from __future__ import annotations

import gzip
import logging
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

from fastapi import UploadFile

from app.config import get_settings
from app.exceptions import StudyNotFoundError, UploadFailedError
from app.services.audit_service import write_audit_log

logger = logging.getLogger(__name__)

_PLANNING_DISCLAIMER = b"FOR PLANNING PURPOSES ONLY - NOT FOR DIAGNOSTIC USE"

# Local uploads directory (dev mode — no MinIO)
_UPLOADS_DIR = Path(__file__).resolve().parent.parent.parent / "uploads"

_NIFTI_EXTENSIONS = {".nii", ".nii.gz", ".nrrd"}


def _is_nifti_filename(name: str) -> bool:
    """Check if filename looks like NIfTI/NRRD (no DICOM conversion needed)."""
    lower = name.lower()
    return lower.endswith(".nii.gz") or lower.endswith(".nii") or lower.endswith(".nrrd")


def _extract_nifti_metadata(nifti_path: Path) -> dict[str, object]:
    """Read NIfTI header via nibabel to extract dimensions and voxel spacing."""
    import nibabel as nib
    import numpy as np

    img = nib.load(str(nifti_path))
    header = img.header
    dims = img.shape[:3]  # (x, y, z)
    spacing = header.get_zooms()[:3]  # type: ignore[union-attr]

    return {
        "dimensions": (int(dims[0]), int(dims[1]), int(dims[2])),
        "voxel_spacing": (
            round(float(spacing[0]), 4),
            round(float(spacing[1]), 4),
            round(float(spacing[2]), 4),
        ),
    }


def _detect_modality_from_nifti(nifti_path: Path) -> str:
    """Heuristic: check HU range to guess CT vs MR. Fallback to 'unknown'."""
    try:
        import nibabel as nib
        import numpy as np

        img = nib.load(str(nifti_path))
        data = np.asarray(img.dataobj, dtype=np.float32)
        mn, mx = float(np.min(data)), float(np.max(data))
        # CT typically has min < -500 (air ~-1000)
        if mn < -500:
            return "CT"
        # MR typically 0+ with high max
        if mn >= 0 and mx > 100:
            return "MR"
    except Exception:
        logger.debug("Modality detection failed; defaulting to unknown")
    return "unknown"


def _save_local(study_id: str, nifti_data: bytes, filename: str = "volume.nii.gz") -> Path:
    """Save NIfTI to local uploads directory."""
    study_dir = _UPLOADS_DIR / study_id
    study_dir.mkdir(parents=True, exist_ok=True)
    out_path = study_dir / filename
    out_path.write_bytes(nifti_data)
    logger.debug("Saved local NIfTI: %s (%d bytes)", out_path, len(nifti_data))
    return out_path


def _try_upload_minio(study_id: str, nifti_data: bytes) -> bool:
    """Attempt MinIO upload. Returns True on success, False on failure (dev mode)."""
    try:
        from app.services.storage_service import upload_bytes

        nifti_key = f"studies/{study_id}/volume.nii.gz"
        upload_bytes(nifti_key, nifti_data, content_type="application/gzip")
        logger.info("NIfTI uploaded to MinIO: %s", nifti_key)
        return True
    except Exception as exc:
        logger.warning("MinIO upload skipped (dev mode): %s", exc)
        return False


def _try_trigger_qc(study_id: str, nifti_key: str) -> str | None:
    """Attempt to enqueue QC Celery task. Returns task_id or None in dev mode."""
    try:
        from app.tasks.qc_tasks import run_scan_qc_task

        task = run_scan_qc_task.apply_async(args=[study_id, nifti_key], queue="cpu_queue")
        return task.id  # type: ignore[no-any-return]
    except Exception as exc:
        logger.warning("Celery QC task skipped (dev mode): %s", exc)
        return None


def get_local_nifti_path(study_id: str, filename: str) -> Path:
    """Resolve the local path for a study file. Raises StudyNotFoundError if missing."""
    file_path = _UPLOADS_DIR / study_id / filename
    if not file_path.exists():
        raise StudyNotFoundError(f"File not found: {study_id}/{filename}")
    return file_path


async def ingest_dicom(
    file: UploadFile,
    study_id: str,
    session_id: str,
) -> dict[str, object]:
    """
    Accept a DICOM ZIP, single DCM, or NIfTI file.

    - NIfTI files are stored directly (no conversion).
    - DICOM files are converted via dcm2niix.
    - Saves to local filesystem (always) and MinIO (if available).
    - Extracts metadata with nibabel.
    - Triggers QC task if Celery/Redis are available.

    Returns dict ready for FinalizeResponse serialization.
    """
    write_audit_log(session_id, "upload_start", study_id)

    filename = file.filename or "upload"
    contents = await file.read()

    if _is_nifti_filename(filename):
        # NIfTI / NRRD — save directly, no dcm2niix needed
        nifti_data = contents
        out_filename = "volume.nii.gz" if filename.lower().endswith(".nii.gz") else filename
        # If it's a plain .nii, gzip it
        if filename.lower().endswith(".nii") and not filename.lower().endswith(".nii.gz"):
            nifti_data = gzip.compress(contents)
            out_filename = "volume.nii.gz"
    else:
        # DICOM — try dcm2niix conversion
        nifti_data, out_filename = await _convert_dicom(contents, filename)

    # Save locally (always works)
    local_path = _save_local(study_id, nifti_data, out_filename)

    # Extract metadata
    try:
        meta = _extract_nifti_metadata(local_path)
        modality = _detect_modality_from_nifti(local_path)
    except Exception as exc:
        logger.warning("Metadata extraction failed: %s — using defaults", exc)
        meta = {"dimensions": (0, 0, 0), "voxel_spacing": (1.0, 1.0, 1.0)}
        modality = "unknown"

    # Try MinIO upload (non-blocking; skipped in dev if unavailable)
    nifti_key = f"studies/{study_id}/{out_filename}"
    _try_upload_minio(study_id, nifti_data)

    # Try Celery QC task
    qc_task_id = _try_trigger_qc(study_id, nifti_key)

    # Build NIfTI URL: local dev serves via /api/upload/files/{study_id}/{filename}
    nifti_url = f"/api/upload/files/{study_id}/{out_filename}"

    write_audit_log(session_id, "upload_complete", study_id)

    return {
        "study_id": study_id,
        "nifti_url": nifti_url,
        "voxel_spacing": meta["voxel_spacing"],
        "dimensions": meta["dimensions"],
        "modality": modality,
        "qc_task_id": qc_task_id,
        "qc_warnings": [],
    }


async def _convert_dicom(contents: bytes, filename: str) -> tuple[bytes, str]:
    """Run dcm2niix on DICOM data, with pydicom fallback for JPEG 2000."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        raw_path = tmp_path / "input_dicom"
        raw_path.mkdir()

        raw_file = raw_path / filename
        raw_file.write_bytes(contents)

        # If it's a ZIP, extract it then remove the archive
        if filename.lower().endswith(".zip"):
            shutil.unpack_archive(str(raw_file), str(raw_path))
            raw_file.unlink(missing_ok=True)

        nifti_out = tmp_path / "nifti"
        nifti_out.mkdir()

        # Try dcm2niix first (fast C tool)
        nifti_data = _try_dcm2niix(raw_path, nifti_out)

        if nifti_data is not None:
            return (nifti_data, "volume.nii.gz")

        # Fallback: pydicom + nibabel (handles JPEG 2000, JPEG-LS, etc.)
        logger.info("dcm2niix failed — falling back to pydicom + nibabel")
        nifti_data = _convert_with_pydicom(raw_path, nifti_out)

        if nifti_data is not None:
            return (nifti_data, "volume.nii.gz")

        raise UploadFailedError(
            "DICOM conversion failed with both dcm2niix and pydicom. "
            "The DICOM files may use an unsupported format."
        )


def _try_dcm2niix(raw_path: Path, nifti_out: Path) -> bytes | None:
    """Attempt dcm2niix conversion. Returns NIfTI bytes or None on failure."""
    dcm2niix_cmd = shutil.which("dcm2niix")
    if dcm2niix_cmd is None:
        logger.warning("dcm2niix not found on PATH")
        return None

    result = subprocess.run(  # noqa: S603
        [
            dcm2niix_cmd,
            "-r", "y",
            "-z", "y",
            "-f", "volume",
            "-o", str(nifti_out),
            str(raw_path),
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )

    combined_output = (result.stdout or "") + (result.stderr or "")
    logger.debug("dcm2niix output:\n%s", combined_output[:2000])

    nifti_files = sorted(
        nifti_out.glob("*.nii.gz"),
        key=lambda p: p.stat().st_size,
        reverse=True,
    )

    if not nifti_files:
        logger.warning(
            "dcm2niix produced no NIfTI. returncode=%d output:\n%s",
            result.returncode,
            combined_output[:500],
        )
        return None

    chosen = nifti_files[0]
    logger.info(
        "dcm2niix produced %d NIfTI file(s), using %s (%d bytes)",
        len(nifti_files),
        chosen.name,
        chosen.stat().st_size,
    )
    return chosen.read_bytes()


def _convert_with_pydicom(raw_path: Path, nifti_out: Path) -> bytes | None:
    """
    Fallback DICOM→NIfTI conversion using pydicom + nibabel.
    Handles JPEG 2000, JPEG-LS, and other compressed transfer syntaxes
    that dcm2niix cannot decode.
    """
    import nibabel as nib
    import numpy as np
    import pydicom

    # Collect all DICOM files recursively
    dcm_paths = sorted(
        p for p in raw_path.rglob("*")
        if p.is_file() and p.suffix.lower() in {".dcm", ".ima", ""}
    )
    # Also try files without extension (some DICOMs have no extension)
    if not dcm_paths:
        dcm_paths = sorted(
            p for p in raw_path.rglob("*")
            if p.is_file() and not p.suffix
        )

    if not dcm_paths:
        logger.error("No DICOM files found in %s", raw_path)
        return None

    logger.info("pydicom fallback: loading %d DICOM files", len(dcm_paths))

    # Read all slices
    slices: list[pydicom.Dataset] = []
    for p in dcm_paths:
        try:
            ds = pydicom.dcmread(str(p))
            if hasattr(ds, "pixel_array") and hasattr(ds, "ImagePositionPatient"):
                slices.append(ds)
        except Exception as exc:
            logger.debug("Skipping %s: %s", p.name, exc)
            continue

    if not slices:
        logger.error("No valid DICOM image slices found")
        return None

    # Sort by ImagePositionPatient Z coordinate (slice location)
    slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))

    logger.info(
        "pydicom: %d slices, %dx%d, sorted by Z position",
        len(slices),
        slices[0].Rows,
        slices[0].Columns,
    )

    # Stack pixel data into 3D volume
    try:
        volume = np.stack([s.pixel_array for s in slices], axis=0)  # (Z, Y, X)
    except Exception as exc:
        logger.error("Failed to stack DICOM pixel arrays: %s", exc)
        return None

    # Apply rescale slope/intercept (CT Hounsfield units)
    slope = float(getattr(slices[0], "RescaleSlope", 1.0))
    intercept = float(getattr(slices[0], "RescaleIntercept", 0.0))
    if slope != 1.0 or intercept != 0.0:
        volume = volume.astype(np.float32) * slope + intercept

    # Build affine matrix from DICOM metadata
    ds0 = slices[0]
    ipp = [float(x) for x in ds0.ImagePositionPatient]          # (x, y, z) mm
    iop = [float(x) for x in ds0.ImageOrientationPatient]       # 6 direction cosines
    ps = [float(x) for x in getattr(ds0, "PixelSpacing", [1.0, 1.0])]  # row, col spacing

    row_cosine = np.array(iop[:3])
    col_cosine = np.array(iop[3:])

    # Slice spacing: distance between first two slices
    if len(slices) > 1:
        ipp1 = [float(x) for x in slices[1].ImagePositionPatient]
        slice_spacing = np.linalg.norm(np.array(ipp1) - np.array(ipp))
    else:
        slice_spacing = float(getattr(ds0, "SliceThickness", 1.0))

    # NIfTI affine (RAS orientation)
    affine = np.eye(4)
    affine[:3, 0] = row_cosine * ps[1]       # column direction
    affine[:3, 1] = col_cosine * ps[0]       # row direction
    affine[:3, 2] = np.cross(row_cosine, col_cosine) * slice_spacing  # slice direction
    affine[:3, 3] = ipp                       # origin

    # Transpose volume from (Z, Y, X) to (X, Y, Z) for NIfTI convention
    volume_xyz = np.transpose(volume, (2, 1, 0))

    nifti_img = nib.Nifti1Image(volume_xyz, affine)
    out_path = nifti_out / "volume.nii.gz"
    nib.save(nifti_img, str(out_path))

    logger.info(
        "pydicom fallback: saved NIfTI %s (%d bytes), shape=%s",
        out_path.name,
        out_path.stat().st_size,
        volume_xyz.shape,
    )
    return out_path.read_bytes()
