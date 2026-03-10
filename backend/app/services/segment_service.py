from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

from app.exceptions import SegmentFailedError, SegmentOomError, StudyNotFoundError
from app.services.feature_flags import get_feature_flags
from app.services.storage_service import download_bytes, upload_bytes

logger = logging.getLogger(__name__)

# Dispatch table: surgeon hint → TotalSegmentator task name
_HINT_TO_TASK: dict[str, str] = {
    "skull": "craniofacial_structures",
    "mandible": "craniofacial_structures",
    "teeth": "teeth",
    "spine": "total",
    "pelvis": "total",
    "appendicular": "appendicular_bones",
    "trunk": "trunk_cavities",
}


def _resolve_totalseg_task(hint_labels: list[str]) -> str:
    flags = get_feature_flags()
    for hint in hint_labels:
        task = _HINT_TO_TASK.get(hint.lower())
        if task == "appendicular_bones" and not flags.appendicular_bones:
            logger.warning("appendicular_bones requested but license not available — falling back to total")
            return "total"
        if task:
            return task
    return "total"


def run_segment(
    study_id: str,
    session_id: str,
    roi: dict[str, float],
    hint_labels: list[str],
) -> list[dict]:
    """
    Tier-2 TotalSegmentator AI segmentation on the ROI crop.
    Reads volume.nii.gz, crops to ROI, runs TotalSegmentator, uploads label NIfTIs.
    Returns list of SegLabel dicts.
    """
    try:
        import nibabel as nib
        import numpy as np
        from totalsegmentator.python_api import totalsegmentator  # type: ignore[import]
    except ImportError as exc:
        raise SegmentFailedError(f"TotalSegmentator not available: {exc}") from exc

    logger.info("Segment start study=%s roi=%s", study_id, roi)
    nifti_key = f"studies/{study_id}/volume.nii.gz"

    try:
        nifti_bytes = download_bytes(nifti_key)
    except Exception as exc:
        raise StudyNotFoundError(f"NIfTI not found for study {study_id}") from exc

    task_name = _resolve_totalseg_task(hint_labels)
    labels: list[dict] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        nifti_path = tmp / "volume.nii.gz"
        nifti_path.write_bytes(nifti_bytes)

        # Crop to ROI before running TotalSegmentator (10-58× smaller volume → no GPU OOM)
        img = nib.load(str(nifti_path))
        data = np.asarray(img.dataobj)
        _apply_roi_crop(img, data, roi, tmp / "roi.nii.gz")

        out_dir = tmp / "seg_out"
        out_dir.mkdir()

        try:
            totalsegmentator(
                input=str(tmp / "roi.nii.gz"),
                output=str(out_dir),
                task=task_name,
                fast=False,
                statistics=False,
                radiomics=False,
            )
        except RuntimeError as exc:
            if "CUDA out of memory" in str(exc) or "OOM" in str(exc):
                raise SegmentOomError(str(exc)) from exc
            raise SegmentFailedError(str(exc)) from exc

        # Upload each label NIfTI to MinIO and build manifest
        for label_nii in sorted(out_dir.glob("*.nii.gz")):
            label_name = label_nii.stem.replace(".nii", "")
            label_key = f"studies/{study_id}/labels/{label_name}.nii.gz"
            upload_bytes(label_key, label_nii.read_bytes())
            labels.append({
                "label_id": label_name,
                "name": label_name.replace("_", " ").title(),
                "color": "#4fc3f7",
                "visible": True,
                "nifti_key": label_key,
            })

    manifest_key = f"studies/{study_id}/labels/manifest.json"
    upload_bytes(manifest_key, json.dumps(labels).encode(), content_type="application/json")
    logger.info("Segment complete study=%s labels=%d task=%s", study_id, len(labels), task_name)
    return labels


def _apply_roi_crop(img: object, data: object, roi: dict[str, float], out_path: Path) -> None:
    """Crop NIfTI volume to the surgeon-defined ROI before TotalSegmentator inference."""
    import nibabel as nib  # type: ignore[import]
    import numpy as np

    nii: nib.Nifti1Image = img  # type: ignore[assignment]
    arr: np.ndarray = data  # type: ignore[assignment]
    affine = nii.affine
    affine_inv = np.linalg.inv(affine)

    def _to_vox(ras_mm: tuple[float, float, float]) -> tuple[int, int, int]:
        v = affine_inv @ np.array([*ras_mm, 1.0])
        return (int(np.clip(round(v[0]), 0, arr.shape[0]-1)),
                int(np.clip(round(v[1]), 0, arr.shape[1]-1)),
                int(np.clip(round(v[2]), 0, arr.shape[2]-1)))

    lo = _to_vox((roi["x_min"], roi["y_min"], roi["z_min"]))
    hi = _to_vox((roi["x_max"], roi["y_max"], roi["z_max"]))

    xs = slice(min(lo[0], hi[0]), max(lo[0], hi[0]) + 1)
    ys = slice(min(lo[1], hi[1]), max(lo[1], hi[1]) + 1)
    zs = slice(min(lo[2], hi[2]), max(lo[2], hi[2]) + 1)

    cropped = arr[xs, ys, zs]
    new_affine = affine.copy()
    new_affine[:3, 3] = (affine @ np.array([lo[0], lo[1], lo[2], 1.0]))[:3]
    nib.save(nib.Nifti1Image(cropped, new_affine, nii.header), str(out_path))


async def fetch_label_list(study_id: str, session_id: str) -> list[dict]:
    manifest_key = f"studies/{study_id}/labels/manifest.json"
    try:
        return json.loads(download_bytes(manifest_key))
    except Exception as exc:
        raise StudyNotFoundError(f"Label manifest not found for study {study_id}") from exc
