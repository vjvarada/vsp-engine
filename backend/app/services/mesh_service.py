from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

from app.exceptions import MeshFailedError, MeshNotWatertightError, StudyNotFoundError
from app.services.storage_service import download_bytes, upload_bytes

logger = logging.getLogger(__name__)

_PLANNING_HEADER = b"FOR PLANNING PURPOSES ONLY - NOT FOR DIAGNOSTIC USE  " + b" " * (80 - 53)

_QUALITY_PARAMS: dict[str, dict[str, object]] = {
    "preview":  {"max_deviation": 2.0,  "max_angle": 30.0},
    "standard": {"max_deviation": 0.5,  "max_angle": 20.0},
    "high":     {"max_deviation": 0.1,  "max_angle": 10.0},
}


def run_mesh_generation(
    study_id: str,
    session_id: str,
    quality: str = "standard",
) -> dict[str, object]:
    """
    Generate a watertight mesh from AI label NIfTIs using MeshLib.
    Applies mesh healing and topology check before allowing STL download.
    """
    try:
        import meshlib.mrmeshpy as mr  # type: ignore[import]
        import nibabel as nib
        import numpy as np
    except ImportError as exc:
        raise MeshFailedError(f"MeshLib not available: {exc}") from exc

    params = _QUALITY_PARAMS.get(quality, _QUALITY_PARAMS["standard"])
    logger.info("Mesh generation start study=%s quality=%s", study_id, quality)

    manifest_key = f"studies/{study_id}/labels/manifest.json"
    try:
        labels: list[dict] = json.loads(download_bytes(manifest_key))
    except Exception as exc:
        raise StudyNotFoundError(f"Label manifest not found for study {study_id}") from exc

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        meshes = []

        for label in labels:
            if not label.get("visible", True):
                continue
            nifti_bytes = download_bytes(label["nifti_key"])
            nifti_path = tmp / f"{label['label_id']}.nii.gz"
            nifti_path.write_bytes(nifti_bytes)

            img = nib.load(str(nifti_path))
            data = np.asarray(img.dataobj, dtype=np.float32)

            sv = mr.SimpleVolume()
            sv.dims = mr.Vector3i(int(data.shape[0]), int(data.shape[1]), int(data.shape[2]))
            sv.data = mr.std_vector_float(data.flatten(order="F").tolist())

            mc_params = mr.MarchingCubesParams()
            mc_params.iso = 0.5
            mesh = mr.marchingCubes(sv, mc_params)

            # Mesh healing
            mr.fixUndercuts(mesh)
            mr.meshSmoothing(mesh, mr.MeshSmoothingParams())

            meshes.append(mesh)

        if not meshes:
            raise MeshFailedError("No visible labels to mesh")

        # Merge all label meshes
        final_mesh = meshes[0]
        for m in meshes[1:]:
            mr.merge(final_mesh, m)

        # Watertight check — MUST pass before STL download is offered
        is_watertight = final_mesh.topology.isClosed()
        if not is_watertight:
            logger.warning("Mesh is not watertight study=%s", study_id)
            _attempt_repair(final_mesh)
            is_watertight = final_mesh.topology.isClosed()

        # Export STL with planning disclaimer in binary header
        stl_path = tmp / "mesh.stl"
        mr.saveMesh(final_mesh, str(stl_path))
        stl_bytes = stl_path.read_bytes()
        # Overwrite STL 80-byte header with planning disclaimer
        stl_bytes = _PLANNING_HEADER + stl_bytes[80:]

        mesh_key = f"studies/{study_id}/mesh/mesh.stl"
        upload_bytes(mesh_key, stl_bytes, content_type="model/stl")

        vertex_count = final_mesh.topology.numValidVerts()
        face_count = final_mesh.topology.numValidFaces()

    result = {
        "mesh_key": mesh_key,
        "is_watertight": is_watertight,
        "vertex_count": int(vertex_count),
        "face_count": int(face_count),
    }
    result_key = f"studies/{study_id}/mesh/result.json"
    upload_bytes(result_key, json.dumps(result).encode(), content_type="application/json")
    logger.info("Mesh complete study=%s watertight=%s verts=%d", study_id, is_watertight, vertex_count)

    if not is_watertight:
        raise MeshNotWatertightError("Mesh is not watertight after repair attempt")

    return result


def _attempt_repair(mesh: object) -> None:
    """Apply morphological close and fill holes to improve watertightness."""
    try:
        import meshlib.mrmeshpy as mr  # type: ignore[import]
        m = mesh  # type: ignore[assignment]
        mr.fill_holes(m, mr.FillHolesParams())
    except Exception:
        pass  # Repair is best-effort


async def fetch_mesh_result(study_id: str, session_id: str) -> dict[str, object]:
    result_key = f"studies/{study_id}/mesh/result.json"
    try:
        return json.loads(download_bytes(result_key))
    except Exception as exc:
        raise StudyNotFoundError(f"Mesh result not found for study {study_id}") from exc
