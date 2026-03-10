from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from app.exceptions import ScoutFailedError, StudyNotFoundError
from app.services.storage_service import download_bytes, upload_bytes

logger = logging.getLogger(__name__)

# MeshLib HU threshold for bone isolation
_BONE_HU_MIN = 200.0
_BONE_HU_MAX = 3000.0


@dataclass
class IslandMeta:
    island_id: str
    label: str
    voxel_count: int
    mesh_key: str
    aabb: dict[str, float]


def run_scout(study_id: str, session_id: str) -> list[dict]:  # type: ignore[return]
    """
    Tier-1 MeshLib HU-threshold scout pass.
    Reads volume.nii.gz from MinIO, extracts bone islands, uploads per-island PLY meshes.
    Returns list of IslandMeta dicts.
    """
    try:
        import meshlib.mrmeshpy as mr  # type: ignore[import]
        import nibabel as nib
        import numpy as np
        import tempfile
        from pathlib import Path
    except ImportError as exc:
        raise ScoutFailedError(f"Required library not available: {exc}") from exc

    logger.info("Scout start study=%s", study_id)

    nifti_key = f"studies/{study_id}/volume.nii.gz"
    try:
        nifti_bytes = download_bytes(nifti_key)
    except Exception as exc:
        raise StudyNotFoundError(f"NIfTI not found for study {study_id}") from exc

    islands: list[dict] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        nifti_path = tmp / "volume.nii.gz"
        nifti_path.write_bytes(nifti_bytes)

        img = nib.load(str(nifti_path))
        data = np.asarray(img.dataobj, dtype=np.float32)
        affine = img.affine

        # Build MeshLib mesh from HU threshold
        vdb = mr.FloatGrid()
        mask = (data >= _BONE_HU_MIN) & (data <= _BONE_HU_MAX)

        # Use MeshLib to create mesh from voxels
        params = mr.MarchingCubesParams()
        params.iso = 0.5

        # Convert mask to MeshLib SimpleVolume
        sv = mr.SimpleVolume()
        sv.dims = mr.Vector3i(int(data.shape[0]), int(data.shape[1]), int(data.shape[2]))
        flat = mask.astype(np.float32).flatten(order="F")
        sv.data = mr.std_vector_float(flat.tolist())

        mesh = mr.marchingCubes(sv, params)

        if mesh.topology.numFaces() == 0:
            logger.warning("Scout produced empty mesh for study=%s", study_id)
            return islands

        # Split into connected components (bone islands)
        components = mr.MeshComponents.getAllComponentsVerts(mesh, mr.MeshComponents.FaceIncidence.PerEdge)

        for idx, comp in enumerate(components):
            island_id = f"island_{idx:04d}"
            # Extract sub-mesh for this component
            sub_mesh = mr.Mesh(mesh)
            mr.MeshComponents.getComponent(sub_mesh, idx, mr.MeshComponents.FaceIncidence.PerEdge)

            # Compute AABB
            bb = sub_mesh.getBoundingBox()
            aabb = {
                "x_min": float(bb.min.x), "x_max": float(bb.max.x),
                "y_min": float(bb.min.y), "y_max": float(bb.max.y),
                "z_min": float(bb.min.z), "z_max": float(bb.max.z),
            }

            # Save PLY to MinIO
            ply_path = tmp / f"{island_id}.ply"
            mr.saveMesh(sub_mesh, str(ply_path))
            mesh_key = f"studies/{study_id}/islands/{island_id}.ply"
            upload_bytes(mesh_key, ply_path.read_bytes(), content_type="application/octet-stream")

            islands.append({
                "island_id": island_id,
                "label": f"Bone Island {idx + 1}",
                "voxel_count": int(sub_mesh.topology.numFaces()),
                "mesh_key": mesh_key,
                "aabb": aabb,
            })

    # Persist island manifest to MinIO
    manifest_key = f"studies/{study_id}/islands/manifest.json"
    upload_bytes(manifest_key, json.dumps(islands).encode(), content_type="application/json")
    logger.info("Scout complete study=%s islands=%d", study_id, len(islands))
    return islands


async def fetch_island_list(study_id: str, session_id: str) -> list[dict]:
    """Load island manifest from MinIO."""
    manifest_key = f"studies/{study_id}/islands/manifest.json"
    try:
        data = download_bytes(manifest_key)
        return json.loads(data)
    except Exception as exc:
        raise StudyNotFoundError(f"Island manifest not found for study {study_id}") from exc
