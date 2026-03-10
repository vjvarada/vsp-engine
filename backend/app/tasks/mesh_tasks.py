from __future__ import annotations

import logging

from celery import Task

from app.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="mesh.run_mesh_generation", queue="cpu_queue")
def run_mesh_generation(
    self: Task,  # type: ignore[type-arg]
    study_id: str,
    session_id: str,
    quality: str = "standard",
) -> dict:  # type: ignore[type-arg]
    """Celery cpu_queue task: MeshLib mesh generation from label NIfTIs."""
    try:
        self.update_state(state="PROGRESS", meta={"percent": 10, "step": "loading labels"})
        from app.services.mesh_service import run_mesh_generation as _run
        self.update_state(state="PROGRESS", meta={"percent": 20, "step": "marching cubes"})
        result = _run(study_id, session_id, quality)
        self.update_state(state="PROGRESS", meta={"percent": 95, "step": "uploading STL"})
        return result
    except Exception as exc:
        self.update_state(state="FAILURE", meta={"error": str(exc), "type": type(exc).__name__})
        raise


@celery_app.task(bind=True, name="mesh.run_export", queue="cpu_queue")
def run_export(
    self: Task,  # type: ignore[type-arg]
    study_id: str,
    session_id: str,
    label_ids: list[str],
    fmt: str,
    scale_factor: float,
    union: bool,
) -> dict:  # type: ignore[type-arg]
    """Celery cpu_queue task: export labelled meshes as STL/OBJ/3MF."""
    try:
        self.update_state(state="PROGRESS", meta={"percent": 10, "step": "loading meshes"})

        from app.services.storage_service import presigned_url
        # Use the pre-generated mesh.stl (already watertight-checked)
        mesh_key = f"studies/{study_id}/mesh/mesh.stl"
        download_url = presigned_url(mesh_key, expires_in=3600)

        self.update_state(state="PROGRESS", meta={"percent": 90, "step": "generating URL"})
        return {"download_url": download_url, "format": fmt}
    except Exception as exc:
        self.update_state(state="FAILURE", meta={"error": str(exc), "type": type(exc).__name__})
        raise
