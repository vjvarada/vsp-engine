from __future__ import annotations

import logging

from celery import Task

from app.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="scout.run_scout_pass", queue="cpu_queue")
def run_scout_pass(self: Task, study_id: str, session_id: str) -> dict:  # type: ignore[type-arg]
    """Celery cpu_queue task: MeshLib HU-threshold scout pass."""
    try:
        self.update_state(state="PROGRESS", meta={"percent": 5, "step": "loading volume"})
        from app.services.scout_service import run_scout
        self.update_state(state="PROGRESS", meta={"percent": 20, "step": "thresholding bone"})
        islands = run_scout(study_id, session_id)
        self.update_state(state="PROGRESS", meta={"percent": 90, "step": "uploading meshes"})
        return {"islands": islands, "count": len(islands)}
    except Exception as exc:
        self.update_state(state="FAILURE", meta={"error": str(exc), "type": type(exc).__name__})
        raise
