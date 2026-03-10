from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from celery import Task

from app.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="qc.run_scan_qc", queue="cpu_queue")
def run_scan_qc_task(
    self: Task,  # type: ignore[type-arg]
    study_id: str,
    nifti_key: str,
) -> dict:  # type: ignore[type-arg]
    """Celery cpu_queue task: non-blocking scan quality check."""
    try:
        self.update_state(state="PROGRESS", meta={"percent": 20, "step": "downloading volume"})
        from app.services.storage_service import download_bytes
        nifti_bytes = download_bytes(nifti_key)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / "volume.nii.gz"
            tmp_path.write_bytes(nifti_bytes)

            self.update_state(state="PROGRESS", meta={"percent": 60, "step": "checking QC"})
            from app.services.scan_qc import run_scan_qc
            result = run_scan_qc(str(tmp_path))

        return {"passed": result.passed, "warnings": result.warnings}
    except Exception as exc:
        self.update_state(state="FAILURE", meta={"error": str(exc), "type": type(exc).__name__})
        raise
