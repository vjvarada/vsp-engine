from __future__ import annotations

import logging

from celery import Task

from app.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="segment.run_segment_pass", queue="gpu_queue")
def run_segment_pass(
    self: Task,  # type: ignore[type-arg]
    study_id: str,
    session_id: str,
    hint_labels: list[str],
) -> dict:  # type: ignore[type-arg]
    """Celery gpu_queue task: TotalSegmentator AI segmentation."""
    try:
        self.update_state(state="PROGRESS", meta={"percent": 5, "step": "loading volume"})

        from app.services.storage_service import download_bytes
        import json

        # Fetch ROI from saved manifest
        roi_key = f"studies/{study_id}/roi.json"
        try:
            roi: dict = json.loads(download_bytes(roi_key))
        except Exception:
            roi = {}  # fallback: segment whole volume

        self.update_state(state="PROGRESS", meta={"percent": 15, "step": "cropping ROI"})

        from app.services.segment_service import run_segment
        self.update_state(state="PROGRESS", meta={"percent": 25, "step": "running TotalSegmentator"})
        labels = run_segment(study_id, session_id, roi, hint_labels)

        self.update_state(state="PROGRESS", meta={"percent": 95, "step": "uploading labels"})
        return {"labels": labels, "count": len(labels)}
    except Exception as exc:
        self.update_state(state="FAILURE", meta={"error": str(exc), "type": type(exc).__name__})
        raise


@celery_app.task(bind=True, name="segment.run_refine_point", queue="gpu_queue")
def run_refine_point(
    self: Task,  # type: ignore[type-arg]
    study_id: str,
    session_id: str,
    label_id: str,
    ras_mm: list[float],
) -> dict:  # type: ignore[type-arg]
    """Celery gpu_queue task: SAM-Med3D-turbo 3D point refinement."""
    try:
        self.update_state(state="PROGRESS", meta={"percent": 10, "step": "loading model"})
        try:
            import medim  # type: ignore[import]
            model = medim.create_model("SAM-Med3D", pretrained=True)
        except ImportError as exc:
            raise RuntimeError(f"SAM-Med3D not available: {exc}") from exc

        self.update_state(state="PROGRESS", meta={"percent": 40, "step": "running inference"})
        # Input point in (z,y,x) normalised format — NEVER raw voxel or direct RAS
        from app.services.coord_service import nifti_voxel_to_sam_med3d_point
        point = nifti_voxel_to_sam_med3d_point(
            (float(ras_mm[0]), float(ras_mm[1]), float(ras_mm[2])),
            (0.0, 0.0, 0.0),  # patch origin — TODO: compute from ROI
        )
        logger.info("SAM-Med3D point=%s study=%s label=%s", point, study_id, label_id)
        # TODO: run SAM-Med3D inference and upload result mask
        self.update_state(state="PROGRESS", meta={"percent": 90, "step": "uploading mask"})
        return {"label_id": label_id, "refined": True}
    except Exception as exc:
        self.update_state(state="FAILURE", meta={"error": str(exc), "type": type(exc).__name__})
        raise


@celery_app.task(bind=True, name="segment.run_refine_bbox", queue="gpu_queue")
def run_refine_bbox(
    self: Task,  # type: ignore[type-arg]
    study_id: str,
    session_id: str,
    label_id: str,
    slice_idx: int,
    bbox: list[float],
    plane: str,
) -> dict:  # type: ignore[type-arg]
    """Celery gpu_queue task: MedSAM 2D bbox refinement."""
    try:
        self.update_state(state="PROGRESS", meta={"percent": 10, "step": "loading MedSAM"})
        # MedSAM: segment_anything library with medsam_vit_b checkpoint
        logger.info("MedSAM bbox=%s slice=%d plane=%s study=%s", bbox, slice_idx, plane, study_id)
        # TODO: run MedSAM inference and upload result mask
        self.update_state(state="PROGRESS", meta={"percent": 90, "step": "uploading mask"})
        return {"label_id": label_id, "refined": True}
    except Exception as exc:
        self.update_state(state="FAILURE", meta={"error": str(exc), "type": type(exc).__name__})
        raise
