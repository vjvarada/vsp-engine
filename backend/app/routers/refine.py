from __future__ import annotations
import logging
from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict
from app.services.auth_service import require_session
from app.tasks.segment_tasks import run_refine_point, run_refine_bbox
router = APIRouter()
logger = logging.getLogger(__name__)
class PointRequest(BaseModel):
    label_id: str
    ras_mm: list[float]
class BboxRequest(BaseModel):
    label_id: str
    slice_idx: int
    bbox: list[float]
    plane: str = "axial"
class RefineResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    task_id: str
@router.post("/{study_id}/point", response_model=RefineResponse)
async def refine_point(
    study_id: str, body: PointRequest, session_id: str = Depends(require_session)
) -> RefineResponse:
    task = run_refine_point.apply_async(
        args=[study_id, session_id, body.label_id, body.ras_mm], queue="gpu_queue"
    )
    return RefineResponse(task_id=task.id)
@router.post("/{study_id}/bbox", response_model=RefineResponse)
async def refine_bbox(
    study_id: str, body: BboxRequest, session_id: str = Depends(require_session)
) -> RefineResponse:
    task = run_refine_bbox.apply_async(
        args=[study_id, session_id, body.label_id, body.slice_idx, body.bbox, body.plane],
        queue="gpu_queue",
    )
    return RefineResponse(task_id=task.id)
