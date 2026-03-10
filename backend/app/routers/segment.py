from __future__ import annotations
import logging
from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict
from app.services.auth_service import require_session
from app.tasks.segment_tasks import run_segment_pass
router = APIRouter()
logger = logging.getLogger(__name__)
class SegmentRequest(BaseModel):
    hint_labels: list[str] = []
class SegmentRunResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    task_id: str
class SegLabel(BaseModel):
    model_config = ConfigDict(frozen=True)
    label_id: str
    name: str
    color: str
    visible: bool = True
@router.post("/{study_id}/run", response_model=SegmentRunResponse)
async def run_segment(
    study_id: str, body: SegmentRequest, session_id: str = Depends(require_session)
) -> SegmentRunResponse:
    task = run_segment_pass.apply_async(
        args=[study_id, session_id, body.hint_labels], queue="gpu_queue"
    )
    return SegmentRunResponse(task_id=task.id)
@router.get("/{study_id}/labels", response_model=list[SegLabel])
async def get_labels(study_id: str, session_id: str = Depends(require_session)) -> list[SegLabel]:
    from app.services.segment_service import fetch_label_list
    return await fetch_label_list(study_id, session_id)
