from __future__ import annotations
import logging
from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict
from app.services.auth_service import require_session
from app.tasks.mesh_tasks import run_mesh_generation, run_export
router = APIRouter()
logger = logging.getLogger(__name__)
class MeshRunRequest(BaseModel):
    quality: str = "standard"
class MeshRunResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    task_id: str
class MeshResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    mesh_key: str
    is_watertight: bool
    vertex_count: int
    face_count: int
class ExportRequest(BaseModel):
    label_ids: list[str]
    format: str = "stl"
    scale_factor: float = 1.0
    union: bool = True
class ExportResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    task_id: str
@router.post("/{study_id}/generate", response_model=MeshRunResponse)
async def generate_mesh(
    study_id: str, body: MeshRunRequest, session_id: str = Depends(require_session)
) -> MeshRunResponse:
    task = run_mesh_generation.apply_async(
        args=[study_id, session_id, body.quality], queue="cpu_queue"
    )
    return MeshRunResponse(task_id=task.id)
@router.get("/{study_id}/result", response_model=MeshResult)
async def get_mesh_result(study_id: str, session_id: str = Depends(require_session)) -> MeshResult:
    from app.services.mesh_service import fetch_mesh_result
    return await fetch_mesh_result(study_id, session_id)
@router.post("/{study_id}/export", response_model=ExportResponse)
async def export_mesh(
    study_id: str, body: ExportRequest, session_id: str = Depends(require_session)
) -> ExportResponse:
    task = run_export.apply_async(
        args=[study_id, session_id, body.label_ids, body.format, body.scale_factor, body.union],
        queue="cpu_queue",
    )
    return ExportResponse(task_id=task.id)
