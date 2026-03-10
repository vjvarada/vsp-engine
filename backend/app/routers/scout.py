from __future__ import annotations
import logging
from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict
from app.services.auth_service import require_session
from app.tasks.scout_tasks import run_scout_pass
router = APIRouter()
logger = logging.getLogger(__name__)
class ScoutRunResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    task_id: str
class IslandMeta(BaseModel):
    model_config = ConfigDict(frozen=True)
    island_id: str
    label: str
    voxel_count: int
    mesh_key: str
    aabb: dict[str, float]
@router.post("/{study_id}/run", response_model=ScoutRunResponse)
async def run_scout(study_id: str, session_id: str = Depends(require_session)) -> ScoutRunResponse:
    task = run_scout_pass.apply_async(args=[study_id, session_id], queue="cpu_queue")
    return ScoutRunResponse(task_id=task.id)
@router.get("/{study_id}/islands", response_model=list[IslandMeta])
async def get_islands(study_id: str, session_id: str = Depends(require_session)) -> list[IslandMeta]:
    from app.services.scout_service import fetch_island_list
    return await fetch_island_list(study_id, session_id)
