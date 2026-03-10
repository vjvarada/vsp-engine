from __future__ import annotations
import asyncio, logging
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict
from app.tasks.celery_app import celery_app
logger = logging.getLogger(__name__)
router = APIRouter()
class JobStatusResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    task_id: str
    status: str
    percent: int | None = None
    result: object = None
    error: str | None = None
@router.get("/{task_id}/status", response_model=JobStatusResponse)
async def get_job_status(task_id: str) -> JobStatusResponse:
    result = celery_app.AsyncResult(task_id)
    meta: dict = result.info if isinstance(result.info, dict) else {}
    return JobStatusResponse(
        task_id=task_id, status=result.state,
        percent=meta.get("percent"), result=meta.get("result"),
        error=meta.get("error"),
    )
async def _sse_generator(task_id: str):
    poll_interval = 1.0
    while True:
        result = celery_app.AsyncResult(task_id)
        meta: dict = result.info if isinstance(result.info, dict) else {}
        data = {"task_id": task_id, "status": result.state, **meta}
        yield f"data: {data}\n\n"
        if result.state in ("SUCCESS", "FAILURE", "REVOKED"):
            break
        await asyncio.sleep(poll_interval)
@router.get("/{task_id}/stream")
async def stream_job(task_id: str, request: Request) -> StreamingResponse:
    return StreamingResponse(_sse_generator(task_id), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
