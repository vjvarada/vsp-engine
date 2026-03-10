from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import configure_logging, get_settings
from app.exceptions import (
    AuthError,
    ExportFailedError,
    MeshFailedError,
    MeshNotWatertightError,
    RefineFailedError,
    ScoutFailedError,
    SegmentFailedError,
    SegmentOomError,
    StorageError,
    StudyNotFoundError,
    UploadFailedError,
)
from app.routers import auth, config_router, jobs, mesh, refine, scout, segment, upload

logger = logging.getLogger(__name__)

_settings = get_settings()
configure_logging(_settings)

app = FastAPI(
    title="VSP Engine API",
    version="0.1.0",
    description="Virtual Surgery Planning — FOR PLANNING PURPOSES ONLY",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#  Routers 
app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
app.include_router(upload.router, prefix="/api/upload", tags=["upload"])
app.include_router(scout.router, prefix="/api/scout", tags=["scout"])
app.include_router(segment.router, prefix="/api/segment", tags=["segment"])
app.include_router(refine.router, prefix="/api/refine", tags=["refine"])
app.include_router(mesh.router, prefix="/api/mesh", tags=["mesh"])
app.include_router(jobs.router, prefix="/api/jobs", tags=["jobs"])
app.include_router(config_router.router, prefix="/api/config", tags=["config"])


#  Exception handlers 
@app.exception_handler(StudyNotFoundError)
async def _study_not_found(_: Request, exc: StudyNotFoundError) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.exception_handler(AuthError)
async def _auth_error(_: Request, exc: AuthError) -> JSONResponse:
    return JSONResponse(status_code=403, content={"detail": str(exc)})


@app.exception_handler(UploadFailedError)
async def _upload_failed(_: Request, exc: UploadFailedError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(SegmentOomError)
async def _segment_oom(_: Request, exc: SegmentOomError) -> JSONResponse:
    return JSONResponse(status_code=507, content={"detail": str(exc)})


@app.exception_handler(MeshNotWatertightError)
async def _not_watertight(_: Request, exc: MeshNotWatertightError) -> JSONResponse:
    return JSONResponse(status_code=409, content={"detail": str(exc)})


@app.exception_handler(ScoutFailedError)
@app.exception_handler(SegmentFailedError)
@app.exception_handler(RefineFailedError)
@app.exception_handler(MeshFailedError)
@app.exception_handler(ExportFailedError)
@app.exception_handler(StorageError)
async def _task_failed(_: Request, exc: Exception) -> JSONResponse:
    logger.error("Task failed: %s", exc, exc_info=True)
    return JSONResponse(status_code=500, content={"detail": str(exc)})


@app.get("/api/health", tags=["health"])
async def health_check() -> dict[str, str]:
    return {"status": "ok", "version": "0.1.0"}
