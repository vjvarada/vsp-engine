from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, Depends, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict

from app.services.auth_service import require_session
from app.services.dicom_service import get_local_nifti_path, ingest_dicom

router = APIRouter()
logger = logging.getLogger(__name__)


class FinalizeResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    study_id: str
    nifti_url: str
    voxel_spacing: tuple[float, float, float]
    dimensions: tuple[int, int, int]
    modality: str
    qc_task_id: str | None
    qc_warnings: list[str]


@router.post("/finalize", response_model=FinalizeResponse)
async def finalize_upload(
    file: UploadFile,
    session_id: str = Depends(require_session),
) -> FinalizeResponse:
    """Accept an uploaded file (DICOM ZIP or NIfTI), convert to NIfTI, return metadata."""
    study_id = str(uuid.uuid4())
    result = await ingest_dicom(file, study_id, session_id)
    logger.info("Upload finalized study=%s session=%s", study_id, session_id)
    return FinalizeResponse(**result)


@router.get("/files/{study_id}/{filename}")
async def serve_study_file(
    study_id: str,
    filename: str,
    session_id: str = Depends(require_session),
) -> FileResponse:
    """Serve study files from local uploads dir (dev mode)."""
    file_path = get_local_nifti_path(study_id, filename)
    return FileResponse(
        path=str(file_path),
        media_type="application/gzip",
        filename=filename,
    )
