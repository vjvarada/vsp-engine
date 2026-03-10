from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Response
from jose import jwt
from pydantic import BaseModel, ConfigDict

from app.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter()
_ALGORITHM = "HS256"
_TTL_HOURS = 24


class SessionResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    session_id: str
    expires_at: str


@router.post("/session", response_model=SessionResponse)
async def create_session(response: Response) -> SessionResponse:
    settings = get_settings()
    session_id = str(uuid.uuid4())
    exp = datetime.now(timezone.utc) + timedelta(hours=_TTL_HOURS)
    token = jwt.encode(
        {"sub": session_id, "exp": exp},
        settings.secret_key,
        algorithm=_ALGORITHM,
    )
    response.set_cookie(
        "session_token",
        token,
        httponly=True,
        samesite="strict",
        secure=settings.app_env == "production",
        max_age=_TTL_HOURS * 3600,
    )
    logger.info("Session created: %s", session_id)
    return SessionResponse(session_id=session_id, expires_at=exp.isoformat())
