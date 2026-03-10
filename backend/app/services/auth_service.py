from __future__ import annotations
import logging
from fastapi import Cookie, HTTPException
from jose import JWTError, jwt
from app.config import get_settings
logger = logging.getLogger(__name__)
_ALGORITHM = "HS256"
async def require_session(session_token: str | None = Cookie(None)) -> str:
    from app.exceptions import AuthError
    if not session_token:
        raise AuthError("No session token provided")
    try:
        settings = get_settings()
        payload = jwt.decode(session_token, settings.secret_key, algorithms=[_ALGORITHM])
        session_id: str | None = payload.get("sub")
        if not session_id:
            raise AuthError("Invalid session token")
        return session_id
    except JWTError as exc:
        raise AuthError(str(exc)) from exc
