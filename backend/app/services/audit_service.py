from __future__ import annotations
import logging
logger = logging.getLogger(__name__)
def write_audit_log(session_id: str, action: str, study_id: str | None = None) -> None:
    """Write to audit_log table. PHI MUST NOT appear in any parameter."""
    # TODO: write to PostgreSQL audit_log table via SQLAlchemy
    logger.info("AUDIT session=%s action=%s study=%s", session_id, action, study_id or "N/A")
