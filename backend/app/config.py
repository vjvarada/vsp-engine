from __future__ import annotations

import logging
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Application
    app_env: str = Field("development", alias="APP_ENV")
    secret_key: str = Field(..., alias="SECRET_KEY")
    allowed_origins: list[str] = Field(
        default=["http://localhost:5173"], alias="ALLOWED_ORIGINS"
    )

    # Database
    database_url: str = Field(
        "postgresql+asyncpg://vsp:vsp@localhost:5432/vsp", alias="DATABASE_URL"
    )

    # Redis / Celery
    redis_url: str = Field("redis://localhost:6379/0", alias="REDIS_URL")

    # MinIO / S3
    minio_endpoint: str = Field("localhost:9000", alias="MINIO_ENDPOINT")
    minio_access_key: str = Field("vspengine", alias="MINIO_ACCESS_KEY")
    minio_secret_key: str = Field(..., alias="MINIO_SECRET_KEY")
    minio_bucket: str = Field("vsp-studies", alias="MINIO_BUCKET")
    minio_use_ssl: bool = Field(False, alias="MINIO_USE_SSL")
    study_ttl_days: int = Field(30, alias="STUDY_TTL_DAYS")

    # GPU
    gpu_device: str = Field("cuda:0", alias="GPU_DEVICE")

    # MedSAM2 isolated service
    medsam2_base_url: str = Field("http://medsam2:8001", alias="MEDSAM2_BASE_URL")

    # TotalSegmentator license (for appendicular_bones)
    totalseg_license_key: str | None = Field(None, alias="TOTALSEG_LICENSE_KEY")

    # Logging
    log_level: str = Field("INFO", alias="LOG_LEVEL")


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]


def configure_logging(settings: Settings) -> None:
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )