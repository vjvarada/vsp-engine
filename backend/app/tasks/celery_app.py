from __future__ import annotations

from celery import Celery

from app.config import get_settings

_settings = get_settings()

celery_app = Celery(
    "vsp_engine",
    broker=_settings.redis_url,
    backend=_settings.redis_url,
    include=[
        "app.tasks.scout_tasks",
        "app.tasks.segment_tasks",
        "app.tasks.mesh_tasks",
        "app.tasks.qc_tasks",
    ],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    # Two queues: cpu_queue (concurrency=4) and gpu_queue (concurrency=1)
    task_queues={
        "cpu_queue": {"exchange": "cpu_queue", "routing_key": "cpu_queue"},
        "gpu_queue": {"exchange": "gpu_queue", "routing_key": "gpu_queue"},
    },
    task_default_queue="cpu_queue",
)
