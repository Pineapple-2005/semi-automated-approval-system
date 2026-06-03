from celery import Celery
from app.config import settings

celery_app = Celery("verification")
celery_app.conf.update(
    broker_url=settings.REDIS_URL,
    result_backend=settings.REDIS_URL,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    task_routes={
        "app.tasks.process_document.*": {"queue": "document_processing"},
        "app.tasks.webhook_sender.*": {"queue": "webhooks"},
    },
)
celery_app.autodiscover_tasks(["app.tasks"])
