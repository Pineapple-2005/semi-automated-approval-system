import logging

import httpx

from app.config import settings
from app.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(
    bind=True,
    max_retries=3,
    name="app.tasks.webhook_sender.send_webhook",
)
def send_webhook(self, job_id: str, callback_url: str, payload: dict):
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "PhilID-Verification-API/1.0",
        "X-Job-ID": job_id,
    }
    try:
        with httpx.Client(timeout=settings.WEBHOOK_TIMEOUT_SECONDS) as client:
            response = client.post(callback_url, json=payload, headers=headers)
            response.raise_for_status()
            logger.info(
                f"Webhook sent for job {job_id}, status {response.status_code}"
            )
    except Exception as exc:
        countdown = 2 ** self.request.retries * 30
        logger.warning(
            f"Webhook failed for job {job_id}, retry {self.request.retries}: {exc}"
        )
        raise self.retry(exc=exc, countdown=countdown)
