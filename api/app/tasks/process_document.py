import logging

from app.database import SessionLocal
from app.models import VerificationJob
from app.services.flagging import generate_approval_signals, generate_flags
from app.services.ocr import run_ocr
from app.services.template_matcher import run_template_matching
from app.tasks.celery_app import celery_app
from app.utils.minio_client import download_file

logger = logging.getLogger(__name__)


@celery_app.task(
    bind=True,
    name="app.tasks.process_document.process_document",
)
def process_document(self, job_id: str):
    db = SessionLocal()
    job = None
    try:
        # a. Fetch the verification job
        job = (
            db.query(VerificationJob)
            .filter(VerificationJob.job_id == job_id)
            .first()
        )

        # b. Guard: job must exist
        if not job:
            logger.error(
                f"process_document: job {job_id} not found in database."
            )
            return

        # c. Download document image from MinIO
        image_bytes = download_file(job.image_path)

        # d. Run template matching first so government_id umbrella can be resolved
        match_result = run_template_matching(image_bytes, job.document_type)

        # e. Resolve the effective OCR doc-type:
        #    For government_id, use the classifier's detected subtype if confident.
        from app.services.system_requirements import get_effective_ocr_type
        effective_type = get_effective_ocr_type(
            job.document_type, match_result["detected_type"]
        )

        # f. Run OCR with the resolved document type
        ocr_result = run_ocr(image_bytes, effective_type)

        # f. Generate flags (concerns) and approval signals (confirmations)
        shared_args = (
            ocr_result["extracted_data"],
            ocr_result["confidence_scores"],
            ocr_result.get("quality_metrics", {}),
            match_result["detected_type"],
            job.document_type,
            job.expected_name,
            job.expected_dob,
            match_result.get("region_analysis"),
        )
        flags = generate_flags(*shared_args)
        approval_signals = generate_approval_signals(*shared_args)

        # g. Always route to needs_review — human makes the final decision
        job.status = "needs_review"
        job.flags = flags
        job.approval_signals = approval_signals
        job.extracted_data = ocr_result["extracted_data"]
        job.confidence_scores = ocr_result["confidence_scores"]
        job.document_type_detected = match_result["detected_type"]
        db.commit()

    except Exception as e:
        logger.exception("process_document: unhandled error for job %s", job_id)
        if job is not None:
            job.status = "processing_error"
            job.flags = [f"processing_error: {str(e)[:500]}"]
            try:
                db.commit()
            except Exception:
                logger.exception(
                    "process_document: failed to persist error state for job %s", job_id
                )
    finally:
        db.close()
