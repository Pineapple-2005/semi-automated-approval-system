from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class JobStatus(str, Enum):
    processing = "processing"
    needs_review = "needs_review"
    approved = "approved"
    rejected = "rejected"
    processing_error = "processing_error"


class DocumentType(str, Enum):
    # Government-issued personal IDs (also accepted as government_id umbrella)
    umid = "umid"
    passport = "passport"
    drivers_license = "drivers_license"
    philsys = "philsys"
    prc = "prc"
    # Umbrella type — resolved to a subtype during processing
    government_id = "government_id"
    # Specialised types
    medical_license = "medical_license"
    organizational_certificate = "organizational_certificate"
    sec_registration = "sec_registration"


# ---------------------------------------------------------------------------
# Request / response helpers
# ---------------------------------------------------------------------------

class VerifyResponse(BaseModel):
    """Returned immediately after a verification job is submitted."""

    job_id: UUID
    status: str
    message: str


class JobStatusResponse(BaseModel):
    """Full representation of a single verification job."""

    model_config = ConfigDict(from_attributes=True)

    job_id: UUID
    system_id: str
    user_id: str
    document_type: str
    expected_name: str
    expected_dob: Optional[str] = None
    callback_url: Optional[str] = None
    status: str
    flags: list[Any] = []
    approval_signals: list[Any] = []
    extracted_data: dict[str, Any] = {}
    confidence_scores: dict[str, Any] = {}
    document_type_detected: Optional[str] = None
    image_path: str
    created_at: Optional[datetime] = None
    reviewed_by: Optional[str] = None
    review_comment: Optional[str] = None
    reviewed_at: Optional[datetime] = None

    # Populated at query time with a MinIO presigned URL; not stored in the DB
    presigned_image_url: Optional[str] = None


class ReviewQueueResponse(BaseModel):
    """Paginated list of jobs awaiting human review."""

    items: list[JobStatusResponse]
    total: int


class ReviewRequest(BaseModel):
    """Payload sent by a human reviewer to approve or reject a job."""

    action: Literal["approve", "reject"]
    comment: Optional[str] = None
    reviewed_by: Optional[str] = None


class WebhookPayload(BaseModel):
    """Outbound payload delivered to the tenant callback URL."""

    job_id: str
    status: str
    flags: list[Any]
    extracted_data: dict[str, Any]
    system_id: str
    document_type: str
    timestamp: str


class ErrorResponse(BaseModel):
    """Standard error envelope returned on 4xx / 5xx responses."""

    error: str
    message: str
    detail: Optional[str] = None
