from sqlalchemy import (
    Column,
    DateTime,
    Index,
    String,
    Text,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID

from app.database import Base


class VerificationJob(Base):
    """ORM model for the *verification_jobs* table."""

    __tablename__ = "verification_jobs"

    # ------------------------------------------------------------------
    # Primary key
    # ------------------------------------------------------------------
    job_id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
        nullable=False,
    )

    # ------------------------------------------------------------------
    # Tenant / user identity
    # ------------------------------------------------------------------
    system_id = Column(String(100), nullable=False, index=True)
    user_id = Column(String(100), nullable=False)

    # ------------------------------------------------------------------
    # Document metadata
    # ------------------------------------------------------------------
    document_type = Column(String(50), nullable=False)
    expected_name = Column(String(200), nullable=False)
    expected_dob = Column(String(20), nullable=True)
    callback_url = Column(String(500), nullable=True)

    # ------------------------------------------------------------------
    # Processing state
    # ------------------------------------------------------------------
    status = Column(String(20), nullable=False, default="processing")
    flags = Column(JSONB, nullable=False, default=list)
    approval_signals = Column(JSONB, nullable=False, default=list)
    extracted_data = Column(JSONB, nullable=False, default=dict)
    confidence_scores = Column(JSONB, nullable=False, default=dict)

    # ------------------------------------------------------------------
    # Detection results
    # ------------------------------------------------------------------
    document_type_detected = Column(String(50), nullable=True)
    image_path = Column(String(500), nullable=False)

    # ------------------------------------------------------------------
    # Timestamps
    # ------------------------------------------------------------------
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # ------------------------------------------------------------------
    # Review audit trail
    # ------------------------------------------------------------------
    reviewed_by = Column(String(100), nullable=True)
    review_comment = Column(Text, nullable=True)
    reviewed_at = Column(DateTime(timezone=True), nullable=True)

    # ------------------------------------------------------------------
    # Explicit composite / secondary indexes
    # ------------------------------------------------------------------
    __table_args__ = (
        # system_id is already indexed via the column-level `index=True`;
        # add a named index here so it is easy to reference in migrations.
        Index("ix_verification_jobs_system_id", "system_id"),
    )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<VerificationJob job_id={self.job_id} "
            f"system_id={self.system_id!r} "
            f"document_type={self.document_type!r} "
            f"status={self.status!r}>"
        )
