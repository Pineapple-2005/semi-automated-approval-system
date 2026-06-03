"""
Philippine ID Document Verification API - Application Entry Point
=================================================================

This module bootstraps the FastAPI application for the Philippine ID
Document Verification microservice. It is responsible for:

  * Initialising the Supabase PostgreSQL schema via :func:`init_db` on
    startup and tearing down any long-lived resources on shutdown.
  * Ensuring the MinIO object-storage bucket exists before the first
    request is served.
  * Registering CORS, request-logging, and global error-handling
    middleware so every downstream router inherits consistent behaviour.
  * Mounting the ``/api/v1`` verification and review routers.
  * Exposing lightweight ``GET /`` (index) and ``GET /health``
    (liveness / readiness probe) endpoints.

Environment variables consumed (see :mod:`app.config` for full list):

  ``DATABASE_URL``        - psycopg2 DSN for Supabase PostgreSQL (SSL)
  ``REDIS_URL``           - Celery / cache broker URL
  ``MINIO_ENDPOINT``      - host[:port] of the MinIO server
  ``MINIO_ACCESS_KEY``    - MinIO access-key ID
  ``MINIO_SECRET_KEY``    - MinIO secret access key
  ``MINIO_BUCKET``        - bucket name used for document images
  ``MINIO_SECURE``        - ``"true"`` to use TLS for MinIO connections
  ``API_KEY_SYSTEM_MAP``  - JSON object mapping API keys → system_id
  ``ALLOWED_API_KEYS``    - comma-separated list of valid API keys
  ``MAX_IMAGE_SIZE_MB``   - maximum accepted upload size in megabytes
  ``WEBHOOK_TIMEOUT_SECONDS`` - HTTP timeout for outbound callback POSTs

Tech stack
----------
Python 3.11, FastAPI, Celery/Redis, SQLAlchemy / psycopg2,
Supabase PostgreSQL (SSL), MinIO, Tesseract OCR, OpenCV.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from app.config import settings
from app.database import init_db
from app.routers.review import router as review_router
from app.routers.verification import router as verification_router
from app.schemas import ErrorResponse
from app.utils.minio_client import ensure_bucket_exists

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan context manager (replaces deprecated on_event)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:  # noqa: ARG001
    """Application lifespan handler.

    Startup sequence
    ~~~~~~~~~~~~~~~~
    1. ``init_db()`` - creates or migrates the ``verification_jobs`` table
       (and any other schema objects) in Supabase PostgreSQL.
    2. ``ensure_bucket_exists(bucket)`` - idempotently creates the MinIO
       bucket that stores uploaded document images.

    Shutdown sequence
    ~~~~~~~~~~~~~~~~~
    Any cleanup (connection-pool disposal, etc.) should be added in the
    block after the ``yield``.
    """
    # ---- startup -----------------------------------------------------------
    logger.info("Starting Philippine ID Verification API …")

    try:
        logger.info("Initialising database schema …")
        init_db()
        logger.info("Database schema ready.")
    except Exception as exc:  # pragma: no cover
        logger.critical("Failed to initialise database: %s", exc, exc_info=True)
        raise

    try:
        logger.info(
            "Ensuring MinIO bucket '%s' exists …", settings.MINIO_BUCKET
        )
        ensure_bucket_exists(settings.MINIO_BUCKET)
        logger.info("MinIO bucket ready.")
    except Exception as exc:  # pragma: no cover
        logger.critical(
            "Failed to ensure MinIO bucket existence: %s", exc, exc_info=True
        )
        raise

    logger.info("Application startup complete.")

    yield  # ---- application running ----------------------------------------

    # ---- shutdown ----------------------------------------------------------
    logger.info("Shutting down Philippine ID Verification API …")
    # SQLAlchemy engine disposal and other cleanup would go here.
    logger.info("Shutdown complete.")


# ---------------------------------------------------------------------------
# Request logging middleware
# ---------------------------------------------------------------------------

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log every inbound HTTP request with method, path, status, and latency."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        start = time.perf_counter()
        response: Response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1_000

        logger.info(
            "%s %s → %d (%.1f ms) | client=%s",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
            request.client.host if request.client else "unknown",
        )
        return response


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Philippine ID Document Verification API",
    description=(
        "Automated verification of Philippine government-issued IDs "
        "(UMID, Passport, Driver's License, PhilSys, PRC) using OCR "
        "and template matching. Supports multi-tenant deployments via "
        "API-key → system_id mapping."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Middleware registration order matters: added last → executed first
# ---------------------------------------------------------------------------

# 1. CORS - must wrap everything so pre-flight OPTIONS responses are handled
#    before any auth or business logic.
app.add_middleware(
    CORSMiddleware,
    allow_origins=getattr(settings, "CORS_ALLOW_ORIGINS", ["*"]),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Request / response logging.
app.add_middleware(RequestLoggingMiddleware)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(
    verification_router,
    prefix="/api/v1",
    tags=["Verification"],
)
app.include_router(
    review_router,
    prefix="/api/v1",
    tags=["Review"],
)

# ---------------------------------------------------------------------------
# Global exception handler
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """Catch-all handler that converts unhandled exceptions to a structured
    :class:`~app.schemas.ErrorResponse` with HTTP 500 so clients always
    receive valid JSON rather than an HTML traceback.
    """
    logger.exception(
        "Unhandled exception on %s %s",
        request.method,
        request.url.path,
    )
    error_body = ErrorResponse(
        error="internal_server_error",
        message="An unexpected error occurred. Please try again later.",
        detail=str(exc) if settings.DEBUG else None,
    )
    return JSONResponse(
        status_code=500,
        content=error_body.model_dump(exclude_none=True),
    )


# ---------------------------------------------------------------------------
# Built-in endpoints
# ---------------------------------------------------------------------------

@app.get(
    "/",
    summary="API index",
    response_description="Basic service information.",
    tags=["General"],
)
async def index() -> dict:
    """Return a short description of the service and a link to interactive
    API documentation."""
    return {
        "message": "Philippine ID Verification API",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get(
    "/health",
    summary="Health check",
    response_description="Liveness/readiness status of backing services.",
    tags=["General"],
)
async def health_check() -> JSONResponse:
    """Probe the database and MinIO connections.

    Returns HTTP 200 with ``status: ok`` when all backing services are
    reachable, or HTTP 200 with ``status: degraded`` when one or more
    checks fail (the individual boolean flags indicate which service is
    unavailable).

    This endpoint intentionally never returns a 5xx so that orchestration
    platforms (Kubernetes, Railway, etc.) can distinguish between a
    completely unresponsive pod (no response) and a degraded-but-live
    service (200 degraded).
    """
    db_ok = False
    minio_ok = False

    # ---- database probe ----------------------------------------------------
    try:
        from app.database import get_engine  # local import to avoid circular
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(__import__("sqlalchemy").text("SELECT 1"))
        db_ok = True
    except Exception as exc:  # pragma: no cover
        logger.warning("Health check: database probe failed: %s", exc)

    # ---- MinIO probe -------------------------------------------------------
    try:
        from app.utils.minio_client import get_minio_client  # local import
        client = get_minio_client()
        client.bucket_exists(settings.MINIO_BUCKET)
        minio_ok = True
    except Exception as exc:  # pragma: no cover
        logger.warning("Health check: MinIO probe failed: %s", exc)

    overall = "ok" if (db_ok and minio_ok) else "degraded"
    timestamp = datetime.now(timezone.utc).isoformat()

    return JSONResponse(
        status_code=200,
        content={
            "status": overall,
            "database": db_ok,
            "minio": minio_ok,
            "timestamp": timestamp,
        },
    )
