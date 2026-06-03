import logging

import psycopg2
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import SQLAlchemyError

from app.config import settings

logger = logging.getLogger(__name__)


def _make_engine():
    """
    Build the SQLAlchemy engine.

    Prefers individual DB_* vars (no URL encoding needed) over DATABASE_URL.
    Both paths enforce sslmode=require for Supabase.
    """
    if settings.DB_HOST and settings.DB_USER and settings.DB_PASSWORD:
        # Use a psycopg2 creator so the password is passed as a plain string —
        # no URL encoding, no parsing, no surprises.
        def _creator():
            return psycopg2.connect(
                host=settings.DB_HOST,
                port=settings.DB_PORT,
                user=settings.DB_USER,
                password=settings.DB_PASSWORD,
                dbname=settings.DB_NAME,
                sslmode="require",
            )

        return create_engine(
            "postgresql+psycopg2://",
            creator=_creator,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
        )

    if not settings.DATABASE_URL:
        raise RuntimeError(
            "No database configuration found. "
            "Set DB_HOST/DB_USER/DB_PASSWORD/DB_NAME or DATABASE_URL in .env"
        )

    # Fallback: DATABASE_URL path (password must be URL-encoded if it has special chars)
    url = settings.DATABASE_URL
    if "sslmode" not in url:
        url += ("&" if "?" in url else "?") + "sslmode=require"

    return create_engine(
        url,
        connect_args={"sslmode": "require"},
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
    )


engine = _make_engine()

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_engine():
    return engine


def init_db() -> None:
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables initialised successfully.")
    except SQLAlchemyError as exc:
        logger.error("Failed to initialise database tables: %s", exc)
        raise
