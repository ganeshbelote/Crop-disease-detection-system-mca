"""
SQLAlchemy engine and session management for MySQL.

Uses SQLAlchemy 2.0 style with PyMySQL as the driver. Connection details are
pulled from environment variables via app.config.settings — no credentials
are hardcoded anywhere in this module.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import settings

engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,  # detect and recover from stale MySQL connections
    pool_recycle=3600,
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    """Base class for all ORM models."""


def get_db():
    """FastAPI dependency that yields a database session and always closes it."""
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create all tables that don't exist yet. Called by scripts/init_db.py
    and, defensively, on application startup."""
    # Import models here so they are registered on Base.metadata before
    # create_all is called.
    from app.models import prediction  # noqa: F401

    Base.metadata.create_all(bind=engine)
