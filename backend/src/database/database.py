"""Database session and engine setup.

Provides SQLAlchemy engine, base model, and a `get_db` generator that yields
a database session and ensures it is closed after use.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from src.config import DATABASE_URL

Base = declarative_base()

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    pool_size=20,
    max_overflow=10,
    pool_timeout=60,
)
SESSION_LOCAL = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """Yield a database session and ensure it is closed afterwards."""
    db = SESSION_LOCAL()
    try:
        yield db
    finally:
        db.close()
