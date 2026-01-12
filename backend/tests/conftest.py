"""Pytest fixtures for NFTool backend tests."""

# pylint: disable=redefined-outer-name

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.api import app
from src.database.database import Base, get_db
from src.config import API_KEY

# Setup an in-memory SQLite database for testing
SQLALCHEMY_DATABASE_URL = "sqlite:///./test_nftool.db"

engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TESTING_SESSION_LOCAL = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture(scope="session")
def db_engine():
    """Create and destroy the test database engine for the session."""
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)

@pytest.fixture
def db(db_engine):
    """Provide a transactional database session for each test."""
    connection = db_engine.connect()
    transaction = connection.begin()
    session = TESTING_SESSION_LOCAL(bind=connection)

    yield session

    session.close()
    transaction.rollback()
    connection.close()

@pytest.fixture
def client(db):
    """Create a TestClient that uses the test DB session fixture."""

    def override_get_db():
        """Override dependency to provide the test DB session."""
        yield db

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()

@pytest.fixture
def auth_headers():
    """Fixture to provide authentication headers for API requests."""
    return {"X-API-Key": API_KEY}
