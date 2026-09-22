"""
Unit tests for user authentication and profile management in ResearchLens AI.
"""

import pytest
from pathlib import Path
import tempfile
from services.auth_service import AuthService


@pytest.fixture
def temp_auth_service():
    """Provides an isolated AuthService with a temporary SQLite database."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test_users.db"
        yield AuthService(db_path=db_path)


def test_register_user_success(temp_auth_service):
    success, msg, user = temp_auth_service.register_user(
        name="Marie Curie",
        email="marie.curie@sorbonne.fr",
        password="RadiumPassword123!",
        institution="University of Paris",
    )
    assert success is True
    assert "successfully" in msg.lower()
    assert user is not None
    assert user["name"] == "Marie Curie"
    assert user["email"] == "marie.curie@sorbonne.fr"
    assert user["institution"] == "University of Paris"
    assert "id" in user


def test_register_user_validation(temp_auth_service):
    # Invalid name
    s, m, _ = temp_auth_service.register_user("", "test@test.com", "pass123")
    assert s is False
    assert "name" in m.lower()

    # Invalid email
    s, m, _ = temp_auth_service.register_user("Jane", "invalid-email", "pass123")
    assert s is False
    assert "email" in m.lower()

    # Short password
    s, m, _ = temp_auth_service.register_user("Jane", "jane@test.com", "123")
    assert s is False
    assert "password" in m.lower()


def test_register_duplicate_email(temp_auth_service):
    temp_auth_service.register_user("User One", "dup@test.com", "Password123")
    success, msg, user = temp_auth_service.register_user("User Two", "dup@test.com", "AnotherPassword123")
    assert success is False
    assert "already exists" in msg.lower()
    assert user is None


def test_authenticate_user_success(temp_auth_service):
    temp_auth_service.register_user("Alan Turing", "alan@cambridge.ac.uk", "EnigmaCipher123", "Cambridge")
    success, msg, user = temp_auth_service.authenticate_user("alan@cambridge.ac.uk", "EnigmaCipher123")
    assert success is True
    assert "successful" in msg.lower()
    assert user is not None
    assert user["name"] == "Alan Turing"


def test_authenticate_user_wrong_password(temp_auth_service):
    temp_auth_service.register_user("Alan Turing", "alan@cambridge.ac.uk", "EnigmaCipher123")
    success, msg, user = temp_auth_service.authenticate_user("alan@cambridge.ac.uk", "WrongPassword")
    assert success is False
    assert "incorrect" in msg.lower()
    assert user is None


def test_authenticate_user_nonexistent(temp_auth_service):
    success, msg, user = temp_auth_service.authenticate_user("ghost@cambridge.ac.uk", "Password123")
    assert success is False
    assert "no account found" in msg.lower()
    assert user is None


def test_get_or_create_demo_user(temp_auth_service):
    demo1 = temp_auth_service.get_or_create_demo_user()
    assert demo1 is not None
    assert demo1["email"] == "researcher@researchlens.ai"

    # Calling again returns the existing demo user without duplicate error
    demo2 = temp_auth_service.get_or_create_demo_user()
    assert demo2["id"] == demo1["id"]
    assert demo2["email"] == demo1["email"]
