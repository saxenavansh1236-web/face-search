"""
test_user_store.py

Tests the SQLite-backed user store.
"""

import uuid
from app.services.user_store import create_user, verify_user, user_exists


def test_create_and_verify_user():
    username = f"pytest_user_{uuid.uuid4().hex[:8]}"
    assert create_user(username, "correct_password") is True
    assert verify_user(username, "correct_password") is True
    assert verify_user(username, "wrong_password") is False


def test_duplicate_username_rejected():
    username = f"pytest_dup_{uuid.uuid4().hex[:8]}"
    assert create_user(username, "password123") is True
    assert create_user(username, "different_password") is False


def test_user_exists():
    username = f"pytest_exists_{uuid.uuid4().hex[:8]}"
    assert user_exists(username) is False
    create_user(username, "password123")
    assert user_exists(username) is True