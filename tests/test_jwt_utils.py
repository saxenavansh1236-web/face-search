"""
test_jwt_utils.py

Tests JWT token creation and decoding.
"""

from app.services.jwt_utils import create_access_token, decode_access_token


def test_create_and_decode_token():
    token = create_access_token("testuser")
    username = decode_access_token(token)
    assert username == "testuser"


def test_invalid_token_returns_none():
    assert decode_access_token("not-a-real-token") is None


def test_tampered_token_rejected():
    token = create_access_token("testuser")
    tampered = token[:-3] + "xyz"
    assert decode_access_token(tampered) is None