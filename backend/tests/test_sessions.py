"""
backend/tests/test_sessions.py
──────────────────────────────
Tests for session lifecycle and URL-blocking logic.
"""

import pytest


@pytest.mark.asyncio
async def test_create_session(client):
    """Session can be created for a valid user."""
    # Create a user first
    user_resp = await client.post(
        "/users/",
        json={"email": "alice@test.com", "full_name": "Alice", "role": "candidate"},
    )
    assert user_resp.status_code == 201
    user_id = user_resp.json()["id"]

    # Start session with a (whitelisted) URL
    resp = await client.post(
        "/sessions/",
        json={"user_id": user_id, "exam_url": "https://exam.example.edu/test/1"},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["status"] == "active"
    assert data["exam_url"] == "https://exam.example.edu/test/1"


@pytest.mark.asyncio
async def test_end_session(client):
    """Session can be ended and counters are set."""
    user_resp = await client.post(
        "/users/",
        json={"email": "bob@test.com", "full_name": "Bob", "role": "candidate"},
    )
    user_id = user_resp.json()["id"]

    sess_resp = await client.post(
        "/sessions/",
        json={"user_id": user_id, "exam_url": "https://exam.example.edu/test/2"},
    )
    session_id = sess_resp.json()["id"]

    end_resp = await client.post(
        f"/sessions/{session_id}/end", json={"status": "completed"}
    )
    assert end_resp.status_code == 200
    assert end_resp.json()["status"] == "completed"
    assert end_resp.json()["ended_at"] is not None


@pytest.mark.asyncio
async def test_get_session_not_found(client):
    """Non-existent session returns 404."""
    resp = await client.get(
        "/sessions/00000000-0000-0000-0000-000000000000"
    )
    assert resp.status_code == 404
