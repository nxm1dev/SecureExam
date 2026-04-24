"""
backend/tests/test_violations.py
──────────────────────────────────
Tests for violation recording and reporting.
"""

import pytest


@pytest.fixture
async def session_with_user(client):
    """Helper: create user + session, return (client, user_id, session_id)."""
    user_resp = await client.post(
        "/users/",
        json={"email": "carol@test.com", "full_name": "Carol", "role": "candidate"},
    )
    user_id = user_resp.json()["id"]

    sess_resp = await client.post(
        "/sessions/",
        json={"user_id": user_id, "exam_url": "https://exam.example.edu/test/3"},
    )
    session_id = sess_resp.json()["id"]
    return user_id, session_id


@pytest.mark.asyncio
async def test_log_violation(client, session_with_user):
    """A violation can be recorded."""
    user_id, session_id = session_with_user
    resp = await client.post(
        "/violations/",
        json={
            "session_id": session_id,
            "user_id": user_id,
            "event_type": "tab_switch",
            "severity": "medium",
            "event_metadata": {"window_title": "Chrome"},
        },
    )
    assert resp.status_code == 201
    assert resp.json()["event_type"] == "tab_switch"


@pytest.mark.asyncio
async def test_log_multiple_faces_violation(client, session_with_user):
    """Critical multiple-faces violation is recorded correctly."""
    user_id, session_id = session_with_user
    resp = await client.post(
        "/violations/",
        json={
            "session_id": session_id,
            "user_id": user_id,
            "event_type": "multiple_faces",
            "severity": "critical",
            "event_metadata": {"face_count": 2},
        },
    )
    assert resp.status_code == 201
    assert resp.json()["severity"] == "critical"


@pytest.mark.asyncio
async def test_report_generation(client, session_with_user):
    """Report returns violation timeline and counts."""
    user_id, session_id = session_with_user

    # Log a no_face violation
    await client.post(
        "/violations/",
        json={
            "session_id": session_id,
            "user_id": user_id,
            "event_type": "no_face",
            "severity": "high",
            "event_metadata": {"seconds_missing": 15},
        },
    )

    # End session
    await client.post(
        f"/sessions/{session_id}/end", json={"status": "completed"}
    )

    # Fetch report
    resp = await client.get(f"/reports/{session_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_violations"] >= 1
    assert len(data["timeline"]) >= 1
    event_types = [v["event_type"] for v in data["timeline"]]
    assert "no_face" in event_types
