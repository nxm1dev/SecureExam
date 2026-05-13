-- ============================================================
-- Exam Anti-cheating Database Schema
-- PostgreSQL 15
-- ============================================================

-- Enable UUID generation
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ──────────────────────────────────────────
-- Users (exam candidates + admins)
-- ──────────────────────────────────────────
CREATE TABLE IF NOT EXISTS users (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email       VARCHAR(255) UNIQUE NOT NULL,
    full_name   VARCHAR(255) NOT NULL,
    role        VARCHAR(20) NOT NULL DEFAULT 'candidate', -- candidate | admin
    -- Base64-encoded face embedding vector (512-dim InsightFace)
    face_embedding  TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ──────────────────────────────────────────
-- Exam Sessions
-- ──────────────────────────────────────────
CREATE TABLE IF NOT EXISTS sessions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    exam_url        VARCHAR(2048) NOT NULL,
    status          VARCHAR(20) NOT NULL DEFAULT 'active', -- active | completed | terminated
    started_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ended_at        TIMESTAMPTZ,
    -- Summary counters (updated at end of session)
    total_violations    INTEGER NOT NULL DEFAULT 0,
    critical_count      INTEGER NOT NULL DEFAULT 0,
    high_count          INTEGER NOT NULL DEFAULT 0,
    medium_count        INTEGER NOT NULL DEFAULT 0,
    low_count           INTEGER NOT NULL DEFAULT 0
);

-- ──────────────────────────────────────────
-- Violations / Events Log
-- ──────────────────────────────────────────
CREATE TYPE violation_severity AS ENUM ('low', 'medium', 'high', 'critical');

CREATE TABLE IF NOT EXISTS violations (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id  UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    user_id     UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    -- Event type: tab_switch | fullscreen_exit | url_blocked | no_face |
    --             multiple_faces | identity_mismatch | speech_detected |
    --             multiple_voices | voice_overlap | rapid_voice_change
    event_type  VARCHAR(64) NOT NULL,
    severity    violation_severity NOT NULL DEFAULT 'medium',
    -- Free-form JSON for extra context (face count, similarity score, etc.)
    metadata    JSONB NOT NULL DEFAULT '{}',
    occurred_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ──────────────────────────────────────────
-- Indexes for fast reporting queries
-- ──────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_violations_session  ON violations(session_id);
CREATE INDEX IF NOT EXISTS idx_violations_user     ON violations(user_id);
CREATE INDEX IF NOT EXISTS idx_violations_event    ON violations(event_type);
CREATE INDEX IF NOT EXISTS idx_violations_time     ON violations(occurred_at DESC);
CREATE INDEX IF NOT EXISTS idx_sessions_user       ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_status     ON sessions(status);

-- ──────────────────────────────────────────
-- Seed admin user for local dev
-- ──────────────────────────────────────────
INSERT INTO users (email, full_name, role)
VALUES ('admin@examac.local', 'Admin', 'admin')
ON CONFLICT DO NOTHING;
