# SecureExam – Anti-Fraud Online Exam System

## Overview

Build a cross-platform (Windows/macOS/Linux) anti-cheating desktop exam application with the following stack:

| Layer | Technology |
|---|---|
| Desktop App | Electron + React (TypeScript) |
| AI Service | Python + FastAPI (camera/mic analysis) |
| Backend API | FastAPI + PostgreSQL (sessions, logs, reports) |
| Communication | Local HTTP REST + WebSocket |
| AI Models | InsightFace (face detection/recognition), WebRTC VAD + SpeechBrain (voice analysis) |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                  Electron Desktop App                    │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐  │
│  │  BrowserView │  │  React UI    │  │  IPC Bridge    │  │
│  │  (Exam Page) │  │  (Monitor)   │  │  (Main↔Render) │  │
│  └─────────────┘  └──────────────┘  └────────────────┘  │
└──────────────┬──────────────────────────────┬───────────┘
               │ HTTP/WS (localhost)           │ HTTP/WS
               ▼                              ▼
┌──────────────────────┐      ┌───────────────────────────┐
│   AI Service :8001   │      │   Backend API :8000        │
│  FastAPI              │      │   FastAPI                  │
│  - Face Detection     │      │   - Session Management     │
│  - Face Recognition   │      │   - Violation Logging      │
│  - VAD                │      │   - Report Generation      │
│  - Voice Analysis     │      │   - User Management        │
└──────────────────────┘      └──────────────┬────────────┘
                                              │
                                     ┌────────▼──────┐
                                     │  PostgreSQL    │
                                     │  :5432         │
                                     └───────────────┘
```

---

## Directory Structure

```
secure-exam/
├── desktop/                        # Electron + React app
│   ├── package.json
│   ├── tsconfig.json
│   ├── webpack.config.js
│   ├── electron/
│   │   ├── main.ts                 # Electron main process
│   │   ├── preload.ts              # IPC preload script
│   │   ├── browser-guard.ts        # URL whitelist + block logic
│   │   ├── session-manager.ts      # Exam session lifecycle
│   │   └── ipc-handlers.ts         # IPC event handlers
│   └── src/
│       ├── App.tsx
│       ├── index.tsx
│       ├── pages/
│       │   ├── SetupPage.tsx       # URL + profile setup
│       │   ├── ExamPage.tsx        # Live monitoring UI
│       │   └── ReportPage.tsx      # Post-exam report
│       ├── components/
│       │   ├── CameraMonitor.tsx
│       │   ├── AlertBanner.tsx
│       │   └── ViolationList.tsx
│       ├── hooks/
│       │   ├── useCamera.ts
│       │   ├── useMicrophone.ts
│       │   └── useViolations.ts
│       └── services/
│           ├── api.ts              # Backend API client
│           └── ai-service.ts       # AI service client
│
├── ai-service/                     # Python AI analysis service
│   ├── requirements.txt
│   ├── config.yaml                 # AI thresholds + settings
│   ├── main.py                     # FastAPI entry point
│   ├── core/
│   │   ├── config.py
│   │   └── logger.py
│   ├── modules/
│   │   ├── face/
│   │   │   ├── detector.py         # Face detection (InsightFace)
│   │   │   ├── recognizer.py       # Face recognition
│   │   │   └── analyzer.py         # Frame analysis orchestrator
│   │   └── audio/
│   │       ├── vad.py              # Voice Activity Detection
│   │       ├── feature_extractor.py # Audio feature extraction
│   │       └── analyzer.py         # Audio analysis orchestrator
│   ├── api/
│   │   ├── routes/
│   │   │   ├── face.py             # /analyze/face endpoint
│   │   │   └── audio.py            # /analyze/audio endpoint
│   │   └── schemas.py              # Pydantic models
│   └── tests/
│       ├── test_face.py
│       └── test_audio.py
│
├── backend/                        # FastAPI backend + PostgreSQL
│   ├── requirements.txt
│   ├── config.yaml
│   ├── main.py
│   ├── core/
│   │   ├── config.py
│   │   ├── database.py             # SQLAlchemy setup
│   │   └── logger.py
│   ├── models/
│   │   ├── session.py
│   │   ├── violation.py
│   │   └── user.py
│   ├── api/
│   │   ├── routes/
│   │   │   ├── sessions.py
│   │   │   ├── violations.py
│   │   │   ├── reports.py
│   │   │   └── users.py
│   │   └── schemas.py
│   ├── services/
│   │   ├── session_service.py
│   │   ├── violation_service.py
│   │   └── report_service.py
│   ├── migrations/
│   │   └── init.sql                # DB schema
│   └── tests/
│       ├── test_sessions.py
│       ├── test_violations.py
│       └── test_reports.py
│
├── config/                         # Shared configuration
│   ├── whitelist.yaml              # Allowed URLs
│   ├── camera.yaml                 # Camera thresholds
│   ├── audio.yaml                  # Audio thresholds
│   └── app.yaml                    # Global app settings
│
├── docker-compose.yml              # Local dev orchestration
└── README.md                       # Setup + run instructions
```

---

## Proposed Changes

### 1. Project Scaffold & Config
#### [NEW] config/whitelist.yaml
#### [NEW] config/camera.yaml
#### [NEW] config/audio.yaml
#### [NEW] config/app.yaml
#### [NEW] docker-compose.yml

### 2. Backend API (FastAPI + PostgreSQL)
#### [NEW] backend/migrations/init.sql — DB schema with sessions, violations, users tables
#### [NEW] backend/core/database.py — SQLAlchemy async engine
#### [NEW] backend/models/ — ORM models
#### [NEW] backend/api/routes/ — REST endpoints
#### [NEW] backend/services/ — Business logic
#### [NEW] backend/tests/ — Pytest tests

### 3. AI Service (Python)
#### [NEW] ai-service/modules/face/ — InsightFace detection + recognition
#### [NEW] ai-service/modules/audio/ — WebRTC VAD + feature extraction
#### [NEW] ai-service/api/routes/ — Analysis endpoints
#### [NEW] ai-service/tests/ — Unit tests

### 4. Desktop App (Electron + React)
#### [NEW] desktop/electron/main.ts — Main process, BrowserView, URL guard
#### [NEW] desktop/electron/preload.ts — Secure IPC bridge
#### [NEW] desktop/src/pages/ — Setup, Exam, Report pages
#### [NEW] desktop/src/components/ — UI building blocks
#### [NEW] desktop/src/services/ — API clients

---

## Models Used

| Purpose | Library | Why |
|---|---|---|
| Face Detection | `insightface` (buffalo_sc model) | Lightweight, accurate, free |
| Face Recognition | `insightface` embedding comparison | Same library, no extra cost |
| Voice Activity Detection | `webrtcvad` | Ultra-lightweight Google VAD |
| Audio Features | `librosa` (MFCCs) | Standard, well-maintained |
| Speaker clustering | cosine similarity on MFCCs | No training needed, real-time |

---

## Verification Plan

### Automated Tests
- `pytest ai-service/tests/` — face + audio analysis units
- `pytest backend/tests/` — API endpoint tests

### Manual Verification
- Open valid URL → loads in BrowserView ✓
- Open blocked URL → redirect + log entry ✓  
- Cover camera 5s → warning alert ✓
- Show 2 faces → strong alert ✓
- Play background speech → anomaly detected ✓
