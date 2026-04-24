# SecureExam Build Tasks

## Phase 1: Scaffold & Config
- [x] Create project directory structure
- [x] Write config files (whitelist, camera, audio, app)
- [x] Write docker-compose.yml
- [x] Write README.md

## Phase 2: Backend API
- [x] DB schema (init.sql)
- [x] SQLAlchemy models
- [x] FastAPI routes (sessions, violations, reports, users)
- [x] Services layer
- [x] Tests

## Phase 3: AI Service
- [x] Face detector module
- [x] Face recognizer module
- [x] Audio VAD module
- [x] Audio feature extractor
- [x] FastAPI routes
- [x] Tests

## Phase 4: Desktop App
- [x] Electron main process
- [x] Preload / IPC bridge
- [x] URL guard / browser-guard
- [x] React UI pages (Setup, Exam, Report)
- [x] Camera + Mic hooks
- [x] API service clients

## Phase 5: Integration & Verification
- [ ] Run docker-compose up
- [ ] Smoke test all endpoints
- [ ] Manual browser flow test
