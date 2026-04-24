# SecureExam – Anti-Fraud Online Exam System

Hệ thống thi trực tuyến chống gian lận chạy trên Windows/macOS/Linux.

## Kiến trúc tổng thể

```
┌─────────────────────────────────────────────────────────┐
│                  Electron Desktop App                    │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐  │
│  │  BrowserView │  │  React UI    │  │  IPC Bridge    │  │
│  │  (Exam Page) │  │  (Monitor)   │  │  (Main↔Render) │  │
│  └─────────────┘  └──────────────┘  └────────────────┘  │
└──────────────┬──────────────────────────────┬───────────┘
               │ HTTP (localhost)              │ HTTP
               ▼                              ▼
┌──────────────────────┐      ┌───────────────────────────┐
│   AI Service :8001   │      │   Backend API :8000        │
│  - Face Detection    │      │   - Session Management     │
│  - Face Recognition  │      │   - Violation Logging      │
│  - Voice VAD         │      │   - Report Generation      │
│  - Audio Analysis    │      └──────────────┬────────────┘
└──────────────────────┘                     │
                                    ┌────────▼──────┐
                                    │  PostgreSQL    │
                                    │  :5432         │
                                    └───────────────┘
```

## Yêu cầu hệ thống

- **Docker + Docker Compose** (cho backend + AI service + DB)
- **Node.js ≥ 18** (cho desktop app)
- **Python ≥ 3.11** (nếu chạy AI service không qua Docker)

## Hướng dẫn chạy

### 1. Clone và di chuyển vào thư mục

```bash
cd "secure app"
```

### 2. Chạy backend, AI service, và PostgreSQL bằng Docker

```bash
docker-compose up --build -d
```

Kiểm tra services đang hoạt động:
```bash
curl http://localhost:8000/health   # {"status":"ok","service":"backend"}
curl http://localhost:8001/health   # {"status":"ok","service":"ai-service"}
```

> ⚠️ Lần đầu khởi động, AI service sẽ tải model InsightFace (~90MB). Chờ vài phút.

### 3. Cài đặt và chạy desktop app

```bash
cd desktop
npm install
npm run dev
```

Ứng dụng Electron sẽ mở lên.

### 4. Cấu hình URL được phép thi

Chỉnh sửa `config/whitelist.yaml`:
```yaml
whitelist:
  - "https://your-exam-platform.edu/*"
```

---

## Cấu trúc thư mục

```
secure-exam/
├── config/               # YAML config (whitelist, camera, audio, app)
├── backend/              # FastAPI + PostgreSQL API
│   ├── main.py           # Entry point
│   ├── core/             # Config, DB, Logger
│   ├── models/           # SQLAlchemy ORM
│   ├── api/routes/       # REST endpoints
│   ├── services/         # Business logic
│   ├── migrations/       # init.sql schema
│   └── tests/            # Pytest tests
├── ai-service/           # Python AI analysis
│   ├── main.py
│   ├── core/
│   ├── modules/
│   │   ├── face/         # InsightFace detection + recognition
│   │   └── audio/        # WebRTC VAD + MFCC analysis
│   ├── api/routes/
│   └── tests/
└── desktop/              # Electron + React
    ├── electron/         # Main process, preload, IPC handlers
    └── src/              # React UI (pages, components, hooks)
```

---

## Chạy tests

### Backend tests
```bash
cd backend
pip install -r requirements.txt
pip install pytest pytest-asyncio httpx aiosqlite
pytest tests/ -v
```

### AI service tests
```bash
cd ai-service
pip install -r requirements.txt
python -m pytest tests/ -v
```

---

## Cấu hình chi tiết

| File | Mô tả |
|------|-------|
| `config/whitelist.yaml` | Danh sách URL được phép mở trong bài thi |
| `config/camera.yaml` | Thời gian capture, ngưỡng phát hiện khuôn mặt |
| `config/audio.yaml` | VAD mode, ngưỡng phát hiện giọng, MFCC settings |
| `config/app.yaml` | Port services, session config |

---

## Models AI sử dụng

| Module | Model | Kích cỡ | Ghi chú |
|--------|-------|---------|---------|
| Phát hiện khuôn mặt | InsightFace `buffalo_sc` | ~90MB | Chạy trên CPU |
| Nhận diện khuôn mặt | InsightFace embedding | (cùng model) | Cosine similarity |
| Phát hiện giọng nói | WebRTC VAD | ~1MB | Google, C-based, rất nhẹ |
| Trích đặc trưng âm thanh | MFCC (librosa) | 0MB | Thuật toán, không có model |

---

## Mở rộng / Nâng cấp

- **Thêm URL vào whitelist**: Sửa `config/whitelist.yaml`, khởi động lại desktop app.
- **Thay đổi ngưỡng cảnh báo**: Sửa `config/camera.yaml` hoặc `config/audio.yaml`.
- **Thêm module giám sát khác**: Tạo module mới trong `ai-service/modules/`, thêm route, kết nối từ IPC handler.
- **Dùng GPU**: Thay `ctx_id=-1` bằng `ctx_id=0` trong InsightFace `prepare()`.

---

## API Reference

### Backend (port 8000)

| Method | URL | Mô tả |
|--------|-----|-------|
| POST | `/users/` | Đăng ký thí sinh |
| PUT | `/users/{id}/face` | Lưu face embedding |
| POST | `/sessions/` | Bắt đầu phiên thi |
| POST | `/sessions/{id}/end` | Kết thúc phiên thi |
| POST | `/violations/` | Ghi một vi phạm |
| POST | `/violations/batch` | Ghi nhiều vi phạm cùng lúc |
| GET | `/reports/{session_id}` | Lấy báo cáo đầy đủ |

### AI Service (port 8001)

| Method | URL | Mô tả |
|--------|-----|-------|
| POST | `/analyze/face/` | Phân tích frame camera |
| POST | `/analyze/audio/` | Phân tích chunk âm thanh |
| POST | `/analyze/audio/clear` | Xóa session state âm thanh |
