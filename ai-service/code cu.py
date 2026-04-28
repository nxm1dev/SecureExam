from __future__ import annotations

"""
ai-service/code cu.py
─────────────────────────────
Multimodal AI Exam Anti-Cheating System.

Gộp 3 phần:
  1. Frontend React/TS (VAD + Camera) – code mẫu trong docstring
  2. Backend Python – ExamFaceDetector nâng cấp (MAR + Pose)
  3. Cross-check Controller – Logic xác nhận chéo
"""

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  PHAN 1: FRONTEND (React / TypeScript) - Code mau trong docstring       ║
# ║  TOI UU: WebSocket + Throttle 500ms + JPEG 70% + Scale 640x480          ║
# ╚════════════════════════════════════════════════════════════════════════════╝

FRONTEND_CODE = """
// ─── ExamMonitor.tsx ────────────────────────────────────────────────────────
// npm install @ricky0123/vad-react @ricky0123/vad-web

import React, { useRef, useState, useCallback, useEffect } from 'react';
import { useMicVAD } from '@ricky0123/vad-react';

interface ExamMonitorProps {
  webSocketUrl: string;  // vd: 'ws://localhost:8001/ws/monitor'
}

const ExamMonitor: React.FC<ExamMonitorProps> = ({ webSocketUrl }) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [ws, setWs] = useState<WebSocket | null>(null);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const captureIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // Khoi tao WebSocket (nhe hon REST, tranh overhead HTTP headers moi request)
  useEffect(() => {
    const socket = new WebSocket(webSocketUrl);
    socket.onopen = () => console.log('[ExamMonitor] WebSocket Connected');
    socket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      console.log('[ExamMonitor] Verdict:', data);
    };
    setWs(socket);
    return () => socket.close();
  }, [webSocketUrl]);

  // Khoi tao Camera (chi video, audio do VAD xu ly rieng)
  useEffect(() => {
    navigator.mediaDevices.getUserMedia({ video: true, audio: false })
      .then((stream) => {
        if (videoRef.current) videoRef.current.srcObject = stream;
      })
      .catch(console.error);
  }, []);

  // ── Ham capture va gui du lieu TOI UU ──────────────────────────────────
  // - Scale xuong 640x480 de giam payload (thay vi full HD)
  // - JPEG chat luong 0.7 (giam ~33% kich thuoc so voi PNG)
  // - Gui qua WebSocket (tranh overhead HTTP moi request)
  const captureAndSend = useCallback(() => {
    if (!videoRef.current || !canvasRef.current || !ws || ws.readyState !== WebSocket.OPEN) return;

    const canvas = canvasRef.current;
    const video = videoRef.current;
    const context = canvas.getContext('2d');

    // Scale down kich thuoc anh de giam payload va tang toc do xu ly
    canvas.width = 640;
    canvas.height = 480;
    context?.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Dung JPEG chat luong 0.7 thay vi PNG de nhe hon
    const base64Image = canvas.toDataURL('image/jpeg', 0.7);

    ws.send(JSON.stringify({
      speech_detected: true,
      timestamp: Date.now(),
      image: base64Image
    }));
  }, [ws]);

  // ── Tich hop VAD voi THROTTLE ──────────────────────────────────────────
  // BOTTLENECK FIX: Khong gui moi frame (30 FPS se lam nghen)
  // Chi gui 2 frame/giay (moi 500ms) khi dang noi
  const vad = useMicVAD({
    startOnLoad: true,
    onSpeechStart: () => {
      console.log('[VAD] User started speaking');
      setIsSpeaking(true);
      // Gui frame dau tien ngay lap tuc
      captureAndSend();
      // Bat dau throttle: chi gui 2 frame / giay (moi 500ms) khi dang noi
      captureIntervalRef.current = setInterval(captureAndSend, 500);
    },
    onSpeechEnd: () => {
      console.log('[VAD] User stopped speaking');
      setIsSpeaking(false);
      // Dung gui frame khi het noi
      if (captureIntervalRef.current) {
        clearInterval(captureIntervalRef.current);
        captureIntervalRef.current = null;
      }
    },
    onVADMisfire: () => {
      setIsSpeaking(false);
      if (captureIntervalRef.current) {
        clearInterval(captureIntervalRef.current);
        captureIntervalRef.current = null;
      }
    }
  });

  return (
    <div style={{ position: 'relative' }}>
      <video ref={videoRef} autoPlay playsInline muted style={{ width: '100%', maxWidth: '640px' }} />
      {/* Canvas an dung de lay frame */}
      <canvas ref={canvasRef} style={{ display: 'none' }} />
      <div style={{
        position: 'absolute', top: 10, right: 10, padding: '8px 16px',
        background: isSpeaking ? '#ef4444' : '#22c55e',
        color: 'white', borderRadius: '4px', fontWeight: 'bold'
      }}>
        VAD: {isSpeaking ? 'SPEAKING' : 'Silent'}
      </div>
    </div>
  );
};

export default ExamMonitor;
"""

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  PHẦN 2: BACKEND – ExamFaceDetector nâng cấp (MAR + Head Pose)          ║
# ╚════════════════════════════════════════════════════════════════════════════╝


import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Any

import cv2
import numpy as np

# Type aliases
BBox = List[int]
OnMultipleFacesCB = Callable[[np.ndarray, int], None]
OnNoFaceCB = Callable[[np.ndarray], None]
OnNormalCB = Callable[[np.ndarray, BBox], None]


@dataclass
class DetectionResult:
    """Kết quả detect của một lần inference."""
    face_count: int = 0
    bboxes: List[BBox] = field(default_factory=list)
    timestamp: float = field(default_factory=time.monotonic)


@dataclass
class MultimodalResult:
    """Kết quả phân tích đa phương thức (hình ảnh + âm thanh)."""
    face_count: int = 0
    bboxes: List[BBox] = field(default_factory=list)
    # MAR (Mouth Aspect Ratio)
    is_mouth_open: bool = False
    mar_value: float = 0.0
    # Head Pose
    is_looking_away: bool = False
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0
    # Landmarks raw (cho debug)
    has_landmarks: bool = False


class ExamFaceDetector:
    """
    Giám sát khuôn mặt + phân tích MAR & Head Pose cho hệ thống chống gian lận.

    Nâng cấp so với bản gốc:
      - Bật module 'landmark_2d_106' để có 106 điểm landmark
      - Hàm calculate_mar() tính Mouth Aspect Ratio
      - Hàm analyze_pose() phân tích góc quay đầu
    """

    _PREPROCESS_W: int = 640
    _PREPROCESS_H: int = 360

    def __init__(
        self,
        model_root: str = "~/.insightface",
        on_multiple_faces: Optional[OnMultipleFacesCB] = None,
        on_no_face: Optional[OnNoFaceCB] = None,
        on_normal: Optional[OnNormalCB] = None,
        det_size: Tuple[int, int] = (640, 640),
        det_thresh: float = 0.5,
        throttle_interval: float = 0.25,
        ctx_id: int = -1,
        # ── Ngưỡng mới cho Multimodal ──
        mar_threshold: float = 0.20,
        yaw_threshold: float = 30.0,
        pitch_threshold: float = 20.0,
    ) -> None:
        self._on_multiple_faces = on_multiple_faces
        self._on_no_face = on_no_face
        self._on_normal = on_normal

        self._det_size = det_size
        self._det_thresh = det_thresh
        self._throttle_interval = throttle_interval
        self._ctx_id = ctx_id
        self._model_root = model_root

        # ── Ngưỡng cấu hình (có thể chỉnh runtime) ──
        # MAR > 0.20 → miệng mở. Tăng nếu false-positive nhiều.
        self.mar_threshold: float = mar_threshold
        # |yaw| > 30° → quay ngang. Giảm nếu muốn nhạy hơn.
        self.yaw_threshold: float = yaw_threshold
        # |pitch| > 20° → cúi/ngửa quá mức.
        self.pitch_threshold: float = pitch_threshold

        # Throttle state
        self._last_inference_time: float = 0.0
        self._last_result: DetectionResult = DetectionResult()
        self._last_multimodal: MultimodalResult = MultimodalResult()

        # Lazy-load
        self._app = None

    # ── Model loading ──────────────────────────────────────────────────────
    def _load_model(self) -> None:
        """Lazy-load InsightFace với detection + landmark_2d_106."""
        if self._app is not None:
            return

        from insightface.app import FaceAnalysis

        self._app = FaceAnalysis(
            name="buffalo_l",
            root=self._model_root,
            # QUAN TRỌNG: Thêm 'landmark_2d_106' để có 106 điểm landmark
            allowed_modules=["detection", "landmark_2d_106"],
        )
        self._app.prepare(
            ctx_id=self._ctx_id,
            det_thresh=self._det_thresh,
            det_size=self._det_size,
        )

    # ── Preprocessing ──────────────────────────────────────────────────────
    @staticmethod
    def _preprocess(frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        tw, th = ExamFaceDetector._PREPROCESS_W, ExamFaceDetector._PREPROCESS_H
        if w == tw and h == th:
            return frame
        return cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA)

    # ── MAR Calculation ────────────────────────────────────────────────────
    def calculate_mar(self, landmarks: Optional[np.ndarray]) -> Tuple[bool, float]:
        """
        Tính Mouth Aspect Ratio (MAR) từ InsightFace 106-point landmarks.

        Công thức:
            MAR = khoảng_cách_dọc_môi / khoảng_cách_ngang_khóe_miệng

        InsightFace 106-point landmark indices cho vùng miệng:
            - Khóe miệng trái:  điểm 84
            - Khóe miệng phải:  điểm 90
            - Môi trên (giữa):  điểm 87
            - Môi dưới (giữa):  điểm 93

        Returns:
            (is_mouth_open, mar_value)
        """
        if landmarks is None or len(landmarks) < 106:
            return False, 0.0

        # Khóe miệng trái và phải
        left_corner = landmarks[84]   # Khóe trái
        right_corner = landmarks[90]  # Khóe phải

        # Khoảng cách ngang (horizontal) giữa 2 khóe miệng
        horizontal = np.linalg.norm(left_corner - right_corner)
        if horizontal < 1e-6:
            return False, 0.0

        # Điểm giữa môi trên và môi dưới
        top_lip_center = landmarks[87]     # Giữa môi trên
        bottom_lip_center = landmarks[93]  # Giữa môi dưới

        # Khoảng cách dọc (vertical) giữa môi trên và môi dưới
        vertical = np.linalg.norm(top_lip_center - bottom_lip_center)

        mar = float(vertical / horizontal)
        is_mouth_open = mar > self.mar_threshold

        return is_mouth_open, mar

    # ── Head Pose Analysis ─────────────────────────────────────────────────
    def analyze_pose(self, pose: Optional[np.ndarray]) -> Tuple[bool, Tuple[float, float, float]]:
        """
        Phân tích góc quay đầu từ InsightFace.

        InsightFace trả về pose = [pitch, yaw, roll] (đơn vị: độ).
          - pitch: cúi (âm) / ngửa (dương)
          - yaw:   quay trái (âm) / quay phải (dương)
          - roll:  nghiêng đầu

        Returns:
            (is_looking_away, (pitch, yaw, roll))
        """
        if pose is None or len(pose) < 3:
            return False, (0.0, 0.0, 0.0)

        pitch, yaw, roll = float(pose[0]), float(pose[1]), float(pose[2])

        # Thí sinh "nhìn đi chỗ khác" khi yaw hoặc pitch vượt ngưỡng
        is_looking_away = abs(yaw) > self.yaw_threshold or abs(pitch) > self.pitch_threshold

        return is_looking_away, (pitch, yaw, roll)

    # ── Inference ──────────────────────────────────────────────────────────
    def _run_inference(self, small_frame: np.ndarray) -> Tuple[DetectionResult, MultimodalResult]:
        """Chạy InsightFace và trích xuất BBox + MAR + Pose."""
        self._load_model()

        try:
            faces = self._app.get(small_frame)
        except Exception:
            return DetectionResult(), MultimodalResult()

        bboxes: List[BBox] = []
        for face in faces:
            bboxes.append(face.bbox.astype(int).tolist())

        det = DetectionResult(face_count=len(bboxes), bboxes=bboxes, timestamp=time.monotonic())

        # Phan tich MAR va Pose cho khuon mat LON NHAT (thi sinh chinh)
        # Chon face co bounding box lon nhat thay vi faces[0]
        mm = MultimodalResult(face_count=len(bboxes), bboxes=bboxes)

        if faces:
            face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))

            # MAR từ landmark_2d_106
            lmk = getattr(face, "landmark_2d_106", None)
            if lmk is not None:
                mm.has_landmarks = True
                mm.is_mouth_open, mm.mar_value = self.calculate_mar(lmk)

            # Head Pose
            pose = getattr(face, "pose", None)
            if pose is not None:
                mm.is_looking_away, (mm.pitch, mm.yaw, mm.roll) = self.analyze_pose(pose)

        return det, mm

    # ── Callback dispatcher ────────────────────────────────────────────────
    def _dispatch(self, frame: np.ndarray, result: DetectionResult) -> None:
        count = result.face_count
        if count == 0:
            if self._on_no_face:
                self._on_no_face(frame)
        elif count == 1:
            if self._on_normal:
                self._on_normal(frame, result.bboxes[0])
        else:
            if self._on_multiple_faces:
                self._on_multiple_faces(frame, count)

    # ── Public API ─────────────────────────────────────────────────────────
    def process_frame(self, frame: np.ndarray) -> DetectionResult:
        """Xử lý frame (có throttle). Tương thích API cũ."""
        now = time.monotonic()
        if now - self._last_inference_time >= self._throttle_interval:
            small = self._preprocess(frame)
            self._last_result, self._last_multimodal = self._run_inference(small)
            self._last_inference_time = now

        self._dispatch(frame, self._last_result)
        return self._last_result

    def process_frame_multimodal(self, frame: np.ndarray) -> MultimodalResult:
        """Xử lý frame và trả về kết quả đa phương thức (MAR + Pose)."""
        now = time.monotonic()
        if now - self._last_inference_time >= self._throttle_interval:
            small = self._preprocess(frame)
            self._last_result, self._last_multimodal = self._run_inference(small)
            self._last_inference_time = now

        self._dispatch(frame, self._last_result)
        return self._last_multimodal

    def reset(self) -> None:
        self._last_inference_time = 0.0
        self._last_result = DetectionResult()
        self._last_multimodal = MultimodalResult()

    @property
    def last_result(self) -> DetectionResult:
        return self._last_result

    @property
    def last_multimodal(self) -> MultimodalResult:
        return self._last_multimodal

    @staticmethod
    def decode_base64_image(base64_str: str) -> np.ndarray:
        """
        Decode anh base64 (tu frontend) thanh numpy array BGR.
        Ho tro ca format 'data:image/jpeg;base64,...' va raw base64.
        """
        import base64 as b64mod
        # Tach header 'data:image/jpeg;base64,' neu co
        if "," in base64_str:
            base64_str = base64_str.split(",", 1)[1]
        img_data = b64mod.b64decode(base64_str)
        np_arr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return frame


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  PHẦN 3: CROSS-CHECK CONTROLLER – Logic xác nhận chéo                   ║
# ╚════════════════════════════════════════════════════════════════════════════╝

@dataclass
class CheatingVerdict:
    """Kết quả phân tích hành vi gian lận."""
    status: str       # Mã trạng thái (NORMAL, MILD_WARNING, WARNING_L1, WARNING_L2)
    message: str      # Mô tả tiếng Việt
    level: int         # 0=bình thường, 1=nhẹ, 2=mức 1, 3=mức 2 khẩn cấp
    details: Dict[str, Any] = field(default_factory=dict)


def analyze_cheating_behavior(
    frame: np.ndarray,
    speech_detected: bool,
    detector: ExamFaceDetector,
) -> CheatingVerdict:
    """
    Hàm xác nhận chéo (Cross-check) giữa VAD (âm thanh) và Camera (hình ảnh).

    Ma trận logic:
    ┌──────────────────┬────────────────┬──────────────────┬─────────────────────────────┐
    │ speech_detected  │ is_mouth_open  │ is_looking_away  │ Kết luận                    │
    ├──────────────────┼────────────────┼──────────────────┼─────────────────────────────┤
    │ False            │ *              │ *                │ Bình thường (bỏ qua)        │
    │ True             │ True           │ False            │ Cảnh báo nhẹ (đọc nhẩm)    │
    │ True             │ True           │ True             │ Mức 1 (quay sang nói)       │
    │ True             │ False          │ *                │ Mức 2 (người khác nhắc bài) │
    └──────────────────┴────────────────┴──────────────────┴─────────────────────────────┘

    Args:
        frame: Frame BGR từ camera/upload.
        speech_detected: Cờ VAD từ client (True = có tiếng nói).
        detector: Instance ExamFaceDetector đã khởi tạo.

    Returns:
        CheatingVerdict chứa status, message, level và details.
    """
    # ── Không có tiếng nói → Bình thường ──
    if not speech_detected:
        return CheatingVerdict(
            status="NORMAL",
            message="Bình thường – không phát hiện tiếng nói.",
            level=0,
        )

    # ── Có tiếng nói → Phân tích hình ảnh ──
    mm: MultimodalResult = detector.process_frame_multimodal(frame)

    details: Dict[str, Any] = {
        "face_count": mm.face_count,
        "mar": mm.mar_value,
        "is_mouth_open": mm.is_mouth_open,
        "is_looking_away": mm.is_looking_away,
        "pose": {"pitch": mm.pitch, "yaw": mm.yaw, "roll": mm.roll},
        "has_landmarks": mm.has_landmarks,
    }

    # Không thấy mặt nhưng có tiếng nói → cũng đáng ngờ
    if mm.face_count == 0:
        return CheatingVerdict(
            status="WARNING_NO_FACE",
            message="Có tiếng nói nhưng không tìm thấy khuôn mặt trong khung hình.",
            level=2,
            details=details,
        )

    # ── Áp dụng ma trận logic ──

    if mm.is_mouth_open and not mm.is_looking_away:
        # speech=True, mouth=open, looking=straight → đọc nhẩm
        return CheatingVerdict(
            status="MILD_WARNING",
            message="Cảnh báo nhẹ: Thí sinh đang đọc nhẩm.",
            level=1,
            details=details,
        )

    if mm.is_mouth_open and mm.is_looking_away:
        # speech=True, mouth=open, looking=away → quay sang nói chuyện
        return CheatingVerdict(
            status="WARNING_LEVEL_1",
            message="Cảnh báo Mức 1: Thí sinh đang quay sang nói chuyện.",
            level=2,
            details=details,
        )

    if not mm.is_mouth_open:
        # speech=True, mouth=closed → có người khác nhắc bài
        return CheatingVerdict(
            status="WARNING_LEVEL_2_URGENT",
            message="Cảnh báo Mức 2 Khẩn cấp: Có tiếng người khác nhắc bài!",
            level=3,
            details=details,
        )

    # Fallback (khong bao gio den day theo logic tren)
    return CheatingVerdict(status="UNKNOWN", message="Khong xac dinh.", level=0, details=details)


class ExamCheatController:
    """
    Controller tong hop - dung cho WebSocket/API endpoint.
    Nhan payload JSON tu frontend (base64 image + speech_detected),
    decode anh, chay InsightFace, ap dung ma tran logic.

    Su dung:
        controller = ExamCheatController(mar_threshold=0.2, yaw_threshold=20.0)
        result = controller.process_payload(payload_dict)
    """

    def __init__(
        self,
        model_root: str = "~/.insightface",
        ctx_id: int = -1,
        mar_threshold: float = 0.20,
        yaw_threshold: float = 20.0,
        pitch_threshold: float = 20.0,
    ) -> None:
        self.detector = ExamFaceDetector(
            model_root=model_root,
            ctx_id=ctx_id,
            mar_threshold=mar_threshold,
            yaw_threshold=yaw_threshold,
            pitch_threshold=pitch_threshold,
            throttle_interval=0.0,  # Khong throttle vi frontend da throttle 500ms
        )

    def process_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Xu ly payload JSON tu WebSocket/API.

        payload = {
            "speech_detected": true,
            "image": "data:image/jpeg;base64,...",
            "timestamp": 1234567890
        }
        """
        speech_detected = payload.get("speech_detected", False)

        # Khong co tieng noi -> bo qua, tiet kiem tai nguyen
        if not speech_detected:
            return {"status": "NORMAL", "message": "Binh thuong", "level": 0}

        base64_image = payload.get("image")
        if not base64_image:
            return {"status": "ERROR", "message": "Thieu frame anh", "level": -1}

        # Decode base64 -> numpy BGR
        frame = ExamFaceDetector.decode_base64_image(base64_image)
        if frame is None:
            return {"status": "ERROR", "message": "Khong the decode anh", "level": -1}

        # Chay cross-check
        verdict = analyze_cheating_behavior(frame, speech_detected, self.detector)
        return {
            "status": verdict.status,
            "message": verdict.message,
            "level": verdict.level,
            "details": verdict.details,
        }


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  CAMERA MONITOR GUI – Giao dien giam sat thoi gian thuc                 ║
# ╚════════════════════════════════════════════════════════════════════════════╝

import argparse
import os
import sys
import threading
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# ── Mau sac overlay ───────────────────────────────────────────────────────────
_COLOR_NORMAL = (0, 220, 80)       # Xanh la  - binh thuong
_COLOR_NO_FACE = (60, 60, 220)     # Do       - khong co mat
_COLOR_MULTIPLE = (0, 165, 255)    # Cam      - nhieu mat
_COLOR_WARNING_L1 = (0, 255, 255)  # Vang     - canh bao nhe
_COLOR_WARNING_L2 = (0, 140, 255)  # Cam dam  - muc 1
_COLOR_WARNING_L3 = (0, 0, 255)    # Do       - muc 2 khan cap

_DET_SIZE = (640, 640)
_PREPROCESS_SIZE = (640, 360)

# ── Shared overlay state (thread-safe) ─────────────────────────────────────
_overlay_lock = threading.Lock()
_overlay_state: dict = {
    "label": "Initializing...",
    "color": (200, 200, 200),
    "bbox": None,
    "face_count": 0,
    "last_update": 0.0,
    "verdict": None,         # CheatingVerdict hien tai
    "multimodal": None,      # MultimodalResult hien tai
    "speech_on": False,
}

# ── Callbacks cho ExamFaceDetector ─────────────────────────────────────────

def _on_multiple_faces(frame: np.ndarray, face_count: int) -> None:
    with _overlay_lock:
        _overlay_state["label"] = f"CANH BAO: {face_count} khuon mat!"
        _overlay_state["color"] = _COLOR_MULTIPLE
        _overlay_state["face_count"] = face_count
        _overlay_state["bbox"] = None
        _overlay_state["last_update"] = time.monotonic()


def _on_no_face(frame: np.ndarray) -> None:
    with _overlay_lock:
        _overlay_state["label"] = "CANH BAO: Khong co thi sinh!"
        _overlay_state["color"] = _COLOR_NO_FACE
        _overlay_state["face_count"] = 0
        _overlay_state["bbox"] = None
        _overlay_state["last_update"] = time.monotonic()


def _on_normal(frame: np.ndarray, bbox: BBox) -> None:
    with _overlay_lock:
        _overlay_state["label"] = "Binh thuong"
        _overlay_state["color"] = _COLOR_NORMAL
        _overlay_state["face_count"] = 1
        _overlay_state["bbox"] = bbox
        _overlay_state["last_update"] = time.monotonic()


# ── Font helpers ───────────────────────────────────────────────────────────

def _load_unicode_font(size: int = 22) -> ImageFont.FreeTypeFont:
    """Tai font ho tro Unicode (tieng Viet)."""
    candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/tahoma.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    ]
    for fp in candidates:
        if os.path.isfile(fp):
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                continue
    return ImageFont.load_default()


# Cache font
_FONT_BANNER = _load_unicode_font(24)
_FONT_LABEL = _load_unicode_font(18)
_FONT_SMALL = _load_unicode_font(14)


def _put_text_pil(
    draw: ImageDraw.ImageDraw,
    text: str,
    pos: tuple,
    font: ImageFont.FreeTypeFont,
    color_bgr: tuple,
) -> None:
    """Ve text Unicode dung ImageDraw co san (tiet kiem chuyen doi)."""
    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
    draw.text(pos, text, font=font, fill=color_rgb)


# ── Bbox normalization ────────────────────────────────────────────────────

def _normalize_bbox_to_frame(
    bbox: BBox, frame_w: int, frame_h: int
) -> tuple:
    """Map bbox tu he toa do detector (640x640) ve frame goc."""
    x1, y1, x2, y2 = map(int, bbox)
    det_w, det_h = _DET_SIZE
    pp_w, pp_h = _PREPROCESS_SIZE

    lb_scale = min(det_w / float(pp_w), det_h / float(pp_h))
    pad_x = (det_w - pp_w * lb_scale) / 2.0
    pad_y = (det_h - pp_h * lb_scale) / 2.0

    pp_x1 = (x1 - pad_x) / lb_scale
    pp_y1 = (y1 - pad_y) / lb_scale
    pp_x2 = (x2 - pad_x) / lb_scale
    pp_y2 = (y2 - pad_y) / lb_scale

    sx = frame_w / float(pp_w)
    sy = frame_h / float(pp_h)

    return (
        max(0, int(pp_x1 * sx)),
        max(0, int(pp_y1 * sy)),
        min(frame_w, int(pp_x2 * sx)),
        min(frame_h, int(pp_y2 * sy)),
    )


# ── Draw overlay ──────────────────────────────────────────────────────────

def _level_color(level: int) -> tuple:
    """Tra ve mau BGR theo muc canh bao."""
    if level <= 0:
        return _COLOR_NORMAL
    elif level == 1:
        return _COLOR_WARNING_L1
    elif level == 2:
        return _COLOR_WARNING_L2
    else:
        return _COLOR_WARNING_L3


def _draw_overlay(frame: np.ndarray) -> np.ndarray:
    """Ve status label, bbox, va panel multimodal len frame (da toi uu)."""
    display = frame.copy()
    h, w = display.shape[:2]

    with _overlay_lock:
        label = _overlay_state["label"]
        color = _overlay_state["color"]
        bbox = _overlay_state["bbox"]
        verdict: Optional[CheatingVerdict] = _overlay_state["verdict"]
        mm: Optional[MultimodalResult] = _overlay_state["multimodal"]
        speech_on = _overlay_state["speech_on"]

    # 1. Ve cac hinh khoi bang OpenCV (nhanh hon PIL)
    # Top banner
    overlay = display.copy()
    cv2.rectangle(overlay, (0, 0), (w, 50), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)

    # Bounding box
    if bbox is not None:
        x1, y1, x2, y2 = _normalize_bbox_to_frame(bbox, w, h)
        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

    # Panel Multimodal
    panel_w, panel_h = 420, 130
    panel_y, panel_x = h - panel_h - 10, 10
    panel_overlay = display.copy()
    cv2.rectangle(panel_overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (20, 20, 20), -1)
    cv2.addWeighted(panel_overlay, 0.7, display, 0.3, 0, display)

    # 2. Chuyen sang PIL 1 LAN DUY NHAT de ve toan bo text Unicode
    pil_img = Image.fromarray(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    # Ve status label
    _put_text_pil(draw, label, (16, 12), _FONT_BANNER, color)

    # Ve nhan "Thi sinh"
    if bbox is not None:
        x1, y1, x2, y2 = _normalize_bbox_to_frame(bbox, w, h)
        _put_text_pil(draw, "Thi sinh", (x1, max(y1 - 24, 2)), _FONT_LABEL, color)

    # Ve thong tin speech
    speech_color = (0, 0, 255) if speech_on else (0, 220, 80)
    speech_text = "[S] Speech: ON" if speech_on else "[S] Speech: OFF"
    _put_text_pil(draw, speech_text, (panel_x + 8, panel_y + 8), _FONT_SMALL, speech_color)

    if mm is not None and mm.face_count > 0:
        mar_text = f"MAR: {mm.mar_value:.3f} | Mouth: {'OPEN' if mm.is_mouth_open else 'Closed'}"
        _put_text_pil(draw, mar_text, (panel_x + 8, panel_y + 32), _FONT_SMALL, (200, 200, 200))
        pose_text = f"Yaw: {mm.yaw:.1f}  Pitch: {mm.pitch:.1f}  Look: {'AWAY' if mm.is_looking_away else 'OK'}"
        _put_text_pil(draw, pose_text, (panel_x + 8, panel_y + 56), _FONT_SMALL, (0, 0, 255) if mm.is_looking_away else (0, 220, 80))
    else:
        _put_text_pil(draw, "MAR: --- | Pose: ---", (panel_x + 8, panel_y + 32), _FONT_SMALL, (120, 120, 120))

    if verdict is not None and speech_on:
        _put_text_pil(draw, f"[Lv.{verdict.level}] {verdict.message}", (panel_x + 8, panel_y + 85), _FONT_LABEL, _level_color(verdict.level))
    else:
        _put_text_pil(draw, "Binh thuong", (panel_x + 8, panel_y + 85), _FONT_LABEL, _COLOR_NORMAL)

    _put_text_pil(draw, "Q: Thoat | S: Toggle Speech", (w - 250, h - 25), _FONT_SMALL, (150, 150, 150))

    # 3. Chuyen nguoc lai BGR
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# ── Main monitor loop ─────────────────────────────────────────────────────

def run_monitor(
    camera_index: int = 0,
    model_root: str = "~/.insightface",
    ctx_id: int = -1,
    throttle_interval: float = 0.25,
    simulate_speech: bool = False,
) -> None:
    """
    Khoi chay vong lap giam sat camera voi giao dien multimodal (Threaded).
    """
    print("[ExamMonitor] Khoi tao ExamFaceDetector (Threaded)...")

    detector = ExamFaceDetector(
        model_root=model_root,
        on_multiple_faces=_on_multiple_faces,
        on_no_face=_on_no_face,
        on_normal=_on_normal,
        det_size=_DET_SIZE,
        det_thresh=0.5,
        throttle_interval=throttle_interval,
        ctx_id=ctx_id,
    )

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Khong the mo camera index={camera_index}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # State cho threading
    shared_data = {
        "frame": None,
        "speech_on": simulate_speech,
        "running": True
    }

    def inference_worker():
        """Luong xu ly AI chay song song."""
        print("[ExamMonitor] Inference thread started.")
        while shared_data["running"]:
            if shared_data["frame"] is not None:
                # Lay frame moi nhat de xu ly
                f = shared_data["frame"].copy()
                s_on = shared_data["speech_on"]
                
                # Chay AI
                mm = detector.process_frame_multimodal(f)
                verdict = analyze_cheating_behavior(f, s_on, detector)
                
                # Cap nhat overlay state
                with _overlay_lock:
                    _overlay_state["verdict"] = verdict
                    _overlay_state["multimodal"] = mm
                    _overlay_state["speech_on"] = s_on
            
            time.sleep(0.01)

    # Chay thread inference
    thread = threading.Thread(target=inference_worker, daemon=True)
    thread.start()

    frame_count = 0
    fps_timer = time.monotonic()

    print("[ExamMonitor] Camera san sang. Nhan 'q' de thoat, 's' de toggle speech.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            # Day frame vao queue cho thread inference
            shared_data["frame"] = frame

            # Tinh FPS hiển thị
            frame_count += 1
            elapsed = time.monotonic() - fps_timer
            if elapsed >= 2.0:
                fps = frame_count / elapsed
                print(f"[ExamMonitor] Display FPS: {fps:.1f}")
                frame_count = 0
                fps_timer = time.monotonic()

            # Hien thi luon frame hien tai voi ket qua AI moi nhat (khong cho doi)
            display = _draw_overlay(frame)
            cv2.imshow("SecureExam - Multimodal AI (Optimized)", display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            if key == ord("s"):
                shared_data["speech_on"] = not shared_data["speech_on"]
                print(f"[ExamMonitor] Speech: {'ON' if shared_data['speech_on'] else 'OFF'}")

    finally:
        shared_data["running"] = False
        thread.join(timeout=1.0)
        cap.release()
        cv2.destroyAllWindows()
        print("[ExamMonitor] Da dung.")


# ── CLI entry point ───────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SecureExam - Giam sat khuon mat ky thi (Multimodal AI)"
    )
    parser.add_argument("--camera", type=int, default=0, help="Index camera (mac dinh: 0)")
    parser.add_argument("--gpu", action="store_true", help="Dung GPU (CUDA), mac dinh CPU")
    parser.add_argument(
        "--model-root",
        default=str(Path(__file__).resolve().parent / "models_cache"),
        help="Thu muc model InsightFace",
    )
    parser.add_argument(
        "--throttle", type=float, default=0.25,
        help="Khoang cach giua 2 lan inference (giay, mac dinh 0.25)",
    )
    parser.add_argument(
        "--simulate-speech", action="store_true",
        help="Bat dau voi speech=ON de test logic cross-check",
    )
    args = parser.parse_args()

    run_monitor(
        camera_index=args.camera,
        model_root=args.model_root,
        ctx_id=0 if args.gpu else -1,
        throttle_interval=args.throttle,
        simulate_speech=args.simulate_speech,
    )