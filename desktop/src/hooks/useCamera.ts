/**
 * src/hooks/useCamera.ts
 * ───────────────────────
 * Custom hook that manages camera capture and periodic frame analysis.
 * Captures one frame every N seconds and sends to AI service for analysis.
 */

import { useRef, useState, useEffect, useCallback } from "react";

const api = (window as any).electronAPI;

export interface CameraStatus {
  isRunning: boolean;
  faceCount: number;
  faceDetected: boolean;
  multipleFaces: boolean;
  identityMatch: boolean | null; // null = not checked
  identityDistance: number;
  lastError: string | null;
}

interface Options {
  sessionId: string;
  userId: string;
  referenceEmbeddingB64?: string;
  captureIntervalSeconds?: number;
  onViolation?: (type: string, severity: string, metadata: Record<string, unknown>) => void;
}

export function useCamera({
  sessionId,
  userId,
  referenceEmbeddingB64,
  captureIntervalSeconds = 2,
  onViolation,
}: Options) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  // Tracks how many consecutive frames had no face (for threshold logic)
  const noFaceCountRef = useRef(0);

  const [status, setStatus] = useState<CameraStatus>({
    isRunning: false,
    faceCount: 0,
    faceDetected: false,
    multipleFaces: false,
    identityMatch: null,
    identityDistance: 1,
    lastError: null,
  });

  // ── Start camera ────────────────────────────────────────────────
  const startCamera = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: "user" },
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setStatus(s => ({ ...s, isRunning: true, lastError: null }));
    } catch (e: any) {
      const msg = "Camera không khả dụng: " + (e.message || "quyền bị từ chối");
      setStatus(s => ({ ...s, lastError: msg }));
      onViolation?.("camera_unavailable", "high", { error: e.message });
    }
  }, [onViolation]);

  // ── Stop camera ─────────────────────────────────────────────────
  const stopCamera = useCallback(() => {
    streamRef.current?.getTracks().forEach(t => t.stop());
    streamRef.current = null;
    if (intervalRef.current) clearInterval(intervalRef.current);
    setStatus(s => ({ ...s, isRunning: false }));
  }, []);

  // ── Capture and analyze one frame ───────────────────────────────
  const captureFrame = useCallback(async () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || video.readyState < 2) return;

    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    canvas.getContext("2d")!.drawImage(video, 0, 0);
    const frameB64 = canvas.toDataURL("image/jpeg", 0.7).split(",")[1];

    try {
      const result = await api.analyzeFrame(frameB64, referenceEmbeddingB64);

      setStatus(s => ({
        ...s,
        faceCount: result.face_count,
        faceDetected: result.face_detected,
        multipleFaces: result.multiple_faces,
        identityMatch: result.identity_checked ? result.identity_match : null,
        identityDistance: result.identity_distance,
      }));

      // ── Violation logic ─────────────────────────────────────────
      if (!result.face_detected) {
        noFaceCountRef.current += 1;
        const consecutiveMiss = noFaceCountRef.current * captureIntervalSeconds;
        if (consecutiveMiss >= 30) {
          onViolation?.("no_face", "high", { seconds_missing: consecutiveMiss });
        } else if (consecutiveMiss >= 10) {
          onViolation?.("no_face", "medium", { seconds_missing: consecutiveMiss });
        }
      } else {
        noFaceCountRef.current = 0;
      }

      if (result.multiple_faces) {
        onViolation?.("multiple_faces", "critical", { face_count: result.face_count });
      }

      if (result.identity_checked && !result.identity_match) {
        onViolation?.("identity_mismatch", "critical", {
          distance: result.identity_distance,
        });
      }
    } catch (e: any) {
      console.warn("[useCamera] Analysis error:", e.message);
      onViolation?.("camera_analysis_failed", "medium", { error: e.message });
    }
  }, [referenceEmbeddingB64, captureIntervalSeconds, onViolation]);

  // ── Start periodic capture when camera is running ───────────────
  useEffect(() => {
    if (!status.isRunning) return;
    intervalRef.current = setInterval(captureFrame, captureIntervalSeconds * 1000);
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [status.isRunning, captureFrame, captureIntervalSeconds]);

  return { videoRef, canvasRef, status, startCamera, stopCamera };
}
