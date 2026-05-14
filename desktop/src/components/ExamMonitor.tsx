import React, { useRef, useState, useCallback, useEffect } from "react";
import { useMicVAD } from "@ricky0123/vad-react";
import { useRollingBuffer } from "../hooks/useRollingBuffer";

interface ExamMonitorProps {
  webSocketUrl: string;
  sessionId: string;
  onVerdict?: (verdict: MonitorVerdict, captureViolationClip: (violationId: string) => void) => void;
}

export interface MonitorVerdict {
  status: string;
  message: string;
  level: number;
  details?: Record<string, unknown>;
}

type MonitorPayload = {
  speech_detected: boolean;
  timestamp: number;
  sequence: number;
  image: string;
};

const STATUS_LABELS: Record<string, string> = {
  NORMAL: "Bình thường",
  MILD_WARNING: "Cảnh báo nhẹ",
  WARNING_LEVEL_1: "Cảnh báo mức 1",
  WARNING_LEVEL_2_URGENT: "Cảnh báo mức 2",
};

const levelColors: Record<number, string> = {
  0: "#45d59a",
  1: "#ffbf5f",
  2: "#ff9b5c",
  3: "#ff5d7d",
};

const FRAME_WIDTH = 512;
const FRAME_HEIGHT = 384;
const FRAME_QUALITY = 0.6;
const IDLE_CAPTURE_INTERVAL_MS = 1500;
const SPEAKING_CAPTURE_INTERVAL_MS = 700;
const VERDICT_TIMEOUT_MS = 5000;
const MAX_SOCKET_BUFFERED_BYTES = 512 * 1024;
const api = (window as any).electronAPI;

const ExamMonitor: React.FC<ExamMonitorProps> = ({ webSocketUrl, sessionId, onVerdict }) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const speechDebounceRef = useRef<NodeJS.Timeout | null>(null);
  const isSpeakingRef = useRef(false);
  const captureIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectDelayRef = useRef(2000);
  const awaitingVerdictRef = useRef(false);
  const verdictTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const sequenceRef = useRef(0);

  const [isSpeaking, setIsSpeaking] = useState(false);
  const [verdict, setVerdict] = useState<MonitorVerdict | null>(null);
  const [wsConnected, setWsConnected] = useState(false);
  const [fallbackConnected, setFallbackConnected] = useState(false);

  const { startRecording, captureViolationClip } = useRollingBuffer({
    sessionId,
    bufferDuration: 5,
    chunkIntervalMs: 1000,
  });

  const clearVerdictTimeout = useCallback(() => {
    if (verdictTimeoutRef.current) {
      clearTimeout(verdictTimeoutRef.current);
      verdictTimeoutRef.current = null;
    }
  }, []);

  const markVerdictSettled = useCallback(() => {
    awaitingVerdictRef.current = false;
    clearVerdictTimeout();
  }, [clearVerdictTimeout]);

  const armVerdictTimeout = useCallback(() => {
    clearVerdictTimeout();
    verdictTimeoutRef.current = setTimeout(() => {
      if (!awaitingVerdictRef.current) {
        return;
      }

      console.warn("[ExamMonitor] AI verdict timeout. Recycling WebSocket connection.");
      awaitingVerdictRef.current = false;

      const socket = wsRef.current;
      if (socket && socket.readyState === WebSocket.OPEN) {
        socket.close(4000, "verdict-timeout");
      }
    }, VERDICT_TIMEOUT_MS);
  }, [clearVerdictTimeout]);

  const handleVerdict = useCallback(
    (data: MonitorVerdict) => {
      setVerdict(data);
      onVerdict?.(data, captureViolationClip);
    },
    [captureViolationClip, onVerdict]
  );

  const sendViaHttpFallback = useCallback(
    async (payload: MonitorPayload) => {
      try {
        const data: MonitorVerdict = await api.analyzeMonitorFrame(sessionId, payload);
        markVerdictSettled();
        setFallbackConnected(true);
        handleVerdict(data);
      } catch (error) {
        markVerdictSettled();
        setFallbackConnected(false);
        console.warn("[ExamMonitor] HTTP fallback analysis failed:", error);
      }
    },
    [handleVerdict, markVerdictSettled, sessionId]
  );

  useEffect(() => {
    let reconnectTimer: NodeJS.Timeout;
    let isUnmounted = false;

    const connect = () => {
      if (isUnmounted) return;
      
      const socket = new WebSocket(webSocketUrl);

      socket.onopen = () => {
        console.log("[ExamMonitor] WebSocket connected");
        reconnectDelayRef.current = 2000;
        awaitingVerdictRef.current = false;
        clearVerdictTimeout();
        setWsConnected(true);
        setFallbackConnected(false);
      };

      socket.onmessage = (event) => {
        markVerdictSettled();
        try {
          const data: MonitorVerdict = JSON.parse(event.data);
          handleVerdict(data);
        } catch {
          console.warn("[ExamMonitor] Invalid verdict JSON");
        }
      };

      socket.onclose = () => {
        const delay = reconnectDelayRef.current;
        console.log(`[ExamMonitor] WebSocket disconnected. Reconnecting in ${delay}ms...`);
        setWsConnected(false);
        markVerdictSettled();
        wsRef.current = null;
        if (!isUnmounted) {
          reconnectTimer = setTimeout(connect, delay);
          reconnectDelayRef.current = Math.min(delay * 1.5, 5000);
        }
      };

      socket.onerror = (err) => {
        console.error("[ExamMonitor] WebSocket error:", err);
        // onclose is usually called immediately after onerror
      };

      wsRef.current = socket;
    };

    connect();

    return () => {
      isUnmounted = true;
      clearVerdictTimeout();
      if (reconnectTimer) clearTimeout(reconnectTimer);
      if (wsRef.current) {
        // Remove the onclose handler so it doesn't trigger a reconnect when the component unmounts
        wsRef.current.onclose = null;
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [clearVerdictTimeout, handleVerdict, markVerdictSettled, webSocketUrl]);

  useEffect(() => {
    navigator.mediaDevices
      .getUserMedia({
        video: { width: 854, height: 480, frameRate: 15 },
        audio: true,
      })
      .then((stream) => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
        startRecording(stream);
      })
      .catch(console.error);
  }, [startRecording]);

  const captureAndSend = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ws = wsRef.current;

    if (!video || !canvas) {
      return;
    }

    if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
      return;
    }

    if (awaitingVerdictRef.current) {
      return;
    }

    const canUseWebSocket = ws?.readyState === WebSocket.OPEN;

    if (canUseWebSocket && ws.bufferedAmount > MAX_SOCKET_BUFFERED_BYTES) {
      console.warn("[ExamMonitor] WebSocket buffer is saturated. Skipping this frame.");
      return;
    }

    const context = canvas.getContext("2d");
    if (!context) {
      return;
    }

    canvas.width = FRAME_WIDTH;
    canvas.height = FRAME_HEIGHT;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    const base64Image = canvas.toDataURL("image/jpeg", FRAME_QUALITY);
    sequenceRef.current += 1;
    const payload: MonitorPayload = {
      speech_detected: isSpeakingRef.current,
      timestamp: Date.now(),
      sequence: sequenceRef.current,
      image: base64Image,
    };

    try {
      awaitingVerdictRef.current = true;
      if (!canUseWebSocket) {
        void sendViaHttpFallback(payload);
        armVerdictTimeout();
        return;
      }

      ws.send(JSON.stringify(payload));
      armVerdictTimeout();
    } catch (error) {
      console.error("[ExamMonitor] Failed to send frame:", error);
      void sendViaHttpFallback(payload);
      armVerdictTimeout();
    }
  }, [armVerdictTimeout, sendViaHttpFallback]);

  const stopCapture = useCallback(() => {
    if (captureIntervalRef.current) {
      clearInterval(captureIntervalRef.current);
      captureIntervalRef.current = null;
    }
  }, []);

  const startCapture = useCallback(() => {
    stopCapture();
    captureAndSend();
    captureIntervalRef.current = setInterval(captureAndSend, SPEAKING_CAPTURE_INTERVAL_MS);
  }, [captureAndSend, stopCapture]);

  const vad = useMicVAD({
    startOnLoad: true,
    onnxWASMBasePath: "./",
    baseAssetPath: "./",
    model: "legacy",
    ortConfig: (ort: any) => {
      ort.env.wasm.numThreads = 1;
      ort.env.wasm.wasmPaths = {
        "ort-wasm-simd-threaded.wasm": "./ort-wasm-simd-threaded.wasm",
        "ort-wasm-simd.wasm": "./ort-wasm-simd-threaded.wasm",
        "ort-wasm.wasm": "./ort-wasm-simd-threaded.wasm",
        "ort-wasm-threaded.wasm": "./ort-wasm-simd-threaded.wasm",
      };
    },
    onSpeechStart: () => {
      console.log("[VAD] Speech started");
      isSpeakingRef.current = true;
      setIsSpeaking(true);
      startCapture();
    },
    onSpeechEnd: () => {
      if (isSpeakingRef.current) {
        console.log("[VAD] Speech ended");
        isSpeakingRef.current = false;
        setIsSpeaking(false);
        stopCapture();
        captureAndSend();
      }
    },
    onVADMisfire: () => {
      if (isSpeakingRef.current) {
        isSpeakingRef.current = false;
        setIsSpeaking(false);
        stopCapture();
        captureAndSend();
      }
    },
  });

  useEffect(() => {
    const bgTimer = setInterval(() => {
      if (!isSpeakingRef.current) {
        captureAndSend();
      }
    }, IDLE_CAPTURE_INTERVAL_MS);

    return () => clearInterval(bgTimer);
  }, [captureAndSend]);

  useEffect(() => {
    return () => {
      stopCapture();
      clearVerdictTimeout();
      if (speechDebounceRef.current) {
        clearTimeout(speechDebounceRef.current);
      }
    };
  }, [clearVerdictTimeout, stopCapture]);

  const verdictColor = verdict ? levelColors[verdict.level] || "#6b7280" : "#6b7280";
  const speechDetected = Boolean(verdict?.details?.speech_detected);
  const aiConnected = wsConnected || fallbackConnected;

  return (
    <div style={styles.shell}>
      <div style={styles.frame}>
        <video ref={videoRef} autoPlay playsInline muted style={styles.video} />
        <canvas ref={canvasRef} style={{ display: "none" }} />

        <div style={styles.topRow}>
          <div
            style={{
              ...styles.statusChip,
              background: aiConnected ? "rgba(69, 213, 154, 0.18)" : "rgba(109, 124, 146, 0.18)",
              color: aiConnected ? "#9cf0cc" : "#c8d3e3",
            }}
          >
            {aiConnected ? "Kết nối AI ổn định" : "Đang kết nối AI"}
          </div>

          <div
            style={{
              ...styles.statusChip,
              background: isSpeaking ? "rgba(255, 93, 125, 0.18)" : "rgba(69, 213, 154, 0.18)",
              color: isSpeaking ? "#ffb1c0" : "#9cf0cc",
            }}
          >
            {isSpeaking ? "Đang phát hiện giọng nói" : "Âm thanh yên tĩnh"}
          </div>
        </div>

        {vad.loading && <div style={styles.loadingHint}>Đang khởi tạo mô-đun nhận diện giọng nói...</div>}

        {vad.errored && (
          <div style={styles.errorHint}>
            Không thể khởi tạo microphone: {(vad.errored as any).message || String(vad.errored)}
          </div>
        )}

        {!vad.loading && !vad.errored && !vad.listening && (
          <button onClick={() => vad.start()} style={styles.micButton}>
            Bật lại microphone
          </button>
        )}

        {verdict && verdict.level > 0 && (
          <div style={{ ...styles.verdictPanel, borderLeftColor: verdictColor }}>
            <div style={{ ...styles.verdictHeader, color: verdictColor }}>
              {STATUS_LABELS[verdict.status] || verdict.status}
            </div>
            <div style={styles.verdictMessage}>{verdict.message}</div>
            <div style={styles.verdictMeta}>
              <span>Mức: {verdict.level}</span>
              <span>Số khuôn mặt: {String(verdict.details?.face_count ?? 0)}</span>
              <span>Âm thanh: {speechDetected ? "Có" : "Không"}</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

const styles: Record<string, React.CSSProperties> = {
  shell: {
    display: "flex",
    flexDirection: "column",
    gap: 12,
  },
  frame: {
    position: "relative",
    borderRadius: 20,
    overflow: "hidden",
    border: "1px solid rgba(158, 192, 245, 0.16)",
    background: "rgba(4, 11, 19, 0.92)",
    boxShadow: "0 20px 44px rgba(0, 0, 0, 0.28)",
  },
  video: {
    display: "block",
    width: "100%",
    maxWidth: 640,
    aspectRatio: "4 / 3",
    objectFit: "cover",
    background: "#02070d",
  },
  topRow: {
    position: "absolute",
    top: 12,
    left: 12,
    right: 12,
    display: "flex",
    justifyContent: "space-between",
    gap: 8,
    flexWrap: "wrap",
  },
  statusChip: {
    padding: "8px 12px",
    borderRadius: 999,
    border: "1px solid rgba(255, 255, 255, 0.08)",
    backdropFilter: "blur(10px)",
    fontSize: 12,
    fontWeight: 700,
  },
  loadingHint: {
    position: "absolute",
    left: 12,
    bottom: 12,
    padding: "8px 12px",
    borderRadius: 12,
    background: "rgba(9, 18, 32, 0.78)",
    color: "#d8e7ff",
    fontSize: 12,
  },
  errorHint: {
    position: "absolute",
    left: 12,
    bottom: 12,
    right: 12,
    padding: "10px 12px",
    borderRadius: 14,
    background: "rgba(255, 93, 125, 0.18)",
    border: "1px solid rgba(255, 93, 125, 0.26)",
    color: "#ffd1da",
    fontSize: 12,
    lineHeight: 1.5,
  },
  micButton: {
    position: "absolute",
    right: 12,
    bottom: 12,
    border: "none",
    borderRadius: 12,
    background: "linear-gradient(135deg, #57a6ff, #2dd4bf)",
    color: "#04111d",
    fontWeight: 700,
    padding: "10px 14px",
    cursor: "pointer",
  },
  verdictPanel: {
    position: "fixed",
    bottom: 16,
    right: 16,
    width: 218, // Fits within 250px side panel (250 - 16 - 16)
    zIndex: 9999,
    padding: "12px 16px",
    borderRadius: 12,
    background: "rgba(4, 11, 19, 0.95)",
    borderLeft: "4px solid transparent",
    boxShadow: "0 10px 40px rgba(0, 0, 0, 0.5)",
  },
  verdictHeader: {
    fontSize: 12,
    fontWeight: 800,
    textTransform: "uppercase",
    letterSpacing: "0.06em",
    marginBottom: 6,
  },
  verdictMessage: {
    fontSize: 13,
    color: "#f2f7ff",
    lineHeight: 1.5,
    marginBottom: 6,
  },
  verdictMeta: {
    display: "flex",
    flexWrap: "wrap",
    gap: 10,
    fontSize: 11,
    color: "#9fb0c7",
  },
};

export default ExamMonitor;
