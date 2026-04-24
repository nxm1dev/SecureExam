/**
 * src/components/CameraMonitor.tsx
 * ──────────────────────────────────
 * Renders the camera preview thumbnail and status indicators.
 */

import React from "react";
import { CameraStatus } from "../hooks/useCamera";

interface Props {
  videoRef: React.RefObject<HTMLVideoElement>;
  status: CameraStatus;
}

export default function CameraMonitor({ videoRef, status }: Props) {
  const faceStatusColor = !status.isRunning
    ? "var(--color-text-dim)"
    : status.lastError
    ? "var(--color-danger)"
    : status.multipleFaces
    ? "var(--color-critical)"
    : !status.faceDetected
    ? "var(--color-warning)"
    : status.identityMatch === false
    ? "var(--color-critical)"
    : "var(--color-success)";

  const faceStatusText = !status.isRunning
    ? "Chưa khởi động"
    : status.lastError
    ? "Lỗi camera"
    : status.multipleFaces
    ? `⚠️ ${status.faceCount} khuôn mặt`
    : !status.faceDetected
    ? "Không thấy mặt"
    : status.identityMatch === false
    ? "⚠️ Không khớp danh tính"
    : `✓ ${status.faceCount} khuôn mặt`;

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <span style={styles.label}>📷 Camera</span>
        <div style={{ ...styles.dot, background: faceStatusColor }} className={status.isRunning ? "pulse" : ""} />
      </div>

      {/* Video preview */}
      <div style={styles.videoWrapper}>
        <video
          ref={videoRef}
          muted
          playsInline
          style={styles.video}
        />
        {!status.isRunning && (
          <div style={styles.overlay}>Chưa khởi động</div>
        )}
      </div>

      {/* Status text */}
      <div style={{ ...styles.statusText, color: faceStatusColor }}>
        {faceStatusText}
      </div>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    background: "var(--color-surface)",
    border: "1px solid var(--color-border)",
    borderRadius: 12,
    padding: 12,
    display: "flex",
    flexDirection: "column",
    gap: 8,
  },
  header: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
  },
  label: {
    fontSize: 12,
    fontWeight: 600,
    color: "var(--color-text-muted)",
    textTransform: "uppercase",
    letterSpacing: "0.5px",
  },
  dot: {
    width: 10,
    height: 10,
    borderRadius: "50%",
  },
  videoWrapper: {
    position: "relative",
    width: "100%",
    aspectRatio: "4/3",
    borderRadius: 8,
    overflow: "hidden",
    background: "#000",
  },
  video: {
    width: "100%",
    height: "100%",
    objectFit: "cover",
  },
  overlay: {
    position: "absolute",
    inset: 0,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    color: "var(--color-text-dim)",
    fontSize: 12,
    background: "rgba(0,0,0,0.7)",
  },
  statusText: {
    fontSize: 12,
    fontWeight: 500,
    textAlign: "center",
  },
};
