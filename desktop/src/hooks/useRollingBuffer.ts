/**
 * src/hooks/useRollingBuffer.ts
 * ──────────────────────────────
 * Ring buffer for camera + microphone recording.
 * Continuously records at 480p 15FPS + full audio.
 * When a violation is triggered, captures a ~10s clip (5s before + 5s after).
 * 
 * FIX: Uses Overlapping Recorders instead of Chunk Stitching to ensure 100% valid WebM files with correct durations.
 */

import { useRef, useCallback, useEffect, useState } from "react";

const api = (window as any).electronAPI;

interface UseRollingBufferOptions {
  sessionId: string;
  /** Seconds of pre-violation footage to keep in memory */
  bufferDuration?: number;
  /** Chunk interval in ms */
  chunkIntervalMs?: number;
}

interface RollingBufferAPI {
  /** Whether the buffer is actively recording */
  isRecording: boolean;
  /** Start continuous background recording */
  startRecording: (stream: MediaStream) => void;
  /** Stop recording and cleanup */
  stopRecording: () => void;
  /** Capture a violation clip (~10s total) and upload */
  captureViolationClip: (violationId: string) => void;
}

type RecorderSession = {
  id: string;
  recorder: MediaRecorder;
  chunks: Blob[];
  startTime: number;
  isCapturing: boolean;
  violationId?: string;
};

export function useRollingBuffer({
  sessionId,
  bufferDuration = 5,
  chunkIntervalMs = 1000,
}: UseRollingBufferOptions): RollingBufferAPI {
  const [isRecording, setIsRecording] = useState(false);

  const streamRef = useRef<MediaStream | null>(null);
  const sessionsRef = useRef<RecorderSession[]>([]);
  const spawnIntervalRef = useRef<NodeJS.Timeout | null>(null);

  const uploadBlob = useCallback(async (blob: Blob, violationId: string) => {
    try {
      console.log(`[RollingBuffer] Clip size: ${(blob.size / 1024).toFixed(1)}KB`);
      const arrayBuffer = await blob.arrayBuffer();
      const uint8 = new Uint8Array(arrayBuffer);
      
      // Efficiently convert to base64
      let binary = "";
      // Use chunking to prevent stack overflow on huge arrays
      const chunkSize = 8192;
      for (let i = 0; i < uint8.length; i += chunkSize) {
        binary += String.fromCharCode.apply(null, Array.from(uint8.subarray(i, i + chunkSize)));
      }
      
      const base64 = btoa(binary);

      await api.uploadViolationVideo(sessionId, violationId, base64);
      console.log(`[RollingBuffer] Clip for violation ${violationId} uploaded successfully`);
    } catch (err: any) {
      console.error("[RollingBuffer] Upload failed:", err.message);
    }
  }, [sessionId]);

  const spawnRecorder = useCallback(() => {
    if (!streamRef.current) return;

    const mimeType = MediaRecorder.isTypeSupported("video/webm;codecs=vp8,opus")
      ? "video/webm;codecs=vp8,opus"
      : "video/webm";

    const recorder = new MediaRecorder(streamRef.current, {
      mimeType,
      videoBitsPerSecond: 500_000, // 500kbps – low quality but visible
    });

    const session: RecorderSession = {
      id: Math.random().toString(36).slice(2),
      recorder,
      chunks: [],
      startTime: Date.now(),
      isCapturing: false,
    };

    recorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        session.chunks.push(event.data);
      }
    };

    recorder.onstop = () => {
      if (session.isCapturing && session.violationId) {
        const blob = new Blob(session.chunks, { type: "video/webm" });
        void uploadBlob(blob, session.violationId);
      }
    };

    recorder.start(chunkIntervalMs);
    sessionsRef.current.push(session);
  }, [chunkIntervalMs, uploadBlob]);

  const cleanupOldRecorders = useCallback(() => {
    const now = Date.now();
    sessionsRef.current = sessionsRef.current.filter((session) => {
      // Keep capturing sessions alive until they finish
      if (session.isCapturing) return true;

      // If a session is older than (bufferDuration * 2 + 2) seconds, it's too old
      // (e.g., 12 seconds). We stop it and remove it.
      if (now - session.startTime > (bufferDuration * 2 + 2) * 1000) {
        if (session.recorder.state !== "inactive") {
          session.recorder.stop();
        }
        return false;
      }
      return true;
    });
  }, [bufferDuration]);

  const startRecording = useCallback(
    (stream: MediaStream) => {
      if (isRecording) return;
      
      streamRef.current = stream;
      setIsRecording(true);
      console.log("[RollingBuffer] Started recording (Overlapping Mode)");

      // Spawn the first recorder immediately
      spawnRecorder();

      // Spawn a new recorder every `bufferDuration` seconds
      spawnIntervalRef.current = setInterval(() => {
        spawnRecorder();
        cleanupOldRecorders();
      }, bufferDuration * 1000);
    },
    [isRecording, bufferDuration, spawnRecorder, cleanupOldRecorders]
  );

  const captureViolationClip = useCallback(
    (violationId: string) => {
      const now = Date.now();

      // Find sessions that are NOT already capturing
      const available = sessionsRef.current.filter((s) => !s.isCapturing);
      if (available.length === 0) {
        console.warn("[RollingBuffer] No available recorders for clip capture.");
        return;
      }

      // Sort by how close their history is to the desired pre-violation `bufferDuration`
      available.sort((a, b) => {
        const historyA = now - a.startTime;
        const historyB = now - b.startTime;
        return Math.abs(historyA - bufferDuration * 1000) - Math.abs(historyB - bufferDuration * 1000);
      });

      const bestSession = available[0];
      bestSession.isCapturing = true;
      bestSession.violationId = violationId;

      console.log(
        `[RollingBuffer] Capturing violation ${violationId}. Pre-history: ${
          ((now - bestSession.startTime) / 1000).toFixed(1)
        }s`
      );

      // Let it run for the post-violation duration, then stop it to trigger upload
      setTimeout(() => {
        if (bestSession.recorder.state !== "inactive") {
          bestSession.recorder.stop();
        }
        // Remove it from the active tracking array
        sessionsRef.current = sessionsRef.current.filter((s) => s.id !== bestSession.id);
      }, bufferDuration * 1000);
    },
    [bufferDuration]
  );

  const stopRecording = useCallback(() => {
    if (spawnIntervalRef.current) {
      clearInterval(spawnIntervalRef.current);
      spawnIntervalRef.current = null;
    }

    sessionsRef.current.forEach((session) => {
      if (session.recorder.state !== "inactive") {
        session.recorder.stop();
      }
    });

    sessionsRef.current = [];
    streamRef.current = null;
    setIsRecording(false);
    console.log("[RollingBuffer] Stopped recording");
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopRecording();
    };
  }, [stopRecording]);

  return {
    isRecording,
    startRecording,
    stopRecording,
    captureViolationClip,
  };
}
