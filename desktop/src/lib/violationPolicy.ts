export const NO_FACE_LOG_THRESHOLD_MS = 10_000;
export const LEVEL_TWO_LOG_COOLDOWN_MS = 12_000;

function toNumber(value: unknown): number {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === "string") {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) {
      return parsed;
    }
  }
  return 0;
}

export function shouldPersistViolation(
  eventType: string,
  severity: string,
  metadata: Record<string, unknown> = {}
): boolean {
  // Critical và high luôn được lưu (mức 2 trở lên)
  if (severity === "critical" || severity === "high") {
    return true;
  }

  // ai_cheating_l1 = mức 2 (medium) – luôn lưu kèm video
  if (eventType === "ai_cheating_l1") {
    return true;
  }

  if (eventType === "multiple_faces") {
    return toNumber(metadata.confirmed_face_count ?? metadata.face_count) >= 2;
  }

  if (eventType === "no_face") {
    return toNumber(metadata.no_face_duration_ms) >= NO_FACE_LOG_THRESHOLD_MS;
  }

  return false;
}
