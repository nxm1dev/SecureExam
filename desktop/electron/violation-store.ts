/**
 * desktop/electron/violation-store.ts
 * ────────────────────────────────────
 * Persistent local violation store.
 * Stores ALL violations per session in JSON files that are NEVER deleted.
 * Used as the authoritative source for post-exam reports.
 */

import { app } from "electron";
import * as fs from "fs";
import * as path from "path";

export interface StoredViolation {
  id: string;
  session_id: string;
  event_type: string;
  severity: string;
  message?: string;
  metadata: Record<string, unknown>;
  occurred_at: string;
}

export interface LocalSessionReport {
  session_id: string;
  violations: StoredViolation[];
  total_violations: number;
  violations_by_severity: Record<string, number>;
  violations_by_type: Array<{
    event_type: string;
    count: number;
    severity: string;
  }>;
}

let storeDir: string | null = null;

function getStoreDir(): string {
  if (!storeDir) {
    storeDir = path.join(app.getPath("userData"), "secureexam-reports");
    fs.mkdirSync(storeDir, { recursive: true });
  }
  return storeDir;
}

function getSessionFilePath(sessionId: string): string {
  // Sanitize sessionId to prevent path traversal
  const safe = sessionId.replace(/[^a-zA-Z0-9\-]/g, "_");
  return path.join(getStoreDir(), `${safe}.json`);
}

function readSessionViolations(sessionId: string): StoredViolation[] {
  const filePath = getSessionFilePath(sessionId);
  try {
    const data = fs.readFileSync(filePath, "utf-8");
    const violations = JSON.parse(data);
    return Array.isArray(violations) ? violations : [];
  } catch {
    return [];
  }
}

function writeSessionViolations(
  sessionId: string,
  violations: StoredViolation[]
): void {
  const filePath = getSessionFilePath(sessionId);
  try {
    fs.writeFileSync(filePath, JSON.stringify(violations, null, 2));
  } catch (err: any) {
    console.error("[ViolationStore] Write error:", err.message);
  }
}

/**
 * Append a violation to the persistent local store.
 * Deduplicates by ID and by event_type within a 3-second window.
 */
export function appendViolation(violation: {
  id?: string;
  session_id: string;
  event_type: string;
  severity: string;
  message?: string;
  metadata?: Record<string, unknown>;
}): void {
  const sessionId = violation.session_id;
  if (!sessionId) return;

  const violations = readSessionViolations(sessionId);
  const id = violation.id ?? crypto.randomUUID();

  // Dedup by ID
  if (violations.some((v) => v.id === id)) {
    return;
  }

  // Dedup by event_type within 3s window (prevents double-logging from
  // main process + renderer for the same real-world event)
  const now = Date.now();
  const recentDuplicate = violations.some(
    (v) =>
      v.event_type === violation.event_type &&
      Math.abs(new Date(v.occurred_at).getTime() - now) < 3000
  );
  if (recentDuplicate) {
    return;
  }

  const entry: StoredViolation = {
    id,
    session_id: sessionId,
    event_type: violation.event_type,
    severity: violation.severity,
    message: violation.message,
    metadata: violation.metadata || {},
    occurred_at: new Date().toISOString(),
  };

  violations.push(entry);
  writeSessionViolations(sessionId, violations);

  console.log(
    `[ViolationStore] Stored: ${violation.event_type} (${violation.severity}) for session ${sessionId.slice(0, 8)}… [total=${violations.length}]`
  );
}

/**
 * Build a full report from the local store for a given session.
 */
export function getSessionReport(sessionId: string): LocalSessionReport {
  const violations = readSessionViolations(sessionId);

  // Count by severity
  const severityCounts: Record<string, number> = {};
  for (const v of violations) {
    severityCounts[v.severity] = (severityCounts[v.severity] || 0) + 1;
  }

  // Count by type (group by event_type, pick the highest severity seen)
  const typeMap: Record<string, { count: number; severity: string }> = {};
  const severityRank: Record<string, number> = {
    critical: 4,
    high: 3,
    medium: 2,
    low: 1,
  };

  for (const v of violations) {
    if (!typeMap[v.event_type]) {
      typeMap[v.event_type] = { count: 0, severity: v.severity };
    }
    typeMap[v.event_type].count++;
    // Keep highest severity seen for this event type
    if (
      (severityRank[v.severity] || 0) >
      (severityRank[typeMap[v.event_type].severity] || 0)
    ) {
      typeMap[v.event_type].severity = v.severity;
    }
  }

  return {
    session_id: sessionId,
    violations,
    total_violations: violations.length,
    violations_by_severity: severityCounts,
    violations_by_type: Object.entries(typeMap)
      .map(([event_type, data]) => ({
        event_type,
        count: data.count,
        severity: data.severity,
      }))
      .sort((a, b) => b.count - a.count),
  };
}

/**
 * Delete the persistent local store for a given session.
 */
export function deleteSessionReport(sessionId: string): void {
  const filePath = getSessionFilePath(sessionId);
  try {
    if (fs.existsSync(filePath)) {
      fs.unlinkSync(filePath);
      console.log(`[ViolationStore] Deleted session report: ${sessionId.slice(0, 8)}…`);
    }
  } catch (err: any) {
    console.error("[ViolationStore] Delete error:", err.message);
  }
}
