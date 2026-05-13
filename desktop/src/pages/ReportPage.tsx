/**
 * src/pages/ReportPage.tsx
 * ─────────────────────────
 * Post-exam report page showing violation summary and timeline.
 * Reads from persistent local violation store (primary) with backend fallback.
 */

import React, { useEffect, useState } from "react";

const api = (window as any).electronAPI;

interface ViolationItem {
  id: string;
  event_type: string;
  severity: string;
  message?: string;
  metadata: Record<string, any>;
  occurred_at: string;
}

interface LocalReportData {
  session_id: string;
  violations: ViolationItem[];
  total_violations: number;
  violations_by_severity: Record<string, number>;
  violations_by_type: Array<{
    event_type: string;
    count: number;
    severity: string;
  }>;
}

interface BackendReportData {
  session_id: string;
  status: string;
  started_at: string;
  ended_at: string | null;
  total_violations: number;
  violations_by_severity: Record<string, number>;
  violations_by_type: Array<{
    event_type: string;
    count: number;
    severity: string;
  }>;
  timeline: ViolationItem[];
}

const EVENT_LABELS: Record<string, string> = {
  ai_cheating_l2: "AI: Nhắc bài / Nhìn tài liệu",
  ai_cheating_l1: "AI: Nói chuyện / Trao đổi",
  ai_cheating_mild: "AI: Đọc nhẩm",
  no_face: "Không thấy khuôn mặt",
  multiple_faces: "Nhiều khuôn mặt",
  identity_mismatch: "Không khớp danh tính",
  speech_detected: "Phát hiện giọng nói",
  voice_overlap: "Chồng giọng",
  rapid_voice_change: "Thay đổi giọng đột ngột",
  app_focus_lost: "Rời cửa sổ thi",
  blocked_shortcut: "Phím tắt bị chặn",
  blacklisted_process: "Ứng dụng bị cấm",
  fullscreen_exit: "Thoát toàn màn hình",
  app_close_attempt: "Cố đóng cửa sổ",
  multi_monitor_blocked: "Nhiều màn hình (khóa)",
  multi_monitor_connected: "Cắm thêm màn hình",
  exam_cancelled: "Hủy thi do vi phạm",
  exam_auto_submit: "Tự động nộp bài",
  tab_switch: "Chuyển ứng dụng",
  url_blocked: "Truy cập URL bị chặn",
  popup_attempt: "Cố mở tab/popup",
  vm_detected: "Phát hiện máy ảo",
  process_killed: "Ứng dụng cấm bị tắt",
};

const SEVERITY_META: Record<
  string,
  { label: string; color: string; icon: string; desc: string }
> = {
  low: {
    label: "Thấp",
    color: "var(--color-success)",
    icon: "🟢",
    desc: "Đọc nhẩm, cử chỉ nhẹ",
  },
  medium: {
    label: "Trung bình",
    color: "var(--color-warning)",
    icon: "🟡",
    desc: "Nói chuyện, trao đổi",
  },
  high: {
    label: "Cao",
    color: "var(--color-danger)",
    icon: "🟠",
    desc: "Rời màn hình, phím tắt, ứng dụng cấm",
  },
  critical: {
    label: "Nghiêm trọng",
    color: "var(--color-critical)",
    icon: "🔴",
    desc: "Nhắc bài, nhiều mặt, nhiều màn hình",
  },
};

interface Props {
  sessionId: string;
  onNewExam: () => void;
}

export default function ReportPage({ sessionId, onNewExam }: Props) {
  const [localReport, setLocalReport] = useState<LocalReportData | null>(null);
  const [backendReport, setBackendReport] = useState<BackendReportData | null>(
    null
  );
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const fetchReports = async () => {
    setLoading(true);
    setError("");

    try {
      // Primary: read from local persistent store (always available)
      const local: LocalReportData = await api.getLocalReport(sessionId);
      setLocalReport(local);
    } catch (e: any) {
      console.warn("[ReportPage] Local report failed:", e.message);
    }

    try {
      // Secondary: also fetch backend for session timing info
      const backend: BackendReportData = await api.getReport(sessionId);
      setBackendReport(backend);
    } catch (e: any) {
      console.warn("[ReportPage] Backend report failed:", e.message);
    }

    setLoading(false);

    // Schedule deletion of the local store file 10 seconds after successful load
    setTimeout(() => {
      api.deleteLocalReport(sessionId).catch((err: any) => {
        console.warn("[ReportPage] Failed to delete local report:", err);
      });
    }, 10000);
  };

  useEffect(() => {
    fetchReports();
  }, [sessionId]);

  if (loading) {
    return (
      <div style={styles.center}>
        <div style={styles.spinner} />
        <p style={{ color: "var(--color-text-muted)" }}>Đang tải báo cáo...</p>
      </div>
    );
  }

  // Use local report as primary source, backend as supplement
  const report = localReport;
  const violations = report?.violations ?? [];
  const totalViolations = report?.total_violations ?? 0;
  const severityCounts = report?.violations_by_severity ?? {};
  const typeCounts = report?.violations_by_type ?? [];

  // Session timing from backend
  const startedAt = backendReport?.started_at;
  const endedAt = backendReport?.ended_at;
  const duration =
    endedAt && startedAt
      ? Math.round(
          (new Date(endedAt).getTime() - new Date(startedAt).getTime()) / 1000
        )
      : null;

  if (!report && error) {
    return (
      <div style={styles.center}>
        <p style={{ color: "var(--color-danger)" }}>
          {error || "Lỗi không xác định"}
        </p>
        <div style={{ display: "flex", gap: 12 }}>
          <button className="btn btn-primary" onClick={() => fetchReports()}>
            🔄 Thử lại
          </button>
          <button className="btn btn-ghost" onClick={onNewExam}>
            ← Về trang chủ
          </button>
        </div>
      </div>
    );
  }

  const severityOrder = ["low", "medium", "high", "critical"] as const;

  return (
    <div style={styles.page}>
      <div style={styles.container}>
        {/* Header */}
        <div style={styles.header}>
          <div>
            <h1 style={styles.title}>📋 Báo cáo bài thi</h1>
            <p style={{ color: "var(--color-text-muted)", fontSize: 13 }}>
              Mã phiên:{" "}
              <code style={{ fontFamily: "var(--font-mono)" }}>
                {sessionId.slice(0, 8)}…
              </code>
            </p>
          </div>
          <button className="btn btn-ghost" onClick={onNewExam}>
            + Thi mới
          </button>
        </div>

        {/* Summary cards */}
        <div style={styles.statsGrid}>
          <StatCard
            label="Tổng vi phạm"
            value={totalViolations}
            color={
              totalViolations === 0
                ? "var(--color-success)"
                : "var(--color-warning)"
            }
          />
          <StatCard
            label="Nghiêm trọng"
            value={severityCounts.critical || 0}
            color="var(--color-critical)"
          />
          <StatCard
            label="Cao"
            value={severityCounts.high || 0}
            color="var(--color-danger)"
          />
          <StatCard
            label="Thời gian thi"
            value={
              duration
                ? `${Math.floor(duration / 60)}m ${duration % 60}s`
                : "—"
            }
            color="var(--color-primary)"
          />
        </div>

        {/* By type */}
        {typeCounts.length > 0 && (
          <div className="card" style={{ marginBottom: 16 }}>
            <h2 style={styles.sectionTitle}>Vi phạm theo loại</h2>
            <div
              style={{ display: "flex", flexDirection: "column", gap: 8 }}
            >
              {typeCounts.map((t) => (
                <div key={t.event_type} style={styles.typeRow}>
                  <span className={`badge badge-${t.severity}`}>
                    {t.severity}
                  </span>
                  <span style={{ flex: 1, fontSize: 13 }}>
                    {EVENT_LABELS[t.event_type] || t.event_type}
                  </span>
                  <span
                    style={{
                      fontFamily: "var(--font-mono)",
                      fontSize: 13,
                      color: "var(--color-text-muted)",
                    }}
                  >
                    ×{t.count}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Timeline */}
        <div className="card">
          <h2 style={styles.sectionTitle}>📜 Dòng thời gian vi phạm</h2>

          {/* Always show severity summary using real data */}
          <div
            className="card"
            style={{
              background: "var(--color-surface-2)",
              padding: "16px",
              borderRadius: 8,
              border: "1px solid var(--color-border)",
              marginBottom: violations.length > 0 ? 16 : 0,
            }}
          >
            {totalViolations === 0 && (
              <p
                style={{
                  color: "var(--color-success)",
                  fontSize: 14,
                  fontWeight: 600,
                  marginBottom: 16,
                  textAlign: "center",
                }}
              >
                ✅ Không có vi phạm nào được ghi nhận trong suốt bài thi
              </p>
            )}
            {totalViolations > 0 && violations.length === 0 && (
              <p
                style={{
                  color: "var(--color-warning)",
                  fontSize: 14,
                  fontWeight: 600,
                  marginBottom: 16,
                  textAlign: "center",
                }}
              >
                ⚠️ Có {totalViolations} vi phạm được ghi nhận
              </p>
            )}
            <div
              style={{ display: "flex", flexDirection: "column", gap: 8 }}
            >
              {severityOrder.map((sev) => {
                const meta = SEVERITY_META[sev];
                const count = severityCounts[sev] || 0;
                return (
                  <div key={sev} style={styles.summaryRow}>
                    <span
                      style={{
                        ...styles.summaryLabel,
                        color: count > 0 ? meta.color : undefined,
                      }}
                    >
                      {meta.icon}{" "}
                      <strong>{meta.label}</strong>
                    </span>
                    <span
                      style={{
                        ...styles.summaryCount,
                        color: count > 0 ? meta.color : "var(--color-text-dim)",
                      }}
                    >
                      {count}
                    </span>
                    <span style={styles.summaryDesc}>{meta.desc}</span>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Timeline entries */}
          {violations.length > 0 && (
            <div
              style={{
                display: "flex",
                flexDirection: "column",
                gap: 6,
                maxHeight: 400,
                overflowY: "auto",
              }}
            >
              {violations.map((v) => {
                const sevMeta = SEVERITY_META[v.severity] || {
                  label: v.severity,
                  color: "var(--color-text-muted)",
                  icon: "⚪",
                };
                return (
                  <div key={v.id} style={styles.timelineItem}>
                    <span style={styles.timelineTime}>
                      {new Date(v.occurred_at).toLocaleTimeString("vi-VN")}
                    </span>
                    <span
                      style={{
                        ...styles.severityBadge,
                        color: sevMeta.color,
                        fontWeight: 600,
                        fontSize: 12,
                      }}
                    >
                      {sevMeta.icon} {sevMeta.label}
                    </span>
                    <span
                      style={{
                        fontSize: 13,
                        color: "var(--color-text)",
                        flex: 1,
                      }}
                    >
                      {v.message ||
                        EVENT_LABELS[v.event_type] ||
                        v.event_type}
                    </span>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function StatCard({
  label,
  value,
  color,
}: {
  label: string;
  value: number | string;
  color: string;
}) {
  return (
    <div className="card" style={{ textAlign: "center" }}>
      <div style={{ fontSize: 28, fontWeight: 700, color }}>{value}</div>
      <div
        style={{ fontSize: 12, color: "var(--color-text-muted)", marginTop: 4 }}
      >
        {label}
      </div>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  page: {
    minHeight: "100vh",
    background: "var(--color-bg)",
    padding: "32px 16px",
    overflowY: "auto",
  },
  container: {
    maxWidth: 720,
    margin: "0 auto",
  },
  center: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    height: "100vh",
    gap: 16,
  },
  header: {
    display: "flex",
    alignItems: "flex-start",
    justifyContent: "space-between",
    marginBottom: 24,
  },
  title: {
    fontSize: 24,
    fontWeight: 700,
    margin: 0,
    color: "var(--color-text)",
  },
  statsGrid: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))",
    gap: 12,
    marginBottom: 16,
  },
  sectionTitle: {
    fontSize: 14,
    fontWeight: 600,
    color: "var(--color-text-muted)",
    textTransform: "uppercase" as const,
    letterSpacing: "0.5px",
    margin: "0 0 12px",
  },
  typeRow: {
    display: "flex",
    alignItems: "center",
    gap: 10,
    padding: "8px 10px",
    background: "var(--color-surface-2)",
    borderRadius: 6,
  },
  timelineItem: {
    display: "flex",
    alignItems: "center",
    gap: 10,
    padding: "6px 8px",
    background: "var(--color-surface-2)",
    borderRadius: 6,
    border: "1px solid var(--color-border)",
  },
  timelineTime: {
    fontFamily: "var(--font-mono)",
    fontSize: 11,
    color: "var(--color-text-dim)",
    minWidth: 72,
  },
  spinner: {
    width: 36,
    height: 36,
    borderRadius: "50%",
    border: "3px solid var(--color-border)",
    borderTopColor: "var(--color-primary)",
    animation: "spin 0.8s linear infinite",
  },
  summaryRow: {
    display: "flex",
    alignItems: "center",
    gap: 12,
    padding: "10px 12px",
    background: "var(--color-bg)",
    borderRadius: 6,
  },
  summaryLabel: {
    fontSize: 13,
    fontWeight: 600,
    minWidth: 120,
  },
  summaryCount: {
    fontFamily: "var(--font-mono)",
    fontSize: 18,
    fontWeight: 700,
    color: "var(--color-text)",
    minWidth: 36,
    textAlign: "center" as const,
  },
  summaryDesc: {
    fontSize: 12,
    color: "var(--color-text-muted)",
    flex: 1,
  },
  severityBadge: {
    fontFamily: "var(--font-mono)",
    padding: "2px 8px",
    borderRadius: 4,
    minWidth: 110,
  },
};
