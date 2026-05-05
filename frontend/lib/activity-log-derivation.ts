/**
 * Issue 004 of the active-and-queued-jobs-ux-redesign PRD.
 *
 * Pure function that diffs two ``jobsById`` snapshots into a
 * stream of activity-log entries. The page wraps this in a
 * ``useEffect`` that watches ``projectJobs.jobsById`` and feeds
 * each derived entry to ``activityLog.log()``.
 *
 * Design choices:
 *
 * - **Pure**: no React, no IO, no time. Tests are table-driven.
 * - **Phase changes are the primary signal**, not status changes.
 *   The pipeline emits a stream of phase strings (``queued``,
 *   ``generating``, ``finalizing``, ``room_started``,
 *   ``room_completed``, ``project_completed``, ...) that map 1:1
 *   to "the job moved" — exactly what the activity log surfaces.
 * - **Progress-percentage ticks are dropped.** A job emitting
 *   ``progress=10, 20, 30`` stays on the same phase; we don't
 *   generate three log lines.
 * - **Terminal failures route through ``error-kind-copy``** so the
 *   log shows friendly copy instead of a raw enum token. The raw
 *   ``error`` message lands in ``detail`` for the expandable row.
 * - **``opts.bootstrap``** suppresses derivation on the very first
 *   call (the SSE seed restoring prior state). Without this, a
 *   reload of the page mid-run would falsely backfill log entries
 *   for the running job.
 * - **``opts.staleness``** is reserved for issue 005's heartbeat-
 *   stale detector. Defined but currently a no-op.
 */

import { getErrorKindCopy } from "./error-kind-copy";

export interface JobLike {
  id: string;
  status: string;
  phase?: string | null;
  progress?: number | null;
  error?: string | null;
  error_kind?: string | null;
  kind?: string;
  room_id?: string;
  variation_id?: string;
}

export interface DerivedLogEntry {
  level: "info" | "success" | "error" | "warn";
  message: string;
  /** Optional detail line — used for raw error messages or job ids. */
  detail?: string;
  /** Optional emoji-style icon hint for the log renderer. */
  icon?: string;
}

export interface DeriveLogEntriesOptions {
  /**
   * When true, suppress all derivation. Used on the initial SSE
   * seed so a page reload mid-run doesn't backfill log entries
   * for jobs that started before the page loaded.
   */
  bootstrap?: boolean;
  /**
   * Reserved for issue 005's staleness detector. The page can
   * pass a per-job staleness map and the derivation will emit
   * one-shot warning entries on the rising edge. Defined but not
   * consumed in issue 004.
   */
  staleness?: Record<string, boolean>;
}

const TERMINAL_STATUSES = new Set(["succeeded", "failed", "cancelled"]);

/**
 * Title-case + spaces-instead-of-underscores for snake_case phase
 * tokens emitted by the pipeline (``room_started`` →
 * "Room started", ``project_completed`` → "Project completed").
 *
 * Falls through to the raw token if it's already human-friendly.
 */
function humanizePhase(phase: string): string {
  if (!phase) return phase;
  // ``project_completed`` → "Project completed"
  const spaced = phase.replace(/_/g, " ");
  return spaced.charAt(0).toUpperCase() + spaced.slice(1);
}

function describePhaseTransition(
  job: JobLike,
  previousPhase: string | null | undefined,
): DerivedLogEntry | null {
  const phase = job.phase ?? null;
  if (!phase) return null;
  if (phase === previousPhase) return null;
  return {
    level: "info",
    message: humanizePhase(phase),
    detail: job.id,
  };
}

function describeTerminal(job: JobLike): DerivedLogEntry | null {
  const status = job.status;
  if (!TERMINAL_STATUSES.has(status)) return null;
  if (status === "succeeded") {
    return {
      level: "success",
      message: "Generation completed",
      detail: job.id,
    };
  }
  if (status === "cancelled") {
    return {
      level: "warn",
      message: "Generation cancelled",
      detail: job.id,
    };
  }
  // status === "failed"
  if (job.error_kind) {
    const copy = getErrorKindCopy(job.error_kind);
    return {
      level: "error",
      message: copy.friendlyTitle,
      detail: job.error || copy.userMessage,
    };
  }
  return {
    level: "error",
    message: "Generation failed",
    detail: job.error || job.id,
  };
}

/**
 * Diff two ``jobsById`` snapshots and emit derived log entries
 * for the user-visible transitions.
 */
export function deriveLogEntries(
  prev: Record<string, JobLike>,
  current: Record<string, JobLike>,
  opts: DeriveLogEntriesOptions = {},
): DerivedLogEntry[] {
  if (opts.bootstrap) return [];

  const entries: DerivedLogEntry[] = [];

  for (const id of Object.keys(current)) {
    const cur = current[id];
    const before = prev[id];

    const wasTerminal = before
      ? TERMINAL_STATUSES.has(before.status)
      : false;
    const isTerminal = TERMINAL_STATUSES.has(cur.status);

    // Phase transition entry (drops progress-only changes).
    const phaseEntry = describePhaseTransition(cur, before?.phase ?? null);
    if (phaseEntry) {
      entries.push(phaseEntry);
    }

    // Terminal transition (only on the rising edge — don't repeat
    // the same terminal event on every snapshot).
    if (isTerminal && !wasTerminal) {
      const terminalEntry = describeTerminal(cur);
      if (terminalEntry) {
        entries.push(terminalEntry);
      }
    }
  }

  return entries;
}
