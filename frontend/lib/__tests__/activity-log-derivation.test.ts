/**
 * Issue 004 of the active-and-queued-jobs-ux-redesign PRD: deep
 * module that diffs two ``jobsById`` snapshots into a list of
 * activity-log entries.
 *
 * The function is pure — no React, no IO — so it can be hammered
 * with table-driven tests. The page calls it inside a ``useEffect``
 * watching ``projectJobs.jobsById`` and emits each derived entry
 * via ``activityLog.log()``.
 *
 * Tests pin the contract from PRD slice 5:
 *
 * - Phase changes → entries.
 * - Progress-percentage ticks → no entry (we don't pollute the log
 *   with synthetic progress noise).
 * - Terminal failure → error entry; if ``error_kind`` is present
 *   the friendly title from ``error-kind-copy`` is preferred over
 *   the raw enum string.
 * - Terminal success → success entry.
 * - Bootstrap-seed (first non-empty snapshot) → suppressed via
 *   opts.bootstrap so reloads don't backfill stale entries.
 * - Multiple jobs → multiple entries in stable order.
 */
import { describe, expect, it } from "vitest";

import { deriveLogEntries } from "../activity-log-derivation";
import type { JobLike } from "../activity-log-derivation";

type JobMap = Record<string, JobLike>;

function job(over: Partial<JobLike> & { id: string }): JobLike {
  return {
    id: over.id,
    status: over.status ?? "pending",
    phase: over.phase ?? null,
    progress: over.progress,
    error: over.error,
    error_kind: over.error_kind,
    kind: over.kind ?? "generate_project",
    room_id: over.room_id ?? "__project__",
    variation_id: over.variation_id ?? "__project__",
  };
}

describe("deriveLogEntries — no-change cases", () => {
  it("returns no entries when prev and current are identical", () => {
    const snapshot: JobMap = {
      a: job({ id: "a", status: "running", phase: "generating" }),
    };
    expect(deriveLogEntries(snapshot, snapshot)).toEqual([]);
  });

  it("returns no entries when current is empty", () => {
    expect(deriveLogEntries({}, {})).toEqual([]);
  });

  it("drops progress-only changes (no phase or status flip)", () => {
    // PRD AC: "drops progress-percentage tick events". A worker
    // emits progress=10, 20, 30 — none of those should show up in
    // the log; only phase/status transitions do.
    const prev: JobMap = {
      a: job({ id: "a", status: "running", phase: "generating", progress: 10 }),
    };
    const cur: JobMap = {
      a: job({ id: "a", status: "running", phase: "generating", progress: 50 }),
    };
    expect(deriveLogEntries(prev, cur)).toEqual([]);
  });
});

describe("deriveLogEntries — phase changes emit info entries", () => {
  it("emits an entry when phase changes from null → 'queued'", () => {
    const prev: JobMap = {};
    const cur: JobMap = { a: job({ id: "a", status: "pending", phase: "queued" }) };
    const entries = deriveLogEntries(prev, cur);
    expect(entries).toHaveLength(1);
    expect(entries[0].level).toBe("info");
    expect(entries[0].message).toMatch(/queued/i);
  });

  it("emits an entry when phase changes from 'queued' → 'generating'", () => {
    const prev: JobMap = { a: job({ id: "a", status: "running", phase: "queued" }) };
    const cur: JobMap = {
      a: job({ id: "a", status: "running", phase: "generating" }),
    };
    const entries = deriveLogEntries(prev, cur);
    expect(entries).toHaveLength(1);
    expect(entries[0].level).toBe("info");
    expect(entries[0].message.toLowerCase()).toContain("generating");
  });

  it("emits an entry when phase changes from 'generating' → 'finalizing'", () => {
    const prev: JobMap = {
      a: job({ id: "a", status: "running", phase: "generating" }),
    };
    const cur: JobMap = {
      a: job({ id: "a", status: "running", phase: "finalizing" }),
    };
    const entries = deriveLogEntries(prev, cur);
    expect(entries).toHaveLength(1);
    expect(entries[0].message.toLowerCase()).toContain("finalizing");
  });

  it("renders pipeline-emitted phase strings (room_started etc.) as friendly text", () => {
    // The pipeline emits ``room_started`` / ``room_completed`` etc.
    // The derivation must title-case / normalize them so the log
    // doesn't show raw snake_case enum tokens.
    const prev: JobMap = {
      r1: job({ id: "r1", kind: "generate_room_v1", phase: "queued" }),
    };
    const cur: JobMap = {
      r1: job({ id: "r1", kind: "generate_room_v1", phase: "room_started" }),
    };
    const entries = deriveLogEntries(prev, cur);
    expect(entries).toHaveLength(1);
    expect(entries[0].message).not.toMatch(/room_started/);
  });
});

describe("deriveLogEntries — terminal transitions", () => {
  it("emits a success entry on phase 'project_completed' or status 'succeeded'", () => {
    const prev: JobMap = {
      a: job({ id: "a", status: "running", phase: "finalizing" }),
    };
    const cur: JobMap = {
      a: job({ id: "a", status: "succeeded", phase: "project_completed" }),
    };
    const entries = deriveLogEntries(prev, cur);
    // We allow the implementation to emit a single combined entry
    // for "phase advanced AND status flipped to terminal" — the
    // contract is "at least one success entry, no error entry".
    expect(entries.length).toBeGreaterThanOrEqual(1);
    const successEntries = entries.filter((e) => e.level === "success");
    expect(successEntries.length).toBeGreaterThanOrEqual(1);
    expect(entries.every((e) => e.level !== "error")).toBe(true);
  });

  it("emits an error entry on terminal failure with friendly copy from error_kind", () => {
    const prev: JobMap = {
      a: job({ id: "a", status: "running", phase: "generating" }),
    };
    const cur: JobMap = {
      a: job({
        id: "a",
        status: "failed",
        phase: "generating",
        error: "Cosmos write timed out",
        error_kind: "STORE_FAILED",
      }),
    };
    const entries = deriveLogEntries(prev, cur);
    const errEntries = entries.filter((e) => e.level === "error");
    expect(errEntries).toHaveLength(1);
    // Friendly title (not the raw enum) drives the message:
    expect(errEntries[0].message).not.toMatch(/STORE_FAILED/);
    // Detail carries the raw error so the log row can be expanded
    // for debugging.
    expect(errEntries[0].detail).toBeTruthy();
  });

  it("falls back to a generic 'failed' message when error_kind is absent", () => {
    const prev: JobMap = {
      a: job({ id: "a", status: "running", phase: "generating" }),
    };
    const cur: JobMap = {
      a: job({ id: "a", status: "failed", error: "Something blew up" }),
    };
    const entries = deriveLogEntries(prev, cur);
    const errEntries = entries.filter((e) => e.level === "error");
    expect(errEntries).toHaveLength(1);
    expect(errEntries[0].message).toBeTruthy();
  });

  it("emits a 'cancelled' entry on status 'cancelled'", () => {
    const prev: JobMap = {
      a: job({ id: "a", status: "running", phase: "generating" }),
    };
    const cur: JobMap = {
      a: job({ id: "a", status: "cancelled", phase: "generating" }),
    };
    const entries = deriveLogEntries(prev, cur);
    expect(entries.length).toBeGreaterThanOrEqual(1);
    expect(entries.some((e) => /cancel/i.test(e.message))).toBe(true);
  });
});

describe("deriveLogEntries — multiple jobs", () => {
  it("derives entries for each job independently", () => {
    const prev: JobMap = {
      a: job({ id: "a", status: "running", phase: "queued" }),
      b: job({ id: "b", status: "running", phase: "queued" }),
    };
    const cur: JobMap = {
      a: job({ id: "a", status: "running", phase: "generating" }),
      b: job({ id: "b", status: "running", phase: "finalizing" }),
    };
    const entries = deriveLogEntries(prev, cur);
    // Two distinct phase changes → at least two entries.
    expect(entries.length).toBeGreaterThanOrEqual(2);
  });

  it("emits new-job entries for jobs that did not exist before", () => {
    const prev: JobMap = {
      a: job({ id: "a", status: "running", phase: "generating" }),
    };
    const cur: JobMap = {
      a: job({ id: "a", status: "running", phase: "generating" }),
      b: job({ id: "b", status: "pending", phase: "queued" }),
    };
    const entries = deriveLogEntries(prev, cur);
    expect(entries.length).toBeGreaterThanOrEqual(1);
    // The unchanged job ('a') must NOT generate an entry — only
    // the new one ('b').
    expect(entries.some((e) => e.detail?.includes("a"))).toBe(false);
  });
});

describe("deriveLogEntries — bootstrap suppression", () => {
  it("returns no entries when opts.bootstrap is true (initial seed)", () => {
    // PRD AC: activity log resets on reload. The first SSE seed
    // restores prior jobs but must NOT backfill log entries — that
    // would falsely show "queued / running" lines for jobs already
    // in flight when the page loaded.
    const prev: JobMap = {};
    const cur: JobMap = {
      a: job({ id: "a", status: "running", phase: "generating" }),
      b: job({ id: "b", status: "pending", phase: "queued" }),
    };
    const entries = deriveLogEntries(prev, cur, { bootstrap: true });
    expect(entries).toEqual([]);
  });

  it("emits entries normally when opts.bootstrap is false / undefined", () => {
    const prev: JobMap = {};
    const cur: JobMap = {
      a: job({ id: "a", status: "running", phase: "generating" }),
    };
    expect(deriveLogEntries(prev, cur).length).toBeGreaterThanOrEqual(1);
    expect(
      deriveLogEntries(prev, cur, { bootstrap: false }).length,
    ).toBeGreaterThanOrEqual(1);
  });
});
