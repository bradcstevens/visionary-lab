/**
 * Issue 005 of the active-and-queued-jobs-ux-redesign PRD.
 *
 * Pure module that classifies the freshness of in-flight jobs by
 * comparing the wall clock against ``lastBackendActivityByJobId``,
 * a baseline the jobs-context maintains. The detector emits a
 * ``StalenessKind`` per job and a project-level worst-state
 * reduction the page header consumes to render the staleness
 * subline + "Cancel queued jobs" button.
 *
 * Why pure:
 *
 * - Testable as a plain function — no React, no DOM, no clock
 *   needed beyond the ``now`` parameter (every call site that has
 *   to read wall clock injects its own ``Date.now()``).
 * - The thresholds (45_000 / 120_000) and severity ordering live
 *   in ONE file. Adding a third tier (e.g. >300_000ms => "critical")
 *   only requires changing this module.
 *
 * Why ``lastBackendActivityByJobId`` and NOT ``updated_at``
 * directly:
 *
 * - Detector A (pending jobs): we only know the job is pending
 *   from the moment WE first heard about it (REST seed, SSE seed,
 *   SSE event, or polling). The wall-clock between the queue
 *   receiving the message and the front-end seeing it is irrelevant
 *   to "is the WORKER stuck?".
 * - Detector B (running jobs): ``updated_at`` is the worker's
 *   wall clock; baseline is the front-end's wall clock at the time
 *   the latest doc was merged. Falling back to ``created_at`` for
 *   running jobs would mark a healthy long-running job hard-stale
 *   just because it's old (rubber-duck blocking finding).
 *
 * Severity ordering: ``fresh < soft-pending = soft-running <
 * hard-pending = hard-running``. The worst-state reduction picks
 * the highest tier across all non-terminal ``generate_project``
 * jobs (rubber-duck blocking finding: tier-3 multi-pending case
 * still surfaces staleness even when the canonical
 * ``inFlightProjectGeneration`` slice returns null).
 */

import { describe, expect, test } from "vitest";

import {
  computeStaleness,
  deriveProjectWorstStaleness,
  STALENESS_SOFT_MS,
  STALENESS_HARD_MS,
  type JobLike,
  type StalenessKind,
  type StalenessState,
} from "../job-staleness";

describe("computeStaleness", () => {
  const baseRunning: JobLike = {
    id: "j1",
    status: "running",
    kind: "generate_project",
    updated_at: "2026-05-04T12:00:00Z",
  };
  const basePending: JobLike = {
    id: "j2",
    status: "pending",
    kind: "generate_project",
    updated_at: "2026-05-04T12:00:00Z",
  };

  describe("empty / no-op cases", () => {
    test("returns empty array when no jobs", () => {
      expect(computeStaleness([], {}, 0)).toEqual([]);
    });

    test("treats missing baseline as fresh, NOT hard (rubber-duck #1)", () => {
      const result = computeStaleness([baseRunning], {}, 999_999_999);
      expect(result).toEqual<StalenessState[]>([
        { jobId: "j1", kind: "fresh", secondsAgo: 0 },
      ]);
    });

    test("ignores terminal jobs (succeeded / failed / cancelled)", () => {
      const jobs: JobLike[] = [
        { ...baseRunning, status: "succeeded" },
        { ...baseRunning, id: "f", status: "failed" },
        { ...baseRunning, id: "c", status: "cancelled" },
      ];
      const baseline = { j1: 0, f: 0, c: 0 };
      expect(computeStaleness(jobs, baseline, 999_999_999)).toEqual([]);
    });

    test("ignores non-generate_project kinds", () => {
      const jobs: JobLike[] = [
        { ...baseRunning, id: "x", kind: "render_variation" },
      ];
      expect(computeStaleness(jobs, { x: 0 }, 999_999_999)).toEqual([]);
    });
  });

  describe("running jobs (Detector B)", () => {
    test("running fresh below 45s soft threshold", () => {
      const baseline = { j1: 100_000 };
      const result = computeStaleness([baseRunning], baseline, 144_000);
      expect(result).toEqual<StalenessState[]>([
        { jobId: "j1", kind: "fresh", secondsAgo: 44 },
      ]);
    });

    test("running soft at 45s exactly", () => {
      const baseline = { j1: 100_000 };
      const result = computeStaleness([baseRunning], baseline, 145_000);
      expect(result).toEqual<StalenessState[]>([
        { jobId: "j1", kind: "soft-running", secondsAgo: 45 },
      ]);
    });

    test("running hard at 120s exactly", () => {
      const baseline = { j1: 100_000 };
      const result = computeStaleness([baseRunning], baseline, 220_000);
      expect(result).toEqual<StalenessState[]>([
        { jobId: "j1", kind: "hard-running", secondsAgo: 120 },
      ]);
    });

    test("running hard well past 120s", () => {
      const baseline = { j1: 100_000 };
      const result = computeStaleness([baseRunning], baseline, 350_000);
      expect(result).toEqual<StalenessState[]>([
        { jobId: "j1", kind: "hard-running", secondsAgo: 250 },
      ]);
    });
  });

  describe("pending jobs (Detector A)", () => {
    test("pending soft at 45s baseline-relative", () => {
      const baseline = { j2: 100_000 };
      const result = computeStaleness([basePending], baseline, 145_000);
      expect(result).toEqual<StalenessState[]>([
        { jobId: "j2", kind: "soft-pending", secondsAgo: 45 },
      ]);
    });

    test("pending hard at 120s", () => {
      const baseline = { j2: 100_000 };
      const result = computeStaleness([basePending], baseline, 220_000);
      expect(result).toEqual<StalenessState[]>([
        { jobId: "j2", kind: "hard-pending", secondsAgo: 120 },
      ]);
    });

    test("pending fresh under 45s", () => {
      const baseline = { j2: 100_000 };
      const result = computeStaleness([basePending], baseline, 130_000);
      expect(result).toEqual<StalenessState[]>([
        { jobId: "j2", kind: "fresh", secondsAgo: 30 },
      ]);
    });
  });

  describe("multi-job + diverse states", () => {
    test("emits one entry per non-terminal generate_project job", () => {
      const jobs: JobLike[] = [
        { ...basePending, id: "p1" },
        { ...baseRunning, id: "r1" },
        { ...baseRunning, id: "r2" },
      ];
      const baseline = { p1: 0, r1: 100_000, r2: 50_000 };
      const result = computeStaleness(jobs, baseline, 200_000);
      expect(result).toHaveLength(3);
      expect(result.map((s) => s.jobId).sort()).toEqual(["p1", "r1", "r2"]);
    });
  });
});

describe("deriveProjectWorstStaleness", () => {
  const make = (
    jobId: string,
    kind: StalenessKind,
    secondsAgo: number,
  ): StalenessState => ({ jobId, kind, secondsAgo });

  test("returns null on empty input", () => {
    expect(deriveProjectWorstStaleness([])).toBeNull();
  });

  test("picks hard-running over soft-running over fresh", () => {
    const states: StalenessState[] = [
      make("a", "fresh", 10),
      make("b", "soft-running", 50),
      make("c", "hard-running", 130),
    ];
    expect(deriveProjectWorstStaleness(states)).toEqual(states[2]);
  });

  test("picks hard-pending over soft-pending", () => {
    const states: StalenessState[] = [
      make("a", "soft-pending", 50),
      make("b", "hard-pending", 130),
    ];
    expect(deriveProjectWorstStaleness(states)).toEqual(states[1]);
  });

  test("treats hard-running and hard-pending as same severity, picks oldest", () => {
    const states: StalenessState[] = [
      make("a", "hard-running", 130),
      make("b", "hard-pending", 200),
    ];
    expect(deriveProjectWorstStaleness(states)).toEqual(states[1]);
  });

  test("treats soft-running and soft-pending as same severity, picks oldest", () => {
    const states: StalenessState[] = [
      make("a", "soft-running", 50),
      make("b", "soft-pending", 90),
    ];
    expect(deriveProjectWorstStaleness(states)).toEqual(states[1]);
  });

  test("returns fresh when only fresh entries are present", () => {
    const states: StalenessState[] = [make("a", "fresh", 5)];
    expect(deriveProjectWorstStaleness(states)).toEqual(states[0]);
  });

  test(
    "tier-3 multi-pending case: cancel-all most-needed when canonical " +
      "inFlightProjectGeneration returns null (rubber-duck #3)",
    () => {
      // Multiple pending generate_project jobs => the canonical 3-tier
      // slice in jobs-context returns null. The page header still
      // needs the worst-staleness signal so the user can hit
      // "Cancel queued jobs".
      const states: StalenessState[] = [
        make("p1", "hard-pending", 200),
        make("p2", "hard-pending", 150),
      ];
      const worst = deriveProjectWorstStaleness(states);
      expect(worst).toEqual(states[0]);
    },
  );
});

describe("threshold constants exported for jobs-context", () => {
  test("thresholds match PRD: 45s soft, 120s hard", () => {
    expect(STALENESS_SOFT_MS).toBe(45_000);
    expect(STALENESS_HARD_MS).toBe(120_000);
  });
});
