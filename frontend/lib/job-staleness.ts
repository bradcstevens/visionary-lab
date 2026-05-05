/**
 * Issue 005 of the active-and-queued-jobs-ux-redesign PRD.
 *
 * Pure staleness detector. Compares the wall clock against
 * ``lastBackendActivityByJobId`` (a per-job baseline maintained
 * by ``jobs-context.mergeJobs``) and classifies each non-terminal
 * ``generate_project`` job into a ``StalenessKind``.
 *
 * The page header consumes ``deriveProjectWorstStaleness`` to
 * render the staleness subline + "Cancel queued jobs" button.
 */

export const STALENESS_SOFT_MS = 45_000;
export const STALENESS_HARD_MS = 120_000;

const TERMINAL_STATUSES: ReadonlySet<string> = new Set([
  "succeeded",
  "failed",
  "cancelled",
]);

export type StalenessKind =
  | "fresh"
  | "soft-pending"
  | "soft-running"
  | "hard-pending"
  | "hard-running";

export interface StalenessState {
  jobId: string;
  kind: StalenessKind;
  secondsAgo: number;
}

export interface JobLike {
  id: string;
  status: string;
  kind?: string;
  updated_at?: string;
}

const SEVERITY_RANK: Record<StalenessKind, number> = {
  "fresh": 0,
  "soft-pending": 1,
  "soft-running": 1,
  "hard-pending": 2,
  "hard-running": 2,
};

/**
 * Classify each non-terminal generate_project job by the time
 * elapsed since its last observed backend activity. Returns one
 * entry per job; jobs without a baseline are reported as "fresh"
 * (rubber-duck blocking finding: missing baseline is NOT hard).
 *
 * Pending jobs (Detector A): the wait-time-since-enqueue is the
 * baseline-relative elapsed time. A pending job that's never been
 * heard from in 45s is soft-pending; 120s is hard-pending.
 *
 * Running jobs (Detector B): a running job whose last update was
 * 45s ago is soft-running; 120s is hard-running. Baseline tracks
 * front-end wall clock at merge time, NOT job.updated_at directly,
 * so NTP drift between worker and client doesn't poison the
 * detector.
 */
export function computeStaleness(
  jobs: readonly JobLike[],
  lastBackendActivityByJobId: Readonly<Record<string, number>>,
  now: number,
): StalenessState[] {
  const out: StalenessState[] = [];
  for (const job of jobs) {
    if (job.kind && job.kind !== "generate_project") continue;
    if (TERMINAL_STATUSES.has(job.status)) continue;
    if (job.status !== "pending" && job.status !== "running") continue;

    const baseline = lastBackendActivityByJobId[job.id];
    if (baseline == null) {
      out.push({ jobId: job.id, kind: "fresh", secondsAgo: 0 });
      continue;
    }

    const elapsedMs = Math.max(0, now - baseline);
    const secondsAgo = Math.floor(elapsedMs / 1000);

    let kind: StalenessKind;
    if (elapsedMs >= STALENESS_HARD_MS) {
      kind = job.status === "running" ? "hard-running" : "hard-pending";
    } else if (elapsedMs >= STALENESS_SOFT_MS) {
      kind = job.status === "running" ? "soft-running" : "soft-pending";
    } else {
      kind = "fresh";
    }

    out.push({ jobId: job.id, kind, secondsAgo });
  }
  return out;
}

/**
 * Reduce a per-job array of staleness states to the project-level
 * worst-state, picking the highest severity tier and (within a
 * tier) the oldest job. Returns null when the input is empty.
 *
 * The page header consumes this to render exactly ONE staleness
 * subline regardless of how many jobs are stale (showing N
 * sublines for N stuck jobs would be visual noise — the user only
 * needs to know "the most-stuck job is N seconds stale").
 */
export function deriveProjectWorstStaleness(
  states: readonly StalenessState[],
): StalenessState | null {
  if (states.length === 0) return null;
  let best: StalenessState = states[0];
  for (let i = 1; i < states.length; i++) {
    const s = states[i];
    const sRank = SEVERITY_RANK[s.kind];
    const bestRank = SEVERITY_RANK[best.kind];
    if (sRank > bestRank) {
      best = s;
    } else if (sRank === bestRank && s.secondsAgo > best.secondsAgo) {
      best = s;
    }
  }
  return best;
}
