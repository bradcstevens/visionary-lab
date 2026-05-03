"use client"

import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { StagingProject } from "@/services/stagingApi";
import {
  type ProjectJob,
  TERMINAL_JOB_STATUSES,
} from "@/context/jobs-context";
import { cn } from "@/utils/cn";
import { RoomStatusPill } from "./RoomStatusPill";
import type { RecoveryState } from "@/utils/recovery-state";

// ---------------------------------------------------------------------------
// Issue 009 — image-pipeline-and-project-ux-overhaul
// ---------------------------------------------------------------------------
//
// ProgressTracker exposes three render modes, selected by ``kind``:
//
//   - undefined / "summary" (LEGACY): the original room-rollup card driven
//     by ``project`` + ``isGenerating``. Preserved verbatim so the existing
//     header card on the project page keeps working without per-call-site
//     changes.
//
//   - "per-image": a thin overlay bar for a single variation tile, driven
//     by one ProjectJob. Queued state renders as an indeterminate / striped
//     bar (the worker hasn't picked the job up yet, no determinate value
//     to show); running state renders a determinate bar fed by job.progress.
//     Returns null when the job is missing or in a terminal state, so the
//     caller can mount this unconditionally without conditionally hiding.
//
//   - "per-project": an aggregate bar for the project header, driven by the
//     full list of ProjectJob docs from useProjectJobs. Averages progress
//     across active (non-terminal) jobs only — terminal jobs are 100% by
//     definition and including them in the denominator would falsely lift
//     the bar past true completion. Returns null when there are no active
//     jobs so the header collapses cleanly when the queue drains.

function isJobActive(job: ProjectJob): boolean {
  return !TERMINAL_JOB_STATUSES.has(job.status);
}

function isJobQueued(job: ProjectJob): boolean {
  // A job is "queued" until either its status flips to running OR its
  // phase advances past "queued". The worker may set phase="generating"
  // before flipping status to "running", and we want the bar to leave
  // the indeterminate state at the earliest signal.
  if (job.status === "running") return false;
  if (job.phase && job.phase !== "queued") return false;
  return job.status === "pending" || job.phase === "queued";
}

interface PerImageProgressProps {
  kind: "per-image";
  job: ProjectJob | null | undefined;
  className?: string;
}

interface PerProjectProgressProps {
  kind: "per-project";
  jobs: readonly ProjectJob[];
  className?: string;
}

interface SummaryProgressProps {
  kind?: "summary";
  project: StagingProject;
  isGenerating?: boolean;
  /**
   * Issue 003 of projects-page-stalled-stream-error-cleanup PRD: project-
   * level recovery classification, threaded into RoomStatusPill so the
   * summary card's per-room pills render the amber stalled treatment in
   * sync with the room-list cards. Optional and defaults to
   * `{ kind: 'none' }` for callers that haven't been migrated.
   */
  projectRecoveryState?: RecoveryState;
}

export type ProgressTrackerProps =
  | PerImageProgressProps
  | PerProjectProgressProps
  | SummaryProgressProps;

export function ProgressTracker(props: ProgressTrackerProps) {
  if (props.kind === "per-image") {
    return <PerImageBar job={props.job} className={props.className} />;
  }
  if (props.kind === "per-project") {
    return <PerProjectBar jobs={props.jobs} className={props.className} />;
  }
  return (
    <SummaryTracker
      project={props.project}
      isGenerating={props.isGenerating}
      projectRecoveryState={props.projectRecoveryState}
    />
  );
}

function PerImageBar({
  job,
  className,
}: {
  job: ProjectJob | null | undefined;
  className?: string;
}) {
  if (!job || !isJobActive(job)) return null;
  const queued = isJobQueued(job);
  const value = queued
    ? null
    : Math.max(0, Math.min(100, Math.round((job.progress ?? 0) * 100)));

  return (
    <div
      data-testid="per-image-progress"
      data-phase={queued ? "queued" : "running"}
      role="progressbar"
      aria-busy="true"
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={value ?? undefined}
      aria-label={
        queued ? "Queued for generation" : `Generating: ${value ?? 0}%`
      }
      className={cn(
        "absolute bottom-0 left-0 right-0 h-1 overflow-hidden rounded-b-lg bg-black/30 backdrop-blur-sm",
        className,
      )}
    >
      {queued ? (
        <div
          data-testid="per-image-progress-indeterminate"
          className="h-full w-1/3 animate-[shimmer_1.4s_linear_infinite] bg-gradient-to-r from-transparent via-white/80 to-transparent"
          style={{
            backgroundSize: "200% 100%",
          }}
        />
      ) : (
        <div
          data-testid="per-image-progress-determinate"
          className="h-full bg-primary transition-[width] duration-300 ease-out"
          style={{ width: `${value ?? 0}%` }}
        />
      )}
    </div>
  );
}

function PerProjectBar({
  jobs,
  className,
}: {
  jobs: readonly ProjectJob[];
  className?: string;
}) {
  const activeJobs = jobs.filter(isJobActive);
  if (activeJobs.length === 0) return null;

  const queuedCount = activeJobs.filter(isJobQueued).length;
  const runningCount = activeJobs.length - queuedCount;
  // Average progress across active jobs only; queued jobs contribute 0 by
  // design so the bar reflects "of the work currently active, how done".
  const avg =
    activeJobs.reduce((s, j) => s + (j.progress ?? 0), 0) / activeJobs.length;
  const value = Math.max(0, Math.min(100, Math.round(avg * 100)));

  return (
    <div
      data-testid="per-project-progress"
      className={cn(
        "space-y-2 p-3 bg-muted/30 rounded-lg border",
        className,
      )}
    >
      <div className="flex items-center justify-between text-xs">
        <span className="font-medium">
          {activeJobs.length} active job{activeJobs.length === 1 ? "" : "s"}
        </span>
        <span className="text-muted-foreground" data-testid="per-project-progress-counts">
          {runningCount} running · {queuedCount} queued
        </span>
      </div>
      <Progress
        value={value}
        className="h-2"
        data-testid="per-project-progress-bar"
        aria-label={`Project generation progress: ${value}%`}
      />
      <div className="text-[10px] text-muted-foreground text-right">
        {value}%
      </div>
    </div>
  );
}

function SummaryTracker({
  project,
  isGenerating,
  projectRecoveryState,
}: {
  project: StagingProject;
  isGenerating?: boolean;
  projectRecoveryState?: RecoveryState;
}) {
  // Only show if project is processing
  if (project.status !== 'processing') {
    return null;
  }

  // Compute totals from rooms data (backend doesn't populate top-level fields)
  const totalVariations = project.rooms.reduce((sum, r) => sum + r.variations.length, 0);
  const completedVariations = project.rooms.reduce(
    (sum, r) => sum + r.variations.filter(v => v.status === 'completed').length, 0
  );

  const progressPercentage = totalVariations > 0 
    ? (completedVariations / totalVariations) * 100 
    : 0;

  // Stale = project is processing but no active SSE stream in this tab
  const isStale = !isGenerating;

  return (
    <div className="space-y-4 p-4 bg-muted/30 rounded-lg border">
      <div className="flex items-center justify-between">
        <h3 className="font-medium text-sm text-muted-foreground">Generation Progress</h3>
        {isStale ? (
          <Badge variant="outline" className="text-amber-600 border-amber-500/30">
            Interrupted
          </Badge>
        ) : (
          <Badge variant="secondary" className="animate-pulse">
            Processing...
          </Badge>
        )}
      </div>

      {/* Overall Progress Bar */}
      <div className="space-y-2">
        <div className="flex items-center justify-between text-sm">
          <span>Overall Progress</span>
          <span className="font-medium">
            {completedVariations}/{totalVariations} variations
          </span>
        </div>
        <Progress value={progressPercentage} className="h-2" />
        <div className="text-xs text-muted-foreground text-right">
          {Math.round(progressPercentage)}% complete
        </div>
      </div>

      {/* Per-room Status Pills */}
      <div className="space-y-2">
        <div className="text-sm font-medium text-muted-foreground">Room Status</div>
        <div className="flex flex-wrap gap-2">
          {project.rooms.map((room) => {
            const roomCompletedVariations = room.variations.filter(v => v.status === 'completed').length;
            const roomTotalVariations = room.variations.length;

            return (
              <div key={room.id} className="flex items-center gap-2">
                <RoomStatusPill
                  status={room.status}
                  label={room.label}
                  projectRecoveryState={
                    projectRecoveryState ?? { kind: "none" }
                  }
                />
                <span className="text-xs text-muted-foreground">
                  {roomCompletedVariations}/{roomTotalVariations}
                </span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}