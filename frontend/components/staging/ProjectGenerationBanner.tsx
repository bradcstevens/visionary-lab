"use client";

import * as React from "react";
import { cn } from "@/utils/cn";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";

// Issue 010 of project-generation-async-queue-cutover PRD.
//
// Pure presentational banner for the in-flight project-generation job.
// The page wires the inFlightProjectGeneration slice from jobs-context
// (issue 009) to these props 1:1 — no business logic lives here, just
// rendering, label derivation, and a11y wiring.
//
// Mounting / unmounting is the page's responsibility (the slice is null
// when no job is in flight). The terminal-status null return below is
// belt-and-suspenders against a stale render between change-feed events.

export type ProjectGenerationBannerProps = {
  progress: number;
  phase: string;
  status: string;
  onCancel: () => void;
  cancelling?: boolean;
};

const TERMINAL_STATUSES = new Set(["succeeded", "failed", "cancelled"]);

// Backend pipeline emits these phase strings (see staging_pipeline.py +
// job_worker.py). Any phase NOT in this map is title-cased verbatim so
// a future backend addition (e.g. "composing_brief") shows up legibly
// without a frontend ship.
const PHASE_LABELS: Record<string, string> = {
  queued: "Queued",
  room_started: "Generating",
  room_completed: "Generating",
  room_failed: "Generating",
  variation_failed: "Generating",
  finalizing: "Finalizing",
};

function derivePhaseLabel(phase: string): string {
  if (PHASE_LABELS[phase]) {
    return PHASE_LABELS[phase];
  }
  if (!phase || phase.trim().length === 0) {
    return "Generating";
  }
  return phase
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

export function ProjectGenerationBanner({
  progress,
  phase,
  status,
  onCancel,
  cancelling = false,
}: ProjectGenerationBannerProps): React.JSX.Element | null {
  if (TERMINAL_STATUSES.has(status)) {
    return null;
  }

  const label = derivePhaseLabel(phase);

  return (
    <div
      data-testid="project-generation-banner"
      data-status={status}
      data-phase={phase}
      role="status"
      aria-live="polite"
      className={cn(
        "flex flex-col gap-2 rounded-md border bg-card p-4 shadow-sm",
      )}
    >
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-baseline gap-2">
          <span className="text-sm font-medium">{label}</span>
          <span className="text-sm text-muted-foreground">
            {Math.round(progress)}%
          </span>
        </div>
        <Button
          type="button"
          variant="outline"
          size="sm"
          onClick={onCancel}
          disabled={cancelling}
          aria-label="Cancel project generation"
        >
          {cancelling ? "Cancelling…" : "Cancel"}
        </Button>
      </div>
      <Progress value={progress} aria-valuenow={progress} />
    </div>
  );
}
