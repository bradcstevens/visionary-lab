"use client";

import * as React from "react";
import { Button } from "@/components/ui/button";
import { cn } from "@/utils/cn";
import {
  getStalenessSublineCopy,
  type StalenessSeverity,
} from "@/lib/staleness-subline-copy";
import type { StalenessState } from "@/lib/job-staleness";

/**
 * Issue 005 of the active-and-queued-jobs-ux-redesign PRD.
 *
 * Renders the staleness subline + (at the 120s hard threshold) the
 * one-tap "Cancel queued jobs" button. Lives INSIDE the page header
 * area — REPLACING the ``{N}/{M} variations complete`` counter line
 * when staleness is non-fresh, FALLING BACK to the counter line when
 * fresh (caller-supplied via ``fallback`` prop).
 *
 * Per rubber-duck blocking finding #5, this is NOT a separate banner
 * component mounted under ``ProjectGenerationBanner``; it lives
 * inside the header's flex stack so the visual hierarchy is:
 *   <h1>Project name</h1>
 *   <CollapsiblePrompt />
 *   <StalenessSubline /> ← here
 *
 * Three local states (driven by props from the page wiring effect):
 * - ``staleness``: pure staleness state from jobs-context. Drives copy.
 * - ``cancelling``: page sets true synchronously on click; sustained
 *    until the server-side cancel confirms (job flips terminal) OR
 *    the 10s timeout fallback fires. While true, the button shows
 *    a spinner + "Cancelling…" copy and is disabled.
 * - ``dismissedAfterTimeout``: page sets true after the 10s timeout
 *    so we don't re-mount the subline immediately; cleared when the
 *    live set of non-terminal jobs no longer contains the cancelled
 *    job (terminal arrival OR new job starts). Owned by the page
 *    because the suppression keying is per-jobId.
 *
 * Component is presentation-only: it does NOT call ``cancelAllProjectJobs``
 * itself (the page wiring owns the cancel + 10s timer + suppression).
 */

const SEVERITY_CLASS: Record<StalenessSeverity, string> = {
  info: "text-muted-foreground",
  warning: "text-amber-600 dark:text-amber-400",
  danger: "text-destructive",
};

export interface StalenessSublineProps {
  staleness: StalenessState | null;
  cancelling: boolean;
  /** Page wiring-supplied click handler. Component does NOT call cancelAllProjectJobs directly. */
  onCancelAllClick: () => void;
  /** Rendered when staleness is null/fresh (caller passes the existing counter line). */
  fallback?: React.ReactNode;
}

export function StalenessSubline({
  staleness,
  cancelling,
  onCancelAllClick,
  fallback,
}: StalenessSublineProps): React.JSX.Element | null {
  // While cancelling is true we render a fixed "Cancelling…" subline
  // regardless of staleness state, so a stale-running flip during the
  // cancel doesn't flash conflicting copy.
  if (cancelling) {
    return (
      <p
        data-testid="staleness-subline"
        data-state="cancelling"
        role="status"
        aria-live="polite"
        className={cn("text-xs text-muted-foreground")}
      >
        Cancelling…
      </p>
    );
  }

  const copy = staleness ? getStalenessSublineCopy(staleness.kind, staleness.secondsAgo) : null;

  if (!copy) {
    // Fresh / null — render the existing counter line (or nothing).
    return <>{fallback ?? null}</>;
  }

  return (
    <div
      data-testid="staleness-subline"
      data-state={staleness?.kind}
      role="status"
      aria-live="polite"
      className="flex items-center gap-2"
    >
      <p className={cn("text-xs", SEVERITY_CLASS[copy.severity])}>
        {copy.message}
      </p>
      {copy.showCancelButton && (
        <Button
          data-testid="cancel-queued-jobs-button"
          size="sm"
          variant="outline"
          onClick={onCancelAllClick}
          className="text-xs h-7"
        >
          Cancel queued jobs
        </Button>
      )}
    </div>
  );
}
