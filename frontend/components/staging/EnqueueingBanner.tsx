"use client";

import * as React from "react";
import { cn } from "@/utils/cn";

/**
 * Issue 004 of the active-and-queued-jobs-ux-redesign PRD.
 *
 * Preflight banner that mounts synchronously when the user clicks
 * Generate, before the producer has confirmed (202) and before SSE
 * has delivered any job state. Replaces the current ``isEnqueueing``
 * silent state.
 *
 * Phased copy (PRD slice 5):
 *   - 0..14 s → "Composing design brief…"
 *   - 15+ s   → "Submitting to queue…"
 *
 * The 14-second cutover is timer-driven, not SSE-driven — brief
 * composition happens INSIDE the producer (server-side) BEFORE
 * 202; the worker hasn't seen the job yet, so we cannot read a
 * ``phase`` field. The banner's job is to set honest expectations
 * during the silent gap.
 *
 * Why no progress bar: we don't have an honest percentage to show
 * (the brief composition is one LLM call, not a long-running
 * pipeline). A fake progress bar would be more dishonest than no
 * bar at all.
 *
 * Why no Cancel button: cancelling a single in-flight HTTP request
 * isn't useful — the user gets an error or success in seconds.
 * Once the producer returns 202, the page mounts the post-202
 * `ProjectGenerationBanner` which DOES carry a Cancel.
 */

const PHASE_TWO_DELAY_MS = 15_000;

export function EnqueueingBanner(): React.JSX.Element {
  const [phaseTwo, setPhaseTwo] = React.useState(false);

  React.useEffect(() => {
    const timer = setTimeout(() => setPhaseTwo(true), PHASE_TWO_DELAY_MS);
    return () => clearTimeout(timer);
  }, []);

  const message = phaseTwo
    ? "Submitting to queue…"
    : "Composing design brief…";

  return (
    <div
      data-testid="enqueueing-banner"
      data-phase={phaseTwo ? "submitting" : "composing"}
      role="status"
      aria-live="polite"
      className={cn(
        "flex items-center gap-3 rounded-md border bg-card p-4 shadow-sm",
      )}
    >
      <span
        aria-hidden="true"
        className="inline-block h-3 w-3 animate-pulse rounded-full bg-primary"
      />
      <span className="text-sm font-medium">{message}</span>
    </div>
  );
}
