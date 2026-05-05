/**
 * Issue 005 of the active-and-queued-jobs-ux-redesign PRD.
 *
 * Pure copy mapping for the staleness subline + cancel button. Drives
 * the dynamic subline that renders BENEATH the existing header counters
 * (the "{N}/{M} variations complete" line) per the rubber-duck blocking
 * finding #5 — staleness is rendered INSIDE the header area, not as a
 * separate banner.
 *
 * Copy table (PRD slice 4):
 *
 *                 | 45s soft                                       | 120s hard
 * ----------------|------------------------------------------------|----------
 * pending (A)     | "Waiting for worker to pick up your job…"      | "Worker may be unavailable. Try cancelling and starting again."
 * running (B)     | "Generation paused — last update 45s ago"      | "Worker stopped responding. Cancel to free the queue and retry."
 *
 * For soft-running (Detector B) we splice ``secondsAgo`` into the copy
 * so the user sees an honest count: "last update 47s ago" not the
 * stuck "45s ago" string. For all other states the copy is static.
 *
 * Pure design: no React, no DOM, no jobs-context. Returns a stable
 * plain object ``{ message, showCancelButton, severity }`` that the
 * page header consumes via JSX. Severity drives styling
 * (text-muted-foreground vs text-destructive). showCancelButton flips
 * true ONLY at the hard threshold.
 */

import type { StalenessKind } from "./job-staleness";

export type StalenessSeverity = "info" | "warning" | "danger";

export interface StalenessSublineCopy {
  message: string;
  showCancelButton: boolean;
  severity: StalenessSeverity;
}

/**
 * Map a staleness kind (+ secondsAgo for the running-soft variable
 * splice) to user-visible copy.
 *
 * Returns null for ``"fresh"`` so callers can naturally distinguish
 * "render the subline" from "render the existing counter". The page
 * uses ``getStalenessSublineCopy(state) ?? counterFallbackJsx``.
 */
export function getStalenessSublineCopy(
  kind: StalenessKind,
  secondsAgo: number,
): StalenessSublineCopy | null {
  switch (kind) {
    case "fresh":
      return null;
    case "soft-pending":
      return {
        message: "Waiting for worker to pick up your job\u2026",
        showCancelButton: false,
        severity: "warning",
      };
    case "soft-running":
      return {
        message: `Generation paused \u2014 last update ${secondsAgo}s ago`,
        showCancelButton: false,
        severity: "warning",
      };
    case "hard-pending":
      return {
        message:
          "Worker may be unavailable. Try cancelling and starting again.",
        showCancelButton: true,
        severity: "danger",
      };
    case "hard-running":
      return {
        message:
          "Worker stopped responding. Cancel to free the queue and retry.",
        showCancelButton: true,
        severity: "danger",
      };
    default: {
      // Forward-compat: an unrecognized kind degrades to a generic
      // warning rather than rendering raw enum tokens.
      const exhaustive: never = kind;
      void exhaustive;
      return {
        message: "Generation may be stalled.",
        showCancelButton: false,
        severity: "warning",
      };
    }
  }
}
