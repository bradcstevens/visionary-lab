import { describe, expect, test } from "vitest";

import { getStalenessSublineCopy } from "../staleness-subline-copy";

describe("getStalenessSublineCopy", () => {
  test("returns null for fresh — caller falls back to the existing counter line", () => {
    expect(getStalenessSublineCopy("fresh", 0)).toBeNull();
  });

  test("soft-pending: 'Waiting for worker to pick up your job…' with no cancel button", () => {
    const copy = getStalenessSublineCopy("soft-pending", 60);
    expect(copy).toEqual({
      message: "Waiting for worker to pick up your job\u2026",
      showCancelButton: false,
      severity: "warning",
    });
  });

  test("soft-running splices secondsAgo into the message", () => {
    expect(getStalenessSublineCopy("soft-running", 47)).toEqual({
      message: "Generation paused \u2014 last update 47s ago",
      showCancelButton: false,
      severity: "warning",
    });
    expect(getStalenessSublineCopy("soft-running", 119)?.message).toContain(
      "119s ago",
    );
  });

  test("hard-pending shows cancel button + danger severity + verbatim PRD copy", () => {
    expect(getStalenessSublineCopy("hard-pending", 130)).toEqual({
      message: "Worker may be unavailable. Try cancelling and starting again.",
      showCancelButton: true,
      severity: "danger",
    });
  });

  test("hard-running shows cancel button + danger severity + verbatim PRD copy", () => {
    expect(getStalenessSublineCopy("hard-running", 200)).toEqual({
      message:
        "Worker stopped responding. Cancel to free the queue and retry.",
      showCancelButton: true,
      severity: "danger",
    });
  });

  test("only the hard-* tiers expose the cancel button (rubber-duck PRD AC)", () => {
    expect(getStalenessSublineCopy("soft-pending", 50)?.showCancelButton).toBe(
      false,
    );
    expect(getStalenessSublineCopy("soft-running", 50)?.showCancelButton).toBe(
      false,
    );
    expect(getStalenessSublineCopy("hard-pending", 130)?.showCancelButton).toBe(
      true,
    );
    expect(getStalenessSublineCopy("hard-running", 130)?.showCancelButton).toBe(
      true,
    );
  });
});
