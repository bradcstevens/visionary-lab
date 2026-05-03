import { describe, it, expect, afterEach } from "vitest";
import { render, cleanup, screen } from "@testing-library/react";
import { RoomStatusPill } from "../RoomStatusPill";
import type { RecoveryState } from "@/utils/recovery-state";

afterEach(() => cleanup());

const NONE: RecoveryState = { kind: "none" };
const INTERRUPTED: RecoveryState = { kind: "interrupted" };
const STREAM_LOST: RecoveryState = {
  kind: "stream-lost",
  lostOpId: "lost-1",
};
const ERROR: RecoveryState = { kind: "error" };

describe("RoomStatusPill — per (status, projectRecoveryState) combinations", () => {
  // Pre-existing four visuals: stalled treatment must NOT fire.
  it.each([
    ["pending", NONE],
    ["processing", NONE],
    ["completed", NONE],
    ["failed", NONE],
    // Non-processing statuses never get the stalled treatment, regardless
    // of the project-level recovery state.
    ["pending", INTERRUPTED],
    ["completed", STREAM_LOST],
    ["failed", INTERRUPTED],
    // `processing` + `error` does NOT fire the stalled treatment — only
    // `interrupted` and `stream-lost` do per the PRD trigger.
    ["processing", ERROR],
  ] as const)(
    "status=%s with %o → data-stalled is absent",
    (status, recovery) => {
      render(
        <RoomStatusPill status={status} projectRecoveryState={recovery} />,
      );
      const pill = screen.getByTestId("room-status-pill");
      expect(pill.getAttribute("data-status")).toBe(status);
      expect(pill.getAttribute("data-stalled")).toBeNull();
      // The pill always shows the literal status word (the four pre-
      // existing visuals are unchanged).
      expect(pill.textContent).toBe(status);
    },
  );

  // Stalled treatment fires only on processing + (interrupted | stream-lost).
  it.each([
    ["processing", INTERRUPTED],
    ["processing", STREAM_LOST],
  ] as const)(
    "status=%s with %o → data-stalled='true' (amber)",
    (status, recovery) => {
      render(
        <RoomStatusPill status={status} projectRecoveryState={recovery} />,
      );
      const pill = screen.getByTestId("room-status-pill");
      expect(pill.getAttribute("data-status")).toBe(status);
      expect(pill.getAttribute("data-stalled")).toBe("true");
    },
  );
});
