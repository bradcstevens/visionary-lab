import { describe, it, expect, afterEach, vi } from "vitest";
import { render, cleanup, screen, act } from "@testing-library/react";

import { EnqueueingBanner } from "../EnqueueingBanner";

afterEach(() => {
  cleanup();
  vi.useRealTimers();
});

// ---------------------------------------------------------------------------
// Issue 004 of the active-and-queued-jobs-ux-redesign PRD.
//
// Preflight banner that mounts synchronously when the user clicks Generate
// (`isEnqueueing && !inFlightProjectGeneration`) so the "click → silent gap
// → tile appears 30s later" behaviour is replaced with an honest, phased
// status update.
//
// Contract:
//   - 0..14 s: "Composing design brief…"
//   - 15+ s: "Submitting to queue…"
//   - No progress bar (we don't have honest percentages yet).
//   - No Cancel button (the enqueue is a single network call; cancelling
//     it is meaningless — the page-level button is the post-202 surface).
//   - role=status / aria-live=polite for screen readers.
//   - data-testid="enqueueing-banner" so e2e specs can pin the surface.
// ---------------------------------------------------------------------------

describe("EnqueueingBanner — initial render", () => {
  it("renders the brief-composition copy at t=0", () => {
    render(<EnqueueingBanner />);
    expect(screen.getByTestId("enqueueing-banner")).toBeTruthy();
    expect(screen.getByText(/composing design brief/i)).toBeTruthy();
  });

  it("does NOT render a progress bar (intentional — no honest percentage)", () => {
    render(<EnqueueingBanner />);
    expect(screen.queryByRole("progressbar")).toBeNull();
  });

  it("does NOT render a Cancel button (intentional — preflight has no cancel)", () => {
    render(<EnqueueingBanner />);
    expect(
      screen.queryByRole("button", { name: /cancel/i }),
    ).toBeNull();
  });

  it("has role=status and aria-live=polite for screen readers", () => {
    render(<EnqueueingBanner />);
    const banner = screen.getByTestId("enqueueing-banner");
    expect(banner.getAttribute("role")).toBe("status");
    expect(banner.getAttribute("aria-live")).toBe("polite");
  });
});

describe("EnqueueingBanner — phased copy", () => {
  it("switches to 'Submitting to queue…' after 15 seconds", () => {
    vi.useFakeTimers();
    render(<EnqueueingBanner />);
    expect(screen.getByText(/composing design brief/i)).toBeTruthy();
    expect(screen.queryByText(/submitting to queue/i)).toBeNull();

    act(() => {
      vi.advanceTimersByTime(15_001);
    });

    expect(screen.getByText(/submitting to queue/i)).toBeTruthy();
    expect(screen.queryByText(/composing design brief/i)).toBeNull();
  });

  it("stays on the brief-composition copy at t=14s (boundary)", () => {
    vi.useFakeTimers();
    render(<EnqueueingBanner />);

    act(() => {
      vi.advanceTimersByTime(14_000);
    });

    expect(screen.getByText(/composing design brief/i)).toBeTruthy();
    expect(screen.queryByText(/submitting to queue/i)).toBeNull();
  });
});
