import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { render, cleanup } from "@testing-library/react";
import {
  BodyLockGuard,
  shouldScheduleForRecords,
} from "../BodyLockGuard";

/**
 * Focused DOM-level tests for the wrapper component. The pure
 * decision logic is exhaustively unit-tested in
 * `frontend/utils/__tests__/computeLockReleaseActions.test.ts`. These
 * tests cover the wrapper's three observable behaviors that the
 * pure tests cannot:
 *
 *   1. Does NOT clear the body lock while a Radix overlay is open
 *      (the "never fight a live overlay" invariant in a real DOM).
 *   2. DOES clear the body lock once no overlay is open and a
 *      mutation triggers re-evaluation (the close-then-recover path).
 *   3. Clears a pre-existing stuck lock via the initial mount check
 *      (catches the no-MutationObserver-fires-on-mount blind spot).
 *
 * The mutation-record filter is also tested directly so a regression
 * that broadens the filter (re-introducing app-wide wakeup noise) or
 * narrows it (re-introducing the missed-overlay-close blind spot) is
 * caught explicitly.
 */

function setStuckBodyLock(): void {
  document.body.style.pointerEvents = "none";
  document.body.style.overflow = "hidden";
  document.body.setAttribute("data-scroll-locked", "1");
}

function clearBodyLock(): void {
  document.body.style.pointerEvents = "";
  document.body.style.overflow = "";
  document.body.removeAttribute("data-scroll-locked");
}

function makeOpenOverlay(id: string): HTMLDivElement {
  const el = document.createElement("div");
  el.setAttribute("role", "dialog");
  el.setAttribute("data-state", "open");
  el.id = id;
  document.body.appendChild(el);
  return el;
}

/**
 * Wait for the next two animation frames. The guard schedules its
 * evaluation via a single rAF; tests need to advance past at least
 * one frame for the callback to run. Two frames also covers the
 * reentrancy follow-up cycle (write → mutation → next rAF).
 */
function waitForRafCycles(count = 2): Promise<void> {
  return new Promise((resolve) => {
    const tick = (remaining: number) => {
      if (remaining === 0) {
        resolve();
        return;
      }
      requestAnimationFrame(() => tick(remaining - 1));
    };
    tick(count);
  });
}

describe("BodyLockGuard — DOM behavior", () => {
  beforeEach(() => {
    clearBodyLock();
    document.body
      .querySelectorAll('[role="dialog"]')
      .forEach((el) => el.remove());
  });

  afterEach(() => {
    cleanup();
    clearBodyLock();
    document.body
      .querySelectorAll('[role="dialog"]')
      .forEach((el) => el.remove());
  });

  it("does NOT clear a stuck body lock while an overlay matching the selector is open", async () => {
    // Set up: an open overlay + a stuck-looking body lock. This
    // mirrors the legitimate state during a real overlay's lifetime.
    // The guard MUST leave the body alone — otherwise it would fight
    // the live overlay and cause its own bugs (pointer-events
    // getting cleared while an overlay still expects them blocked).
    makeOpenOverlay("overlay-1");
    setStuckBodyLock();

    render(<BodyLockGuard />);
    await waitForRafCycles();

    // Body lock signature must be untouched.
    expect(document.body.style.pointerEvents).toBe("none");
    expect(document.body.hasAttribute("data-scroll-locked")).toBe(true);
    expect(document.body.style.overflow).toBe("hidden");
  });

  it("DOES clear a stuck body lock once the open overlay is removed and a mutation fires", async () => {
    // Set up: open overlay + stuck body lock. Render the guard.
    // Initial check sees overlay open → no action. Then remove the
    // overlay (mutation) → guard re-evaluates → no overlay open →
    // clears the lock. This is the regression path the PRD's "Q1
    // blocker" was about: the wrapper must wake up on overlay-side
    // mutations, not just body-side ones.
    const overlay = makeOpenOverlay("overlay-1");
    setStuckBodyLock();

    render(<BodyLockGuard />);
    await waitForRafCycles();

    // Sanity: still locked while overlay is present.
    expect(document.body.style.pointerEvents).toBe("none");

    // Simulate a Radix close: data-state flips to closed (which
    // makes the overlay no longer match the open-selector). In real
    // Radix the element later unmounts, but for the guard's purposes
    // the data-state attribute change alone is enough.
    overlay.setAttribute("data-state", "closed");
    await waitForRafCycles();

    // Body lock must now be released.
    expect(document.body.style.pointerEvents).toBe("");
    expect(document.body.hasAttribute("data-scroll-locked")).toBe(false);
    expect(document.body.style.overflow).toBe("");
  });

  it("clears a pre-existing stuck body lock on initial mount via the rAF check", async () => {
    // The MutationObserver-doesn't-fire-on-mount blind spot. If a
    // stuck lock is already present at hydration time (HMR, stale
    // SSR, prior session leftover), no future mutation may happen
    // to wake the observer. The initial-mount rAF closes this gap.
    setStuckBodyLock();
    // Note: no overlay in the document.

    render(<BodyLockGuard />);
    await waitForRafCycles();

    expect(document.body.style.pointerEvents).toBe("");
    expect(document.body.hasAttribute("data-scroll-locked")).toBe(false);
    expect(document.body.style.overflow).toBe("");
  });

  it("disconnects the observer on unmount (cleanup is wired)", async () => {
    // Behavioral proof of cleanup: render the guard, unmount it,
    // then create a new stuck lock. The guard is gone, so the lock
    // must NOT be cleared.
    const { unmount } = render(<BodyLockGuard />);
    await waitForRafCycles();

    unmount();

    // After unmount, set a new stuck lock with no overlay.
    setStuckBodyLock();
    await waitForRafCycles();

    // Lock should still be present — the now-disconnected guard
    // should not have touched it.
    expect(document.body.style.pointerEvents).toBe("none");
  });
});

describe("BodyLockGuard — shouldScheduleForRecords filter", () => {
  beforeEach(() => {
    document.body
      .querySelectorAll('[role="dialog"]')
      .forEach((el) => el.remove());
  });

  it("schedules on body style attribute change (the lock signature)", () => {
    const record = {
      type: "attributes",
      target: document.body,
      attributeName: "style",
    } as unknown as MutationRecord;
    expect(shouldScheduleForRecords([record])).toBe(true);
  });

  it("schedules on body data-scroll-locked attribute change", () => {
    const record = {
      type: "attributes",
      target: document.body,
      attributeName: "data-scroll-locked",
    } as unknown as MutationRecord;
    expect(shouldScheduleForRecords([record])).toBe(true);
  });

  it("schedules on data-state change anywhere in the subtree (overlay open/closed)", () => {
    const overlay = document.createElement("div");
    overlay.setAttribute("role", "dialog");
    overlay.setAttribute("data-state", "closed");
    document.body.appendChild(overlay);
    try {
      const record = {
        type: "attributes",
        target: overlay,
        attributeName: "data-state",
      } as unknown as MutationRecord;
      expect(shouldScheduleForRecords([record])).toBe(true);
    } finally {
      overlay.remove();
    }
  });

  it("ignores style attribute changes on elements other than body (app-wide noise)", () => {
    const noisyChild = document.createElement("div");
    document.body.appendChild(noisyChild);
    try {
      const record = {
        type: "attributes",
        target: noisyChild,
        attributeName: "style",
      } as unknown as MutationRecord;
      expect(shouldScheduleForRecords([record])).toBe(false);
    } finally {
      noisyChild.remove();
    }
  });

  it("ignores childList records (we react to data-state on the same node instead)", () => {
    // A Radix portal mount/unmount produces childList events. The
    // attendant data-state attribute change on the new/removed
    // element fires its own mutation record, so we don't need to
    // also process the childList notification — and processing it
    // would trip on every toast / activity log entry / streaming
    // chunk that hits document.body.
    const record = {
      type: "childList",
      target: document.body,
      attributeName: null,
    } as unknown as MutationRecord;
    expect(shouldScheduleForRecords([record])).toBe(false);
  });

  it("schedules when at least one record matches, even if others do not", () => {
    const noise = {
      type: "childList",
      target: document.body,
      attributeName: null,
    } as unknown as MutationRecord;
    const real = {
      type: "attributes",
      target: document.body,
      attributeName: "style",
    } as unknown as MutationRecord;
    expect(shouldScheduleForRecords([noise, real])).toBe(true);
  });

  it("returns false for an empty record list", () => {
    expect(shouldScheduleForRecords([])).toBe(false);
  });
});
