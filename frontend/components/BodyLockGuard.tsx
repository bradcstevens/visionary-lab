"use client";

import { useEffect } from "react";
import {
  computeLockReleaseActions,
  type BodyLockState,
  type LockReleaseAction,
} from "@/utils/computeLockReleaseActions";

/**
 * BodyLockGuard — defense-in-depth body-lock release for stuck Radix
 * overlay state. Issue 002 of the radix-dialog-body-lock-fix PRD
 * (`prds/2026-05-01-radix-dialog-body-lock-fix-prd.md`).
 *
 * # Bug family being defended against
 *
 * Radix UI's Dialog (and Sheet, which is built on top of Radix Dialog)
 * uses `react-remove-scroll` to lock the body's scroll and pointer
 * events while a modal overlay is open. The lock is implemented as:
 *
 *   - `body.style.pointerEvents = 'none'` (inline style)
 *   - `body.setAttribute('data-scroll-locked', '1')`
 *   - `body.style.overflow = 'hidden'` (inline style, paired with the
 *     attribute above)
 *
 * On certain unmount-while-open paths under React 19 + Radix 1.1.x +
 * react-remove-scroll, the cleanup that unwinds these mutations does
 * NOT run, leaving the page non-interactive even though every overlay
 * has closed. The user has to refresh the page to recover. This is a
 * well-known upstream issue family; rather than vendor-patch Radix or
 * upgrade aggressively, we run a single layout-level guard that
 * detects the stuck signature and clears it.
 *
 * # How it works
 *
 * 1. A single MutationObserver is attached to `<body>` watching:
 *    - `style` and `data-scroll-locked` attribute changes on body itself
 *      (the lock signature)
 *    - `data-state` attribute changes anywhere in the body subtree
 *      (catches every Radix overlay's open→closed transition,
 *      including portaled content under document.body)
 *
 * 2. Mutation bursts during a Radix close transition (typically
 *    several writes within ~16ms) are coalesced through a single
 *    `requestAnimationFrame` callback.
 *
 * 3. The rAF callback samples body state and counts open overlays
 *    via the selector contract (see below), then delegates to the
 *    pure function `computeLockReleaseActions` which decides the
 *    list of clear-actions to apply.
 *
 * 4. If the pure function returns clear-actions (only when no
 *    overlay is open AND a stuck signature is present), the wrapper
 *    applies them to the live DOM. Those writes will themselves
 *    fire mutation events, but the next rAF will see a clean body
 *    and return no actions — so reentrancy terminates after one
 *    additional cycle.
 *
 * 5. On mount (and only on mount) the guard runs an initial check
 *    via the same rAF path. This catches a stuck lock that was
 *    already present at hydration time (e.g., after HMR or stale
 *    SSR client state) — without it, `MutationObserver` would never
 *    fire because no further mutation occurs.
 *
 * # Selector contract with Radix
 *
 * The "is any overlay open?" check uses
 * `[role="dialog"][data-state="open"]`. This relies on Radix
 * consistently emitting both `role="dialog"` AND `data-state="open"`
 * for both Dialog and Sheet primitives. Sheet is built on top of
 * react-dialog and inherits the contract. This has been stable
 * across several Radix major versions.
 *
 * If a future Radix upgrade changes either the role or the
 * data-state attribute, this selector must be updated. The
 * safest place to broaden in the future is to a union:
 * `:is([role="dialog"], [role="alertdialog"])[data-state="open"]`.
 *
 * # When to retire this guard
 *
 * If the upstream Radix + react-remove-scroll cleanup race is fixed
 * — verifiable by running the project detail page through every
 * overlay close path with this guard removed and observing no stuck
 * locks — this component can be deleted. Track that with a manual
 * smoke test rather than a unit test, since the bug requires a real
 * browser to reproduce.
 *
 * # What this guard never touches
 *
 * - `body.className` (only inline style + the specific attribute)
 * - any inline body style other than `pointerEvents` and `overflow`
 *   (and `overflow` only when paired with the scroll-lock attribute)
 * - any element other than `document.body`
 *
 * Mounted exactly once inside `<body>` in `frontend/app/layout.tsx`.
 * Zero props.
 */

const OPEN_OVERLAY_SELECTOR = '[role="dialog"][data-state="open"]';

function readBodyState(body: HTMLBodyElement): BodyLockState {
  return {
    pointerEventsInline: body.style.pointerEvents,
    scrollLockedAttr: body.hasAttribute("data-scroll-locked"),
    overflowInline: body.style.overflow,
  };
}

function applyAction(body: HTMLBodyElement, action: LockReleaseAction): void {
  switch (action.kind) {
    case "clear-pointer-events":
      body.style.pointerEvents = "";
      break;
    case "remove-scroll-locked-attr":
      body.removeAttribute("data-scroll-locked");
      break;
    case "clear-overflow":
      body.style.overflow = "";
      break;
  }
}

/**
 * Returns true if the mutation records contain at least one record
 * the guard cares about. Cuts noise from app-wide DOM churn (toasts,
 * Suspense swaps, activity log updates) by ignoring pure childList
 * mutations and attribute mutations on attributes the guard does
 * not care about.
 *
 * Exported for testability.
 */
export function shouldScheduleForRecords(records: MutationRecord[]): boolean {
  for (const record of records) {
    if (record.type !== "attributes") continue;
    const name = record.attributeName;
    if (name === "style" || name === "data-scroll-locked") {
      // Only on body itself.
      if (record.target === document.body) return true;
    } else if (name === "data-state") {
      // Anywhere in the subtree — Radix overlay open/closed transitions.
      return true;
    }
  }
  return false;
}

export function BodyLockGuard(): null {
  useEffect(() => {
    if (typeof document === "undefined") return;
    const body = document.body as HTMLBodyElement;

    let rafToken: number | null = null;

    const evaluate = () => {
      rafToken = null;
      const overlayCount = document.querySelectorAll(OPEN_OVERLAY_SELECTOR).length;
      const actions = computeLockReleaseActions(readBodyState(body), overlayCount);
      for (const action of actions) {
        applyAction(body, action);
      }
    };

    const schedule = () => {
      if (rafToken !== null) return;
      rafToken = requestAnimationFrame(evaluate);
    };

    const observer = new MutationObserver((records) => {
      if (shouldScheduleForRecords(records)) {
        schedule();
      }
    });

    observer.observe(body, {
      attributes: true,
      attributeFilter: ["style", "data-scroll-locked", "data-state"],
      childList: false,
      subtree: true,
    });

    schedule();

    return () => {
      observer.disconnect();
      if (rafToken !== null) {
        cancelAnimationFrame(rafToken);
        rafToken = null;
      }
    };
  }, []);

  return null;
}
