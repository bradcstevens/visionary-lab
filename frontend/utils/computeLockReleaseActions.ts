/**
 * Pure decision logic for the layout-level body-lock guard.
 *
 * Issue 002 of the radix-dialog-body-lock-fix PRD
 * (`prds/2026-05-01-radix-dialog-body-lock-fix-prd.md`).
 *
 * Given a synthetic snapshot of `<body>`'s lock-relevant state and
 * a count of currently-open Radix overlays, this function returns
 * the list of clear-actions the live-DOM wrapper should apply.
 *
 * The function has zero DOM access. The only piece of the guard whose
 * correctness is non-obvious is encoded here so it can be exhaustively
 * unit-tested against synthetic inputs instead of a real browser.
 *
 * Decision rules (per PRD § "Defense-in-depth body-lock guard"):
 *
 *   1. If `openOverlayCount > 0` — at least one Radix overlay is
 *      semantically open — the guard takes no action. Live overlays
 *      legitimately need the body lock; we never fight them.
 *
 *   2. Otherwise (no open overlays):
 *
 *      a. If `pointerEventsInline === 'none'` (set inline on the body
 *         by `react-remove-scroll`'s leftover state), emit
 *         `clear-pointer-events`.
 *
 *      b. If `scrollLockedAttr === true` (the data-scroll-locked
 *         attribute that `react-remove-scroll` failed to clean up),
 *         emit `remove-scroll-locked-attr`. ALSO emit `clear-overflow`
 *         when AND ONLY when `overflowInline === 'hidden'` (the paired
 *         signature). The paired condition matters in this codebase
 *         specifically because `<body>` already has the
 *         `overflow-hidden` Tailwind class set in
 *         `frontend/app/layout.tsx` — a bare inline `overflow: hidden`
 *         is not by itself a stuck-lock signature here, since
 *         `body.style.overflow` reads `''` when the value comes from
 *         a class. Only the paired signature is the bug.
 *
 * The action list order is stable and meaningful: the wrapper applies
 * actions in array order. Pointer-events first (cheapest unlock,
 * fixes the most common symptom), then scroll-lock attribute, then
 * overflow. Ordering is asserted by the truth-table tests.
 */

export interface BodyLockState {
  /**
   * What `body.style.pointerEvents` reads. The interesting value is
   * `'none'`. Empty string `''` means no inline override (browser
   * default).
   */
  pointerEventsInline: string | null;
  /**
   * Whether `body.hasAttribute('data-scroll-locked')` is true.
   */
  scrollLockedAttr: boolean;
  /**
   * What `body.style.overflow` reads. The interesting value is
   * `'hidden'`. Note: a Tailwind `overflow-hidden` class on `<body>`
   * does NOT set this — class-based styling never appears in
   * `style.overflow`. Only `react-remove-scroll`'s inline write does.
   */
  overflowInline: string | null;
}

export type LockReleaseAction =
  | { kind: "clear-pointer-events" }
  | { kind: "remove-scroll-locked-attr" }
  | { kind: "clear-overflow" };

export function computeLockReleaseActions(
  bodyState: BodyLockState,
  openOverlayCount: number,
): LockReleaseAction[] {
  if (openOverlayCount > 0) {
    return [];
  }

  const actions: LockReleaseAction[] = [];

  if (bodyState.pointerEventsInline === "none") {
    actions.push({ kind: "clear-pointer-events" });
  }

  if (bodyState.scrollLockedAttr) {
    actions.push({ kind: "remove-scroll-locked-attr" });
    if (bodyState.overflowInline === "hidden") {
      actions.push({ kind: "clear-overflow" });
    }
  }

  return actions;
}
