import { describe, it, expect } from "vitest";
import {
  computeLockReleaseActions,
  type BodyLockState,
  type LockReleaseAction,
} from "../computeLockReleaseActions";

/**
 * Issue 002 of the radix-dialog-body-lock-fix PRD.
 *
 * Exhaustive truth-table coverage of `computeLockReleaseActions`. The
 * decision rules form a small enumerable matrix:
 *
 *   pointerEventsInline ∈ { 'none', '' }            (2 cells)
 *   scrollLockedAttr    ∈ { true, false }           (2 cells)
 *   overflowInline      ∈ { 'hidden', '' }          (2 cells)
 *   openOverlayCount    ∈ { 0, 1, 2 }                (3 cells)
 *
 * 2³ × 3 = 24 cells. The first describe block enumerates every cell
 * explicitly so a regression on any single combination cannot slip
 * through. Subsequent describe blocks add semantic / ordering /
 * regression assertions.
 */

const FLAG_VALUES = {
  pointerEventsInline: ["none", ""] as const,
  scrollLockedAttr: [true, false] as const,
  overflowInline: ["hidden", ""] as const,
} as const;

const OVERLAY_COUNTS = [0, 1, 2] as const;

/**
 * Reference implementation of the PRD's decision rules, used to compute
 * the EXPECTED action list for each truth-table cell. The production
 * implementation MUST match this output for every cell. If the two
 * diverge, either the production code or this reference (and the PRD it
 * encodes) is wrong — the test failure pinpoints which cell.
 */
function expectedFor(
  bodyState: BodyLockState,
  openOverlayCount: number,
): LockReleaseAction[] {
  if (openOverlayCount > 0) return [];
  const out: LockReleaseAction[] = [];
  if (bodyState.pointerEventsInline === "none") {
    out.push({ kind: "clear-pointer-events" });
  }
  if (bodyState.scrollLockedAttr) {
    out.push({ kind: "remove-scroll-locked-attr" });
    if (bodyState.overflowInline === "hidden") {
      out.push({ kind: "clear-overflow" });
    }
  }
  return out;
}

describe("computeLockReleaseActions — exhaustive truth table", () => {
  for (const pe of FLAG_VALUES.pointerEventsInline) {
    for (const sl of FLAG_VALUES.scrollLockedAttr) {
      for (const ov of FLAG_VALUES.overflowInline) {
        for (const count of OVERLAY_COUNTS) {
          const bodyState: BodyLockState = {
            pointerEventsInline: pe,
            scrollLockedAttr: sl,
            overflowInline: ov,
          };
          const expected = expectedFor(bodyState, count);
          const cellLabel =
            `pe=${JSON.stringify(pe)} sl=${sl} ov=${JSON.stringify(ov)} count=${count}`;
          it(`returns ${JSON.stringify(expected.map((a) => a.kind))} for ${cellLabel}`, () => {
            expect(computeLockReleaseActions(bodyState, count)).toEqual(expected);
          });
        }
      }
    }
  }
});

describe("computeLockReleaseActions — semantic invariants", () => {
  it("returns no actions whenever any overlay is open, regardless of body state", () => {
    // The "never fight a live overlay" invariant from the PRD. If
    // a Radix overlay is semantically open, the body lock is
    // legitimate and we must not touch it.
    const fullyStuckBody: BodyLockState = {
      pointerEventsInline: "none",
      scrollLockedAttr: true,
      overflowInline: "hidden",
    };
    expect(computeLockReleaseActions(fullyStuckBody, 1)).toEqual([]);
    expect(computeLockReleaseActions(fullyStuckBody, 2)).toEqual([]);
    expect(computeLockReleaseActions(fullyStuckBody, 100)).toEqual([]);
  });

  it("returns no actions when there are no open overlays AND body is in a clean state", () => {
    expect(
      computeLockReleaseActions(
        { pointerEventsInline: "", scrollLockedAttr: false, overflowInline: "" },
        0,
      ),
    ).toEqual([]);
  });

  it("clears pointer-events independently of scroll-lock state", () => {
    expect(
      computeLockReleaseActions(
        { pointerEventsInline: "none", scrollLockedAttr: false, overflowInline: "" },
        0,
      ),
    ).toEqual([{ kind: "clear-pointer-events" }]);
  });

  it("removes scroll-lock attribute independently of pointer-events state", () => {
    expect(
      computeLockReleaseActions(
        { pointerEventsInline: "", scrollLockedAttr: true, overflowInline: "" },
        0,
      ),
    ).toEqual([{ kind: "remove-scroll-locked-attr" }]);
  });

  it("clears all three when the body has the full stuck signature and no overlay is open", () => {
    expect(
      computeLockReleaseActions(
        { pointerEventsInline: "none", scrollLockedAttr: true, overflowInline: "hidden" },
        0,
      ),
    ).toEqual([
      { kind: "clear-pointer-events" },
      { kind: "remove-scroll-locked-attr" },
      { kind: "clear-overflow" },
    ]);
  });
});

describe("computeLockReleaseActions — paired clear-overflow condition", () => {
  it("clears inline overflow ONLY when paired with the data-scroll-locked attribute", () => {
    // Per PRD § Defense-in-depth body-lock guard: clear-overflow only
    // fires alongside remove-scroll-locked-attr. A bare inline
    // overflow:hidden without the scroll-lock attribute is NOT a stuck
    // lock signature in this codebase — and would also be wrong to
    // clear because react-remove-scroll's restoration path may set
    // overflow back to its prior value via ITS prior-style snapshot.
    expect(
      computeLockReleaseActions(
        {
          pointerEventsInline: "",
          scrollLockedAttr: false,
          overflowInline: "hidden",
        },
        0,
      ),
    ).toEqual([]);
  });

  it("does NOT clear overflow when the scroll-lock attribute is present but inline overflow is empty", () => {
    // The paired condition is a strict AND: both signs must match
    // before the wrapper touches body.style.overflow. If the scroll
    // attribute is stuck without the corresponding inline overflow
    // (a partial-cleanup state), only the attribute is removed.
    expect(
      computeLockReleaseActions(
        {
          pointerEventsInline: "",
          scrollLockedAttr: true,
          overflowInline: "",
        },
        0,
      ),
    ).toEqual([{ kind: "remove-scroll-locked-attr" }]);
  });

  it("does NOT clear inline overflow when an overlay is open, even if both paired signs match", () => {
    // Reinforces the live-overlay invariant: paired condition is
    // gated on no overlays open.
    expect(
      computeLockReleaseActions(
        {
          pointerEventsInline: "",
          scrollLockedAttr: true,
          overflowInline: "hidden",
        },
        1,
      ),
    ).toEqual([]);
  });
});

describe("computeLockReleaseActions — action ordering", () => {
  it("orders actions as [clear-pointer-events, remove-scroll-locked-attr, clear-overflow]", () => {
    // The wrapper applies actions in array order. Pointer-events
    // unlocks the page first (the most user-visible symptom), then
    // the scroll-lock attribute, then overflow. Asserting the order
    // pins this contract so a refactor of the pure function can't
    // silently re-order the writes.
    const result = computeLockReleaseActions(
      { pointerEventsInline: "none", scrollLockedAttr: true, overflowInline: "hidden" },
      0,
    );
    expect(result.map((a) => a.kind)).toEqual([
      "clear-pointer-events",
      "remove-scroll-locked-attr",
      "clear-overflow",
    ]);
  });
});

describe("computeLockReleaseActions — null/undefined safety", () => {
  it("treats null pointerEventsInline as not-stuck", () => {
    // Defensive case: element style reads return '' for unset inline
    // styles, but a future caller might pass null (e.g., from a
    // typed snapshot helper that uses `getAttribute('style')` parsing).
    expect(
      computeLockReleaseActions(
        { pointerEventsInline: null, scrollLockedAttr: false, overflowInline: null },
        0,
      ),
    ).toEqual([]);
  });
});
