## Parent PRD

`prds/2026-05-01-radix-dialog-body-lock-fix-prd.md`

## What to build

Build a defense-in-depth guard component that catches stuck Radix /
`react-remove-scroll` body-locks generically and clears them within
one animation frame, mounted exactly once at the root layout level.

The guard has two layers:

1. A **pure decision function** — given the current state of `<body>`
   (whether `pointer-events: none` is set inline, whether
   `data-scroll-locked` is present, whether `overflow: hidden` is set
   inline) and the count of currently-open Radix overlays
   (`[role="dialog"][data-state="open"]`), it returns a list of clear
   actions to perform. This function has no DOM access. It is the
   piece whose correctness is non-obvious and is exhaustively
   unit-tested.
2. A **thin DOM-touching client component** — wires a
   `MutationObserver` to `<body>`'s `style` and `data-scroll-locked`
   attributes, coalesces bursts of mutations through a single
   `requestAnimationFrame` callback, calls the pure decision
   function, and applies its returned actions to the live DOM. The
   component takes zero props and is mounted exactly once inside
   `<body>` in the root layout.

The guard takes no action when at least one Radix overlay is actually
open, so it never fights a live overlay. It clears only the specific
attributes Radix and `react-remove-scroll` are known to leak — never
arbitrary inline body styles. The guard's source file documents
which Radix bug family it defends against and the
selector-contract assumption (`[role="dialog"][data-state="open"]`)
so a future contributor knows when it can be retired.

See PRD sections "Defense-in-depth body-lock guard" and "Selector
contract with Radix" for the precise behavior, and "Test philosophy"
and "Modules under test" for the unit-test scope.

## Acceptance criteria

- [ ] A pure decision function (e.g. `computeLockReleaseActions`) is
      exported from a new module and takes only synthetic inputs
      (body-state flags + open-overlay count), returning a list of
      clear-action descriptors with no DOM access
- [ ] Vitest unit tests exhaustively cover the truth table of
      (pointer-events-none ✓/✗) × (scroll-locked attribute ✓/✗) ×
      (overflow-hidden ✓/✗) × (open-overlay count 0 / 1 / N), in the
      style of the existing `computeProjectSettingsDiff` pure-helper
      test
- [ ] A thin client component owns the `MutationObserver` on
      `<body>`'s `style` and `data-scroll-locked` attributes,
      coalesces firings through a single
      `requestAnimationFrame` callback per burst, and applies the
      pure function's returned actions to the live DOM
- [ ] The guard is mounted exactly once inside `<body>` in the root
      layout (`frontend/app/layout.tsx`); no per-page or per-overlay
      mount
- [ ] When the guard observes that one or more Radix overlays are
      open, it takes no action against `<body>`
- [ ] When the guard observes no open Radix overlays and a stuck
      lock signature on `<body>`, it clears only `pointer-events`
      (when set inline to `none`), the `data-scroll-locked` attribute,
      and `overflow` (only when paired with `data-scroll-locked` and
      set inline to `hidden`) — no other inline body styles are
      touched
- [ ] The guard's source file includes a doc-comment explaining
      which Radix bug family it defends against and identifying the
      selector contract (`[role="dialog"][data-state="open"]`) so a
      future contributor can decide whether to retire it
- [ ] Manual verification: on the project detail page, injecting
      `document.body.style.pointerEvents = 'none'` from the devtools
      console with no overlay open results in the style being
      cleared within one animation frame

## Blocked by

None - can start immediately.

## User stories addressed

Reference by number from the parent PRD:

- User story 16
- User story 17
- User story 18
- User story 19
- User story 20
- User story 21
- User story 22
- User story 23
