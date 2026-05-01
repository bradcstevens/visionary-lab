# Failed-variation Retry Queue

## Problem Statement

When a user generates staged variations for a project and one variation
fails mid-stream (typically because Azure OpenAI returns a 429 / 500 after
the backend exhausts its 5-attempt retry budget), the failed thumbnail
renders a Retry button. The user clicks Retry and **nothing visibly
happens**. No spinner, no toast, no error — the click just disappears.

From the user's perspective: "I see four out of five variations
generated, one failed with a clear retry button, but clicking it does
nothing. The button is broken."

The actual cause: the global generation SSE stream is still open for the
remaining variations, so the page-level `isGenerating` flag is still
`true`, and the click handler short-circuits silently. The button has no
disabled state and no feedback, so the user can't tell that the click was
ignored or why.

## Solution

Introduce a per-page in-memory **retry queue** for failed-variation
Retry clicks that arrive while generation is still in flight. The click
is registered into the queue immediately, the user sees a "Queued —
will retry when generation finishes" indicator on the thumbnail, and the
queued retries fire serially once the global stream terminates.

For the adjacent regen entry points (Regenerate dropdown on completed
thumbnails, Regenerate menu inside the lightbox), a different fix
applies: those buttons get an honest disabled state with a tooltip while
generation is in flight, instead of silently no-op'ing. They are
discretionary "try something different" actions where queueing would
surprise users.

The underlying Azure rate limit is **out of scope** — the backend
already retries 429s with full-jitter exponential backoff (5 attempts,
120s cap, honors `Retry-After`). This PRD is purely about closing the
UX feedback loop on the click side.

## User Stories

1. As a user staging a project, I want a clear visual signal that my Retry
   click was registered when generation is still running, so that I know
   the system received my intent and isn't broken.

2. As a user, I want my queued retry to fire automatically once the
   in-flight generation completes, so that I don't have to babysit the
   page and re-click after the spinner stops.

3. As a user, I want clicking Retry multiple times on the same failed
   variation to be safe (no double-firing, no toast spam), so that an
   impatient extra click doesn't waste capacity or duplicate work.

4. As a user, I want the queued state to be clearly distinguished from
   the actively-regenerating state on the thumbnail, so that I can tell
   at a glance whether my retry is waiting or running.

5. As a user, I want a toast confirming "Retry queued" the moment I
   click during in-flight generation, so that I have proof my click
   landed even if I scroll away from that thumbnail.

6. As a user reviewing the activity log, I want to see queue events
   alongside the existing lifecycle events, so that I have a complete
   timeline of what happened during the session.

7. As a user who clicks "Regenerate Room" on a room with queued failed
   variations, I want the queued individual retries to be silently
   superseded by the room-level regen, so that I don't get redundant
   per-variation regens after the bigger regen finishes.

8. As a user who clicks "Generate Remaining" on the project header
   while individual retries are queued, I want the queue to be cleared
   for the same reason — the bigger action subsumes them.

9. As a user, when the global generation stream itself fails with an
   error banner, I want the queued retries to be dropped so that the
   system doesn't immediately fire N more requests against the same
   broken upstream — and so that the Retry button on the failed
   variation is restored to clickable state for me to re-trigger
   manually after acknowledging the error.

10. As a user who navigates away from the project page or reloads it, I
    want any pending queued retries to be discarded — the queue is a
    transient "I'm watching this page" convenience, not a persistent
    work order.

11. As a user, I want the regenerate dropdown on completed-variation
    thumbnails (Retry Same Prompt / Try Something New) to be honestly
    disabled-with-tooltip while generation is in flight, instead of
    being hidden with no explanation, so that I understand the option
    exists and why it's currently unavailable. **Note**: today this
    dropdown is hidden entirely during generation; the PRD only changes
    behavior for the lightbox regen, where it currently silently
    no-ops. The thumbnail-dropdown hide is acceptable as-is.

12. As a user with the lightbox open during in-flight generation, I want
    the Regenerate action inside the lightbox to be visibly disabled
    with a tooltip explaining why, instead of silently no-op'ing on
    click, so that the lightbox UX matches the honesty of the rest of
    the page.

13. As a user whose queue is draining after generation completes, I
    want the retries to fire one at a time (not all at once), so that
    the same Azure capacity pressure that caused the original failure
    isn't immediately re-hit.

14. As a user whose failed variation was somehow already marked
    completed by the time the queue drains (race condition with a
    concurrent state change), I want the queue entry to be dropped
    silently rather than triggering an unnecessary regen, so that the
    queue respects the current truth of the project.

15. As a developer maintaining this codebase, I want the queue logic
    extracted into a `useRetryQueue` hook with a small, testable
    interface, so that I can write component-level tests for the state
    machine without spinning up Playwright.

16. As a developer, I want the heavy E2E tests to cover the four
    distinct branches (queue happy path, supersede, dedup, drop-on-
    error), so that regressions in any branch are caught before reaching
    production.

## Implementation Decisions

### Scope boundary

This PRD is **frontend-only**. The backend's typed retry behavior in
`backend/core/retry.py` (5 attempts, full-jitter exponential backoff,
`Retry-After` honoring, 120s cumulative wait cap, allowlist of
`rate_limit` / 5xx / connection categories) and the global
`IMAGE_GEN_SEMAPHORE` (default 3 concurrent image calls) are unchanged.
The 429 / 500 surfaced as `variation_failed` SSE events is the contract
this PRD consumes; tuning that contract is a separate concern.

### Deep module: `useRetryQueue` hook

Extract the queue state machine into a custom React hook in
`frontend/hooks/`. The hook encapsulates:

- An in-memory FIFO queue of variation IDs (React state).
- An `enqueue(variationId)` method that returns one of three outcomes:
  `'dispatched'` (fired immediately because the system is idle),
  `'queued'` (added to the back of the queue), or `'deduped'`
  (already in the queue, no-op).
- A `clear()` method for the supersede and drop-on-error paths.
- A read-only `queuedIds: ReadonlySet<string>` exposed for the UI.
- An internal effect that watches `regeneratingVariationId` and
  `isGenerating`, and when both go idle with a non-empty queue, pops
  the head and invokes the consumer-provided `onDispatch` callback to
  fire the next regen serially.
- A drop-rule when popping: if the variation no longer exists in the
  current `project`, or its status is no longer `'failed'`, the entry
  is dropped (with an activity-log hook called on the consumer side)
  and the next entry is considered.

The hook receives `project`, `isGenerating`, `regeneratingVariationId`,
and an `onDispatch(room, variationIndex, strategy)` callback. It does
not own the SSE stream — dispatch delegates back to the page's existing
`handleRegenerateVariation`.

This hook is the only deep, isolated module introduced. The rest is
surface-level prop wiring through existing components.

### Page integration

The project detail page uses `useRetryQueue` and:

- Routes failed-variation Retry clicks (`handleRetryVariation`)
  through the hook's `enqueue` instead of going directly to
  `handleRegenerateVariation`.
- On `'queued'` outcome, fires a `toast.info("Retry queued — will run
  when generation completes")` and an info-level activity-log entry.
- On `'deduped'` outcome, no toast, no log — silent.
- Calls `clear()` at the top of `handleRegenerateRoom`,
  `startGeneration`, and `handleRegenerateAll`, before the existing
  isGenerating guards. This is the supersede behavior.
- Calls `clear()` in the `handleStreamEvent` `'error'` case, after the
  existing error handling. This is the drop-on-error behavior.
- Computes `queuedVariationIds: Set<string>` from the hook's
  `queuedIds` and passes it into `RoomGroup`.

### Component prop flow

- `RoomGroup` accepts a new optional `queuedVariationIds?: Set<string>`
  prop and passes a derived `isQueued?: boolean` flag down to each
  `VariationThumbnail` (computed per-variation as
  `queuedVariationIds.has(variation.id)`).
- `VariationThumbnail` accepts a new optional `isQueued?: boolean`
  prop. In the failed branch, when `isQueued` is true, the Retry
  button is replaced by a Queued indicator (a Loader2 spinner +
  "Queued" badge + small descriptive caption). The visual is
  deliberately distinct from both the processing-state spinner (so
  the user knows it's waiting, not running) and the failed-state error
  tile (so the user knows their click registered).
- `ImageLightbox` accepts a new optional `isBlocked?: boolean` prop.
  When true, the Regenerate menu/button inside the lightbox is
  rendered disabled with a tooltip explaining why
  ("Generating other variations… regenerate available when complete").
  The lightbox does not need a queue itself — completed-variation
  regen is discretionary and the disabled-with-tooltip pattern is the
  right honesty fix here.

### Page → ImageLightbox wiring

The page computes `isBlocked = isGenerating || regeneratingVariationId
!== null` and passes it to `ImageLightbox`. The existing `onRegenerate`
prop continues to be passed; the lightbox simply chooses not to invoke
it (and renders the disabled UI) when `isBlocked` is true.

### Queue persistence and identity

The queue lives entirely in React state on the project detail page.
There is no `localStorage`, `sessionStorage`, or server-side
persistence. Page unmount or browser reload discards the queue.
Variation identity is by `variation.id` (UUID assigned at room upload
time and stable across reloads), so the queue is robust to project
reloads triggered by `debouncedReload` between SSE events.

### Dispatch ordering

Strictly serial. The hook fires one queued retry, waits for the
`regeneratingVariationId` to clear (which the existing
`handleRegenerateVariation` does on `variation_completed`,
`variation_failed`, `project_completed`, `stream_ended`, or `error`
events from the variation-regen SSE stream), then fires the next.
Parallel dispatch was rejected because the same Azure capacity that
caused the original failure would likely re-throttle multiple
simultaneous client requests.

### Strategy used for queued retries

Queued retries always use `strategy: 'fresh'`, matching the current
behavior of `handleRetryVariation` (the failed-variation Retry button
exists for failure remediation, not for re-running the same prompt
that just failed).

## Testing Decisions

### What makes a good test

Tests should exercise **observable behavior** — the visible thumbnail
state, the toast appearance, the activity-log entries, the network
requests fired, the timing relationships between user actions and SSE
events. Tests should NOT assert on internal hook state, internal React
re-render counts, or the specific names of helper functions.

### Modules under test

**`useRetryQueue` hook (unit-level, React Testing Library + jsdom or
similar)**:
- Enqueue when idle dispatches immediately (returns `'dispatched'`,
  invokes `onDispatch` synchronously).
- Enqueue while busy queues and returns `'queued'`; subsequent
  enqueues for the same id return `'deduped'` and do not change the
  queue.
- When `regeneratingVariationId` and `isGenerating` both go idle and
  the queue is non-empty, the next entry is dispatched.
- `clear()` empties the queue and prevents pending dispatches.
- Drop rule: if the project state at drain time has no failed variation
  matching the queued id, that entry is dropped (no `onDispatch` call)
  and the next entry is considered.

**Page + components (Playwright E2E, heavy)**: a new spec file at
`frontend/tests/e2e/retry-queue-during-generation.spec.ts` covering
four scenarios:

1. **Queue happy path**: mock the global generation SSE to emit four
   `variation_completed` and one `variation_failed` mid-stream → click
   Retry on the failed variation → assert the Queued indicator is
   visible and the Retry button is no longer there → emit
   `project_completed` → assert the variation regen POST fires once →
   assert `variation_completed` updates the thumbnail.

2. **Supersede on Regenerate Room**: queue a Retry → click Regenerate
   on the room header → assert no per-variation regen POST fires →
   assert the Queued indicator clears and the thumbnail enters the
   processing state from the room regen.

3. **Dedup on multi-click**: click Retry three times in rapid
   succession on the same failed variation while in-flight → assert
   only one `toast.info` lands and only one variation regen POST fires
   after the global stream completes.

4. **Drop on global error**: queue a Retry → emit `error` event on the
   global stream → assert the error banner appears, the Queued
   indicator clears (Retry button is restored), and no variation
   regen POST fires.

### Prior art

The Playwright spec `frontend/tests/e2e/regen-failure-preserves-prior-image.spec.ts`
demonstrates the established pattern for mocking the staging SSE
endpoints (`event:`/`data:` framing, `setupMockedRoutes` helper, fake
project fixture). The new queue spec follows the same pattern.

The `frontend/tests/e2e/retry-fallback-toast.spec.ts` and
`frontend/tests/e2e/activity-log-copy.spec.ts` files demonstrate
asserting on toast and activity-log entries, which the queue spec
will need.

For the unit-level hook test, the codebase does not currently have
a hooks/ test directory. The first hook test in this PRD establishes
the pattern: `frontend/hooks/__tests__/useRetryQueue.test.ts` (or
co-located `useRetryQueue.test.ts`) using React Testing Library's
`renderHook` against jsdom — matching the testing stack already
implied by `frontend/package.json`.

## Out of Scope

- Tuning `IMAGE_GEN_RETRY_ATTEMPTS`, `IMAGE_GEN_RETRY_BASE_DELAY`,
  `IMAGE_GEN_RETRY_MAX_TOTAL_WAIT`, or `IMAGE_GEN_MAX_CONCURRENT`.
- Changing how the backend categorizes 429 vs 5xx vs other errors,
  or how it surfaces them as `variation_failed` SSE events.
- Persisting the queue across browser reload, navigation, or session.
- A server-side queue endpoint (e.g., `/queue-retry`) or any backend
  state about pending retries.
- Bulk-retry UX ("retry all failed variations in this project" as a
  single action).
- Changes to the room-level Regenerate button or the project-level
  Generate Remaining / Regenerate All header CTA, beyond the
  supersede behavior of clearing the queue when those actions fire.
- Restructuring the existing `handleRegenerateVariation` callback or
  the `streamVariationRegeneration` API service contract.
- Per-variation cancel UI (canceling a queued retry before it fires).
  If a user wants to cancel, they can navigate away or trigger a
  superseding Regenerate Room.

## Further Notes

The fix is intentionally narrow: only the failed-variation Retry path
is queued. Completed-variation regen entry points (the dropdown on the
thumbnail, the Regenerate menu in the lightbox) get an honest
disabled-with-tooltip treatment instead, because those are
discretionary "try something different" actions where a delayed
auto-fire would surprise users.

The thumbnail dropdown for completed variations is already correctly
hidden during generation (see `RoomGroup.tsx` line 154's
`!isGenerating` guard); only the lightbox regen needs the new
`isBlocked` plumbing.

Activity-log entries follow the existing copy and icon conventions
established in `app/projects/[id]/page.tsx` (`level`, `icon`,
`message`, `detail`).

The `useRetryQueue` hook is intentionally agnostic about *what* gets
dispatched — its `onDispatch` callback signature accepts `(room,
variationIndex, strategy)`, so future use cases (e.g., queueing
"Try Something New" regens, if that policy ever changes) only require
new call sites, not changes to the hook.
