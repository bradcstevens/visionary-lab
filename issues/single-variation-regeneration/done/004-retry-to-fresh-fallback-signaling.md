# Retry-to-fresh fallback signaling

## Parent PRD

`prds/2026-04-29-single-variation-regeneration-prd.md`

## What to build

When a user picks **Retry Same Prompt** on a variation that has no prior prompt on record (legacy variation, or a variation that errored before metadata persistence — though `001` mostly closes that gap), the backend silently falls back to a fresh prompt today. This slice makes the fallback **explicit and visible** to the user.

- **Backend:** before falling back, emit a new `variation_fallback` SSE event with `{ room_id, variation_id, reason: "no_prior_prompt" }`. Then proceed as if `strategy=fresh`.
- **Frontend SSE client:** add `variation_fallback` to the SSE event-type union in `streamVariationRegeneration`.
- **Frontend handler:** on receiving `variation_fallback`, surface a single info toast: *"No previous prompt found — generating a fresh take instead."* The regen continues to completion; this is a notification, not a cancellation.

See PRD sections **Implementation Decisions → Backend** (the regen-endpoint bullet about emitting `variation_fallback`), **Implementation Decisions → Frontend — API layer**, **Implementation Decisions → Frontend — project detail page handler** (the `variation_fallback` event row), and **Further Notes → SSE event additions** for full context.

## Acceptance criteria

- [ ] Backend: regen endpoint with `strategy=retry` and no `adapted_prompt` in metadata emits a `variation_fallback` SSE event before doing fresh-fallback work
- [ ] Backend: payload shape is `{ "type": "variation_fallback", "room_id": "...", "variation_id": "...", "reason": "no_prior_prompt" }`
- [ ] Backend: regen continues normally to terminal SSE event (no early termination)
- [ ] Frontend: SSE event-type union includes `variation_fallback`
- [ ] Frontend: the project page handler renders a single info toast on receipt
- [ ] Frontend: the toast does not block, dismiss, or otherwise interfere with the in-flight regen
- [ ] Backend test: retry against a variation with no `adapted_prompt` emits `variation_fallback` followed by the normal lifecycle events
- [ ] Backend test: retry against a variation with a valid `adapted_prompt` does NOT emit `variation_fallback`
- [ ] Playwright test: load a project with a variation lacking `adapted_prompt`, pick Retry Same Prompt, assert the fallback toast appears and regen still completes

## Blocked by

- Blocked by `001-failed-variation-retry-routes-to-single-variation-regen.md` (without prompt persistence, every retry hits fallback and the test signal is muddied)

## User stories addressed

Reference by number from the parent PRD:

- User story 16 (fallback toast when retry has no prior prompt)
