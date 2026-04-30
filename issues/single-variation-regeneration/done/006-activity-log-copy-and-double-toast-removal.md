# Activity log copy and double-toast removal

## Parent PRD

`prds/2026-04-29-single-variation-regeneration-prd.md`

## What to build

Tighten the user-facing feedback for regen events so the activity log is the source of truth and ephemeral toasts don't fire spurious or duplicated messages.

- **Activity log message** distinguishes the strategy used:
  - `Variation N regenerated (retry)` for a successful Retry Same Prompt
  - `Variation N regenerated (fresh)` for a successful Try Something New
  - `Variation N regenerated (fresh — no prior prompt)` for a Retry that fell back to fresh
- **Activity log detail** appends the first ~60 characters of the adapted prompt (in addition to model / tokens / elapsed) so the user can recall what aesthetic each regen attempted.
- **Drop the success toast on `project_completed`.** Visual feedback (new image appearing, lightbox auto-updating, activity log entry) is sufficient. This also resolves the double-toast bug where a failed regen flashed an error toast immediately followed by a "Variation regenerated!" success toast.
- **Drop the toast on `stream_ended`.** The terminal `stream_ended` event fires when the SSE closes without a normal terminal event (e.g., network blip); it's used internally to clear the regen state and reload the project, but should not flash spurious feedback.
- **Keep the error toast on `variation_failed`.** Failures still need an attention-grabbing toast.

See PRD sections **Implementation Decisions → Frontend — project detail page handler** and **Further Notes → Activity log copy reference** for full context.

## Acceptance criteria

- [ ] Frontend: success activity log entry includes the strategy label (`(retry)` / `(fresh)` / `(fresh — no prior prompt)`)
- [ ] Frontend: success activity log detail includes a 60-char prompt snippet alongside model / tokens / elapsed
- [ ] Frontend: no success toast fires on `project_completed` for a single-variation regen flow
- [ ] Frontend: no toast fires on `stream_ended`
- [ ] Frontend: error toast still fires on `variation_failed`
- [ ] Frontend: a regen that fails followed by `stream_ended` results in exactly one toast (the error toast) — no trailing success toast
- [ ] Playwright test: successful retry produces the right activity log entry with `(retry)` label and prompt snippet
- [ ] Playwright test: successful fresh produces the right activity log entry with `(fresh)` label
- [ ] Playwright test: successful fresh-fallback produces the right activity log entry with `(fresh — no prior prompt)` label
- [ ] Playwright test: a regen failure surfaces exactly one error toast and no follow-up success toast

## Blocked by

- Blocked by `003-prompt-diversity-module-for-try-something-new.md` (need strategy distinction to differentiate `(retry)` vs `(fresh)` in copy)
- Blocked by `004-retry-to-fresh-fallback-signaling.md` (need the fallback event to render the `(fresh — no prior prompt)` variant correctly)

## User stories addressed

Reference by number from the parent PRD:

- User story 17 (activity log distinguishes Retry/Fresh/Fresh-fallback)
- User story 18 (activity log includes prompt snippet)
