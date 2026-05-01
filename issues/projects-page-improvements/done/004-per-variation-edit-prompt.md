# Per-variation Edit Prompt that appends a new variation

## Parent PRD

`prds/2026-04-30-projects-page-improvements-prd.md`

## What to build

Restructure the per-variation regenerate dropdown so users can edit
the prompt that produced a generated image and have the new image
appear as a NEW variation alongside the original — preserving the
original for A/B comparison. Replaces the current "Try Something
New" with an explicit "Edit Prompt" textbox prefilled with the
variation's current `adapted_prompt`. "Retry Same Prompt" keeps its
existing destructive replace-in-place behavior.

End-to-end behavior:

- Backend: a new dedicated endpoint
  `POST /projects/{id}/rooms/{rid}/variations/{vid}/edit-prompt`
  accepts `{ adapted_prompt: string }`. Pipeline path: skip
  `BriefGeneratorService.brief_to_prompts`; pass the user's prompt
  directly through `PromptComposer` (slice 003) as the
  `variation_override`; **append** a new `Variation` to
  `room.variations` with the new image; recompute room and project
  status using `ProjectStatusCalculator` (slice 001). Streams SSE
  events using the existing event vocabulary plus a new
  `staging.variation_edit_prompt.{started, completed, failed}` log
  event family for clean operator forensics.
- Frontend: the variation regenerate dropdown in `VariationThumbnail`
  becomes `Retry Same Prompt` (unchanged) and `Edit Prompt` (new).
  Selecting Edit Prompt opens an inline modal (Dialog primitive)
  with a prefilled `<textarea>` (defaulting to the variation's
  `generation_metadata.adapted_prompt`; if missing, defaults to the
  project-level prompt and the modal shows a small notice), a
  Cancel button, and a Generate button. Generate calls the new
  endpoint. `Try Something New` is removed.
- The existing room grid layout extends naturally past
  `variations_per_room` so a 5-variation room becomes a 6-variation
  room without layout glitches.
- Tests: endpoint test asserting append (not replace) semantics and
  the new SSE log family, plus a Playwright scenario covering
  open dropdown → Edit Prompt → edit text → submit → assert
  variation count grew by one and the original is unchanged.

See PRD sections **"Solution → 4. Per-variation Edit Prompt that
preserves the original"**, **"Implementation Decisions → Backend
modules"** (edit-prompt endpoint bullet), **"Implementation
Decisions → Frontend modules"** (variation regenerate dropdown
bullet), **"Cross-cutting decisions"** (dedicated endpoint
rationale), **"Further Notes"** (textarea default behavior), and
**"Testing Decisions → Backend unit tests"**
(`tests/test_staging_endpoints_edit_prompt.py`).

## Acceptance criteria

- [ ] A new endpoint
      `POST /projects/{id}/rooms/{rid}/variations/{vid}/edit-prompt`
      accepts `{ adapted_prompt: string }`.
- [ ] The endpoint passes the user-supplied prompt through
      `PromptComposer.compose` as the `variation_override`,
      bypassing `BriefGeneratorService.brief_to_prompts`.
- [ ] On success the endpoint **appends** a new `Variation` to
      `room.variations` (does not mutate or replace any existing
      variation). The room's variation count grows past
      `variations_per_room`.
- [ ] Room and project status are recomputed via
      `ProjectStatusCalculator.compute_status` (from slice 001) at
      the end of the path.
- [ ] The endpoint emits a new
      `staging.variation_edit_prompt.{started, completed, failed}`
      structured log event family — these events do not masquerade
      as `regenerate_variation.*` events.
- [ ] The endpoint streams SSE events through the existing event
      vocabulary plus the new edit-prompt-specific log lines.
- [ ] The variation regenerate dropdown in `VariationThumbnail`
      now shows `Retry Same Prompt` (existing destructive behavior
      unchanged) and `Edit Prompt` (new). `Try Something New` is
      removed.
- [ ] Selecting Edit Prompt opens a Dialog with a prefilled
      `<textarea>` containing
      `variation.generation_metadata.adapted_prompt`. If that
      metadata is missing, the textarea defaults to the project-
      level prompt and the modal shows a small notice explaining
      the fallback.
- [ ] The Dialog has Cancel (closes without submitting) and
      Generate (calls the new endpoint, closes on success, surfaces
      errors via the existing toast pattern).
- [ ] The grid layout in `RoomGroup` extends naturally past
      `variations_per_room` — a 5-variation room becomes 6 without
      layout glitches.
- [ ] `tests/test_staging_endpoints_edit_prompt.py` asserts: a new
      variation is appended (existing variations are NOT mutated);
      room variation count increments; the new
      `staging.variation_edit_prompt.{started, completed}` log
      events are emitted; the response includes the updated
      project; status is recomputed via the calculator. The image-
      gen pipeline is mocked.
- [ ] A new Playwright scenario covers: open the variation
      dropdown; click Edit Prompt; observe the textarea prefilled
      with the current prompt; edit; submit; assert the variation
      count grew by one and the original variation is unchanged.
- [ ] Local checks pass before commit:
      `uv run pytest tests/ --ignore=tests/integration -v`,
      `cd frontend && npx playwright test`,
      `cd frontend && npm run build`,
      `cd frontend && npx next lint`.

## Blocked by

- Blocked by `001-project-status-calculator.md` (uses
  `ProjectStatusCalculator.compute_status` at the end of the
  pipeline path).
- Blocked by `003-per-room-prompt-addendum.md` (uses
  `PromptComposer.compose` with the `variation_override`
  parameter introduced in that slice).

## User stories addressed

Reference by number from the parent PRD:

- User story 14
- User story 15
- User story 16
- User story 17
- User story 18
- User story 19
