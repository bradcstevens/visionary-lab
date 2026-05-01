# Per-room prompt addendum (Room field + PromptComposer + pencil-icon UI)

## Parent PRD

`prds/2026-04-30-projects-page-improvements-prd.md`

## What to build

Let users attach a per-image clarification to the project prompt
without polluting the prompts for other rooms. Adds a new
`prompt_addendum` field on each room and a small pencil-icon UI to
edit it. Future generations of that room (full-room generation and
the new edit-prompt path) append the addendum; existing variations
are not regenerated.

End-to-end behavior:

- Backend:
  - `Room.prompt_addendum: Optional[str] = None` is added to
    `backend/models/staging.py`. Persisted alongside other Room
    fields. Existing rooms without the field default to `None`.
  - A new pure helper
    `PromptComposer.compose(project_prompt, design_brief,
    room_addendum, variation_override?) -> adapted_prompt`
    encapsulates the precedence rules: `variation_override` (when
    present) wins outright; otherwise the room addendum is
    appended to the brief-adapted prompt. (The `variation_override`
    parameter is implemented in this slice but only exercised in
    slice 004.)
  - `BriefGeneratorService.brief_to_prompts()` consumes the
    composer when generating per-room prompts so future
    regenerations of an addendum-bearing room respect the addendum.
  - The room addendum is editable through the existing project
    update surface — either by extending `PATCH /projects/{id}` to
    accept nested room updates or by adding a dedicated
    `PATCH /projects/{id}/rooms/{rid}` endpoint. Either path is
    acceptable per the PRD's "Further Notes" — the implementer
    chooses based on whether other room-level fields are expected
    to become editable soon (in which case the dedicated endpoint
    is cleaner).
- Frontend: a small pencil icon is placed next to each room title.
  Clicking it opens a popover with a textarea bound to
  `room.prompt_addendum`. Saving calls the room/project update
  endpoint. The textarea is prefilled with the current value when
  one exists.
- Tests: table-driven `PromptComposer` unit tests covering all
  precedence layers, plus a Playwright scenario adding an addendum
  and asserting it appears in the next regeneration's prompt.

See PRD sections **"Solution → 3. Per-room prompt addendum"**,
**"Implementation Decisions → Backend modules"** (`PromptComposer`
and `Room.prompt_addendum` bullets), **"Implementation Decisions →
Frontend modules"** (pencil-icon bullet), **"Further Notes"**
(addendum precedence and Retry semantics), and **"Testing
Decisions → Backend unit tests"** (`tests/test_prompt_composer.py`).

## Acceptance criteria

- [ ] `Room.prompt_addendum: Optional[str] = None` is added to
      `backend/models/staging.py` and persists alongside other room
      fields. Existing rooms without the field load with the field
      defaulting to `None`.
- [ ] A new pure helper
      `PromptComposer.compose(project_prompt, design_brief,
      room_addendum, variation_override?) -> adapted_prompt` is
      added. Pure (no I/O, no mutation). The `variation_override`
      parameter is wired through and tested but is only exercised
      from a real call site in slice 004.
- [ ] Composer precedence: `variation_override` wins outright;
      otherwise the room addendum is appended to the brief-adapted
      prompt; whitespace and `None`/empty inputs are handled
      cleanly at every layer.
- [ ] `BriefGeneratorService.brief_to_prompts()` consumes the
      composer for every per-room prompt so future regenerations of
      an addendum-bearing room respect the addendum.
- [ ] An update endpoint accepts a `prompt_addendum` change for a
      specific room — either extending `PATCH /projects/{id}` to
      accept nested room updates or adding a dedicated
      `PATCH /projects/{id}/rooms/{rid}` endpoint. The choice is
      called out in the PR description with rationale.
- [ ] A pencil icon is rendered next to each room title in the
      project detail page. Clicking it opens a popover containing a
      textarea bound to `room.prompt_addendum`, prefilled with the
      current value (if any). Save calls the update endpoint and
      closes the popover; cancel discards.
- [ ] Saving an addendum never triggers automatic regeneration —
      the user must explicitly Generate / Regenerate to apply the
      addendum to images.
- [ ] Per the PRD's Retry semantics note: Retry Same Prompt does
      NOT re-run the composer. To pick up a new addendum on an
      existing variation the user must use Edit Prompt (slice 004)
      or regenerate the whole room.
- [ ] `tests/test_prompt_composer.py` is added with table-driven
      tests covering: project prompt only; project + brief; project
      + brief + room addendum; project + brief + room addendum +
      variation override (override wins); empty/None inputs at
      each layer; whitespace handling.
- [ ] A new Playwright scenario covers: open the pencil-icon
      popover; type an addendum; save; trigger a generation on that
      room; assert the rendered prompt or generation metadata
      reflects the addendum.
- [ ] Local checks pass before commit:
      `uv run pytest tests/ --ignore=tests/integration -v`,
      `cd frontend && npx playwright test`,
      `cd frontend && npm run build`,
      `cd frontend && npx next lint`.

## Blocked by

None - can start immediately.

## User stories addressed

Reference by number from the parent PRD:

- User story 10
- User story 11
- User story 12
- User story 13
