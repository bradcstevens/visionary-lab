# Header state machine and overflow menu cleanup

## Parent PRD

`prds/2026-04-30-per-room-generation-control-prd.md`

## What to build

A vertical slice that replaces the project header's misleading
`Generate` / `Regenerate All` ternary with a truthful three-state action,
backed by a small pure helper, and removes the duplicate header action from
the overflow menu.

End-to-end behavior on the project detail page:

- A new pure helper `getHeaderAction` lives next to the existing staging UI
  helpers (e.g. `frontend/utils/` or `frontend/services/stagingApi.ts` —
  pick whichever already houses similar pure helpers). It takes the project's
  rooms (or just the array of statuses) and returns a tagged union:
  - `{ kind: 'generate' }` — every room is `pending`
  - `{ kind: 'generate-remaining', count: number }` — at least one
    `completed` room exists alongside one or more `pending` or `failed` rooms;
    `count` equals `pending + failed`
  - `{ kind: 'hidden' }` — every room is `completed`
  No React, no IO; the helper is exhaustively reasoned about via the rendered
  header button.
- The inline `allPending ? <Generate> : !allPending ? <Regenerate All> : null`
  block in `ProjectDetailPage` becomes a 3-arm switch on
  `getHeaderAction(project.rooms)`:
  - `generate` → `Generate` button (Play icon, primary variant), wired to
    `startGeneration` (existing handler, unchanged)
  - `generate-remaining` → `Generate Remaining (N)` button (`RefreshCw` icon,
    outline variant), wired to `handleRegenerateAll`; `N` comes from the
    helper
  - `hidden` → no header button rendered
- In every visible state, the header button is `disabled` and shows a
  `Loader2` spinner whenever `isGenerating` is true. This preserves the
  existing single-active-stream invariant; no new lock is introduced.
- The duplicate `DropdownMenuItem` for `Regenerate all` in the overflow menu
  (and its surrounding separator wiring) is removed. The remaining items —
  `Add more images` and `Delete project` — keep their separator and
  disabled-while-`isGenerating` semantics.
- Both endpoints are reused unchanged:
  - Header `Generate` / `Generate Remaining (N)` →
    `POST /api/v1/staging/projects/{id}/generate` (already filters to
    `pending` and `failed` rooms server-side, so the renamed label simply
    tells the truth).
  - Per-row buttons (the other half of the PRD) continue to use
    `POST /.../rooms/{room_id}/regenerate`.
- The completed-room scenario where a stale-processing banner triggers
  `Reset & Retry` is untouched; only the header button itself is hidden when
  every room is completed.

See parent PRD sections **Solution → 2**, **Implementation Decisions →
`ProjectDetailPage`**, **Implementation Decisions → Deep module:
`getHeaderAction` helper**, **State table for header copy**, **Endpoints
reused**, and **Further Notes (outline variant rationale)** for full context.

## Acceptance criteria

- [ ] Frontend: a pure `getHeaderAction(rooms)` helper exists alongside
      existing staging UI helpers, returns the three-variant tagged union
      described above, and contains no React or IO
- [ ] Frontend: when every room is `pending`, the header renders a primary
      `Generate` button (Play icon)
- [ ] Frontend: when at least one room is `completed` and at least one room
      is `pending` or `failed`, the header renders an outline-variant
      `Generate Remaining (N)` button (`RefreshCw` icon) where N equals the
      count of `pending + failed` rooms
- [ ] Frontend: when every room is `completed`, no header button is rendered
- [ ] Frontend: in every visible state, the header button is disabled and
      shows a `Loader2` spinner while `isGenerating` is true
- [ ] Frontend: clicking `Generate Remaining (N)` calls
      `handleRegenerateAll`, which posts to
      `POST /api/v1/staging/projects/{id}/generate`; completed rooms remain
      untouched at stream end
- [ ] Frontend: the duplicate `Regenerate all` `DropdownMenuItem` is removed
      from the overflow menu; only `Add more images` and `Delete project`
      remain (with the separator preserved)
- [ ] Playwright: `frontend/tests/e2e/header-generate-remaining-label.spec.ts`
      covers a parameterized fixture and asserts the rendered header label /
      presence for each state:
      - All-pending → header label is `Generate` (no count)
      - One completed + 12 pending → header label is `Generate Remaining (12)`
      - All completed → header button is not present
      - In-flight (`isGenerating` true) → header button is disabled and shows
        a spinner
- [ ] Existing specs `frontend/tests/e2e/project-generation.spec.ts` and
      `frontend/tests/e2e/regenerate-preserves-overrides.spec.ts` still pass
- [ ] Frontend: `cd frontend && npm run build`, `npx next lint`, and the new
      Playwright spec all succeed locally before commit

## Blocked by

None - can start immediately. (Independent of `001-per-row-generate-on-pending-rooms.md`;
both edit `ProjectDetailPage` but in non-overlapping regions.)

## User stories addressed

Reference by number from the parent PRD:

- User story 2 (`Generate Remaining (N)` count is exactly pending + failed)
- User story 3 (header button hidden when every room is completed)
- User story 9 (clicking `Generate Remaining (12)` processes only those 12)
- User story 10 (all-pending project shows `Generate` without a count)
- User story 11 (header button shows spinner / disabled during active stream)
- User story 14 (overflow menu shows only `Add more images` and `Delete project`)
- User story 15 (no Generate / Regenerate Remaining action silently resets a
  completed room — frontend hides the header button when everything is done)
