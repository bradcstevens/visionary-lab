# 004 — Chat-driven overrides and regenerate preserves overrides

## Parent PRD

`issues/2026-04-29-per-image-object-quantities-prd.md`

## What to build

The "AI does the right thing" half of the PRD. Two related flows ship together because they share the same generator-service plumbing (name-tagged overrides, name-to-UUID substitution, name-based reconciliation):

1. **Chat-driven pre-fill.** When the design chat conversation explicitly differentiates quantities or placement between specific images ("eight in the side yard, three in the front yard"), the LLM populates `per_image_objects` up front so the user lands in the editor with their intent already roughed in.
2. **Regenerate preserves overrides.** When the user clicks regenerate after further chat, the surviving palette is matched against the prior palette by case-insensitive whitespace-trimmed name, and matching per-image overrides are carried forward with their `object_id` rewritten to the new UUID. Unmatched overrides are dropped, and a non-blocking toast tells the user how many were preserved and how many were dropped.

End-to-end behaviour delivered by this slice:

- LLM emits per-image overrides tagged by `object_name` (not `object_id`, since UUIDs do not exist in LLM output). The generator substitutes each `object_name` with the corresponding palette entry's UUID at parse time.
- Override entries whose `object_name` does not match any palette entry are dropped. Override entries whose `room_id` is not in `image_analyses` are also dropped.
- The regenerate flow passes the previous brief alongside the new chat context. The new brief is assembled, then per-image overrides are reconciled by name. The endpoint response includes a `reconciliation_summary` so the wizard can show a non-blocking toast: "Carried forward N per-image quantity overrides; M were dropped because their objects are no longer in the palette." The toast appears only when M > 0.

See PRD sections "AI brief generation" and "API" (generate-brief endpoint), and "Frontend" (regenerate flow + toast).

## Acceptance criteria

- [ ] `BRIEF_GENERATION_PROMPT` is extended so the LLM may populate `per_image_objects` only when the conversation explicitly differentiates quantities or placement between specific images. Each LLM-emitted override carries an `object_name` tag instead of an `object_id`.
- [ ] `BriefGeneratorService.generate_brief` walks `per_image_objects` after assigning palette UUIDs and substitutes each `object_name` reference with the corresponding assigned UUID. Override entries whose `object_name` does not match any palette entry are dropped at parse time. Override entries whose `room_id` is not in `image_analyses` are dropped.
- [ ] `brief_resolver.reconcile_overrides_by_name(prev_brief, new_brief) -> (new_brief, ReconcileSummary)` exists. For each prior override, it looks up the prior `object_id` to get the prior name, then matches that name (case-insensitive, whitespace-trimmed) against the new palette. Matches have their `object_id` rewritten to the new UUID and are carried forward; non-matches are dropped. `ReconcileSummary` carries `carried_forward: int` and `dropped: int` counts.
- [ ] `BriefGeneratorService.generate_brief` accepts an optional `previous_brief` argument. When supplied, it runs `reconcile_overrides_by_name` after assembling the new brief and returns a `(brief, ReconcileSummary)` tuple.
- [ ] The generate-brief API endpoint accepts an optional `previous_brief` payload field and includes a `reconciliation_summary` in the response.
- [ ] The wizard's regenerate-brief call passes the current brief as `previous_brief`. After a successful response, if `reconciliation_summary.dropped > 0`, the wizard surfaces a non-blocking toast: "Carried forward N per-image quantity overrides; M were dropped because their objects are no longer in the palette." If `dropped == 0`, no toast is shown.
- [ ] Unit tests for `brief_resolver.reconcile_overrides_by_name`: identical-name carry-forward; renamed objects dropped; case-insensitive matching; whitespace-trimmed matching; summary counts.
- [ ] Unit tests for `BriefGeneratorService.generate_brief`: `object_name` references in LLM-emitted `per_image_objects` are substituted to `object_id`; entries with unknown names or unknown `room_id`s are dropped.
- [ ] New Playwright E2E scenario folder under `tests/projects/regenerate-preserves-overrides/`. The user generates a brief, edits qty=8 on an object for image A, triggers regenerate from chat with a deterministic LLM stub so the regenerated palette name matches, and asserts the qty=8 override survived and the reconciliation toast was surfaced when at least one override was dropped. Reuses the existing activity-log SSE event-capture pattern.
- [ ] `uv run pytest tests/ --ignore=tests/integration -v`, `cd frontend && npx next lint`, `cd frontend && npm run build`, and `cd frontend && npx playwright test` all pass. Save Playwright reports under `tests/playwright/<YYYY-MM-DD-HHMMSS>/`.

## Blocked by

- Blocked by `issues/003-per-image-object-overrides.md`

## User stories addressed

- User story 12
- User story 16
- User story 17
- User story 30 (regenerate-preserves-overrides half)
