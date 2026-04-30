# Regenerate Brief Preserves Overrides — Scenario

Reference scenario for issue 004 of `prds/2026-04-29-per-image-object-quantities-design-prd.md`. Covers the regenerate-brief flow where the user edits per-image overrides on step 4 of the wizard, goes back to step 3, chats more, and re-generates the brief. Surviving overrides (matched by case-insensitive, whitespace-trimmed object name) carry forward onto the new palette UUIDs; unmatched overrides surface as a non-blocking toast.

## Why this scenario

The issue 003 flow only covers a single brief generation. In practice users iterate: they generate a brief, edit qty=8 on a Lavender row in their backyard, look at the result, decide they want a different overall palette, return to chat, regenerate.

Without reconciliation, regenerating discards every per-image override the user just typed in — even ones whose object survived under a new UUID — because the resolver matches by `object_id`. With reconciliation:

- Lavender survives a regenerate that swaps the LLM-generated palette UUIDs (case-insensitive name match against the new palette).
- "Pine" → "Pine Tree" rename drops the override (no fuzzy match), and the toast tells the user how many were dropped so they can re-edit if needed.
- Ambiguous duplicate names (e.g., two Lavender entries in either palette) are dropped to avoid silent misattribution.

## Test coverage that uses this scenario

### Backend

- `tests/test_brief_resolver.py::TestReconcileOverridesByName` — covers identical names, case-insensitive match, whitespace trim, orphan-in-prev → drop, duplicate names in either palette → ambiguous drop, `enabled=False` survives reconcile, `quantity=0` survives reconcile, prev wins on (room_id, name) conflict, append-when-room-not-in-new, empty-prev no-op, input briefs not mutated.
- `tests/test_brief_generator.py::TestPerImageObjectsParsing` and `TestGenerateBriefWithPreviousBrief` — name → UUID substitution at LLM-parse time, unknown-name and unknown-room drops, malformed-row narrow try/except, ambiguous-duplicate-name drops, tuple return, final `valid_room_ids` filter after reconcile.
- `tests/test_staging_api.py::test_post_brief_returns_reconciliation_summary_zero_zero_when_no_previous_brief` and `test_post_brief_reconciles_previous_brief_overrides_by_name` — round-trip the API contract: response always includes `reconciliation_summary`; sending a `previous_brief` with an override on the prior Lavender UUID emerges with the override re-keyed onto the new Lavender UUID and a `dropped` count for the prior Pine override that is no longer in the regenerated palette.

### Frontend

- `frontend/tests/e2e/regenerate-preserves-overrides.spec.ts` — drives the wizard through step 1 (name) → step 2 (upload) → step 3 (chat → Generate Design Brief) → step 4 (edit Lavender qty=8) → Back → step 3 (chat again → Generate Design Brief) → step 4 again. Asserts: (a) the second `/brief` POST carries `previous_brief` with the qty=8 override; (b) the new Lavender row carries forward qty=8 (now under a new UUID); (c) the renamed "Pine Tree" row sits at its palette default (its prev override was dropped); (d) the broadened toast copy "could not be matched in the regenerated palette" is visible.

## Acceptance criteria check-list

The scenario satisfies issue 004's acceptance criteria:

- [x] `BRIEF_GENERATION_PROMPT` extended with strict `per_image_objects` schema and "MUST be omitted unless conversation differentiates" instruction; `room_id` must come from `image_analyses`; `object_name` must exactly match the palette (case-insensitive).
- [x] LLM-emitted `per_image_objects` are parsed with narrow per-row `try/except` so a single malformed row does not poison the whole brief.
- [x] `reconcile_overrides_by_name(prev, new) -> Tuple[DesignBrief, ReconcileSummary]` — case-insensitive whitespace-trimmed match; ambiguous duplicate names dropped on either side; prev wins on (room_id, name) conflict (user edits beat LLM auto-suggestions); empty-prev is a clean no-op.
- [x] `BriefGeneratorService.generate_brief` returns `Tuple[DesignBrief, ReconcileSummary]` always, with zero-counts when `previous_brief is None`. Final `valid_room_ids` filter applied after reconcile to clean up overrides for rooms no longer present.
- [x] POST `/projects/{id}/brief` accepts an optional `previous_brief` field and returns `{"brief", "reconciliation_summary"}`.
- [x] `stagingApi.generateBrief(projectId, conversationHistory, previousBrief?)` returns `{brief, reconciliation_summary}` and forwards `previous_brief` only when defined.
- [x] `NewProjectWizard.transitionToBriefEditor` passes `designBrief` as `previousBrief` when non-null. When `reconciliation_summary.dropped > 0` a non-blocking sonner toast surfaces the broadened copy: "Carried forward N per-image quantity overrides; M were dropped because their objects could not be matched in the regenerated palette."
