## Parent PRD

`prds/2026-05-01-image-pipeline-and-project-ux-overhaul-prd.md`

## What to build

`PromptSummarizer` — given a long prompt, returns a ≤240-char summary
via the existing LLM client; deterministic truncation fallback when the
LLM is unavailable. `StagingProject` schema gains `prompt_summary`.
`PATCH /api/v1/staging/projects/{id}` accepts `prompt`, `prompt_summary`
(server regenerates via `PromptSummarizer` if not provided), and
structured brief sections. Editing a prompt MUST NOT trigger image
regeneration.

See PRD sections "PromptSummarizer", "Modified — `models/staging.py`",
"API contracts", and user stories 6, 9, 10, 11.

## Acceptance criteria

- [ ] `PromptSummarizer` returns ≤240-char summary; truncation fallback used when LLM client raises
- [ ] `StagingProject` model exposes `prompt_summary`; persisted to Cosmos
- [ ] `PATCH /projects/{id}` updates prompt and refreshes `prompt_summary` (provided or regenerated)
- [ ] PATCH never enqueues regeneration jobs
- [ ] Unit test: LLM stub raises → truncation fallback applied
- [ ] API test: PATCH updates summary and returns it; no jobs created

## Blocked by

None - can start immediately.

## User stories addressed

- User story 6
- User story 9
- User story 10
- User story 11
