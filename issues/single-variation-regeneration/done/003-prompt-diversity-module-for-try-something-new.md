# Prompt diversity module for Try Something New

## Parent PRD

`prds/2026-04-29-single-variation-regeneration-prd.md`

## What to build

Make "Try Something New" **demonstrably different** from the rejected variation, instead of relying on LLM temperature to (hopefully) produce divergence.

Extract a small **pure deep module** — `prompt_diversity` — exposing a single function `build_diversifying_prompt(rejected_prompt, base_prompt_or_brief, room_analysis) -> str`. The function takes the rejected prompt as negative context and the base prompt or design brief as the user's intent, and returns a steering instruction that asks the LLM to depart meaningfully from the rejected aesthetic while staying faithful to the base intent.

The endpoint and pipeline are wired to use this module on the fresh-regen path:

- The regen endpoint passes the previously-rejected `adapted_prompt` (if any) into the pipeline call.
- The pipeline call threads it down to both prompt-generation paths: the design-brief path (`BriefGeneratorService.brief_to_prompts`) and the no-brief path (`adapt_prompt`), each via a thin wrapper around `build_diversifying_prompt`.

The new module is pure (no I/O, no Azure clients) so it can be tested in isolation with hand-crafted inputs.

See PRD sections **Implementation Decisions → Backend** (the `prompt_diversity` extraction bullet) and **User Stories** items 3, 4, 25 for full context.

## Acceptance criteria

- [ ] New module exposes `build_diversifying_prompt(rejected_prompt, base, room_analysis) -> str`
- [ ] Module is pure: no Azure SDK calls, no LLM client calls, no I/O
- [ ] When `rejected_prompt` is `None` or empty, the function returns the base content unchanged
- [ ] When `rejected_prompt` is non-empty, the output includes the rejected prompt as exclusion / negative-direction context
- [ ] When `rejected_prompt` is non-empty, the base prompt or brief content survives in the output (user intent is not dropped)
- [ ] Endpoint passes the prior `adapted_prompt` into `process_single_variation` on `fresh` strategy
- [ ] Pipeline threads the prior prompt to both the brief path and the no-brief path
- [ ] Unit tests cover the three input modes above
- [ ] Integration test: regen with `strategy=fresh` and a known prior prompt — the LLM call site receives a prompt that includes the prior prompt as negative context

## Blocked by

None - can start immediately.

## User stories addressed

Reference by number from the parent PRD:

- User story 3 (Try Something New)
- User story 4 (demonstrably different)
- User story 25 (prompt_diversity testable in isolation)
