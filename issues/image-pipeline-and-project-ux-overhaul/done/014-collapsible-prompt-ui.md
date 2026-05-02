## Parent PRD

`prds/2026-05-01-image-pipeline-and-project-ux-overhaul-prd.md`

## What to build

`frontend/app/projects/[id]/page.tsx` shows the `prompt_summary` by
default with a Show full prompt affordance. Expanding reveals the full
prompt in an editable view; saving calls the PATCH endpoint and the
collapsed summary refreshes from the response. Saving an edited prompt
explicitly does NOT trigger regeneration.

See PRD sections "Modified — `app/projects/[id]/page.tsx`" and user
stories 6, 7, 8, 9, 10.

## Acceptance criteria

- [ ] Project page renders `prompt_summary` collapsed by default with a Show full prompt control
- [ ] Expanding shows the full prompt in an editable textarea
- [ ] Save calls `PATCH /projects/{id}`; collapsed summary updates from the response
- [ ] No regeneration jobs are created on save (assert via Playwright)
- [ ] Playwright test: collapsed summary visible → expand → edit → save → updated summary, no new jobs
- [ ] `cd frontend && npm run build` and `npx next lint` pass

## Blocked by

- Blocked by `013-prompt-summarizer-and-patch.md`

## User stories addressed

- User story 6
- User story 7
- User story 8
- User story 9
- User story 10

## Notes (frontend slice complete)

- Component CollapsiblePrompt.tsx renders prompt_summary collapsed by
  default with "Show full prompt" toggle, expands to editable textarea,
  Save calls handleProjectSettingsSave({ prompt }) which uses the
  existing PATCH endpoint. Backend regenerates prompt_summary
  server-side (issue 013); the parent re-renders with the refreshed
  summary on success.
- 11 vitest unit tests pin: collapsed default, summary fallback to
  truncated prompt, expand-edit-save flow, summary refresh from
  parent re-render, cancel discards edits, save disabled while
  in-flight / unchanged / empty, save failure keeps editor open,
  no regeneration callback exposed.
- Playwright spec deferred — needs running backend with
  PromptSummarizer wired (consistent with other slices in this PRD
  that deferred Playwright pending integration env).
