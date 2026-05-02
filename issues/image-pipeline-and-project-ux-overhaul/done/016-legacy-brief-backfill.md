## Parent PRD

`prds/2026-05-01-image-pipeline-and-project-ux-overhaul-prd.md`

## What to build

Lazy backfill of the eight-section structure for projects created before
this feature shipped. `backend/core/brief_generator.py` extended to
extract the eight canonical sections from the legacy free-form brief on
first read, persist the structured form back to the project record, and
serve it from then on. No batch migration job.

See PRD sections "Modified — `brief_generator.py`", "Out of Scope — A
migration job for legacy project briefs", and user story 37.

## Acceptance criteria

- [ ] On read, a project lacking structured sections has them extracted from its existing brief and persisted
- [ ] Subsequent reads return the persisted structured sections without re-extracting
- [ ] Extraction is deterministic for a given input (cache-friendly)
- [ ] Unit test: legacy brief → 8 populated sections → persisted; second read returns same sections without invoking extractor
- [ ] No regeneration jobs created during backfill

## Blocked by

- Blocked by `015-brief-section-registry-and-composer.md`

## User stories addressed

- User story 37
