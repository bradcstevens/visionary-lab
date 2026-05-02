## Parent PRD

`prds/2026-05-01-image-pipeline-and-project-ux-overhaul-prd.md`

## What to build

Lazy backfill so historical projects gain thumbnails on first view. When
a `Variation` is read without `thumb_url` / `md_url`, the read path
enqueues a thumbnail-only job (or runs `ThumbnailDeriver` inline if the
original blob is small enough) and patches the variation record. No
batch migration job; backfill is purely lazy on read.

See PRD sections "Out of Scope — A migration job for legacy project
briefs" (same lazy principle) and user story 4.

## Acceptance criteria

- [ ] Reading a variation with missing variants triggers derivation and persists `thumb_url` / `md_url`
- [ ] Repeat reads of the same variation do not re-derive
- [ ] No additional Azure resources required for the backfill path
- [ ] Integration test: seed a legacy variation without variants, read it via the API, assert variants exist on the second read

## Blocked by

- Blocked by `010-thumbnail-deriver-and-pipeline.md`

## User stories addressed

- User story 4
