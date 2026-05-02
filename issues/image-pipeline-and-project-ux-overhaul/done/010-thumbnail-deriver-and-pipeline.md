## Parent PRD

`prds/2026-05-01-image-pipeline-and-project-ux-overhaul-prd.md`

## What to build

`ThumbnailDeriver` — pure transform that takes a source image (bytes or
blob ref) and produces sibling `thumb.webp` (512px max-edge, q70) and
`md.webp` (1024px max-edge, q80) blobs. Wired into the tail of every
variation job in `staging_pipeline.py`, run synchronously before the
job reports `succeeded`. `Variation` model gains `thumb_url`, `md_url`,
and `revision` fields populated when the variant blobs land. Pillow is
already a dependency.

See PRD sections "ThumbnailDeriver", "Modified — `staging_pipeline.py`,
`models/staging.py`", and "Further Notes".

## Acceptance criteria

- [ ] `ThumbnailDeriver` produces both webp variants with correct max-edge dimensions and quality settings
- [ ] Sibling blob names follow a predictable pattern (e.g. `<original>.thumb.webp`, `<original>.md.webp`)
- [ ] `staging_pipeline.py` invokes the deriver at the tail of each variation job before marking the job `succeeded`
- [ ] `Variation` model exposes `thumb_url`, `md_url`, `revision`; populated for all newly generated variations
- [ ] Unit test: feed a sample PNG, assert both variants exist with correct dimensions
- [ ] `uv run pytest tests/ --ignore=tests/integration -v` passes

## Blocked by

- Blocked by `003-jobworker-consumer.md`

## User stories addressed

- User story 1
- User story 3
