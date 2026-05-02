## Parent PRD

`prds/2026-05-01-image-pipeline-and-project-ux-overhaul-prd.md`

## What to build

`StorageImage.tsx` gains a `variant` prop (`thumb` | `md` | `original`)
and a status-aware skeleton shown whenever the chosen variant URL is
missing or still loading. Failed loads surface a retry control rather
than a silent gap. Grid views switch to `variant="thumb"`; lightboxes
switch to `variant="md"`. Every code path that previously could render
`<img src="undefined">` is eliminated.

See PRD sections "Modified — `StorageImage.tsx`", "VariationThumbnail",
and user stories 1, 2, 3, 5.

## Acceptance criteria

- [ ] `StorageImage` accepts `variant` and selects the corresponding URL from props
- [ ] Skeleton shown while variant URL is missing or `<img>` is loading; reflects job status when available
- [ ] On `<img>` `onError`, a retry button is shown; clicking it re-attempts the load
- [ ] Project grid uses `variant="thumb"`; lightbox uses `variant="md"`
- [ ] Playwright test: thumbnail grid renders URLs containing `.thumb.` and never renders `<img src="undefined">`
- [ ] Playwright test: simulated load failure surfaces the retry control
- [ ] `cd frontend && npm run build` and `npx next lint` pass

## Blocked by

- Blocked by `010-thumbnail-deriver-and-pipeline.md`

## User stories addressed

- User story 1
- User story 2
- User story 3
- User story 5

## Notes (closed)

Shipped: StorageImage variant prop + thumbUrl/mdUrl/originalUrl;
skeleton shown when chosen variant URL is missing or while loading;
on terminal load failure a Retry button (data-testid storage-image-retry)
invalidates the SAS cache and remounts the <img>. VariationThumbnail
now passes variant="thumb" and thumb_url; ImageLightbox passes
variant="md" and the new mdUrl alongside the original. Variation type
gained thumb_url/md_url; LightboxImage gained mdUrl.

Playwright e2e tests for the grid (URLs contain .thumb.) and simulated
load failure (retry control surfaces) are DEFERRED — they need an
e2e fixture project with thumbnail-enabled variations and a backend
that serves the .thumb. URLs. Behavior is pinned by 15 vitest tests
covering variant resolution, skeleton states, the auto+manual retry
flow, and src-change reset.

