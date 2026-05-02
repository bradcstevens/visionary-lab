## Parent PRD

`prds/2026-05-01-image-pipeline-and-project-ux-overhaul-prd.md`

## What to build

`RegionalPackLoader` — loads JSON content packs keyed by `(domain,
region)` from the backend. Ship the Castle Rock plant pack as the
reference example. A simple domain classifier seeds the wizard's first
step (e.g. landscaping → loads `landscaping/castle-rock` pack), and the
loaded pack feeds quick-reply chips into subsequent wizard steps. Packs
ship as code via PR (no user upload).

See PRD sections "RegionalPackLoader", "Out of Scope — User-uploaded
regional packs", and user stories 27–30.

## Acceptance criteria

- [ ] `RegionalPackLoader` returns a pack by `(domain, region)`; unknown key returns empty
- [ ] Castle Rock landscaping pack ships as JSON in the repo
- [ ] Domain classifier maps a free-text user description to a domain id
- [ ] Wizard step config consumes the pack to surface quick-reply chips
- [ ] Unit test: pack lookup hit/miss; classifier returns expected domain for landscaping prompt
- [ ] `uv run pytest tests/ --ignore=tests/integration -v` passes

## Blocked by

- Blocked by `015-brief-section-registry-and-composer.md`

## User stories addressed

- User story 27
- User story 28
- User story 29
- User story 30
