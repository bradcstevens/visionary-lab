# Inline-feedback Retry on the projects-list error banner

## Context

Companion to commit `3752238` (`fix(projects): inline spinner on
interrupted-banner Refresh`), which fixed the same anti-pattern on the
project detail page. The list page still has the bug.

## Problem

`frontend/app/projects/page.tsx:150-153` — the load-error banner's
Retry button calls `loadProjects` directly:

```tsx
<Button onClick={loadProjects} variant="outline" size="sm" className="mt-1">
  <RefreshCw className="h-3.5 w-3.5 mr-1.5" />
  Retry
</Button>
```

`loadProjects` flips an `isLoading` state that is consumed by a
page-level guard which replaces the entire view with a generic
"Loading projects…" spinner. Effect: clicking Retry destroys the error
banner (including the technical details the user just expanded), shows
a full-page spinner, and on failure re-renders the banner from scratch.
There is no button-local feedback and the user loses any state they had
in the banner.

## What to build

Mirror the pattern landed for the project-detail page in commit
`3752238`:

1. Add an `isRetrying` state alongside whatever local state the page
   already tracks for the banner.
2. Refactor `loadProjects` to accept `{ blocking?: boolean } = {}` and
   only call `setIsLoading(true)` when `blocking` is true. Defaults to
   `true` so existing callers (initial mount, route changes, any
   non-banner caller) keep their current full-page-loader behavior.
3. The `finally` block must always reset `isLoading` on the latest
   load id (use the same race-safe pattern as the project-detail page),
   so a stranded `isLoading=true` from a prior blocking call cannot
   persist after a later non-blocking call wins.
4. Replace the bare `onClick={loadProjects}` with an explicit arrow
   function that toggles `setIsRetrying`:

   ```tsx
   onClick={() => {
     setIsRetrying(true);
     loadProjects({ blocking: false }).finally(() => setIsRetrying(false));
   }}
   ```

5. Render `<Loader2 className="h-3.5 w-3.5 mr-1.5 animate-spin" />`
   when `isRetrying` is true; render the existing `<RefreshCw />`
   otherwise. Set `disabled={isRetrying}`.

## Acceptance criteria

- [ ] Clicking Retry on the projects-list load-error banner shows an
  inline spinner inside the button and disables it for the duration of
  the in-flight fetch.
- [ ] The banner stays mounted during the retry. The page does NOT
  flash to a generic "Loading projects…" spinner.
- [ ] On a successful retry, the banner unmounts and the projects grid
  renders as today.
- [ ] On a failed retry, the banner stays mounted with the new error
  detail and the Retry button is re-enabled with the `<RefreshCw />`
  icon.
- [ ] All other call sites of `loadProjects` (initial mount, anything
  triggered by route changes or auth changes) still gate the page-level
  loader as today (`blocking` defaults to `true`).
- [ ] No regressions in existing Playwright specs that touch
  `/projects`.

## Out of scope

- Any refactor of the page-level loader pattern itself.
- Any change to `/projects/[id]` (already shipped in `3752238`).
- Folding `isLoading` and `isRetrying` into a single state object — the
  two states model orthogonal things and merging them tends to leak
  bugs across callers.

## Pointers

- Commit `3752238` for the exact pattern, race-safety reasoning, and
  spinner convention.
- `frontend/app/projects/[id]/page.tsx` — `loadProject` `blocking`
  parameter and the interrupted-banner Refresh button.
- `frontend/components/ui/button.tsx` — Button does NOT have a
  `loading` prop; the codebase convention is the inline `<Loader2 />`
  pattern.
