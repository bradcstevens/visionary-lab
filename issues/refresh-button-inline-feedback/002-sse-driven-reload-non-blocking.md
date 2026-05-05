# SSE-driven reloads should be non-blocking

## Context

Latent issue surfaced while reviewing commit `3752238`
(`fix(projects): inline spinner on interrupted-banner Refresh`). The
fix added a `blocking` parameter to `loadProject` but only opted out
the one Refresh button. SSE-driven reloads still default to
`blocking=true` and continue to flash the full-page "Loading
project…" loader on every reconcile.

## Problem

`frontend/app/projects/[id]/page.tsx:223-230` — `debouncedReload` is
called by SSE event handlers (`runs.event`, `runs.refresh`,
`projects.changed`, etc.) several times during a normal generation
run:

```tsx
const debouncedReload = useCallback(() => {
  if (reloadTimerRef.current) clearTimeout(reloadTimerRef.current);
  reloadTimerRef.current = setTimeout(() => {
    reloadTimerRef.current = null;
    loadProject();
  }, 500);
}, [loadProject]);
```

`loadProject()` with no arguments resolves to `blocking=true`, which
trips the page-level guard `if (isLoading || !project)` at
~line 1265 and replaces the entire view (header, room grid,
in-flight panel, recovery banner) with a generic "Loading project…"
spinner for the duration of the fetch. Even with a 500ms debounce
this is visible flicker — the user sees their grid disappear and
reappear several times during a normal run.

The original page-level `isLoading` guard exists to handle:

1. Initial mount on route navigation (`project === null`).
2. Route changes within the same dynamic segment (Next App Router
   reuses the page component across `/projects/[id1]` →
   `/projects/[id2]`, and showing stale Project A while Project B
   loads in the same component instance is wrong).

Both cases also satisfy `!project` (case 1 directly; case 2 because
the projectId-change effect resets `project` before the next fetch
fires). The `isLoading` arm of the guard is therefore redundant for
its stated purpose, and harmful when SSE-driven background reloads
fire it.

## What to build

1. **Make `debouncedReload` non-blocking.** It is always a background
   reconcile triggered by SSE; it should never replace the view.
   Change line 228 to `loadProject({ blocking: false });`.
2. **Audit other `loadProject()` call sites** and convert any that
   are background reconciles (post-cancel reload, dismiss-action
   reload, anything that fires after the user has already seen the
   project) to `{ blocking: false }`. Keep `blocking: true` only on:
   - The initial mount effect.
   - The projectId-change effect (route navigation within
     `/projects/[id]`).
3. **Verify the page-level guard.** Once the only remaining
   `blocking: true` callers are mount/route-change paths, confirm
   that simplifying the guard to `if (!project)` is safe (it should
   be — the projectId-change effect already nulls `project` before
   the next fetch). If safe, simplify; if not, leave the
   `isLoading || !project` guard and document why in a comment.
4. **Optional follow-up:** if step 3 simplifies cleanly, the
   `blocking` parameter on `loadProject` can be removed entirely and
   non-blocking becomes the only mode. Decide based on whether any
   future caller would legitimately want the full-page loader (the
   answer is probably "no, only mount/route-change does, and those
   resolve via `!project`").

## Acceptance criteria

- [ ] During a normal generation run with SSE active, the page
  header, room grid, and any visible recovery banner stay mounted
  throughout the run. No full-page "Loading project…" flash on
  every reconcile.
- [ ] Initial mount on `/projects/[id]` still shows the full-page
  loader until the first fetch resolves (unchanged behavior).
- [ ] Route navigation `/projects/[id1]` → `/projects/[id2]` does
  not show stale Project A's data while Project B loads. The page
  shows the full-page loader during the transition (unchanged
  behavior).
- [ ] Post-cancel reload does not flash the full-page loader.
- [ ] Stream-lost banner Dismiss → reload does not flash the
  full-page loader.
- [ ] If the page-level guard is simplified, all existing Playwright
  specs that assert mount/route behaviors still pass:
  `recovery-banner-single`, `project-generation`,
  `project-generation-resume`, `project-generation-watchdog-regression`,
  `project-generation-staleness`, `queued-project-stays-processing`,
  `concurrent-room-generation`.

## Out of scope

- Refactoring the SSE subscription pattern itself.
- Changing the 500ms debounce window in `debouncedReload`.
- Touching `/projects` list page (covered by issue 001 in this
  folder).

## Pointers

- Commit `3752238` introduced the `blocking` parameter and the
  race-safe `finally` block on the latest load id.
- `frontend/app/projects/[id]/page.tsx:184-221` — current
  `loadProject` definition.
- `frontend/app/projects/[id]/page.tsx:223-230` — `debouncedReload`
  call site that should flip to `blocking: false`.
- `frontend/app/projects/[id]/page.tsx:~1265` — page-level
  `if (isLoading || !project)` guard to potentially simplify.
- `frontend/app/projects/[id]/page.tsx:132-136` — projectId-change
  effect that nulls `project` before the next fetch.
