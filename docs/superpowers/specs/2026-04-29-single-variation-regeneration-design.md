# Single Variation Regeneration — Design Spec

**Date:** 2026-04-29
**Status:** Approved
**Scope:** Allow users to regenerate a single variation image independently, with choice to retry the same prompt or get a fresh creative take.

---

## Problem

Currently, regenerating a variation requires regenerating the entire room (all variations re-analyzed and re-generated). This is wasteful when only one variation needs a redo and destroys other variations the user may have liked.

## Solution

A dedicated backend endpoint for single-variation regeneration, with frontend affordances in both the variation thumbnail hover and the image lightbox.

## Architecture Approach

**Approach A — New dedicated variation regeneration endpoint.** Clean API, minimal blast radius, reuses existing SSE infrastructure.

---

## Backend

### New Endpoint

```
POST /api/v1/staging/projects/{project_id}/rooms/{room_id}/variations/{variation_id}/regenerate
Query params: strategy=retry|fresh (default: fresh)
Response: SSE stream (text/event-stream)
```

### Behavior

1. Look up project → room → variation. Return 404 if any not found.
2. **Strategy: `retry`**
   - Read `variation.generation_metadata.adapted_prompt` from the existing variation.
   - If no previous prompt exists, fall back to `fresh` strategy.
   - Reset the single variation to pending (clear image_url, error, metadata).
   - Generate using the saved prompt — no LLM call needed.
3. **Strategy: `fresh`**
   - Reset the single variation to pending.
   - Fetch the room's original image from blob storage.
   - If project has a `design_brief`, use `BriefGeneratorService.brief_to_prompts()` to generate 1 prompt for this room (consistent with the design brief flow).
   - Otherwise, re-analyze the room image (existing `analyze_room`) and generate 1 new adapted prompt via LLM (existing `adapt_prompt` with `n_variations=1`).
   - Generate using the new prompt.
4. Process through the existing image pipeline (`ImagePipelineRequest` with `action=EDIT`).
5. Stream SSE events:
   - `variation_started` → `variation_completed` or `variation_failed` → `project_completed`
6. After completion, recalculate room status (from variation statuses) and project status.
7. Persist all status changes to Cosmos DB.

### Pipeline Method

Add `process_single_variation()` to `StagingPipeline`:
- Accepts: project, room, variation, adapted_prompt (string)
- Reuses the existing image generation logic from `process_room()` but for a single variation
- Yields SSE events for that one variation

### No Changes to Existing Endpoints

- `POST /projects/{id}/generate` — unchanged
- `POST /projects/{id}/rooms/{room_id}/regenerate` — unchanged

---

## Frontend

### API Layer (`stagingApi.ts`)

New function:
```typescript
export function streamVariationRegeneration(
  projectId: string,
  roomId: string,
  variationId: string,
  strategy: 'retry' | 'fresh',
  onEvent: StagingStreamEventCallback,
): () => void
```

Same SSE streaming pattern as `streamRoomRegeneration`. Returns cleanup function.

### State Management (Project Detail Page)

New state: `regeneratingVariationId: string | null`
- Tracks which specific variation is being regenerated
- **Not** the global `isGenerating` — allows other variations to remain interactive
- When set, only that variation's thumbnail shows a processing state

New handler: `handleRegenerateVariation(room: Room, variationIndex: number, strategy: 'retry' | 'fresh')`
- Sets `regeneratingVariationId` to the variation's ID
- Calls `streamVariationRegeneration`
- On completion: clears `regeneratingVariationId`, reloads project data

### Variation Thumbnail (`VariationThumbnail.tsx`)

**New prop:** `onRegenerate?: (strategy: 'retry' | 'fresh') => void`

**Completed state — hover overlay:**
- Semi-transparent dark overlay appears on hover
- Contains a `RefreshCw` icon button (centered or bottom-right)
- Clicking the refresh icon shows a small dropdown/popover:
  - "🔄 Retry Same" — calls `onRegenerate('retry')`
  - "✨ Try Something New" — calls `onRegenerate('fresh')`
- The existing click-to-lightbox behavior remains on the image itself (not the overlay button)

**New prop:** `isRegenerating?: boolean`
- When true, renders the existing `processing` state (spinner + badge) instead of the completed image
- Replaces the image temporarily until the new one arrives

### Image Lightbox (`ImageLightbox.tsx`)

**New props:**
- `onRegenerate?: (strategy: 'retry' | 'fresh') => void`
- `isRegenerating?: boolean`

**Top bar additions:**
- `RefreshCw` button added next to the existing "open in new tab" button
- Clicking it shows a dropdown with "Retry Same" / "Try Something New"
- While regenerating: image area shows a centered spinner overlay, buttons disabled
- When regeneration completes: new image replaces the old one (lightbox stays open)

### RoomGroup Component (`RoomGroup.tsx`)

**New prop:** `onRegenerateVariation?: (room: Room, variationIndex: number, strategy: 'retry' | 'fresh') => void`
**New prop:** `regeneratingVariationId?: string | null`

Passes through to `VariationThumbnail`:
- `onRegenerate` prop wired to the parent handler
- `isRegenerating` computed from `regeneratingVariationId === variation.id`

---

## Edge Cases

1. **No previous prompt (retry strategy):** Fall back to `fresh` strategy. Show toast: "No previous prompt found — generating a fresh take instead."
2. **Regeneration while global generation is running:** Disabled. The `isGenerating` flag disables the regenerate overlay/button. User must wait for batch to finish.
3. **Variation already processing:** Endpoint returns 409 Conflict if variation is already in `processing` state.
4. **Lightbox open during regeneration:** Lightbox stays open, shows spinner, auto-updates with new image URL on completion.
5. **Network error during SSE stream:** Existing error handling pattern applies — toast notification + error state on the variation.

---

## Files Changed

### Backend
- `backend/api/endpoints/staging.py` — new endpoint
- `backend/core/staging_pipeline.py` — new `process_single_variation()` method

### Frontend
- `frontend/services/stagingApi.ts` — new `streamVariationRegeneration` function
- `frontend/app/projects/[id]/page.tsx` — new state, handler, updated props
- `frontend/components/staging/VariationThumbnail.tsx` — hover overlay, regenerate props
- `frontend/components/staging/ImageLightbox.tsx` — regenerate button, loading state
- `frontend/components/staging/RoomGroup.tsx` — pass-through props

### Tests
- Backend: unit test for new endpoint (retry + fresh strategies, 404 cases, 409 conflict)
- Frontend: Playwright test for regeneration flow from thumbnail and lightbox
