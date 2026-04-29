# Single Variation Regeneration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow users to regenerate a single variation image independently with choice to retry the same prompt or get a fresh creative take.

**Architecture:** New dedicated backend endpoint for single-variation regeneration with SSE streaming, plus frontend regeneration affordances on both thumbnail hover and lightbox. State management uses a separate `regeneratingVariationId` to allow targeted UI updates without blocking the rest of the page.

**Tech Stack:** Python/FastAPI (backend), Next.js/React/TypeScript (frontend), SSE streaming, Radix UI components

---

## File Map

### Backend (modified)
| File | Responsibility |
|------|---------------|
| `backend/core/staging_pipeline.py` | New `process_single_variation()` method — generates one variation from a given prompt |
| `backend/api/endpoints/staging.py` | New `POST /projects/{id}/rooms/{room_id}/variations/{variation_id}/regenerate` endpoint |

### Frontend (modified)
| File | Responsibility |
|------|---------------|
| `frontend/services/stagingApi.ts` | New `streamVariationRegeneration()` function — SSE client for single variation |
| `frontend/components/staging/VariationThumbnail.tsx` | Hover overlay with regenerate dropdown on completed variations |
| `frontend/components/staging/ImageLightbox.tsx` | Regenerate button with strategy dropdown in lightbox top bar |
| `frontend/components/staging/RoomGroup.tsx` | Pass-through props for variation regeneration |
| `frontend/app/projects/[id]/page.tsx` | New state + handler for single-variation regeneration |

### Tests (new/modified)
| File | Responsibility |
|------|---------------|
| `tests/test_staging_api.py` | New tests for variation regeneration endpoint |
| `tests/test_staging_pipeline.py` | New test for `process_single_variation()` |

---

### Task 1: Backend — `process_single_variation()` pipeline method

**Files:**
- Modify: `backend/core/staging_pipeline.py` (add method after `process_room`, around line 275)
- Test: `tests/test_staging_pipeline.py`

- [ ] **Step 1: Write failing test for `process_single_variation`**

Add to `tests/test_staging_pipeline.py`:

```python
@pytest.mark.asyncio
async def test_process_single_variation_completes():
    """process_single_variation should yield variation_completed with image URL."""
    project = _make_project(n_rooms=1, n_variations=3)
    room = project.rooms[0]
    variation = room.variations[1]  # Target the second variation
    adapted_prompt = "Add a modern sofa with warm wood tones"

    pipeline_response = _make_pipeline_response()

    with patch("backend.core.staging_pipeline.StagingPipeline.__init__", return_value=None):
        pipeline = StagingPipeline.__new__(StagingPipeline)
        pipeline.image_pipeline = AsyncMock()
        pipeline.image_pipeline.process_pipeline = AsyncMock(return_value=pipeline_response)
        pipeline.blob_service = MagicMock()
        pipeline.blob_service.get_asset_content.return_value = (b"fake-image-bytes", "image/png")
        pipeline.storage_service = MagicMock()
        pipeline.semaphore = asyncio.Semaphore(1)

        events = []
        async for event in pipeline.process_single_variation(project, room, variation, adapted_prompt):
            events.append(event)

    event_types = [e["type"] for e in events]
    assert "variation_completed" in event_types
    completed_event = next(e for e in events if e["type"] == "variation_completed")
    assert completed_event["variation_index"] == 1
    assert completed_event["room_id"] == room.id
    assert variation.status == ItemStatus.COMPLETED
    assert variation.image_url is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/bradcstevens/code/github/bradcstevens/visionary-lab && uv run pytest tests/test_staging_pipeline.py::test_process_single_variation_completes -v`
Expected: FAIL with `AttributeError: 'StagingPipeline' object has no attribute 'process_single_variation'`

- [ ] **Step 3: Implement `process_single_variation` in staging_pipeline.py**

Add this method to the `StagingPipeline` class, after `process_room` (after line 275):

```python
    async def process_single_variation(
        self,
        project: StagingProject,
        room: Room,
        variation: Variation,
        adapted_prompt: str,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Regenerate a single variation using the provided prompt. Yields SSE events."""
        variation_index = next(
            (i for i, v in enumerate(room.variations) if v.id == variation.id), 0
        )

        variation.status = ItemStatus.PROCESSING
        self._update_room_in_project(project, room)

        start_time = time.monotonic()
        result = None
        elapsed_ms = 0
        try:
            image_content, _ = self.blob_service.get_asset_content(
                blob_name=self._extract_blob_name(room.original_image_url),
                container_name=settings.AZURE_BLOB_IMAGE_CONTAINER,
            )
            if image_content is None:
                raise RuntimeError(f"Image not found in blob storage: {room.original_image_url}")
            image_b64 = base64.b64encode(image_content).decode("utf-8")

            pipeline_request = ImagePipelineRequest(
                action=PipelineAction.EDIT,
                prompt=adapted_prompt,
                model=project.settings.model,
                n=1,
                size=project.settings.size,
                quality=project.settings.quality,
                response_format="b64_json",
                output_format="png",
                source_image_base64=[image_b64],
                save_options=PipelineSaveOptions(
                    enabled=True,
                    folder_path=f"staging/{project.id}/variations/{room.id}",
                ),
                analysis_options=PipelineAnalysisOptions(enabled=False),
            )

            result = await self.image_pipeline.process_pipeline(
                pipeline_request=pipeline_request,
                azure_storage_service=self.blob_service,
            )

            elapsed_ms = int((time.monotonic() - start_time) * 1000)

            if result.generation and result.save:
                saved = result.save
                saved_url = (
                    saved.saved_images[0].get("url")
                    if saved.saved_images
                    else None
                )
                if saved_url:
                    variation.image_url = saved_url
                    variation.status = ItemStatus.COMPLETED
                    variation.generation_metadata = {
                        "model": project.settings.model,
                        "adapted_prompt": adapted_prompt,
                        "generation_time_ms": elapsed_ms,
                    }
                else:
                    variation.status = ItemStatus.FAILED
                    variation.error = "Save succeeded but no image URL returned"
            else:
                variation.status = ItemStatus.FAILED
                variation.error = "Pipeline returned no generation result"

        except Exception as e:
            logger.error(f"Single variation regen failed for {variation.id}: {e}")
            variation.status = ItemStatus.FAILED
            variation.error = str(e)
            elapsed_ms = int((time.monotonic() - start_time) * 1000)

        token_usage = None
        if result and result.generation and result.generation.token_usage:
            tu = result.generation.token_usage
            token_usage = tu.get("total_tokens") if isinstance(tu, dict) else getattr(tu, "total_tokens", None)

        self._update_room_in_project(project, room)

        yield {
            "type": f"variation_{'completed' if variation.status == ItemStatus.COMPLETED else 'failed'}",
            "room_id": room.id,
            "variation_index": variation_index,
            "image_url": variation.image_url,
            "error": variation.error,
            "elapsed_ms": elapsed_ms,
            "tokens_used": token_usage,
            "model": project.settings.model,
        }
```

- [ ] **Step 4: Add missing imports to the test file**

Ensure these imports are at the top of `tests/test_staging_pipeline.py`:

```python
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from backend.core.staging_pipeline import StagingPipeline
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /Users/bradcstevens/code/github/bradcstevens/visionary-lab && uv run pytest tests/test_staging_pipeline.py::test_process_single_variation_completes -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
cd /Users/bradcstevens/code/github/bradcstevens/visionary-lab
git add backend/core/staging_pipeline.py tests/test_staging_pipeline.py
git commit -m "feat(backend): add process_single_variation pipeline method

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 2: Backend — Variation regeneration endpoint

**Files:**
- Modify: `backend/api/endpoints/staging.py` (add endpoint after `regenerate_room`, around line 339)
- Test: `tests/test_staging_api.py`

- [ ] **Step 1: Write failing tests for the new endpoint**

Add to `tests/test_staging_api.py`:

```python
def _project_with_completed_variation():
    """Helper: project with one room, one completed variation with metadata."""
    return {
        "id": "proj-123",
        "name": "Test",
        "prompt": "Modern minimalist",
        "status": "completed",
        "rooms": [{
            "id": "room-1",
            "label": "Living Room",
            "original_image_url": "https://acct.blob.core.windows.net/images/staging/proj/originals/photo.png",
            "status": "completed",
            "variations": [{
                "id": "var-1",
                "status": "completed",
                "image_url": "https://acct.blob.core.windows.net/images/staging/proj/variations/room-1/img.png",
                "generation_metadata": {
                    "model": "gpt-image-2",
                    "adapted_prompt": "Add a cozy reading nook with warm lighting",
                    "generation_time_ms": 5000,
                },
            }],
        }],
        "settings": {"variations_per_room": 1, "model": "gpt-image-2", "quality": "high", "size": "auto"},
    }


def test_regenerate_variation_not_found_project(client, mock_staging_deps):
    mock_container = mock_staging_deps["container"]
    mock_container.read_item.return_value = None
    response = client.post("/api/v1/staging/projects/nope/rooms/room-1/variations/var-1/regenerate?strategy=fresh")
    assert response.status_code == 404


def test_regenerate_variation_not_found_room(client, mock_staging_deps):
    mock_container = mock_staging_deps["container"]
    mock_container.read_item.return_value = _project_with_completed_variation()
    response = client.post("/api/v1/staging/projects/proj-123/rooms/bad-room/variations/var-1/regenerate?strategy=fresh")
    assert response.status_code == 404


def test_regenerate_variation_not_found_variation(client, mock_staging_deps):
    mock_container = mock_staging_deps["container"]
    mock_container.read_item.return_value = _project_with_completed_variation()
    response = client.post("/api/v1/staging/projects/proj-123/rooms/room-1/variations/bad-var/regenerate?strategy=fresh")
    assert response.status_code == 404
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/bradcstevens/code/github/bradcstevens/visionary-lab && uv run pytest tests/test_staging_api.py::test_regenerate_variation_not_found_project tests/test_staging_api.py::test_regenerate_variation_not_found_room tests/test_staging_api.py::test_regenerate_variation_not_found_variation -v`
Expected: FAIL with 404 (Method Not Allowed or similar — endpoint doesn't exist yet)

- [ ] **Step 3: Implement the endpoint**

Add to `backend/api/endpoints/staging.py` after the `regenerate_room` endpoint (after line 339):

```python
@router.post("/projects/{project_id}/rooms/{room_id}/variations/{variation_id}/regenerate")
async def regenerate_variation(
    project_id: str,
    room_id: str,
    variation_id: str,
    strategy: str = "fresh",
    storage: StagingStorageService = Depends(get_staging_storage),
    pipeline=Depends(get_staging_pipeline),
):
    """Regenerate a single variation. strategy=retry reuses the previous prompt; strategy=fresh generates a new one."""
    if strategy not in ("retry", "fresh"):
        raise HTTPException(status_code=400, detail="strategy must be 'retry' or 'fresh'")

    project_data = storage.get_project(project_id)
    if not project_data:
        raise HTTPException(status_code=404, detail="Project not found")

    clean = {k: v for k, v in project_data.items() if k != "doc_type" and not k.startswith("_")}
    project = StagingProject(**clean)

    room = next((r for r in project.rooms if r.id == room_id), None)
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")

    variation = next((v for v in room.variations if v.id == variation_id), None)
    if not variation:
        raise HTTPException(status_code=404, detail="Variation not found")

    if variation.status == ItemStatus.PROCESSING:
        raise HTTPException(status_code=409, detail="Variation is already being processed")

    # Determine the prompt to use
    adapted_prompt = None
    fallback_to_fresh = False

    if strategy == "retry":
        if variation.generation_metadata and isinstance(variation.generation_metadata, dict):
            adapted_prompt = variation.generation_metadata.get("adapted_prompt")
        elif hasattr(variation.generation_metadata, "adapted_prompt"):
            adapted_prompt = variation.generation_metadata.adapted_prompt
        if not adapted_prompt:
            fallback_to_fresh = True

    # Reset the variation
    variation.status = ItemStatus.PENDING
    variation.image_url = None
    variation.error = None

    # Update room status to processing
    room.status = ItemStatus.PROCESSING
    storage.update_project(project_id, json.loads(project.json()))

    async def event_stream():
        nonlocal adapted_prompt

        try:
            if strategy == "fresh" or fallback_to_fresh:
                # Check for design brief first
                if project.design_brief:
                    from backend.core.brief_generator import BriefGeneratorService
                    from backend.core import async_llm_client
                    from backend.models.design_brief import DesignBrief as DBModel, ImageAnalysis

                    brief = DBModel(**project.design_brief)
                    analyses = [ImageAnalysis(**a) for a in (project.analyses or [])]
                    if analyses:
                        brief_service = BriefGeneratorService(
                            async_llm_client=async_llm_client,
                            llm_deployment=settings.LLM_DEPLOYMENT,
                        )
                        brief_prompts = await brief_service.brief_to_prompts(
                            brief=brief,
                            image_analyses=analyses,
                            n_variations=1,
                        )
                        if room.id in brief_prompts and brief_prompts[room.id]:
                            adapted_prompt = brief_prompts[room.id][0]

                if not adapted_prompt:
                    # Fall back to standard prompt adaptation
                    import base64
                    image_content, _ = pipeline.blob_service.get_asset_content(
                        blob_name=pipeline._extract_blob_name(room.original_image_url),
                        container_name=settings.AZURE_BLOB_IMAGE_CONTAINER,
                    )
                    if image_content is None:
                        raise RuntimeError("Original image not found in storage")
                    image_b64 = base64.b64encode(image_content).decode("utf-8")
                    analysis = await pipeline.analyze_room(image_b64)
                    room_description = analysis.get("description", "A room")
                    prompts = await pipeline.adapt_prompt(
                        user_prompt=project.prompt,
                        room_analysis=room_description,
                        n_variations=1,
                    )
                    adapted_prompt = prompts[0]

            async for event in pipeline.process_single_variation(
                project, room, variation, adapted_prompt
            ):
                yield _sse_event(event["type"], event)

        finally:
            # Recalculate room and project status
            fresh = storage.get_project(project_id)
            if fresh:
                clean_fresh = {k: v for k, v in fresh.items() if k != "doc_type" and not k.startswith("_")}
                fresh_project = StagingProject(**clean_fresh)
                target_room = next((r for r in fresh_project.rooms if r.id == room_id), None)
                if target_room:
                    any_completed = any(v.status == "completed" for v in target_room.variations)
                    any_pending = any(v.status in ("pending", "processing") for v in target_room.variations)
                    if any_pending:
                        target_room.status = "processing"
                    elif any_completed:
                        target_room.status = "completed"
                    else:
                        target_room.status = "failed"
                any_room_processing = any(r.status in ("pending", "processing") for r in fresh_project.rooms)
                if not any_room_processing:
                    any_room_completed = any(r.status == "completed" for r in fresh_project.rooms)
                    fresh_project.status = "completed" if any_room_completed else "failed"
                storage.update_project(project_id, json.loads(fresh_project.json()))

        yield _sse_event("project_completed", {"status": "completed"})

    return StreamingResponse(event_stream(), media_type="text/event-stream")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/bradcstevens/code/github/bradcstevens/visionary-lab && uv run pytest tests/test_staging_api.py -v -k "regenerate_variation"`
Expected: All 3 tests PASS

- [ ] **Step 5: Run full backend test suite to check for regressions**

Run: `cd /Users/bradcstevens/code/github/bradcstevens/visionary-lab && uv run pytest tests/ --ignore=tests/integration -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
cd /Users/bradcstevens/code/github/bradcstevens/visionary-lab
git add backend/api/endpoints/staging.py tests/test_staging_api.py
git commit -m "feat(backend): add single variation regeneration endpoint

POST /projects/{id}/rooms/{room_id}/variations/{variation_id}/regenerate
Supports strategy=retry (reuse prompt) and strategy=fresh (new prompt).

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 3: Frontend — `streamVariationRegeneration` API function

**Files:**
- Modify: `frontend/services/stagingApi.ts` (add function after `streamRoomRegeneration`, around line 546)

- [ ] **Step 1: Add `streamVariationRegeneration` function**

Add after the `streamRoomRegeneration` function (after line 546) in `frontend/services/stagingApi.ts`:

```typescript
/**
 * Stream single variation regeneration
 */
export function streamVariationRegeneration(
  projectId: string,
  roomId: string,
  variationId: string,
  strategy: 'retry' | 'fresh',
  onEvent: StagingStreamEventCallback,
): () => void {
  const url = `${API_BASE_URL}/staging/projects/${projectId}/rooms/${roomId}/variations/${variationId}/regenerate?strategy=${strategy}`;
  
  if (API_DEBUG) {
    console.log(`Starting SSE stream for variation regeneration (${strategy})`);
    console.log(`POST ${url}`);
  }

  const abortController = new AbortController();
  let receivedTerminalEvent = false;

  fetch(url, {
    method: 'POST',
    signal: abortController.signal,
  })
    .then(async (response) => {
      if (!response.ok) {
        const errorText = await response.text();
        onEvent({ type: 'error', error: `HTTP ${response.status}: ${errorText}` });
        return;
      }

      const reader = response.body?.getReader();
      if (!reader) {
        onEvent({ type: 'error', error: 'No response body' });
        return;
      }

      const decoder = new TextDecoder();
      let buffer = '';
      let currentEventType: string | null = null;
      let currentData: string | null = null;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('event: ')) {
            currentEventType = line.slice(7).trim();
          } else if (line.startsWith('data: ')) {
            currentData = line.slice(6);
          } else if (line === '' && currentEventType && currentData) {
            try {
              const parsedData = JSON.parse(currentData);
              const event: StagingStreamEvent = {
                type: currentEventType as StagingStreamEventType,
                ...parsedData,
              };

              if (currentEventType === 'project_completed' || currentEventType === 'error') {
                receivedTerminalEvent = true;
              }

              if (API_DEBUG) {
                console.log('SSE event:', event);
              }

              onEvent(event);
            } catch (parseError) {
              console.error('Failed to parse SSE data:', currentData, parseError);
            }

            currentEventType = null;
            currentData = null;
          }
        }
      }

      if (!receivedTerminalEvent) {
        onEvent({ type: 'stream_ended' });
      }
    })
    .catch((error) => {
      if (error.name === 'AbortError') return;
      console.error('SSE stream error:', error);
      onEvent({ type: 'error', error: error.message || 'Stream error' });
    });

  return () => {
    abortController.abort();
  };
}
```

- [ ] **Step 2: Verify build passes**

Run: `cd /Users/bradcstevens/code/github/bradcstevens/visionary-lab/frontend && npm run build`
Expected: Build succeeds

- [ ] **Step 3: Commit**

```bash
cd /Users/bradcstevens/code/github/bradcstevens/visionary-lab
git add frontend/services/stagingApi.ts
git commit -m "feat(frontend): add streamVariationRegeneration API function

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 4: Frontend — VariationThumbnail hover overlay with regenerate

**Files:**
- Modify: `frontend/components/staging/VariationThumbnail.tsx`

- [ ] **Step 1: Add new props and hover overlay to VariationThumbnail**

Update `frontend/components/staging/VariationThumbnail.tsx`:

1. Add imports at the top (merge with existing):
```typescript
import { AlertCircle, RefreshCw, Loader2, RotateCcw, Sparkles } from "lucide-react";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
```

2. Update the interface to add new props:
```typescript
interface VariationThumbnailProps {
  imageUrl?: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  error?: string;
  index: number;
  onClick?: () => void;
  onRetry?: () => void;
  onRegenerate?: (strategy: 'retry' | 'fresh') => void;
  isRegenerating?: boolean;
}
```

3. Update the destructuring to include new props:
```typescript
export function VariationThumbnail({ 
  imageUrl, 
  status, 
  error, 
  index, 
  onClick, 
  onRetry,
  onRegenerate,
  isRegenerating,
}: VariationThumbnailProps) {
```

4. Replace the `completed` case in `renderContent()` with:
```typescript
      case 'completed':
        if (isRegenerating) {
          return (
            <div className="w-full h-full bg-muted rounded-lg flex items-center justify-center">
              <div className="flex flex-col items-center gap-2">
                <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
                <Badge variant="secondary" className="text-xs">
                  {index + 1}
                </Badge>
              </div>
            </div>
          );
        }
        return (
          <div className="relative w-full h-full group cursor-pointer" onClick={onClick}>
            <StorageImage
              src={imageUrl}
              alt={`Variation ${index + 1}`}
              className="w-full h-full object-cover rounded-lg"
              fallbackClassName="w-full h-full rounded-lg"
              fallbackText="Preview unavailable"
              overlay={
                <>
                  <Badge 
                    variant="secondary" 
                    className="absolute top-2 right-2 bg-black/70 text-white text-xs"
                  >
                    {index + 1}
                  </Badge>
                  {onRegenerate && (
                    <div className="absolute inset-0 bg-black/0 group-hover:bg-black/40 transition-colors duration-200 rounded-lg flex items-center justify-center">
                      <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                          <Button
                            size="sm"
                            variant="secondary"
                            className="opacity-0 group-hover:opacity-100 transition-opacity duration-200 h-8 w-8 p-0 rounded-full bg-white/90 hover:bg-white text-gray-700 shadow-md"
                            onClick={(e) => e.stopPropagation()}
                          >
                            <RefreshCw className="h-4 w-4" />
                          </Button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent align="center" side="top" className="w-48">
                          <DropdownMenuItem onClick={(e) => { e.stopPropagation(); onRegenerate('retry'); }}>
                            <RotateCcw className="h-4 w-4 mr-2" />
                            Retry Same Prompt
                          </DropdownMenuItem>
                          <DropdownMenuItem onClick={(e) => { e.stopPropagation(); onRegenerate('fresh'); }}>
                            <Sparkles className="h-4 w-4 mr-2" />
                            Try Something New
                          </DropdownMenuItem>
                        </DropdownMenuContent>
                      </DropdownMenu>
                    </div>
                  )}
                </>
              }
            />
          </div>
        );
```

- [ ] **Step 2: Verify build passes**

Run: `cd /Users/bradcstevens/code/github/bradcstevens/visionary-lab/frontend && npm run build`
Expected: Build succeeds (new props are optional so existing callers don't break)

- [ ] **Step 3: Commit**

```bash
cd /Users/bradcstevens/code/github/bradcstevens/visionary-lab
git add frontend/components/staging/VariationThumbnail.tsx
git commit -m "feat(frontend): add regenerate overlay to VariationThumbnail

Hover shows refresh button with dropdown for retry/fresh strategies.
isRegenerating prop shows spinner in place of image during regeneration.

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 5: Frontend — ImageLightbox regenerate button

**Files:**
- Modify: `frontend/components/staging/ImageLightbox.tsx`

- [ ] **Step 1: Update ImageLightbox with regenerate button**

Update `frontend/components/staging/ImageLightbox.tsx`:

1. Add imports:
```typescript
import { X, ExternalLink, RefreshCw, RotateCcw, Sparkles, Loader2 } from "lucide-react";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
```

2. Update the props interface:
```typescript
interface ImageLightboxProps {
  image: LightboxImage | null;
  onClose: () => void;
  onRegenerate?: (strategy: 'retry' | 'fresh') => void;
  isRegenerating?: boolean;
}
```

3. Update the function signature:
```typescript
export function ImageLightbox({ image, onClose, onRegenerate, isRegenerating }: ImageLightboxProps) {
```

4. In the top bar `<div className="flex items-center gap-2">` section, add the regenerate dropdown before the external link button:
```typescript
              {onRegenerate && (
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button
                      size="sm"
                      variant="ghost"
                      className="text-white/70 hover:text-white hover:bg-white/10 h-8 px-2"
                      disabled={isRegenerating}
                      aria-label="Regenerate this variation"
                    >
                      {isRegenerating ? (
                        <Loader2 className="h-4 w-4 animate-spin" />
                      ) : (
                        <RefreshCw className="h-4 w-4" />
                      )}
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end" className="w-48">
                    <DropdownMenuItem onClick={() => onRegenerate('retry')}>
                      <RotateCcw className="h-4 w-4 mr-2" />
                      Retry Same Prompt
                    </DropdownMenuItem>
                    <DropdownMenuItem onClick={() => onRegenerate('fresh')}>
                      <Sparkles className="h-4 w-4 mr-2" />
                      Try Something New
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              )}
```

5. Wrap the image container with a loading overlay when regenerating. Replace the image container `{image && (` block:
```typescript
          {image && (
            <div className="relative flex items-center justify-center w-full max-w-[90vw] max-h-[80vh]">
              <StorageImage
                src={image.url}
                alt={`${image.roomLabel} variation ${image.variationIndex + 1}`}
                className={cn(
                  "max-w-full max-h-[80vh] w-auto h-auto object-contain rounded-lg",
                  isRegenerating && "opacity-40"
                )}
                fallbackClassName="w-64 h-64 rounded-lg bg-muted"
                fallbackText="Image could not be loaded"
              />
              {isRegenerating && (
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="flex flex-col items-center gap-3">
                    <Loader2 className="h-8 w-8 animate-spin text-white" />
                    <span className="text-sm text-white/80 font-medium">Regenerating...</span>
                  </div>
                </div>
              )}
            </div>
          )}
```

- [ ] **Step 2: Verify build passes**

Run: `cd /Users/bradcstevens/code/github/bradcstevens/visionary-lab/frontend && npm run build`
Expected: Build succeeds

- [ ] **Step 3: Commit**

```bash
cd /Users/bradcstevens/code/github/bradcstevens/visionary-lab
git add frontend/components/staging/ImageLightbox.tsx
git commit -m "feat(frontend): add regenerate button to ImageLightbox

Dropdown with retry/fresh options. Loading overlay during regeneration.

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 6: Frontend — RoomGroup prop pass-through

**Files:**
- Modify: `frontend/components/staging/RoomGroup.tsx`

- [ ] **Step 1: Update RoomGroup to pass regeneration props to VariationThumbnail**

Update `frontend/components/staging/RoomGroup.tsx`:

1. Update the interface:
```typescript
interface RoomGroupProps {
  room: Room;
  onVariationClick?: (room: Room, variationIndex: number) => void;
  onRetryVariation?: (room: Room, variationIndex: number) => void;
  onRegenerateRoom?: (room: Room) => void;
  onRegenerateVariation?: (room: Room, variationIndex: number, strategy: 'retry' | 'fresh') => void;
  regeneratingVariationId?: string | null;
  isGenerating?: boolean;
}
```

2. Update the destructuring:
```typescript
export function RoomGroup({ room, onVariationClick, onRetryVariation, onRegenerateRoom, onRegenerateVariation, regeneratingVariationId, isGenerating }: RoomGroupProps) {
```

3. Update the `VariationThumbnail` rendering in the room grid (the `{room.variations.map(...)}` block) to pass the new props:
```typescript
        {room.variations.map((variation, index) => (
          <VariationThumbnail
            key={variation.id}
            imageUrl={variation.image_url}
            status={variation.status}
            error={variation.error}
            index={index}
            onClick={
              variation.status === 'completed' && onVariationClick
                ? () => onVariationClick(room, index)
                : undefined
            }
            onRetry={
              variation.status === 'failed' && onRetryVariation
                ? () => onRetryVariation(room, index)
                : undefined
            }
            onRegenerate={
              variation.status === 'completed' && onRegenerateVariation && !isGenerating
                ? (strategy) => onRegenerateVariation(room, index, strategy)
                : undefined
            }
            isRegenerating={regeneratingVariationId === variation.id}
          />
        ))}
```

- [ ] **Step 2: Verify build passes**

Run: `cd /Users/bradcstevens/code/github/bradcstevens/visionary-lab/frontend && npm run build`
Expected: Build succeeds

- [ ] **Step 3: Commit**

```bash
cd /Users/bradcstevens/code/github/bradcstevens/visionary-lab
git add frontend/components/staging/RoomGroup.tsx
git commit -m "feat(frontend): pass regeneration props through RoomGroup

Wires onRegenerateVariation and regeneratingVariationId to VariationThumbnail.

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 7: Frontend — Project detail page state & handlers (wire everything together)

**Files:**
- Modify: `frontend/app/projects/[id]/page.tsx`

- [ ] **Step 1: Add imports, state, and handler**

1. Add `streamVariationRegeneration` to the import from `@/services/stagingApi` (line 20):
```typescript
import { getProject, deleteProject, resetProject, streamGeneration, streamRoomRegeneration, streamVariationRegeneration, StagingProject, Room, StagingStreamEvent } from "@/services/stagingApi";
```

2. Add new state after `const [lightboxImage, setLightboxImage] = useState<LightboxImage | null>(null);` (line 38):
```typescript
  const [regeneratingVariationId, setRegeneratingVariationId] = useState<string | null>(null);
```

3. Add the `LightboxImage` interface extension. Currently `LightboxImage` is defined in `ImageLightbox.tsx` without room/variation IDs. We need the room and variation references for the lightbox regeneration. Update the `handleVariationClick` handler AND the lightbox rendering to pass room/variation identity.

Update `handleVariationClick` to also store room and variation references:
```typescript
  const [lightboxContext, setLightboxContext] = useState<{ room: Room; variationIndex: number } | null>(null);

  const handleVariationClick = (room: Room, variationIndex: number) => {
    const variation = room.variations[variationIndex];
    if (variation.status === 'completed' && variation.image_url) {
      setLightboxImage({
        url: variation.image_url,
        roomLabel: room.label,
        variationIndex,
      });
      setLightboxContext({ room, variationIndex });
    }
  };
```

4. Add the single-variation regeneration handler after `handleRetryVariation` (around line 256):
```typescript
  const handleRegenerateVariation = useCallback((room: Room, variationIndex: number, strategy: 'retry' | 'fresh') => {
    if (isGenerating || regeneratingVariationId) return;
    const variation = room.variations[variationIndex];
    setRegeneratingVariationId(variation.id);

    const cleanup = streamVariationRegeneration(
      projectId,
      room.id,
      variation.id,
      strategy,
      (event) => {
        switch (event.type) {
          case 'variation_completed':
            activityLog.log({
              level: 'success',
              icon: '✓',
              message: `Variation ${variationIndex + 1} regenerated`,
              detail: [
                (event as any).model,
                (event as any).tokens_used ? `${Number((event as any).tokens_used).toLocaleString()} tokens` : null,
                (event as any).elapsed_ms ? `${((event as any).elapsed_ms / 1000).toFixed(1)}s` : null,
              ].filter(Boolean).join(' · ') || undefined,
            });
            break;
          case 'variation_failed':
            activityLog.log({
              level: 'error',
              icon: '✕',
              message: `Variation ${variationIndex + 1} regeneration failed`,
              detail: (event as any).error || 'Unknown error',
            });
            toast.error(`Regeneration failed: ${(event as any).error || 'Unknown error'}`);
            break;
          case 'project_completed':
          case 'stream_ended':
            setRegeneratingVariationId(null);
            loadProject();
            if (event.type === 'project_completed') {
              toast.success('Variation regenerated!');
            }
            break;
          case 'error':
            setRegeneratingVariationId(null);
            toast.error(event.error || 'Regeneration failed');
            loadProject();
            break;
        }
      },
    );

    // Store cleanup for abort on unmount (reuse existing ref pattern)
    const previousCleanup = streamCleanupRef.current;
    streamCleanupRef.current = () => {
      cleanup();
      previousCleanup?.();
    };
  }, [isGenerating, regeneratingVariationId, projectId, activityLog, loadProject]);

  const handleLightboxRegenerate = useCallback((strategy: 'retry' | 'fresh') => {
    if (!lightboxContext) return;
    handleRegenerateVariation(lightboxContext.room, lightboxContext.variationIndex, strategy);
  }, [lightboxContext, handleRegenerateVariation]);
```

5. Update the `RoomGroup` rendering to pass new props (around line 498-507):
```typescript
            <RoomGroup
              key={room.id}
              room={room}
              onVariationClick={handleVariationClick}
              onRetryVariation={handleRetryVariation}
              onRegenerateRoom={handleRegenerateRoom}
              onRegenerateVariation={handleRegenerateVariation}
              regeneratingVariationId={regeneratingVariationId}
              isGenerating={isGenerating}
            />
```

6. Update the `ImageLightbox` rendering (around line 538) to pass regeneration props:
```typescript
      <ImageLightbox
        image={lightboxImage}
        onClose={() => { setLightboxImage(null); setLightboxContext(null); }}
        onRegenerate={lightboxContext ? handleLightboxRegenerate : undefined}
        isRegenerating={
          lightboxContext
            ? regeneratingVariationId === lightboxContext.room.variations[lightboxContext.variationIndex]?.id
            : false
        }
      />
```

- [ ] **Step 2: Update the lightbox to refresh its image when regeneration completes**

The lightbox image URL needs to update when the project reloads after regeneration. The current `lightboxImage` state holds the URL, but after `loadProject()` re-fetches, the URL changes. Add an effect to sync:

After the existing `useEffect` blocks, add:
```typescript
  // Sync lightbox image URL with project data after reload
  useEffect(() => {
    if (lightboxContext && project) {
      const room = project.rooms.find(r => r.id === lightboxContext.room.id);
      if (room) {
        const variation = room.variations[lightboxContext.variationIndex];
        if (variation?.status === 'completed' && variation.image_url) {
          setLightboxImage(prev => prev ? {
            ...prev,
            url: variation.image_url!,
          } : null);
          // Keep lightboxContext room reference fresh
          setLightboxContext(prev => prev ? { ...prev, room } : null);
        }
      }
    }
  }, [project]);
```

- [ ] **Step 3: Verify build passes**

Run: `cd /Users/bradcstevens/code/github/bradcstevens/visionary-lab/frontend && npm run build`
Expected: Build succeeds

- [ ] **Step 4: Run lint**

Run: `cd /Users/bradcstevens/code/github/bradcstevens/visionary-lab/frontend && npx next lint`
Expected: No errors

- [ ] **Step 5: Commit**

```bash
cd /Users/bradcstevens/code/github/bradcstevens/visionary-lab
git add frontend/app/projects/[id]/page.tsx
git commit -m "feat(frontend): wire single variation regeneration into project detail page

Adds regeneratingVariationId state, handleRegenerateVariation handler,
lightbox regeneration support, and auto-sync of lightbox image after reload.

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 8: Full integration verification

- [ ] **Step 1: Run full backend test suite**

Run: `cd /Users/bradcstevens/code/github/bradcstevens/visionary-lab && uv run pytest tests/ --ignore=tests/integration -v`
Expected: All tests PASS

- [ ] **Step 2: Run frontend build**

Run: `cd /Users/bradcstevens/code/github/bradcstevens/visionary-lab/frontend && npm run build`
Expected: Build succeeds with no errors

- [ ] **Step 3: Run frontend lint**

Run: `cd /Users/bradcstevens/code/github/bradcstevens/visionary-lab/frontend && npx next lint`
Expected: No lint errors
