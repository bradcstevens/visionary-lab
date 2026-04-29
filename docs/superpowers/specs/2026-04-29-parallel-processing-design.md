# Parallel Image/Video Processing — Design Spec

## Problem

The staging pipeline processes rooms sequentially (`for room in pending_rooms`), and variations within each room sequentially (`for idx, adapted_prompt in ...`). A project with 5 rooms × 5 variations makes 25 sequential API calls, each taking 10–30 seconds — up to 12+ minutes total when most of that time is waiting.

## Approach: Room-Level Parallelism (Approach A)

Process multiple rooms concurrently using `asyncio.gather` + an asyncio semaphore. Keep variations sequential within each room. This saturates Azure AI Foundry's default 9 RPM rate limit with 3 concurrent rooms and avoids over-complexity.

### Why not also parallelize variations?

At 3 concurrent rooms × 1 variation each = 3 concurrent API calls. Each takes ~20 seconds, yielding ~9 completions/min — right at the default 9 RPM limit. Adding variation-level parallelism risks 429s with marginal speedup.

## Rate Limits (from Azure docs)

| Model | Default Quota |
|-------|--------------|
| GPT-image-1 series | 9 RPM |
| GPT-image-1-mini | 12 RPM |
| GPT-image-2 | 9 RPM |
| Sora 2 | 2 jobs/min |

## Changes

### 1. Backend: `staging_pipeline.py` — Parallel Room Processing

**Current** (line 302):
```python
for room in pending_rooms:
    async for event in self.process_room(project, room, brief_prompts=brief_prompts):
        yield event
```

**New**: Launch room tasks into an `asyncio.Queue`-based event bus. Each room runs as an independent `asyncio.Task` guarded by the existing `self.semaphore`. Events are yielded to the SSE stream as they arrive (interleaved).

The frontend already debounces SSE events via `debouncedReload()` and handles any event type/order — no frontend SSE changes needed.

### 2. Backend: `image_pipeline.py` — 429 Retry with Backoff

Wrap `asyncio.to_thread(client.generate_image, ...)` and `asyncio.to_thread(client.edit_image, ...)` with retry logic that catches HTTP 429 (rate limit) errors and retries with exponential backoff, respecting `Retry-After` headers.

### 3. Backend: `config.py` — New Settings

```python
IMAGE_GEN_RETRY_ATTEMPTS: int = 3          # max retries on 429
IMAGE_GEN_RETRY_BASE_DELAY: float = 2.0    # seconds, doubles each retry
```

### 4. Backend: New `/api/v1/images/batch` Endpoint

Accept an array of image generation/edit requests, process them concurrently with a configurable semaphore, and return all results. Uses a new `IMAGE_BATCH_MAX_CONCURRENT: int = 3` setting.

### 5. Backend: Batch Request/Response Models

```python
class ImageBatchRequest(BaseModel):
    requests: List[ImagePipelineRequest]

class ImageBatchResponse(BaseModel):
    results: List[ImagePipelineResponse]
    total: int
    succeeded: int
    failed: int
```

### 6. Frontend: Batch API Client (optional)

Add `batchProcess()` to `stagingApi.ts` for the new batch endpoint. The staging SSE flow needs no frontend changes.

## What Does NOT Change

- `process_room()` internal logic stays the same (variations still sequential within a room)
- SSE event schema (types, fields) stays the same
- Frontend `handleStreamEvent` / `debouncedReload` — already handles interleaved events
- Cosmos DB update pattern — each room still updates independently
- Video (Sora) processing — not part of staging pipeline

## Testing Strategy

- Unit tests for parallel room orchestration (mock rooms, verify concurrent execution + event merging)
- Unit tests for 429 retry logic (mock HTTP 429 → verify retry + backoff)
- Unit tests for batch endpoint (multiple requests → concurrent execution)
- Existing tests must continue to pass unchanged
