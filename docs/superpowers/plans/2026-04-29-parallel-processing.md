# Parallel Image/Video Processing — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Process multiple rooms and image batch requests concurrently instead of sequentially, respecting Azure AI Foundry rate limits.

**Architecture:** Room-level parallelism via asyncio.Queue event bus + semaphore. 429 retry with exponential backoff on image generation API calls. New `/images/batch` endpoint for non-staging parallel processing.

**Tech Stack:** Python asyncio, FastAPI, pytest + pytest-asyncio

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `backend/core/config.py` | Modify | Add retry + batch concurrency settings |
| `backend/core/image_pipeline.py` | Modify | Add 429 retry wrapper around API calls |
| `backend/core/staging_pipeline.py` | Modify | Parallel room orchestration via event queue |
| `backend/models/images.py` | Modify | Add `ImageBatchRequest` / `ImageBatchResponse` models |
| `backend/api/endpoints/images.py` | Modify | Add `/batch` endpoint |
| `tests/test_retry_logic.py` | Create | Tests for 429 retry with backoff |
| `tests/test_parallel_rooms.py` | Create | Tests for parallel room orchestration |
| `tests/test_batch_endpoint.py` | Create | Tests for batch image API |

---

### Task 1: Add Configuration Settings

**Files:**
- Modify: `backend/core/config.py:54-58`

- [ ] **Step 1: Write failing test**

```python
# tests/test_config_settings.py
from backend.core.config import Settings

def test_retry_settings_have_defaults():
    s = Settings()
    assert s.IMAGE_GEN_RETRY_ATTEMPTS == 3
    assert s.IMAGE_GEN_RETRY_BASE_DELAY == 2.0

def test_batch_concurrency_setting_has_default():
    s = Settings()
    assert s.IMAGE_BATCH_MAX_CONCURRENT == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config_settings.py -v`
Expected: FAIL with `AttributeError: 'Settings' object has no attribute 'IMAGE_GEN_RETRY_ATTEMPTS'`

- [ ] **Step 3: Write minimal implementation**

Add these fields to the `Settings` class in `backend/core/config.py`, after line 58 (after `STAGING_STALE_PROCESSING_MINUTES`):

```python
    # Rate-limit retry for image generation API calls (429 handling)
    IMAGE_GEN_RETRY_ATTEMPTS: int = 3
    IMAGE_GEN_RETRY_BASE_DELAY: float = 2.0  # seconds; doubles each retry

    # Batch image API concurrency
    IMAGE_BATCH_MAX_CONCURRENT: int = 3
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_config_settings.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/core/config.py tests/test_config_settings.py
git commit -m "feat: add retry and batch concurrency config settings

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 2: Add 429 Retry Logic to Image Pipeline

**Files:**
- Create: `tests/test_retry_logic.py`
- Modify: `backend/core/image_pipeline.py:49-96`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_retry_logic.py
"""Tests for image generation 429 retry with exponential backoff."""
import asyncio
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from openai import RateLimitError
from httpx import Response, Request

from backend.core.image_pipeline import ImagePipelineService
from backend.models.images import ImageGenerationRequest


def _make_rate_limit_error(retry_after: str = "2"):
    """Build a realistic OpenAI RateLimitError."""
    mock_response = Response(
        status_code=429,
        headers={"retry-after": retry_after},
        request=Request("POST", "https://example.com"),
    )
    return RateLimitError(
        message="Rate limit exceeded",
        response=mock_response,
        body={"error": {"message": "Rate limit exceeded"}},
    )


@pytest.mark.asyncio
async def test_retry_on_429_succeeds_after_retries():
    """Should retry on 429 and eventually succeed."""
    service = ImagePipelineService()
    request = ImageGenerationRequest(prompt="a cat", model="gpt-image-2")

    call_count = 0
    good_response = {"created": 1, "data": [{"b64_json": "AAAA"}],
                     "usage": {"total_tokens": 100}}

    def mock_generate_image(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise _make_rate_limit_error("0.01")
        return MagicMock(created=1, data=[MagicMock(b64_json="AAAA")])

    with patch("backend.core.image_pipeline.asyncio.to_thread") as mock_to_thread:
        mock_to_thread.side_effect = mock_generate_image
        # Patch sleep to avoid actual delays
        with patch("backend.core.image_pipeline.asyncio.sleep", new_callable=AsyncMock):
            result = await service.generate(request)

    assert call_count == 3
    assert result.success


@pytest.mark.asyncio
async def test_retry_exhausted_raises():
    """Should raise after exhausting all retry attempts."""
    service = ImagePipelineService()
    request = ImageGenerationRequest(prompt="a cat", model="gpt-image-2")

    def always_429(**kwargs):
        raise _make_rate_limit_error("0.01")

    with patch("backend.core.image_pipeline.asyncio.to_thread", side_effect=always_429):
        with patch("backend.core.image_pipeline.asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(RateLimitError):
                await service.generate(request)


@pytest.mark.asyncio
async def test_non_429_errors_not_retried():
    """Non-rate-limit errors should propagate immediately."""
    service = ImagePipelineService()
    request = ImageGenerationRequest(prompt="a cat", model="gpt-image-2")

    call_count = 0

    def fail_with_value_error(**kwargs):
        nonlocal call_count
        call_count += 1
        raise ValueError("bad input")

    with patch("backend.core.image_pipeline.asyncio.to_thread", side_effect=fail_with_value_error):
        with pytest.raises(Exception):  # Wrapped by HTTPException in generate()
            await service.generate(request)

    assert call_count == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_retry_logic.py -v`
Expected: FAIL — `_generate_with_retry` does not exist

- [ ] **Step 3: Write minimal implementation**

In `backend/core/image_pipeline.py`, add this method to `ImagePipelineService` and update `generate()` and `edit()` to use it:

```python
import time as _time
from openai import RateLimitError

async def _call_with_retry(self, coro_fn, *args, **kwargs):
    """Wrap an async callable with 429 retry + exponential backoff."""
    max_attempts = settings.IMAGE_GEN_RETRY_ATTEMPTS
    base_delay = settings.IMAGE_GEN_RETRY_BASE_DELAY

    for attempt in range(max_attempts):
        try:
            return await coro_fn(*args, **kwargs)
        except RateLimitError as exc:
            if attempt >= max_attempts - 1:
                raise
            retry_after = None
            if hasattr(exc, 'response') and exc.response is not None:
                retry_after_str = exc.response.headers.get("retry-after")
                if retry_after_str:
                    try:
                        retry_after = float(retry_after_str)
                    except (ValueError, TypeError):
                        pass
            delay = retry_after if retry_after else base_delay * (2 ** attempt)
            logger.warning(
                "Rate limited (429), attempt %d/%d. Retrying in %.1fs",
                attempt + 1, max_attempts, delay,
            )
            await asyncio.sleep(delay)
    raise RuntimeError("Unreachable")  # pragma: no cover
```

Then update `generate()` line 85 to use the retry wrapper:

```python
# Before:
response = await asyncio.to_thread(client.generate_image, **params)

# After:
response = await self._call_with_retry(
    asyncio.to_thread, client.generate_image, **params
)
```

Apply the same change in `edit()` for the `edit_image` call.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_retry_logic.py -v`
Expected: PASS

- [ ] **Step 5: Run all existing tests to verify no regressions**

Run: `uv run pytest tests/ --ignore=tests/integration -v`
Expected: All existing tests PASS

- [ ] **Step 6: Commit**

```bash
git add backend/core/image_pipeline.py tests/test_retry_logic.py
git commit -m "feat: add 429 retry with exponential backoff for image generation

Wraps generate/edit API calls with retry logic that respects
Retry-After headers from Azure AI Foundry rate limit responses.

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 3: Parallel Room Orchestration in Staging Pipeline

**Files:**
- Create: `tests/test_parallel_rooms.py`
- Modify: `backend/core/staging_pipeline.py:277-309`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_parallel_rooms.py
"""Tests for parallel room processing in the staging pipeline."""
import asyncio
import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.models.images import (
    ImageGenerationResponse,
    ImageSaveResponse,
    ImagePipelineResponse,
    PipelineStepResult,
)
from backend.models.staging import (
    ItemStatus,
    ProjectStatus,
    Room,
    StagingProject,
    StagingSettings,
    Variation,
)


def _make_project(n_rooms=3, n_variations=2) -> StagingProject:
    rooms = []
    for i in range(n_rooms):
        variations = [Variation(id=f"var-{i}-{j}") for j in range(n_variations)]
        rooms.append(
            Room(
                id=f"room-{i}",
                label=f"Room {i+1}",
                original_image_url="https://acct.blob.core.windows.net/images/staging/proj/originals/photo.png",
                variations=variations,
            )
        )
    return StagingProject(
        id="proj-parallel",
        name="Parallel Test",
        prompt="Modern minimalist",
        settings=StagingSettings(variations_per_room=n_variations),
        rooms=rooms,
    )


def _make_pipeline_response(room_id="room-0"):
    url = f"https://acct.blob.core.windows.net/images/staging/proj/variations/{room_id}/img.png"
    gen = ImageGenerationResponse(
        success=True, message="ok",
        imgen_model_response={"data": [{"b64_json": "AAAA"}], "usage": {"total_tokens": 100}},
    )
    save = ImageSaveResponse(
        success=True, message="Saved",
        saved_images=[{"url": url, "blob_name": f"staging/proj/variations/{room_id}/img.png"}],
        total_saved=1,
    )
    return ImagePipelineResponse(
        success=True, message="Pipeline completed",
        steps=[PipelineStepResult(step="edit", success=True), PipelineStepResult(step="save", success=True)],
        generation=gen, save=save,
    )


def _build_staging_pipeline():
    """Construct a StagingPipeline with all deps mocked."""
    from backend.core.staging_pipeline import StagingPipeline

    mock_pipeline = AsyncMock()
    mock_pipeline.process_pipeline.side_effect = lambda **kw: _make_pipeline_response()

    mock_blob = MagicMock()
    mock_blob.get_asset_content.return_value = (b"\x89PNG\r\n", "image/png")

    mock_storage = MagicMock()
    mock_storage.update_project = MagicMock()

    mock_analyzer = AsyncMock()
    mock_analyzer.async_image_chat.return_value = {"description": "A room", "features": []}

    mock_llm = AsyncMock()
    mock_llm.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content='["Add plants", "Add art"]'))]
    )

    return StagingPipeline(
        async_llm_client=mock_llm,
        llm_deployment="gpt-4o",
        image_analyzer=mock_analyzer,
        image_pipeline=mock_pipeline,
        storage_service=mock_storage,
        blob_service=mock_blob,
    )


class TestParallelRoomProcessing:

    @pytest.mark.asyncio
    async def test_all_rooms_processed(self):
        """All rooms should be processed and emit events."""
        staging = _build_staging_pipeline()
        project = _make_project(n_rooms=3, n_variations=1)

        events = []
        async for event in staging.generate_project(project):
            events.append(event)

        room_completed = [e for e in events if e["type"] == "room_completed"]
        assert len(room_completed) == 3
        assert events[-1]["type"] == "project_completed"

    @pytest.mark.asyncio
    async def test_rooms_run_concurrently(self):
        """Rooms should overlap in time, not run sequentially."""
        staging = _build_staging_pipeline()
        project = _make_project(n_rooms=3, n_variations=1)

        # Add a small delay to each room to make timing measurable
        original_process_room = staging.process_room

        async def slow_process_room(proj, room, brief_prompts=None):
            await asyncio.sleep(0.1)  # 100ms per room
            async for event in original_process_room(proj, room, brief_prompts=brief_prompts):
                yield event

        staging.process_room = slow_process_room

        start = time.monotonic()
        events = []
        async for event in staging.generate_project(project):
            events.append(event)
        elapsed = time.monotonic() - start

        # If sequential: 3 × 0.1s = 0.3s minimum
        # If parallel: ~0.1s (all 3 rooms overlap)
        # Allow generous margin but it must be faster than sequential
        assert elapsed < 0.25, f"Rooms appear sequential: took {elapsed:.2f}s (expected < 0.25s)"

    @pytest.mark.asyncio
    async def test_semaphore_limits_concurrency(self):
        """No more than STAGING_CONCURRENT_ROOMS rooms should run at once."""
        staging = _build_staging_pipeline()
        project = _make_project(n_rooms=5, n_variations=1)

        peak_concurrent = 0
        current_concurrent = 0
        lock = asyncio.Lock()

        # Patch process_pipeline (the actual work inside the semaphore) to
        # measure gated concurrency, not wrapper concurrency.
        original_process_pipeline = staging.image_pipeline.process_pipeline

        async def counting_process_pipeline(**kwargs):
            nonlocal peak_concurrent, current_concurrent
            async with lock:
                current_concurrent += 1
                peak_concurrent = max(peak_concurrent, current_concurrent)
            try:
                await asyncio.sleep(0.05)
                return await original_process_pipeline(**kwargs)
            finally:
                async with lock:
                    current_concurrent -= 1

        staging.image_pipeline.process_pipeline = counting_process_pipeline

        events = []
        async for event in staging.generate_project(project):
            events.append(event)

        # STAGING_CONCURRENT_ROOMS defaults to 3
        assert peak_concurrent <= 3, f"Peak concurrency {peak_concurrent} exceeded limit of 3"

    @pytest.mark.asyncio
    async def test_one_room_failure_doesnt_stop_others(self):
        """If one room fails, remaining rooms should still complete."""
        staging = _build_staging_pipeline()
        project = _make_project(n_rooms=3, n_variations=1)

        original_process_room = staging.process_room
        call_count = 0

        async def failing_process_room(proj, room, brief_prompts=None):
            nonlocal call_count
            call_count += 1
            if room.id == "room-1":
                raise RuntimeError("Simulated failure")
            async for event in original_process_room(proj, room, brief_prompts=brief_prompts):
                yield event

        staging.process_room = failing_process_room

        events = []
        async for event in staging.generate_project(project):
            events.append(event)

        # room-0 and room-2 should complete, room-1 should fail
        completed = [e for e in events if e["type"] == "room_completed"]
        failed = [e for e in events if e["type"] == "room_failed"]
        assert len(completed) == 2
        assert len(failed) == 1
        assert failed[0]["room_id"] == "room-1"

    @pytest.mark.asyncio
    async def test_project_status_set_correctly(self):
        """Project should be COMPLETED if any room completed."""
        staging = _build_staging_pipeline()
        project = _make_project(n_rooms=2, n_variations=1)

        events = []
        async for event in staging.generate_project(project):
            events.append(event)

        final = events[-1]
        assert final["type"] == "project_completed"
        assert project.status == ProjectStatus.COMPLETED
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_parallel_rooms.py -v`
Expected: `test_rooms_run_concurrently` FAILS (rooms still sequential, takes ≥0.3s)

- [ ] **Step 3: Write minimal implementation**

Replace `generate_project()` in `backend/core/staging_pipeline.py` (lines 277–309):

```python
async def generate_project(self, project: StagingProject) -> AsyncGenerator[Dict[str, Any], None]:
    """Process all pending rooms in parallel. Yields SSE events as they arrive."""
    project.status = ProjectStatus.PROCESSING
    self.storage_service.update_project(project.id, self._serialize_project(project))

    pending_rooms = [r for r in project.rooms if r.status in (ItemStatus.PENDING, ItemStatus.FAILED)]

    # If project has a design_brief, use BriefGeneratorService for prompt adaptation
    brief_prompts = {}
    if project.design_brief:
        from backend.core.brief_generator import BriefGeneratorService
        from backend.models.design_brief import DesignBrief as DBModel, ImageAnalysis

        brief = DBModel(**project.design_brief)
        analyses = [ImageAnalysis(**a) for a in (project.analyses or [])]
        brief_service = BriefGeneratorService(
            async_llm_client=self.async_llm_client,
            llm_deployment=self.llm_deployment,
        )
        brief_prompts = await brief_service.brief_to_prompts(
            brief=brief,
            image_analyses=analyses,
            n_variations=project.settings.variations_per_room,
        )

    if not pending_rooms:
        project.status = ProjectStatus.COMPLETED
        self.storage_service.update_project(project.id, self._serialize_project(project))
        yield {"type": "project_completed", "status": project.status}
        return

    # Event queue: rooms push events here, generator yields them to SSE.
    # _WORKER_DONE is a sentinel pushed in `finally` so we count worker
    # completion instead of semantic events — robust against cancellation.
    _WORKER_DONE = object()
    event_queue: asyncio.Queue = asyncio.Queue()

    async def _room_worker(room: Room) -> None:
        """Process one room and push all its events into the shared queue."""
        try:
            async for event in self.process_room(project, room, brief_prompts=brief_prompts):
                await event_queue.put(event)
        except BaseException as exc:
            # BaseException catches CancelledError too, preventing silent hangs
            if not isinstance(exc, asyncio.CancelledError):
                logger.error("Room %s failed: %s", room.id, exc)
            room.status = ItemStatus.FAILED
            room.error = str(exc)
            self._update_room_in_project(project, room)
            await event_queue.put({"type": "room_failed", "room_id": room.id, "error": str(exc)})
        finally:
            await event_queue.put(_WORKER_DONE)

    # Launch all rooms — the semaphore inside process_room gates concurrency
    tasks = [asyncio.create_task(_room_worker(room)) for room in pending_rooms]

    # Yield events as they arrive until every worker signals done
    workers_done = 0
    total_workers = len(pending_rooms)
    while workers_done < total_workers:
        event = await event_queue.get()
        if event is _WORKER_DONE:
            workers_done += 1
            continue
        yield event

    # Await tasks to propagate any unhandled exceptions
    await asyncio.gather(*tasks, return_exceptions=True)

    any_room_completed = any(r.status == ItemStatus.COMPLETED for r in project.rooms)
    project.status = ProjectStatus.COMPLETED if any_room_completed else ProjectStatus.FAILED
    self.storage_service.update_project(project.id, self._serialize_project(project))
    yield {"type": "project_completed", "status": project.status}
```

- [ ] **Step 4: Run new tests to verify they pass**

Run: `uv run pytest tests/test_parallel_rooms.py -v`
Expected: All PASS

- [ ] **Step 5: Run ALL existing tests to verify no regressions**

Run: `uv run pytest tests/ --ignore=tests/integration -v`
Expected: All tests PASS (including the 4 existing tests in `test_staging_pipeline.py`)

- [ ] **Step 6: Commit**

```bash
git add backend/core/staging_pipeline.py tests/test_parallel_rooms.py
git commit -m "feat: process staging rooms in parallel via asyncio event queue

Rooms now run as concurrent tasks guarded by the existing
STAGING_CONCURRENT_ROOMS semaphore. Events are yielded to the SSE
stream as they arrive (interleaved). One room failing does not
block others.

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 4: Batch Image Processing Models

**Files:**
- Modify: `backend/models/images.py`
- Create: `tests/test_batch_models.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_batch_models.py
"""Tests for batch request/response models."""
from backend.models.images import ImageBatchRequest, ImageBatchResponse, ImagePipelineRequest, ImagePipelineResponse, PipelineAction


def test_batch_request_accepts_list_of_pipeline_requests():
    req = ImageBatchRequest(requests=[
        ImagePipelineRequest(action=PipelineAction.GENERATE, prompt="a cat"),
        ImagePipelineRequest(action=PipelineAction.GENERATE, prompt="a dog"),
    ])
    assert len(req.requests) == 2


def test_batch_response_tracks_counts():
    resp = ImageBatchResponse(
        results=[],
        total=2,
        succeeded=1,
        failed=1,
    )
    assert resp.total == 2
    assert resp.succeeded == 1
    assert resp.failed == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_batch_models.py -v`
Expected: FAIL with `ImportError: cannot import name 'ImageBatchRequest'`

- [ ] **Step 3: Write minimal implementation**

Add to `backend/models/images.py` (at the end of the file):

```python
class ImageBatchRequest(BaseModel):
    """Request containing multiple image pipeline operations to process in parallel."""
    requests: List[ImagePipelineRequest] = Field(
        ...,
        description="List of pipeline requests to execute concurrently",
        min_length=1,
        max_length=20,
    )


class ImageBatchResponse(BaseModel):
    """Response for batch image processing."""
    results: List[ImagePipelineResponse] = Field(
        default_factory=list,
        description="Results for each request, in the same order as input",
    )
    total: int = Field(0, description="Total number of requests processed")
    succeeded: int = Field(0, description="Number of successful results")
    failed: int = Field(0, description="Number of failed results")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_batch_models.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/models/images.py tests/test_batch_models.py
git commit -m "feat: add ImageBatchRequest/Response models for batch processing

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 5: Batch Image API Endpoint

**Files:**
- Create: `tests/test_batch_endpoint.py`
- Modify: `backend/api/endpoints/images.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_batch_endpoint.py
"""Tests for the /api/v1/images/batch endpoint."""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient

from backend.models.images import (
    ImagePipelineRequest,
    ImagePipelineResponse,
    ImageGenerationResponse,
    ImageSaveResponse,
    PipelineAction,
    PipelineStepResult,
)


def _make_pipeline_response(prompt: str) -> ImagePipelineResponse:
    gen = ImageGenerationResponse(
        success=True, message="ok",
        imgen_model_response={"data": [{"b64_json": "AAAA"}]},
    )
    return ImagePipelineResponse(
        success=True, message=f"Generated: {prompt}",
        steps=[PipelineStepResult(step="generate", success=True)],
        generation=gen,
    )


@pytest.fixture
def client():
    from backend.main import app
    return TestClient(app)


class TestBatchEndpoint:

    def test_batch_processes_multiple_requests(self, client):
        """Batch endpoint should accept multiple requests and return all results."""
        with patch(
            "backend.api.endpoints.images.pipeline_service.process_pipeline",
            new_callable=AsyncMock,
        ) as mock_pipeline:
            mock_pipeline.side_effect = lambda pipeline_request, **kw: _make_pipeline_response(
                pipeline_request.prompt
            )

            response = client.post(
                "/api/v1/images/batch",
                json={
                    "requests": [
                        {"action": "generate", "prompt": "a cat"},
                        {"action": "generate", "prompt": "a dog"},
                    ]
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["total"] == 2
            assert data["succeeded"] == 2
            assert data["failed"] == 0
            assert len(data["results"]) == 2

    def test_batch_handles_partial_failures(self, client):
        """If some requests fail, batch should still return results for others."""
        call_count = 0

        async def flaky_pipeline(pipeline_request, **kw):
            nonlocal call_count
            call_count += 1
            if "fail" in pipeline_request.prompt:
                raise RuntimeError("Generation failed")
            return _make_pipeline_response(pipeline_request.prompt)

        with patch(
            "backend.api.endpoints.images.pipeline_service.process_pipeline",
            new_callable=AsyncMock,
            side_effect=flaky_pipeline,
        ):
            response = client.post(
                "/api/v1/images/batch",
                json={
                    "requests": [
                        {"action": "generate", "prompt": "a cat"},
                        {"action": "generate", "prompt": "fail this one"},
                        {"action": "generate", "prompt": "a dog"},
                    ]
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["total"] == 3
            assert data["succeeded"] == 2
            assert data["failed"] == 1

    def test_batch_rejects_empty_requests(self, client):
        """Batch endpoint should reject empty request lists."""
        response = client.post("/api/v1/images/batch", json={"requests": []})
        assert response.status_code == 422
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_batch_endpoint.py -v`
Expected: FAIL (404 — endpoint doesn't exist yet)

- [ ] **Step 3: Write minimal implementation**

Add to `backend/api/endpoints/images.py`:

```python
from backend.models.images import ImageBatchRequest, ImageBatchResponse

@router.post("/batch", response_model=ImageBatchResponse)
async def batch_process(
    batch_request: ImageBatchRequest,
    cosmos_service: Optional[CosmosDBService] = Depends(get_cosmos_service),
):
    """Process multiple image pipeline requests concurrently."""
    azure_storage = AzureBlobStorageService() if settings.AZURE_BLOB_SERVICE_URL else None
    semaphore = asyncio.Semaphore(settings.IMAGE_BATCH_MAX_CONCURRENT)
    results: list = [None] * len(batch_request.requests)

    async def _process_one(idx: int, req: ImagePipelineRequest):
        async with semaphore:
            try:
                result = await pipeline_service.process_pipeline(
                    pipeline_request=req,
                    azure_storage_service=azure_storage,
                    cosmos_service=cosmos_service,
                )
                results[idx] = result
            except Exception as exc:
                logger.error("Batch item %d failed: %s", idx, exc)
                results[idx] = ImagePipelineResponse(
                    success=False,
                    message=str(exc),
                    steps=[],
                )

    tasks = [
        asyncio.create_task(_process_one(i, req))
        for i, req in enumerate(batch_request.requests)
    ]
    await asyncio.gather(*tasks, return_exceptions=True)

    succeeded = sum(1 for r in results if r and r.success)
    failed = len(results) - succeeded

    return ImageBatchResponse(
        results=results,
        total=len(results),
        succeeded=succeeded,
        failed=failed,
    )
```

Also add `import asyncio` at the top of the file if not already present.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_batch_endpoint.py -v`
Expected: All PASS

- [ ] **Step 5: Run ALL tests**

Run: `uv run pytest tests/ --ignore=tests/integration -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add backend/api/endpoints/images.py tests/test_batch_endpoint.py
git commit -m "feat: add /api/v1/images/batch endpoint for parallel processing

Accepts multiple pipeline requests and executes them concurrently,
limited by IMAGE_BATCH_MAX_CONCURRENT (default 3). Partial failures
don't block successful results.

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 6: Final Validation

- [ ] **Step 1: Run full test suite**

```bash
uv run pytest tests/ --ignore=tests/integration -v
```
Expected: All tests PASS

- [ ] **Step 2: Verify frontend build**

```bash
cd frontend && npm run build
```
Expected: Build succeeds (no frontend changes that would break build)

- [ ] **Step 3: Final commit with changelog**

Update `CHANGELOG.md` with a new entry describing the parallel processing feature, then commit.

```bash
git add CHANGELOG.md
git commit -m "docs: add parallel processing entry to changelog

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```
