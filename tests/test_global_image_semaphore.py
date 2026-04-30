"""Tests for the module-level global image-call semaphore.

The shared semaphore lives in ``backend.core.image_pipeline.IMAGE_GEN_SEMAPHORE``
and is acquired by ``call_with_retry`` for every image-gen / image-edit call
site:

* ``ImagePipelineService.generate``  (ad-hoc generate)
* ``ImagePipelineService.edit``       (ad-hoc edit, JSON payload)
* ``ImagePipelineService.edit_with_uploads`` → ``_invoke_edit_with_files``
  (ad-hoc edit, multipart upload)

The staging pipeline (``StagingPipeline.process_room`` and
``process_single_variation``) and single-variation regeneration both flow
through ``ImagePipelineService.process_pipeline``, which delegates to
``generate``/``edit``/``edit_with_uploads`` — so they observably share the
same cap by routing.

Tests use ``asyncio.Event``-gated mocks and observe externally-visible
behavior (in-flight count via the mocked SDK call, slot-released ordering
via task completion). Internal counter values (``Semaphore._value``) are
never asserted.

See parallel-processing PRD § Global image-call cap (rate-limit bound).
"""
from __future__ import annotations

import asyncio
import io
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException, UploadFile
from httpx import Request, Response
from openai import RateLimitError

from backend.core import image_pipeline as image_pipeline_module
from backend.core.image_pipeline import ImagePipelineService
from backend.models.images import (
    ImageEditRequest,
    ImageGenerationRequest,
)


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def small_global_semaphore(monkeypatch):
    """Replace the module-level IMAGE_GEN_SEMAPHORE with a fresh cap=2 sem.

    Tests that rely on observable concurrency need a deterministic cap and
    isolation from other tests' use of the global semaphore. A fresh
    asyncio.Semaphore is bound to the current event loop on first contended
    acquire.
    """
    fresh = asyncio.Semaphore(2)
    monkeypatch.setattr(
        image_pipeline_module, "IMAGE_GEN_SEMAPHORE", fresh
    )
    return fresh


@pytest.fixture
def isolated_global_semaphore(monkeypatch):
    """Replace IMAGE_GEN_SEMAPHORE with a fresh cap=3 sem (matches default).

    Used by tests where the actual cap value doesn't matter, but loop-binding
    isolation does.
    """
    fresh = asyncio.Semaphore(3)
    monkeypatch.setattr(
        image_pipeline_module, "IMAGE_GEN_SEMAPHORE", fresh
    )
    return fresh


def _generation_response_payload(b64: str = "AAAA") -> dict:
    return {
        "created": 1,
        "data": [{"b64_json": b64}],
        "usage": {"total_tokens": 100, "input_tokens": 50, "output_tokens": 50},
    }


def _make_rate_limit_error(retry_after: str = "0.01") -> RateLimitError:
    response = Response(
        status_code=429,
        headers={"retry-after": retry_after},
        request=Request("POST", "https://example.com"),
    )
    return RateLimitError(
        message="Rate limit exceeded",
        response=response,
        body={"error": {"message": "Rate limit exceeded"}},
    )


# ---------------------------------------------------------------------------
# Burst test — global cap enforces in-flight ceiling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_burst_observes_at_most_cap_in_flight(small_global_semaphore):
    """A burst of N concurrent generate() calls observes at most cap in flight."""
    cap = 2
    n_tasks = 5

    in_flight = {"count": 0, "max": 0}
    saturated = asyncio.Event()
    release = asyncio.Event()

    async def gated_to_thread(fn, **kwargs):
        in_flight["count"] += 1
        in_flight["max"] = max(in_flight["max"], in_flight["count"])
        if in_flight["count"] >= cap:
            saturated.set()
        try:
            await release.wait()
        finally:
            in_flight["count"] -= 1
        return _generation_response_payload()

    service = ImagePipelineService()
    request = ImageGenerationRequest(prompt="a cat", model="gpt-image-2")

    with patch(
        "backend.core.image_pipeline.asyncio.to_thread",
        side_effect=gated_to_thread,
    ):
        tasks = [
            asyncio.create_task(service.generate(request)) for _ in range(n_tasks)
        ]
        # Wait until the cap is fully saturated.
        await asyncio.wait_for(saturated.wait(), timeout=5.0)
        # Give the scheduler a few cycles to confirm extras are blocked, not running.
        for _ in range(20):
            await asyncio.sleep(0)

        assert in_flight["count"] == cap, (
            f"Expected {cap} concurrent in-flight, got {in_flight['count']}"
        )
        assert in_flight["max"] == cap, (
            f"Observed {in_flight['max']} concurrent — exceeded cap of {cap}"
        )

        # Release all gated tasks; they will complete one batch at a time.
        release.set()
        results = await asyncio.gather(*tasks)

    assert len(results) == n_tasks
    assert all(r.success for r in results)
    # Even after all tasks completed, the high-water mark must equal the cap.
    assert in_flight["max"] == cap


# ---------------------------------------------------------------------------
# Slot-released semantics — observable via post-burst latency-free completion
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_slot_released_on_terminal_success(small_global_semaphore):
    """After a burst completes, the semaphore is fully released.

    Observable behavior: a follow-up call goes through immediately without
    waiting on any prior task. We verify by saturating the cap, releasing,
    waiting for everyone to finish, then firing one more call without any
    gating and confirming it completes.
    """
    cap = 2

    release = asyncio.Event()
    in_flight = {"count": 0, "peak": 0}

    async def gated_to_thread(fn, **kwargs):
        in_flight["count"] += 1
        in_flight["peak"] = max(in_flight["peak"], in_flight["count"])
        try:
            await release.wait()
        finally:
            in_flight["count"] -= 1
        return _generation_response_payload()

    service = ImagePipelineService()
    request = ImageGenerationRequest(prompt="x", model="gpt-image-2")

    with patch(
        "backend.core.image_pipeline.asyncio.to_thread",
        side_effect=gated_to_thread,
    ):
        # Saturate and complete a burst.
        burst = [
            asyncio.create_task(service.generate(request)) for _ in range(cap * 2)
        ]
        # Let burst saturate.
        for _ in range(50):
            await asyncio.sleep(0)
            if in_flight["count"] >= cap:
                break
        release.set()
        await asyncio.gather(*burst)

    # All slots released. A fresh non-gated call should complete immediately.
    async def free_to_thread(fn, **kwargs):
        return _generation_response_payload()

    with patch(
        "backend.core.image_pipeline.asyncio.to_thread",
        side_effect=free_to_thread,
    ):
        result = await asyncio.wait_for(service.generate(request), timeout=1.0)
    assert result.success


@pytest.mark.asyncio
async def test_slot_released_on_terminal_failure(small_global_semaphore):
    """When retries are exhausted and a call raises, the slot is released."""
    cap = 2

    async def always_429(fn, **kwargs):
        raise _make_rate_limit_error("0.001")

    service = ImagePipelineService()
    request = ImageGenerationRequest(prompt="x", model="gpt-image-2")

    with patch(
        "backend.core.image_pipeline.asyncio.to_thread", side_effect=always_429
    ), patch(
        "backend.core.retry.asyncio.sleep", new_callable=AsyncMock
    ):
        # Run cap+1 failing tasks in sequence — if the slot wasn't released,
        # the second one would never get to the ``call_with_retry`` body. We
        # use sequential calls (rather than concurrent) to keep the assertion
        # free of timing concerns.
        for _ in range(cap + 1):
            with pytest.raises(HTTPException) as exc_info:
                await service.generate(request)
            assert exc_info.value.status_code == 429

    # If we got here, every call acquired and released cleanly.


# ---------------------------------------------------------------------------
# Cross-path test — generate, edit, and edit_with_uploads share one cap
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cross_path_generate_edit_share_global_cap(
    small_global_semaphore, tmp_path
):
    """generate() and edit() concurrent calls share the same global cap.

    Demonstrates that two distinct ImagePipelineService entry points (the
    ad-hoc generate path and the JSON-payload edit path that the staging
    pipeline and single-variation regeneration both delegate to via
    ``process_pipeline``) hold one shared semaphore.
    """
    cap = 2

    in_flight = {"count": 0, "max": 0}
    saturated = asyncio.Event()
    release = asyncio.Event()

    async def gated_to_thread(fn, **kwargs):
        in_flight["count"] += 1
        in_flight["max"] = max(in_flight["max"], in_flight["count"])
        if in_flight["count"] >= cap:
            saturated.set()
        try:
            await release.wait()
        finally:
            in_flight["count"] -= 1
        return _generation_response_payload()

    # Build a real PNG-ish file for the edit path.
    img_path = tmp_path / "input.png"
    img_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)

    service = ImagePipelineService()
    gen_req = ImageGenerationRequest(prompt="a cat", model="gpt-image-2")
    edit_req = ImageEditRequest(prompt="modify", model="gpt-image-2", image=str(img_path))

    with patch(
        "backend.core.image_pipeline.asyncio.to_thread",
        side_effect=gated_to_thread,
    ):
        tasks = [
            asyncio.create_task(service.generate(gen_req)),
            asyncio.create_task(service.edit(edit_req)),
            asyncio.create_task(service.generate(gen_req)),
            asyncio.create_task(service.edit(edit_req)),
        ]

        await asyncio.wait_for(saturated.wait(), timeout=5.0)
        for _ in range(20):
            await asyncio.sleep(0)

        assert in_flight["count"] == cap, (
            f"Cross-path concurrency exceeded cap: {in_flight['count']} vs {cap}"
        )
        assert in_flight["max"] == cap

        release.set()
        results = await asyncio.gather(*tasks)

    assert all(r.success for r in results)
    assert in_flight["max"] == cap


@pytest.mark.asyncio
async def test_edit_with_uploads_shares_global_cap(small_global_semaphore, tmp_path):
    """edit_with_uploads() (multipart upload path) holds the global cap too.

    This is the path used by /api/v1/images/edit/upload. Until this slice it
    bypassed the retry util and the global semaphore entirely.
    """
    cap = 2
    in_flight = {"count": 0, "max": 0}
    saturated = asyncio.Event()
    release = asyncio.Event()

    async def gated_to_thread(fn, **kwargs):
        in_flight["count"] += 1
        in_flight["max"] = max(in_flight["max"], in_flight["count"])
        if in_flight["count"] >= cap:
            saturated.set()
        try:
            await release.wait()
        finally:
            in_flight["count"] -= 1
        return _generation_response_payload()

    service = ImagePipelineService()

    def _make_upload() -> UploadFile:
        # Real PNG header so _determine_extension returns 'png'.
        contents = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
        return UploadFile(
            filename="input.png",
            file=io.BytesIO(contents),
            headers={"content-type": "image/png"},
        )

    # Run 3 concurrent edit_with_uploads calls with cap=2.
    with patch(
        "backend.core.image_pipeline.asyncio.to_thread",
        side_effect=gated_to_thread,
    ):
        tasks = [
            asyncio.create_task(
                service.edit_with_uploads(
                    prompt="modify",
                    model="gpt-image-2",
                    n=1,
                    size="1024x1024",
                    quality="auto",
                    output_format="png",
                    input_fidelity="low",
                    images=[_make_upload()],
                )
            )
            for _ in range(3)
        ]

        await asyncio.wait_for(saturated.wait(), timeout=5.0)
        for _ in range(20):
            await asyncio.sleep(0)

        assert in_flight["count"] == cap
        assert in_flight["max"] == cap

        release.set()
        results = await asyncio.gather(*tasks)

    assert all(r.success for r in results)
    assert in_flight["max"] == cap


# ---------------------------------------------------------------------------
# Staging-routing test — process_pipeline delegates through the global cap
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_staging_path_via_process_pipeline_shares_global_cap(
    small_global_semaphore,
):
    """The staging pipeline path (process_pipeline action=GENERATE, save off)
    flows through the same module-level semaphore as a direct generate() call.

    This proves that single-variation regen and staging room workers — which
    both call ``ImagePipelineService.process_pipeline(...)`` — share the
    global cap with ad-hoc generate / edit endpoints.
    """
    cap = 2

    in_flight = {"count": 0, "max": 0}
    saturated = asyncio.Event()
    release = asyncio.Event()

    async def gated_to_thread(fn, **kwargs):
        in_flight["count"] += 1
        in_flight["max"] = max(in_flight["max"], in_flight["count"])
        if in_flight["count"] >= cap:
            saturated.set()
        try:
            await release.wait()
        finally:
            in_flight["count"] -= 1
        return _generation_response_payload()

    service = ImagePipelineService()

    from backend.models.images import (
        ImagePipelineRequest,
        PipelineAction,
        PipelineSaveOptions,
        PipelineAnalysisOptions,
    )

    # Build a process_pipeline GENERATE request (save disabled — we only care
    # about the image-call slice).
    pipeline_req = ImagePipelineRequest(
        action=PipelineAction.GENERATE,
        prompt="a modern living room",
        model="gpt-image-2",
        n=1,
        size="1024x1024",
        save_options=PipelineSaveOptions(enabled=False),
        analysis_options=PipelineAnalysisOptions(enabled=False),
    )

    direct_req = ImageGenerationRequest(prompt="x", model="gpt-image-2")

    with patch(
        "backend.core.image_pipeline.asyncio.to_thread",
        side_effect=gated_to_thread,
    ):
        # Mix two staging-style tasks (process_pipeline) with two direct tasks
        # (service.generate). All four go through the same global cap.
        tasks = [
            asyncio.create_task(service.process_pipeline(pipeline_req)),
            asyncio.create_task(service.generate(direct_req)),
            asyncio.create_task(service.process_pipeline(pipeline_req)),
            asyncio.create_task(service.generate(direct_req)),
        ]

        await asyncio.wait_for(saturated.wait(), timeout=5.0)
        for _ in range(20):
            await asyncio.sleep(0)

        assert in_flight["count"] == cap, (
            f"Staging-path concurrency exceeded cap: "
            f"{in_flight['count']} vs {cap}"
        )
        assert in_flight["max"] == cap

        release.set()
        results = await asyncio.gather(*tasks)

    # process_pipeline returns ImagePipelineResponse; generate returns
    # ImageGenerationResponse. Both have a `success` attribute.
    assert all(r.success for r in results)
    assert in_flight["max"] == cap


# ---------------------------------------------------------------------------
# Rate-limit error mapping for upload path — exhausted retries surface as 429
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_edit_with_uploads_maps_429_after_retry_exhaustion(
    isolated_global_semaphore,
):
    """edit_with_uploads must surface exhausted-retry RateLimitError as HTTP 429.

    Before this slice the multipart upload path bypassed call_with_retry. After
    wrapping it, RateLimitError can escape from call_with_retry once the
    attempt budget is consumed. The endpoint's catch-all ``except Exception``
    block would map it to HTTP 500 unless we explicitly catch and re-raise as
    429 — mirroring the pattern in generate()/edit().
    """
    service = ImagePipelineService()

    async def always_429(fn, **kwargs):
        raise _make_rate_limit_error("0.001")

    contents = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
    upload = UploadFile(
        filename="input.png",
        file=io.BytesIO(contents),
        headers={"content-type": "image/png"},
    )

    with patch(
        "backend.core.image_pipeline.asyncio.to_thread", side_effect=always_429
    ), patch(
        "backend.core.retry.asyncio.sleep", new_callable=AsyncMock
    ):
        with pytest.raises(HTTPException) as exc_info:
            await service.edit_with_uploads(
                prompt="modify",
                model="gpt-image-2",
                n=1,
                size="1024x1024",
                quality="auto",
                output_format="png",
                input_fidelity="low",
                images=[upload],
            )

    assert exc_info.value.status_code == 429, (
        f"Expected 429 from exhausted retries, got {exc_info.value.status_code} — "
        "the rate-limit error must not be wrapped as 500 by edit_with_uploads"
    )


# ---------------------------------------------------------------------------
# Module-level semaphore exists with the documented shape
# ---------------------------------------------------------------------------


def test_module_level_semaphore_exists():
    """``IMAGE_GEN_SEMAPHORE`` is exported and is an ``asyncio.Semaphore``."""
    assert isinstance(image_pipeline_module.IMAGE_GEN_SEMAPHORE, asyncio.Semaphore)
