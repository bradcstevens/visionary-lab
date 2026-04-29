"""Tests for image generation 429 retry with exponential backoff."""
from unittest.mock import AsyncMock, patch, MagicMock
import io

import pytest
from fastapi import HTTPException
from httpx import Response, Request
from openai import RateLimitError

from backend.core.image_pipeline import ImagePipelineService
from backend.models.images import ImageGenerationRequest, ImageEditRequest


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

    def mock_to_thread(fn, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise _make_rate_limit_error("0.01")
        # Build a minimal response dict that ImagePipelineService.generate expects
        return {
            "created": 1,
            "data": [{"b64_json": "AAAA"}]
        }

    async def fake_to_thread(fn, **kwargs):
        return mock_to_thread(fn, **kwargs)

    with patch("backend.core.image_pipeline.asyncio.to_thread", side_effect=fake_to_thread):
        with patch("backend.core.image_pipeline.asyncio.sleep", new_callable=AsyncMock):
            result = await service.generate(request)

    assert call_count == 3
    assert result.success


@pytest.mark.asyncio
async def test_retry_exhausted_raises_http_exception():
    """After exhausting all retries, RateLimitError should propagate as HTTP 429."""
    service = ImagePipelineService()
    request = ImageGenerationRequest(prompt="a cat", model="gpt-image-2")

    async def always_429(fn, **kwargs):
        raise _make_rate_limit_error("0.01")

    with patch("backend.core.image_pipeline.asyncio.to_thread", side_effect=always_429):
        with patch("backend.core.image_pipeline.asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(HTTPException) as exc_info:
                await service.generate(request)
    assert exc_info.value.status_code == 429


@pytest.mark.asyncio
async def test_non_429_errors_not_retried():
    """Non-rate-limit errors should propagate immediately (no retry)."""
    service = ImagePipelineService()
    request = ImageGenerationRequest(prompt="a cat", model="gpt-image-2")

    call_count = 0

    async def fail_with_value_error(fn, **kwargs):
        nonlocal call_count
        call_count += 1
        raise ValueError("bad input")

    with patch("backend.core.image_pipeline.asyncio.to_thread", side_effect=fail_with_value_error):
        with pytest.raises(HTTPException):
            await service.generate(request)

    assert call_count == 1


@pytest.mark.asyncio
async def test_edit_retry_reopens_files(tmp_path):
    """edit() must reopen file paths on each retry attempt, not reuse closed handles."""
    # Create a minimal PNG-like file
    img_path = tmp_path / "input.png"
    img_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)

    service = ImagePipelineService()
    request = ImageEditRequest(prompt="modify", model="gpt-image-2", image=str(img_path))

    call_count = 0

    async def to_thread_with_retry(fn, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise _make_rate_limit_error("0.01")
        # On second call, invoke the closure to verify files reopen successfully
        try:
            fn(**kwargs)
        except Exception as e:
            # If it's a file handle error, let it surface
            if "closed file" in str(e):
                raise
        return {"created": 1, "data": [{"b64_json": "AAAA"}]}

    with patch("backend.core.gpt_image.GPTImageClient.edit_image", return_value={"created": 1, "data": [{"b64_json": "AAAA"}]}):
        with patch("backend.core.image_pipeline.asyncio.sleep", new_callable=AsyncMock):
            with patch("backend.core.image_pipeline.asyncio.to_thread", side_effect=to_thread_with_retry):
                result = await service.edit(request)

    assert call_count >= 2
    assert result.success

