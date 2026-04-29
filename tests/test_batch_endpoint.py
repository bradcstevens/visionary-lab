"""Tests for the /api/v1/images/batch endpoint."""
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from backend.models.images import (
    ImageGenerationResponse,
    ImagePipelineResponse,
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
        async def fake_pipeline(pipeline_request, **kw):
            return _make_pipeline_response(pipeline_request.prompt)

        with patch(
            "backend.api.endpoints.images.pipeline_service.process_pipeline",
            new_callable=AsyncMock,
            side_effect=fake_pipeline,
        ):
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
        async def flaky_pipeline(pipeline_request, **kw):
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

    def test_batch_respects_concurrency_limit(self, client):
        """No more than IMAGE_BATCH_MAX_CONCURRENT requests should run in parallel."""
        import asyncio
        peak = 0
        current = 0
        lock = asyncio.Lock()

        async def counting_pipeline(pipeline_request, **kw):
            nonlocal peak, current
            async with lock:
                current += 1
                peak = max(peak, current)
            try:
                await asyncio.sleep(0.05)
                return _make_pipeline_response(pipeline_request.prompt)
            finally:
                async with lock:
                    current -= 1

        with patch(
            "backend.api.endpoints.images.pipeline_service.process_pipeline",
            new_callable=AsyncMock,
            side_effect=counting_pipeline,
        ):
            response = client.post(
                "/api/v1/images/batch",
                json={"requests": [{"action": "generate", "prompt": f"p{i}"} for i in range(6)]},
            )
        assert response.status_code == 200
        # IMAGE_BATCH_MAX_CONCURRENT defaults to 3
        assert peak <= 3, f"peak={peak}"

    def test_batch_semaphore_is_module_level(self):
        """Verify the semaphore is declared at module level to be shared across requests."""
        from backend.api.endpoints import images
        import asyncio
        # Module-level semaphore should exist
        assert hasattr(images, "_batch_semaphore")
        assert isinstance(images._batch_semaphore, asyncio.Semaphore)
        # Should be initialized with the config value
        from backend.core.config import settings
        assert images._batch_semaphore._value == settings.IMAGE_BATCH_MAX_CONCURRENT
