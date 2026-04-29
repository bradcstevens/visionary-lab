"""Tests for staging pipeline variation URL extraction and SSE event flow."""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.core.staging_pipeline import StagingPipeline
from backend.models.images import ImageGenerationResponse, ImageSaveResponse, ImagePipelineResponse, PipelineStepResult
from backend.models.staging import (
    ItemStatus,
    Room,
    StagingProject,
    StagingSettings,
    Variation,
)


def _make_project(n_rooms=1, n_variations=2) -> StagingProject:
    """Create a minimal StagingProject for testing."""
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
        id="proj-test",
        name="Test Project",
        prompt="Modern minimalist",
        settings=StagingSettings(variations_per_room=n_variations),
        rooms=rooms,
    )


def _make_pipeline_response(image_url="https://acct.blob.core.windows.net/images/staging/proj/variations/room-0/img.png"):
    """Build a realistic ImagePipelineResponse with generation + save."""
    gen = ImageGenerationResponse(
        success=True,
        message="ok",
        imgen_model_response={
            "data": [{"b64_json": "AAAA"}],
            "usage": {"total_tokens": 100, "input_tokens": 50, "output_tokens": 50},
        },
    )
    save = ImageSaveResponse(
        success=True,
        message="Saved 1 image(s)",
        saved_images=[
            {
                "file_id": "img-001",
                "blob_name": "staging/proj/variations/room-0/img.png",
                "container": "images",
                "url": image_url,
                "size": 12345,
                "content_type": "image/png",
            }
        ],
        total_saved=1,
    )
    return ImagePipelineResponse(
        success=True,
        message="Pipeline completed",
        steps=[
            PipelineStepResult(step="edit", success=True),
            PipelineStepResult(step="save", success=True),
        ],
        generation=gen,
        save=save,
    )


class TestVariationUrlExtraction:
    """Verify staging pipeline correctly extracts image URLs from save responses."""

    @pytest.mark.asyncio
    async def test_variation_gets_url_from_saved_images(self):
        """The pipeline must read `saved.saved_images[0]['url']`, not `saved.files`."""
        from backend.core.staging_pipeline import StagingPipeline

        project = _make_project(n_rooms=1, n_variations=1)
        room = project.rooms[0]
        expected_url = "https://acct.blob.core.windows.net/images/staging/proj/variations/room-0/img.png"

        pipeline_response = _make_pipeline_response(image_url=expected_url)

        # Mock all dependencies
        mock_pipeline = AsyncMock()
        mock_pipeline.process_pipeline.return_value = pipeline_response

        mock_blob = MagicMock()
        mock_blob.get_asset_content.return_value = (b"\x89PNG\r\n", "image/png")

        mock_storage = MagicMock()
        mock_storage.update_project = MagicMock()

        mock_analyzer = AsyncMock()
        mock_analyzer.async_image_chat.return_value = {
            "description": "A modern room",
            "features": ["sofa", "table"],
        }

        mock_llm = AsyncMock()
        mock_llm.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content='["Add a plant in the corner"]'))]
        )

        staging = StagingPipeline(
            async_llm_client=mock_llm,
            llm_deployment="gpt-4o",
            image_analyzer=mock_analyzer,
            image_pipeline=mock_pipeline,
            storage_service=mock_storage,
            blob_service=mock_blob,
        )

        events = []
        async for event in staging.process_room(project, room):
            events.append(event)

        # The variation should be completed with the correct URL
        assert room.variations[0].status == ItemStatus.COMPLETED
        assert room.variations[0].image_url == expected_url
        assert room.variations[0].error is None

        # SSE events should include a variation_completed event
        completed_events = [e for e in events if e["type"] == "variation_completed"]
        assert len(completed_events) == 1
        assert completed_events[0]["image_url"] == expected_url

    @pytest.mark.asyncio
    async def test_variation_fails_gracefully_when_no_saved_images(self):
        """If saved_images is empty, variation should be marked failed, not crash."""
        from backend.core.staging_pipeline import StagingPipeline

        project = _make_project(n_rooms=1, n_variations=1)
        room = project.rooms[0]

        # Build a response where generation succeeded but save has empty saved_images
        gen = ImageGenerationResponse(
            success=True,
            message="ok",
            imgen_model_response={"data": [{"b64_json": "AAAA"}]},
        )
        save = ImageSaveResponse(
            success=True,
            message="Saved 0 image(s)",
            saved_images=[],
            total_saved=0,
        )
        pipeline_response = ImagePipelineResponse(
            success=True,
            message="Pipeline completed",
            steps=[PipelineStepResult(step="edit", success=True), PipelineStepResult(step="save", success=True)],
            generation=gen,
            save=save,
        )

        mock_pipeline = AsyncMock()
        mock_pipeline.process_pipeline.return_value = pipeline_response

        mock_blob = MagicMock()
        mock_blob.get_asset_content.return_value = (b"\x89PNG\r\n", "image/png")

        mock_storage = MagicMock()
        mock_llm = AsyncMock()
        mock_llm.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content='["Add a sofa"]'))]
        )
        mock_analyzer = AsyncMock()
        mock_analyzer.async_image_chat.return_value = {"description": "A room", "features": []}

        staging = StagingPipeline(
            async_llm_client=mock_llm,
            llm_deployment="gpt-4o",
            image_analyzer=mock_analyzer,
            image_pipeline=mock_pipeline,
            storage_service=mock_storage,
            blob_service=mock_blob,
        )

        events = []
        async for event in staging.process_room(project, room):
            events.append(event)

        # Variation should be failed, not completed with None URL
        assert room.variations[0].status == ItemStatus.FAILED
        assert room.variations[0].image_url is None

    @pytest.mark.asyncio
    async def test_all_sse_event_types_emitted(self):
        """Verify that process_room emits room_started, variation events, and room_completed."""
        from backend.core.staging_pipeline import StagingPipeline

        project = _make_project(n_rooms=1, n_variations=2)
        room = project.rooms[0]
        pipeline_response = _make_pipeline_response()

        mock_pipeline = AsyncMock()
        mock_pipeline.process_pipeline.return_value = pipeline_response

        mock_blob = MagicMock()
        mock_blob.get_asset_content.return_value = (b"\x89PNG\r\n", "image/png")

        mock_storage = MagicMock()
        mock_llm = AsyncMock()
        mock_llm.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content='["Add plants", "Add art"]'))]
        )
        mock_analyzer = AsyncMock()
        mock_analyzer.async_image_chat.return_value = {"description": "A room", "features": []}

        staging = StagingPipeline(
            async_llm_client=mock_llm,
            llm_deployment="gpt-4o",
            image_analyzer=mock_analyzer,
            image_pipeline=mock_pipeline,
            storage_service=mock_storage,
            blob_service=mock_blob,
        )

        events = []
        async for event in staging.process_room(project, room):
            events.append(event)

        event_types = [e["type"] for e in events]
        assert event_types[0] == "room_started"
        assert "room_completed" in event_types
        # Should have variation events for each variation
        variation_events = [e for e in events if e["type"].startswith("variation_")]
        assert len(variation_events) == 2

    @pytest.mark.asyncio
    async def test_sse_events_include_timing_and_token_data(self):
        """Variation SSE events must include elapsed_ms, tokens_used, and model."""
        from backend.core.staging_pipeline import StagingPipeline

        project = _make_project(n_rooms=1, n_variations=1)
        room = project.rooms[0]

        gen = ImageGenerationResponse(
            success=True,
            message="ok",
            imgen_model_response={
                "data": [{"b64_json": "AAAA"}],
                "usage": {"total_tokens": 1500, "input_tokens": 800, "output_tokens": 700},
            },
            token_usage={"total_tokens": 1500, "input_tokens": 800, "output_tokens": 700},
        )
        save = ImageSaveResponse(
            success=True,
            message="Saved 1 image(s)",
            saved_images=[{"url": "https://example.com/img.png", "blob_name": "img.png"}],
            total_saved=1,
        )
        pipeline_response = ImagePipelineResponse(
            success=True,
            message="Pipeline completed",
            steps=[PipelineStepResult(step="edit", success=True), PipelineStepResult(step="save", success=True)],
            generation=gen,
            save=save,
        )

        mock_pipeline = AsyncMock()
        mock_pipeline.process_pipeline.return_value = pipeline_response

        mock_blob = MagicMock()
        mock_blob.get_asset_content.return_value = (b"\x89PNG\r\n", "image/png")

        mock_storage = MagicMock()
        mock_llm = AsyncMock()
        mock_llm.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content='["Add plants"]'))]
        )
        mock_analyzer = AsyncMock()
        mock_analyzer.async_image_chat.return_value = {"description": "A room", "features": []}

        staging = StagingPipeline(
            async_llm_client=mock_llm,
            llm_deployment="gpt-4o",
            image_analyzer=mock_analyzer,
            image_pipeline=mock_pipeline,
            storage_service=mock_storage,
            blob_service=mock_blob,
        )

        events = []
        async for event in staging.process_room(project, room):
            events.append(event)

        completed_events = [e for e in events if e["type"] == "variation_completed"]
        assert len(completed_events) == 1
        evt = completed_events[0]
        assert "elapsed_ms" in evt
        assert isinstance(evt["elapsed_ms"], int)
        assert "tokens_used" in evt
        assert evt["tokens_used"] == 1500
        assert "model" in evt
        assert evt["model"] == "gpt-image-2"


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
