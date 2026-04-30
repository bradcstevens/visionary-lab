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


def _persisted_variation_for(mock_storage, room_id, variation_id):
    """Return the most recent persisted variation dict for (room_id, variation_id)."""
    found = None
    for call in mock_storage.update_project.call_args_list:
        # update_project signature: (project_id, project_dict)
        args = call.args if call.args else ()
        if len(args) < 2:
            continue
        project_dict = args[1]
        for r in project_dict.get("rooms", []):
            if r.get("id") != room_id:
                continue
            for v in r.get("variations", []):
                if v.get("id") == variation_id:
                    found = v
    return found


class TestFailedVariationPersistsAdaptedPrompt:
    """Regression: failed variations must persist their attempted adapted_prompt
    BEFORE the image-gen call, so a subsequent retry can re-use it."""

    @pytest.mark.asyncio
    async def test_process_room_persists_adapted_prompt_on_image_gen_failure(self):
        """When image_pipeline.process_pipeline raises, the variation's
        adapted_prompt must already be persisted to Cosmos so a later retry
        can read it back."""
        from backend.core.staging_pipeline import StagingPipeline

        project = _make_project(n_rooms=1, n_variations=1)
        room = project.rooms[0]
        target_variation_id = room.variations[0].id

        # Track the persisted adapted_prompt at the moment process_pipeline is invoked,
        # not just after the failure has been handled.
        persisted_prompt_at_call_time: dict = {}

        mock_blob = MagicMock()
        mock_blob.get_asset_content.return_value = (b"\x89PNG\r\n", "image/png")

        mock_storage = MagicMock()
        mock_storage.update_project = MagicMock()

        mock_analyzer = AsyncMock()
        mock_analyzer.async_image_chat.return_value = {
            "description": "A modern room",
            "features": [],
        }

        mock_llm = AsyncMock()
        # The LLM will produce one adapted prompt
        mock_llm.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content='["Add a velvet teal sofa near the window"]'))]
        )

        async def _fail_after_inspecting_persistence(*_args, **_kwargs):
            # By the time the pipeline is actually called, persistence must already
            # contain the adapted_prompt for this variation.
            v = _persisted_variation_for(mock_storage, room.id, target_variation_id)
            persisted_prompt_at_call_time["value"] = (
                (v or {}).get("generation_metadata", {}) or {}
            ).get("adapted_prompt")
            raise RuntimeError("simulated image-gen failure")

        mock_pipeline = AsyncMock()
        mock_pipeline.process_pipeline.side_effect = _fail_after_inspecting_persistence

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

        # Persistence happened before the pipeline call, capturing the prompt:
        assert persisted_prompt_at_call_time.get("value") == \
            "Add a velvet teal sofa near the window", \
            "adapted_prompt must be persisted to Cosmos BEFORE the image-gen call"

        # Final state: variation is failed and still has the prompt on record:
        assert room.variations[0].status == ItemStatus.FAILED
        assert room.variations[0].generation_metadata is not None
        # Existing convention in this codebase stores generation_metadata as a dict.
        meta = room.variations[0].generation_metadata
        meta_prompt = meta.get("adapted_prompt") if isinstance(meta, dict) else meta.adapted_prompt
        assert meta_prompt == "Add a velvet teal sofa near the window"

        # And the most recent persisted snapshot also carries the prompt:
        final = _persisted_variation_for(mock_storage, room.id, target_variation_id)
        assert final is not None
        assert (final.get("generation_metadata") or {}).get("adapted_prompt") == \
            "Add a velvet teal sofa near the window"

    @pytest.mark.asyncio
    async def test_process_single_variation_persists_adapted_prompt_on_image_gen_failure(self):
        """Same regression for the single-variation regen pipeline."""
        project = _make_project(n_rooms=1, n_variations=2)
        room = project.rooms[0]
        variation = room.variations[1]
        adapted_prompt = "Add a sculptural pendant lamp over the dining table"

        persisted_prompt_at_call_time: dict = {}

        mock_storage = MagicMock()

        async def _fail_after_inspecting_persistence(*_args, **_kwargs):
            v = _persisted_variation_for(mock_storage, room.id, variation.id)
            persisted_prompt_at_call_time["value"] = (
                (v or {}).get("generation_metadata", {}) or {}
            ).get("adapted_prompt")
            raise RuntimeError("simulated image-gen failure")

        with patch("backend.core.staging_pipeline.StagingPipeline.__init__", return_value=None):
            pipeline = StagingPipeline.__new__(StagingPipeline)
            pipeline.image_pipeline = AsyncMock()
            pipeline.image_pipeline.process_pipeline.side_effect = _fail_after_inspecting_persistence
            pipeline.blob_service = MagicMock()
            pipeline.blob_service.get_asset_content.return_value = (b"fake-image-bytes", "image/png")
            pipeline.storage_service = mock_storage
            pipeline.semaphore = asyncio.Semaphore(1)

            events = []
            async for event in pipeline.process_single_variation(project, room, variation, adapted_prompt):
                events.append(event)

        # Persistence happened before the pipeline call:
        assert persisted_prompt_at_call_time.get("value") == adapted_prompt, \
            "adapted_prompt must be persisted to Cosmos BEFORE the image-gen call"

        # Final state: under the rollback-on-failure contract (issue 002), the
        # variation is restored to its pre-regen visible state — but the new
        # adapted_prompt remains in generation_metadata so a retry can re-use it.
        # The variation here started in PENDING (default), so rollback returns it
        # to PENDING, NOT FAILED. The dedicated rollback test class
        # (TestSingleVariationFailureRollbackAndCleanup) covers the prior-FAILED
        # case explicitly.
        assert variation.status == ItemStatus.PENDING
        assert variation.generation_metadata is not None
        meta = variation.generation_metadata
        meta_prompt = meta.get("adapted_prompt") if isinstance(meta, dict) else meta.adapted_prompt
        assert meta_prompt == adapted_prompt

        # And the most recent persisted snapshot carries the prompt:
        final = _persisted_variation_for(mock_storage, room.id, variation.id)
        assert final is not None
        assert (final.get("generation_metadata") or {}).get("adapted_prompt") == adapted_prompt


async def _drain_cleanup_tasks(pipeline) -> None:
    """Wait for any fire-and-forget cleanup tasks (e.g., blob deletes) to settle."""
    pending = getattr(pipeline, "_cleanup_tasks", None)
    if not pending:
        return
    await asyncio.gather(*list(pending), return_exceptions=True)


class TestSingleVariationFailureRollbackAndCleanup:
    """Issue 002 (single-variation-regeneration): failure preserves the prior
    user-visible state (status / image_url / error), success deletes the prior
    blob, and disconnect mid-flight does not strand the variation in PROCESSING.
    """

    PRIOR_URL = "https://acct.blob.core.windows.net/images/staging/proj/variations/room-0/prior.png"
    PRIOR_BLOB_NAME = "staging/proj/variations/room-0/prior.png"
    NEW_URL = "https://acct.blob.core.windows.net/images/staging/proj/variations/room-0/new.png"

    def _make_pipeline_with_prior_image(self, mock_blob, mock_storage, mock_pipeline_call):
        """Construct a StagingPipeline with mocked deps. Returns (pipeline, project, room, variation)."""
        project = _make_project(n_rooms=1, n_variations=2)
        room = project.rooms[0]
        variation = room.variations[1]
        # Prior state: COMPLETED with an image
        variation.status = ItemStatus.COMPLETED
        variation.image_url = self.PRIOR_URL
        variation.error = None

        with patch("backend.core.staging_pipeline.StagingPipeline.__init__", return_value=None):
            pipeline = StagingPipeline.__new__(StagingPipeline)
            pipeline.image_pipeline = AsyncMock()
            pipeline.image_pipeline.process_pipeline = mock_pipeline_call
            pipeline.blob_service = mock_blob
            pipeline.blob_service.get_asset_content.return_value = (b"\x89PNG\r\n", "image/png")
            pipeline.storage_service = mock_storage
            pipeline.semaphore = asyncio.Semaphore(1)
            pipeline._cleanup_tasks = set()
        return pipeline, project, room, variation

    @pytest.mark.asyncio
    async def test_failure_preserves_prior_image_url_and_status(self):
        """When image-gen raises, the variation must end with prior status/url/error
        so the UI continues to show the prior image (not a failure tile)."""
        mock_blob = MagicMock()
        mock_blob.delete_asset = MagicMock(return_value=True)
        mock_storage = MagicMock()

        async def _raise(*_args, **_kwargs):
            raise RuntimeError("simulated image-gen failure")

        pipeline, project, room, variation = self._make_pipeline_with_prior_image(
            mock_blob, mock_storage, AsyncMock(side_effect=_raise),
        )
        adapted_prompt = "Add a velvet teal sofa near the window"

        events = []
        async for event in pipeline.process_single_variation(project, room, variation, adapted_prompt):
            events.append(event)
        await _drain_cleanup_tasks(pipeline)

        # Variation is restored to its pre-regen visible state:
        assert variation.status == ItemStatus.COMPLETED, \
            "Failure must restore prior status so the UI keeps the prior image visible"
        assert variation.image_url == self.PRIOR_URL, \
            "Failure must preserve the prior image_url"
        assert variation.error is None, "Failure must restore the prior error (None for COMPLETED)"

        # SSE event reports the regen failure, but the persisted state shows the prior image:
        failed_events = [e for e in events if e["type"] == "variation_failed"]
        assert len(failed_events) == 1
        assert "simulated image-gen failure" in (failed_events[0].get("error") or "")

        # The prior blob is NOT deleted on failure:
        mock_blob.delete_asset.assert_not_called()

        # The new adapted_prompt is preserved in metadata so retry can re-use it:
        meta = variation.generation_metadata
        meta_prompt = meta.get("adapted_prompt") if isinstance(meta, dict) else meta.adapted_prompt
        assert meta_prompt == adapted_prompt

    @pytest.mark.asyncio
    async def test_success_deletes_prior_blob(self):
        """On successful regen, the prior blob is deleted fire-and-forget."""
        mock_blob = MagicMock()
        mock_blob.delete_asset = MagicMock(return_value=True)
        mock_storage = MagicMock()

        pipeline_response = _make_pipeline_response(image_url=self.NEW_URL)
        pipeline, project, room, variation = self._make_pipeline_with_prior_image(
            mock_blob, mock_storage, AsyncMock(return_value=pipeline_response),
        )

        events = []
        async for event in pipeline.process_single_variation(
            project, room, variation, "new prompt",
        ):
            events.append(event)
        await _drain_cleanup_tasks(pipeline)

        # Variation transitioned to COMPLETED with the new URL:
        assert variation.status == ItemStatus.COMPLETED
        assert variation.image_url == self.NEW_URL

        # Prior blob deleted exactly once with the right name + container:
        from backend.core.config import settings as _settings
        mock_blob.delete_asset.assert_called_once_with(
            self.PRIOR_BLOB_NAME, _settings.AZURE_BLOB_IMAGE_CONTAINER,
        )

        completed_events = [e for e in events if e["type"] == "variation_completed"]
        assert len(completed_events) == 1
        assert completed_events[0]["image_url"] == self.NEW_URL

    @pytest.mark.asyncio
    async def test_no_prior_image_no_delete(self):
        """When the variation had no prior image (e.g., regenning a previously
        FAILED variation), success path must NOT call delete_asset."""
        mock_blob = MagicMock()
        mock_blob.delete_asset = MagicMock(return_value=True)
        mock_storage = MagicMock()

        pipeline_response = _make_pipeline_response(image_url=self.NEW_URL)
        pipeline, project, room, variation = self._make_pipeline_with_prior_image(
            mock_blob, mock_storage, AsyncMock(return_value=pipeline_response),
        )
        # No prior image (override the COMPLETED-with-prior-image setup)
        variation.image_url = None
        variation.status = ItemStatus.FAILED
        variation.error = "previous attempt failed"

        async for _ in pipeline.process_single_variation(project, room, variation, "new prompt"):
            pass
        await _drain_cleanup_tasks(pipeline)

        assert variation.status == ItemStatus.COMPLETED
        assert variation.image_url == self.NEW_URL
        assert variation.error is None
        mock_blob.delete_asset.assert_not_called()

    @pytest.mark.asyncio
    async def test_save_without_url_preserves_prior_state(self):
        """If the pipeline returns a 'save succeeded but no URL' response, the
        regen counts as a failure and prior state is preserved."""
        mock_blob = MagicMock()
        mock_blob.delete_asset = MagicMock(return_value=True)
        mock_storage = MagicMock()

        # Build a response with empty saved_images
        gen = ImageGenerationResponse(success=True, message="ok",
                                       imgen_model_response={"data": [{"b64_json": "AAAA"}]})
        save = ImageSaveResponse(success=True, message="Saved 0", saved_images=[], total_saved=0)
        empty_response = ImagePipelineResponse(
            success=True, message="ok",
            steps=[PipelineStepResult(step="edit", success=True), PipelineStepResult(step="save", success=True)],
            generation=gen, save=save,
        )

        pipeline, project, room, variation = self._make_pipeline_with_prior_image(
            mock_blob, mock_storage, AsyncMock(return_value=empty_response),
        )

        events = []
        async for event in pipeline.process_single_variation(project, room, variation, "new prompt"):
            events.append(event)
        await _drain_cleanup_tasks(pipeline)

        # Prior state is preserved:
        assert variation.status == ItemStatus.COMPLETED
        assert variation.image_url == self.PRIOR_URL
        # The SSE event reports the failure:
        failed_events = [e for e in events if e["type"] == "variation_failed"]
        assert len(failed_events) == 1
        # No blob delete on failure:
        mock_blob.delete_asset.assert_not_called()

    @pytest.mark.asyncio
    async def test_failure_from_prior_failed_state_keeps_failed(self):
        """If the variation was FAILED before regen and the regen also fails,
        the persisted state must remain FAILED with the prior error message."""
        mock_blob = MagicMock()
        mock_blob.delete_asset = MagicMock(return_value=True)
        mock_storage = MagicMock()

        async def _raise(*_args, **_kwargs):
            raise RuntimeError("regen also failed")

        pipeline, project, room, variation = self._make_pipeline_with_prior_image(
            mock_blob, mock_storage, AsyncMock(side_effect=_raise),
        )
        # Prior state: FAILED with a specific error
        variation.status = ItemStatus.FAILED
        variation.image_url = None
        variation.error = "earlier rate limit"

        async for _ in pipeline.process_single_variation(project, room, variation, "new prompt"):
            pass
        await _drain_cleanup_tasks(pipeline)

        assert variation.status == ItemStatus.FAILED
        assert variation.image_url is None
        assert variation.error == "earlier rate limit"
        mock_blob.delete_asset.assert_not_called()

    @pytest.mark.asyncio
    async def test_pre_shield_cancellation_restores_prior_state(self):
        """If the SSE client disconnects DURING image-gen (before the shielded
        persist runs), the variation must end up in its prior visible state, not
        stranded in PROCESSING. Reconcile cannot recover this case because
        single-variation regen does not elevate project.status."""
        mock_blob = MagicMock()
        mock_blob.delete_asset = MagicMock(return_value=True)
        mock_storage = MagicMock()

        async def _raise_cancelled(*_args, **_kwargs):
            raise asyncio.CancelledError()

        pipeline, project, room, variation = self._make_pipeline_with_prior_image(
            mock_blob, mock_storage, AsyncMock(side_effect=_raise_cancelled),
        )

        # Drive the generator until it raises CancelledError out
        gen = pipeline.process_single_variation(project, room, variation, "new prompt")
        with pytest.raises(asyncio.CancelledError):
            async for _ in gen:
                pass
        await _drain_cleanup_tasks(pipeline)

        # Variation is no longer stranded in PROCESSING:
        assert variation.status != ItemStatus.PROCESSING
        # And the prior visible state is preserved:
        assert variation.status == ItemStatus.COMPLETED
        assert variation.image_url == self.PRIOR_URL
        # No blob delete on cancellation:
        mock_blob.delete_asset.assert_not_called()

    @pytest.mark.asyncio
    async def test_invariant_does_not_elevate_project_status(self):
        """process_single_variation must never set project.status = processing.
        That invariant is what keeps reconcile from interfering mid-regen."""
        mock_blob = MagicMock()
        mock_blob.delete_asset = MagicMock(return_value=True)
        mock_storage = MagicMock()

        pipeline_response = _make_pipeline_response(image_url=self.NEW_URL)
        pipeline, project, room, variation = self._make_pipeline_with_prior_image(
            mock_blob, mock_storage, AsyncMock(return_value=pipeline_response),
        )
        # Set project.status to a non-processing baseline
        from backend.models.staging import ProjectStatus
        project.status = ProjectStatus.COMPLETED

        async for _ in pipeline.process_single_variation(project, room, variation, "new prompt"):
            # During processing project.status must NOT be processing:
            assert project.status != ProjectStatus.PROCESSING
        await _drain_cleanup_tasks(pipeline)
        # And it remains non-processing afterwards:
        assert project.status != ProjectStatus.PROCESSING

    @pytest.mark.asyncio
    async def test_delete_asset_failure_is_swallowed(self):
        """If best-effort cleanup of the prior blob raises, the regen must
        still complete successfully — cleanup failures must never propagate or
        flip the variation back to FAILED."""
        mock_blob = MagicMock()
        # delete_asset raises, simulating a transient storage error
        mock_blob.delete_asset = MagicMock(side_effect=RuntimeError("storage offline"))
        mock_storage = MagicMock()

        pipeline_response = _make_pipeline_response(image_url=self.NEW_URL)
        pipeline, project, room, variation = self._make_pipeline_with_prior_image(
            mock_blob, mock_storage, AsyncMock(return_value=pipeline_response),
        )

        events = []
        async for event in pipeline.process_single_variation(
            project, room, variation, "new prompt"
        ):
            events.append(event)
        # Drain so the cleanup task's exception is observed (and swallowed).
        await _drain_cleanup_tasks(pipeline)

        # Regen still reports success and the variation is COMPLETED with the
        # new image — the delete failure does not surface to the caller.
        assert any(e["type"] == "variation_completed" for e in events)
        assert variation.status == ItemStatus.COMPLETED
        assert variation.image_url == self.NEW_URL
        # The cleanup task did attempt the delete (and the mock raised).
        mock_blob.delete_asset.assert_called_once()


class TestAdaptPromptWithRejectedPrompt:
    """Issue 003 of single-variation-regeneration PRD: ``StagingPipeline.adapt_prompt``
    accepts an optional ``rejected_prompt`` and threads it into the LLM
    system message so "Try Something New" diverges from the rejected
    aesthetic on the no-brief regen path.
    """

    def _make_pipeline(self, mock_llm: AsyncMock) -> StagingPipeline:
        return StagingPipeline(
            async_llm_client=mock_llm,
            llm_deployment="gpt-4o",
            image_analyzer=MagicMock(),
            image_pipeline=MagicMock(),
            storage_service=MagicMock(),
            blob_service=MagicMock(),
        )

    @pytest.mark.asyncio
    async def test_default_rejected_prompt_is_none_no_steering(self):
        mock_llm = AsyncMock()
        mock_llm.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content='{"prompts":["A","B"]}'))]
        )
        pipeline = self._make_pipeline(mock_llm)
        await pipeline.adapt_prompt(
            user_prompt="Modern minimalist",
            room_analysis="A sunlit living room",
            n_variations=2,
        )
        sent = mock_llm.chat.completions.create.call_args.kwargs["messages"][0]["content"]
        assert "REJECTED_PRIOR_DIRECTION" not in sent
        assert "REGENERATION STEERING" not in sent

    @pytest.mark.asyncio
    async def test_rejected_prompt_appears_in_llm_call_site(self):
        """Integration acceptance criterion: assert at the actual LLM
        client (not at the boundary of ``adapt_prompt``) that the rejected
        prompt is embedded in the system message."""
        mock_llm = AsyncMock()
        mock_llm.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content='{"prompts":["new A","new B"]}'))]
        )
        pipeline = self._make_pipeline(mock_llm)
        await pipeline.adapt_prompt(
            user_prompt="Modern minimalist",
            room_analysis="A sunlit living room",
            n_variations=2,
            rejected_prompt="MAGENTA-AND-CHROME MAXIMALIST AESTHETIC",
        )
        sent = mock_llm.chat.completions.create.call_args.kwargs["messages"][0]["content"]
        assert "MAGENTA-AND-CHROME MAXIMALIST AESTHETIC" in sent
        assert "REJECTED_PRIOR_DIRECTION" in sent

    @pytest.mark.asyncio
    async def test_rejected_prompt_does_not_drop_user_intent(self):
        """The user's ``user_prompt`` (e.g. "Modern minimalist") must
        survive in the system message alongside the steering block."""
        mock_llm = AsyncMock()
        mock_llm.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content='{"prompts":["A","B"]}'))]
        )
        pipeline = self._make_pipeline(mock_llm)
        await pipeline.adapt_prompt(
            user_prompt="USER_INTENT_SENTINEL_TOKEN",
            room_analysis="A modern living room",
            n_variations=2,
            rejected_prompt="industrial concrete jungle",
        )
        sent = mock_llm.chat.completions.create.call_args.kwargs["messages"][0]["content"]
        assert "USER_INTENT_SENTINEL_TOKEN" in sent
        assert "industrial concrete jungle" in sent

    @pytest.mark.asyncio
    async def test_empty_rejected_prompt_treated_as_none(self):
        mock_llm = AsyncMock()
        mock_llm.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content='{"prompts":["A","B"]}'))]
        )
        pipeline = self._make_pipeline(mock_llm)
        await pipeline.adapt_prompt(
            user_prompt="Modern minimalist",
            room_analysis="A sunlit living room",
            n_variations=2,
            rejected_prompt="",
        )
        sent = mock_llm.chat.completions.create.call_args.kwargs["messages"][0]["content"]
        assert "REJECTED_PRIOR_DIRECTION" not in sent
