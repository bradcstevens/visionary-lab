"""Staging pipeline: image analysis → prompt adaptation → fan-out generation."""
import asyncio
import base64
import json
import logging
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import UploadFile

from backend.core.analyze import ImageAnalyzer
from backend.core.azure_storage import AzureBlobStorageService
from backend.core.config import settings
from backend.core.image_pipeline import ImagePipelineService
from backend.core.staging_storage import StagingStorageService
from backend.models.images import ImagePipelineRequest, PipelineAction, PipelineSaveOptions, PipelineAnalysisOptions
from backend.models.staging import ItemStatus, ProjectStatus, Room, StagingProject, Variation

logger = logging.getLogger(__name__)

PROMPT_ADAPTATION_TEMPLATE = """You are a virtual staging assistant. The user wants to visualize decorating ideas for their space.

ROOM ANALYSIS: {room_analysis}
USER'S STYLE DIRECTION: {user_prompt}

Generate {n} distinct variation prompts for an image editing model. Each prompt should:
- ADD items to the existing scene (furniture, decor, plants, landscaping)
- NOT remove or replace existing structures visible in the analysis
- Interpret the user's style direction differently in each variation
- Be specific about what to add and where to place it
- Reference the existing room features from the analysis

Return ONLY a JSON array of {n} strings. No other text."""


class StagingPipeline:
    """Orchestrates virtual staging: analyze → adapt prompt → generate variations."""

    def __init__(
        self,
        async_llm_client,
        llm_deployment: str,
        image_analyzer: ImageAnalyzer,
        image_pipeline: ImagePipelineService,
        storage_service: StagingStorageService,
        blob_service: AzureBlobStorageService,
    ):
        self.async_llm_client = async_llm_client
        self.llm_deployment = llm_deployment
        self.image_analyzer = image_analyzer
        self.image_pipeline = image_pipeline
        self.storage_service = storage_service
        self.blob_service = blob_service
        self.semaphore = asyncio.Semaphore(settings.STAGING_CONCURRENT_ROOMS)

    async def analyze_room(self, image_base64: str) -> Dict[str, Any]:
        """Use ImageAnalyzer to describe what's in the uploaded photo."""
        system_msg = (
            "Describe this room or outdoor space in detail. Include: "
            "existing furniture, decor, colors, flooring, lighting, plants, "
            "architectural features, and any empty areas where items could be added. "
            "Return JSON with keys: description (string), features (list of strings)."
        )
        return await self.image_analyzer.async_image_chat(
            image_base64=image_base64,
            system_message=system_msg,
        )

    async def adapt_prompt(
        self, user_prompt: str, room_analysis: str, n_variations: int,
    ) -> List[str]:
        """Use LLM to create n distinct variation prompts for this room."""
        system_content = PROMPT_ADAPTATION_TEMPLATE.format(
            room_analysis=room_analysis,
            user_prompt=user_prompt,
            n=n_variations,
        )
        for attempt in range(3):
            if attempt:
                await asyncio.sleep(1)
            response = await self.async_llm_client.chat.completions.create(
                model=self.llm_deployment,
                messages=[{"role": "system", "content": system_content}],
                temperature=0.8,
                response_format={"type": "json_object"},
            )
            try:
                content = response.choices[0].message.content
                parsed = json.loads(content)
                if isinstance(parsed, list):
                    return [str(p) for p in parsed[:n_variations]]
                if isinstance(parsed, dict) and "prompts" in parsed:
                    return [str(p) for p in parsed["prompts"][:n_variations]]
            except (json.JSONDecodeError, KeyError, IndexError):
                logger.warning(f"Prompt adaptation attempt {attempt+1} returned invalid JSON, retrying")
                continue
        raise RuntimeError("Failed to adapt prompt after 3 attempts")

    async def process_room(
        self, project: StagingProject, room: Room,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Process a single room: analyze → adapt → generate variations. Yields SSE events."""
        async with self.semaphore:
            yield {"type": "room_started", "room_id": room.id, "label": room.label}

            room.status = ItemStatus.PROCESSING
            self._update_room_in_project(project, room)

            try:
                image_bytes = await self.blob_service.get_asset_content(
                    blob_name=room.original_image_url.split("/")[-2] + "/" + room.original_image_url.split("/")[-1],
                    container_name=settings.AZURE_BLOB_IMAGE_CONTAINER,
                )
                image_b64 = base64.b64encode(image_bytes).decode("utf-8") if isinstance(image_bytes, bytes) else image_bytes

                analysis = await self.analyze_room(image_b64)
                room_description = analysis.get("description", "A room")

                adapted_prompts = await self.adapt_prompt(
                    user_prompt=project.prompt,
                    room_analysis=room_description,
                    n_variations=project.settings.variations_per_room,
                )

                for idx, adapted_prompt in enumerate(adapted_prompts):
                    variation = room.variations[idx]
                    variation.status = ItemStatus.PROCESSING
                    self._update_room_in_project(project, room)

                    start_time = time.monotonic()
                    try:
                        pipeline_request = ImagePipelineRequest(
                            action=PipelineAction.EDIT,
                            prompt=adapted_prompt,
                            model=project.settings.model,
                            n=1,
                            size=project.settings.size,
                            quality=project.settings.quality,
                            response_format="b64_json",
                            output_format="png",
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
                            variation.image_url = saved.files[0].url if saved.files else None
                            variation.status = ItemStatus.COMPLETED
                            variation.generation_metadata = {
                                "model": project.settings.model,
                                "adapted_prompt": adapted_prompt,
                                "generation_time_ms": elapsed_ms,
                            }
                        else:
                            variation.status = ItemStatus.FAILED
                            variation.error = "Pipeline returned no generation result"

                    except Exception as e:
                        logger.error(f"Variation {idx} failed for room {room.id}: {e}")
                        variation.status = ItemStatus.FAILED
                        variation.error = str(e)

                    self._update_room_in_project(project, room)

                    yield {
                        "type": f"variation_{'completed' if variation.status == ItemStatus.COMPLETED else 'failed'}",
                        "room_id": room.id,
                        "variation_index": idx,
                        "image_url": variation.image_url,
                        "error": variation.error,
                    }

                all_done = all(v.status == ItemStatus.COMPLETED for v in room.variations)
                room.status = ItemStatus.COMPLETED if all_done else ItemStatus.FAILED
                self._update_room_in_project(project, room)
                yield {"type": "room_completed", "room_id": room.id, "status": room.status}

            except Exception as e:
                logger.error(f"Room {room.id} failed: {e}")
                room.status = ItemStatus.FAILED
                room.error = str(e)
                self._update_room_in_project(project, room)
                yield {"type": "room_failed", "room_id": room.id, "error": str(e)}

    async def generate_project(self, project: StagingProject) -> AsyncGenerator[Dict[str, Any], None]:
        """Process all pending rooms in the project. Yields SSE events."""
        project.status = ProjectStatus.PROCESSING
        self.storage_service.update_project(project.id, project.dict())

        pending_rooms = [r for r in project.rooms if r.status in (ItemStatus.PENDING, ItemStatus.FAILED)]

        for room in pending_rooms:
            async for event in self.process_room(project, room):
                yield event

        all_completed = all(r.status == ItemStatus.COMPLETED for r in project.rooms)
        project.status = ProjectStatus.COMPLETED if all_completed else ProjectStatus.FAILED
        self.storage_service.update_project(project.id, project.dict())
        yield {"type": "project_completed", "status": project.status}

    def _update_room_in_project(self, project: StagingProject, room: Room):
        """Persist room updates to Cosmos DB."""
        for i, r in enumerate(project.rooms):
            if r.id == room.id:
                project.rooms[i] = room
                break
        try:
            self.storage_service.update_project(project.id, project.dict())
        except Exception as e:
            logger.error(f"Failed to persist room update: {e}")