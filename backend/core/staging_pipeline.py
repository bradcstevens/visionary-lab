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
from backend.core.retry import call_with_retry
from backend.core.staging_storage import StagingStorageService
from backend.models.images import ImagePipelineRequest, PipelineAction, PipelineSaveOptions, PipelineAnalysisOptions
from backend.models.staging import ItemStatus, ProjectStatus, Room, StagingProject, Variation

logger = logging.getLogger(__name__)

# Module-level per-project lock registry. Keyed by project.id, this dict is
# shared across ALL StagingPipeline instances in the process so concurrent
# requests for the same project (each given its own pipeline by the FastAPI
# Depends factory) still serialize at the storage boundary. In-process
# serialization is correct because the backend container app is pinned to a
# single replica — see parallel-processing PRD § Single-replica deployment
# constraint and § Per-project lock for Cosmos updates. The dict has no
# eviction in this slice; projects are short-lived enough that growth is
# acceptable per the PRD's accepted-trade list.
_PROJECT_LOCKS: Dict[str, asyncio.Lock] = {}


def _get_project_lock(project_id: str) -> asyncio.Lock:
    """Return the process-wide per-project asyncio.Lock, creating it lazily.

    Safe under concurrent first-use because asyncio is single-threaded: the
    get/assignment pair has no `await` between it, so two concurrently-
    scheduled tasks cannot both observe `None` and both create a fresh Lock.
    """
    lock = _PROJECT_LOCKS.get(project_id)
    if lock is None:
        lock = asyncio.Lock()
        _PROJECT_LOCKS[project_id] = lock
    return lock

INDOOR_PROMPT_TEMPLATE = """You are a virtual staging assistant. The user wants to visualize decorating ideas for their space.

ROOM ANALYSIS: {room_analysis}
USER'S STYLE DIRECTION: {user_prompt}

Generate {n} distinct variation prompts for an image editing model. Each prompt should:
- ADD items to the existing scene (furniture, decor, plants)
- NOT remove or replace existing structures visible in the analysis
- Interpret the user's style direction differently in each variation
- Be specific about what to add and where to place it
- Reference the existing room features from the analysis

Return ONLY a JSON array of {n} strings. No other text."""

OUTDOOR_PROMPT_TEMPLATE = """You are a landscape visualization assistant. The user wants to visualize landscaping and outdoor design ideas.

SCENE ANALYSIS: {room_analysis}
USER'S DESIGN DIRECTION: {user_prompt}

Generate {n} distinct variation prompts for an image editing model. Each prompt should:
- ADD plants, trees, shrubs, hardscaping, or outdoor elements to the existing scene
- NOT remove or replace existing structures (patios, fences, pergolas, fire pits)
- Specify plant species with visual characteristics (leaf color, form, texture, size)
- Describe placement using landscape terms (back row, border, along fence, flanking)
- Interpret the design direction differently in each variation
- Reference the existing outdoor features from the analysis

Return ONLY a JSON array of {n} strings. No other text."""

OUTDOOR_KEYWORDS = {"backyard", "fence", "patio", "pergola", "turf", "lawn", "garden",
                     "yard", "outdoor", "landscape", "deck", "driveway", "tree", "shrub"}


def build_adaptation_template(room_analysis: str, is_outdoor: bool = False) -> str:
    """Return the appropriate prompt template based on context."""
    if is_outdoor:
        return OUTDOOR_PROMPT_TEMPLATE
    analysis_lower = room_analysis.lower()
    if any(kw in analysis_lower for kw in OUTDOOR_KEYWORDS):
        return OUTDOOR_PROMPT_TEMPLATE
    return INDOOR_PROMPT_TEMPLATE


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
        # Fire-and-forget cleanup tasks (e.g. prior-blob deletes after a
        # successful single-variation regen). We hold strong references here so
        # the tasks are not garbage-collected before they run, and so tests can
        # await them deterministically. See `_schedule_blob_cleanup` and
        # `process_single_variation`.
        self._cleanup_tasks: set = set()

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
        template = build_adaptation_template(room_analysis)
        system_content = template.format(
            room_analysis=room_analysis,
            user_prompt=user_prompt,
            n=n_variations,
        )
        for attempt in range(3):
            if attempt:
                await asyncio.sleep(1)
            # Wrap the network call with retry util; the outer JSON-parse loop
            # remains as an inner-style loop (per the parallel-processing PRD,
            # JSON-parse retries inside prompt adaptation stay).
            response = await call_with_retry(
                lambda: self.async_llm_client.chat.completions.create(
                    model=self.llm_deployment,
                    messages=[{"role": "system", "content": system_content}],
                    temperature=0.8,
                    response_format={"type": "json_object"},
                ),
                semaphore=None,
                model=self.llm_deployment,
                attempts=settings.IMAGE_GEN_RETRY_ATTEMPTS,
                base_delay=settings.IMAGE_GEN_RETRY_BASE_DELAY,
                max_total_wait=settings.IMAGE_GEN_RETRY_MAX_TOTAL_WAIT,
            )
            try:
                content = response.choices[0].message.content
                parsed = json.loads(content)
                if isinstance(parsed, list):
                    return [str(p) for p in parsed[:n_variations]]
                if isinstance(parsed, dict):
                    # Try common key names the LLM might use
                    for key in ("prompts", "variations", "results", "data"):
                        if key in parsed and isinstance(parsed[key], list):
                            return [str(p) for p in parsed[key][:n_variations]]
                    # If dict has string values, collect them
                    values = [v for v in parsed.values() if isinstance(v, (str, list))]
                    if values and isinstance(values[0], list):
                        return [str(p) for p in values[0][:n_variations]]
            except (json.JSONDecodeError, KeyError, IndexError):
                logger.warning(f"Prompt adaptation attempt {attempt+1} returned invalid JSON, retrying")
                continue
        raise RuntimeError("Failed to adapt prompt after 3 attempts")

    async def process_room(
        self, project: StagingProject, room: Room, brief_prompts: dict = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Process a single room: analyze → adapt → generate variations. Yields SSE events."""
        async with self.semaphore:
            yield {"type": "room_started", "room_id": room.id, "label": room.label}

            room.status = ItemStatus.PROCESSING
            await self._update_room_in_project(project, room)

            try:
                image_content, _ = self.blob_service.get_asset_content(
                    blob_name=self._extract_blob_name(room.original_image_url),
                    container_name=settings.AZURE_BLOB_IMAGE_CONTAINER,
                )
                if image_content is None:
                    raise RuntimeError(f"Image not found in blob storage: {room.original_image_url}")
                image_b64 = base64.b64encode(image_content).decode("utf-8")

                analysis = await self.analyze_room(image_b64)
                room_description = analysis.get("description", "A room")

                # Use brief-generated prompts if available, else fall back to adapt_prompt
                if brief_prompts and room.id in brief_prompts:
                    adapted_prompts = brief_prompts[room.id]
                else:
                    adapted_prompts = await self.adapt_prompt(
                        user_prompt=project.prompt,
                        room_analysis=room_description,
                        n_variations=project.settings.variations_per_room,
                    )

                for idx, adapted_prompt in enumerate(adapted_prompts):
                    if idx >= len(room.variations):
                        logger.warning(
                            "More adapted prompts (%d) than variations (%d) for room %s; "
                            "ignoring excess prompts",
                            len(adapted_prompts), len(room.variations), room.id,
                        )
                        break
                    variation = room.variations[idx]
                    variation.status = ItemStatus.PROCESSING
                    # Persist the attempted prompt BEFORE the image-gen call so a
                    # subsequent retry can re-use it even if generation fails or
                    # the worker process dies mid-call. See PRD Implementation
                    # Decisions → Backend (`adapted_prompt` persistence bullet).
                    variation.generation_metadata = {
                        "model": project.settings.model,
                        "adapted_prompt": adapted_prompt,
                    }
                    await self._update_room_in_project(project, room)

                    start_time = time.monotonic()
                    result = None
                    elapsed_ms = 0
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
                        logger.error(f"Variation {idx} failed for room {room.id}: {e}")
                        variation.status = ItemStatus.FAILED
                        variation.error = str(e)
                        elapsed_ms = int((time.monotonic() - start_time) * 1000)

                    # Extract token usage from generation response
                    token_usage = None
                    if result and result.generation and result.generation.token_usage:
                        tu = result.generation.token_usage
                        token_usage = tu.get("total_tokens") if isinstance(tu, dict) else getattr(tu, "total_tokens", None)

                    await self._update_room_in_project(project, room)

                    yield {
                        "type": f"variation_{'completed' if variation.status == ItemStatus.COMPLETED else 'failed'}",
                        "room_id": room.id,
                        "variation_index": idx,
                        "image_url": variation.image_url,
                        "error": variation.error,
                        "elapsed_ms": elapsed_ms,
                        "tokens_used": token_usage,
                        "model": project.settings.model,
                    }

                # Mark any unprocessed variations (fewer prompts than variations) as failed
                for v in room.variations[len(adapted_prompts):]:
                    if v.status == ItemStatus.PENDING:
                        v.status = ItemStatus.FAILED
                        v.error = "No adapted prompt generated for this variation"

                any_completed = any(v.status == ItemStatus.COMPLETED for v in room.variations)
                room.status = ItemStatus.COMPLETED if any_completed else ItemStatus.FAILED
                await self._update_room_in_project(project, room)
                yield {"type": "room_completed", "room_id": room.id, "status": room.status}

            except Exception as e:
                logger.error(f"Room {room.id} failed: {e}")
                room.status = ItemStatus.FAILED
                room.error = str(e)
                await self._update_room_in_project(project, room)
                yield {"type": "room_failed", "room_id": room.id, "error": str(e)}

    async def process_single_variation(
        self,
        project: StagingProject,
        room: Room,
        variation: Variation,
        adapted_prompt: str,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Regenerate a single variation using the provided prompt. Yields SSE events.

        Failure / cancellation semantics (issue 002 of the single-variation-regen PRD):
        - On image-gen failure or pre-shield client disconnect, the variation is
          restored to its pre-regen visible state (status / image_url / error).
          The new `adapted_prompt` is preserved in `generation_metadata` so a
          subsequent retry can re-use it.
        - On success, the prior blob is deleted fire-and-forget.
        - The post-image-gen state write is wrapped in `asyncio.shield()` so a
          client disconnect during persistence cannot strand the variation in
          PROCESSING.

        Reconcile invariant: this method does NOT elevate `project.status` to
        PROCESSING. Reconcile only fires when `project.status == 'processing'`
        AND the doc is older than `STAGING_STALE_PROCESSING_MINUTES` (default
        5 min). Therefore reconcile cannot interfere with this regen mid-flight.
        That is what makes the explicit cancellation rollback above necessary —
        without it, a stranded variation would never recover.
        """
        variation_index = next(
            (i for i, v in enumerate(room.variations) if v.id == variation.id), None
        )
        if variation_index is None:
            logger.warning(f"Variation {variation.id} not found in room {room.id}, defaulting to index 0")
            variation_index = 0

        async with self.semaphore:
            # Capture pre-regen visible state so we can restore it on failure.
            prior_image_url = variation.image_url
            prior_status = variation.status
            prior_error = variation.error

            variation.status = ItemStatus.PROCESSING
            # Persist the attempted prompt BEFORE the image-gen call so a
            # subsequent retry can re-use it even if generation fails or the
            # worker dies mid-call. See PRD Implementation Decisions → Backend
            # (`adapted_prompt` persistence bullet).
            variation.generation_metadata = {
                "model": project.settings.model,
                "adapted_prompt": adapted_prompt,
            }
            await self._update_room_in_project(project, room)

            start_time = time.monotonic()
            result = None
            image_gen_error: Optional[Exception] = None
            shield_started = False

            try:
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

                    # Cancellable: a client disconnect mid-image-gen propagates
                    # CancelledError out of this await, past the inner Exception
                    # handler (CancelledError is BaseException, not Exception).
                    result = await self.image_pipeline.process_pipeline(
                        pipeline_request=pipeline_request,
                        azure_storage_service=self.blob_service,
                    )
                except Exception as e:
                    image_gen_error = e

                elapsed_ms = int((time.monotonic() - start_time) * 1000)

                # Shield the persist write so a client disconnect after image
                # generation but before persistence cannot strand the variation.
                shield_started = True
                event_type, regen_error_message = await asyncio.shield(
                    self._persist_single_variation_outcome(
                        project=project,
                        room=room,
                        variation=variation,
                        adapted_prompt=adapted_prompt,
                        prior_image_url=prior_image_url,
                        prior_status=prior_status,
                        prior_error=prior_error,
                        result=result,
                        image_gen_error=image_gen_error,
                        elapsed_ms=elapsed_ms,
                    )
                )
            except asyncio.CancelledError:
                if not shield_started:
                    # Pre-shield cancellation: image-gen was killed by the
                    # client disconnect before we built the persist payload.
                    # Persist a rollback (shielded) so the variation isn't
                    # stranded in PROCESSING (reconcile cannot recover us
                    # because we did not elevate `project.status`).
                    elapsed_ms = int((time.monotonic() - start_time) * 1000)
                    await asyncio.shield(
                        self._persist_single_variation_outcome(
                            project=project,
                            room=room,
                            variation=variation,
                            adapted_prompt=adapted_prompt,
                            prior_image_url=prior_image_url,
                            prior_status=prior_status,
                            prior_error=prior_error,
                            result=None,
                            image_gen_error=RuntimeError("Cancelled by client disconnect"),
                            elapsed_ms=elapsed_ms,
                        )
                    )
                # If shield_started is True the inner _persist_outcome is still
                # running to completion via asyncio.shield(); do not re-call it.
                raise

            token_usage = None
            if result and result.generation and result.generation.token_usage:
                tu = result.generation.token_usage
                token_usage = tu.get("total_tokens") if isinstance(tu, dict) else getattr(tu, "total_tokens", None)

            yield {
                "type": event_type,
                "room_id": room.id,
                "variation_index": variation_index,
                "image_url": variation.image_url,
                "error": regen_error_message,
                "elapsed_ms": elapsed_ms,
                "tokens_used": token_usage,
                "model": project.settings.model,
            }

    async def _persist_single_variation_outcome(
        self,
        *,
        project: StagingProject,
        room: Room,
        variation: Variation,
        adapted_prompt: str,
        prior_image_url: Optional[str],
        prior_status: str,
        prior_error: Optional[str],
        result,
        image_gen_error: Optional[Exception],
        elapsed_ms: int,
    ) -> tuple:
        """Persist the regen outcome atomically.

        Returns (event_type, regen_error_message):
        - ("variation_completed", None) on success — variation now points at the
          new image, COMPLETED, prior blob delete scheduled fire-and-forget.
        - ("variation_failed", "<reason>") on any failure (image-gen exception,
          empty saved_images, missing generation, cancellation). Variation is
          restored to its prior visible state (status / image_url / error) so
          the UI keeps showing the prior image. The new `adapted_prompt` stays
          in `generation_metadata` so a retry can re-use it.
        """
        saved_url = None
        if image_gen_error is None and result and result.generation and result.save:
            saved = result.save
            saved_url = (
                saved.saved_images[0].get("url") if saved.saved_images else None
            )
            if not saved_url:
                image_gen_error = RuntimeError("Save succeeded but no image URL returned")
        elif image_gen_error is None:
            image_gen_error = RuntimeError("Pipeline returned no generation result")

        if image_gen_error is None and saved_url:
            # Success path
            variation.image_url = saved_url
            variation.status = ItemStatus.COMPLETED
            variation.error = None
            variation.generation_metadata = {
                "model": project.settings.model,
                "adapted_prompt": adapted_prompt,
                "generation_time_ms": elapsed_ms,
            }
            await self._update_room_in_project(project, room)

            # Fire-and-forget delete of the prior blob (best-effort).
            if prior_image_url and prior_image_url != saved_url:
                self._schedule_blob_cleanup(prior_image_url)

            return ("variation_completed", None)

        # Failure path: restore prior visible state. Keep the new
        # adapted_prompt in generation_metadata for retry.
        regen_error_message = str(image_gen_error) if image_gen_error else "Unknown error"
        logger.error(
            f"Single variation regen failed for {variation.id}: {regen_error_message}"
        )
        variation.status = prior_status
        variation.image_url = prior_image_url
        variation.error = prior_error
        # NOTE: variation.generation_metadata already carries the new
        # adapted_prompt (persisted before the image-gen call). Leave it.
        await self._update_room_in_project(project, room)
        return ("variation_failed", regen_error_message)

    def _schedule_blob_cleanup(self, blob_url: str) -> None:
        """Schedule a fire-and-forget delete of the given blob URL.

        Best-effort: warnings are logged on failure; never raises. The task is
        held in `self._cleanup_tasks` to prevent garbage collection and to
        allow tests to await pending cleanups deterministically.
        """
        try:
            blob_name = self._extract_blob_name(blob_url)
        except Exception as e:
            logger.warning(f"Could not extract blob name from {blob_url}: {e}")
            return

        async def _delete():
            try:
                await asyncio.to_thread(
                    self.blob_service.delete_asset,
                    blob_name,
                    settings.AZURE_BLOB_IMAGE_CONTAINER,
                )
            except Exception as e:
                logger.warning(f"Failed to delete prior blob {blob_name}: {e}")

        try:
            task = asyncio.create_task(_delete())
        except RuntimeError:
            # No running loop — fall back to a synchronous best-effort delete.
            # This path is unreachable during request handling but keeps
            # cleanup robust if invoked from non-async contexts.
            try:
                self.blob_service.delete_asset(
                    blob_name, settings.AZURE_BLOB_IMAGE_CONTAINER,
                )
            except Exception as e:
                logger.warning(f"Failed to delete prior blob {blob_name}: {e}")
            return
        self._cleanup_tasks.add(task)
        task.add_done_callback(self._cleanup_tasks.discard)

    async def generate_project(self, project: StagingProject) -> AsyncGenerator[Dict[str, Any], None]:
        """Process all pending rooms in parallel. Yields SSE events as they arrive."""
        project.status = ProjectStatus.PROCESSING
        await self._persist_project_locked(project)

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
            await self._persist_project_locked(project)
            yield {"type": "project_completed", "status": project.status}
            return

        # _WORKER_DONE sentinel pushed in `finally` so we count worker completion
        # rather than semantic events — robust against task cancellation.
        _WORKER_DONE = object()
        event_queue: asyncio.Queue = asyncio.Queue()

        async def _room_worker(room: Room) -> None:
            """Process one room, pushing all its events to the shared queue."""
            try:
                async for event in self.process_room(project, room, brief_prompts=brief_prompts):
                    await event_queue.put(event)
            except BaseException as exc:
                # BaseException catches CancelledError too — prevents silent hangs.
                if not isinstance(exc, asyncio.CancelledError):
                    logger.error("Room %s failed: %s", room.id, exc)
                room.status = ItemStatus.FAILED
                room.error = str(exc) if not isinstance(exc, asyncio.CancelledError) else "cancelled"
                await self._update_room_in_project(project, room)
                await event_queue.put({"type": "room_failed", "room_id": room.id, "error": str(exc)})
            finally:
                await event_queue.put(_WORKER_DONE)

        # Launch all rooms concurrently; the semaphore inside process_room gates real concurrency
        tasks = [asyncio.create_task(_room_worker(room)) for room in pending_rooms]
        try:
            # Drain the queue, yielding events as they arrive, until every worker signals done
            workers_done = 0
            total_workers = len(pending_rooms)
            while workers_done < total_workers:
                event = await event_queue.get()
                if event is _WORKER_DONE:
                    workers_done += 1
                    continue
                yield event
        finally:
            # Ensure tasks are cancelled if the generator is closed early (consumer disconnect)
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

        any_room_completed = any(r.status == ItemStatus.COMPLETED for r in project.rooms)
        project.status = ProjectStatus.COMPLETED if any_room_completed else ProjectStatus.FAILED
        await self._persist_project_locked(project)
        yield {"type": "project_completed", "status": project.status}

    @staticmethod
    def _extract_blob_name(blob_url: str) -> str:
        """Extract blob name from a full Azure Blob Storage URL.

        Input:  https://account.blob.core.windows.net/images/staging/proj/originals/file.png
        Output: staging/proj/originals/file.png
        """
        # Find the container name segment and return everything after it
        parts = blob_url.split("/")
        try:
            # URL format: https://account.blob.core.windows.net/{container}/{blob_path...}
            # The container is at index 3 (after protocol, empty, host)
            net_idx = next(i for i, p in enumerate(parts) if p.endswith(".net"))
            return "/".join(parts[net_idx + 2:])  # skip container name
        except (StopIteration, IndexError):
            # Fallback: assume the last 2+ segments after /images/ or /videos/
            for container in ("images", "videos"):
                if f"/{container}/" in blob_url:
                    return blob_url.split(f"/{container}/")[1]
            return "/".join(parts[-2:])

    @staticmethod
    def _serialize_project(project: StagingProject) -> dict:
        """Serialize project to a JSON-safe dict (datetime → ISO string)."""
        return json.loads(project.json())

    def _get_project_lock(self, project_id: str) -> asyncio.Lock:
        """Return the per-project asyncio.Lock, delegating to the module-level
        registry so locks are shared across all StagingPipeline instances in
        this process. See `staging_pipeline._PROJECT_LOCKS` for rationale.
        """
        return _get_project_lock(project_id)

    async def _persist_project_locked(self, project: StagingProject) -> Dict[str, Any]:
        """Persist the full project document under the per-project lock.

        Wraps `storage_service.update_project` (which performs a Cosmos
        read-modify-write) inside the lock so concurrent persists for the
        same project are serialized. The blocking Cosmos SDK call is
        dispatched onto the default thread pool via `asyncio.to_thread`
        so the lock has a real await window — ensuring concurrent persists
        on the same project ID actually serialize at the storage boundary
        rather than blocking the event loop.

        The lock is per-project, so persists for *different* project IDs
        run in parallel.

        Cancellation safety: once the thread starts, Python threads cannot
        be cancelled. If the caller's task is cancelled mid-`to_thread`,
        we must keep the lock held until the thread truly completes —
        otherwise the next writer could enter the lock and race the still-
        in-flight write. We use `asyncio.shield` to detach the inner
        Future from caller-cancellation and a re-entry loop to wait for
        completion before re-raising the original cancellation.
        """
        async with self._get_project_lock(project.id):
            fut = asyncio.ensure_future(
                asyncio.to_thread(
                    self.storage_service.update_project,
                    project.id,
                    self._serialize_project(project),
                )
            )
            try:
                return await asyncio.shield(fut)
            except asyncio.CancelledError:
                while not fut.done():
                    try:
                        await asyncio.shield(fut)
                    except asyncio.CancelledError:
                        # Cascading cancels are absorbed; we keep waiting
                        # so the lock isn't released early.
                        continue
                raise

    async def _update_room_in_project(self, project: StagingProject, room: Room):
        """Persist room updates to Cosmos DB under the per-project lock."""
        for i, r in enumerate(project.rooms):
            if r.id == room.id:
                project.rooms[i] = room
                break
        try:
            await self._persist_project_locked(project)
        except Exception as e:
            logger.error(f"Failed to persist room update: {e}")