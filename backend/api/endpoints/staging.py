"""FastAPI endpoints for virtual staging projects."""
import json
import logging
import time
import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from backend.core.azure_storage import AzureBlobStorageService
from backend.core.brief_resolver import migrate_legacy_plant_palette
from backend.core.config import settings
from backend.core.project_status import ProjectStatusCalculator
from backend.core.staging_pipeline import _get_project_lock
from backend.core.staging_reconcile import reconcile_project
from backend.core.staging_storage import StagingStorageService
from backend.models.design_brief import (
    ChatRequest,
    ChatResponse,
    DesignBrief,
    GenerateBriefRequest,
    ImageAnalysis,
)
from backend.models.staging import (
    CreateProjectRequest,
    ItemStatus,
    ProjectListResponse,
    ProjectResponse,
    Room,
    StagingProject,
    UploadRoomsResponse,
    Variation,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _migrate_design_brief_in_place(project: dict) -> bool:
    """Apply ``migrate_legacy_plant_palette`` to ``project['design_brief']`` if
    present. Returns True if the project was mutated (legacy → generic shape),
    False otherwise. Idempotent: already-migrated briefs are left untouched
    via the ``migrated is original`` short-circuit in the resolver.
    """
    brief = project.get("design_brief")
    if not isinstance(brief, dict):
        return False
    migrated = migrate_legacy_plant_palette(brief)
    if migrated is brief:
        return False
    project["design_brief"] = migrated
    return True


def get_staging_storage() -> StagingStorageService:
    return StagingStorageService()


def get_image_analyzer():
    from backend.core import async_llm_client
    from backend.core.analyze import ImageAnalyzer
    return ImageAnalyzer(
        openai_client=None,
        model=settings.LLM_DEPLOYMENT,
        async_openai_client=async_llm_client,
    )


def get_staging_pipeline():
    from backend.core import async_llm_client
    from backend.core.analyze import ImageAnalyzer
    from backend.core.image_pipeline import ImagePipelineService
    from backend.core.staging_pipeline import StagingPipeline

    analyzer = ImageAnalyzer(
        openai_client=None,
        model=settings.LLM_DEPLOYMENT,
        async_openai_client=async_llm_client,
    )
    image_pipeline = ImagePipelineService()
    blob_service = AzureBlobStorageService()
    storage = StagingStorageService()

    return StagingPipeline(
        async_llm_client=async_llm_client,
        llm_deployment=settings.LLM_DEPLOYMENT,
        image_analyzer=analyzer,
        image_pipeline=image_pipeline,
        storage_service=storage,
        blob_service=blob_service,
    )


@router.post("/projects", response_model=ProjectResponse, status_code=201)
async def create_project(
    request: CreateProjectRequest,
    storage: StagingStorageService = Depends(get_staging_storage),
):
    project_data = {
        "id": str(uuid.uuid4()),
        "name": request.name,
        "prompt": request.prompt,
        "status": "uploading",
        "rooms": [],
        "settings": request.settings.dict(),
        "folder_path": f"staging/{request.name.lower().replace(' ', '-')}",
    }
    created = storage.create_project(project_data)
    return ProjectResponse(project=StagingProject(**{k: v for k, v in created.items() if k != "doc_type" and not k.startswith("_")}))


@router.get("/projects", response_model=ProjectListResponse)
async def list_projects(
    limit: int = 50,
    offset: int = 0,
    storage: StagingStorageService = Depends(get_staging_storage),
):
    projects_raw = storage.list_projects(limit=limit, offset=offset)
    total = storage.count_projects()
    projects = []
    for p in projects_raw:
        # Combine reconcile + legacy-brief-migration into a single optional
        # writeback. Either pass alone may mutate; if both pass mutate we
        # only persist once. ``or`` short-circuits, but we want both calls
        # to run, so we OR the results explicitly.
        reconciled = reconcile_project(p)
        migrated = _migrate_design_brief_in_place(p)
        if reconciled or migrated:
            try:
                storage.update_project(p["id"], p)
            except Exception as e:
                logger.warning("Failed to persist reconciled/migrated project %s: %s", p.get("id"), e)
        clean = {k: v for k, v in p.items() if k != "doc_type" and not k.startswith("_")}
        projects.append(StagingProject(**clean))
    return ProjectListResponse(projects=projects, total=total)


@router.get("/projects/{project_id}", response_model=ProjectResponse)
async def get_project(
    project_id: str,
    storage: StagingStorageService = Depends(get_staging_storage),
):
    project = storage.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Auto-heal stale processing states + opportunistically migrate legacy
    # plant_palette → object_palette on read (single combined writeback).
    reconciled = reconcile_project(project)
    migrated = _migrate_design_brief_in_place(project)
    if reconciled or migrated:
        try:
            storage.update_project(project_id, project)
        except Exception as e:
            logger.warning("Failed to persist reconciled/migrated project %s: %s", project_id, e)

    clean = {k: v for k, v in project.items() if k != "doc_type" and not k.startswith("_")}
    return ProjectResponse(project=StagingProject(**clean))


@router.delete("/projects/{project_id}")
async def delete_project(
    project_id: str,
    storage: StagingStorageService = Depends(get_staging_storage),
):
    """Delete a project and all associated blob storage artifacts."""
    # Get project first to find blob paths
    project_data = storage.get_project(project_id)
    if not project_data:
        raise HTTPException(status_code=404, detail="Project not found")

    # Delete all blobs under staging/{project_id}/
    deleted_blobs = 0
    try:
        blob_service = AzureBlobStorageService()
        container_client = blob_service.blob_service_client.get_container_client(settings.AZURE_BLOB_IMAGE_CONTAINER)
        prefix = f"staging/{project_id}/"
        blobs = container_client.list_blobs(name_starts_with=prefix)
        for blob in blobs:
            try:
                container_client.delete_blob(blob.name)
                deleted_blobs += 1
            except Exception as e:
                logger.warning(f"Failed to delete blob {blob.name}: {e}")
    except Exception as e:
        logger.warning(f"Failed to clean up blobs for project {project_id}: {e}")

    # Delete Cosmos document
    if not storage.delete_project(project_id):
        raise HTTPException(status_code=404, detail="Project not found")

    return {"status": "deleted", "project_id": project_id, "blobs_deleted": deleted_blobs}


@router.post("/projects/{project_id}/reset", response_model=ProjectResponse)
async def reset_project(
    project_id: str,
    storage: StagingStorageService = Depends(get_staging_storage),
):
    """Force-reset a stuck project: all processing/failed items → pending."""
    project = storage.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    reconcile_project(project, force=True)
    try:
        storage.update_project(project_id, project)
    except Exception as e:
        logger.error("Failed to persist reset for project %s: %s", project_id, e)
        raise HTTPException(status_code=500, detail="Failed to reset project")

    clean = {k: v for k, v in project.items() if k != "doc_type" and not k.startswith("_")}
    return ProjectResponse(project=StagingProject(**clean))


@router.post("/projects/{project_id}/rooms", response_model=UploadRoomsResponse)
async def upload_rooms(
    project_id: str,
    images: List[UploadFile] = File(...),
    labels: Optional[str] = Form(None),
    storage: StagingStorageService = Depends(get_staging_storage),
):
    project_data = storage.get_project(project_id)
    if not project_data:
        raise HTTPException(status_code=404, detail="Project not found")

    if settings.STAGING_MAX_ROOMS_PER_PROJECT > 0 and len(images) > settings.STAGING_MAX_ROOMS_PER_PROJECT:
        raise HTTPException(status_code=400, detail=f"Maximum {settings.STAGING_MAX_ROOMS_PER_PROJECT} rooms per project")

    label_list = json.loads(labels) if labels else []
    blob_service = AzureBlobStorageService()
    new_rooms = []

    for idx, image_file in enumerate(images):
        room_id = str(uuid.uuid4())
        label = label_list[idx] if idx < len(label_list) else f"Room {len(project_data.get('rooms', [])) + idx + 1}"

        upload_result = await blob_service.upload_asset(
            file=image_file,
            asset_type="image",
            folder_path=f"staging/{project_id}/originals",
        )

        n_vars = project_data.get("settings", {}).get("variations_per_room", 5)
        variations = [Variation(id=str(uuid.uuid4())).dict() for _ in range(n_vars)]

        room = Room(
            id=room_id,
            label=label,
            original_image_url=upload_result["url"],
            variations=variations,
        )
        new_rooms.append(room)

    existing_rooms = project_data.get("rooms", [])
    existing_rooms.extend([r.dict() for r in new_rooms])

    # Transition status from "uploading" to "pending" now that rooms exist
    updates = {"rooms": existing_rooms}
    if project_data.get("status") == "uploading":
        updates["status"] = "pending"

    storage.update_project(project_id, updates)

    return UploadRoomsResponse(project_id=project_id, rooms_added=len(new_rooms), rooms=new_rooms)


def _sse_event(event_type: str, data: dict) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


def _log_regen_event(
    *,
    event: str,
    project_id: str,
    room_id: str,
    variation_id: str,
    strategy: str,
    effective_strategy: str,
    elapsed_ms: Optional[int] = None,
    tokens_used: Optional[int] = None,
) -> None:
    """Emit an operator-facing structured log line for a regen lifecycle event.

    Issue 008 of the single-variation-regeneration PRD: log analytics must be
    able to answer questions about regen usage rates, success rates, fallback
    frequency, and elapsed time without spelunking through unstructured logs.

    Field contract (PRD § Implementation Decisions → Backend, structured-
    logging bullet):

    - ``project_id``, ``room_id``, ``variation_id`` — entity identifiers.
      All UUIDs; no PII.
    - ``strategy`` — the *requested* strategy ("retry" or "fresh").
    - ``effective_strategy`` — the strategy *actually used* after any fallback
      (e.g., retry that fell back to fresh has ``effective_strategy="fresh"``).
    - ``elapsed_ms`` — wall-clock milliseconds from regen acceptance to the
      terminal event. Sourced from ``time.monotonic()`` deltas in the
      endpoint, NOT from the pipeline's image-gen-only ``elapsed_ms``: log
      analytics for "how long does a regen take" needs the operator-facing
      total (prompt-gen + image-gen + persistence), not just the image-gen
      slice. Included on ``completed`` / ``failed`` lines; omitted on
      ``started`` / ``fallback_to_fresh``.
    - ``tokens_used`` — image-gen token usage from the pipeline's terminal
      event (``None`` for failed image-gen and any path that never reaches
      the pipeline). LLM prompt-generation tokens are NOT included; the
      field is image-gen-only. ALWAYS carried on ``completed`` / ``failed``
      (even when ``None``, so consumers distinguish "no image-gen happened"
      from "field missing"). Omitted on ``started`` / ``fallback_to_fresh``.

    Fields are projected onto the ``LogRecord`` via ``extra=`` (for
    structured-log aggregators like App Insights) AND mirrored as
    ``key=value`` pairs in the human-readable message (mirroring the
    ``backend.core.retry`` pattern). Promotion to a dedicated metrics sink
    is explicitly out of scope; see PRD § Out of Scope.
    """
    extra = {
        "event": event,
        "project_id": project_id,
        "room_id": room_id,
        "variation_id": variation_id,
        "strategy": strategy,
        "effective_strategy": effective_strategy,
    }
    parts = [
        f"event={event}",
        f"project_id={project_id}",
        f"room_id={room_id}",
        f"variation_id={variation_id}",
        f"strategy={strategy}",
        f"effective_strategy={effective_strategy}",
    ]
    # ``completed`` and ``failed`` are the terminal lifecycle events that
    # carry timing + token metrics. ``started`` and ``fallback_to_fresh``
    # are mid-flight markers and intentionally omit those fields.
    is_terminal = event.endswith(".completed") or event.endswith(".failed")
    if is_terminal:
        extra["elapsed_ms"] = elapsed_ms
        parts.append(f"elapsed_ms={elapsed_ms}")
        # Carried even when None so consumers can distinguish
        # "no-LLM-call flow / failed before token tally" from "field missing".
        extra["tokens_used"] = tokens_used
        parts.append(f"tokens_used={tokens_used}")

    logger.info(" ".join(parts), extra=extra)


@router.post("/projects/{project_id}/generate")
async def generate_project(
    project_id: str,
    storage: StagingStorageService = Depends(get_staging_storage),
    pipeline=Depends(get_staging_pipeline),
):
    project_data = storage.get_project(project_id)
    if not project_data:
        raise HTTPException(status_code=404, detail="Project not found")

    # Auto-heal stale processing before starting
    if reconcile_project(project_data):
        try:
            storage.update_project(project_id, project_data)
        except Exception as e:
            logger.warning("Failed to persist reconciled project %s: %s", project_id, e)

    clean = {k: v for k, v in project_data.items() if k != "doc_type" and not k.startswith("_")}
    project = StagingProject(**clean)

    if not project.rooms:
        raise HTTPException(status_code=400, detail="No rooms uploaded yet")

    async def event_stream():
        async for event in pipeline.generate_project(project):
            yield _sse_event(event["type"], event)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/projects/{project_id}/rooms/{room_id}/regenerate")
async def regenerate_room(
    project_id: str,
    room_id: str,
    storage: StagingStorageService = Depends(get_staging_storage),
    pipeline=Depends(get_staging_pipeline),
):
    project_data = storage.get_project(project_id)
    if not project_data:
        raise HTTPException(status_code=404, detail="Project not found")

    clean = {k: v for k, v in project_data.items() if k != "doc_type" and not k.startswith("_")}
    project = StagingProject(**clean)

    room = next((r for r in project.rooms if r.id == room_id), None)
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")

    room.status = ItemStatus.PENDING
    for v in room.variations:
        v.status = ItemStatus.PENDING
        v.image_url = None
        v.error = None

    # Compute brief prompts if design brief exists
    brief_prompts = None
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
                n_variations=project.settings.variations_per_room,
            )

    async def event_stream():
        try:
            async for event in pipeline.process_room(project, room, brief_prompts=brief_prompts):
                yield _sse_event(event["type"], event)
        finally:
            # Recalculate project-level status after room regeneration via
            # the single ProjectStatusCalculator helper (issue 001 of the
            # projects-page-improvements PRD). Pre-fix this block had its
            # own inline any_processing/any_completed branch that drifted
            # from the parallel branch in regenerate_variation; both now
            # delegate to the calculator. We persist on every transition
            # so the badge stays correct after a refresh, including the
            # PENDING case (work outstanding) that the previous code
            # silently skipped.
            #
            # Wrapped in the per-project lock to serialize with the
            # pipeline's `_persist_project_locked` writes from any
            # concurrent room workers (e.g. a regenerate_room finishing
            # while a separate regenerate_variation worker is mid-write).
            # Without this lock the read-modify-write race could clobber
            # fresher per-room state because Cosmos writes are full-doc
            # replacements.
            final_status = project.status
            async with _get_project_lock(project_id):
                fresh = storage.get_project(project_id)
                if fresh:
                    clean_fresh = {k: v for k, v in fresh.items() if k != "doc_type" and not k.startswith("_")}
                    fresh_project = StagingProject(**clean_fresh)
                    fresh_project.status = ProjectStatusCalculator.compute_status(fresh_project.rooms)
                    storage.update_project(project_id, json.loads(fresh_project.json()))
                    final_status = fresh_project.status
        # Emit the freshly-computed status, not the stale local
        # `project.status` snapshot that pre-dates the worker's writes.
        yield _sse_event("project_completed", {"status": final_status})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


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

    # Determine the prompt to use, and capture the previously-rejected
    # prompt regardless of strategy. Issue 003 of single-variation-regen
    # PRD: on the fresh path, the rejected prior prompt is threaded down
    # to ``brief_to_prompts`` / ``adapt_prompt`` as ``rejected_prompt`` so
    # the new generation diverges from the rejected aesthetic.
    adapted_prompt = None
    prior_adapted_prompt: Optional[str] = None
    fallback_to_fresh = False

    if variation.generation_metadata and isinstance(variation.generation_metadata, dict):
        prior_adapted_prompt = variation.generation_metadata.get("adapted_prompt")
    elif hasattr(variation.generation_metadata, "adapted_prompt"):
        prior_adapted_prompt = variation.generation_metadata.adapted_prompt

    if strategy == "retry":
        adapted_prompt = prior_adapted_prompt
        if not adapted_prompt:
            fallback_to_fresh = True

    # Issue 008: ``effective_strategy`` reflects the strategy that will
    # *actually* be used: a retry that lacks a prior prompt is silently a
    # fresh, so its effective_strategy is "fresh" even though the requested
    # strategy is "retry". Computed pre-preflight so it's available to the
    # ``started`` log line below and to ``event_stream`` via closure.
    effective_strategy = "fresh" if (strategy == "fresh" or fallback_to_fresh) else "retry"
    # Wall-clock anchor for ``elapsed_ms`` reported on terminal log lines.
    # Captured BEFORE preflight so the operator-facing total includes the
    # preflight-persist round-trip. Started log fires only after a
    # successful preflight (see below) so a write-failure path does not
    # leave a stranded ``started`` line in the logs.
    regen_start_time = time.monotonic()

    # Preflight: mark variation/room as PROCESSING so concurrent regen requests
    # see the 409 mutex on `variation.status == PROCESSING`. Deliberately do
    # NOT clear `variation.image_url` — the pipeline captures it as
    # `prior_image_url` for failure rollback and old-blob cleanup. See
    # PRD: Implementation Decisions → Backend (`process_single_variation`
    # rollback semantics) and issue 002 of the single-variation regen PRD.
    variation.status = ItemStatus.PROCESSING
    variation.error = None

    # Update room status to processing
    room.status = ItemStatus.PROCESSING
    storage.update_project(project_id, json.loads(project.json()))

    # Issue 008 of the single-variation-regeneration PRD: emit the
    # ``started`` log line AFTER the preflight write succeeds. If the
    # preflight write raises, we never reach this line — that's intentional,
    # since a regen that never entered durable processing should not appear
    # in operator dashboards as having "started".
    _log_regen_event(
        event="staging.variation_regen.started",
        project_id=project_id,
        room_id=room_id,
        variation_id=variation_id,
        strategy=strategy,
        effective_strategy=effective_strategy,
    )

    async def event_stream():
        nonlocal adapted_prompt
        final_status = "completed"
        # Track the terminal pipeline event so the post-loop log line knows
        # whether the regen completed or failed. ``elapsed_ms`` is captured
        # at terminal-event time as wall-clock from ``regen_start_time`` —
        # the duck-checked rationale for wall-clock over the pipeline's
        # ``elapsed_ms`` field is in ``_log_regen_event``'s docstring.
        # ``tokens_used`` IS sourced from the pipeline event since that's
        # an image-gen-internal value the endpoint can't compute itself.
        # All three are captured BEFORE the corresponding ``yield`` so a
        # client disconnect after the SSE event is sent (but before the
        # generator resumes) still leaves ``finally`` with enough state to
        # emit the matching ``completed`` / ``failed`` log line.
        terminal_event_type: Optional[str] = None
        terminal_elapsed_ms: Optional[int] = None
        terminal_tokens_used: Optional[int] = None

        try:
            if fallback_to_fresh:
                # Issue 004 of single-variation-regeneration PRD: surface
                # the silent retry→fresh fallback as a dedicated SSE event
                # so the frontend can toast "no previous prompt found —
                # generating a fresh take instead." Continues normally to a
                # terminal ``project_completed`` event — this is a
                # notification, not a cancellation.
                #
                # Issue 008: pair the SSE event with a structured log line
                # so log analytics can count silent retry→fresh fallbacks
                # without parsing the SSE stream. Log line emits BEFORE
                # the yield so a client disconnect mid-yield doesn't
                # silently drop the operator log.
                _log_regen_event(
                    event="staging.variation_regen.fallback_to_fresh",
                    project_id=project_id,
                    room_id=room_id,
                    variation_id=variation_id,
                    strategy=strategy,
                    effective_strategy=effective_strategy,
                )
                yield _sse_event("variation_fallback", {
                    "room_id": room_id,
                    "variation_id": variation_id,
                    "reason": "no_prior_prompt",
                })

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
                            rejected_prompt=prior_adapted_prompt,
                        )
                        if room.id in brief_prompts and brief_prompts[room.id]:
                            adapted_prompt = brief_prompts[room.id][0]

                if not adapted_prompt:
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
                        rejected_prompt=prior_adapted_prompt,
                    )
                    adapted_prompt = prompts[0]

            if not adapted_prompt:
                # Issue 008: prompt-generation never produced a usable
                # prompt — this is a terminal failure of the regen flow.
                # No pipeline event will fire. Set terminal state BEFORE
                # the SSE yield so a client disconnect mid-yield still
                # leaves enough state in ``finally`` to log ``failed``.
                terminal_event_type = "variation_failed"
                terminal_elapsed_ms = int((time.monotonic() - regen_start_time) * 1000)
                terminal_tokens_used = None
                yield _sse_event("error", {"error": "Failed to generate or retrieve adapted prompt"})
                return

            async for event in pipeline.process_single_variation(
                project, room, variation, adapted_prompt
            ):
                # Issue 008: capture the pipeline's terminal event BEFORE
                # the SSE yield. The pipeline contract is one terminal
                # event per call; the last one observed wins if (against
                # contract) more than one fires. ``elapsed_ms`` is wall-
                # clock from regen acceptance to terminal-event observation
                # — see ``_log_regen_event``'s docstring for the wall-clock
                # vs pipeline-elapsed_ms rationale. ``tokens_used`` is taken
                # from the pipeline event (image-gen-internal value).
                if event.get("type") in ("variation_completed", "variation_failed"):
                    terminal_event_type = event["type"]
                    terminal_elapsed_ms = int((time.monotonic() - regen_start_time) * 1000)
                    terminal_tokens_used = event.get("tokens_used")
                yield _sse_event(event["type"], event)

        except Exception:
            # Issue 008 defense-in-depth: if the pipeline (or any of the
            # prompt-generation helpers above) raises an exception instead
            # of yielding a terminal SSE event, the regen still terminated
            # — operator analytics must see a ``failed`` log line, not a
            # stranded ``started`` with no terminal partner. Only synthesize
            # terminal state if a real terminal event hasn't already been
            # observed (covers the case where the pipeline yielded
            # ``variation_failed`` and then raised on a subsequent yield —
            # we keep the real terminal_elapsed_ms from the pipeline event).
            if terminal_event_type is None:
                terminal_event_type = "variation_failed"
                terminal_elapsed_ms = int((time.monotonic() - regen_start_time) * 1000)
                terminal_tokens_used = None
            raise

        finally:
            # Recalculate room and project status. Wrapped in the per-
            # project lock so this read-modify-write serializes with any
            # concurrent room-worker `_persist_project_locked` writes —
            # without the lock, two writers can interleave and the
            # finalizer's full-doc replace would clobber fresher state.
            async with _get_project_lock(project_id):
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
                    # Issue 001 (projects-page-improvements PRD): single
                    # source of truth for project-level status via
                    # ProjectStatusCalculator. Replaces an inline branch
                    # that only updated project.status when ALL rooms were
                    # terminal — meaning the persisted status was stale on
                    # mixed projects (one room mid-regen with peers still
                    # pending). The calculator unconditionally returns the
                    # truthful status (PENDING when work is outstanding)
                    # so the persisted value matches the badge after refresh.
                    fresh_project.status = ProjectStatusCalculator.compute_status(fresh_project.rooms)
                    storage.update_project(project_id, json.loads(fresh_project.json()))
                    final_status = fresh_project.status

            # Issue 008: emit the terminal log line ONCE we've observed the
            # pipeline's terminal SSE event, OR the prompt-gen-failure path
            # set ``terminal_event_type = "variation_failed"`` directly, OR
            # the ``except Exception`` defense-in-depth branch synthesized
            # a failure for an unhandled raise. Lives inside ``finally`` so
            # it fires even when the client disconnects mid-stream — a
            # regen that the worker finished server-side still gets its
            # operator log line. ``elapsed_ms`` is wall-clock from regen
            # acceptance to terminal-event observation (NOT the pipeline's
            # image-gen-only ``elapsed_ms``); see ``_log_regen_event``'s
            # docstring for the rationale.
            if terminal_event_type is not None:
                terminal_log_event = (
                    "staging.variation_regen.completed"
                    if terminal_event_type == "variation_completed"
                    else "staging.variation_regen.failed"
                )
                _log_regen_event(
                    event=terminal_log_event,
                    project_id=project_id,
                    room_id=room_id,
                    variation_id=variation_id,
                    strategy=strategy,
                    effective_strategy=effective_strategy,
                    elapsed_ms=terminal_elapsed_ms,
                    tokens_used=terminal_tokens_used,
                )

        yield _sse_event("project_completed", {"status": final_status})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/projects/{project_id}/analyze")
async def analyze_project_images(
    project_id: str,
    storage: StagingStorageService = Depends(get_staging_storage),
):
    """Analyze all uploaded images in the project."""
    import asyncio
    import base64

    project_data = storage.get_project(project_id)
    if not project_data:
        raise HTTPException(status_code=404, detail="Project not found")

    rooms = project_data.get("rooms", [])
    if not rooms:
        raise HTTPException(status_code=400, detail="No images uploaded yet")

    analyzer = get_image_analyzer()
    blob_service = AzureBlobStorageService()

    system_msg = (
        "Describe this scene in detail for a design assistant. Include: "
        "existing structures, plants, materials, colors, spatial layout, "
        "and identifiable zones where items could be added. "
        "Return JSON with keys: description (string), features (list of strings), "
        "zones (list of strings identifying areas suitable for placing new objects)."
    )

    async def analyze_one(room: dict) -> dict:
        url = room["original_image_url"]
        # Extract blob name: everything after the container name in the URL
        for container in ("images", "videos"):
            if f"/{container}/" in url:
                blob_name = url.split(f"/{container}/")[1]
                break
        else:
            blob_name = "/".join(url.split("/")[-2:])
        image_content, _ = blob_service.get_asset_content(
            blob_name=blob_name,
            container_name=settings.AZURE_BLOB_IMAGE_CONTAINER,
        )
        if image_content is None:
            raise RuntimeError(f"Image not found in blob storage: {url}")
        image_b64 = base64.b64encode(image_content).decode("utf-8")
        result = await analyzer.async_image_chat(image_base64=image_b64, system_message=system_msg)
        return {
            "room_id": room["id"],
            "description": result.get("description", ""),
            "features": result.get("features", []),
            "zones": result.get("zones", []),
        }

    analyses = await asyncio.gather(*[analyze_one(r) for r in rooms], return_exceptions=True)
    valid_analyses = [a for a in analyses if isinstance(a, dict)]
    failed_count = len(analyses) - len(valid_analyses)
    if failed_count > 0:
        logger.warning(f"analyze_project_images: {failed_count}/{len(analyses)} image analyses failed")

    storage.update_project(project_id, {"analyses": valid_analyses})

    return {"analyses": valid_analyses, "failed_count": failed_count if failed_count > 0 else 0}


@router.post("/projects/{project_id}/chat", response_model=ChatResponse)
async def chat_with_project(
    project_id: str,
    request: ChatRequest,
    storage: StagingStorageService = Depends(get_staging_storage),
):
    """Conversational AI endpoint for the Design Session."""
    from backend.core import async_llm_client
    from backend.core.design_chat import DesignChatService

    project_data = storage.get_project(project_id)
    if not project_data:
        raise HTTPException(status_code=404, detail="Project not found")

    raw_analyses = project_data.get("analyses", [])
    analyses = [ImageAnalysis(**a) for a in raw_analyses]

    service = DesignChatService(
        async_llm_client=async_llm_client,
        llm_deployment=settings.LLM_DEPLOYMENT,
        image_analyses=analyses,
    )

    return await service.chat(
        message=request.message,
        conversation_history=request.conversation_history,
        focused_image_id=request.focused_image_id,
    )


@router.post("/projects/{project_id}/brief")
async def generate_brief(
    project_id: str,
    request: GenerateBriefRequest = None,
    storage: StagingStorageService = Depends(get_staging_storage),
):
    """Generate a structured Design Brief from the conversation."""
    from backend.core import async_llm_client
    from backend.core.brief_generator import BriefGeneratorService

    project_data = storage.get_project(project_id)
    if not project_data:
        raise HTTPException(status_code=404, detail="Project not found")

    raw_analyses = project_data.get("analyses", [])
    analyses = [ImageAnalysis(**a) for a in raw_analyses]

    conversation_history = []
    previous_brief = None
    if request and request.conversation_history:
        conversation_history = request.conversation_history
    if request and request.previous_brief is not None:
        previous_brief = request.previous_brief

    service = BriefGeneratorService(
        async_llm_client=async_llm_client,
        llm_deployment=settings.LLM_DEPLOYMENT,
    )

    brief, reconcile_summary = await service.generate_brief(
        conversation_history=conversation_history,
        image_analyses=analyses,
        previous_brief=previous_brief,
    )

    brief_dict = brief.dict()
    storage.update_project(project_id, {"design_brief": brief_dict})

    return {
        "brief": brief_dict,
        "reconciliation_summary": reconcile_summary.dict(),
    }


@router.put("/projects/{project_id}/brief")
async def update_brief(
    project_id: str,
    brief: DesignBrief,
    storage: StagingStorageService = Depends(get_staging_storage),
):
    """Save user edits to the Design Brief."""
    project_data = storage.get_project(project_id)
    if not project_data:
        raise HTTPException(status_code=404, detail="Project not found")

    brief_dict = brief.dict()
    storage.update_project(project_id, {"design_brief": brief_dict})

    return {"brief": brief_dict}