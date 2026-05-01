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
from backend.core.prompt_composer import PromptComposer
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
    EditPromptRequest,
    ItemStatus,
    ProjectListResponse,
    ProjectResponse,
    Room,
    StagingProject,
    UpdateProjectRequest,
    UpdateRoomRequest,
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


@router.patch("/projects/{project_id}/rooms/{room_id}", response_model=ProjectResponse)
async def update_room(
    project_id: str,
    room_id: str,
    body: UpdateRoomRequest,
    storage: StagingStorageService = Depends(get_staging_storage),
):
    """Partial-update editable Room fields (currently just
    ``prompt_addendum``).

    Issue 003 of the projects-page-improvements PRD. Per the PRD's
    Further Notes the implementer chose a dedicated room-scoped endpoint
    over extending ``PATCH /projects/{id}`` because:

    - ``PATCH /projects/{id}`` doesn't exist yet (slice 002).
    - The URL semantics match the resource being edited.
    - Keeps room-scoped concerns out of the project-level PATCH.

    The endpoint:

    - Updates only ``room.prompt_addendum`` on the target room.
    - Normalizes ``""``, ``None``, and whitespace-only to ``None`` so the
      persisted shape stays consistent with the composer's "absent" rule.
    - Leaves variations / status / image_url / label untouched.
    - Leaves all sibling rooms byte-for-byte unchanged.
    - Leaves project-level status / prompt / settings untouched.
    - Does NOT trigger any generation.

    The read-modify-write is wrapped in the per-project asyncio lock from
    ``staging_pipeline._get_project_lock`` so concurrent edits across
    different rooms (or a parallel regen finalizer) cannot clobber each
    other through Cosmos's full-doc replacement semantics. This mirrors
    the protection already in place on the regen-finalizer write paths.
    """
    async with _get_project_lock(project_id):
        project_data = storage.get_project(project_id)
        if not project_data:
            raise HTTPException(status_code=404, detail="Project not found")

        rooms = project_data.get("rooms", [])
        room = next((r for r in rooms if r.get("id") == room_id), None)
        if not room:
            raise HTTPException(status_code=404, detail="Room not found")

        # Normalize empty / whitespace-only addendum to None so the
        # persisted shape stays clean. Mirrors ``PromptComposer``'s
        # "absent if it strips to empty" treatment.
        addendum = body.prompt_addendum
        if isinstance(addendum, str) and not addendum.strip():
            addendum = None
        elif isinstance(addendum, str):
            addendum = addendum.strip()
        room["prompt_addendum"] = addendum

        storage.update_project(project_id, project_data)

    clean = {k: v for k, v in project_data.items() if k != "doc_type" and not k.startswith("_")}
    return ProjectResponse(project=StagingProject(**clean))


@router.patch("/projects/{project_id}", response_model=ProjectResponse)
async def update_project(
    project_id: str,
    body: UpdateProjectRequest,
    storage: StagingStorageService = Depends(get_staging_storage),
):
    """Partial-update editable project-level fields (``name``,
    ``prompt``, ``settings``, ``design_brief``).

    Issue 002 of the projects-page-improvements PRD. Per the PRD's
    Solution → 2 paragraph, saved changes apply only to FUTURE
    generations — every existing variation and its prompt stays exactly
    as it was. The endpoint:

    - Updates ONLY the fields the client actually sent (uses
      ``__fields_set__`` to distinguish "absent" from "explicit null").
    - For ``settings``: MERGES the supplied keys onto the persisted
      settings rather than replacing the whole object. This means a
      partial update like ``{settings: {variations_per_room: 3}}``
      changes only that key — ``model``/``quality``/``size`` keep their
      persisted values. Without the merge, defaults from
      ``StagingSettings.__init__`` would silently overwrite whatever the
      user previously chose.
    - For ``design_brief``: ``None`` is meaningful — it clears the brief.
    - NEVER modifies ``rooms``, ``analyses``, or ``status``. The shape
      of ``UpdateProjectRequest`` intentionally has no fields for them
      so a misbehaving client cannot edit them through this endpoint.
    - Does NOT trigger any generation — plain JSON response, no SSE.

    The read-modify-write is wrapped in the per-project asyncio lock
    from ``staging_pipeline._get_project_lock`` so concurrent writes
    (PATCH, regen finalizers, pipeline persists) serialize at the
    storage boundary. This is necessary because Cosmos writes are full-
    doc replacements; without serialization a finalizer write that
    started reading state BEFORE this PATCH could clobber the PATCH on
    its way out. Pairs with the surgical fix in
    ``StagingPipeline._persist_project_locked`` that scopes worker
    writes to ``{rooms, status}`` only — the latter prevents the
    *content* clobber while this lock prevents the *write-order*
    clobber.
    """
    async with _get_project_lock(project_id):
        project_data = storage.get_project(project_id)
        if not project_data:
            raise HTTPException(status_code=404, detail="Project not found")

        # Use ``__fields_set__`` to distinguish "absent" from "explicit
        # null". The Pydantic v1 validators on ``UpdateProjectRequest``
        # already rejected explicit null for name/prompt/settings (those
        # raise 422 at parse time), so by the time we get here all
        # values in the set are non-None for those three fields.
        sent = body.__fields_set__

        if "name" in sent:
            project_data["name"] = body.name
        if "prompt" in sent:
            project_data["prompt"] = body.prompt
        if "settings" in sent:
            # Merge supplied keys onto the persisted settings. The
            # ``body.dict(exclude_unset=True)["settings"]`` form returns
            # only the keys the client actually sent (Pydantic v1 tracks
            # this on the nested model too), so we don't accidentally
            # overwrite ``model``/``quality``/``size`` with defaults
            # when the client only wanted to change ``variations_per_room``.
            settings_update = body.dict(exclude_unset=True).get("settings", {})
            existing_settings = project_data.get("settings") or {}
            project_data["settings"] = {**existing_settings, **settings_update}
        if "design_brief" in sent:
            # ``None`` is meaningful here — clears the brief.
            project_data["design_brief"] = body.design_brief

        # Issue 001 of project-settings-completeness PRD:
        # mirror ``project.prompt`` and
        # ``project.design_brief.global_instructions`` so the user sees
        # one coherent "prompt" across Settings, Brief, gallery
        # dialogs, project cards, regenerate flows, and any future
        # snapshot-restore path. The mirror runs AFTER the field-
        # application block so it operates on the fully-applied
        # incoming state (it reads ``project_data["design_brief"]``,
        # not ``body.design_brief``, so the "brief explicitly cleared
        # to None" branch falls through cleanly).
        #
        # The mirror is intentionally scoped to PATCH /projects/{id}
        # and PUT /brief — the two USER-FACING inbound update paths.
        # POST /brief (generate_brief) is system-driven brief-synthesis
        # whose output is downstream of the prompt the user already
        # controls via PATCH; mirroring it would risk overwriting the
        # user's prompt with an LLM-synthesized variant on every
        # regenerate-brief, which the PRD explicitly forbids ("the
        # user keeps their edits").
        _mirror_prompt_and_brief_in_place(
            project_data=project_data,
            sent=sent,
            body_prompt=body.prompt,
        )

        storage.update_project(project_id, project_data)

    clean = {k: v for k, v in project_data.items() if k != "doc_type" and not k.startswith("_")}
    return ProjectResponse(project=StagingProject(**clean))


def _is_nonempty_str(value: object) -> bool:
    """Mirror gate: a value is "non-empty" when it's a string AND
    contains at least one non-whitespace character. The PRD says the
    mirror skips empty global_instructions; we extend "empty" to
    whitespace-only so we don't propagate visual garbage between the
    prompt and the brief."""
    return isinstance(value, str) and bool(value.strip())


def _mirror_prompt_and_brief_in_place(
    *,
    project_data: dict,
    sent: set,
    body_prompt: Optional[str],
) -> None:
    """Apply the prompt ↔ design_brief.global_instructions mirror to
    ``project_data`` in place. See the inline comment at the call site
    in ``update_project`` for the rationale and scope decisions.

    Rules (verbatim from PRD § Backend mirror behavior):

    1. Both ``prompt`` and ``design_brief`` in ``sent``:
       brief wins. If the persisted brief is a dict and its
       ``global_instructions`` is non-empty (after strip) →
       ``project_data["prompt"]`` is set from it. Otherwise the
       user-supplied ``body_prompt`` is preserved (already applied by
       the caller; nothing to do).
    2. Only ``prompt`` in ``sent``:
       If a brief is currently persisted (dict), copy ``body_prompt``
       into ``brief["global_instructions"]``. If no brief, no-op (the
       caller has already updated ``project_data["prompt"]``).
    3. Only ``design_brief`` in ``sent``:
       If brief is dict and ``global_instructions`` is non-empty →
       ``project_data["prompt"]`` mirrors. Otherwise (brief cleared via
       ``None``, brief is empty ``{}``, ``global_instructions`` missing
       or empty/whitespace-only) → ``project_data["prompt"]`` is
       untouched.
    """
    prompt_in = "prompt" in sent
    brief_in = "design_brief" in sent
    if not (prompt_in or brief_in):
        return

    brief = project_data.get("design_brief")
    brief_is_dict = isinstance(brief, dict)

    if prompt_in and brief_in:
        # Brief wins when it has something to win with.
        if brief_is_dict and _is_nonempty_str(brief.get("global_instructions")):
            project_data["prompt"] = brief["global_instructions"]
        # Else: keep the user-supplied prompt the caller already wrote.
    elif prompt_in:
        # Only prompt sent. If a brief exists, mirror prompt INTO brief.
        # We mutate the existing dict in place — the storage layer does
        # a full ``replace_item`` on the project doc so reference identity
        # doesn't matter, and the existing endpoint code already mutates
        # nested dicts in place (see the settings-merge a few lines up).
        if brief_is_dict:
            brief["global_instructions"] = body_prompt
            project_data["design_brief"] = brief
    else:
        # Only design_brief sent. Mirror non-empty global_instructions
        # OUT to project.prompt. ``brief`` here is whatever the caller
        # just wrote — possibly None (clear), possibly a dict.
        if brief_is_dict and _is_nonempty_str(brief.get("global_instructions")):
            project_data["prompt"] = brief["global_instructions"]


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

                # Issue 003 (projects-page-improvements PRD): per-room
                # ``prompt_addendum`` is composed onto the freshly-
                # generated base prompt at the LAST MILE (after either
                # the brief OR adapt_prompt path produced the base) so
                # both source paths get the addendum uniformly. The
                # composer is a no-op when ``room.prompt_addendum`` is
                # None / empty / whitespace-only — existing rooms
                # without an addendum see the original behavior.
                # Retry path is intentionally NOT touched: it uses the
                # prior ``generation_metadata.adapted_prompt`` verbatim
                # (which already includes whatever addendum was in
                # effect at the original generation time), so re-
                # composing would double-append.
                if adapted_prompt is not None:
                    adapted_prompt = PromptComposer.compose(
                        project_prompt=project.prompt,
                        design_brief=adapted_prompt,
                        room_addendum=room.prompt_addendum,
                    )

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


def _log_edit_prompt_event(
    *,
    event: str,
    project_id: str,
    room_id: str,
    new_variation_id: str,
    source_variation_id: str,
    elapsed_ms: Optional[int] = None,
    tokens_used: Optional[int] = None,
) -> None:
    """Emit a structured log line for an Edit Prompt lifecycle event.

    Issue 004 of the projects-page-improvements PRD § Solution → 4 +
    Implementation Decisions → Backend modules: log analytics needs to
    count Edit Prompt usage SEPARATELY from regen usage so dashboards
    can answer "how often do users actually edit prompts vs retry vs
    try-something-new". Therefore the event names form a dedicated
    family ``staging.variation_edit_prompt.{started, completed, failed}``
    instead of aliasing the existing ``staging.variation_regen.*`` lines.

    Field contract (mirrors ``_log_regen_event`` for analytics
    consistency, minus the regen-specific ``strategy`` /
    ``effective_strategy`` fields which don't apply here):

    - ``project_id`` / ``room_id`` / ``new_variation_id`` /
      ``source_variation_id`` — entity identifiers. ``source_variation_id``
      lets analytics trace which existing variation triggered each Edit
      Prompt; ``new_variation_id`` identifies the freshly-appended one.
    - ``elapsed_ms`` — wall-clock milliseconds from request acceptance
      to terminal-event observation. Sourced from ``time.monotonic()``
      deltas in the endpoint so the operator-facing total includes
      preflight + image-gen + persistence (NOT the pipeline's image-gen-
      only ``elapsed_ms``). Carried on ``completed`` / ``failed``.
    - ``tokens_used`` — image-gen token usage from the pipeline's
      terminal event (``None`` for failed image-gen). Carried on
      ``completed`` / ``failed`` even when None so consumers can
      distinguish "no image-gen happened" from "field missing".
    """
    extra = {
        "event": event,
        "project_id": project_id,
        "room_id": room_id,
        "new_variation_id": new_variation_id,
        "source_variation_id": source_variation_id,
    }
    parts = [
        f"event={event}",
        f"project_id={project_id}",
        f"room_id={room_id}",
        f"new_variation_id={new_variation_id}",
        f"source_variation_id={source_variation_id}",
    ]
    is_terminal = event.endswith(".completed") or event.endswith(".failed")
    if is_terminal:
        extra["elapsed_ms"] = elapsed_ms
        parts.append(f"elapsed_ms={elapsed_ms}")
        extra["tokens_used"] = tokens_used
        parts.append(f"tokens_used={tokens_used}")

    logger.info(" ".join(parts), extra=extra)


@router.post("/projects/{project_id}/rooms/{room_id}/variations/{variation_id}/edit-prompt")
async def edit_variation_prompt(
    project_id: str,
    room_id: str,
    variation_id: str,
    body: EditPromptRequest,
    storage: StagingStorageService = Depends(get_staging_storage),
    pipeline=Depends(get_staging_pipeline),
):
    """Append a NEW variation generated from a user-edited prompt.

    Issue 004 of the projects-page-improvements PRD: lets users edit
    the prompt that produced a generated image and have the new image
    appear as a fresh variation alongside the original — preserving
    the original for A/B comparison.

    Distinct from ``regenerate_variation`` which mutates the existing
    variation in place. The ``variation_id`` URL segment identifies the
    "source" variation the user clicked Edit on but is NOT mutated by
    this endpoint; a fresh ``Variation`` is appended to
    ``room.variations`` with its own UUID.

    Pipeline path bypasses ``BriefGeneratorService.brief_to_prompts``
    entirely: the user's text flows straight through
    ``PromptComposer.compose`` as ``variation_override``. The room's
    ``prompt_addendum`` (issue 003) is still composed onto the end so
    the addendum constraint applies even to user-typed prompts.

    Concurrency: the preflight read-append-write is wrapped in the
    per-project asyncio lock (issue 002 pattern) — appending to a list
    is more racy than the regen preflight's status flip, so the lock
    is mandatory here even though ``regenerate_variation`` historically
    didn't have one. The finalizer write is also wrapped in the lock
    (mirrors the regen finalizer pattern).

    Status fix-up: ``process_single_variation``'s built-in failure
    rollback restores ``prior_status`` for the variation, which for an
    appended variation pre-set to PROCESSING means it would strand in
    PROCESSING after a failure. The endpoint's finally block detects
    this and forces the variation to FAILED with the captured error
    message so the room/project status calculator sees a truthful
    terminal state.
    """
    project_data = storage.get_project(project_id)
    if not project_data:
        raise HTTPException(status_code=404, detail="Project not found")

    clean = {k: v for k, v in project_data.items() if k != "doc_type" and not k.startswith("_")}
    project = StagingProject(**clean)

    room = next((r for r in project.rooms if r.id == room_id), None)
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")

    source_variation = next((v for v in room.variations if v.id == variation_id), None)
    if not source_variation:
        raise HTTPException(status_code=404, detail="Variation not found")

    # 409 mutex: don't accept Edit Prompt while the source variation is
    # mid-regen. Mirrors regenerate_variation's PROCESSING guard.
    if source_variation.status == ItemStatus.PROCESSING:
        raise HTTPException(
            status_code=409,
            detail="Source variation is currently being processed; wait for it to finish before editing its prompt",
        )

    # Compose the final adapted_prompt. The user's text is the
    # variation_override (highest precedence in PromptComposer); the
    # room's prompt_addendum is layered on top per slice 003 contract.
    # design_brief=None bypasses BriefGeneratorService entirely per
    # PRD § Solution → 4.
    composed_prompt = PromptComposer.compose(
        project_prompt=project.prompt,
        design_brief=None,
        room_addendum=room.prompt_addendum,
        variation_override=body.adapted_prompt,
    )

    # Append a fresh variation. Preset status=PROCESSING so concurrent
    # callers see it as in-flight (the 409 mutex above keys off this
    # state). Capture the new ID before appending so we can locate the
    # variation in fresh storage reads later.
    new_variation = Variation(
        id=str(uuid.uuid4()),
        status=ItemStatus.PROCESSING,
    )

    # Preflight write — wrapped in the per-project lock per the rubber-
    # duck-flagged blocker. Appending to room.variations is a list-
    # append; outside the lock a concurrent worker write to room.status
    # or sibling-variation state could clobber the append (Cosmos
    # writes are full-doc replacements). The per-project lock pattern
    # (issue 002) serializes us with both pipeline workers
    # (`_persist_project_locked`) and PATCH endpoints.
    async with _get_project_lock(project_id):
        # Re-read inside the lock so we observe any concurrent worker
        # writes that landed between our pre-lock read and now.
        fresh = storage.get_project(project_id)
        if not fresh:
            raise HTTPException(status_code=404, detail="Project not found")
        clean_fresh = {k: v for k, v in fresh.items() if k != "doc_type" and not k.startswith("_")}
        project = StagingProject(**clean_fresh)
        room = next((r for r in project.rooms if r.id == room_id), None)
        if not room:
            raise HTTPException(status_code=404, detail="Room not found")
        # Re-validate the source variation under the lock — a concurrent
        # delete or status flip could have raced past our pre-lock check.
        source_variation = next((v for v in room.variations if v.id == variation_id), None)
        if not source_variation:
            raise HTTPException(status_code=404, detail="Variation not found")
        if source_variation.status == ItemStatus.PROCESSING:
            raise HTTPException(status_code=409, detail="Source variation is currently being processed")
        room.variations.append(new_variation)
        room.status = ItemStatus.PROCESSING
        storage.update_project(project_id, json.loads(project.json()))

    # Started log fires AFTER the preflight write succeeds — same
    # pattern as regenerate_variation: a write-failure path doesn't
    # leave a stranded "started" log line with no terminal partner.
    edit_prompt_start_time = time.monotonic()
    _log_edit_prompt_event(
        event="staging.variation_edit_prompt.started",
        project_id=project_id,
        room_id=room_id,
        new_variation_id=new_variation.id,
        source_variation_id=variation_id,
    )

    async def event_stream():
        terminal_event_type: Optional[str] = None
        terminal_elapsed_ms: Optional[int] = None
        terminal_tokens_used: Optional[int] = None
        terminal_error_message: Optional[str] = None
        final_status = "completed"

        try:
            async for event in pipeline.process_single_variation(
                project, room, new_variation, composed_prompt
            ):
                if event.get("type") in ("variation_completed", "variation_failed"):
                    terminal_event_type = event["type"]
                    terminal_elapsed_ms = int((time.monotonic() - edit_prompt_start_time) * 1000)
                    terminal_tokens_used = event.get("tokens_used")
                    if event["type"] == "variation_failed":
                        terminal_error_message = event.get("error") or "Generation failed"
                yield _sse_event(event["type"], event)
        except Exception:
            # Defense-in-depth: if the pipeline raises instead of
            # yielding a terminal event, synthesize a failure so the
            # log line + finalizer fix-up still fire.
            if terminal_event_type is None:
                terminal_event_type = "variation_failed"
                terminal_elapsed_ms = int((time.monotonic() - edit_prompt_start_time) * 1000)
                terminal_tokens_used = None
                terminal_error_message = "Generation failed (pipeline exception)"
            raise
        finally:
            # Recompute room + project status under the per-project
            # lock so we serialize with any concurrent worker writes.
            # ALSO: force the appended variation to FAILED if it didn't
            # reach a terminal-success state — process_single_variation's
            # built-in rollback restores prior_status (PROCESSING for an
            # appended variation), which would otherwise strand the new
            # variation forever. See test
            # `test_edit_prompt_failure_marks_appended_variation_failed_not_stranded`.
            async with _get_project_lock(project_id):
                fresh = storage.get_project(project_id)
                if fresh:
                    clean_fresh = {k: v for k, v in fresh.items() if k != "doc_type" and not k.startswith("_")}
                    fresh_project = StagingProject(**clean_fresh)
                    target_room = next((r for r in fresh_project.rooms if r.id == room_id), None)
                    if target_room:
                        # Find our appended variation in the fresh state.
                        appended = next(
                            (v for v in target_room.variations if v.id == new_variation.id),
                            None,
                        )
                        if appended is not None and appended.status != ItemStatus.COMPLETED:
                            # Force to FAILED so the room calculator sees a
                            # terminal state. Use the captured error or a
                            # generic fallback if no terminal SSE event was
                            # observed (e.g. client disconnect before any
                            # event landed).
                            appended.status = ItemStatus.FAILED
                            appended.error = (
                                terminal_error_message
                                or appended.error
                                or "Edit Prompt generation did not complete"
                            )
                        # Recompute room status (mirrors
                        # regenerate_variation's per-room rule).
                        any_completed = any(v.status == "completed" for v in target_room.variations)
                        any_pending = any(v.status in ("pending", "processing") for v in target_room.variations)
                        if any_pending:
                            target_room.status = "processing"
                        elif any_completed:
                            target_room.status = "completed"
                        else:
                            target_room.status = "failed"
                    # Single source of truth for project status (issue 001).
                    fresh_project.status = ProjectStatusCalculator.compute_status(fresh_project.rooms)
                    storage.update_project(project_id, json.loads(fresh_project.json()))
                    final_status = fresh_project.status

            if terminal_event_type is not None:
                terminal_log_event = (
                    "staging.variation_edit_prompt.completed"
                    if terminal_event_type == "variation_completed"
                    else "staging.variation_edit_prompt.failed"
                )
                _log_edit_prompt_event(
                    event=terminal_log_event,
                    project_id=project_id,
                    room_id=room_id,
                    new_variation_id=new_variation.id,
                    source_variation_id=variation_id,
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
    """Save user edits to the Design Brief.

    Issue 001 of the project-settings-completeness PRD: the brief and
    ``project.prompt`` are kept in sync — when the user saves a brief
    whose ``global_instructions`` is non-empty, ``project.prompt``
    mirrors that value in the same Cosmos write. This keeps the user's
    "prompt" coherent across Settings, Brief, gallery dialogs, project
    cards, and regenerate flows regardless of which surface they
    edited from.

    The handler now runs inside ``_get_project_lock`` for the same
    reason ``update_project`` does: Cosmos writes are full-document
    replacements, and adding mirror-driven ``prompt`` writes to this
    path widened the loss surface beyond just ``design_brief``. Without
    the lock, a concurrent ``PATCH /projects/{id}`` (which holds the
    lock and also writes ``prompt``) could be clobbered by an
    unserialized PUT here.

    Returns ``{"brief": <persisted brief dict>}`` so the frontend's
    ``updateBrief`` wrapper (``frontend/services/stagingApi.ts``) can
    do its expected ``return data.brief``. The pre-fix handler had no
    explicit return — FastAPI sent ``null`` as the body, and the
    frontend's ``data.brief`` access crashed silently into the wizard's
    "Failed to save Design Brief" toast. Adding the return statement
    is a tightly-coupled bug fix to the same handler this issue
    rewrites for the lock + mirror.
    """
    async with _get_project_lock(project_id):
        project_data = storage.get_project(project_id)
        if not project_data:
            raise HTTPException(status_code=404, detail="Project not found")

        brief_dict = brief.dict()
        updates = {"design_brief": brief_dict}
        if _is_nonempty_str(brief_dict.get("global_instructions")):
            updates["prompt"] = brief_dict["global_instructions"]
        storage.update_project(project_id, updates)

    return {"brief": brief_dict}