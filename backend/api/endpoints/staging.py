"""FastAPI endpoints for virtual staging projects."""
import json
import logging
import time
import uuid
from typing import List, Optional

import asyncio

from fastapi import APIRouter, Cookie, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import StreamingResponse

from backend.core.azure_storage import AzureBlobStorageService
from backend.core.brief_generator import backfill_legacy_brief_sections
from backend.core.brief_resolver import migrate_legacy_plant_palette
from backend.core.config import settings
from backend.core.job_queue import JobQueue
from backend.core.job_store import JobStore, deterministic_job_id
from backend.core.sse_hub import SSEHub, get_sse_hub
from backend.core.project_status import ProjectStatusCalculator
from backend.core.prompt_composer import PromptComposer
from backend.core.prompt_summarizer import PromptSummarizer, truncate_to_summary
from backend.core.staging_pipeline import _get_project_lock
from backend.core.staging_reconcile import reconcile_project
from backend.core.staging_storage import StagingStorageService
from backend.core.thumbnail_backfill import backfill_project_thumbnails
from backend.core.thumbnail_deriver import ThumbnailDeriver
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


def get_job_store() -> JobStore:
    """FastAPI dependency: ``JobStore`` for the async-queue endpoints
    (issue 004 of the image-pipeline PRD). Patched in tests."""
    return JobStore()


def get_thumbnail_backfill_deps():
    """FastAPI dependency: ``(thumbnail_deriver, blob_service)`` for the
    lazy thumbnail backfill (issue 012). Patched in tests via
    ``app.dependency_overrides`` so the read path can be exercised
    without real blob I/O.
    """
    blob_service = AzureBlobStorageService()
    return (ThumbnailDeriver(blob_service), blob_service)


def get_prompt_summarizer() -> PromptSummarizer:
    """FastAPI dependency: ``PromptSummarizer`` for ``POST /projects``
    and ``PATCH /projects/{id}`` (issue 013 of the image-pipeline PRD).
    Patched in tests via ``app.dependency_overrides`` so the endpoint
    flow can be exercised without an LLM round-trip."""
    from backend.core import async_llm_client

    return PromptSummarizer(
        async_llm_client=async_llm_client,
        llm_deployment=settings.LLM_DEPLOYMENT,
    )


def get_job_queue() -> JobQueue:
    """FastAPI dependency: ``JobQueue`` for the async-queue endpoints
    (issue 004 of the image-pipeline PRD). Patched in tests."""
    return JobQueue()


async def get_sse_hub_dep() -> SSEHub:
    """FastAPI dependency: per-replica ``SSEHub`` (issue 005). Patched
    in tests via ``app.dependency_overrides`` so an in-memory feed can
    be installed without booting Cosmos."""
    return await get_sse_hub()


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
    summarizer: PromptSummarizer = Depends(get_prompt_summarizer),
):
    # Issue 013: seed ``prompt_summary`` at create time so the project
    # page can render the collapsed-summary view from the very first
    # read. Short prompts (≤240 chars — the common case) skip the LLM
    # round-trip via the summarizer's pass-through optimization.
    prompt_summary = await summarizer.summarize(request.prompt)
    project_data = {
        "id": str(uuid.uuid4()),
        "name": request.name,
        "prompt": request.prompt,
        "prompt_summary": prompt_summary,
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
        # Combine reconcile + legacy-brief-migration + section backfill
        # into a single optional writeback. Each pass alone may mutate;
        # if multiple mutate we only persist once. ``or`` short-circuits,
        # but we want every call to run, so we OR the results explicitly.
        reconciled = reconcile_project(p)
        migrated = _migrate_design_brief_in_place(p)
        sections_backfilled = backfill_legacy_brief_sections(p)
        if reconciled or migrated or sections_backfilled:
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
    backfill_deps: tuple = Depends(get_thumbnail_backfill_deps),
):
    project = storage.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Auto-heal stale processing states + opportunistically migrate legacy
    # plant_palette → object_palette + lazy-backfill canonical brief
    # sections (issue 016) on read (single combined writeback).
    reconciled = reconcile_project(project)
    migrated = _migrate_design_brief_in_place(project)
    sections_backfilled = backfill_legacy_brief_sections(project)
    # Issue 012: lazy thumbnail backfill. Variations created before issue
    # 010 wired the deriver into the pipeline have ``image_url`` but no
    # ``thumb_url`` / ``md_url``. Derive them on first read so the grid
    # renders proper thumbnails; subsequent reads short-circuit because
    # the fields are now populated. Best-effort — per-variation failures
    # log at WARNING and the read still returns the project.
    deriver, blob_service = backfill_deps
    try:
        backfilled = await backfill_project_thumbnails(
            project,
            thumbnail_deriver=deriver,
            blob_service=blob_service,
            container_name=settings.AZURE_BLOB_IMAGE_CONTAINER,
        )
    except Exception as e:
        logger.warning("Thumbnail backfill raised for project %s: %s", project_id, e)
        backfilled = False

    if reconciled or migrated or backfilled or sections_backfilled:
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
    store: JobStore = Depends(get_job_store),
):
    """Delete a project and all associated blob storage artifacts.

    Cascade (issue 007): before deleting the project document, mark
    every non-terminal job for ``project_id`` with
    ``cancel_requested=True``. The ``JobWorker`` (issue 003) observes
    the flag at the next safe point and transitions each job to
    ``cancelled``, preventing leaked Azure compute / blob writes for
    a project that no longer exists.

    Best-effort: ``JobStore`` failures (transient Cosmos errors,
    individual ``update_job`` raises) are logged at WARNING and do
    NOT block the project delete. The Cosmos document and blob
    cleanup happen regardless — a stuck-pending job after a delete
    is recoverable on the next deploy; a failed delete is a UX bug
    the user has to retry.

    Gated on ``FEATURE_ASYNC_QUEUE`` so the legacy in-process
    pipeline (no JobStore container provisioned) is unaffected.
    """
    # Get project first to find blob paths
    project_data = storage.get_project(project_id)
    if not project_data:
        raise HTTPException(status_code=404, detail="Project not found")

    # Cascade-cancel non-terminal jobs for this project. Best-effort.
    if settings.FEATURE_ASYNC_QUEUE:
        _cascade_cancel_project_jobs(project_id, store)

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
    """Partial-update editable Room fields.

    Issue history:

    - Originally added in issue 003 of the projects-page-improvements
      PRD with a single editable field (``prompt_addendum``). The
      implementer chose a dedicated room-scoped endpoint over
      extending ``PATCH /projects/{id}`` because:

        * ``PATCH /projects/{id}`` doesn't exist yet (slice 002).
        * The URL semantics match the resource being edited.
        * Keeps room-scoped concerns out of the project-level PATCH.

    - Extended in issue 004 of the project-settings-completeness PRD
      with ``label`` so the new ``ProjectRoomsManager`` UI on the
      Project Settings sheet can rename rooms in place. The endpoint
      contract becomes ``__fields_set__``-aware for BOTH ``label``
      AND ``prompt_addendum`` — load-bearing per the rubber-duck
      blocker call: a label-only PATCH must NOT silently clear an
      existing addendum (which would happen if the handler defaulted
      the absent field to ``None`` and unconditionally wrote it back).

    The endpoint:

    - Updates ONLY the fields the client actually sent (uses
      ``body.__fields_set__`` to distinguish "absent" from "explicit
      null", same pattern as ``update_project``).
    - For ``label``: trims surrounding whitespace before persisting.
      Empty / whitespace-only / explicit-null is rejected at parse
      time by ``UpdateRoomRequest._label_non_empty`` (422).
    - For ``prompt_addendum``: normalizes ``""``, ``None``, and
      whitespace-only to ``None`` so the persisted shape stays
      consistent with the composer's "absent" rule.
    - Leaves variations / status / image_url untouched.
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

        # Field-set-aware updates (issue 004). Both fields are
        # independent and only touched when the client actually sent
        # them. The rubber-duck regression
        # ``test_patch_room_label_only_preserves_existing_addendum``
        # pins the load-bearing claim that an absent ``prompt_addendum``
        # leaves the persisted value untouched.
        sent = body.__fields_set__

        if "label" in sent:
            # The validator already rejected null / empty / whitespace,
            # so ``body.label`` is a non-empty string here. Trim before
            # persisting (matches the existing addendum-trim behavior
            # and the ``name`` / ``prompt`` rules in
            # ``UpdateProjectRequest``).
            room["label"] = body.label.strip()

        if "prompt_addendum" in sent:
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


def _extract_blob_name_for_cleanup(blob_url: str) -> Optional[str]:
    """Extract the blob name (everything after the container segment)
    from a full Azure Blob Storage URL. Returns ``None`` if the URL
    doesn't parse cleanly so the caller can skip a malformed entry
    rather than crashing the whole cleanup pass.

    Mirrors ``StagingPipeline._extract_blob_name`` (kept local to
    avoid pulling in the pipeline module for an endpoint helper).
    """
    if not blob_url or not isinstance(blob_url, str):
        return None
    try:
        parts = blob_url.split("/")
        net_idx = next(i for i, p in enumerate(parts) if p.endswith(".net"))
        return "/".join(parts[net_idx + 2:])  # skip container segment
    except (StopIteration, IndexError):
        for container in ("images", "videos"):
            if f"/{container}/" in blob_url:
                return blob_url.split(f"/{container}/")[1]
        return None


def _prune_room_metadata_in_place(project_data: dict, room_id: str) -> None:
    """Remove all room-keyed metadata for ``room_id`` from a project
    document, in-place.

    Issue 005 of the project-settings-completeness PRD (rubber-duck
    blocker): without pruning, deleting a room from ``project.rooms``
    leaves stale references in:

      - ``project.analyses[*]`` where ``room_id == room_id`` (used by
        the brief generator and regenerate flows).
      - ``project.design_brief.per_image_notes[room_id]`` (used by the
        brief composer when rebuilding prompts).
      - ``project.design_brief.per_image_objects[room_id]`` (used by
        the per-image objects UI and the brief).

    Those stale references would then leak into future brief / regen /
    composer operations and silently re-introduce the deleted room's
    state. Defensive: handles None / missing entries cleanly so legacy
    or unmigrated projects don't crash.
    """
    analyses = project_data.get("analyses")
    if isinstance(analyses, list):
        project_data["analyses"] = [
            entry for entry in analyses
            if entry.get("room_id") != room_id
        ]

    brief = project_data.get("design_brief")
    if isinstance(brief, dict):
        notes = brief.get("per_image_notes")
        if isinstance(notes, dict) and room_id in notes:
            del notes[room_id]
        objects = brief.get("per_image_objects")
        if isinstance(objects, dict) and room_id in objects:
            del objects[room_id]


def _cleanup_room_blobs(project_id: str, room: dict) -> None:
    """Best-effort blob cleanup for a deleted room. Runs OUTSIDE the
    project lock (issue 005 rubber-duck non-blocking finding) so blob
    I/O latency cannot block other room edits / regens on the project.

    Cleans up:
      - The room's ``original_image_url`` blob.
      - The room's ``original_thumbnail_url`` blob (when present).
      - All blobs under the ``staging/{project_id}/variations/{room_id}/``
        prefix — covers all variations even if some have null
        ``image_url`` (incomplete generation, edit-prompt artifacts,
        regen artifacts).

    Failures are LOGGED but do NOT bubble — the metadata delete already
    succeeded by the time this runs. Mirrors ``delete_project``'s
    try/except pattern.
    """
    try:
        blob_service = AzureBlobStorageService()
        container_client = blob_service.blob_service_client.get_container_client(
            settings.AZURE_BLOB_IMAGE_CONTAINER
        )

        # Per-blob deletes for the originals (these live under
        # ``staging/{project_id}/originals/`` which is shared across
        # rooms — a prefix sweep would risk other rooms' originals).
        for url_field in ("original_image_url", "original_thumbnail_url"):
            blob_name = _extract_blob_name_for_cleanup(room.get(url_field))
            if blob_name:
                try:
                    container_client.delete_blob(blob_name)
                except Exception as e:
                    logger.warning(
                        f"Failed to delete blob {blob_name} for room {room.get('id')}: {e}"
                    )

        # Prefix sweep for variations (this prefix is room-scoped so
        # it's safe to bulk-delete everything under it).
        room_id = room.get("id")
        if room_id:
            variations_prefix = f"staging/{project_id}/variations/{room_id}/"
            for blob in container_client.list_blobs(name_starts_with=variations_prefix):
                try:
                    container_client.delete_blob(blob.name)
                except Exception as e:
                    logger.warning(
                        f"Failed to delete variation blob {blob.name}: {e}"
                    )
    except Exception as e:
        logger.warning(
            f"Blob cleanup failed for room {room.get('id')} of project {project_id}: {e}"
        )


@router.delete("/projects/{project_id}/rooms/{room_id}", response_model=ProjectResponse)
async def remove_room(
    project_id: str,
    room_id: str,
    storage: StagingStorageService = Depends(get_staging_storage),
):
    """Cascading delete of a room and its associated metadata.

    Issue 005 of the project-settings-completeness PRD. Backs the
    "Delete with confirm" affordance in ``ProjectRoomsManager`` on
    the Project Settings sheet. The PRD/issue described this endpoint
    as already existing on the worktree branch but it did not exist
    on ``main`` — same adaptation pattern as 001/002/003/004.

    Contract (asserted by ``tests/test_staging_endpoints_delete_room.py``):

    - REJECTS with 409 Conflict when ``project.status == "processing"``.
      Rubber-duck blocker: the project lock alone does not protect
      against an in-flight pipeline worker that started BEFORE the
      delete carrying a stale ``rooms`` snapshot in memory and
      reintroducing the deleted room when it eventually writes its
      accumulated state back. The frontend's issue 007 will also
      disable the affordance during processing; this guard is the
      authoritative protection against a programmatic / racing
      client that bypasses the UI.

    - INSIDE the project lock: validates project + room exist (404
      otherwise), prunes ``project.rooms`` of the target room AND
      prunes room-keyed metadata in ``analyses`` and
      ``design_brief.per_image_notes`` / ``design_brief.per_image_objects``
      via ``_prune_room_metadata_in_place``, then persists the result.

    - OUTSIDE the lock: best-effort blob cleanup via
      ``_cleanup_room_blobs``. Blob I/O latency does not block other
      room edits / regens on the project (rubber-duck non-blocking
      finding). Cleanup failures are logged but the response is still
      200 — the metadata delete already succeeded.

    The lock semantic mirrors ``update_room`` and ``update_project``.
    """
    async with _get_project_lock(project_id):
        project_data = storage.get_project(project_id)
        if not project_data:
            raise HTTPException(status_code=404, detail="Project not found")

        # Reject delete during processing (rubber-duck blocker).
        if project_data.get("status") == "processing":
            raise HTTPException(
                status_code=409,
                detail=(
                    "Cannot delete a room while the project is processing. "
                    "Wait for generation to complete, then try again."
                ),
            )

        rooms = project_data.get("rooms", [])
        room = next((r for r in rooms if r.get("id") == room_id), None)
        if not room:
            raise HTTPException(status_code=404, detail="Room not found")

        # Hold a reference for the post-lock blob cleanup pass.
        deleted_room = room

        # Mutate metadata IN-PLACE: remove the room from rooms and
        # prune all room-keyed entries from analyses + design brief.
        project_data["rooms"] = [r for r in rooms if r.get("id") != room_id]
        _prune_room_metadata_in_place(project_data, room_id)

        storage.update_project(project_id, project_data)

    # Best-effort blob cleanup runs OUTSIDE the lock so blob I/O latency
    # doesn't block other room operations on this project. The metadata
    # delete is already persisted at this point; cleanup failures are
    # logged but do not bubble.
    _cleanup_room_blobs(project_id, deleted_room)

    clean = {k: v for k, v in project_data.items() if k != "doc_type" and not k.startswith("_")}
    return ProjectResponse(project=StagingProject(**clean))


@router.patch("/projects/{project_id}", response_model=ProjectResponse)
async def update_project(
    project_id: str,
    body: UpdateProjectRequest,
    storage: StagingStorageService = Depends(get_staging_storage),
    summarizer: PromptSummarizer = Depends(get_prompt_summarizer),
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

        # Issue 013 — prompt_summary maintenance. Three cases:
        #
        #   1. Client sent prompt_summary explicitly: that wins. We
        #      still pass it through ``truncate_to_summary`` so a
        #      client overshoot (>240 chars) is normalized to the
        #      same contract the rest of the system enforces, rather
        #      than 422-ing the request the user can otherwise satisfy.
        #
        #   2. Client sent prompt but NOT prompt_summary: server
        #      regenerates via PromptSummarizer (which itself short-
        #      circuits short prompts and falls back to truncation
        #      on LLM failure). This is the common UX path — the
        #      collapsed-summary view stays accurate after a prompt
        #      edit without the client having to know about it.
        #
        #   3. Neither sent: leave persisted summary untouched.
        #
        # Per the PRD's AC, none of these branches enqueues
        # regeneration jobs — prompt edits explicitly do NOT trigger
        # image regeneration. The user clicks a separate Regenerate
        # button (issue 019) when they want that.
        if "prompt_summary" in sent:
            project_data["prompt_summary"] = truncate_to_summary(body.prompt_summary)
        elif "prompt" in sent:
            project_data["prompt_summary"] = await summarizer.summarize(body.prompt)

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

# ---------------------------------------------------------------------------
# Async-queue REST surface (issue 004 — image-pipeline-and-project-ux-overhaul)
# ---------------------------------------------------------------------------


_TERMINAL_JOB_STATUSES = frozenset({"succeeded", "failed", "cancelled"})


def _cascade_cancel_project_jobs(project_id: str, store: "JobStore") -> None:
    """Mark every non-terminal job for ``project_id`` as
    ``cancel_requested=True``.

    Issue 007 cascade. Best-effort: any failure in
    ``list_jobs_by_project`` aborts the cascade with a WARNING log
    and the caller (``delete_project``) still completes the project
    delete. A failure inside an individual ``update_job`` is logged
    but does NOT abort the loop — partial cascade is strictly better
    than no cascade.

    The ``JobWorker`` (issue 003) re-reads the doc on its
    ``is_cancelled`` probe each tick, so the flag is observed
    regardless of which replica was holding the queue lease.
    """
    try:
        jobs = store.list_jobs_by_project(project_id)
    except Exception as exc:  # noqa: BLE001 — best-effort cascade
        logger.warning(
            "staging.delete_project.cascade.list_failed project_id=%s error=%s",
            project_id, exc,
        )
        return

    cancelled = 0
    for job in jobs:
        if job.get("status") in _TERMINAL_JOB_STATUSES:
            continue
        job_id = job.get("id")
        try:
            store.update_job(job_id, project_id, cancel_requested=True)
            cancelled += 1
        except Exception as exc:  # noqa: BLE001 — best-effort cascade
            logger.warning(
                "staging.delete_project.cascade.update_failed job_id=%s error=%s",
                job_id, exc,
            )
    logger.info(
        "staging.delete_project.cascade.done project_id=%s cancelled=%d",
        project_id, cancelled,
    )


def _require_async_queue_enabled() -> None:
    """Gate the new endpoints behind ``FEATURE_ASYNC_QUEUE``.

    Per PRD § Feature flags: the flag defaults true in dev/staging
    (``Settings.FEATURE_ASYNC_QUEUE = True``); production flips it via
    azd env var after a smoke test. When off, return 503 so a
    misconfigured production deploy fails loud rather than silently
    queueing into a worker that isn't running.
    """
    if not settings.FEATURE_ASYNC_QUEUE:
        raise HTTPException(
            status_code=503,
            detail="Async image-job queue is disabled (FEATURE_ASYNC_QUEUE=false)",
        )


def _select_revision_for_idempotent_regen(
    existing_jobs: list[dict],
    *,
    room_id: str,
    variation_id: str,
) -> int:
    """Pick the deterministic revision for a regenerate request so a
    rapid retry of the same call returns the same job ids.

    Rule: among jobs already persisted for this (room, variation),
    take the highest revision; if it is non-terminal (still in flight),
    re-use it — that's the active regen and we want the second caller
    to receive the SAME id (idempotent on retry). If the latest is
    terminal, increment by 1 — that's a "do it again" request.

    Returns 0 when no prior jobs exist.
    """
    matching = [
        j for j in existing_jobs
        if j.get("room_id") == room_id and j.get("variation_id") == variation_id
    ]
    if not matching:
        return 0
    latest = max(matching, key=lambda j: j.get("revision", 0))
    if latest.get("status") in _TERMINAL_JOB_STATUSES:
        return int(latest.get("revision", 0)) + 1
    return int(latest.get("revision", 0))


def _job_summary(doc: dict) -> dict:
    """Project a JobStore doc to the shape the frontend reads.

    Whitelists the operationally-meaningful fields so we never leak
    payload internals or future-added private fields through the
    public list endpoint.
    """
    return {
        "id": doc.get("id"),
        "project_id": doc.get("project_id"),
        "room_id": doc.get("room_id"),
        "variation_id": doc.get("variation_id"),
        "revision": doc.get("revision"),
        "kind": doc.get("kind"),
        "status": doc.get("status"),
        "progress": doc.get("progress"),
        "phase": doc.get("phase"),
        "attempts": doc.get("attempts"),
        "error": doc.get("error"),
        "result": doc.get("result"),
        "cancel_requested": doc.get("cancel_requested", False),
        "created_at": doc.get("created_at"),
        "updated_at": doc.get("updated_at"),
    }


@router.post("/projects/{project_id}/jobs/regenerate", status_code=202)
async def enqueue_regenerate_jobs(
    project_id: str,
    body: Optional[dict] = None,
    storage: StagingStorageService = Depends(get_staging_storage),
    store: JobStore = Depends(get_job_store),
    queue: JobQueue = Depends(get_job_queue),
):
    """Enqueue one regenerate-variation job per matching variation.

    Body shape (all fields optional):

        {
            "room_ids":      [str, ...],   # filter — only these rooms
            "variation_ids": [str, ...],   # filter — only these variations
        }

    Empty body / no filter → enqueue for every variation in every room.

    Returns ``{"job_ids": [...]}``. Deterministic + idempotent on retry
    (see ``_select_revision_for_idempotent_regen``).
    """
    _require_async_queue_enabled()

    project_data = storage.get_project(project_id)
    if not project_data:
        raise HTTPException(status_code=404, detail="Project not found")

    body = body or {}
    room_filter = set(body.get("room_ids") or [])
    variation_filter = set(body.get("variation_ids") or [])

    existing_jobs = store.list_jobs_by_project(project_id)

    job_ids: list[str] = []
    for room in project_data.get("rooms") or []:
        room_id = room.get("id")
        if room_filter and room_id not in room_filter:
            continue
        for variation in room.get("variations") or []:
            variation_id = variation.get("id")
            if variation_filter and variation_id not in variation_filter:
                continue
            revision = _select_revision_for_idempotent_regen(
                existing_jobs, room_id=room_id, variation_id=variation_id
            )
            doc = store.create_job(
                project_id=project_id,
                room_id=room_id,
                variation_id=variation_id,
                revision=revision,
                kind="regenerate_variation",
                payload={
                    "room_id": room_id,
                    "variation_id": variation_id,
                    "revision": revision,
                },
            )
            job_id = doc["id"]
            queue.enqueue(job_id=job_id, project_id=project_id)
            job_ids.append(job_id)

    logger.info(
        "staging.jobs.regenerate.enqueued project_id=%s count=%d",
        project_id, len(job_ids),
    )
    return {"job_ids": job_ids}


@router.get("/projects/{project_id}/jobs")
async def list_project_jobs(
    project_id: str,
    storage: StagingStorageService = Depends(get_staging_storage),
    store: JobStore = Depends(get_job_store),
):
    """List jobs for a project with status + progress + phase + kind.

    Partition-scoped (single Cosmos partition by ``/project_id``).
    """
    _require_async_queue_enabled()

    project_data = storage.get_project(project_id)
    if not project_data:
        raise HTTPException(status_code=404, detail="Project not found")

    docs = store.list_jobs_by_project(project_id)
    return {"jobs": [_job_summary(d) for d in docs]}


@router.delete("/jobs/{job_id}", status_code=202)
async def cancel_job(
    job_id: str,
    store: JobStore = Depends(get_job_store),
):
    """Flip ``cancel_requested=True`` on a job.

    Per PRD AC: returns 202 even if the job has already reached a
    terminal state (no-op in that case — the worker will never see the
    flag, and the response is informational). 404 only when the id is
    truly unknown.

    Recovers ``project_id`` from the deterministic id format
    ``{project_id}:{room_id}:{variation_id}:{revision}`` so we don't
    have to fan-out across partitions.
    """
    _require_async_queue_enabled()

    parts = job_id.split(":")
    if len(parts) < 4:
        raise HTTPException(status_code=400, detail="Malformed job_id")
    project_id = parts[0]

    doc = store.get_job(job_id, project_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Job not found")

    if doc.get("status") in _TERMINAL_JOB_STATUSES:
        # Already terminal — ack the request but do not modify state.
        # Returning 202 (not 409) per PRD AC: the caller's intent
        # ("please cancel") is honored either way.
        logger.info(
            "staging.jobs.cancel.noop_terminal job_id=%s status=%s",
            job_id, doc.get("status"),
        )
        return {"status": "accepted", "job_id": job_id, "already_terminal": True}

    store.update_job(job_id, project_id, cancel_requested=True)
    logger.info("staging.jobs.cancel.requested job_id=%s", job_id)
    return {"status": "accepted", "job_id": job_id, "already_terminal": False}


# ---------------------------------------------------------------------------
# SSE stream (issue 005 — image-pipeline-and-project-ux-overhaul)
# ---------------------------------------------------------------------------


# Per-session soft cap on concurrent ``/jobs/stream`` connections. The
# count is process-local — each replica enforces its own cap. Front Door
# routes a session to a sticky replica via cookie affinity in the normal
# case; if it doesn't, the limit becomes (cap × replicas) which is still
# fine for runaway-script protection.
_MAX_STREAMS_PER_SESSION = 10
_session_stream_counts: dict[str, int] = {}
_session_stream_lock = asyncio.Lock()

_SSE_HEARTBEAT_INTERVAL_SECONDS = 15.0
_SSE_LOOP_POLL_SECONDS = 0.5


def _extract_session_token(
    session_cookie: Optional[str], access_token: Optional[str]
) -> Optional[str]:
    """Return the cap-key for a stream connection.

    The codebase has no real auth system yet; both inputs are treated as
    opaque strings. Cookie wins when both are present (matches a typical
    browser EventSource flow where the cookie is auto-attached and the
    query param is only a fallback for non-cookie transports).
    """
    if session_cookie:
        return session_cookie
    if access_token:
        return access_token
    return None


async def _acquire_session_slot(token: str) -> bool:
    """Reserve one stream slot for ``token``. Returns False over the cap."""
    async with _session_stream_lock:
        current = _session_stream_counts.get(token, 0)
        if current >= _MAX_STREAMS_PER_SESSION:
            return False
        _session_stream_counts[token] = current + 1
        return True


async def _release_session_slot(token: str) -> None:
    async with _session_stream_lock:
        current = _session_stream_counts.get(token, 0)
        if current <= 1:
            _session_stream_counts.pop(token, None)
        else:
            _session_stream_counts[token] = current - 1


def _format_sse(event: str, data: str) -> bytes:
    """Encode one SSE message frame."""
    return f"event: {event}\ndata: {data}\n\n".encode("utf-8")


async def _sse_event_stream(
    *,
    seed_jobs: list,
    subscription,
    is_disconnected,
    heartbeat_interval: Optional[float] = None,
    poll_interval: Optional[float] = None,
):
    """Inner generator for ``/jobs/stream``, factored out so it can be
    tested directly without going through an ASGI HTTP client (httpx's
    ``ASGITransport`` buffers responses to completion, which deadlocks
    on infinite SSE generators).

    Wire format:
      - one ``event: seed`` carrying ``seed_jobs``
      - one ``event: job`` per item delivered to ``subscription.queue``
      - a ``:heartbeat`` SSE comment line every ``heartbeat_interval``
        seconds while the queue is quiet
    Exits the loop when ``is_disconnected()`` first returns True so
    cleanup in the caller's ``finally`` block runs promptly.
    """
    hb = heartbeat_interval if heartbeat_interval is not None else _SSE_HEARTBEAT_INTERVAL_SECONDS
    pi = poll_interval if poll_interval is not None else _SSE_LOOP_POLL_SECONDS

    yield _format_sse("seed", json.dumps({"jobs": seed_jobs}))
    loop = asyncio.get_running_loop()
    last_heartbeat = loop.time()
    while True:
        if await is_disconnected():
            return
        try:
            item = await asyncio.wait_for(subscription.queue.get(), timeout=pi)
            yield _format_sse("job", json.dumps(_job_summary(item)))
        except asyncio.TimeoutError:
            now = loop.time()
            if now - last_heartbeat >= hb:
                yield b":heartbeat\n\n"
                last_heartbeat = now


@router.get("/projects/{project_id}/jobs/stream")
async def stream_project_jobs(
    project_id: str,
    request: Request,
    access_token: Optional[str] = Query(default=None),
    session_id: Optional[str] = Cookie(default=None),
    storage: StagingStorageService = Depends(get_staging_storage),
    store: JobStore = Depends(get_job_store),
):
    """Server-Sent Events stream of job state for a project.

    Emits one ``event: seed`` carrying the current job list, then one
    ``event: job`` per change-feed delivery for this ``project_id``,
    plus a ``:heartbeat`` comment line every 15s to keep proxies from
    closing the connection during quiet periods.

    Auth: requires either a ``session_id`` cookie or an ``access_token``
    query parameter (per PRD § Frontend transport — the EventSource API
    cannot send custom headers, so the query param is the documented
    escape hatch). The token is opaque to the server and only used as
    the per-session cap key.

    Per-session cap: 10 concurrent streams; the 11th returns 429.

    Headers: ``Cache-Control: no-cache, no-transform`` and
    ``X-Accel-Buffering: no`` so Front Door / nginx-style proxies do
    not buffer the chunked response.

    Note on dependency resolution: the SSE hub is resolved manually
    AFTER the auth / 404 / 503 / 429 gates rather than via ``Depends``
    so an unauthorized request never starts the per-replica change-feed
    pump (the lazy singleton is constructed on first ``get_sse_hub()``
    call, and we don't want to pay that cost on bot traffic).
    """
    _require_async_queue_enabled()

    token = _extract_session_token(session_id, access_token)
    if not token:
        raise HTTPException(
            status_code=401,
            detail="Missing session cookie or access_token",
        )

    project = storage.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if not await _acquire_session_slot(token):
        raise HTTPException(
            status_code=429,
            detail=(
                f"Too many concurrent streams for this session "
                f"(max {_MAX_STREAMS_PER_SESSION})"
            ),
        )

    try:
        hub = await get_sse_hub_dep()
    except Exception:
        # Hub failed to start — release the slot we just acquired so a
        # transient failure doesn't permanently consume the per-session cap.
        await _release_session_slot(token)
        raise

    subscription = await hub.subscribe(project_id)
    seed = [_job_summary(d) for d in store.list_jobs_by_project(project_id)]

    async def _gen():
        try:
            async for chunk in _sse_event_stream(
                seed_jobs=seed,
                subscription=subscription,
                is_disconnected=request.is_disconnected,
            ):
                yield chunk
        finally:
            await subscription.aclose()
            await _release_session_slot(token)
            logger.info(
                "staging.jobs.stream.closed project_id=%s token_hash=%d",
                project_id, hash(token),
            )

    headers = {
        "Cache-Control": "no-cache, no-transform",
        "X-Accel-Buffering": "no",
        "Connection": "keep-alive",
    }
    return StreamingResponse(
        _gen(),
        media_type="text/event-stream",
        headers=headers,
    )
