"""``JobStore`` — deep persistence module wrapping the Cosmos ``jobs``
container.

Single responsibility: durable per-job state for the async image-job
queue. The queue carries only a small reference (the deterministic job
id + project_id) — all real state (status, progress, phase, attempts,
payload, result, error) lives here so the worker can survive replica
churn and resume jobs via visibility-timeout re-delivery.

Public surface (per PRD § JobStore + issue 002 AC):

  - ``deterministic_job_id(project_id, room_id, variation_id, revision)``
    — colon-joined id used as both the Cosmos doc id AND the queue
    payload reference. Stable input → stable id, so a retried enqueue
    is idempotent end-to-end (queue dedupes by message body too, but
    the load-bearing dedupe is the Cosmos ``If-None-Match: *`` insert).

  - ``create_job(...)`` — idempotent insert. Re-creating the same
    deterministic id returns the EXISTING doc rather than raising;
    callers (the regenerate endpoint, retry paths) can fire-and-forget.

  - ``get_job`` / ``update_job`` / ``list_jobs_by_project`` — standard
    read/patch/list. ``list`` is partition-scoped (no cross-partition
    fan-out) since the partition key is ``/project_id``.

  - ``subscribe_change_feed`` — yields ``(items, continuation_token)``
    batches. ``SSEHub`` (issue 005) consumes this once per replica and
    fans events out to per-project EventSource subscribers.

Auth is managed-identity only (matches the rest of this codebase).
Cosmos key fallback exists ONLY for local dev parity with the existing
``StagingStorageService`` — production uses ``DefaultAzureCredential``.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Iterator, Optional

from azure.cosmos import ContainerProxy, CosmosClient, exceptions
from azure.identity import DefaultAzureCredential

from backend.core.config import settings

logger = logging.getLogger(__name__)


# Allow-list of fields that ``update_job`` may patch. Anything else is a
# programming error (e.g. typo of ``progres``) — fail fast rather than
# silently writing garbage into Cosmos.
_UPDATABLE_FIELDS = frozenset(
    {
        "status",
        "progress",
        "phase",
        "attempts",
        "payload",
        "result",
        "error",
        "cancel_requested",
    }
)

JOBS_CONTAINER_ID = "jobs"


def deterministic_job_id(
    project_id: str, room_id: str, variation_id: str, revision: int
) -> str:
    """Return the canonical deterministic id for a job.

    Format: ``{project_id}:{room_id}:{variation_id}:{revision}``.

    Used as the Cosmos doc id (so ``create_item`` with this id will 409
    on a duplicate enqueue, which ``create_job`` swallows). Also used
    as the queue message body so the worker can look up state without
    carrying it in the queue.
    """
    return f"{project_id}:{room_id}:{variation_id}:{revision}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobStore:
    """Cosmos-backed store for image-pipeline job docs.

    Construct with ``container=`` for tests (mock or emulator). Default
    construction uses managed identity to reach the configured Cosmos
    account and creates the ``jobs`` container if missing (partition
    key ``/project_id``).
    """

    def __init__(self, container: Optional[ContainerProxy] = None):
        if container is not None:
            self.container = container
            return

        cosmos_key = getattr(settings, "AZURE_COSMOS_DB_KEY", None) or None
        if cosmos_key:
            client = CosmosClient(
                url=settings.AZURE_COSMOS_DB_ENDPOINT, credential=cosmos_key
            )
        else:
            client = CosmosClient(
                url=settings.AZURE_COSMOS_DB_ENDPOINT,
                credential=DefaultAzureCredential(),
            )
        database = client.get_database_client(settings.AZURE_COSMOS_DB_ID)
        self.container = database.create_container_if_not_exists(
            id=JOBS_CONTAINER_ID,
            partition_key={"paths": ["/project_id"], "kind": "Hash"},
        )

    # ------------------------------------------------------------------
    # Create
    # ------------------------------------------------------------------

    def create_job(
        self,
        *,
        project_id: str,
        room_id: str,
        variation_id: str,
        revision: int,
        kind: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Idempotently create a job doc keyed by the deterministic id.

        On 409 Conflict (the doc already exists from a previous enqueue
        of the exact same logical job), reads and returns the existing
        doc rather than raising — making callers safe to retry.
        """
        job_id = deterministic_job_id(project_id, room_id, variation_id, revision)
        now = _now_iso()
        doc: dict[str, Any] = {
            "id": job_id,
            "project_id": project_id,
            "room_id": room_id,
            "variation_id": variation_id,
            "revision": revision,
            "kind": kind,
            "status": "pending",
            "progress": 0,
            "phase": None,
            "attempts": 0,
            "payload": payload,
            "result": None,
            "error": None,
            "created_at": now,
            "updated_at": now,
        }
        try:
            return self.container.create_item(body=doc)
        except exceptions.CosmosResourceExistsError:
            existing = self.container.read_item(item=job_id, partition_key=project_id)
            logger.info(
                "create_job idempotent hit: job_id=%s already exists (status=%s)",
                job_id,
                existing.get("status"),
            )
            return existing

    # ------------------------------------------------------------------
    # Read / update / list
    # ------------------------------------------------------------------

    def get_job(self, job_id: str, project_id: str) -> Optional[dict[str, Any]]:
        try:
            return self.container.read_item(item=job_id, partition_key=project_id)
        except exceptions.CosmosResourceNotFoundError:
            return None

    def update_job(
        self, job_id: str, project_id: str, **patches: Any
    ) -> dict[str, Any]:
        """Read-modify-write patch. Whitelisted fields only.

        Raises ``ValueError`` for unknown fields (programming error) and
        ``LookupError`` if the job doesn't exist.
        """
        unknown = set(patches) - _UPDATABLE_FIELDS
        if unknown:
            raise ValueError(f"unknown update fields: {sorted(unknown)}")
        try:
            existing = self.container.read_item(
                item=job_id, partition_key=project_id
            )
        except exceptions.CosmosResourceNotFoundError as exc:
            raise LookupError(f"job not found: {job_id}") from exc
        existing.update(patches)
        existing["updated_at"] = _now_iso()
        return self.container.replace_item(item=job_id, body=existing)

    def list_jobs_by_project(self, project_id: str) -> list[dict[str, Any]]:
        """Return all jobs for a project, newest first.

        Partition-scoped — the container is partitioned on
        ``/project_id`` so this is a single-partition query.
        """
        query = (
            "SELECT * FROM c WHERE c.project_id = @pid ORDER BY c.created_at DESC"
        )
        params = [{"name": "@pid", "value": project_id}]
        items = self.container.query_items(
            query=query,
            parameters=params,
            partition_key=project_id,
        )
        return list(items)

    # ------------------------------------------------------------------
    # Change feed
    # ------------------------------------------------------------------

    def subscribe_change_feed(
        self,
        start_time: Optional[str] = None,
        *,
        continuation: Optional[str] = None,
    ) -> Iterator[tuple[list[dict[str, Any]], Optional[str]]]:
        """Yield ``(items, continuation_token)`` batches from the change
        feed across all partitions.

        Exactly one of the mutually-exclusive Cosmos resume kwargs
        (``continuation``, ``start_time``, or ``is_start_from_beginning``)
        is forwarded to the SDK, in that priority order. Passing more
        than one raises ``ValueError`` from the SDK
        (``is_start_from_beginning and start_time are exclusive``) which
        is the bug this method's contract exists to prevent.

        SSEHub uses the continuation token to resume after replica
        restarts so no events are missed across the gap.
        """
        kwargs: dict[str, Any]
        if continuation:
            kwargs = {"continuation": continuation}
        elif start_time:
            kwargs = {"start_time": start_time}
        else:
            kwargs = {"is_start_from_beginning": True}
        iterator = self.container.query_items_change_feed(**kwargs)
        for page in iterator.by_page():
            items = list(page)
            yield items, _extract_continuation_token(iterator)


def _extract_continuation_token(iterator: Any) -> Optional[str]:
    """Pull the continuation token from a Cosmos change-feed iterator.

    The Cosmos SDK exposes the resume token via ``response_headers["etag"]``
    after a ``by_page()`` step; older code paths and some mocks expose it
    as ``iterator.continuation_token``. Prefer the header, fall back to
    the attribute, then ``None``.
    """
    headers = getattr(iterator, "response_headers", None)
    if headers:
        try:
            etag = headers.get("etag")
        except AttributeError:
            etag = None
        if etag:
            return etag
    return getattr(iterator, "continuation_token", None)
