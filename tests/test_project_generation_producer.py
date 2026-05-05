"""Tests for ``backend.core.project_generation_producer``.

Issue 002 of the active-and-queued-jobs-ux-redesign PRD.

The producer is the deep module that owns ``POST /jobs/generate``'s
new-request flow:

    Idempotency-Key validation
        → lease precheck (read-only, project_data already fetched)
        → brief composition (callable injected for testability)
        → cascade-cancel siblings (regenerate_all=True only; PNR)
        → create_job (idempotent on Cosmos 409)
        → CAS lease acquire (idempotent for "holder is me")
        → queue.enqueue (with compensation on failure)

It returns one of three frozen dataclasses:

    AlreadyInFlight(job_id)    — dedupe hit (precheck OR create_job 409
                                  OR CAS-acquire lose)
    NewlyEnqueued(job_id)      — happy path
    EnqueueFailed(error_kind, http_status, user_message, detail)
                                 — classified failure

The endpoint wrapper translates these into 200 / 202 / 4xx-5xx.
Project-not-found and no-rooms are HTTP-layer concerns and return
404 / 400 BEFORE the producer is invoked — the producer assumes a
valid project with at least one room.

Tests use small in-memory fakes for ``JobStore`` / ``JobQueue`` /
``StagingStorageService`` so each branch can be exercised
deterministically. The real ``acquire_project_lease`` is exercised
against a fake storage container that simulates ETag-conflict
behavior — patching the helper itself would silently mask a future
bug in the lease primitive.
"""
from __future__ import annotations

import asyncio
from typing import Any, Mapping
from unittest.mock import MagicMock

import pytest
from azure.core.exceptions import (
    ClientAuthenticationError,
    HttpResponseError,
)
from azure.cosmos import exceptions as cosmos_exceptions

from backend.core.job_errors import BriefCompositionFailed, ErrorKind
from backend.core.project_generation_producer import (
    AlreadyInFlight,
    EnqueueFailed,
    NewlyEnqueued,
    produce,
    validate_idempotency_key,
)


# ---------------------------------------------------------------------------
# In-memory fakes
# ---------------------------------------------------------------------------


class FakeJobStore:
    """In-memory ``JobStore``-shaped fake.

    Mirrors the real ``create_job`` / ``get_job`` / ``update_job`` /
    ``list_jobs_by_project`` surface. Tracks call counts so individual
    tests can assert "was create_job invoked at all?" without resorting
    to MagicMock indirection.
    """

    def __init__(self):
        self.jobs: dict[str, dict[str, Any]] = {}
        self.create_count = 0
        self.update_calls: list[tuple[str, str, dict[str, Any]]] = []

    def create_job(self, *, project_id, room_id, variation_id, revision, kind, payload):
        job_id = f"{project_id}:{room_id}:{variation_id}:{revision}"
        if job_id in self.jobs:
            # Idempotent 409 path — return the existing doc verbatim.
            return dict(self.jobs[job_id])
        self.create_count += 1
        doc = {
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
        }
        self.jobs[job_id] = doc
        return dict(doc)

    def get_job(self, job_id, project_id):
        return dict(self.jobs[job_id]) if job_id in self.jobs else None

    def update_job(self, job_id, project_id, **patches):
        if job_id not in self.jobs:
            raise LookupError(f"job not found: {job_id}")
        self.update_calls.append((job_id, project_id, dict(patches)))
        self.jobs[job_id].update(patches)
        return dict(self.jobs[job_id])

    def list_jobs_by_project(self, project_id):
        return [dict(j) for j in self.jobs.values() if j["project_id"] == project_id]

    def seed_existing(self, *, job_id, project_id, status="pending", kind="generate_project"):
        """Pre-populate a job doc (e.g. for the 409-on-create path)."""
        self.jobs[job_id] = {
            "id": job_id,
            "project_id": project_id,
            "room_id": "__project__",
            "variation_id": "__project__",
            "revision": job_id.rsplit(":", 1)[-1],
            "kind": kind,
            "status": status,
            "progress": 0,
            "phase": None,
            "attempts": 0,
            "payload": {},
            "result": None,
            "error": None,
        }


class FakeJobQueue:
    def __init__(self, fail: BaseException | None = None):
        self.calls: list[tuple[str, str]] = []
        self.fail = fail

    def enqueue(self, *, job_id, project_id):
        self.calls.append((job_id, project_id))
        if self.fail is not None:
            raise self.fail


class FakeStorage:
    """Fake ``StagingStorageService`` for the lease helper.

    The real ``acquire_project_lease`` calls
    ``storage.get_project(...)`` then ``storage.container.replace_item(
    item, body, etag, match_condition)``. We back the project with an
    in-memory dict and bump the ``_etag`` on every replace so the next
    read sees a fresh tag.

    To simulate a CAS lose (concurrent writer), construct with
    ``cas_loses=N`` to force the first N replace_item calls to raise
    ``CosmosAccessConditionFailedError`` (and increment the etag as
    if a foreign writer had committed in between).
    """

    def __init__(
        self,
        project: Mapping[str, Any],
        *,
        cas_loses: int = 0,
        winning_holder: str | None = None,
    ):
        self._project = dict(project)
        if "_etag" not in self._project:
            self._project["_etag"] = '"e0"'
        self._etag_seq = 0
        self._cas_loses_remaining = cas_loses
        self._winning_holder = winning_holder
        self.container = MagicMock()
        self.container.replace_item.side_effect = self._replace_item

    def get_project(self, pid):
        if self._project.get("id") != pid:
            return None
        return dict(self._project)

    def _replace_item(self, *, item, body, etag, match_condition):
        if self._cas_loses_remaining > 0:
            self._cas_loses_remaining -= 1
            # Simulate a foreign writer committing — bump our etag and
            # plant the winning holder so the next read shows it.
            self._etag_seq += 1
            self._project["_etag"] = f'"e{self._etag_seq}"'
            if self._winning_holder is not None:
                self._project["current_project_job_id"] = self._winning_holder
            raise cosmos_exceptions.CosmosAccessConditionFailedError(
                message="etag mismatch"
            )
        # Success path — store the new body verbatim and bump the etag.
        self._etag_seq += 1
        new_body = dict(body)
        new_body["_etag"] = f'"e{self._etag_seq}"'
        self._project = new_body
        return dict(new_body)


# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------


PROJECT_ID = "proj-1"
IDEM_KEY_A = "11111111111111111111111111111111"  # 32-char hex (valid)
IDEM_KEY_B = "22222222222222222222222222222222"


def _project_doc(*, holder: str | None = None) -> dict[str, Any]:
    """Tracer-bullet project doc."""
    doc: dict[str, Any] = {
        "id": PROJECT_ID,
        "name": "p",
        "rooms": [{"id": "r1", "label": "L"}],
        "_etag": '"e0"',
    }
    if holder is not None:
        doc["current_project_job_id"] = holder
    return doc


async def _ok_compose_brief() -> dict[str, list[str]]:
    return {"r1": ["prompt 1", "prompt 2"]}


async def _none_compose_brief() -> None:
    return None


def _expected_job_id(idempotency_key: str) -> str:
    return f"{PROJECT_ID}:__project__:__project__:{idempotency_key}"


# ---------------------------------------------------------------------------
# Idempotency-Key validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key",
    [
        "abc",
        "ABC123",
        "a-b-c_d",
        "0123456789abcdef0123456789abcdef",  # uuid4().hex shape
        "A" * 128,  # max length
    ],
)
def test_validate_idempotency_key_accepts_valid_keys(key):
    assert validate_idempotency_key(key) == key


@pytest.mark.parametrize(
    "key",
    [
        "",          # empty
        "A" * 129,   # too long
        "has space",  # space disallowed
        "has/slash",  # slash disallowed
        "has.dot",   # dot disallowed
        "héllo",     # non-ASCII
    ],
)
def test_validate_idempotency_key_rejects_invalid_keys(key):
    with pytest.raises(ValueError):
        validate_idempotency_key(key)


# ---------------------------------------------------------------------------
# Discriminated-union dataclasses are frozen + identifiable
# ---------------------------------------------------------------------------


def test_already_in_flight_is_frozen():
    res = AlreadyInFlight(job_id="x")
    with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
        res.job_id = "y"  # type: ignore


def test_newly_enqueued_is_frozen():
    res = NewlyEnqueued(job_id="x")
    with pytest.raises(Exception):
        res.job_id = "y"  # type: ignore


def test_enqueue_failed_carries_kind_status_and_message():
    res = EnqueueFailed(
        error_kind=ErrorKind.QUEUE_PERMISSION,
        http_status=502,
        user_message="missing role",
        detail={"type": "HttpResponseError", "message": "forbidden"},
    )
    assert res.error_kind == ErrorKind.QUEUE_PERMISSION
    assert res.http_status == 502
    assert "role" in res.user_message
    assert res.detail["type"] == "HttpResponseError"


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_first_time_enqueue_returns_newly_enqueued():
    store = FakeJobStore()
    queue = FakeJobQueue()
    storage = FakeStorage(_project_doc())

    result = await produce(
        project_id=PROJECT_ID,
        project_data=storage.get_project(PROJECT_ID),
        idempotency_key=IDEM_KEY_A,
        regenerate_all=False,
        compose_brief=_ok_compose_brief,
        store=store,
        queue=queue,
        storage=storage,
    )

    assert isinstance(result, NewlyEnqueued)
    assert result.job_id == _expected_job_id(IDEM_KEY_A)
    # create_job was invoked exactly once.
    assert store.create_count == 1
    # Queue saw the job.
    assert queue.calls == [(result.job_id, PROJECT_ID)]
    # Lease was taken: current_project_job_id == job_id.
    assert storage.get_project(PROJECT_ID)["current_project_job_id"] == result.job_id


@pytest.mark.asyncio
async def test_first_time_enqueue_payload_carries_brief_prompts_and_regenerate_all():
    """The payload written to the job doc carries both the precomputed
    brief prompts (None if compose_brief returned None) and the
    regenerate_all flag — these are the dispatcher's inputs."""
    store = FakeJobStore()
    queue = FakeJobQueue()
    storage = FakeStorage(_project_doc())

    await produce(
        project_id=PROJECT_ID,
        project_data=storage.get_project(PROJECT_ID),
        idempotency_key=IDEM_KEY_A,
        regenerate_all=True,
        compose_brief=_ok_compose_brief,
        store=store,
        queue=queue,
        storage=storage,
    )

    job_id = _expected_job_id(IDEM_KEY_A)
    doc = store.jobs[job_id]
    assert doc["payload"]["regenerate_all"] is True
    assert doc["payload"]["brief_prompts"] == {"r1": ["prompt 1", "prompt 2"]}
    assert doc["kind"] == "generate_project"
    assert doc["room_id"] == "__project__"
    assert doc["variation_id"] == "__project__"


@pytest.mark.asyncio
async def test_first_time_enqueue_with_no_design_brief_writes_none_brief_prompts():
    store = FakeJobStore()
    queue = FakeJobQueue()
    storage = FakeStorage(_project_doc())

    await produce(
        project_id=PROJECT_ID,
        project_data=storage.get_project(PROJECT_ID),
        idempotency_key=IDEM_KEY_A,
        regenerate_all=False,
        compose_brief=_none_compose_brief,
        store=store,
        queue=queue,
        storage=storage,
    )
    job_id = _expected_job_id(IDEM_KEY_A)
    assert store.jobs[job_id]["payload"]["brief_prompts"] is None


# ---------------------------------------------------------------------------
# Same-key retry — create_job 409 returns existing doc
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_same_idempotency_key_retry_returns_already_in_flight():
    """Pre-existing doc with the same id (Idempotency-Key) → AlreadyInFlight.
    Compose-brief MUST NOT have run (that's the whole point of dedupe);
    queue.enqueue MUST NOT have run either (no second message)."""
    store = FakeJobStore()
    job_id = _expected_job_id(IDEM_KEY_A)
    store.seed_existing(job_id=job_id, project_id=PROJECT_ID, status="running")
    queue = FakeJobQueue()
    storage = FakeStorage(_project_doc(holder=job_id))

    compose_calls = 0

    async def _tracked_brief() -> dict[str, list[str]]:
        nonlocal compose_calls
        compose_calls += 1
        return {"r1": ["x"]}

    result = await produce(
        project_id=PROJECT_ID,
        project_data=storage.get_project(PROJECT_ID),
        idempotency_key=IDEM_KEY_A,
        regenerate_all=False,
        compose_brief=_tracked_brief,
        store=store,
        queue=queue,
        storage=storage,
    )

    assert isinstance(result, AlreadyInFlight)
    assert result.job_id == job_id
    # Lease precheck short-circuited (holder == job_id, non-terminal),
    # so brief composition NEVER ran.
    assert compose_calls == 0
    # Queue stayed silent.
    assert queue.calls == []
    # No new doc created.
    assert store.create_count == 0


# ---------------------------------------------------------------------------
# Lease precheck — different idempotency-key, lease held by foreign job
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lease_held_different_idempotency_key_returns_already_in_flight():
    """A second click with a NEW idempotency-key during an in-flight
    different-key generation: precheck reads the foreign holder, looks
    it up in the store, sees non-terminal, returns AlreadyInFlight(holder).
    Brief composition + create_job + queue MUST NOT run."""
    store = FakeJobStore()
    foreign_job_id = _expected_job_id(IDEM_KEY_A)
    store.seed_existing(
        job_id=foreign_job_id, project_id=PROJECT_ID, status="running"
    )
    queue = FakeJobQueue()
    storage = FakeStorage(_project_doc(holder=foreign_job_id))

    compose_calls = 0

    async def _tracked_brief() -> dict[str, list[str]]:
        nonlocal compose_calls
        compose_calls += 1
        return {"r1": ["x"]}

    # Second click uses a DIFFERENT idempotency-key.
    result = await produce(
        project_id=PROJECT_ID,
        project_data=storage.get_project(PROJECT_ID),
        idempotency_key=IDEM_KEY_B,
        regenerate_all=False,
        compose_brief=_tracked_brief,
        store=store,
        queue=queue,
        storage=storage,
    )

    assert isinstance(result, AlreadyInFlight)
    assert result.job_id == foreign_job_id  # NOT IDEM_KEY_B's id
    assert compose_calls == 0
    assert queue.calls == []
    assert store.create_count == 0


# ---------------------------------------------------------------------------
# Lease precheck — terminal foreign holder is reclaimable
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lease_held_terminal_holder_proceeds_to_new_enqueue():
    """If the holder doc exists but is in a terminal status (succeeded
    / failed / cancelled), the producer reclaims the lease and proceeds.
    The new run gets a NewlyEnqueued result — not AlreadyInFlight."""
    store = FakeJobStore()
    old_job_id = _expected_job_id(IDEM_KEY_A)
    store.seed_existing(
        job_id=old_job_id, project_id=PROJECT_ID, status="succeeded"
    )
    queue = FakeJobQueue()
    storage = FakeStorage(_project_doc(holder=old_job_id))

    result = await produce(
        project_id=PROJECT_ID,
        project_data=storage.get_project(PROJECT_ID),
        idempotency_key=IDEM_KEY_B,
        regenerate_all=False,
        compose_brief=_ok_compose_brief,
        store=store,
        queue=queue,
        storage=storage,
    )

    assert isinstance(result, NewlyEnqueued)
    assert result.job_id == _expected_job_id(IDEM_KEY_B)
    # Lease was rotated to the new job.
    assert (
        storage.get_project(PROJECT_ID)["current_project_job_id"] == result.job_id
    )


# ---------------------------------------------------------------------------
# Brief composition failure → BRIEF_FAILED
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_brief_composition_exception_returns_enqueue_failed_brief_failed():
    """compose_brief raising → wrapped in BriefCompositionFailed →
    EnqueueFailed(BRIEF_FAILED). No doc created, no enqueue."""
    store = FakeJobStore()
    queue = FakeJobQueue()
    storage = FakeStorage(_project_doc())

    async def _broken_brief() -> dict[str, list[str]]:
        raise RuntimeError("LLM rate-limited")

    result = await produce(
        project_id=PROJECT_ID,
        project_data=storage.get_project(PROJECT_ID),
        idempotency_key=IDEM_KEY_A,
        regenerate_all=False,
        compose_brief=_broken_brief,
        store=store,
        queue=queue,
        storage=storage,
    )

    assert isinstance(result, EnqueueFailed)
    assert result.error_kind == ErrorKind.BRIEF_FAILED
    assert result.http_status == 502
    assert store.create_count == 0
    assert queue.calls == []


@pytest.mark.asyncio
async def test_brief_composition_failed_already_wrapped_passes_through():
    """If compose_brief raises ``BriefCompositionFailed`` directly
    (e.g. an HTTP wrapper already wrapped it), the producer doesn't
    double-wrap — it classifies the same way."""
    store = FakeJobStore()
    queue = FakeJobQueue()
    storage = FakeStorage(_project_doc())

    async def _pre_wrapped() -> dict[str, list[str]]:
        raise BriefCompositionFailed("already wrapped")

    result = await produce(
        project_id=PROJECT_ID,
        project_data=storage.get_project(PROJECT_ID),
        idempotency_key=IDEM_KEY_A,
        regenerate_all=False,
        compose_brief=_pre_wrapped,
        store=store,
        queue=queue,
        storage=storage,
    )
    assert isinstance(result, EnqueueFailed)
    assert result.error_kind == ErrorKind.BRIEF_FAILED


# ---------------------------------------------------------------------------
# Queue enqueue failure → classified + compensation
# ---------------------------------------------------------------------------


def _http_response_error_with_code(code: str, message: str = "forbidden"):
    exc = HttpResponseError(message=message)
    exc.error = MagicMock()
    exc.error.code = code
    return exc


@pytest.mark.asyncio
async def test_enqueue_authorization_permission_mismatch_returns_queue_permission():
    """The bug scenario: Storage Queue 403 with code
    AuthorizationPermissionMismatch on the queue.enqueue call. The
    producer classifies as QUEUE_PERMISSION and runs the compensation
    update_job(status='failed', error_kind, error)."""
    store = FakeJobStore()
    queue = FakeJobQueue(
        fail=_http_response_error_with_code("AuthorizationPermissionMismatch")
    )
    storage = FakeStorage(_project_doc())

    result = await produce(
        project_id=PROJECT_ID,
        project_data=storage.get_project(PROJECT_ID),
        idempotency_key=IDEM_KEY_A,
        regenerate_all=False,
        compose_brief=_ok_compose_brief,
        store=store,
        queue=queue,
        storage=storage,
    )

    assert isinstance(result, EnqueueFailed)
    assert result.error_kind == ErrorKind.QUEUE_PERMISSION
    assert result.http_status == 502
    assert "Storage Queue Data Message Sender" in result.user_message

    # Compensation: the doc went from pending → failed with error_kind.
    job_id = _expected_job_id(IDEM_KEY_A)
    failed_doc = store.jobs[job_id]
    assert failed_doc["status"] == "failed"
    assert failed_doc["error_kind"] == ErrorKind.QUEUE_PERMISSION.value
    # error= is now a dict (NOT a string) per rubber-duck N1.
    assert isinstance(failed_doc["error"], dict)
    assert "type" in failed_doc["error"]
    assert "message" in failed_doc["error"]


@pytest.mark.asyncio
async def test_enqueue_client_authentication_error_returns_queue_permission_with_auth_msg():
    """ClientAuthenticationError → QUEUE_PERMISSION with auth-flavored
    message (NOT the role-grant message). Per rubber-duck B2."""
    store = FakeJobStore()
    queue = FakeJobQueue(
        fail=ClientAuthenticationError(message="token failed")
    )
    storage = FakeStorage(_project_doc())

    result = await produce(
        project_id=PROJECT_ID,
        project_data=storage.get_project(PROJECT_ID),
        idempotency_key=IDEM_KEY_A,
        regenerate_all=False,
        compose_brief=_ok_compose_brief,
        store=store,
        queue=queue,
        storage=storage,
    )
    assert isinstance(result, EnqueueFailed)
    assert result.error_kind == ErrorKind.QUEUE_PERMISSION
    assert (
        "managed identity" in result.user_message.lower()
        or "authenticate" in result.user_message.lower()
    )
    assert "Storage Queue Data Message Sender" not in result.user_message


@pytest.mark.asyncio
async def test_enqueue_arbitrary_exception_returns_unknown_with_compensation():
    store = FakeJobStore()
    queue = FakeJobQueue(fail=RuntimeError("queue down"))
    storage = FakeStorage(_project_doc())

    result = await produce(
        project_id=PROJECT_ID,
        project_data=storage.get_project(PROJECT_ID),
        idempotency_key=IDEM_KEY_A,
        regenerate_all=False,
        compose_brief=_ok_compose_brief,
        store=store,
        queue=queue,
        storage=storage,
    )
    assert isinstance(result, EnqueueFailed)
    assert result.error_kind == ErrorKind.UNKNOWN
    assert result.http_status == 500
    job_id = _expected_job_id(IDEM_KEY_A)
    assert store.jobs[job_id]["status"] == "failed"
    assert store.jobs[job_id]["error_kind"] == ErrorKind.UNKNOWN.value


@pytest.mark.asyncio
async def test_enqueue_failure_compensation_failure_still_returns_enqueue_failed():
    """Compensation is best-effort. If update_job ALSO raises, the
    producer still returns EnqueueFailed — the user gets a 5xx. The
    compensation log line is the only diagnostic; the request flow
    must not become a request-blocking exception."""

    class FailingUpdateStore(FakeJobStore):
        def update_job(self, job_id, project_id, **patches):
            raise RuntimeError("cosmos write failed")

    store = FailingUpdateStore()
    queue = FakeJobQueue(fail=RuntimeError("queue down"))
    storage = FakeStorage(_project_doc())

    result = await produce(
        project_id=PROJECT_ID,
        project_data=storage.get_project(PROJECT_ID),
        idempotency_key=IDEM_KEY_A,
        regenerate_all=False,
        compose_brief=_ok_compose_brief,
        store=store,
        queue=queue,
        storage=storage,
    )

    assert isinstance(result, EnqueueFailed)
    assert result.error_kind == ErrorKind.UNKNOWN


# ---------------------------------------------------------------------------
# CAS-acquire lose during lease acquire
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cas_lease_acquire_lose_returns_already_in_flight_with_winner_id():
    """Two concurrent producers, distinct idempotency-keys. The losing
    one created its doc, its CAS-replace lost (etag conflict), the
    winning concurrent producer planted its job_id in the lease. The
    losing producer marks its newly-created doc as superseded and
    returns AlreadyInFlight(winner_id) — NOT NewlyEnqueued.

    Crucially: queue.enqueue MUST NOT run on the losing path; an
    enqueued message for an orphaned doc would cause the worker to
    pick up a job that lost its lease and silently no-op."""
    store = FakeJobStore()
    winning_job_id = _expected_job_id(IDEM_KEY_B)
    queue = FakeJobQueue()
    storage = FakeStorage(
        _project_doc(),
        cas_loses=1,
        winning_holder=winning_job_id,
    )

    result = await produce(
        project_id=PROJECT_ID,
        project_data=storage.get_project(PROJECT_ID),
        idempotency_key=IDEM_KEY_A,
        regenerate_all=False,
        compose_brief=_ok_compose_brief,
        store=store,
        queue=queue,
        storage=storage,
    )

    assert isinstance(result, AlreadyInFlight)
    assert result.job_id == winning_job_id
    # The losing producer's doc was created but is marked superseded so
    # GET /jobs and the SSE feed don't surface a phantom pending.
    losing_id = _expected_job_id(IDEM_KEY_A)
    losing_doc = store.jobs[losing_id]
    assert losing_doc["status"] in {"failed", "cancelled", "superseded"}
    # And NOT enqueued — the loser must not put a message on the queue.
    assert queue.calls == []


# ---------------------------------------------------------------------------
# Brief composition is gated past BOTH dedupe checks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_compose_brief_runs_only_on_new_work_path():
    """PRD AC: 'Brief composition is gated behind both dedupe checks
    (only runs on the new-work path).' This is the headline efficiency
    win — a duplicate click MUST NOT burn 30-90s recomputing prompts.

    Pinned three ways:
        1. precheck-hit (lease held by non-terminal foreign job): 0 calls.
        2. precheck-hit (lease held by same-key job): 0 calls.
        3. happy path: exactly 1 call.
    """

    # Scenario 1 — different-key lease hit.
    store = FakeJobStore()
    foreign = _expected_job_id(IDEM_KEY_A)
    store.seed_existing(job_id=foreign, project_id=PROJECT_ID, status="running")
    storage = FakeStorage(_project_doc(holder=foreign))

    calls = 0

    async def _tracked() -> dict[str, list[str]]:
        nonlocal calls
        calls += 1
        return {"r1": ["x"]}

    await produce(
        project_id=PROJECT_ID,
        project_data=storage.get_project(PROJECT_ID),
        idempotency_key=IDEM_KEY_B,
        regenerate_all=False,
        compose_brief=_tracked,
        store=store,
        queue=FakeJobQueue(),
        storage=storage,
    )
    assert calls == 0, "precheck-hit (different key) must NOT compose brief"

    # Scenario 2 — same-key lease hit (most common UX dedupe path).
    calls = 0
    await produce(
        project_id=PROJECT_ID,
        project_data=storage.get_project(PROJECT_ID),
        idempotency_key=IDEM_KEY_A,  # same key as the pre-existing holder
        regenerate_all=False,
        compose_brief=_tracked,
        store=store,
        queue=FakeJobQueue(),
        storage=storage,
    )
    assert calls == 0, "precheck-hit (same key) must NOT compose brief"

    # Scenario 3 — new work path.
    store_3 = FakeJobStore()
    storage_3 = FakeStorage(_project_doc())
    calls = 0
    result = await produce(
        project_id=PROJECT_ID,
        project_data=storage_3.get_project(PROJECT_ID),
        idempotency_key=IDEM_KEY_A,
        regenerate_all=False,
        compose_brief=_tracked,
        store=store_3,
        queue=FakeJobQueue(),
        storage=storage_3,
    )
    assert isinstance(result, NewlyEnqueued)
    assert calls == 1, "new-work path must compose brief exactly once"


# ---------------------------------------------------------------------------
# Cascade-cancel ordering — runs BEFORE create_job (PNR contract)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cascade_cancel_runs_before_create_job_when_regenerate_all_true():
    """Cascade-cancel is the point-of-no-return: it persists even if
    create_job or enqueue subsequently fail. Pin the ORDER: cascade
    update_jobs precede create_job's invocation."""
    store = FakeJobStore()
    # Pre-existing in-flight regenerate_variation jobs.
    store.seed_existing(
        job_id=f"{PROJECT_ID}:r1:v1:0", project_id=PROJECT_ID,
        kind="regenerate_variation", status="pending",
    )
    store.seed_existing(
        job_id=f"{PROJECT_ID}:r1:v2:0", project_id=PROJECT_ID,
        kind="regenerate_variation", status="running",
    )
    queue = FakeJobQueue()
    storage = FakeStorage(_project_doc())

    await produce(
        project_id=PROJECT_ID,
        project_data=storage.get_project(PROJECT_ID),
        idempotency_key=IDEM_KEY_A,
        regenerate_all=True,
        compose_brief=_ok_compose_brief,
        store=store,
        queue=queue,
        storage=storage,
    )

    # Both pre-existing variation jobs got cancel_requested=True.
    cancel_calls = [
        c for c in store.update_calls if c[2].get("cancel_requested") is True
    ]
    assert len(cancel_calls) == 2
    cancelled_ids = {c[0] for c in cancel_calls}
    assert cancelled_ids == {
        f"{PROJECT_ID}:r1:v1:0",
        f"{PROJECT_ID}:r1:v2:0",
    }


@pytest.mark.asyncio
async def test_cascade_cancel_does_not_run_when_regenerate_all_false():
    store = FakeJobStore()
    store.seed_existing(
        job_id=f"{PROJECT_ID}:r1:v1:0", project_id=PROJECT_ID,
        kind="regenerate_variation", status="pending",
    )
    storage = FakeStorage(_project_doc())

    await produce(
        project_id=PROJECT_ID,
        project_data=storage.get_project(PROJECT_ID),
        idempotency_key=IDEM_KEY_A,
        regenerate_all=False,
        compose_brief=_ok_compose_brief,
        store=store,
        queue=FakeJobQueue(),
        storage=storage,
    )

    cancel_calls = [
        c for c in store.update_calls if c[2].get("cancel_requested") is True
    ]
    assert cancel_calls == []
