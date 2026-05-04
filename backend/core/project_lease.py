"""Project-level lease + cascade-cancel helpers.

Three pure functions consumed by issue 005 (project dispatcher) and
issue 006 (POST endpoint) of the project-generation-async-queue-cutover
PRD. All dependencies (``storage``, ``store``) are passed in as
keyword-only arguments so the helpers stay easy to test under fakes
and so the worker bootstrap (``backend.worker_main.build_worker``)
can supply them once at startup.

Lease semantics
---------------
The lease is recorded on the project doc as
``current_project_job_id``. ``acquire_project_lease`` writes it via
Cosmos optimistic-concurrency (ETag ``IfNotModified``); a job is
eligible to take ownership if any of the following hold:

* the field is unset, or
* the field already equals the caller's ``job_id`` (idempotent skip),
  in which case we deliberately *skip* the CAS write to save a
  round-trip — the lease is already ours.
* the foreign holder's job doc is in a terminal status
  (``succeeded`` / ``failed`` / ``cancelled``).

A foreign holder whose job doc is *missing* from ``JobStore`` is
treated as **busy**, not reclaimable. A missing doc is not proof the
work finished — it could be a transient read anomaly or eventual
consistency. Failing closed prevents two project-generation runs
ever holding the lease at the same time.

``release_project_lease`` clears the field only if we still hold
the lease. On a single ETag conflict it re-reads the project and
retries; if still self-owned the retry succeeds with the fresh
ETag. If the re-read shows a foreign holder we no-op (someone else
owns it now and we will not stomp on their state). Two consecutive
ETag conflicts → give up; the next dispatcher's takeover path will
reclaim the lease via the terminal-status route.

Cascade-cancel semantics
------------------------
``cascade_cancel_variation_jobs`` finds every non-terminal
``regenerate_variation`` job in the project and sets
``cancel_requested=True`` on the JobStore doc. The worker
(``JobWorker._handle``) already polls ``cancel_requested`` at pickup
(line 179) and during dispatch via ``is_cancelled()`` (line 218),
and is the canonical owner of the terminal ``status="cancelled"``
transition. Setting ``status="cancelled"`` from this helper would
race against an in-flight worker that just transitioned to
``succeeded`` — using ``cancel_requested`` only makes the helper
safe to run at any moment regardless of dispatch state.
"""
from __future__ import annotations

from typing import Any, Mapping

from azure.core import MatchConditions
from azure.cosmos import exceptions as cosmos_exceptions

TERMINAL_JOB_STATUSES: frozenset[str] = frozenset(
    {"succeeded", "failed", "cancelled"}
)

_LEASE_FIELD = "current_project_job_id"


def _holder_is_terminal(store: Any, *, holder: str, project_id: str) -> bool:
    """A foreign holder is reclaimable iff its job doc exists AND
    has a terminal status. Missing doc → return False (busy)."""
    holder_job: Mapping[str, Any] | None = store.get_job(holder, project_id)
    if holder_job is None:
        return False
    return holder_job.get("status") in TERMINAL_JOB_STATUSES


def acquire_project_lease(
    *,
    storage: Any,
    store: Any,
    project_id: str,
    job_id: str,
) -> bool:
    """Attempt to take the project-generation lease for ``job_id``.

    Returns
    -------
    bool
        ``True`` if the caller now (or already) holds the lease;
        ``False`` if the lease is held by a non-terminal foreign job
        or if the CAS write lost a race against a concurrent writer.

    Raises
    ------
    ValueError
        If the project doc does not exist.
    """
    project: Mapping[str, Any] | None = storage.get_project(project_id)
    if project is None:
        raise ValueError(f"Project not found: {project_id}")

    holder: str | None = project.get(_LEASE_FIELD)

    if holder == job_id:
        return True

    if holder is not None and not _holder_is_terminal(
        store, holder=holder, project_id=project_id
    ):
        return False

    body = dict(project)
    body[_LEASE_FIELD] = job_id

    try:
        storage.container.replace_item(
            item=project_id,
            body=body,
            etag=project["_etag"],
            match_condition=MatchConditions.IfNotModified,
        )
    except cosmos_exceptions.CosmosAccessConditionFailedError:
        return False

    return True


def release_project_lease(
    *,
    storage: Any,
    project_id: str,
    job_id: str,
) -> bool:
    """Clear ``current_project_job_id`` iff it still equals ``job_id``.

    Performs at most one ETag-conflict retry; further conflicts mean
    the system is contended enough that the next dispatcher's
    takeover path is the right resolution.

    Returns
    -------
    bool
        ``True`` if the field was successfully cleared. ``False`` if
        the project is missing, a foreign holder owns the lease, or
        both CAS attempts lost their race.
    """
    project: Mapping[str, Any] | None = storage.get_project(project_id)
    if project is None or project.get(_LEASE_FIELD) != job_id:
        return False

    if _try_clear_lease(storage, project=project, project_id=project_id):
        return True

    # One retry on ETag conflict — re-read for fresh ETag and check
    # we still own the lease before attempting again.
    fresh: Mapping[str, Any] | None = storage.get_project(project_id)
    if fresh is None or fresh.get(_LEASE_FIELD) != job_id:
        return False

    return _try_clear_lease(storage, project=fresh, project_id=project_id)


def _try_clear_lease(
    storage: Any, *, project: Mapping[str, Any], project_id: str
) -> bool:
    body = dict(project)
    body[_LEASE_FIELD] = None
    try:
        storage.container.replace_item(
            item=project_id,
            body=body,
            etag=project["_etag"],
            match_condition=MatchConditions.IfNotModified,
        )
    except cosmos_exceptions.CosmosAccessConditionFailedError:
        return False
    return True


def cascade_cancel_variation_jobs(*, store: Any, project_id: str) -> int:
    """Set ``cancel_requested=True`` on every non-terminal
    ``regenerate_variation`` job for ``project_id``.

    The terminal ``status="cancelled"`` transition is the worker's
    responsibility — it polls ``cancel_requested`` at pickup and
    during dispatch and routes to ``complete()`` when set.

    Returns
    -------
    int
        Number of jobs touched.
    """
    cancelled = 0
    for job in store.list_jobs_by_project(project_id):
        if job.get("kind") != "regenerate_variation":
            continue
        if job.get("status") in TERMINAL_JOB_STATUSES:
            continue
        store.update_job(job["id"], project_id, cancel_requested=True)
        cancelled += 1
    return cancelled
