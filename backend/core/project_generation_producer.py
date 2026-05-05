"""Project-generation producer.

Issue 002 of the active-and-queued-jobs-ux-redesign PRD.

A deep module that owns the new-request path of
``POST /projects/{id}/jobs/generate``. The HTTP wrapper validates
the project exists, validates rooms are uploaded, and builds the
``compose_brief`` callable, then hands off to ``produce()`` which
returns a discriminated union the wrapper translates into 200 / 202
/ 4xx-5xx.

Flow (the contract; see the rubber-duck-revised plan)::

    1. Lease precheck — read ``current_project_job_id`` from the
       project doc that the wrapper already fetched. If it points at
       a non-terminal job in the store, return AlreadyInFlight(holder)
       WITHOUT composing the brief or creating a new doc. This is the
       "second click during in-flight different-key generation" case.

    2. Same-key precheck — read store.get_job for the deterministic id
       built from the caller's Idempotency-Key. If the doc exists, the
       caller is retrying with the same key (transport-layer retry).
       Return AlreadyInFlight(existing_id). This catches the dedupe
       case the lease precheck misses (lease released after worker
       finished; same key replayed).

    3. Brief composition — invoke the injected async ``compose_brief``
       callable. Any exception is wrapped in ``BriefCompositionFailed``
       (preserving ``__cause__``), classified, and returned as
       ``EnqueueFailed(BRIEF_FAILED, 502, ...)``. NO side effects up
       to this point.

    4. Cascade cancel (regenerate_all=True only) — POINT OF NO RETURN.
       Mutations PERSIST even if create_job / CAS / enqueue fail.
       Rolling back ``cancel_requested=False`` would race a legitimate
       user-initiated cancel.

    5. create_job — idempotent on Cosmos 409. Returns the doc; the
       fact that we already same-key-prechecked means a 409 here is
       a tiny race window (concurrent producer with the SAME key) —
       still safe because the existing doc IS the deduped result.

    6. CAS lease acquire — ``acquire_project_lease`` is idempotent on
       "holder is me" so a same-key replay that gets past precheck
       will re-acquire its own lease silently. On CAS lose (foreign
       writer planted a different holder concurrent with our flow),
       mark our newly-created doc as ``status=failed`` + error
       ``Superseded`` (NOT cancelled — preserves the user-cancel
       semantic) and return AlreadyInFlight(winner). Queue MUST NOT
       be enqueued on the lose path — that would cause the worker to
       pick up a doc that lost its lease and silently no-op.

    7. queue.enqueue — on failure, classify the exception, run a
       best-effort compensation update (status=failed + error_kind +
       error dict), return EnqueueFailed. Compensation failure does
       NOT raise — the caller's response shape is the only signal,
       and the next reconcile sweep cleans up if needed.

    8. NewlyEnqueued.

The Idempotency-Key is the dedupe knob. Frontend mints
``crypto.randomUUID()`` per ``enqueueProjectGeneration`` call so two
distinct clicks always produce two distinct keys (and therefore two
distinct deterministic ids). The same key only repeats on a transport-
layer retry, where same-id semantics are exactly what we want.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Mapping, Optional, Union

from backend.core.job_errors import (
    BriefCompositionFailed,
    ErrorKind,
    classify,
)
from backend.core.project_lease import (
    TERMINAL_JOB_STATUSES,
    acquire_project_lease,
    cascade_cancel_variation_jobs,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Idempotency-Key validation
# ---------------------------------------------------------------------------

_IDEMPOTENCY_KEY_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


def validate_idempotency_key(key: str) -> str:
    """Return ``key`` unchanged if it matches the idempotency-key regex,
    else raise ``ValueError``.

    The regex (``^[A-Za-z0-9_-]{1,128}$``) intentionally rejects:

    * empty strings (would collapse the deterministic-id namespace);
    * strings longer than 128 chars (oversized Cosmos id risk);
    * any character outside ``[A-Za-z0-9_-]`` (so the key is safe to
      f""-interpolate into the deterministic-id template without any
      escaping — and so a malicious caller can't smuggle a colon and
      forge a different doc id).

    Per RFC, ``Idempotency-Key`` values are case-sensitive — do NOT
    ``.lower()`` here. ``crypto.randomUUID()`` produces a lowercase
    36-char hyphenated string; ``uuid4().hex`` produces a 32-char
    lowercase hex string. Both pass.
    """
    if not isinstance(key, str) or not _IDEMPOTENCY_KEY_RE.fullmatch(key):
        raise ValueError(
            f"Idempotency-Key must match {_IDEMPOTENCY_KEY_RE.pattern}; "
            f"got {key!r}"
        )
    return key


# ---------------------------------------------------------------------------
# Discriminated-union return type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AlreadyInFlight:
    """Dedupe hit. The HTTP wrapper returns 200 ``{job_id, already_in_flight: true}``."""
    job_id: str


@dataclass(frozen=True)
class NewlyEnqueued:
    """Happy path. The HTTP wrapper returns 202 ``{job_id, already_in_flight: false}``."""
    job_id: str


@dataclass(frozen=True)
class EnqueueFailed:
    """Classified failure. The HTTP wrapper returns ``http_status`` with
    ``{error_kind, user_message, detail}``."""
    error_kind: ErrorKind
    http_status: int
    user_message: str
    detail: Optional[Any] = None


ProducerResult = Union[AlreadyInFlight, NewlyEnqueued, EnqueueFailed]


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


_PROJECT_SENTINEL = "__project__"


def _build_job_id(project_id: str, idempotency_key: str) -> str:
    return f"{project_id}:{_PROJECT_SENTINEL}:{_PROJECT_SENTINEL}:{idempotency_key}"


def _exception_detail(exc: BaseException) -> dict[str, str]:
    """Build the ``error`` field written to the job doc + the
    ``detail`` field returned to the HTTP caller. The dict shape
    matches what the worker writes on terminal-failure transitions
    (``error: {type, message}``) so consumers only ever parse one
    shape."""
    return {
        "type": type(exc).__name__,
        "message": str(exc),
    }


async def _compose_brief_or_fail(
    compose_brief: Callable[[], Awaitable[Optional[dict[str, list[str]]]]],
) -> tuple[Optional[dict[str, list[str]]], Optional[EnqueueFailed]]:
    """Invoke ``compose_brief`` and translate any exception into the
    classified discriminated-union value. Returns ``(prompts, None)``
    on success; ``(None, EnqueueFailed(...))`` on any exception."""
    try:
        prompts = await compose_brief()
        return prompts, None
    except BriefCompositionFailed as exc:
        kind, status, msg = classify(exc)
        return None, EnqueueFailed(
            error_kind=kind,
            http_status=status,
            user_message=msg,
            detail=_exception_detail(exc),
        )
    except Exception as exc:
        wrapped = BriefCompositionFailed(str(exc) or type(exc).__name__)
        wrapped.__cause__ = exc
        kind, status, msg = classify(wrapped)
        return None, EnqueueFailed(
            error_kind=kind,
            http_status=status,
            user_message=msg,
            detail=_exception_detail(exc),
        )


def _try_compensation(
    *,
    store: Any,
    job_id: str,
    project_id: str,
    error_kind: ErrorKind,
    exc: BaseException,
) -> None:
    """Best-effort compensation: mark the job ``status=failed`` with
    structured ``error_kind`` + ``error`` so the SSE feed and
    GET /jobs surface the failure. A logging-only failure here MUST
    NOT propagate."""
    try:
        store.update_job(
            job_id, project_id,
            status="failed",
            error_kind=error_kind.value,
            error=_exception_detail(exc),
        )
    except Exception as compensation_exc:
        logger.warning(
            "producer.compensation_failed job_id=%s error=%s",
            job_id, compensation_exc,
        )


def _try_supersede(
    *,
    store: Any,
    job_id: str,
    project_id: str,
    winner_job_id: Optional[str],
) -> None:
    """Mark a doc that lost the CAS lease race as ``status=failed`` with
    a Superseded error type. Distinct from ``cancelled`` (the
    user-cancel state) so the front-end doesn't show "cancelled by
    user" copy for a system-resolved race."""
    try:
        store.update_job(
            job_id, project_id,
            status="failed",
            error_kind=ErrorKind.UNKNOWN.value,
            error={
                "type": "Superseded",
                "message": (
                    f"Lost lease race against {winner_job_id}"
                    if winner_job_id
                    else "Lost lease race"
                ),
            },
        )
    except Exception as exc:
        logger.warning(
            "producer.supersede_failed job_id=%s error=%s",
            job_id, exc,
        )


# ---------------------------------------------------------------------------
# produce
# ---------------------------------------------------------------------------


async def produce(
    *,
    project_id: str,
    project_data: Mapping[str, Any],
    idempotency_key: str,
    regenerate_all: bool,
    compose_brief: Callable[[], Awaitable[Optional[dict[str, list[str]]]]],
    store: Any,
    queue: Any,
    storage: Any,
) -> ProducerResult:
    """Produce a project-generation job (or detect dedupe / failure).

    Args:
        project_id: The project id.
        project_data: The project doc (already fetched by the HTTP
            wrapper for its 404 check). Used for the read-only lease
            precheck. ``acquire_project_lease`` does its own re-fetch
            so a CAS race is correctly detected.
        idempotency_key: Validated key (see ``validate_idempotency_key``).
        regenerate_all: When True, cascade-cancel sibling
            ``regenerate_variation`` jobs BEFORE creating the new doc.
            Point of no return — cancellations persist on later failure.
        compose_brief: Async callable returning the precomputed
            brief prompts (or None when the project lacks
            design_brief / analyses). Closes over the project + LLM
            settings.
        store: ``JobStore``-shaped dependency.
        queue: ``JobQueue``-shaped dependency.
        storage: ``StagingStorageService``-shaped dependency (used
            by the lease helper for ETag-protected writes).

    Returns one of ``AlreadyInFlight``, ``NewlyEnqueued``, or
    ``EnqueueFailed``.
    """
    holder = project_data.get("current_project_job_id")
    if holder:
        holder_doc = store.get_job(holder, project_id)
        if (
            holder_doc is not None
            and holder_doc.get("status") not in TERMINAL_JOB_STATUSES
        ):
            logger.info(
                "producer.dedupe.lease_held project_id=%s holder=%s",
                project_id, holder,
            )
            return AlreadyInFlight(job_id=holder)

    job_id = _build_job_id(project_id, idempotency_key)

    existing = store.get_job(job_id, project_id)
    if existing is not None:
        logger.info(
            "producer.dedupe.same_key project_id=%s job_id=%s status=%s",
            project_id, job_id, existing.get("status"),
        )
        return AlreadyInFlight(job_id=job_id)

    brief_prompts, brief_failure = await _compose_brief_or_fail(compose_brief)
    if brief_failure is not None:
        logger.info(
            "producer.brief_failed project_id=%s error_kind=%s",
            project_id, brief_failure.error_kind.value,
        )
        return brief_failure

    if regenerate_all:
        cancelled = cascade_cancel_variation_jobs(
            store=store, project_id=project_id
        )
        logger.info(
            "producer.cascade_cancel project_id=%s cancelled=%d",
            project_id, cancelled,
        )

    try:
        store.create_job(
            project_id=project_id,
            room_id=_PROJECT_SENTINEL,
            variation_id=_PROJECT_SENTINEL,
            revision=idempotency_key,
            kind="generate_project",
            payload={
                "regenerate_all": regenerate_all,
                "brief_prompts": brief_prompts,
            },
        )
    except Exception as exc:
        kind, status, msg = classify(exc)
        logger.warning(
            "producer.create_job_failed project_id=%s error_kind=%s",
            project_id, kind.value,
        )
        return EnqueueFailed(
            error_kind=kind,
            http_status=status,
            user_message=msg,
            detail=_exception_detail(exc),
        )

    try:
        acquired = acquire_project_lease(
            storage=storage, store=store,
            project_id=project_id, job_id=job_id,
        )
    except Exception as exc:
        kind, status, msg = classify(exc)
        _try_compensation(
            store=store, job_id=job_id, project_id=project_id,
            error_kind=kind, exc=exc,
        )
        return EnqueueFailed(
            error_kind=kind,
            http_status=status,
            user_message=msg,
            detail=_exception_detail(exc),
        )

    if not acquired:
        winner: Optional[str] = None
        try:
            fresh = storage.get_project(project_id)
            if fresh is not None:
                winner = fresh.get("current_project_job_id")
        except Exception as exc:
            logger.warning(
                "producer.lose_winner_lookup_failed project_id=%s error=%s",
                project_id, exc,
            )
        _try_supersede(
            store=store, job_id=job_id, project_id=project_id,
            winner_job_id=winner,
        )
        logger.info(
            "producer.cas_lose project_id=%s loser=%s winner=%s",
            project_id, job_id, winner,
        )
        return AlreadyInFlight(job_id=winner or job_id)

    try:
        queue.enqueue(job_id=job_id, project_id=project_id)
    except Exception as exc:
        kind, status, msg = classify(exc)
        _try_compensation(
            store=store, job_id=job_id, project_id=project_id,
            error_kind=kind, exc=exc,
        )
        logger.warning(
            "producer.enqueue_failed project_id=%s job_id=%s error_kind=%s",
            project_id, job_id, kind.value,
        )
        return EnqueueFailed(
            error_kind=kind,
            http_status=status,
            user_message=msg,
            detail=_exception_detail(exc),
        )

    logger.info(
        "producer.enqueued project_id=%s job_id=%s regenerate_all=%s",
        project_id, job_id, regenerate_all,
    )
    return NewlyEnqueued(job_id=job_id)


__all__ = [
    "AlreadyInFlight",
    "EnqueueFailed",
    "NewlyEnqueued",
    "ProducerResult",
    "produce",
    "validate_idempotency_key",
]
