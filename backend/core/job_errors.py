"""Producer / worker error classification.

Issue 002 of the active-and-queued-jobs-ux-redesign PRD.

Single source of truth for "what user-facing kind is this exception?".
Both the project-generation producer (HTTP 4xx/5xx response body) and
the worker terminal-failure path (Cosmos ``error_kind`` field on the
job document) call ``classify(exc)`` so the front-end never has to
parse free-form messages to decide how to render the error UI.

The classifier is INTENTIONALLY narrow:

  * It distinguishes only between the cases the front-end has
    distinct UI affordances for (queue permission vs brief failure
    vs store failure vs generic unavailable vs unknown).
  * It does NOT attempt to round-trip the underlying Azure SDK error
    structure — the dict ``{"type", "message"}`` already on the job
    document carries the technical detail for "Show details".
  * Adding a new kind requires (1) extending ``ErrorKind`` here,
    (2) extending the dispatch chain in ``classify``, and (3) adding
    paired UI copy in the front-end's ``error-kind-copy`` module
    (issue 004). The enum and the UI copy table are paired by design.

The dispatch order matters: more-specific exceptions (or
exceptions-carrying-specific-codes) are checked BEFORE their more-
generic supertypes. See ``test_job_errors.py``'s
``test_dispatch_order_*`` regressions.
"""
from __future__ import annotations

from enum import Enum
from typing import Tuple

from azure.core.exceptions import (
    ClientAuthenticationError,
    HttpResponseError,
    ServiceRequestError,
)
from azure.cosmos import exceptions as cosmos_exceptions


class ErrorKind(str, Enum):
    """Stable enum of producer/worker terminal-failure categories.

    Inherits from ``str`` so the value JSON-serializes to a stable
    literal — Cosmos containers store ``error_kind`` as a string and
    the front-end matches on the string value, not the enum class.
    """

    QUEUE_PERMISSION = "QUEUE_PERMISSION"
    BRIEF_FAILED = "BRIEF_FAILED"
    STORE_FAILED = "STORE_FAILED"
    UNAVAILABLE = "UNAVAILABLE"
    UNKNOWN = "UNKNOWN"


class BriefCompositionFailed(Exception):
    """Producer-side wrapper raised when ``compose_brief`` fails.

    The producer does ``raise BriefCompositionFailed(str(exc)) from exc``
    so the chain is preserved via ``__cause__`` (visible in logs and
    used by ``classify`` to surface the underlying class name).
    """


# ---------------------------------------------------------------------------
# User-message copy
# ---------------------------------------------------------------------------
#
# The strings below are the BACKEND-FACING copy. They appear in:
#   * the JSON error body returned by ``POST /jobs/generate`` (502/503/500);
#   * the ``error.message`` field on the Cosmos job document;
#   * the ``[error]`` log line emitted by the worker terminal-failure path.
#
# The FRONT-END renders its OWN copy keyed off ``error_kind`` (issue 004's
# ``error-kind-copy`` module). The two surfaces stay aligned by the enum;
# they don't have to be word-for-word identical.

_QUEUE_PERMISSION_RBAC_MSG = (
    "The API's identity is missing the 'Storage Queue Data Message Sender' "
    "role on the project-generation queue. Grant the role on the storage "
    "account and retry."
)

_QUEUE_PERMISSION_AUTH_MSG = (
    "The API couldn't authenticate to Azure. Check that the managed identity "
    "is assigned and the IMDS endpoint is reachable, then retry."
)

_STORE_FAILED_MSG = (
    "Couldn't save the job document. Cosmos may be transiently unavailable; "
    "retry shortly."
)

_UNAVAILABLE_MSG = (
    "An upstream Azure dependency is temporarily unavailable. Try again."
)

_UNKNOWN_MSG = "Couldn't start generation. Try again."


def _brief_failed_message(exc: BriefCompositionFailed) -> str:
    """Build the BRIEF_FAILED user message, naming the inner cause."""
    cause = exc.__cause__
    if cause is None:
        return f"Brief composition failed: {exc}"
    return f"Brief composition failed ({type(cause).__name__}): {cause}"


# ---------------------------------------------------------------------------
# Dispatch helpers
# ---------------------------------------------------------------------------


def _http_error_code(exc: BaseException) -> str | None:
    """Return ``exc.error.code`` if present, else None.

    Azure SDK exceptions carry an ``error`` attribute whose ``code``
    is the canonical Azure error code string. The attribute can be
    None on hand-constructed instances (and on some legacy SDK
    versions); guard accordingly.
    """
    error = getattr(exc, "error", None)
    if error is None:
        return None
    return getattr(error, "code", None)


def _is_authorization_permission_mismatch(exc: HttpResponseError) -> bool:
    """True iff ``exc.error.code == 'AuthorizationPermissionMismatch'``.

    This Azure code surfaces specifically on RBAC denials (Storage
    Queue 403, Cosmos 403 with a missing data-plane role, etc.). The
    producer cares about it because it's the one HTTP-403 case where
    the fix is "grant a role" — distinct from token-acquisition
    failures (``ClientAuthenticationError``).
    """
    return _http_error_code(exc) == "AuthorizationPermissionMismatch"


# ---------------------------------------------------------------------------
# classify
# ---------------------------------------------------------------------------


def classify(exc: BaseException) -> Tuple[ErrorKind, int, str]:
    """Map ``exc`` to ``(error_kind, http_status, user_message)``.

    Args:
        exc: The exception raised by the producer or worker.

    Returns:
        Tuple of:
          * ``ErrorKind`` — stable enum value;
          * HTTP status (int) — for the producer's response body;
          * user message (str) — backend-facing copy.

    Never raises. On unrecognized exceptions falls through to
    ``ErrorKind.UNKNOWN`` so the producer's exception handler stays
    simple (no try/except inside the classifier itself).
    """
    if isinstance(exc, BriefCompositionFailed):
        return ErrorKind.BRIEF_FAILED, 502, _brief_failed_message(exc)

    if isinstance(exc, HttpResponseError) and _is_authorization_permission_mismatch(exc):
        return ErrorKind.QUEUE_PERMISSION, 502, _QUEUE_PERMISSION_RBAC_MSG

    if isinstance(exc, ClientAuthenticationError):
        return ErrorKind.QUEUE_PERMISSION, 502, _QUEUE_PERMISSION_AUTH_MSG

    if isinstance(exc, cosmos_exceptions.CosmosHttpResponseError):
        return ErrorKind.STORE_FAILED, 502, _STORE_FAILED_MSG

    if isinstance(exc, (HttpResponseError, ServiceRequestError)):
        return ErrorKind.UNAVAILABLE, 503, _UNAVAILABLE_MSG

    return ErrorKind.UNKNOWN, 500, _UNKNOWN_MSG


__all__ = [
    "BriefCompositionFailed",
    "ErrorKind",
    "classify",
]
