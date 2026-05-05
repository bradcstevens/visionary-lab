"""Tests for ``backend.core.job_errors`` — the producer/worker error
classifier.

Issue 002 of the active-and-queued-jobs-ux-redesign PRD. The classifier
maps recognized exceptions to a stable ``ErrorKind`` enum + an HTTP
status + a user-facing message. Both the producer (issue 002 itself)
and the worker terminal-failure path consume this single function.

Tests are table-driven over ``(exception, expected_kind, expected_status)``
plus dedicated tests for the user_message variants of QUEUE_PERMISSION
(RBAC-mismatch hint vs token-acquisition hint per rubber-duck B2) and
the BriefCompositionFailed wrapper that preserves the underlying
exception via ``__cause__``.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from azure.core.exceptions import (
    ClientAuthenticationError,
    HttpResponseError,
    ServiceRequestError,
)
from azure.cosmos import exceptions as cosmos_exceptions

from backend.core.job_errors import (
    BriefCompositionFailed,
    ErrorKind,
    classify,
)


# ---------------------------------------------------------------------------
# ErrorKind enum surface
# ---------------------------------------------------------------------------


def test_error_kind_has_exactly_five_values():
    """PRD AC: enum has exactly five values. New categories should be
    added deliberately (with paired UI copy in error-kind-copy module
    that issue 004 lands), not silently."""
    assert {k.value for k in ErrorKind} == {
        "QUEUE_PERMISSION",
        "BRIEF_FAILED",
        "STORE_FAILED",
        "UNAVAILABLE",
        "UNKNOWN",
    }


def test_error_kind_serializes_to_string():
    """Cosmos JSON serialization of ``enum.Enum`` is implementation-
    dependent — the value MUST be a string at the boundary so
    ``store.update_job(error_kind=...)`` writes a stable JSON literal.
    Per rubber-duck N2."""
    for kind in ErrorKind:
        assert isinstance(kind.value, str)
        # Sanity: ``str(ErrorKind.X)`` should not produce ``"ErrorKind.X"``
        # under Python's str(IntEnum) / str(StrEnum)-style coercion.
        assert kind.value == kind  # str-Enum equality
        assert str(kind.value) == kind.value


# ---------------------------------------------------------------------------
# BriefCompositionFailed → BRIEF_FAILED
# ---------------------------------------------------------------------------


def test_brief_composition_failed_routes_to_brief_failed():
    exc = BriefCompositionFailed("something broke")
    kind, status, msg = classify(exc)
    assert kind == ErrorKind.BRIEF_FAILED
    assert status == 502
    assert "brief" in msg.lower()


def test_brief_composition_failed_preserves_cause_in_message():
    """Per rubber-duck N6: ``raise BriefCompositionFailed(...) from exc``
    chains the cause via ``__cause__``. The classifier should surface
    the inner exception's class name in the message so dev triage can
    grep for it without re-reading the chain."""
    inner = ValueError("rate limited")
    wrapper = BriefCompositionFailed("brief failed")
    wrapper.__cause__ = inner
    kind, status, msg = classify(wrapper)
    assert kind == ErrorKind.BRIEF_FAILED
    # The user message names the underlying exception class so
    # the front-end "Show technical details" section (issue 004)
    # has something specific to show.
    assert "ValueError" in msg or "rate limited" in msg


# ---------------------------------------------------------------------------
# AuthorizationPermissionMismatch → QUEUE_PERMISSION (RBAC hint)
# ---------------------------------------------------------------------------


def _http_response_error_with_code(code: str, message: str = "forbidden"):
    """Build an ``HttpResponseError`` with the given Azure error code.

    The Azure SDK's ``error`` attribute is normally an
    ``ODataV4Format``; we substitute a ``MagicMock`` so the test is
    decoupled from the SDK's internal serialization.
    """
    exc = HttpResponseError(message=message)
    exc.error = MagicMock()
    exc.error.code = code
    return exc


def test_authorization_permission_mismatch_routes_to_queue_permission_with_rbac_hint():
    """The bug-report case: Azure Storage Queue 403 with code
    AuthorizationPermissionMismatch. Maps to QUEUE_PERMISSION with a
    developer-targeted message that NAMES the missing role
    (``Storage Queue Data Message Sender``) so an oncall can act on
    the right thing."""
    exc = _http_response_error_with_code("AuthorizationPermissionMismatch")
    kind, status, msg = classify(exc)
    assert kind == ErrorKind.QUEUE_PERMISSION
    assert status == 502
    assert "Storage Queue Data Message Sender" in msg


# ---------------------------------------------------------------------------
# ClientAuthenticationError → QUEUE_PERMISSION (auth hint, NOT role hint)
# ---------------------------------------------------------------------------


def test_client_authentication_error_routes_to_queue_permission_with_auth_hint():
    """Per rubber-duck B2: ``ClientAuthenticationError`` is token-
    acquisition failure (managed identity not assigned, IMDS
    unreachable, AAD outage). NO role assignment can fix it. Same
    ErrorKind (per PRD AC) but branch the user_message so oncall
    isn't misled toward an RBAC change that wouldn't help."""
    exc = ClientAuthenticationError(message="token acquisition failed")
    kind, status, msg = classify(exc)
    assert kind == ErrorKind.QUEUE_PERMISSION
    assert status == 502
    # Message points at managed-identity / token, NOT a role grant.
    assert "managed identity" in msg.lower() or "authenticate" in msg.lower()
    assert "Storage Queue Data Message Sender" not in msg


def test_client_authentication_with_rbac_code_prefers_rbac_hint():
    """Edge case: ``ClientAuthenticationError`` IS an
    ``HttpResponseError`` subclass, so a hypothetical
    ClientAuthenticationError carrying ``code=AuthorizationPermission
    Mismatch`` should route to the RBAC hint (the more specific
    classification). This pins the dispatch order."""
    exc = ClientAuthenticationError(message="forbidden")
    exc.error = MagicMock()
    exc.error.code = "AuthorizationPermissionMismatch"
    kind, status, msg = classify(exc)
    assert kind == ErrorKind.QUEUE_PERMISSION
    assert "Storage Queue Data Message Sender" in msg


# ---------------------------------------------------------------------------
# Cosmos write errors → STORE_FAILED
# ---------------------------------------------------------------------------


def test_cosmos_http_response_error_routes_to_store_failed():
    """Cosmos write errors (NOT 409 — ``create_job`` swallows that as
    the idempotent-retry path). ``CosmosResourceNotFoundError`` for
    ``update_job`` on a missing doc, generic 500s on the container,
    etc. all route to STORE_FAILED so the front-end can show "couldn't
    save your job; try again"."""
    exc = cosmos_exceptions.CosmosHttpResponseError(
        status_code=500, message="cosmos blew up"
    )
    kind, status, msg = classify(exc)
    assert kind == ErrorKind.STORE_FAILED
    assert status == 502
    assert "job" in msg.lower() or "save" in msg.lower() or "persist" in msg.lower()


def test_cosmos_resource_not_found_error_routes_to_store_failed():
    exc = cosmos_exceptions.CosmosResourceNotFoundError(message="missing")
    kind, _, _ = classify(exc)
    assert kind == ErrorKind.STORE_FAILED


# ---------------------------------------------------------------------------
# Generic HttpResponseError → UNAVAILABLE
# ---------------------------------------------------------------------------


def test_generic_http_response_error_routes_to_unavailable():
    """An Azure error with no known code (e.g. transient 503 from a
    dependency) maps to UNAVAILABLE so the front-end can offer a
    'try again' affordance without claiming the user did something
    wrong."""
    exc = _http_response_error_with_code("ServiceUnavailable", message="transient")
    kind, status, msg = classify(exc)
    assert kind == ErrorKind.UNAVAILABLE
    assert status == 503


def test_service_request_error_routes_to_unavailable():
    """Network-level failure (DNS, TLS handshake, etc.) — Azure SDK
    surfaces these as ``ServiceRequestError``. UNAVAILABLE is the
    right kind: nothing the user can fix, retry-with-backoff is the
    correct UX."""
    exc = ServiceRequestError(message="connection refused")
    kind, status, _ = classify(exc)
    assert kind == ErrorKind.UNAVAILABLE
    assert status == 503


# ---------------------------------------------------------------------------
# Unknown exception → UNKNOWN
# ---------------------------------------------------------------------------


def test_arbitrary_exception_routes_to_unknown():
    exc = RuntimeError("something unexpected")
    kind, status, msg = classify(exc)
    assert kind == ErrorKind.UNKNOWN
    assert status == 500
    # Default copy is the PRD's "Couldn't start generation, try again."
    # The unknown-kind fallback must NOT name a specific cause.
    assert "Storage Queue" not in msg
    assert "brief" not in msg.lower()


def test_value_error_routes_to_unknown():
    """Plain ValueError — programming error, no specific recovery."""
    kind, _, _ = classify(ValueError("oops"))
    assert kind == ErrorKind.UNKNOWN


def test_key_error_routes_to_unknown():
    kind, _, _ = classify(KeyError("missing_field"))
    assert kind == ErrorKind.UNKNOWN


# ---------------------------------------------------------------------------
# Table-driven sweep — pin the (kind, status) tuple for every recognized case
# ---------------------------------------------------------------------------


def _table_cases():
    return [
        # (build callable, expected_kind, expected_status)
        (
            lambda: BriefCompositionFailed("x"),
            ErrorKind.BRIEF_FAILED, 502,
        ),
        (
            lambda: _http_response_error_with_code("AuthorizationPermissionMismatch"),
            ErrorKind.QUEUE_PERMISSION, 502,
        ),
        (
            lambda: ClientAuthenticationError(message="token failed"),
            ErrorKind.QUEUE_PERMISSION, 502,
        ),
        (
            lambda: cosmos_exceptions.CosmosHttpResponseError(
                status_code=500, message="cosmos"
            ),
            ErrorKind.STORE_FAILED, 502,
        ),
        (
            lambda: _http_response_error_with_code("ServiceUnavailable"),
            ErrorKind.UNAVAILABLE, 503,
        ),
        (
            lambda: ServiceRequestError(message="net"),
            ErrorKind.UNAVAILABLE, 503,
        ),
        (
            lambda: RuntimeError("wat"),
            ErrorKind.UNKNOWN, 500,
        ),
    ]


@pytest.mark.parametrize(
    "exc_factory, expected_kind, expected_status",
    _table_cases(),
)
def test_classify_table_driven(exc_factory, expected_kind, expected_status):
    """Pin the (kind, status) tuple for every recognized exception
    category. A future contributor adding a new category needs to
    extend the table here (and the corresponding UI copy in issue
    004's ``error-kind-copy`` module)."""
    exc = exc_factory()
    kind, status, msg = classify(exc)
    assert kind == expected_kind, f"kind mismatch for {type(exc).__name__}"
    assert status == expected_status, f"status mismatch for {type(exc).__name__}"
    # Every recognized case has a non-empty user message.
    assert isinstance(msg, str) and msg.strip()


# ---------------------------------------------------------------------------
# Dispatch order regression
# ---------------------------------------------------------------------------


def test_dispatch_order_authorization_mismatch_beats_generic_http_response_error():
    """``AuthorizationPermissionMismatch`` is more specific than the
    generic ``HttpResponseError`` arm. A future refactor that reorders
    the dispatch chain must not silently route RBAC errors to
    UNAVAILABLE."""
    exc = _http_response_error_with_code("AuthorizationPermissionMismatch")
    kind, _, _ = classify(exc)
    assert kind == ErrorKind.QUEUE_PERMISSION


def test_dispatch_order_cosmos_beats_generic_http_response_error():
    """``CosmosHttpResponseError`` is a subclass of ``HttpResponseError``.
    The classifier must check for it explicitly so a Cosmos 500 doesn't
    silently route to the generic UNAVAILABLE arm."""
    exc = cosmos_exceptions.CosmosHttpResponseError(
        status_code=500, message="cosmos"
    )
    kind, _, _ = classify(exc)
    assert kind == ErrorKind.STORE_FAILED


def test_classify_does_not_raise_on_exotic_inputs():
    """The classifier is on the hot path of the producer's exception
    handler — it must NEVER itself raise, even on degenerate input."""
    # HttpResponseError without an .error attribute (constructible).
    exc = HttpResponseError(message="bare")
    kind, _, _ = classify(exc)
    assert isinstance(kind, ErrorKind)
