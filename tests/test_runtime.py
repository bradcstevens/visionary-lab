"""Tests for ``backend.runtime.choose_runtime``."""
from __future__ import annotations

from backend.runtime import choose_runtime


def test_role_worker_returns_worker() -> None:
    assert choose_runtime({"ROLE": "worker"}) == "worker"


def test_role_unset_returns_api() -> None:
    assert choose_runtime({}) == "api"


def test_role_empty_string_returns_api() -> None:
    # Pragma: an unset Container App env var sometimes surfaces as ""
    # rather than missing. Treat both as "api" so a misconfigured
    # worker container can't accidentally start the API server (it
    # also can't accidentally start the worker — both default to api).
    assert choose_runtime({"ROLE": ""}) == "api"


def test_role_unrecognised_value_returns_api() -> None:
    # Anything that isn't literally "worker" is api. Forces the worker
    # role to be set explicitly and avoids typo-driven role drift
    # (e.g. ROLE="workers" silently picking the worker branch).
    assert choose_runtime({"ROLE": "wOrKeR"}) == "api"
    assert choose_runtime({"ROLE": "api"}) == "api"
    assert choose_runtime({"ROLE": "background"}) == "api"
