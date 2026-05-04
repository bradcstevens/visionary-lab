"""``backend.runtime`` — pure helpers for runtime role selection.

The same Docker image runs both Container Apps in production: the
API container (``ROLE=api``, default) and the worker container
(``ROLE=worker``). Each container's bicep entrypoint spawns the
correct process directly (``uvicorn backend.main:app`` for the API,
``python -m backend.worker_main`` for the worker), so this module
exists for any code path that needs to *report* which role it's in
without coupling to ``os.environ`` directly.

Pure: no import-time side effects, no IO. Trivially testable; trivially
mocked (callers pass any ``Mapping[str, str]``).
"""
from __future__ import annotations

from typing import Literal, Mapping


def choose_runtime(env: Mapping[str, str]) -> Literal["api", "worker"]:
    """Pick the runtime role from a process environment mapping.

    Returns ``"worker"`` only when ``env["ROLE"] == "worker"`` exactly;
    every other value (unset, empty, "api", typo of "worker") returns
    ``"api"``. The strict-equality contract is deliberate — silent
    typo handling here would let a misconfigured worker container
    boot the API server and hide the misconfiguration.
    """
    return "worker" if env.get("ROLE") == "worker" else "api"
