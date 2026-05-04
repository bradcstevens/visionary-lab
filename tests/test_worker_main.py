"""Tests for ``backend.worker_main`` — the production worker entry point.

Construction wiring tests for ``build_worker`` were migrated to
``tests/test_worker_factory.py`` when the factory was extracted into
``backend.core.worker_factory`` per issue 001 of the
active-and-queued-jobs-ux-redesign PRD. This module now keeps only
the ``main()`` entry-point smoke tests.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

import backend.worker_main as worker_main_mod


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_main_awaits_worker_run_exactly_once(monkeypatch):
    worker = MagicMock(name="JobWorker_instance")
    worker.run = AsyncMock(name="run", return_value=None)
    monkeypatch.setattr(
        worker_main_mod, "build_worker", MagicMock(return_value=worker)
    )

    await worker_main_mod.main()

    worker.run.assert_awaited_once_with()


def test_main_can_be_invoked_via_asyncio_run(monkeypatch):
    """Smoke test for the ``if __name__ == '__main__': asyncio.run(main())``
    contract — the entry point block in worker_main.py."""
    worker = MagicMock(name="JobWorker_instance")
    worker.run = AsyncMock(name="run", return_value=None)
    monkeypatch.setattr(
        worker_main_mod, "build_worker", MagicMock(return_value=worker)
    )

    # If main() can't be run by asyncio.run, this raises.
    asyncio.run(worker_main_mod.main())
    worker.run.assert_awaited_once_with()


def test_build_worker_is_re_exported_from_worker_factory():
    """``backend.worker_main.build_worker`` is the same object as the
    canonical factory in ``backend.core.worker_factory.build_worker``.
    Importing from either path must not produce two different
    construction codepaths."""
    from backend.core.worker_factory import build_worker as canonical
    assert worker_main_mod.build_worker is canonical
