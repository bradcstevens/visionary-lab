"""Tests for ``backend.core.worker_factory`` — shared JobWorker bootstrap.

The factory is mostly construction wiring: ``JobStore``, ``JobQueue``,
``ProgressEstimator``, the staging pipeline / storage, the
``staging_dispatcher``, and a ``JobWorker``. These tests mock every
constructor so the bootstrap can be verified in isolation without
booting Cosmos, Storage Queues, or the Foundry endpoint.

Coverage focuses on the **wiring**: that each constructor is called
exactly once, that ``configure_dispatcher_dependencies`` is invoked
with factories that resolve to the constructed instances, and that the
constructed JobWorker receives the right dispatcher + estimator.

Pipeline-shape tests live in ``test_staging_dispatcher.py``.

Migrated from the old ``tests/test_worker_main.py`` build_worker tests
when the factory was extracted from ``backend.worker_main`` into
``backend.core.worker_factory`` per issue 001 of the
active-and-queued-jobs-ux-redesign PRD.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import backend.core.worker_factory as worker_factory_mod


@pytest.fixture
def patched_constructors(monkeypatch):
    """Patch every heavy constructor used by ``build_worker``.

    Returns a dict of MagicMocks the test can introspect. Each key
    matches the attribute name on ``backend.core.worker_factory``.
    """
    job_store = MagicMock(name="JobStore_instance")
    job_queue = MagicMock(name="JobQueue_instance")
    estimator = MagicMock(name="ProgressEstimator_instance")
    storage = MagicMock(name="StagingStorageService_instance")
    pipeline = MagicMock(name="StagingPipeline_instance")
    worker = MagicMock(name="JobWorker_instance")

    JobStore = MagicMock(name="JobStore_class", return_value=job_store)
    JobQueue = MagicMock(name="JobQueue_class", return_value=job_queue)
    ProgressEstimator = MagicMock(
        name="ProgressEstimator_class", return_value=estimator
    )
    StagingStorageService = MagicMock(
        name="StagingStorageService_class", return_value=storage
    )
    build_pipeline = MagicMock(
        name="build_staging_pipeline", return_value=pipeline
    )
    JobWorker = MagicMock(name="JobWorker_class", return_value=worker)
    configure = MagicMock(name="configure_dispatcher_dependencies")

    monkeypatch.setattr(worker_factory_mod, "JobStore", JobStore)
    monkeypatch.setattr(worker_factory_mod, "JobQueue", JobQueue)
    monkeypatch.setattr(worker_factory_mod, "ProgressEstimator", ProgressEstimator)
    monkeypatch.setattr(
        worker_factory_mod, "StagingStorageService", StagingStorageService
    )
    monkeypatch.setattr(
        worker_factory_mod, "build_staging_pipeline", build_pipeline
    )
    monkeypatch.setattr(worker_factory_mod, "JobWorker", JobWorker)
    monkeypatch.setattr(
        worker_factory_mod, "configure_dispatcher_dependencies", configure
    )

    return {
        "job_store": job_store,
        "job_queue": job_queue,
        "estimator": estimator,
        "storage": storage,
        "pipeline": pipeline,
        "worker": worker,
        "JobStore": JobStore,
        "JobQueue": JobQueue,
        "ProgressEstimator": ProgressEstimator,
        "StagingStorageService": StagingStorageService,
        "build_pipeline": build_pipeline,
        "JobWorker": JobWorker,
        "configure": configure,
    }


# ---------------------------------------------------------------------------
# build_worker
# ---------------------------------------------------------------------------


def test_build_worker_returns_jobworker_instance(patched_constructors):
    result = worker_factory_mod.build_worker()
    assert result is patched_constructors["worker"]


def test_build_worker_constructs_each_dependency_exactly_once(patched_constructors):
    worker_factory_mod.build_worker()

    assert patched_constructors["JobStore"].call_count == 1
    assert patched_constructors["JobQueue"].call_count == 1
    assert patched_constructors["ProgressEstimator"].call_count == 1
    assert patched_constructors["StagingStorageService"].call_count == 1
    assert patched_constructors["build_pipeline"].call_count == 1
    assert patched_constructors["JobWorker"].call_count == 1


def test_build_worker_configures_dispatcher_dependencies_with_constructed_instances(
    patched_constructors,
):
    worker_factory_mod.build_worker()

    configure = patched_constructors["configure"]
    assert configure.call_count == 1
    kwargs = configure.call_args.kwargs
    assert "storage_factory" in kwargs
    assert "pipeline_factory" in kwargs
    assert "store_factory" in kwargs

    # Factories must resolve to the SAME instances we constructed,
    # not new ones on every call. Sharing matters: pipeline construction
    # is heavy (LLM client + image clients + blob clients). The store
    # is shared across the worker (state writes) and the dispatcher
    # (progress phase writes) — passing a different instance would
    # split the change-feed view.
    assert kwargs["storage_factory"]() is patched_constructors["storage"]
    assert kwargs["pipeline_factory"]() is patched_constructors["pipeline"]
    assert kwargs["store_factory"]() is patched_constructors["job_store"]
    # Calling the factory twice MUST return the same instance — otherwise
    # we'd be paying construction cost on every dispatch call.
    assert kwargs["storage_factory"]() is kwargs["storage_factory"]()
    assert kwargs["pipeline_factory"]() is kwargs["pipeline_factory"]()
    assert kwargs["store_factory"]() is kwargs["store_factory"]()


def test_build_worker_passes_staging_dispatcher_into_jobworker(patched_constructors):
    from backend.core.staging_dispatcher import staging_dispatcher

    worker_factory_mod.build_worker()

    kwargs = patched_constructors["JobWorker"].call_args.kwargs
    assert kwargs["dispatcher"] is staging_dispatcher
    assert kwargs["queue"] is patched_constructors["job_queue"]
    assert kwargs["store"] is patched_constructors["job_store"]
    assert kwargs["estimator"] is patched_constructors["estimator"]


def test_build_worker_configures_dispatcher_before_jobworker_construction(
    patched_constructors,
):
    """Wiring order matters. ``configure_dispatcher_dependencies`` must
    run before ``JobWorker`` is built so that if anything causes the
    worker to dispatch a message synchronously during construction
    (defensive), the dispatcher is already wired."""
    call_order: list[str] = []
    patched_constructors["configure"].side_effect = lambda **_: call_order.append(
        "configure"
    )
    patched_constructors["JobWorker"].side_effect = lambda **_: call_order.append(
        "JobWorker"
    ) or patched_constructors["worker"]

    worker_factory_mod.build_worker()

    assert call_order == ["configure", "JobWorker"]


# ---------------------------------------------------------------------------
# Backward-compat: worker_main still exposes build_worker
# ---------------------------------------------------------------------------


def test_worker_main_re_exports_build_worker():
    """``backend.worker_main.build_worker`` must keep working so that
    ``python -m backend.worker_main`` continues to function and any
    external caller importing from that path doesn't break."""
    import backend.worker_main as worker_main_mod

    assert worker_main_mod.build_worker is worker_factory_mod.build_worker
