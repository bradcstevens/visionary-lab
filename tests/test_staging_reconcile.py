"""Tests for staging project reconciliation logic."""
from unittest.mock import MagicMock

import pytest
from datetime import datetime, timezone, timedelta

from backend.core.staging_reconcile import (
    _derive_status_from_rooms,
    _is_stale,
    compute_project_status_from_jobs,
    reconcile_project,
)


class TestIsStaleness:
    def test_none_updated_at_is_stale(self):
        assert _is_stale(None) is True

    def test_empty_string_is_stale(self):
        assert _is_stale("") is True

    def test_recent_timestamp_is_not_stale(self):
        recent = datetime.now(timezone.utc).isoformat()
        assert _is_stale(recent) is False

    def test_old_timestamp_is_stale(self):
        old = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
        assert _is_stale(old) is True

    def test_custom_threshold(self):
        ts = (datetime.now(timezone.utc) - timedelta(seconds=30)).isoformat()
        assert _is_stale(ts, threshold_seconds=60) is False
        assert _is_stale(ts, threshold_seconds=10) is True


def _make_project(status="processing", rooms=None, updated_at=None):
    """Helper to build a minimal project dict."""
    if updated_at is None:
        # Default to stale so reconciliation triggers
        updated_at = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
    return {
        "id": "proj-1",
        "status": status,
        "updated_at": updated_at,
        "rooms": rooms or [],
    }


def _make_room(room_id="room-1", status="processing", variations=None):
    return {
        "id": room_id,
        "label": f"Room {room_id}",
        "status": status,
        "error": None,
        "variations": variations or [],
    }


def _make_variation(var_id="var-1", status="processing", image_url=None, error=None):
    return {
        "id": var_id,
        "status": status,
        "image_url": image_url,
        "error": error,
        "thumbnail_url": None,
        "generation_metadata": None,
    }


class TestReconcileProject:
    def test_non_processing_project_unchanged(self):
        proj = _make_project(status="completed")
        assert reconcile_project(proj) is False
        assert proj["status"] == "completed"

    def test_pending_project_unchanged(self):
        proj = _make_project(status="pending")
        assert reconcile_project(proj) is False

    def test_recent_processing_not_reconciled(self):
        """If updated_at is recent, don't reconcile (generation may be active)."""
        proj = _make_project(
            status="processing",
            updated_at=datetime.now(timezone.utc).isoformat(),
            rooms=[_make_room(status="processing")],
        )
        assert reconcile_project(proj) is False
        assert proj["status"] == "processing"

    def test_stale_processing_project_with_no_rooms_does_not_mutate_status(self):
        # Issue 003: reconcile_project no longer mutates project status.
        # An empty-rooms project has no variations to clean up; reconcile is
        # a no-op. Status derivation now lives in compute_project_status_from_jobs.
        proj = _make_project(status="processing", rooms=[])
        assert reconcile_project(proj) is False
        assert proj["status"] == "processing"

    def test_stale_processing_room_resets_to_pending(self):
        variations = [_make_variation("v1", "processing"), _make_variation("v2", "pending")]
        room = _make_room("r1", "processing", variations)
        proj = _make_project(rooms=[room])

        assert reconcile_project(proj) is True
        assert room["status"] == "pending"
        assert all(v["status"] == "pending" for v in variations)
        # Issue 003: status no longer mutated by reconcile_project.
        assert proj["status"] == "processing"

    def test_completed_room_no_variation_cleanup_returns_false(self):
        # Issue 003: when there's no variation cleanup work, reconcile_project
        # returns False (under the old code it returned True from the status
        # mutation; that path is gone).
        variations = [
            _make_variation("v1", "completed", image_url="https://example.com/img.png"),
            _make_variation("v2", "completed", image_url="https://example.com/img2.png"),
        ]
        room = _make_room("r1", "completed", variations)
        proj = _make_project(rooms=[room])

        assert reconcile_project(proj) is False
        assert room["status"] == "completed"
        # Status preserved; the project status is computed elsewhere now.
        assert proj["status"] == "processing"

    def test_mix_completed_and_processing_rooms_resets_processing_only(self):
        completed_room = _make_room("r1", "completed", [
            _make_variation("v1", "completed", image_url="https://example.com/img.png"),
        ])
        processing_room = _make_room("r2", "processing", [
            _make_variation("v2", "processing"),
        ])
        proj = _make_project(rooms=[completed_room, processing_room])

        assert reconcile_project(proj) is True
        assert completed_room["status"] == "completed"
        assert processing_room["status"] == "pending"
        # Issue 003: project status NOT mutated by reconcile_project.
        assert proj["status"] == "processing"

    def test_orphan_data_cleared_on_reset(self):
        v = _make_variation("v1", "processing", image_url="https://stale.com/img.png", error="old error")
        v["generation_metadata"] = {"model": "gpt-image-2"}
        room = _make_room("r1", "processing", [v])
        proj = _make_project(rooms=[room])

        reconcile_project(proj)

        assert v["status"] == "pending"
        assert v["image_url"] is None
        assert v["error"] is None
        assert v["generation_metadata"] is None

    def test_force_ignores_staleness_check(self):
        """Force reconciliation even if updated_at is recent."""
        proj = _make_project(
            status="processing",
            updated_at=datetime.now(timezone.utc).isoformat(),
            rooms=[_make_room(status="processing", variations=[_make_variation()])],
        )
        assert reconcile_project(proj) is False  # Not forced, recent = no change
        assert reconcile_project(proj, force=True) is True  # Forced — variations reset
        # Issue 003: status not mutated by reconcile_project (the /reset
        # endpoint now applies status separately).
        assert proj["status"] == "processing"

    def test_all_failed_rooms_no_variation_cleanup_returns_false(self):
        # Issue 003: failed rooms have no processing variations to clean up,
        # so reconcile_project returns False. Project status is NEVER flipped
        # to 'failed' by reconcile (AC#6: only the worker / cancellation
        # cascade / producer hard error can produce 'failed').
        room = _make_room("r1", "failed", [
            _make_variation("v1", "failed", error="boom"),
        ])
        proj = _make_project(rooms=[room])

        assert reconcile_project(proj) is False
        assert proj["status"] == "processing"

    def test_processing_variation_in_completed_room_gets_reset(self):
        """Edge case: completed room with a stuck processing variation."""
        variations = [
            _make_variation("v1", "completed", image_url="https://example.com/img.png"),
            _make_variation("v2", "processing"),
        ]
        room = _make_room("r1", "completed", variations)
        proj = _make_project(rooms=[room])

        assert reconcile_project(proj) is True
        assert room["status"] == "completed"
        assert variations[0]["status"] == "completed"
        assert variations[1]["status"] == "pending"


class TestReconcileProjectNegativeProperty:
    """Negative property: ``reconcile_project`` MUST NOT mutate
    ``project_data["status"]`` on any code path. Status derivation lives
    in ``compute_project_status_from_jobs``.
    """

    @pytest.mark.parametrize(
        "scenario",
        [
            # (initial_status, rooms, force)
            ("processing", [], False),
            ("processing", [], True),
            ("processing", [{"id": "r1", "label": "r1", "status": "pending", "variations": []}], False),
            ("processing", [{"id": "r1", "label": "r1", "status": "pending", "variations": []}], True),
            ("processing", [{"id": "r1", "label": "r1", "status": "processing", "variations": []}], True),
            ("processing", [{"id": "r1", "label": "r1", "status": "completed", "variations": []}], True),
            ("processing", [{"id": "r1", "label": "r1", "status": "failed", "variations": []}], True),
            ("processing", [
                {"id": "r1", "label": "r1", "status": "completed", "variations": []},
                {"id": "r2", "label": "r2", "status": "failed", "variations": []},
            ], True),
            ("completed", [{"id": "r1", "label": "r1", "status": "completed", "variations": []}], True),
            ("pending", [{"id": "r1", "label": "r1", "status": "pending", "variations": []}], True),
            ("failed", [{"id": "r1", "label": "r1", "status": "failed", "variations": []}], True),
        ],
    )
    def test_reconcile_never_mutates_status(self, scenario):
        initial_status, rooms, force = scenario
        proj = _make_project(status=initial_status, rooms=rooms)
        original_status = proj["status"]
        reconcile_project(proj, force=force)
        assert proj["status"] == original_status, (
            f"reconcile_project mutated status from {original_status!r} → "
            f"{proj['status']!r} (scenario={scenario})"
        )


# ---------------------------------------------------------------------------
# Issue 003 (active-and-queued-jobs-ux-redesign PRD): split status logic
# ---------------------------------------------------------------------------


class TestDeriveStatusFromRooms:
    """Pure function: derive project status from room statuses.

    The buggy "mixed room statuses ⇒ failed" branch is removed; mixed states
    fall through to ``pending``. The reconcile path NEVER produces ``failed``
    (that decision belongs to the worker / cancellation cascade / producer
    hard error).
    """

    def test_empty_rooms_yields_pending(self):
        assert _derive_status_from_rooms([]) == "pending"

    def test_all_completed_yields_completed(self):
        rooms = [{"status": "completed"}, {"status": "completed"}]
        assert _derive_status_from_rooms(rooms) == "completed"

    def test_mix_with_completed_yields_completed(self):
        # Preserves "user can regenerate individual rooms that were reset".
        rooms = [{"status": "completed"}, {"status": "pending"}]
        assert _derive_status_from_rooms(rooms) == "completed"

    def test_all_pending_yields_pending(self):
        rooms = [{"status": "pending"}, {"status": "pending"}]
        assert _derive_status_from_rooms(rooms) == "pending"

    def test_all_failed_yields_pending(self):
        # AC#6: reconcile path NEVER produces 'failed'.
        rooms = [{"status": "failed"}, {"status": "failed"}]
        assert _derive_status_from_rooms(rooms) == "pending"

    def test_mixed_pending_and_failed_yields_pending(self):
        # The bug: this used to yield 'failed'.
        rooms = [{"status": "pending"}, {"status": "failed"}]
        assert _derive_status_from_rooms(rooms) == "pending"

    def test_mixed_processing_and_pending_yields_pending(self):
        rooms = [{"status": "processing"}, {"status": "pending"}]
        assert _derive_status_from_rooms(rooms) == "pending"


class TestComputeProjectStatusFromJobs:
    """Issue 003: derive project status by reading the active job from the
    jobs container, falling back to room-derived only when the job is
    terminal or missing.
    """

    def _store(self, *, get_job_return=None, raise_if_called=False):
        """Build a fake JobStore. ``raise_if_called`` asserts the
        short-circuit path doesn't touch the store."""
        store = MagicMock(name="JobStore")
        if raise_if_called:
            store.get_job.side_effect = AssertionError(
                "store.get_job() must not be called on this path"
            )
        else:
            store.get_job.return_value = get_job_return
        return store

    def test_short_circuits_when_status_not_processing(self):
        # Status mutation only flows from the 'processing' state. Other
        # statuses are terminal/at-rest and must not be revisited.
        store = self._store(raise_if_called=True)
        proj = _make_project(status="completed")
        proj["current_project_job_id"] = "proj-1:project:project:rev1"
        assert compute_project_status_from_jobs(proj, store) is None
        assert store.get_job.call_count == 0

    def test_short_circuits_when_pending_status(self):
        store = self._store(raise_if_called=True)
        proj = _make_project(status="pending")
        proj["current_project_job_id"] = "proj-1:project:project:rev1"
        assert compute_project_status_from_jobs(proj, store) is None

    def test_short_circuits_when_current_project_job_id_missing(self):
        # AC: missing current_project_job_id is a short-circuit (legacy
        # projects intentionally left alone here; flipping them to pending
        # based on rooms could erase the user's last-known state).
        store = self._store(raise_if_called=True)
        proj = _make_project(status="processing")
        # No current_project_job_id at all.
        assert compute_project_status_from_jobs(proj, store) is None
        assert store.get_job.call_count == 0

    def test_short_circuits_when_current_project_job_id_empty_string(self):
        # Defensive: a falsy value (None/empty string) also short-circuits.
        store = self._store(raise_if_called=True)
        proj = _make_project(status="processing")
        proj["current_project_job_id"] = ""
        assert compute_project_status_from_jobs(proj, store) is None

    def test_active_non_terminal_job_returns_none(self):
        # Worker is still processing → status stays 'processing'.
        proj = _make_project(status="processing", rooms=[
            _make_room("r1", "completed", [_make_variation("v1", "completed")]),
        ])
        proj["current_project_job_id"] = "proj-1:project:project:rev1"
        store = self._store(get_job_return={
            "id": "proj-1:project:project:rev1",
            "project_id": "proj-1",
            "status": "running",
        })
        assert compute_project_status_from_jobs(proj, store) is None
        store.get_job.assert_called_once_with(
            "proj-1:project:project:rev1", "proj-1"
        )

    def test_pending_job_returns_none(self):
        # 'pending' is also non-terminal — worker hasn't picked it up yet.
        proj = _make_project(status="processing")
        proj["current_project_job_id"] = "proj-1:project:project:rev1"
        store = self._store(get_job_return={
            "id": "proj-1:project:project:rev1",
            "project_id": "proj-1",
            "status": "pending",
        })
        assert compute_project_status_from_jobs(proj, store) is None

    def test_terminal_succeeded_job_derives_from_rooms(self):
        proj = _make_project(status="processing", rooms=[
            _make_room("r1", "completed", [_make_variation("v1", "completed")]),
        ])
        proj["current_project_job_id"] = "proj-1:project:project:rev1"
        store = self._store(get_job_return={
            "id": "proj-1:project:project:rev1",
            "project_id": "proj-1",
            "status": "succeeded",
        })
        assert compute_project_status_from_jobs(proj, store) == "completed"

    def test_terminal_failed_job_derives_from_rooms(self):
        proj = _make_project(status="processing", rooms=[
            _make_room("r1", "failed", []),
        ])
        proj["current_project_job_id"] = "proj-1:project:project:rev1"
        store = self._store(get_job_return={
            "id": "proj-1:project:project:rev1",
            "project_id": "proj-1",
            "status": "failed",
        })
        # Reconcile path NEVER produces 'failed'; mixed/all-failed → pending.
        assert compute_project_status_from_jobs(proj, store) == "pending"

    def test_terminal_cancelled_job_derives_from_rooms(self):
        proj = _make_project(status="processing", rooms=[
            _make_room("r1", "completed", [_make_variation("v1", "completed")]),
            _make_room("r2", "pending", []),
        ])
        proj["current_project_job_id"] = "proj-1:project:project:rev1"
        store = self._store(get_job_return={
            "id": "proj-1:project:project:rev1",
            "project_id": "proj-1",
            "status": "cancelled",
        })
        assert compute_project_status_from_jobs(proj, store) == "completed"

    def test_missing_job_document_derives_from_rooms(self):
        # Job document not found in store (e.g. queue purged) → fall back
        # to room-derived status so the project doesn't stay stuck.
        proj = _make_project(status="processing", rooms=[
            _make_room("r1", "completed", [_make_variation("v1", "completed")]),
        ])
        proj["current_project_job_id"] = "proj-1:project:project:rev1"
        store = self._store(get_job_return=None)
        assert compute_project_status_from_jobs(proj, store) == "completed"

    def test_mixed_room_states_yield_pending_not_failed(self):
        # Headline regression: this is the bug the issue closes.
        proj = _make_project(status="processing", rooms=[
            _make_room("r1", "pending", []),
            _make_room("r2", "failed", []),
        ])
        proj["current_project_job_id"] = "proj-1:project:project:rev1"
        store = self._store(get_job_return={
            "id": "proj-1:project:project:rev1",
            "project_id": "proj-1",
            "status": "failed",
        })
        assert compute_project_status_from_jobs(proj, store) == "pending"

    def test_does_not_call_store_when_not_processing(self):
        # Performance: list_projects iterates N projects and we don't want
        # one Cosmos read per non-processing project.
        store = self._store(raise_if_called=True)
        for status in ("uploading", "pending", "completed", "failed"):
            proj = _make_project(status=status)
            proj["current_project_job_id"] = "proj-1:project:project:rev1"
            assert compute_project_status_from_jobs(proj, store) is None

    def test_does_not_call_store_when_no_job_id(self):
        # Same protection on the missing-job-id short-circuit.
        store = self._store(raise_if_called=True)
        proj = _make_project(status="processing")
        # No current_project_job_id key at all.
        assert compute_project_status_from_jobs(proj, store) is None
