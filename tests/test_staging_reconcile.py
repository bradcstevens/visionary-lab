"""Tests for staging project reconciliation logic."""
import pytest
from datetime import datetime, timezone, timedelta

from backend.core.staging_reconcile import reconcile_project, _is_stale


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

    def test_stale_processing_project_with_no_rooms_resets_to_pending(self):
        proj = _make_project(status="processing", rooms=[])
        assert reconcile_project(proj) is True
        assert proj["status"] == "pending"

    def test_stale_processing_room_resets_to_pending(self):
        variations = [_make_variation("v1", "processing"), _make_variation("v2", "pending")]
        room = _make_room("r1", "processing", variations)
        proj = _make_project(rooms=[room])

        assert reconcile_project(proj) is True
        assert room["status"] == "pending"
        assert all(v["status"] == "pending" for v in variations)

    def test_completed_room_preserved(self):
        variations = [
            _make_variation("v1", "completed", image_url="https://example.com/img.png"),
            _make_variation("v2", "completed", image_url="https://example.com/img2.png"),
        ]
        room = _make_room("r1", "completed", variations)
        proj = _make_project(rooms=[room])

        assert reconcile_project(proj) is True
        assert room["status"] == "completed"
        assert proj["status"] == "completed"

    def test_mix_completed_and_processing_rooms(self):
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
        assert proj["status"] == "completed"  # Has at least one completed room

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
        assert reconcile_project(proj, force=True) is True  # Forced
        assert proj["status"] == "pending"

    def test_all_failed_rooms_results_in_failed_project(self):
        room = _make_room("r1", "failed", [
            _make_variation("v1", "failed", error="boom"),
        ])
        proj = _make_project(rooms=[room])

        assert reconcile_project(proj) is True
        assert proj["status"] == "failed"

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
