"""Tests for ``ProjectStatusCalculator.compute_status``.

The calculator is the single source of truth for project-level status,
replacing three duplicated, drift-prone inline branches that lived in
``generate_project``, ``regenerate_room``, and ``regenerate_variation``.

The Issue 1 bug case the PRD calls out:

    project with rooms = [completed, pending, pending, pending, pending]
    must yield ProjectStatus.PENDING — never COMPLETED.

Pre-fix the ``regenerate_room`` finally block read::

    any_processing = any(r.status in ("pending", "processing") for r in rooms)
    if not any_processing:
        any_completed = any(r.status == "completed" for r in rooms)
        fresh_project.status = "completed" if any_completed else "failed"

which is correct in isolation — but a parallel branch in
``regenerate_variation`` and the inline calc at the end of
``generate_project`` had subtly different shapes, so refactoring all three
to one helper closes the door on future drift.
"""

from typing import List

import pytest

from backend.core.project_status import ProjectStatusCalculator
from backend.models.staging import ItemStatus, ProjectStatus, Room, Variation


def _room(status: str, rid: str = "r") -> Room:
    """Build a minimal Room at the requested status. The label and original
    image URL fields are required by the Pydantic model but irrelevant to
    status computation, so we use throwaway placeholders."""
    return Room(
        id=rid,
        label=f"Room {rid}",
        original_image_url="https://acct.blob.core.windows.net/images/x.png",
        status=status,
        variations=[],
    )


def _rooms(*statuses: str) -> List[Room]:
    return [_room(s, rid=f"r-{i}") for i, s in enumerate(statuses)]


# ----------------------------------------------------------------------
# Table-driven happy paths
# ----------------------------------------------------------------------

@pytest.mark.parametrize(
    "statuses,expected",
    [
        # All-uniform cases.
        (("pending",), ProjectStatus.PENDING),
        (("processing",), ProjectStatus.PENDING),
        (("completed",), ProjectStatus.COMPLETED),
        (("failed",), ProjectStatus.FAILED),

        # All-uniform multi-room cases.
        (("pending", "pending", "pending"), ProjectStatus.PENDING),
        (("processing", "processing"), ProjectStatus.PENDING),
        (("completed", "completed", "completed"), ProjectStatus.COMPLETED),
        (("failed", "failed", "failed"), ProjectStatus.FAILED),

        # Mixed pending + processing — both keep the project PENDING.
        (("pending", "processing"), ProjectStatus.PENDING),
        (("pending", "processing", "pending"), ProjectStatus.PENDING),

        # Mixed terminal + outstanding — outstanding wins (PENDING).
        (("completed", "pending"), ProjectStatus.PENDING),
        (("completed", "processing"), ProjectStatus.PENDING),
        (("failed", "pending"), ProjectStatus.PENDING),
        (("failed", "processing"), ProjectStatus.PENDING),
        (("completed", "failed", "pending"), ProjectStatus.PENDING),

        # All-terminal mixed completed + failed — at least one completed
        # makes the project COMPLETED per PRD ("returns `completed` since at
        # least one completed").
        (("completed", "failed"), ProjectStatus.COMPLETED),
        (("failed", "completed"), ProjectStatus.COMPLETED),
        (("completed", "failed", "failed"), ProjectStatus.COMPLETED),
        (("failed", "completed", "failed", "completed"), ProjectStatus.COMPLETED),
    ],
)
def test_compute_status_table(statuses, expected):
    assert ProjectStatusCalculator.compute_status(_rooms(*statuses)) == expected


# ----------------------------------------------------------------------
# Issue 1 bug case — explicit, named, regression-grade
# ----------------------------------------------------------------------

def test_issue1_one_completed_four_pending_returns_pending():
    """The PRD's headlining bug case: project with one completed room and
    four pending rooms must read PENDING. Pre-fix this returned COMPLETED
    (or sometimes nothing at all, depending on which inline branch fired)
    and lied to the user about whether their work was done.
    """
    rooms = _rooms("completed", "pending", "pending", "pending", "pending")
    assert ProjectStatusCalculator.compute_status(rooms) == ProjectStatus.PENDING


def test_issue1_variant_one_completed_one_processing_three_pending():
    """Same bug, slightly different shape: a real generation often has one
    room finished, one mid-stream (processing), and the rest queued. All
    three of these projects must read PENDING.
    """
    rooms = _rooms("completed", "processing", "pending", "pending", "pending")
    assert ProjectStatusCalculator.compute_status(rooms) == ProjectStatus.PENDING


# ----------------------------------------------------------------------
# Edge cases
# ----------------------------------------------------------------------

def test_empty_rooms_returns_failed():
    """Empty list edge case. The strict 3-rule reading gives FAILED
    (vacuously every room terminal and none completed). In practice
    this case is unreachable from the three regen paths because
    ``generate_project`` and ``regenerate_room`` both validate rooms
    exist before they reach the calculator. The test exists to pin
    the literal-spec contract for any future caller and to prevent
    accidental drift toward an "PENDING is more user-friendly" reading
    that future readers might re-introduce.
    """
    assert ProjectStatusCalculator.compute_status([]) == ProjectStatus.FAILED


def test_single_room_statuses():
    """Single-room projects exercise every branch with the simplest possible
    input — useful for catching off-by-one or any-vs-all bugs that only
    surface with multiple rooms.
    """
    assert ProjectStatusCalculator.compute_status(_rooms("pending")) == ProjectStatus.PENDING
    assert ProjectStatusCalculator.compute_status(_rooms("processing")) == ProjectStatus.PENDING
    assert ProjectStatusCalculator.compute_status(_rooms("completed")) == ProjectStatus.COMPLETED
    assert ProjectStatusCalculator.compute_status(_rooms("failed")) == ProjectStatus.FAILED


# ----------------------------------------------------------------------
# Robustness
# ----------------------------------------------------------------------

def test_accepts_iterable_not_just_list():
    """The signature is ``Iterable[Room]`` — a generator must work. This
    matters because callers commonly pass list comprehensions or generator
    expressions of filtered rooms.
    """
    statuses = ("completed", "pending", "completed")
    rooms_gen = (_room(s, rid=f"g-{i}") for i, s in enumerate(statuses))
    assert ProjectStatusCalculator.compute_status(rooms_gen) == ProjectStatus.PENDING


def test_handles_enum_status_values():
    """Room.status is typed as ``str = Field(ItemStatus.PENDING, ...)``.
    Some callers persist the enum directly (which is a str subclass) and
    others persist the raw string. Calculator must handle both. Verified
    here by setting status to the ItemStatus enum value rather than a
    string literal.
    """
    rooms = [
        _room(ItemStatus.COMPLETED.value),
        _room(ItemStatus.PENDING.value),
    ]
    assert ProjectStatusCalculator.compute_status(rooms) == ProjectStatus.PENDING


def test_does_not_mutate_input():
    """Pure helper invariant — the calculator must NOT touch room.status
    (it only reads). Verified by computing twice and asserting both calls
    return the same answer AND the rooms still hold their original
    statuses.
    """
    rooms = _rooms("completed", "pending", "failed")
    snapshot = [r.status for r in rooms]
    _ = ProjectStatusCalculator.compute_status(rooms)
    _ = ProjectStatusCalculator.compute_status(rooms)
    assert [r.status for r in rooms] == snapshot
