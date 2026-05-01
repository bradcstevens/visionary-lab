"""Tests for the prompt ↔ design_brief.global_instructions mirror added to
``PATCH /api/v1/staging/projects/{id}`` and ``PUT /api/v1/staging/projects/{id}/brief``
in issue 001 of the project-settings-completeness PRD.

Mirror contract (asserted by the tests below — verbatim from the PRD's
"Backend mirror behavior" section):

    Goal: ``project.prompt`` and ``project.design_brief.global_instructions``
    are kept in sync so the user has one coherent "prompt" across Settings,
    Brief, gallery dialogs, project cards, regenerate flows, and any future
    snapshot-restore path.

    PATCH /projects/{id} (update_project), with sent = body.__fields_set__:

        - both prompt and design_brief in sent:
            brief wins. If the brief is a dict and its ``global_instructions``
            is a non-empty string (after ``str.strip()``), then
            ``project.prompt`` is set from it. Otherwise the user's submitted
            ``prompt`` value is preserved (the brief has nothing meaningful
            to override it with).

        - only prompt in sent:
            If a brief is currently persisted on the project (dict), copy the
            new prompt into the brief's ``global_instructions``. If no brief
            is persisted, only ``prompt`` changes (no brief is spawned).

        - only design_brief in sent:
            If brief is dict and ``global_instructions`` is non-empty after
            strip → ``project.prompt`` mirrors. Otherwise (brief cleared via
            None, brief is empty {}, ``global_instructions`` missing or
            empty/whitespace-only) → ``prompt`` is untouched.

    PUT /projects/{id}/brief (update_brief):

        Always: if ``brief.global_instructions`` is non-empty after strip →
        ``project.prompt`` is set from it. Otherwise the persisted ``prompt``
        is left untouched. The brief itself is always saved.

The mirror lives in the endpoint layer (not the storage layer).
``staging_storage.update_project`` does a single-level top-level dict merge
(``existing.update(updates)`` then ``replace_item(body=existing)``), so the
endpoint passes the full mutated ``project_data`` for PATCH and a small
``{prompt, design_brief}`` partial for PUT — both end up as the same
single ``replace_item`` call to Cosmos.

Both handlers run inside ``backend.core.staging_pipeline._get_project_lock``
to serialize against the regen finalizers and pipeline workers — the PUT
handler had no lock pre-fix, and adding mirror-driven ``prompt`` writes
to the PUT path widened the loss surface enough that the lock is
correctness here, not scope creep.

Out of scope for this slice: ``POST /projects/{id}/brief`` (the
``generate_brief`` handler that synthesizes a fresh brief from chat). The
PRD explicitly scopes the mirror to the two user-facing inbound update
endpoints; ``POST /brief`` is a system-driven brief-synthesis path whose
output is downstream of the prompt the user already controls via PATCH.
"""
import json

import pytest


# ---------------------------------------------------------------------------
# Shared fixture: a project with both a prompt and a brief, and a project
# WITHOUT a brief, for the two PATCH branches.
# ---------------------------------------------------------------------------


def _project_with_brief() -> dict:
    """Project with a non-trivial design_brief whose global_instructions
    differ from project.prompt — so a successful mirror is observable as
    one field changing to match the other."""
    return {
        "id": "proj-mirror",
        "name": "Mirror Project",
        "prompt": "ORIGINAL PROMPT",
        "status": "completed",
        "rooms": [],
        "settings": {
            "variations_per_room": 3,
            "model": "gpt-image-2",
            "quality": "high",
            "size": "auto",
        },
        "analyses": [],
        "design_brief": {
            "global_instructions": "ORIGINAL BRIEF GI",
            "object_palette": [],
            "placement_guide": {},
            "per_image_notes": {},
            "per_image_objects": {},
            "preserve_elements": [],
        },
    }


def _project_without_brief() -> dict:
    p = _project_with_brief()
    p["id"] = "proj-no-brief"
    p["design_brief"] = None
    return p


def _captured_replace_body(mock_container) -> dict:
    """Pull the body from the most recent ``replace_item`` call. Matches
    the helper used by the existing PATCH-project test suite."""
    call = mock_container.replace_item.call_args
    return call.kwargs.get("body") or call.args[1]


# ---------------------------------------------------------------------------
# PATCH branch 1: only ``prompt`` is sent.
# ---------------------------------------------------------------------------


def test_patch_prompt_only_with_brief_mirrors_into_brief_global_instructions(
    client, mock_staging_deps,
):
    """PRD case: only ``prompt`` is sent and the project has a brief →
    the new prompt is copied into ``design_brief.global_instructions``.
    Both fields end up equal to the new value.
    """
    mock_container = mock_staging_deps["container"]
    project = _project_with_brief()
    mock_container.read_item.return_value = project
    mock_container.replace_item.side_effect = lambda item, body: body

    response = client.patch(
        "/api/v1/staging/projects/proj-mirror",
        json={"prompt": "NEW PROMPT FROM SETTINGS"},
    )
    assert response.status_code == 200, response.text

    persisted = _captured_replace_body(mock_container)
    assert persisted["prompt"] == "NEW PROMPT FROM SETTINGS"
    assert persisted["design_brief"]["global_instructions"] == "NEW PROMPT FROM SETTINGS"
    # Other brief fields preserved.
    assert persisted["design_brief"]["object_palette"] == []
    assert persisted["design_brief"]["preserve_elements"] == []


def test_patch_prompt_only_without_brief_does_not_spawn_brief(
    client, mock_staging_deps,
):
    """PRD case: only ``prompt`` is sent and the project has NO brief →
    only ``prompt`` changes; the mirror does NOT spawn an empty brief.
    """
    mock_container = mock_staging_deps["container"]
    project = _project_without_brief()
    mock_container.read_item.return_value = project
    mock_container.replace_item.side_effect = lambda item, body: body

    response = client.patch(
        "/api/v1/staging/projects/proj-no-brief",
        json={"prompt": "NEW PROMPT NO BRIEF"},
    )
    assert response.status_code == 200, response.text

    persisted = _captured_replace_body(mock_container)
    assert persisted["prompt"] == "NEW PROMPT NO BRIEF"
    # No brief was created from thin air.
    assert persisted["design_brief"] is None


# ---------------------------------------------------------------------------
# PATCH branch 2: only ``design_brief`` is sent.
# ---------------------------------------------------------------------------


def test_patch_brief_only_with_nonempty_global_instructions_mirrors_to_prompt(
    client, mock_staging_deps,
):
    """PRD case: only ``design_brief`` is sent with non-empty
    ``global_instructions`` → ``project.prompt`` mirrors that value.
    """
    mock_container = mock_staging_deps["container"]
    project = _project_with_brief()
    mock_container.read_item.return_value = project
    mock_container.replace_item.side_effect = lambda item, body: body

    new_brief = {
        "global_instructions": "BRIEF EDIT FROM BRIEF TAB",
        "object_palette": [{"id": "o1", "name": "lamp", "category": "lighting"}],
        "placement_guide": {},
        "per_image_notes": {},
        "per_image_objects": {},
        "preserve_elements": [],
    }

    response = client.patch(
        "/api/v1/staging/projects/proj-mirror",
        json={"design_brief": new_brief},
    )
    assert response.status_code == 200, response.text

    persisted = _captured_replace_body(mock_container)
    assert persisted["design_brief"] == new_brief
    assert persisted["prompt"] == "BRIEF EDIT FROM BRIEF TAB"


def test_patch_brief_only_with_empty_string_global_instructions_does_not_touch_prompt(
    client, mock_staging_deps,
):
    """PRD case: only ``design_brief`` is sent and its
    ``global_instructions`` is the empty string → ``project.prompt`` is
    NOT touched. The brief itself is still persisted.
    """
    mock_container = mock_staging_deps["container"]
    project = _project_with_brief()
    mock_container.read_item.return_value = project
    mock_container.replace_item.side_effect = lambda item, body: body

    new_brief = {
        "global_instructions": "",
        "object_palette": [],
        "placement_guide": {},
        "per_image_notes": {},
        "per_image_objects": {},
        "preserve_elements": [],
    }

    response = client.patch(
        "/api/v1/staging/projects/proj-mirror",
        json={"design_brief": new_brief},
    )
    assert response.status_code == 200, response.text

    persisted = _captured_replace_body(mock_container)
    assert persisted["design_brief"]["global_instructions"] == ""
    assert persisted["prompt"] == "ORIGINAL PROMPT"  # untouched


def test_patch_brief_only_with_whitespace_only_global_instructions_does_not_touch_prompt(
    client, mock_staging_deps,
):
    """Stricter empty check: ``"   "`` is treated as empty by the mirror
    so we don't mirror whitespace garbage into ``project.prompt``.
    """
    mock_container = mock_staging_deps["container"]
    project = _project_with_brief()
    mock_container.read_item.return_value = project
    mock_container.replace_item.side_effect = lambda item, body: body

    new_brief = {
        "global_instructions": "   ",
        "object_palette": [],
        "placement_guide": {},
        "per_image_notes": {},
        "per_image_objects": {},
        "preserve_elements": [],
    }

    response = client.patch(
        "/api/v1/staging/projects/proj-mirror",
        json={"design_brief": new_brief},
    )
    assert response.status_code == 200, response.text

    persisted = _captured_replace_body(mock_container)
    assert persisted["prompt"] == "ORIGINAL PROMPT"


def test_patch_brief_only_explicit_null_clears_brief_does_not_touch_prompt(
    client, mock_staging_deps,
):
    """PRD case: ``design_brief: null`` clears the brief. There is no
    ``global_instructions`` to mirror, so ``project.prompt`` is untouched.
    """
    mock_container = mock_staging_deps["container"]
    project = _project_with_brief()
    mock_container.read_item.return_value = project
    mock_container.replace_item.side_effect = lambda item, body: body

    response = client.patch(
        "/api/v1/staging/projects/proj-mirror",
        json={"design_brief": None},
    )
    assert response.status_code == 200, response.text

    persisted = _captured_replace_body(mock_container)
    assert persisted["design_brief"] is None
    assert persisted["prompt"] == "ORIGINAL PROMPT"


# ---------------------------------------------------------------------------
# PATCH branch 3: BOTH ``prompt`` and ``design_brief`` are sent.
# ---------------------------------------------------------------------------


def test_patch_both_present_brief_wins_when_brief_global_instructions_nonempty(
    client, mock_staging_deps,
):
    """PRD case: both fields present, brief has non-empty
    ``global_instructions`` → brief wins. Both persisted fields equal the
    brief's ``global_instructions`` (the user-supplied ``prompt`` is
    overridden).
    """
    mock_container = mock_staging_deps["container"]
    project = _project_with_brief()
    mock_container.read_item.return_value = project
    mock_container.replace_item.side_effect = lambda item, body: body

    new_brief = {
        "global_instructions": "BRIEF WINS",
        "object_palette": [],
        "placement_guide": {},
        "per_image_notes": {},
        "per_image_objects": {},
        "preserve_elements": [],
    }

    response = client.patch(
        "/api/v1/staging/projects/proj-mirror",
        json={
            "prompt": "USER ALSO SENT THIS",
            "design_brief": new_brief,
        },
    )
    assert response.status_code == 200, response.text

    persisted = _captured_replace_body(mock_container)
    assert persisted["design_brief"]["global_instructions"] == "BRIEF WINS"
    assert persisted["prompt"] == "BRIEF WINS"


def test_patch_both_present_user_prompt_kept_when_brief_global_instructions_empty(
    client, mock_staging_deps,
):
    """Edge case codified per duck's review: both fields present and
    brief's ``global_instructions`` is empty → brief has nothing
    meaningful to override with, so the user-supplied ``prompt`` is
    preserved on ``project.prompt``. The brief is still persisted as-is
    (empty-string ``global_instructions`` and all).
    """
    mock_container = mock_staging_deps["container"]
    project = _project_with_brief()
    mock_container.read_item.return_value = project
    mock_container.replace_item.side_effect = lambda item, body: body

    new_brief = {
        "global_instructions": "",
        "object_palette": [],
        "placement_guide": {},
        "per_image_notes": {},
        "per_image_objects": {},
        "preserve_elements": [],
    }

    response = client.patch(
        "/api/v1/staging/projects/proj-mirror",
        json={
            "prompt": "USER PROMPT KEPT",
            "design_brief": new_brief,
        },
    )
    assert response.status_code == 200, response.text

    persisted = _captured_replace_body(mock_container)
    assert persisted["prompt"] == "USER PROMPT KEPT"
    assert persisted["design_brief"]["global_instructions"] == ""


def test_patch_both_present_user_prompt_kept_when_brief_cleared_to_null(
    client, mock_staging_deps,
):
    """Edge case: both fields present and ``design_brief`` is null →
    brief is cleared. There's no ``global_instructions`` to mirror, so
    the user's submitted ``prompt`` is preserved.
    """
    mock_container = mock_staging_deps["container"]
    project = _project_with_brief()
    mock_container.read_item.return_value = project
    mock_container.replace_item.side_effect = lambda item, body: body

    response = client.patch(
        "/api/v1/staging/projects/proj-mirror",
        json={
            "prompt": "USER PROMPT KEPT EVEN WITH BRIEF NULL",
            "design_brief": None,
        },
    )
    assert response.status_code == 200, response.text

    persisted = _captured_replace_body(mock_container)
    assert persisted["prompt"] == "USER PROMPT KEPT EVEN WITH BRIEF NULL"
    assert persisted["design_brief"] is None


# ---------------------------------------------------------------------------
# PUT /brief mirror.
# ---------------------------------------------------------------------------


def test_put_brief_with_nonempty_global_instructions_mirrors_to_prompt(
    client, mock_staging_deps,
):
    """PRD case: PUT /brief with non-empty ``global_instructions`` →
    ``project.prompt`` mirrors that value, persisted in the same
    Cosmos write that saves the brief.
    """
    mock_container = mock_staging_deps["container"]
    project = _project_with_brief()
    mock_container.read_item.return_value = project
    mock_container.replace_item.side_effect = lambda item, body: body

    new_brief = {
        "global_instructions": "BRIEF TAB DIRECT EDIT",
        "object_palette": [],
        "placement_guide": {},
        "per_image_notes": {},
        "per_image_objects": {},
        "preserve_elements": [],
    }

    response = client.put(
        "/api/v1/staging/projects/proj-mirror/brief",
        json=new_brief,
    )
    assert response.status_code == 200, response.text

    persisted = _captured_replace_body(mock_container)
    assert persisted["design_brief"]["global_instructions"] == "BRIEF TAB DIRECT EDIT"
    assert persisted["prompt"] == "BRIEF TAB DIRECT EDIT"


def test_put_brief_response_includes_brief_for_frontend_wrapper(
    client, mock_staging_deps,
):
    """Tightly-coupled bug fix: the frontend ``updateBrief`` wrapper
    (see ``frontend/services/stagingApi.ts``) does
    ``return data.brief`` on the JSON response. Pre-fix, the handler
    had no explicit ``return``, so FastAPI sent ``null`` as the body
    and the wrapper crashed silently into the wizard's "Failed to
    save Design Brief" toast. The handler now returns
    ``{"brief": <persisted brief dict>}`` so the wrapper works.
    """
    mock_container = mock_staging_deps["container"]
    project = _project_with_brief()
    mock_container.read_item.return_value = project
    mock_container.replace_item.side_effect = lambda item, body: body

    new_brief = {
        "global_instructions": "frontend wrapper expects this shape",
        "object_palette": [],
        "placement_guide": {},
        "per_image_notes": {},
        "per_image_objects": {},
        "preserve_elements": [],
    }
    response = client.put(
        "/api/v1/staging/projects/proj-mirror/brief",
        json=new_brief,
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert isinstance(body, dict)
    assert "brief" in body
    assert body["brief"]["global_instructions"] == "frontend wrapper expects this shape"


def test_put_brief_with_whitespace_only_global_instructions_does_not_touch_prompt(
    client, mock_staging_deps,
):
    """PRD case: PUT /brief with whitespace-only ``global_instructions``
    → brief is still saved, but ``project.prompt`` is left untouched
    (we don't mirror whitespace garbage).

    DesignBrief requires global_instructions: str (no min length), so the
    Pydantic model accepts ``"   "``. The mirror gates on
    ``isinstance(...) and gi.strip()``.
    """
    mock_container = mock_staging_deps["container"]
    project = _project_with_brief()
    mock_container.read_item.return_value = project
    mock_container.replace_item.side_effect = lambda item, body: body

    new_brief = {
        "global_instructions": "   ",
        "object_palette": [],
        "placement_guide": {},
        "per_image_notes": {},
        "per_image_objects": {},
        "preserve_elements": [],
    }

    response = client.put(
        "/api/v1/staging/projects/proj-mirror/brief",
        json=new_brief,
    )
    assert response.status_code == 200, response.text

    persisted = _captured_replace_body(mock_container)
    assert persisted["design_brief"]["global_instructions"] == "   "
    assert persisted["prompt"] == "ORIGINAL PROMPT"


# ---------------------------------------------------------------------------
# Restore-shaped payload regression guard (proxy for PRD AC user story 20).
#
# No version-snapshot/restore endpoint exists in the backend today (the
# PRD's "History/revert" claim is forward-looking). This test guards the
# SHAPE that a future restore path would emit: a single PATCH carrying
# both prompt and design_brief, starting from a project whose persisted
# prompt and brief.global_instructions diverge. After the PATCH, the
# persisted document's prompt MUST equal design_brief.global_instructions
# — proving the mirror leaves no divergence even when the input itself
# was incoherent.
# ---------------------------------------------------------------------------


def test_restore_shaped_patch_payload_ends_coherent(
    client, mock_staging_deps,
):
    """A snapshot-restore-shaped PATCH (both fields, divergent input)
    ends with ``persisted.prompt == persisted.design_brief.global_instructions``
    — the brief-wins rule keeps both fields coherent in a single write.
    Documents the user-story-20 intent without depending on a
    restore endpoint that doesn't exist yet.
    """
    mock_container = mock_staging_deps["container"]
    # Pre-existing project state has its OWN divergent prompt/brief.
    project = _project_with_brief()
    project["prompt"] = "PRE-EXISTING DIVERGENT PROMPT"
    project["design_brief"]["global_instructions"] = "PRE-EXISTING DIVERGENT BRIEF GI"
    mock_container.read_item.return_value = project
    mock_container.replace_item.side_effect = lambda item, body: body

    # The "snapshot" we're restoring TO has its own (also divergent)
    # prompt and brief — exactly what a hand-constructed restore
    # payload would look like before the mirror does its work.
    snapshot_prompt = "SNAPSHOT-ERA PROMPT"
    snapshot_brief = {
        "global_instructions": "SNAPSHOT-ERA BRIEF GI",
        "object_palette": [],
        "placement_guide": {},
        "per_image_notes": {},
        "per_image_objects": {},
        "preserve_elements": [],
    }

    response = client.patch(
        "/api/v1/staging/projects/proj-mirror",
        json={"prompt": snapshot_prompt, "design_brief": snapshot_brief},
    )
    assert response.status_code == 200, response.text

    persisted = _captured_replace_body(mock_container)
    # Coherent: prompt and brief.global_instructions match.
    assert persisted["prompt"] == persisted["design_brief"]["global_instructions"]
    # And brief wins on the value.
    assert persisted["prompt"] == "SNAPSHOT-ERA BRIEF GI"


# ---------------------------------------------------------------------------
# Regression pins for behavior that must remain unchanged after the mirror
# is added.
# ---------------------------------------------------------------------------


def test_patch_does_not_invoke_pipeline_after_mirror_added(
    client, mock_staging_deps,
):
    """The mirror is a pure metadata transform — it must not introduce
    any pipeline call. Existing "patch is a metadata-only op" invariant
    holds.
    """
    mock_container = mock_staging_deps["container"]
    mock_pipeline = mock_staging_deps["pipeline"]
    mock_container.read_item.return_value = _project_with_brief()
    mock_container.replace_item.side_effect = lambda item, body: body

    response = client.patch(
        "/api/v1/staging/projects/proj-mirror",
        json={"prompt": "no-pipeline check"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")

    assert not mock_pipeline.process_room.called
    assert not mock_pipeline.generate_project.called
    assert not mock_pipeline.process_single_variation.called


def test_put_brief_does_not_invoke_pipeline(client, mock_staging_deps):
    """Same invariant for PUT /brief: still a metadata-only write,
    even with the new mirror + lock."""
    mock_container = mock_staging_deps["container"]
    mock_pipeline = mock_staging_deps["pipeline"]
    mock_container.read_item.return_value = _project_with_brief()
    mock_container.replace_item.side_effect = lambda item, body: body

    new_brief = {
        "global_instructions": "no pipeline",
        "object_palette": [],
        "placement_guide": {},
        "per_image_notes": {},
        "per_image_objects": {},
        "preserve_elements": [],
    }
    response = client.put(
        "/api/v1/staging/projects/proj-mirror/brief",
        json=new_brief,
    )
    assert response.status_code == 200
    assert not mock_pipeline.process_room.called
    assert not mock_pipeline.generate_project.called
    assert not mock_pipeline.process_single_variation.called


def test_patch_settings_only_does_not_change_prompt_or_brief(
    client, mock_staging_deps,
):
    """The mirror is gated on prompt/design_brief being in the request.
    A settings-only PATCH must NOT trigger any mirror activity — both
    ``prompt`` and ``design_brief.global_instructions`` are byte-identical
    after the write. This pins the existing "settings-merge" behavior
    from issue 002 of projects-page-improvements against accidental
    interaction with the new mirror.
    """
    mock_container = mock_staging_deps["container"]
    project = _project_with_brief()
    mock_container.read_item.return_value = project
    mock_container.replace_item.side_effect = lambda item, body: body

    prompt_before = project["prompt"]
    brief_before = json.dumps(project["design_brief"], sort_keys=True)

    response = client.patch(
        "/api/v1/staging/projects/proj-mirror",
        json={"settings": {"variations_per_room": 7}},
    )
    assert response.status_code == 200

    persisted = _captured_replace_body(mock_container)
    assert persisted["prompt"] == prompt_before
    assert json.dumps(persisted["design_brief"], sort_keys=True) == brief_before
    assert persisted["settings"]["variations_per_room"] == 7
    # Settings-merge invariant survives.
    assert persisted["settings"]["model"] == "gpt-image-2"


def test_patch_name_only_does_not_change_prompt_or_brief(
    client, mock_staging_deps,
):
    """A name-only PATCH must NOT trigger any mirror activity. Same
    invariant as the settings-only test, pinned independently because
    name and settings live on different code branches in the handler.
    """
    mock_container = mock_staging_deps["container"]
    project = _project_with_brief()
    mock_container.read_item.return_value = project
    mock_container.replace_item.side_effect = lambda item, body: body

    prompt_before = project["prompt"]
    brief_before = json.dumps(project["design_brief"], sort_keys=True)

    response = client.patch(
        "/api/v1/staging/projects/proj-mirror",
        json={"name": "Renamed via Settings"},
    )
    assert response.status_code == 200

    persisted = _captured_replace_body(mock_container)
    assert persisted["name"] == "Renamed via Settings"
    assert persisted["prompt"] == prompt_before
    assert json.dumps(persisted["design_brief"], sort_keys=True) == brief_before
