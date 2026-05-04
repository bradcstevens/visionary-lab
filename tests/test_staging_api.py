"""Tests for staging API endpoints."""
import pytest
from unittest.mock import MagicMock, patch


def test_create_project(client, mock_staging_deps):
    mock_container = mock_staging_deps["container"]
    mock_container.create_item.return_value = {
        "id": "proj-123",
        "name": "Test Project",
        "prompt": "Modern minimalist",
        "status": "uploading",
        "rooms": [],
        "settings": {"variations_per_room": 5, "model": "gpt-image-2", "quality": "high", "size": "auto"},
        "created_at": "2026-04-26T00:00:00Z",
        "updated_at": "2026-04-26T00:00:00Z",
        "doc_type": "staging_project",
    }

    response = client.post("/api/v1/staging/projects", json={
        "name": "Test Project",
        "prompt": "Modern minimalist",
    })
    assert response.status_code == 201
    data = response.json()
    assert data["project"]["name"] == "Test Project"
    assert data["project"]["status"] == "uploading"


def test_list_projects(client, mock_staging_deps):
    mock_container = mock_staging_deps["container"]
    mock_container.query_items.return_value = []
    
    # Mock count query - returns list with a single integer result
    def mock_query_items(query=None, **kwargs):
        if "SELECT VALUE COUNT(1)" in query:
            return [0]  # Count query returns list with integer
        return []
    
    mock_container.query_items = mock_query_items

    response = client.get("/api/v1/staging/projects")
    assert response.status_code == 200
    data = response.json()
    assert data["projects"] == []
    assert data["total"] == 0


def test_get_project(client, mock_staging_deps):
    mock_container = mock_staging_deps["container"]
    mock_container.read_item.return_value = {
        "id": "proj-123",
        "name": "Test",
        "prompt": "Test prompt",
        "status": "uploading",
        "rooms": [],
        "settings": {"variations_per_room": 5, "model": "gpt-image-2", "quality": "high", "size": "auto"},
    }

    response = client.get("/api/v1/staging/projects/proj-123")
    assert response.status_code == 200
    assert response.json()["project"]["id"] == "proj-123"


def test_get_project_not_found(client, mock_staging_deps):
    from azure.cosmos.exceptions import CosmosResourceNotFoundError
    mock_container = mock_staging_deps["container"]
    mock_container.read_item.side_effect = CosmosResourceNotFoundError(status_code=404, message="Not found")
    response = client.get("/api/v1/staging/projects/nonexistent")
    assert response.status_code == 404


def test_delete_project(client, mock_staging_deps):
    mock_container = mock_staging_deps["container"]
    mock_container.read_item.return_value = {"id": "proj-123"}  # Found
    mock_container.delete_item.return_value = None  # Success
    response = client.delete("/api/v1/staging/projects/proj-123")
    assert response.status_code == 200


def test_delete_project_not_found(client, mock_staging_deps):
    from azure.cosmos.exceptions import CosmosResourceNotFoundError
    mock_container = mock_staging_deps["container"]
    # The delete endpoint calls storage.delete_project which tries to delete directly
    mock_container.delete_item.side_effect = CosmosResourceNotFoundError(status_code=404, message="Not found")
    response = client.delete("/api/v1/staging/projects/nonexistent")
    assert response.status_code == 404


# --- Variation regeneration tests ---

def _project_with_completed_variation():
    """Helper: project with one room, one completed variation with metadata."""
    return {
        "id": "proj-123",
        "name": "Test",
        "prompt": "Modern minimalist",
        "status": "completed",
        "rooms": [{
            "id": "room-1",
            "label": "Living Room",
            "original_image_url": "https://acct.blob.core.windows.net/images/staging/proj/originals/photo.png",
            "status": "completed",
            "variations": [{
                "id": "var-1",
                "status": "completed",
                "image_url": "https://acct.blob.core.windows.net/images/staging/proj/variations/room-1/img.png",
                "generation_metadata": {
                    "model": "gpt-image-2",
                    "adapted_prompt": "Add a cozy reading nook with warm lighting",
                    "generation_time_ms": 5000,
                },
            }],
        }],
        "settings": {"variations_per_room": 1, "model": "gpt-image-2", "quality": "high", "size": "auto"},
    }


def test_regenerate_variation_not_found_project(client, mock_staging_deps):
    mock_container = mock_staging_deps["container"]
    mock_container.read_item.return_value = None
    response = client.post("/api/v1/staging/projects/nope/rooms/room-1/variations/var-1/regenerate?strategy=fresh")
    assert response.status_code == 404


def test_regenerate_variation_not_found_room(client, mock_staging_deps):
    mock_container = mock_staging_deps["container"]
    mock_container.read_item.return_value = _project_with_completed_variation()
    response = client.post("/api/v1/staging/projects/proj-123/rooms/bad-room/variations/var-1/regenerate?strategy=fresh")
    assert response.status_code == 404


def test_regenerate_variation_not_found_variation(client, mock_staging_deps):
    mock_container = mock_staging_deps["container"]
    mock_container.read_item.return_value = _project_with_completed_variation()
    response = client.post("/api/v1/staging/projects/proj-123/rooms/room-1/variations/bad-var/regenerate?strategy=fresh")
    assert response.status_code == 404


def test_regenerate_variation_invalid_strategy(client, mock_staging_deps):
    mock_container = mock_staging_deps["container"]
    mock_container.read_item.return_value = _project_with_completed_variation()
    response = client.post("/api/v1/staging/projects/proj-123/rooms/room-1/variations/var-1/regenerate?strategy=invalid")
    assert response.status_code == 400


def test_regenerate_variation_already_processing(client, mock_staging_deps):
    mock_container = mock_staging_deps["container"]
    project_data = _project_with_completed_variation()
    project_data["rooms"][0]["variations"][0]["status"] = "processing"
    mock_container.read_item.return_value = project_data
    response = client.post("/api/v1/staging/projects/proj-123/rooms/room-1/variations/var-1/regenerate?strategy=fresh")
    assert response.status_code == 409


def test_regenerate_variation_preflight_preserves_image_url(client, mock_staging_deps):
    """Issue 002: the endpoint preflight must NOT clear `variation.image_url`.

    The pipeline captures the prior URL for failure rollback and old-blob
    cleanup, so wiping it in the preflight write would defeat that contract.
    Verifies the preflight write keeps the prior image_url intact and sets
    status=processing (not pending).
    """
    mock_container = mock_staging_deps["container"]
    project_data = _project_with_completed_variation()
    prior_image_url = project_data["rooms"][0]["variations"][0]["image_url"]
    mock_container.read_item.return_value = project_data
    # Echo replace_item back so update_project returns valid data
    mock_container.replace_item.side_effect = lambda item, body: body

    # The pipeline's process_single_variation is awaited inside the SSE
    # generator. We don't need it to do anything — the preflight write happens
    # before the generator yields, so we let the response stream to completion
    # but the pipeline is mocked to return an empty async generator.
    mock_pipeline = mock_staging_deps["pipeline"]

    async def _empty_async_gen(*_args, **_kwargs):
        if False:
            yield  # pragma: no cover

    mock_pipeline.process_single_variation = _empty_async_gen

    with client.stream(
        "POST",
        "/api/v1/staging/projects/proj-123/rooms/room-1/variations/var-1/regenerate?strategy=retry",
    ) as response:
        assert response.status_code == 200
        # Drain the stream so the preflight + final writes complete.
        for _ in response.iter_bytes():
            pass

    # The first replace_item call is the preflight write.
    assert mock_container.replace_item.call_count >= 1
    first_call = mock_container.replace_item.call_args_list[0]
    persisted_body = first_call.kwargs.get("body") or first_call.args[1]
    persisted_variation = persisted_body["rooms"][0]["variations"][0]
    # Critical: the prior image_url is preserved through the preflight write.
    assert persisted_variation["image_url"] == prior_image_url, \
        "Preflight write must NOT clear variation.image_url; the pipeline " \
        "needs it for failure rollback and old-blob cleanup (issue 002)."
    # And the variation is now PROCESSING (not PENDING) so the 409 mutex works.
    assert persisted_variation["status"] == "processing"
    # The error field is cleared (in case the variation was previously FAILED).
    assert persisted_variation.get("error") is None

# ============================================================================
# Legacy plant_palette → object_palette migration on read.
#
# Verifies: per-image-object-quantities issue 001 — old persisted briefs are
# transparently migrated when surfaced via the GET project endpoints AND the
# migrated dict is written back so the next read is a no-op.
# ============================================================================


def _legacy_brief_payload():
    """A persisted-shape design brief from the pre-migration era."""
    return {
        "global_instructions": "Add evergreens",
        "plant_palette": [
            {
                "species": "Sequoia",
                "botanical_name": "Sequoiadendron giganteum",
                "quantity": 2,
                "size": "20 ft tall",
                "placement": "north fence",
                "visual_notes": "tall, conical",
            }
        ],
        "placement_guide": {"back_row": "Tall conifers"},
        "preserve_elements": ["patio"],
    }


def _migrated_brief_payload():
    """A persisted-shape design brief that's already been migrated.

    No legacy keys; ``object_palette`` and ``per_image_objects`` set.
    """
    return {
        "global_instructions": "Add evergreens",
        "object_palette": [
            {
                "id": "00000000-0000-0000-0000-000000000001",
                "name": "Sequoia",
                "description": "Sequoiadendron giganteum",
                "category": "tree",
                "default_quantity": 2,
                "size": "20 ft tall",
                "placement": "north fence",
                "visual_notes": "tall, conical",
            }
        ],
        "placement_guide": {"back_row": "Tall conifers"},
        "preserve_elements": ["patio"],
        "per_image_objects": {},
        # Issue 016: a fully-modern brief carries the canonical sections
        # dict so the lazy backfill short-circuits. Without this, the
        # backfill would re-derive sections on every read and trigger a
        # spurious writeback.
        "sections": {"edit_task": "Add evergreens"},
    }


def test_get_project_migrates_legacy_plant_palette_and_writes_back(client, mock_staging_deps):
    """A project persisted with the old plant_palette shape is auto-migrated
    on read AND the migrated dict is written back so the next read is a no-op."""
    mock_container = mock_staging_deps["container"]
    mock_container.read_item.return_value = {
        "id": "proj-legacy",
        "name": "Legacy",
        "prompt": "Test",
        "status": "completed",
        "rooms": [],
        "settings": {"variations_per_room": 5, "model": "gpt-image-2", "quality": "high", "size": "auto"},
        "design_brief": _legacy_brief_payload(),
    }
    mock_container.replace_item.side_effect = lambda item, body: body

    response = client.get("/api/v1/staging/projects/proj-legacy")
    assert response.status_code == 200
    body = response.json()["project"]

    # Response carries object_palette, NOT plant_palette.
    brief = body["design_brief"]
    assert "object_palette" in brief
    assert "plant_palette" not in brief
    assert len(brief["object_palette"]) == 1
    obj = brief["object_palette"][0]
    assert obj["name"] == "Sequoia"
    assert obj["category"] == "tree"
    assert obj["default_quantity"] == 2
    assert obj["description"] == "Sequoiadendron giganteum"
    # Migration assigns a UUID id, regardless of what was persisted before.
    assert isinstance(obj["id"], str) and len(obj["id"]) > 0

    # And the migrated doc was persisted (writeback) — this is what makes the
    # next read a no-op rather than re-running the migration.
    assert mock_container.replace_item.call_count >= 1
    last_call = mock_container.replace_item.call_args_list[-1]
    persisted_body = last_call.kwargs.get("body") or last_call.args[1]
    assert "object_palette" in persisted_body["design_brief"]
    assert "plant_palette" not in persisted_body["design_brief"]


def test_get_project_no_writeback_when_brief_already_migrated(client, mock_staging_deps):
    """Already-migrated briefs MUST NOT trigger an opportunistic writeback —
    avoids needless Cosmos writes on every read."""
    mock_container = mock_staging_deps["container"]
    mock_container.read_item.return_value = {
        "id": "proj-already-migrated",
        "name": "Modern",
        "prompt": "Test",
        "status": "completed",
        "rooms": [],
        "settings": {"variations_per_room": 5, "model": "gpt-image-2", "quality": "high", "size": "auto"},
        "design_brief": _migrated_brief_payload(),
    }

    response = client.get("/api/v1/staging/projects/proj-already-migrated")
    assert response.status_code == 200
    body = response.json()["project"]
    assert "object_palette" in body["design_brief"]

    # No-op writeback — no replace_item calls (reconcile_project also returned
    # False because rooms=[] and the project is already terminal).
    assert mock_container.replace_item.call_count == 0


def test_get_project_no_design_brief_does_not_crash(client, mock_staging_deps):
    """Projects without a design_brief (e.g. uploading state) MUST NOT crash
    or trigger a writeback — the migration helper must short-circuit."""
    mock_container = mock_staging_deps["container"]
    mock_container.read_item.return_value = {
        "id": "proj-no-brief",
        "name": "Brand new",
        "prompt": "Test",
        "status": "uploading",
        "rooms": [],
        "settings": {"variations_per_room": 5, "model": "gpt-image-2", "quality": "high", "size": "auto"},
    }

    response = client.get("/api/v1/staging/projects/proj-no-brief")
    assert response.status_code == 200
    assert mock_container.replace_item.call_count == 0


def test_list_projects_migrates_legacy_plant_palette_and_writes_back(client, mock_staging_deps):
    """list_projects also runs the migration so legacy keys can't leak via
    list endpoints (defense-in-depth — issue 001 explicitly mentions
    surfacing only the new shape)."""
    mock_container = mock_staging_deps["container"]
    legacy_doc = {
        "id": "proj-legacy",
        "name": "Legacy",
        "prompt": "Test",
        "status": "completed",
        "rooms": [],
        "settings": {"variations_per_room": 5, "model": "gpt-image-2", "quality": "high", "size": "auto"},
        "design_brief": _legacy_brief_payload(),
    }

    def mock_query_items(query, parameters=None, enable_cross_partition_query=None):
        if "COUNT" in (query or "").upper():
            return iter([1])
        return iter([legacy_doc])

    mock_container.query_items = mock_query_items
    mock_container.replace_item.side_effect = lambda item, body: body

    response = client.get("/api/v1/staging/projects?limit=10")
    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 1
    project = payload["projects"][0]

    brief = project["design_brief"]
    assert "object_palette" in brief
    assert "plant_palette" not in brief

    # Writeback fired so the next list read is a no-op for this project.
    assert mock_container.replace_item.call_count >= 1


def test_put_brief_round_trip_preserves_typed_per_image_objects(client, mock_staging_deps):
    """Issue 003 of the per-image-object-quantities-design PRD: a brief with
    typed `per_image_objects` overrides survives a PUT round trip with all
    fields intact (object_id, quantity, placement, enabled).
    """
    mock_container = mock_staging_deps["container"]

    project_id = "proj-pio"
    obj_id = "obj-lavender"

    initial = {
        "id": project_id,
        "name": "Per-Image Test",
        "prompt": "",
        "status": "uploading",
        "rooms": [],
        "settings": {"variations_per_room": 1, "model": "gpt-image-2", "quality": "high", "size": "auto"},
        "design_brief": None,
    }
    mock_container.read_item.return_value = initial

    # Capture the payload that would be persisted.
    persisted = {}

    def fake_replace_item(item, body):
        persisted["body"] = body
        return body

    mock_container.replace_item.side_effect = fake_replace_item

    brief_payload = {
        "global_instructions": "Lush greenery",
        "object_palette": [
            {
                "id": obj_id,
                "name": "Lavender",
                "description": "Lavandula",
                "category": "plant",
                "default_quantity": 3,
                "size": "2 ft",
                "placement": "front row",
                "visual_notes": None,
            }
        ],
        "placement_guide": {"back_row": "Tall grasses"},
        "preserve_elements": [],
        "per_image_notes": {"room-1": "Heavy reds"},
        "per_image_objects": {
            "room-1": [
                {
                    "object_id": obj_id,
                    "quantity": 7,
                    "placement": "back row",
                    "enabled": True,
                }
            ],
            "room-2": [
                {
                    "object_id": obj_id,
                    "quantity": 0,
                    "placement": None,
                    "enabled": False,
                }
            ],
        },
        "settings": {"variations_per_room": 1, "model": "gpt-image-2", "quality": "high", "size": "auto"},
    }

    response = client.put(f"/api/v1/staging/projects/{project_id}/brief", json=brief_payload)
    assert response.status_code == 200, response.text

    returned = response.json()["brief"]
    # Round-trip: full structure preserved.
    assert returned["per_image_objects"]["room-1"] == [
        {"object_id": obj_id, "quantity": 7, "placement": "back row", "enabled": True}
    ]
    assert returned["per_image_objects"]["room-2"] == [
        {"object_id": obj_id, "quantity": 0, "placement": None, "enabled": False}
    ]
    assert returned["per_image_notes"] == {"room-1": "Heavy reds"}

    # Persisted Cosmos doc carries the same typed structure (not a free-form
    # `Dict[str, Any]` blob).
    assert persisted["body"]["design_brief"]["per_image_objects"]["room-1"][0]["quantity"] == 7
    assert persisted["body"]["design_brief"]["per_image_objects"]["room-2"][0]["enabled"] is False


def test_put_brief_normalises_placement_whitespace(client, mock_staging_deps):
    """ImageObjectOverride's placement validator (mode='before') strips
    whitespace and turns empty/whitespace-only strings into None. This is
    enforced at the model boundary; assert it actually fires through the
    HTTP layer too.
    """
    mock_container = mock_staging_deps["container"]
    mock_container.read_item.return_value = {
        "id": "proj-ws",
        "name": "WS",
        "prompt": "",
        "status": "uploading",
        "rooms": [],
        "settings": {"variations_per_room": 1, "model": "gpt-image-2", "quality": "high", "size": "auto"},
        "design_brief": None,
    }
    mock_container.replace_item.side_effect = lambda item, body: body

    obj_id = "obj-1"
    brief_payload = {
        "global_instructions": "X",
        "object_palette": [
            {
                "id": obj_id,
                "name": "Lavender",
                "description": "",
                "category": "plant",
                "default_quantity": 3,
                "size": "2 ft",
                "placement": "front",
                "visual_notes": None,
            }
        ],
        "placement_guide": {"back_row": "ZZ"},
        "preserve_elements": [],
        "per_image_notes": {},
        "per_image_objects": {
            "room-1": [
                {"object_id": obj_id, "quantity": 5, "placement": "   ", "enabled": True},
                {"object_id": obj_id + "-dummy-ignored", "quantity": 1, "placement": "  back  row  ", "enabled": True},
            ]
        },
        "settings": {"variations_per_room": 1, "model": "gpt-image-2", "quality": "high", "size": "auto"},
    }

    response = client.put("/api/v1/staging/projects/proj-ws/brief", json=brief_payload)
    assert response.status_code == 200, response.text

    overrides = response.json()["brief"]["per_image_objects"]["room-1"]
    # whitespace-only → None
    assert overrides[0]["placement"] is None
    # interior whitespace preserved; edges stripped
    assert overrides[1]["placement"] == "back  row"


# ---------------------------------------------------------------------------
# POST /projects/{id}/brief — issue 004 of the per-image-object-quantities PRD.
# The endpoint accepts an optional ``previous_brief`` and returns a
# ``reconciliation_summary`` body field. The wizard's regenerate flow uses
# both fields together to carry forward per-image quantity overrides across
# regeneration.
# ---------------------------------------------------------------------------


def test_post_brief_returns_reconciliation_summary_zero_zero_when_no_previous_brief(
    client, mock_staging_deps
):
    """Smoke-check: the new ``reconciliation_summary`` field is always
    present in the response, even on first generation when no
    ``previous_brief`` is supplied. carried_forward / dropped both 0.
    """
    import json as _json
    from unittest.mock import AsyncMock, MagicMock as _MagicMock

    mock_container = mock_staging_deps["container"]
    mock_container.read_item.return_value = {
        "id": "proj-pb-1",
        "name": "X",
        "prompt": "",
        "status": "uploading",
        "rooms": [],
        "settings": {"variations_per_room": 1, "model": "gpt-image-2", "quality": "high", "size": "auto"},
        "design_brief": None,
        "analyses": [
            {"room_id": "room-1", "description": "back yard", "features": [], "zones": []}
        ],
    }
    mock_container.replace_item.side_effect = lambda item, body: body

    fake_llm_response = _MagicMock(
        choices=[_MagicMock(message=_MagicMock(content=_json.dumps({
            "global_instructions": "x",
            "object_palette": [
                {
                    "name": "Lavender",
                    "category": "plant",
                    "default_quantity": 3,
                    "size": "2 ft",
                    "placement": "front",
                    "visual_notes": None,
                    "description": None,
                }
            ],
            "placement_guide": {"back_row": "z"},
            "per_image_notes": {},
            "preserve_elements": [],
        })))]
    )
    fake_llm = AsyncMock()
    fake_llm.chat.completions.create.return_value = fake_llm_response

    with patch("backend.core.async_llm_client", fake_llm):
        response = client.post(
            "/api/v1/staging/projects/proj-pb-1/brief",
            json={"conversation_history": [{"role": "user", "content": "Add lavender"}]},
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert "brief" in body
    assert body["reconciliation_summary"] == {"carried_forward": 0, "dropped": 0}


def test_post_brief_reconciles_previous_brief_overrides_by_name(
    client, mock_staging_deps
):
    """When the request body carries a ``previous_brief``, surviving
    per-image overrides are carried forward by case-insensitive,
    whitespace-trimmed name match against the new palette and the
    ``reconciliation_summary`` reports counts.
    """
    import json as _json
    from unittest.mock import AsyncMock, MagicMock as _MagicMock

    mock_container = mock_staging_deps["container"]
    mock_container.read_item.return_value = {
        "id": "proj-pb-2",
        "name": "X",
        "prompt": "",
        "status": "uploading",
        "rooms": [],
        "settings": {"variations_per_room": 1, "model": "gpt-image-2", "quality": "high", "size": "auto"},
        "design_brief": None,
        "analyses": [
            {"room_id": "room-1", "description": "back yard", "features": [], "zones": []}
        ],
    }
    mock_container.replace_item.side_effect = lambda item, body: body

    # Previous brief: user manually set Lavender qty=8 in room-1 and
    # also has an orphan "Pine" override that won't survive (Pine isn't
    # in the regenerated palette).
    prev_lav_id = "prev-lav-uuid"
    prev_pine_id = "prev-pine-uuid"
    previous_brief = {
        "global_instructions": "x",
        "object_palette": [
            {"id": prev_lav_id, "name": "Lavender", "category": "plant", "default_quantity": 3, "size": "2 ft", "placement": "front", "visual_notes": None, "description": None},
            {"id": prev_pine_id, "name": "Pine", "category": "tree", "default_quantity": 2, "size": "8 ft", "placement": "back", "visual_notes": None, "description": None},
        ],
        "placement_guide": {"back_row": "z"},
        "per_image_notes": {},
        "preserve_elements": [],
        "per_image_objects": {
            "room-1": [
                {"object_id": prev_lav_id, "quantity": 8, "placement": None, "enabled": True},
                {"object_id": prev_pine_id, "quantity": 5, "placement": None, "enabled": True},
            ]
        },
        "settings": {"variations_per_room": 1, "model": "gpt-image-2", "quality": "high", "size": "auto"},
    }

    # New LLM response: Lavender survives (different UUID, same name),
    # Pine is gone.
    fake_llm_response = _MagicMock(
        choices=[_MagicMock(message=_MagicMock(content=_json.dumps({
            "global_instructions": "x",
            "object_palette": [
                {
                    "name": "lavender",  # different case, same normalized name.
                    "category": "plant",
                    "default_quantity": 3,
                    "size": "2 ft",
                    "placement": "front",
                    "visual_notes": None,
                    "description": None,
                }
            ],
            "placement_guide": {"back_row": "z"},
            "per_image_notes": {},
            "preserve_elements": [],
        })))]
    )
    fake_llm = AsyncMock()
    fake_llm.chat.completions.create.return_value = fake_llm_response

    with patch("backend.core.async_llm_client", fake_llm):
        response = client.post(
            "/api/v1/staging/projects/proj-pb-2/brief",
            json={
                "conversation_history": [{"role": "user", "content": "regenerate"}],
                "previous_brief": previous_brief,
            },
        )

    assert response.status_code == 200, response.text
    body = response.json()
    # Reconciliation surfaces: 1 carried (Lavender), 1 dropped (Pine).
    assert body["reconciliation_summary"] == {"carried_forward": 1, "dropped": 1}
    # The carried-forward override now points at the NEW palette UUID.
    new_lavender_id = body["brief"]["object_palette"][0]["id"]
    assert new_lavender_id != prev_lav_id
    overrides = body["brief"]["per_image_objects"]["room-1"]
    assert len(overrides) == 1
    assert overrides[0]["object_id"] == new_lavender_id
    assert overrides[0]["quantity"] == 8


# ============================================================================
# Issue 003 of single-variation-regeneration PRD: prompt_diversity threading.
#
# When the user clicks "Try Something New" (strategy=fresh) on a variation
# that has a previously rejected ``adapted_prompt`` in ``generation_metadata``,
# the prior prompt MUST flow through to the LLM call site as negative
# context. These tests drive the regen endpoint and assert the LLM mock
# receives ``messages[0].content`` containing the prior prompt — the
# integration acceptance criterion.
#
# Test pattern: the existing ``mock_staging_deps`` fixture replaces the
# pipeline with a MagicMock. We configure that mock's blob/analyzer to
# return real data, and replace ``adapt_prompt`` with a thin wrapper that
# delegates to a REAL ``StagingPipeline`` instance whose ``async_llm_client``
# is a captured ``AsyncMock``. The brief path uses ``BriefGeneratorService``
# constructed inside the endpoint with ``backend.core.async_llm_client`` —
# patching that module attribute routes the brief LLM call to the captured
# mock. ``process_single_variation`` is stubbed to an empty async generator
# so the test focuses on the prompt-generation seam.
# ============================================================================

import json as _json_for_regen_tests
from unittest.mock import AsyncMock, MagicMock as _MM, patch as _patch


_PRIOR_PROMPT = "MAGENTA-AND-CHROME MAXIMALIST AESTHETIC FROM REJECTED VARIATION"


def _project_fresh_regen_payload(*, with_brief: bool):
    """Project containing one room with one previously-rejected variation.

    The variation's ``generation_metadata.adapted_prompt`` is the prompt
    the user just rejected. ``with_brief=True`` exercises the
    BriefGeneratorService path; False exercises the no-brief
    (``adapt_prompt``) path.
    """
    project = {
        "id": "proj-regen-fresh",
        "name": "Fresh Regen Test",
        "prompt": "USER_INTENT_SENTINEL — Modern minimalist",
        "status": "completed",
        "rooms": [{
            "id": "room-1",
            "label": "Living Room",
            "original_image_url": "https://acct.blob.core.windows.net/images/staging/proj/originals/photo.png",
            "status": "completed",
            "analysis": {"description": "A sunlit living room with hardwood floors", "features": ["floor", "window"]},
            "variations": [{
                "id": "var-1",
                "status": "completed",
                "image_url": "https://acct.blob.core.windows.net/images/staging/proj/variations/room-1/img.png",
                "generation_metadata": {
                    "model": "gpt-image-2",
                    "adapted_prompt": _PRIOR_PROMPT,
                    "generation_time_ms": 5000,
                },
            }],
        }],
        "settings": {"variations_per_room": 1, "model": "gpt-image-2", "quality": "high", "size": "auto"},
        "analyses": [{"room_id": "room-1", "description": "A sunlit living room with hardwood floors", "features": ["floor", "window"]}],
    }
    if with_brief:
        project["design_brief"] = {
            "global_instructions": "BRIEF_INTENT_SENTINEL — warm scandinavian palette",
            "object_palette": [
                {"name": "Sofa", "category": "furniture", "default_quantity": 1, "size": "3-seater", "placement": "facing window"},
            ],
            "placement_guide": {"back_row": "abstract art"},
            "per_image_notes": {},
            "preserve_elements": ["hardwood floor"],
            "per_image_objects": {},
        }
    return project


def _make_captured_llm():
    """Build an AsyncMock LLM client that records every chat completion."""
    llm = AsyncMock()
    llm.chat.completions.create.return_value = _MM(
        choices=[_MM(message=_MM(content=_json_for_regen_tests.dumps({"prompts": ["new direction"]})))]
    )
    return llm


def test_fresh_regen_threads_prior_prompt_to_brief_llm_call(client, mock_staging_deps):
    """Brief path: the rejected ``adapted_prompt`` from
    ``generation_metadata`` must appear in ``messages[0].content`` sent
    to the LLM by ``BriefGeneratorService.brief_to_prompts``.

    Patches ``backend.core.async_llm_client`` so the BriefGeneratorService
    constructed inside the endpoint routes through our captured mock.
    """
    mock_container = mock_staging_deps["container"]
    project_data = _project_fresh_regen_payload(with_brief=True)
    mock_container.read_item.return_value = project_data
    mock_container.replace_item.side_effect = lambda item, body: body

    captured_llm = _make_captured_llm()

    # ``process_single_variation`` is patched at the class level so the
    # request returns immediately after prompt-generation — we don't want
    # it to run real image-gen.
    async def _empty_psv(self, *_args, **_kwargs):
        if False:
            yield  # pragma: no cover

    from backend.core.staging_pipeline import StagingPipeline

    with _patch("backend.core.async_llm_client", captured_llm), \
         _patch.object(StagingPipeline, "process_single_variation", _empty_psv):
        with client.stream(
            "POST",
            "/api/v1/staging/projects/proj-regen-fresh/rooms/room-1/variations/var-1/regenerate?strategy=fresh",
        ) as response:
            assert response.status_code == 200, response.text
            for _ in response.iter_bytes():
                pass

    assert captured_llm.chat.completions.create.called, (
        "Brief path must reach the LLM for prompt generation"
    )
    sent = captured_llm.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    assert _PRIOR_PROMPT in sent, (
        f"Rejected prior prompt missing from LLM call site. Got: {sent[:400]!r}"
    )
    assert "REJECTED_PRIOR_DIRECTION" in sent
    assert "BRIEF_INTENT_SENTINEL" in sent  # brief intent survives steering


def test_fresh_regen_threads_prior_prompt_to_no_brief_llm_call(client, mock_staging_deps):
    """No-brief path: the rejected ``adapted_prompt`` must appear in the
    LLM's system message via ``StagingPipeline.adapt_prompt``.

    Patches ``backend.core.async_llm_client`` so the pipeline's LLM
    client is the captured mock; patches
    ``AzureBlobStorageService.get_asset_content`` to serve a real bytes
    tuple so ``base64.b64encode`` doesn't crash on the auto-MagicMock
    chain; replaces ``StagingPipeline.analyze_room`` with a stub so we
    don't need to wire the analyzer.
    """
    mock_container = mock_staging_deps["container"]
    project_data = _project_fresh_regen_payload(with_brief=False)
    mock_container.read_item.return_value = project_data
    mock_container.replace_item.side_effect = lambda item, body: body

    captured_llm = _make_captured_llm()

    async def _fake_analyze_room(self, image_b64):
        return {"description": "ROOM_ANALYSIS_SENTINEL — sunlit living room", "features": ["floor", "window"]}

    async def _empty_psv(self, *_args, **_kwargs):
        if False:
            yield  # pragma: no cover

    from backend.core.staging_pipeline import StagingPipeline
    from backend.core.azure_storage import AzureBlobStorageService

    with _patch("backend.core.async_llm_client", captured_llm), \
         _patch.object(StagingPipeline, "process_single_variation", _empty_psv), \
         _patch.object(StagingPipeline, "analyze_room", _fake_analyze_room), \
         _patch.object(AzureBlobStorageService, "get_asset_content",
                       return_value=(b"FAKE_IMG", "image/png")):
        with client.stream(
            "POST",
            "/api/v1/staging/projects/proj-regen-fresh/rooms/room-1/variations/var-1/regenerate?strategy=fresh",
        ) as response:
            assert response.status_code == 200, response.text
            for _ in response.iter_bytes():
                pass

    assert captured_llm.chat.completions.create.called, (
        "No-brief path must reach the LLM for prompt adaptation"
    )
    sent = captured_llm.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    assert _PRIOR_PROMPT in sent, (
        f"Rejected prior prompt missing from no-brief LLM call site. Got: {sent[:400]!r}"
    )
    assert "REJECTED_PRIOR_DIRECTION" in sent
    # User intent survives the steering wrapper.
    assert "USER_INTENT_SENTINEL" in sent


def test_fresh_regen_no_prior_metadata_does_not_inject_steering(client, mock_staging_deps):
    """First-ever generation defense: when ``generation_metadata`` lacks an
    ``adapted_prompt`` (e.g. an old variation persisted before issue 002),
    the LLM call must NOT contain the steering block — defaulting to
    ``rejected_prompt=None`` is the no-op contract."""
    mock_container = mock_staging_deps["container"]
    project_data = _project_fresh_regen_payload(with_brief=False)
    project_data["rooms"][0]["variations"][0]["generation_metadata"] = {"model": "gpt-image-2"}
    mock_container.read_item.return_value = project_data
    mock_container.replace_item.side_effect = lambda item, body: body

    captured_llm = _make_captured_llm()

    async def _fake_analyze_room(self, image_b64):
        return {"description": "A sunlit living room", "features": []}

    async def _empty_psv(self, *_args, **_kwargs):
        if False:
            yield  # pragma: no cover

    from backend.core.staging_pipeline import StagingPipeline
    from backend.core.azure_storage import AzureBlobStorageService

    with _patch("backend.core.async_llm_client", captured_llm), \
         _patch.object(StagingPipeline, "process_single_variation", _empty_psv), \
         _patch.object(StagingPipeline, "analyze_room", _fake_analyze_room), \
         _patch.object(AzureBlobStorageService, "get_asset_content",
                       return_value=(b"FAKE_IMG", "image/png")):
        with client.stream(
            "POST",
            "/api/v1/staging/projects/proj-regen-fresh/rooms/room-1/variations/var-1/regenerate?strategy=fresh",
        ) as response:
            assert response.status_code == 200, response.text
            for _ in response.iter_bytes():
                pass

    sent = captured_llm.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    assert "REJECTED_PRIOR_DIRECTION" not in sent
    assert "REGENERATION STEERING" not in sent


# ============================================================================
# Issue 004 — Retry-to-fresh fallback signaling
# ----------------------------------------------------------------------------
# When a user picks ``Retry Same Prompt`` on a variation that has no prior
# ``adapted_prompt`` recorded (legacy variation, or one that errored before
# issue 001 closed the metadata-persistence gap), the backend silently falls
# back to fresh prompt generation. This slice surfaces the fallback as a
# dedicated ``variation_fallback`` SSE event so the frontend can toast the
# user "no previous prompt found — generating a fresh take instead."
#
# Backend contract:
#  - ``strategy=retry`` AND ``generation_metadata.adapted_prompt`` is missing
#    → emit ``variation_fallback`` BEFORE the fresh-fallback prompt
#    generation work begins. Payload:
#       {"type": "variation_fallback",
#        "room_id": "...", "variation_id": "...",
#        "reason": "no_prior_prompt"}
#  - ``strategy=retry`` AND prior prompt exists → no fallback event.
#  - ``strategy=fresh`` (user explicit choice) → no fallback event.
# ============================================================================


def _parse_sse_stream(byte_chunks):
    """Parse SSE-formatted bytes into a list of ``{type, ...data}`` dicts.

    Joins the chunks into a single buffer and walks the standard
    ``event:`` / ``data:`` framing emitted by ``_sse_event`` in the
    endpoint. Used by the issue 004 fallback tests below.
    """
    buf = b"".join(byte_chunks).decode("utf-8")
    events = []
    current_event = None
    current_data = None
    for raw in buf.split("\n"):
        line = raw.rstrip("\r")
        if line.startswith("event: "):
            current_event = line[7:].strip()
        elif line.startswith("data: "):
            current_data = line[6:]
        elif line == "":
            if current_event and current_data is not None:
                try:
                    parsed = _json_for_regen_tests.loads(current_data)
                except _json_for_regen_tests.JSONDecodeError:
                    parsed = {"raw": current_data}
                if not isinstance(parsed, dict):
                    parsed = {"value": parsed}
                events.append({"type": current_event, **parsed})
            current_event = None
            current_data = None
    return events


def _project_for_fallback_test(*, with_prior_prompt: bool, with_brief: bool = False):
    """Project containing one variation whose ``generation_metadata`` may or
    may not include ``adapted_prompt``. Used by the fallback-event tests."""
    project = {
        "id": "proj-fallback",
        "name": "Fallback Test",
        "prompt": "Modern minimalist living room",
        "status": "completed",
        "rooms": [{
            "id": "room-1",
            "label": "Living Room",
            "original_image_url": "https://acct.blob.core.windows.net/images/staging/proj/originals/photo.png",
            "status": "completed",
            "analysis": {"description": "A sunlit living room", "features": ["floor"]},
            "variations": [{
                "id": "var-1",
                "status": "completed",
                "image_url": "https://acct.blob.core.windows.net/images/staging/proj/variations/room-1/img.png",
                "generation_metadata": (
                    {"model": "gpt-image-2", "adapted_prompt": "PRIOR PROMPT TEXT", "generation_time_ms": 5000}
                    if with_prior_prompt
                    else {"model": "gpt-image-2", "generation_time_ms": 5000}
                ),
            }],
        }],
        "settings": {"variations_per_room": 1, "model": "gpt-image-2", "quality": "high", "size": "auto"},
        "analyses": [{"room_id": "room-1", "description": "A sunlit living room", "features": ["floor"]}],
    }
    if with_brief:
        project["design_brief"] = {
            "global_instructions": "warm scandinavian palette",
            "object_palette": [],
            "placement_guide": {},
            "per_image_notes": {},
            "preserve_elements": [],
            "per_image_objects": {},
        }
    return project


def test_retry_no_prior_prompt_emits_variation_fallback_then_continues_normally(
    client, mock_staging_deps,
):
    """Retry against a variation with no ``adapted_prompt`` must emit a
    ``variation_fallback`` SSE event before doing fresh-fallback work, and
    must continue to a terminal ``project_completed`` event (no early
    termination)."""
    mock_container = mock_staging_deps["container"]
    project_data = _project_for_fallback_test(with_prior_prompt=False, with_brief=False)
    mock_container.read_item.return_value = project_data
    mock_container.replace_item.side_effect = lambda item, body: body

    captured_llm = _make_captured_llm()

    async def _fake_analyze_room(self, image_b64):
        return {"description": "A sunlit living room", "features": []}

    async def _empty_psv(self, *_args, **_kwargs):
        if False:
            yield  # pragma: no cover

    from backend.core.staging_pipeline import StagingPipeline
    from backend.core.azure_storage import AzureBlobStorageService

    chunks: list[bytes] = []
    with _patch("backend.core.async_llm_client", captured_llm), \
         _patch.object(StagingPipeline, "process_single_variation", _empty_psv), \
         _patch.object(StagingPipeline, "analyze_room", _fake_analyze_room), \
         _patch.object(AzureBlobStorageService, "get_asset_content",
                       return_value=(b"FAKE_IMG", "image/png")):
        with client.stream(
            "POST",
            "/api/v1/staging/projects/proj-fallback/rooms/room-1/variations/var-1/regenerate?strategy=retry",
        ) as response:
            assert response.status_code == 200, response.text
            for chunk in response.iter_bytes():
                chunks.append(chunk)

    events = _parse_sse_stream(chunks)
    types = [e["type"] for e in events]

    fallback_events = [e for e in events if e["type"] == "variation_fallback"]
    assert len(fallback_events) == 1, (
        f"Expected exactly one variation_fallback event; got types={types}"
    )
    fb = fallback_events[0]
    assert fb["room_id"] == "room-1"
    assert fb["variation_id"] == "var-1"
    assert fb["reason"] == "no_prior_prompt"

    assert "project_completed" in types, (
        f"Stream must reach terminal project_completed; got types={types}"
    )

    assert types.index("variation_fallback") < types.index("project_completed"), (
        "variation_fallback must precede the terminal project_completed event"
    )


def test_retry_with_prior_prompt_does_not_emit_variation_fallback(
    client, mock_staging_deps,
):
    """Retry against a variation whose ``generation_metadata.adapted_prompt``
    is present must NOT emit ``variation_fallback`` — the prior prompt is
    used as-is and there is no fallback."""
    mock_container = mock_staging_deps["container"]
    project_data = _project_for_fallback_test(with_prior_prompt=True, with_brief=False)
    mock_container.read_item.return_value = project_data
    mock_container.replace_item.side_effect = lambda item, body: body

    async def _empty_psv(self, *_args, **_kwargs):
        if False:
            yield  # pragma: no cover

    from backend.core.staging_pipeline import StagingPipeline

    chunks: list[bytes] = []
    with _patch.object(StagingPipeline, "process_single_variation", _empty_psv):
        with client.stream(
            "POST",
            "/api/v1/staging/projects/proj-fallback/rooms/room-1/variations/var-1/regenerate?strategy=retry",
        ) as response:
            assert response.status_code == 200, response.text
            for chunk in response.iter_bytes():
                chunks.append(chunk)

    events = _parse_sse_stream(chunks)
    types = [e["type"] for e in events]
    assert "variation_fallback" not in types, (
        f"Retry with valid prior prompt must NOT emit fallback event; "
        f"got types={types}"
    )


def test_fresh_strategy_does_not_emit_variation_fallback(client, mock_staging_deps):
    """``strategy=fresh`` is the user's explicit choice — it must not be
    confused with a retry-fallback. Even when there is no prior prompt,
    fresh must emit no fallback event."""
    mock_container = mock_staging_deps["container"]
    project_data = _project_for_fallback_test(with_prior_prompt=False, with_brief=False)
    mock_container.read_item.return_value = project_data
    mock_container.replace_item.side_effect = lambda item, body: body

    captured_llm = _make_captured_llm()

    async def _fake_analyze_room(self, image_b64):
        return {"description": "A sunlit living room", "features": []}

    async def _empty_psv(self, *_args, **_kwargs):
        if False:
            yield  # pragma: no cover

    from backend.core.staging_pipeline import StagingPipeline
    from backend.core.azure_storage import AzureBlobStorageService

    chunks: list[bytes] = []
    with _patch("backend.core.async_llm_client", captured_llm), \
         _patch.object(StagingPipeline, "process_single_variation", _empty_psv), \
         _patch.object(StagingPipeline, "analyze_room", _fake_analyze_room), \
         _patch.object(AzureBlobStorageService, "get_asset_content",
                       return_value=(b"FAKE_IMG", "image/png")):
        with client.stream(
            "POST",
            "/api/v1/staging/projects/proj-fallback/rooms/room-1/variations/var-1/regenerate?strategy=fresh",
        ) as response:
            assert response.status_code == 200, response.text
            for chunk in response.iter_bytes():
                chunks.append(chunk)

    events = _parse_sse_stream(chunks)
    types = [e["type"] for e in events]
    assert "variation_fallback" not in types, (
        f"strategy=fresh must NOT emit fallback event; got types={types}"
    )


# ============================================================================
# Issue 008 — Structured logging at four lifecycle events
# ----------------------------------------------------------------------------
# The regen endpoint must emit operator-facing structured log lines at the
# four key lifecycle events of a single-variation regen, so log analytics can
# answer questions about regen usage rates, success rates, fallback frequency,
# and elapsed time without spelunking through unstructured logs.
#
# Contract (from PRD § Implementation Decisions → Backend, structured-logging
# bullet, and issue 008):
#
#   1. ``staging.variation_regen.started`` — after concurrency / 404 / 400 /
#      409 checks pass, before the pipeline call.
#   2. ``staging.variation_regen.completed`` — on terminal success.
#   3. ``staging.variation_regen.failed`` — on terminal failure.
#   4. ``staging.variation_regen.fallback_to_fresh`` — alongside the
#      ``variation_fallback`` SSE event when retry has no prior prompt.
#
# Each line includes structured fields ``project_id``, ``room_id``,
# ``variation_id``, ``strategy``, ``effective_strategy``. The ``completed``
# and ``failed`` lines additionally include ``elapsed_ms`` (always present)
# and ``tokens_used`` (where available — None for retry-no-LLM-call flows).
# No PII or secrets in the payload.
#
# Fields are duplicated on the LogRecord via ``extra=`` AND in the
# human-readable message via ``key=value`` pairs (mirroring the
# ``backend.core.retry`` pattern in ``test_call_with_retry.py``), so log
# aggregators consuming either form can pick them up. The tests below assert
# on the ``extra=`` projection (``record.event``, ``record.project_id``, …)
# because that's the structured form log analytics actually queries.
# ============================================================================

import logging as _logging_for_regen_logs


_REGEN_LOGGER_NAME = "backend.api.endpoints.staging"


def _regen_log_records(caplog):
    """Filter caplog records to ones that carry the structured ``event`` field
    set to one of the four ``staging.variation_regen.*`` lifecycle events.

    Other unrelated INFO logs from the endpoint (e.g., reconcile warnings,
    blob-cleanup messages) are filtered out by the prefix match so the tests
    can assert on the regen-specific ordering.
    """
    return [
        r for r in caplog.records
        if isinstance(getattr(r, "event", None), str)
        and r.event.startswith("staging.variation_regen.")
    ]


def _psv_yielding(event_dict):
    """Build a class-level ``process_single_variation`` replacement that
    yields exactly one event dict, mirroring the real pipeline's contract
    (one terminal event per call). The ``self`` arg matches the bound-method
    signature so ``patch.object(StagingPipeline, "process_single_variation",
    ...)`` swaps it in cleanly."""
    async def _psv(self, *_args, **_kwargs):
        yield event_dict
    return _psv


def test_regen_logs_started_and_completed_on_happy_retry(
    client, mock_staging_deps, caplog,
):
    """Issue 008 AC: a happy-path retry emits exactly ``started`` and
    ``completed`` log lines (no ``fallback_to_fresh``, no ``failed``).

    With prior ``adapted_prompt`` available, retry uses it directly and skips
    the LLM. ``effective_strategy`` matches the requested ``strategy=retry``.
    ``tokens_used`` reflects the image-gen call's reported usage (the
    retry-no-LLM-call flow only saves on the *prompt*-generation LLM call;
    image generation still consumes tokens).
    """
    caplog.set_level(_logging_for_regen_logs.INFO, logger=_REGEN_LOGGER_NAME)

    mock_container = mock_staging_deps["container"]
    project_data = _project_for_fallback_test(with_prior_prompt=True, with_brief=False)
    mock_container.read_item.return_value = project_data
    mock_container.replace_item.side_effect = lambda item, body: body

    completed_event = {
        "type": "variation_completed",
        "room_id": "room-1",
        "variation_index": 0,
        "image_url": "https://acct.blob.core.windows.net/images/staging/proj/variations/room-1/new.png",
        "error": None,
        "elapsed_ms": 4321,
        "tokens_used": 567,
        "model": "gpt-image-2",
        "adapted_prompt": "PRIOR PROMPT TEXT",
    }

    from backend.core.staging_pipeline import StagingPipeline

    with _patch.object(
        StagingPipeline, "process_single_variation", _psv_yielding(completed_event),
    ):
        with client.stream(
            "POST",
            "/api/v1/staging/projects/proj-fallback/rooms/room-1/variations/var-1/regenerate?strategy=retry",
        ) as response:
            assert response.status_code == 200, response.text
            for _ in response.iter_bytes():
                pass

    records = _regen_log_records(caplog)
    events = [r.event for r in records]
    assert events == [
        "staging.variation_regen.started",
        "staging.variation_regen.completed",
    ], (
        f"Happy retry must emit exactly started+completed; got {events!r}"
    )

    started, completed = records
    assert started.project_id == "proj-fallback"
    assert started.room_id == "room-1"
    assert started.variation_id == "var-1"
    assert started.strategy == "retry"
    assert started.effective_strategy == "retry"

    assert completed.project_id == "proj-fallback"
    assert completed.room_id == "room-1"
    assert completed.variation_id == "var-1"
    assert completed.strategy == "retry"
    assert completed.effective_strategy == "retry"
    # ``elapsed_ms`` is operator-facing wall-clock from regen acceptance to
    # terminal-event observation — NOT the pipeline's image-gen-only
    # ``elapsed_ms`` (4321 in the mocked event above). We assert the field
    # is present, integral, and non-negative; we do NOT pin a specific
    # millisecond value because the wall-clock delta in tests is dominated
    # by mock overhead and is intentionally non-deterministic.
    assert isinstance(completed.elapsed_ms, int)
    assert completed.elapsed_ms >= 0
    assert completed.tokens_used == 567


def test_regen_logs_started_and_completed_on_happy_fresh(
    client, mock_staging_deps, caplog,
):
    """Issue 008 AC: a happy-path fresh emits exactly ``started`` and
    ``completed`` (no fallback, no failed). ``strategy`` and
    ``effective_strategy`` both equal ``fresh``.
    """
    caplog.set_level(_logging_for_regen_logs.INFO, logger=_REGEN_LOGGER_NAME)

    mock_container = mock_staging_deps["container"]
    project_data = _project_for_fallback_test(with_prior_prompt=True, with_brief=False)
    mock_container.read_item.return_value = project_data
    mock_container.replace_item.side_effect = lambda item, body: body

    captured_llm = _make_captured_llm()

    async def _fake_analyze_room(self, image_b64):
        return {"description": "A sunlit living room", "features": []}

    completed_event = {
        "type": "variation_completed",
        "room_id": "room-1",
        "variation_index": 0,
        "image_url": "https://acct.blob.core.windows.net/images/staging/proj/variations/room-1/new.png",
        "error": None,
        "elapsed_ms": 1500,
        "tokens_used": 1200,
        "model": "gpt-image-2",
        "adapted_prompt": "fresh take",
    }

    from backend.core.staging_pipeline import StagingPipeline
    from backend.core.azure_storage import AzureBlobStorageService

    with _patch("backend.core.async_llm_client", captured_llm), \
         _patch.object(StagingPipeline, "process_single_variation", _psv_yielding(completed_event)), \
         _patch.object(StagingPipeline, "analyze_room", _fake_analyze_room), \
         _patch.object(AzureBlobStorageService, "get_asset_content",
                       return_value=(b"FAKE_IMG", "image/png")):
        with client.stream(
            "POST",
            "/api/v1/staging/projects/proj-fallback/rooms/room-1/variations/var-1/regenerate?strategy=fresh",
        ) as response:
            assert response.status_code == 200, response.text
            for _ in response.iter_bytes():
                pass

    records = _regen_log_records(caplog)
    events = [r.event for r in records]
    assert events == [
        "staging.variation_regen.started",
        "staging.variation_regen.completed",
    ], (
        f"Happy fresh must emit exactly started+completed; got {events!r}"
    )

    started, completed = records
    assert started.strategy == "fresh"
    assert started.effective_strategy == "fresh"
    assert completed.strategy == "fresh"
    assert completed.effective_strategy == "fresh"
    assert isinstance(completed.elapsed_ms, int)
    assert completed.elapsed_ms >= 0
    assert completed.tokens_used == 1200


def test_regen_logs_started_fallback_to_fresh_completed_on_retry_no_prior(
    client, mock_staging_deps, caplog,
):
    """Issue 008 AC: retry-with-no-prior-prompt emits ``started``,
    ``fallback_to_fresh``, ``completed`` IN THAT ORDER.

    On this path the user requested ``retry`` but no prior ``adapted_prompt``
    is recorded, so the endpoint silently falls back to fresh prompt
    generation (issue 004) and emits the ``variation_fallback`` SSE event.
    The structured log line pairs with that SSE event.

    ``strategy`` reflects the *requested* strategy ("retry");
    ``effective_strategy`` reflects what was *actually* used ("fresh").
    """
    caplog.set_level(_logging_for_regen_logs.INFO, logger=_REGEN_LOGGER_NAME)

    mock_container = mock_staging_deps["container"]
    project_data = _project_for_fallback_test(with_prior_prompt=False, with_brief=False)
    mock_container.read_item.return_value = project_data
    mock_container.replace_item.side_effect = lambda item, body: body

    captured_llm = _make_captured_llm()

    async def _fake_analyze_room(self, image_b64):
        return {"description": "A sunlit living room", "features": []}

    completed_event = {
        "type": "variation_completed",
        "room_id": "room-1",
        "variation_index": 0,
        "image_url": "https://acct.blob.core.windows.net/images/staging/proj/variations/room-1/new.png",
        "error": None,
        "elapsed_ms": 2000,
        "tokens_used": 800,
        "model": "gpt-image-2",
        "adapted_prompt": "fresh fallback take",
    }

    from backend.core.staging_pipeline import StagingPipeline
    from backend.core.azure_storage import AzureBlobStorageService

    with _patch("backend.core.async_llm_client", captured_llm), \
         _patch.object(StagingPipeline, "process_single_variation", _psv_yielding(completed_event)), \
         _patch.object(StagingPipeline, "analyze_room", _fake_analyze_room), \
         _patch.object(AzureBlobStorageService, "get_asset_content",
                       return_value=(b"FAKE_IMG", "image/png")):
        with client.stream(
            "POST",
            "/api/v1/staging/projects/proj-fallback/rooms/room-1/variations/var-1/regenerate?strategy=retry",
        ) as response:
            assert response.status_code == 200, response.text
            for _ in response.iter_bytes():
                pass

    records = _regen_log_records(caplog)
    events = [r.event for r in records]
    assert events == [
        "staging.variation_regen.started",
        "staging.variation_regen.fallback_to_fresh",
        "staging.variation_regen.completed",
    ], (
        f"Retry with no prior prompt must emit started→fallback→completed "
        f"in order; got {events!r}"
    )

    started, fallback, completed = records
    # All three must agree on strategy="retry", effective_strategy="fresh".
    for r in (started, fallback, completed):
        assert r.project_id == "proj-fallback"
        assert r.room_id == "room-1"
        assert r.variation_id == "var-1"
        assert r.strategy == "retry"
        assert r.effective_strategy == "fresh"

    # Only completed carries elapsed_ms / tokens_used; started + fallback
    # do not (they're not "where applicable").
    assert getattr(started, "elapsed_ms", None) is None
    assert getattr(fallback, "elapsed_ms", None) is None
    assert isinstance(completed.elapsed_ms, int)
    assert completed.elapsed_ms >= 0
    assert completed.tokens_used == 800


def test_regen_logs_started_and_failed_on_failure(
    client, mock_staging_deps, caplog,
):
    """Issue 008 AC: a terminal failure path emits ``started`` and ``failed``
    (no ``completed``). The ``failed`` line still carries ``elapsed_ms``;
    ``tokens_used`` is ``None`` on a failed image-gen path (the pipeline
    surfaces it as None in the SSE event, see staging_pipeline.py).
    """
    caplog.set_level(_logging_for_regen_logs.INFO, logger=_REGEN_LOGGER_NAME)

    mock_container = mock_staging_deps["container"]
    project_data = _project_for_fallback_test(with_prior_prompt=True, with_brief=False)
    mock_container.read_item.return_value = project_data
    mock_container.replace_item.side_effect = lambda item, body: body

    failed_event = {
        "type": "variation_failed",
        "room_id": "room-1",
        "variation_index": 0,
        # On failure rollback, image_url is restored to the prior URL.
        "image_url": "https://acct.blob.core.windows.net/images/staging/proj/variations/room-1/img.png",
        "error": "Image generation failed: upstream 500",
        "elapsed_ms": 800,
        "tokens_used": None,
        "model": "gpt-image-2",
        "adapted_prompt": "PRIOR PROMPT TEXT",
    }

    from backend.core.staging_pipeline import StagingPipeline

    with _patch.object(
        StagingPipeline, "process_single_variation", _psv_yielding(failed_event),
    ):
        with client.stream(
            "POST",
            "/api/v1/staging/projects/proj-fallback/rooms/room-1/variations/var-1/regenerate?strategy=retry",
        ) as response:
            assert response.status_code == 200, response.text
            for _ in response.iter_bytes():
                pass

    records = _regen_log_records(caplog)
    events = [r.event for r in records]
    assert events == [
        "staging.variation_regen.started",
        "staging.variation_regen.failed",
    ], (
        f"Failure path must emit exactly started+failed (no completed); "
        f"got {events!r}"
    )

    started, failed = records
    assert started.strategy == "retry"
    assert started.effective_strategy == "retry"

    assert failed.project_id == "proj-fallback"
    assert failed.room_id == "room-1"
    assert failed.variation_id == "var-1"
    assert failed.strategy == "retry"
    assert failed.effective_strategy == "retry"
    assert isinstance(failed.elapsed_ms, int)
    assert failed.elapsed_ms >= 0
    # tokens_used is explicitly carried even when None — log analytics needs
    # to see "this was a no-LLM-call flow", not "this field was missing".
    assert hasattr(failed, "tokens_used"), (
        "failed log record must carry the tokens_used field even when None"
    )
    assert failed.tokens_used is None


def test_regen_logs_started_and_failed_when_pipeline_raises_unexpectedly(
    client, mock_staging_deps, caplog,
):
    """Issue 008 defense-in-depth (rubber-duck-driven): if the pipeline
    raises an unexpected exception INSTEAD of yielding a terminal SSE event,
    operator logs must still show ``started`` paired with ``failed``.

    Without the ``except Exception`` branch in ``event_stream``, a stray
    exception in ``process_single_variation`` (or in the prompt-generation
    helpers above it) would leave a stranded ``started`` line in the logs
    with no terminal partner — making it impossible for operators to
    distinguish "this regen is still in flight" from "this regen crashed".

    The synthesized ``failed`` line carries ``tokens_used=None`` (no image-
    gen ever produced a token tally) and an integral wall-clock
    ``elapsed_ms`` from regen acceptance to the moment the exception was
    caught.
    """
    caplog.set_level(_logging_for_regen_logs.INFO, logger=_REGEN_LOGGER_NAME)

    mock_container = mock_staging_deps["container"]
    project_data = _project_for_fallback_test(with_prior_prompt=True, with_brief=False)
    mock_container.read_item.return_value = project_data
    mock_container.replace_item.side_effect = lambda item, body: body

    async def _psv_raises(self, *_args, **_kwargs):
        raise RuntimeError("simulated upstream pipeline crash")
        # (unreachable; required to make this a generator function so
        # `async for` over it triggers the exception in event_stream.)
        yield  # pragma: no cover

    from backend.core.staging_pipeline import StagingPipeline

    with _patch.object(StagingPipeline, "process_single_variation", _psv_raises):
        # The exception is raised *inside* the streaming response body.
        # FastAPI/Starlette propagates streaming-body exceptions to the
        # client by closing the connection mid-stream; the test client
        # surfaces this either as the exception itself or as a truncated
        # body. We don't care which — the assertion target is the log,
        # not the HTTP wire format.
        try:
            with client.stream(
                "POST",
                "/api/v1/staging/projects/proj-fallback/rooms/room-1/variations/var-1/regenerate?strategy=retry",
            ) as response:
                assert response.status_code == 200, response.text
                for _ in response.iter_bytes():
                    pass
        except RuntimeError:
            pass

    records = _regen_log_records(caplog)
    events = [r.event for r in records]
    assert events == [
        "staging.variation_regen.started",
        "staging.variation_regen.failed",
    ], (
        f"Pipeline raise must still produce started+failed (NOT a "
        f"stranded started); got {events!r}"
    )

    started, failed = records
    assert started.strategy == "retry"
    assert started.effective_strategy == "retry"
    assert failed.strategy == "retry"
    assert failed.effective_strategy == "retry"
    assert isinstance(failed.elapsed_ms, int)
    assert failed.elapsed_ms >= 0
    # No image-gen ever ran — tokens_used must be None, but the field MUST
    # be present (so log analytics distinguishes "no token tally available"
    # from "field missing on a non-terminal log line").
    assert hasattr(failed, "tokens_used")
    assert failed.tokens_used is None


# ============================================================================
# Issue 001 (projects-page-improvements PRD) — endpoint-level finalizer
# regression tests. The unit tests in
# tests/test_project_status_calculator.py prove the calculator's correctness
# in isolation; the test below in test_staging_pipeline.py
# (TestProjectStatusDelegatesToCalculator) proves generate_project's early-
# out branch delegates to it. These two tests pin the OTHER two call sites
# — the regenerate_room and regenerate_variation finalizer blocks — to the
# calculator. Without them the inline branches could regress without the
# pipeline test catching it.
#
# The shape both tests target: a multi-room project where ONE room is
# completed and another is still pending. After a regen finishes on the
# completed room, the persisted project.status MUST be "pending" — the
# truthful state — not "completed" (the pre-fix lie that survived because
# the inline branch only updated status when ``not any_room_processing``).
# ============================================================================


def _two_room_project_one_completed_one_pending():
    """Helper: 2-room project where room-1 has a completed variation and
    room-2 is still untouched (pending). Persisted ``project.status`` is
    deliberately stamped ``"completed"`` so the test can detect when the
    finalizer fails to overwrite it (the pre-fix bug).
    """
    return {
        "id": "proj-multi",
        "name": "Multi Room Test",
        "prompt": "Modern minimalist",
        # Stale/lying status — pre-fix the finalizer left this in place
        # because ``any_room_processing`` was True (room-2 is pending),
        # so the inline branch's ``if not any_room_processing`` body
        # never ran and the status stayed at this original value.
        "status": "completed",
        "rooms": [
            {
                "id": "room-1",
                "label": "Living Room",
                "original_image_url": "https://acct.blob.core.windows.net/images/staging/proj/originals/lr.png",
                "status": "completed",
                "variations": [{
                    "id": "var-1-1",
                    "status": "completed",
                    "image_url": "https://acct.blob.core.windows.net/images/staging/proj/variations/room-1/img.png",
                    "generation_metadata": {
                        "model": "gpt-image-2",
                        "adapted_prompt": "A cozy reading nook",
                        "generation_time_ms": 5000,
                    },
                }],
            },
            {
                "id": "room-2",
                "label": "Backyard",
                "original_image_url": "https://acct.blob.core.windows.net/images/staging/proj/originals/by.png",
                "status": "pending",
                "variations": [{
                    "id": "var-2-1",
                    "status": "pending",
                }],
            },
        ],
        "settings": {"variations_per_room": 1, "model": "gpt-image-2", "quality": "high", "size": "auto"},
    }


def test_regenerate_variation_finalizer_persists_pending_when_sibling_room_outstanding(
    client, mock_staging_deps
):
    """Issue 001 regression: the regenerate_variation finally block MUST
    persist ``project.status = "pending"`` when a sibling room is still
    pending — even though the regen itself completed cleanly on the
    target variation.

    Pre-fix the inline branch read::

        any_room_processing = any(r.status in ("pending", "processing") ...)
        if not any_room_processing:
            ... fresh_project.status = ...

    so ``fresh_project.status`` was never assigned when ANY peer room was
    still outstanding. The finalizer then persisted whatever stale value
    came back from the prior ``get_project`` call — in the test fixture,
    that's ``"completed"``. The badge would lie.

    Post-fix all paths run through ``ProjectStatusCalculator``, which
    returns PENDING because room-2 is still ``pending``.
    """
    mock_container = mock_staging_deps["container"]
    project_data = _two_room_project_one_completed_one_pending()
    mock_container.read_item.return_value = project_data
    # Echo every write back so update_project responses look real.
    mock_container.replace_item.side_effect = lambda item, body: body

    # Make the pipeline a no-op async generator so the endpoint flows
    # cleanly through preflight → pipeline (does nothing) → finalizer.
    # The exact pipeline behaviour is irrelevant to the contract under
    # test; what matters is the finally block's status calc.
    mock_pipeline = mock_staging_deps["pipeline"]

    async def _empty_async_gen(*_args, **_kwargs):
        if False:
            yield  # pragma: no cover

    mock_pipeline.process_single_variation = _empty_async_gen

    with client.stream(
        "POST",
        "/api/v1/staging/projects/proj-multi/rooms/room-1/variations/var-1-1/regenerate?strategy=retry",
    ) as response:
        assert response.status_code == 200
        for _ in response.iter_bytes():
            pass

    # Collect every replace_item call body. The LAST one is the finalizer's
    # write — it carries the recomputed status. Pre-fix that field would
    # still read "completed" (the value baked into the mocked persisted
    # state). Post-fix it MUST be "pending" because room-2 is outstanding.
    persisted_bodies = [
        (call.kwargs.get("body") or call.args[1])
        for call in mock_container.replace_item.call_args_list
    ]
    assert persisted_bodies, "Expected at least one replace_item write (preflight + finalizer)"
    final_body = persisted_bodies[-1]

    assert final_body["status"] == "pending", (
        "Issue 001 regression: regenerate_variation finalizer must "
        "persist project.status='pending' when a sibling room is still "
        f"pending. Got status={final_body['status']!r}. "
        "Pre-fix the inline ``if not any_room_processing`` branch never "
        "fired in this scenario and left the stale 'completed' value "
        "intact; the calculator now overwrites unconditionally."
    )


def test_regenerate_room_finalizer_persists_pending_when_sibling_room_outstanding(
    client, mock_staging_deps
):
    """Issue 001 regression: same shape as the variation test above but
    for the room-level regen path. After room-1 finishes regenerating,
    project.status MUST read "pending" because room-2 is still pending.

    Pre-fix this finalizer also gated its write on
    ``if not any_processing``, so the persisted status remained the
    stale "completed" value from the read_item mock.
    """
    mock_container = mock_staging_deps["container"]
    project_data = _two_room_project_one_completed_one_pending()
    mock_container.read_item.return_value = project_data
    mock_container.replace_item.side_effect = lambda item, body: body

    mock_pipeline = mock_staging_deps["pipeline"]

    async def _empty_async_gen(*_args, **_kwargs):
        if False:
            yield  # pragma: no cover

    mock_pipeline.process_room = _empty_async_gen

    with client.stream(
        "POST",
        "/api/v1/staging/projects/proj-multi/rooms/room-1/regenerate",
    ) as response:
        assert response.status_code == 200
        for _ in response.iter_bytes():
            pass

    persisted_bodies = [
        (call.kwargs.get("body") or call.args[1])
        for call in mock_container.replace_item.call_args_list
    ]
    assert persisted_bodies, "Expected at least one replace_item write from the finalizer"
    final_body = persisted_bodies[-1]

    assert final_body["status"] == "pending", (
        "Issue 001 regression: regenerate_room finalizer must persist "
        "project.status='pending' when a sibling room is still pending. "
        f"Got status={final_body['status']!r}. Pre-fix the inline "
        "``if not any_processing`` branch never fired in this scenario "
        "and left the stale 'completed' value intact."
    )


# ============================================================================
# Issue 003 (projects-page-improvements PRD) — Per-room prompt addendum
# ----------------------------------------------------------------------------
# Two surfaces to test:
#
#   1. PATCH /projects/{id}/rooms/{rid} accepts ``{prompt_addendum: ...}``,
#      updates only that field, normalizes empty/whitespace to None, leaves
#      sibling rooms / variations / status untouched, and never triggers any
#      generation.
#
#   2. The per-variation regen path (the canonical "future generation"
#      surface) honors ``room.prompt_addendum`` by composing it into the
#      ``adapted_prompt`` that reaches ``process_single_variation``.
#      ``strategy="retry"`` does NOT recompose — it uses the prior
#      ``generation_metadata.adapted_prompt`` verbatim, since that prior
#      value already includes whatever addendum was in effect when it ran.
# ============================================================================


_ADDENDUM_TEXT = "ADDENDUM_SENTINEL — always upright in front of fence"


def _project_with_two_rooms_for_patch():
    """Two-room project so ``patch room A`` tests can assert room B is
    byte-for-byte unchanged after the write."""
    return {
        "id": "proj-patch",
        "name": "Patch Test",
        "prompt": "modern minimalist",
        "status": "completed",
        "rooms": [
            {
                "id": "room-A",
                "label": "Living Room",
                "original_image_url": "https://acct.blob.core.windows.net/images/staging/proj/originals/a.png",
                "status": "completed",
                "variations": [
                    {
                        "id": "var-A1",
                        "status": "completed",
                        "image_url": "https://acct.blob.core.windows.net/images/staging/proj/variations/room-A/v1.png",
                        "generation_metadata": {
                            "model": "gpt-image-2",
                            "adapted_prompt": "earlier prompt for A",
                        },
                    }
                ],
            },
            {
                "id": "room-B",
                "label": "Kitchen",
                "original_image_url": "https://acct.blob.core.windows.net/images/staging/proj/originals/b.png",
                "status": "completed",
                "prompt_addendum": "B's existing addendum",
                "variations": [
                    {
                        "id": "var-B1",
                        "status": "completed",
                        "image_url": "https://acct.blob.core.windows.net/images/staging/proj/variations/room-B/v1.png",
                    }
                ],
            },
        ],
        "settings": {"variations_per_room": 1, "model": "gpt-image-2", "quality": "high", "size": "auto"},
    }


def test_patch_room_addendum_persists_only_target_room(client, mock_staging_deps):
    """PATCH /projects/{id}/rooms/{rid} writes the addendum onto the
    target room only. Sibling room must be byte-identical (existing
    addendum, status, variations all unchanged). The endpoint normalizes
    whitespace-only / empty to None so the model stays clean.
    """
    mock_container = mock_staging_deps["container"]
    project_data = _project_with_two_rooms_for_patch()
    mock_container.read_item.return_value = project_data
    mock_container.replace_item.side_effect = lambda item, body: body

    sibling_before = _json_for_regen_tests.dumps(project_data["rooms"][1], sort_keys=True)

    response = client.patch(
        "/api/v1/staging/projects/proj-patch/rooms/room-A",
        json={"prompt_addendum": "  always upright   "},
    )
    assert response.status_code == 200, response.text

    persisted_body = mock_container.replace_item.call_args.kwargs.get("body")
    if persisted_body is None:
        persisted_body = mock_container.replace_item.call_args.args[1]

    room_a = next(r for r in persisted_body["rooms"] if r["id"] == "room-A")
    room_b = next(r for r in persisted_body["rooms"] if r["id"] == "room-B")

    # Whitespace stripped before persist — model stays clean.
    assert room_a["prompt_addendum"] == "always upright"
    # Sibling room unchanged byte-for-byte.
    assert _json_for_regen_tests.dumps(room_b, sort_keys=True) == sibling_before
    # Project-level fields not touched.
    assert persisted_body["status"] == "completed"
    assert persisted_body["prompt"] == "modern minimalist"


def test_patch_room_addendum_does_not_touch_room_internal_fields(client, mock_staging_deps):
    """The endpoint must not modify the target room's variations, status,
    image URL, or label. Those are owned by the generation pipeline."""
    mock_container = mock_staging_deps["container"]
    project_data = _project_with_two_rooms_for_patch()
    original_room_a = _json_for_regen_tests.dumps(project_data["rooms"][0], sort_keys=True)
    mock_container.read_item.return_value = project_data
    mock_container.replace_item.side_effect = lambda item, body: body

    response = client.patch(
        "/api/v1/staging/projects/proj-patch/rooms/room-A",
        json={"prompt_addendum": "new addendum"},
    )
    assert response.status_code == 200

    persisted_body = mock_container.replace_item.call_args.kwargs.get("body") \
        or mock_container.replace_item.call_args.args[1]
    room_a = next(r for r in persisted_body["rooms"] if r["id"] == "room-A")

    # Reconstruct what room-A "should" look like with only addendum changed
    # and assert all other fields match the original byte-for-byte.
    room_a_minus_addendum = {k: v for k, v in room_a.items() if k != "prompt_addendum"}
    original_minus_addendum = {
        k: v for k, v in _json_for_regen_tests.loads(original_room_a).items()
        if k != "prompt_addendum"
    }
    assert room_a_minus_addendum == original_minus_addendum


def test_patch_room_addendum_normalizes_empty_to_none(client, mock_staging_deps):
    """Empty string and whitespace-only strings are normalized to None
    so the persisted shape stays consistent. This matches the composer's
    treatment of empty/whitespace as 'absent'."""
    mock_container = mock_staging_deps["container"]
    project_data = _project_with_two_rooms_for_patch()
    # Pre-existing addendum on room-B that we'll clear.
    mock_container.read_item.return_value = project_data
    mock_container.replace_item.side_effect = lambda item, body: body

    response = client.patch(
        "/api/v1/staging/projects/proj-patch/rooms/room-B",
        json={"prompt_addendum": "   \n  "},
    )
    assert response.status_code == 200

    persisted_body = mock_container.replace_item.call_args.kwargs.get("body") \
        or mock_container.replace_item.call_args.args[1]
    room_b = next(r for r in persisted_body["rooms"] if r["id"] == "room-B")
    assert room_b["prompt_addendum"] is None


def test_patch_room_addendum_explicit_null_clears_existing(client, mock_staging_deps):
    """Passing ``null`` explicitly clears any existing addendum."""
    mock_container = mock_staging_deps["container"]
    project_data = _project_with_two_rooms_for_patch()
    mock_container.read_item.return_value = project_data
    mock_container.replace_item.side_effect = lambda item, body: body

    response = client.patch(
        "/api/v1/staging/projects/proj-patch/rooms/room-B",
        json={"prompt_addendum": None},
    )
    assert response.status_code == 200

    persisted_body = mock_container.replace_item.call_args.kwargs.get("body") \
        or mock_container.replace_item.call_args.args[1]
    room_b = next(r for r in persisted_body["rooms"] if r["id"] == "room-B")
    assert room_b["prompt_addendum"] is None


def test_patch_room_returns_404_when_project_missing(client, mock_staging_deps):
    from azure.cosmos.exceptions import CosmosResourceNotFoundError
    mock_container = mock_staging_deps["container"]
    mock_container.read_item.side_effect = CosmosResourceNotFoundError(
        status_code=404, message="Not found"
    )
    response = client.patch(
        "/api/v1/staging/projects/nope/rooms/room-A",
        json={"prompt_addendum": "x"},
    )
    assert response.status_code == 404


def test_patch_room_returns_404_when_room_missing(client, mock_staging_deps):
    mock_container = mock_staging_deps["container"]
    mock_container.read_item.return_value = _project_with_two_rooms_for_patch()
    response = client.patch(
        "/api/v1/staging/projects/proj-patch/rooms/room-DOES-NOT-EXIST",
        json={"prompt_addendum": "x"},
    )
    assert response.status_code == 404


def test_patch_room_does_not_trigger_regeneration(client, mock_staging_deps):
    """The endpoint must not enqueue any generation work. Verifies the
    pipeline is never called and the response is a plain JSON
    Project payload (not an SSE stream)."""
    mock_container = mock_staging_deps["container"]
    mock_pipeline = mock_staging_deps["pipeline"]
    project_data = _project_with_two_rooms_for_patch()
    mock_container.read_item.return_value = project_data
    mock_container.replace_item.side_effect = lambda item, body: body

    response = client.patch(
        "/api/v1/staging/projects/proj-patch/rooms/room-A",
        json={"prompt_addendum": "new"},
    )
    assert response.status_code == 200
    # Plain JSON, not text/event-stream.
    assert response.headers["content-type"].startswith("application/json")
    # Pipeline never invoked.
    assert not mock_pipeline.process_room.called
    assert not mock_pipeline.generate_project.called
    assert not mock_pipeline.process_single_variation.called


def test_patch_room_returns_updated_project(client, mock_staging_deps):
    """The response payload must include the freshly-updated project
    so the frontend can update its local state without an extra GET."""
    mock_container = mock_staging_deps["container"]
    project_data = _project_with_two_rooms_for_patch()
    mock_container.read_item.return_value = project_data
    mock_container.replace_item.side_effect = lambda item, body: body

    response = client.patch(
        "/api/v1/staging/projects/proj-patch/rooms/room-A",
        json={"prompt_addendum": "fresh addendum"},
    )
    assert response.status_code == 200
    body = response.json()
    assert "project" in body
    project = body["project"]
    room_a = next(r for r in project["rooms"] if r["id"] == "room-A")
    assert room_a["prompt_addendum"] == "fresh addendum"


# ----------------------------------------------------------------------------
# Generation behavior — the addendum actually reaches the image-gen prompt
# ----------------------------------------------------------------------------


def _project_for_addendum_regen(*, addendum, with_brief=True):
    """One-room project with ``room.prompt_addendum`` set. Used to
    verify the per-variation regen path composes the addendum into the
    final ``adapted_prompt`` that reaches ``process_single_variation``.
    """
    project = {
        "id": "proj-addendum",
        "name": "Addendum Regen Test",
        "prompt": "USER_INTENT_SENTINEL — modern",
        "status": "completed",
        "rooms": [{
            "id": "room-1",
            "label": "Living Room",
            "original_image_url": "https://acct.blob.core.windows.net/images/staging/proj/originals/photo.png",
            "status": "completed",
            "prompt_addendum": addendum,
            "analysis": {"description": "A sunlit room", "features": []},
            "variations": [{
                "id": "var-1",
                "status": "completed",
                "image_url": "https://acct.blob.core.windows.net/images/staging/proj/variations/room-1/img.png",
                "generation_metadata": {
                    "model": "gpt-image-2",
                    "adapted_prompt": _PRIOR_PROMPT,
                    "generation_time_ms": 5000,
                },
            }],
        }],
        "settings": {"variations_per_room": 1, "model": "gpt-image-2", "quality": "high", "size": "auto"},
        "analyses": [{"room_id": "room-1", "description": "A sunlit room", "features": []}],
    }
    if with_brief:
        project["design_brief"] = {
            "global_instructions": "BRIEF_INTENT_SENTINEL — warm scandinavian",
            "object_palette": [
                {"name": "Sofa", "category": "furniture", "default_quantity": 1, "size": "3-seater", "placement": "facing window"},
            ],
            "placement_guide": {"back_row": "art"},
            "per_image_notes": {},
            "preserve_elements": [],
            "per_image_objects": {},
        }
    return project


def test_fresh_regen_composes_room_addendum_into_adapted_prompt(
    client, mock_staging_deps,
):
    """When ``room.prompt_addendum`` is set, the per-variation
    ``strategy=fresh`` regen path must compose the addendum into the
    ``adapted_prompt`` that reaches ``process_single_variation``.

    The composer appends the addendum to whatever base prompt source
    won (brief or adapt_prompt). We assert on the captured
    ``adapted_prompt`` argument because that's what gets persisted to
    ``generation_metadata.adapted_prompt`` and what determines the
    image-gen call.
    """
    mock_container = mock_staging_deps["container"]
    project_data = _project_for_addendum_regen(addendum=_ADDENDUM_TEXT, with_brief=True)
    mock_container.read_item.return_value = project_data
    mock_container.replace_item.side_effect = lambda item, body: body

    captured_llm = _make_captured_llm()

    captured_prompts: list[str] = []

    async def _capturing_psv(self, project, room, variation, adapted_prompt):
        captured_prompts.append(adapted_prompt)
        if False:
            yield  # pragma: no cover

    from backend.core.staging_pipeline import StagingPipeline

    with _patch("backend.core.async_llm_client", captured_llm), \
         _patch.object(StagingPipeline, "process_single_variation", _capturing_psv):
        with client.stream(
            "POST",
            "/api/v1/staging/projects/proj-addendum/rooms/room-1/variations/var-1/regenerate?strategy=fresh",
        ) as response:
            assert response.status_code == 200, response.text
            for _ in response.iter_bytes():
                pass

    assert len(captured_prompts) == 1, (
        f"process_single_variation must be called exactly once; got {captured_prompts!r}"
    )
    final_prompt = captured_prompts[0]
    assert _ADDENDUM_TEXT in final_prompt, (
        f"The room addendum must appear in the final adapted_prompt that "
        f"reaches process_single_variation. Got: {final_prompt!r}"
    )
    # Sanity: the LLM-generated base ("new direction" from the captured
    # mock) is ALSO present, separated by a paragraph break.
    assert "new direction\n\n" + _ADDENDUM_TEXT == final_prompt


def test_fresh_regen_no_brief_path_composes_room_addendum(
    client, mock_staging_deps,
):
    """The no-brief fallback path (``adapt_prompt`` instead of
    ``brief_to_prompts``) must also compose the addendum. The composer
    is at the last mile so any source of base prompt flows through it.
    """
    mock_container = mock_staging_deps["container"]
    project_data = _project_for_addendum_regen(addendum=_ADDENDUM_TEXT, with_brief=False)
    mock_container.read_item.return_value = project_data
    mock_container.replace_item.side_effect = lambda item, body: body

    captured_llm = _make_captured_llm()
    captured_prompts: list[str] = []

    async def _capturing_psv(self, project, room, variation, adapted_prompt):
        captured_prompts.append(adapted_prompt)
        if False:
            yield  # pragma: no cover

    async def _fake_analyze_room(self, image_b64):
        return {"description": "A sunlit room", "features": []}

    from backend.core.staging_pipeline import StagingPipeline
    from backend.core.azure_storage import AzureBlobStorageService

    with _patch("backend.core.async_llm_client", captured_llm), \
         _patch.object(StagingPipeline, "process_single_variation", _capturing_psv), \
         _patch.object(StagingPipeline, "analyze_room", _fake_analyze_room), \
         _patch.object(AzureBlobStorageService, "get_asset_content",
                       return_value=(b"FAKE_IMG", "image/png")):
        with client.stream(
            "POST",
            "/api/v1/staging/projects/proj-addendum/rooms/room-1/variations/var-1/regenerate?strategy=fresh",
        ) as response:
            assert response.status_code == 200, response.text
            for _ in response.iter_bytes():
                pass

    assert len(captured_prompts) == 1
    assert _ADDENDUM_TEXT in captured_prompts[0]


def test_retry_regen_does_not_recompose_addendum(client, mock_staging_deps):
    """``strategy=retry`` uses the prior ``adapted_prompt`` verbatim and
    does NOT pass through the composer. Per PRD § Further Notes:

        > Retry semantics intentionally do not re-run the composer. To pick
        > up a new addendum on an existing variation the user must use Edit
        > Prompt or regenerate the whole room.

    This guards against a regression where someone "helpfully" composes
    the addendum onto a Retry, double-appending the addendum (since the
    prior prompt already includes whatever addendum was in effect when
    it was first generated).
    """
    mock_container = mock_staging_deps["container"]
    # The prior prompt does NOT contain the current addendum text.
    project_data = _project_for_addendum_regen(addendum=_ADDENDUM_TEXT, with_brief=False)
    mock_container.read_item.return_value = project_data
    mock_container.replace_item.side_effect = lambda item, body: body

    captured_prompts: list[str] = []

    async def _capturing_psv(self, project, room, variation, adapted_prompt):
        captured_prompts.append(adapted_prompt)
        if False:
            yield  # pragma: no cover

    from backend.core.staging_pipeline import StagingPipeline

    with _patch.object(StagingPipeline, "process_single_variation", _capturing_psv):
        with client.stream(
            "POST",
            "/api/v1/staging/projects/proj-addendum/rooms/room-1/variations/var-1/regenerate?strategy=retry",
        ) as response:
            assert response.status_code == 200, response.text
            for _ in response.iter_bytes():
                pass

    assert len(captured_prompts) == 1
    final_prompt = captured_prompts[0]
    # Retry path uses the prior prompt VERBATIM.
    assert final_prompt == _PRIOR_PROMPT, (
        f"Retry must use prior adapted_prompt verbatim (no recomposition). "
        f"Got: {final_prompt!r}"
    )
    # The current addendum text must NOT have been appended.
    assert _ADDENDUM_TEXT not in final_prompt


def test_retry_with_no_prior_prompt_falls_back_to_fresh_and_composes_addendum(
    client, mock_staging_deps,
):
    """Retry with no prior ``adapted_prompt`` falls back to the fresh
    path (per issue 004 of single-variation-regen PRD), and the fresh
    path composes the room addendum onto the freshly-generated base
    prompt. This pins the joint behavior across two prior PRDs.
    """
    mock_container = mock_staging_deps["container"]
    project_data = _project_for_addendum_regen(addendum=_ADDENDUM_TEXT, with_brief=False)
    # Strip the prior adapted_prompt so retry MUST fall back to fresh.
    project_data["rooms"][0]["variations"][0]["generation_metadata"] = {
        "model": "gpt-image-2",
        "adapted_prompt": None,
    }
    mock_container.read_item.return_value = project_data
    mock_container.replace_item.side_effect = lambda item, body: body

    captured_prompts: list[str] = []

    async def _capturing_psv(self, project, room, variation, adapted_prompt):
        captured_prompts.append(adapted_prompt)
        if False:
            yield  # pragma: no cover

    async def _stub_adapt(self, user_prompt, room_analysis, n_variations,
                          rejected_prompt=None):
        return ["fresh-fallback base"]

    async def _fake_analyze_room(self, image_b64):
        return {"description": "A room", "features": []}

    from backend.core.staging_pipeline import StagingPipeline
    from backend.core.azure_storage import AzureBlobStorageService

    with _patch.object(StagingPipeline, "process_single_variation", _capturing_psv), \
         _patch.object(StagingPipeline, "adapt_prompt", _stub_adapt), \
         _patch.object(StagingPipeline, "analyze_room", _fake_analyze_room), \
         _patch.object(AzureBlobStorageService, "get_asset_content",
                       return_value=(b"FAKE_IMG", "image/png")):
        with client.stream(
            "POST",
            "/api/v1/staging/projects/proj-addendum/rooms/room-1/variations/var-1/regenerate?strategy=retry",
        ) as response:
            assert response.status_code == 200, response.text
            for _ in response.iter_bytes():
                pass

    # Composer ran on the fresh-fallback base.
    assert captured_prompts == [f"fresh-fallback base\n\n{_ADDENDUM_TEXT}"], (
        f"Retry-with-no-prior must fall back to fresh AND compose the "
        f"addendum. Got: {captured_prompts!r}"
    )


# ----------------------------------------------------------------------------
# Issue 003: queued-projects-stay-processing integration tests.
#
# These tests pin the bug-report scenario at the HTTP boundary: a project
# with status='processing' and pending rooms must NOT flip to 'failed' when
# the staleness window elapses. The reconcile path NEVER produces 'failed'
# in the new design.
# ----------------------------------------------------------------------------


def _build_stale_processing_project(
    *,
    project_id: str = "proj-stuck",
    rooms_status: str = "pending",
    current_project_job_id: str | None = None,
    job_status: str = "pending",
):
    """Build a project doc that LOOKS stale to reconcile_project's
    staleness gate (updated_at well past the threshold) but is genuinely
    queued behind the worker. The pre-fix code flipped this to 'failed';
    the new code must keep it at 'processing'."""
    proj = {
        "id": project_id,
        "name": "Stuck Project",
        "prompt": "A queued project waiting for the worker",
        "status": "processing",
        "rooms": [
            {
                "id": "r1",
                "name": "Living Room",
                "label": "Living Room",
                "original_image_url": "/img/r1-original.png",
                "status": rooms_status,
                "image_url": "/img/r1.png",
                "thumb_url": "/img/r1.png",
                "variations": [],
                "settings": {
                    "variations_per_room": 5,
                    "model": "gpt-image-2",
                    "quality": "high",
                    "size": "auto",
                },
            }
        ],
        "settings": {
            "variations_per_room": 5,
            "model": "gpt-image-2",
            "quality": "high",
            "size": "auto",
        },
        "updated_at": "2020-01-01T00:00:00Z",
        "created_at": "2020-01-01T00:00:00Z",
    }
    if current_project_job_id is not None:
        proj["current_project_job_id"] = current_project_job_id
    return proj


def test_get_project_queued_with_pending_job_stays_processing(client, mock_staging_deps):
    """Bug-report scenario: a project genuinely queued behind the worker
    must not flip to 'failed' (or any other status) just because the
    staleness window elapsed. The active non-terminal job in the jobs
    container is the source of truth."""
    mock_container = mock_staging_deps["container"]
    job_store = mock_staging_deps["job_store"]

    proj = _build_stale_processing_project(
        rooms_status="pending",
        current_project_job_id="proj-stuck:project:project:rev1",
    )
    mock_container.read_item.return_value = proj

    # Active, non-terminal job in the jobs container -> short-circuit
    # to 'no change'. Status must remain 'processing'.
    job_store.get_job.return_value = {
        "id": "proj-stuck:project:project:rev1",
        "project_id": "proj-stuck",
        "status": "pending",
    }

    response = client.get("/api/v1/staging/projects/proj-stuck")
    assert response.status_code == 200
    data = response.json()
    assert data["project"]["status"] == "processing", (
        "Pre-fix bug: project flipped to 'failed' on staleness. New "
        "behavior: active non-terminal job keeps status at 'processing'."
    )
    job_store.get_job.assert_called_once_with(
        "proj-stuck:project:project:rev1", "proj-stuck"
    )


def test_get_project_no_job_id_stays_processing(client, mock_staging_deps):
    """Legacy project without ``current_project_job_id`` is left alone by
    the new derivation path (short-circuits to no-change). Status stays
    where it was; the user's escape hatch is the explicit /reset endpoint.
    """
    mock_container = mock_staging_deps["container"]
    job_store = mock_staging_deps["job_store"]

    proj = _build_stale_processing_project(
        rooms_status="pending",
        current_project_job_id=None,
    )
    mock_container.read_item.return_value = proj

    response = client.get("/api/v1/staging/projects/proj-stuck")
    assert response.status_code == 200
    data = response.json()
    assert data["project"]["status"] == "processing"
    # Short-circuit: don't even consult the jobs container.
    assert job_store.get_job.call_count == 0


def test_get_project_terminal_job_with_pending_rooms_yields_pending_not_failed(
    client, mock_staging_deps
):
    """Worker has finished the job (terminal status) but the rooms ended
    up in mixed/all-pending states. Reconcile path NEVER produces
    'failed' (AC#6); status derives to 'pending' instead."""
    mock_container = mock_staging_deps["container"]
    job_store = mock_staging_deps["job_store"]

    proj = _build_stale_processing_project(
        rooms_status="pending",
        current_project_job_id="proj-stuck:project:project:rev1",
    )
    mock_container.read_item.return_value = proj

    # Job done (any terminal status). Rooms are all-pending -> derived
    # status is 'pending'. The bug used to derive 'failed' here.
    job_store.get_job.return_value = {
        "id": "proj-stuck:project:project:rev1",
        "project_id": "proj-stuck",
        "status": "succeeded",
    }

    response = client.get("/api/v1/staging/projects/proj-stuck")
    assert response.status_code == 200
    data = response.json()
    assert data["project"]["status"] == "pending"


def test_get_project_terminal_failed_job_with_failed_rooms_yields_pending(
    client, mock_staging_deps
):
    """Direct repro of the headline bug: worker reports 'failed', rooms
    are all 'failed' in their last attempt — but the reconcile path must
    still surface 'pending' (the user can retry). Failure is reserved for
    the worker / cancellation cascade / producer-side error paths."""
    mock_container = mock_staging_deps["container"]
    job_store = mock_staging_deps["job_store"]

    proj = _build_stale_processing_project(
        rooms_status="failed",
        current_project_job_id="proj-stuck:project:project:rev1",
    )
    mock_container.read_item.return_value = proj
    job_store.get_job.return_value = {
        "id": "proj-stuck:project:project:rev1",
        "project_id": "proj-stuck",
        "status": "failed",
    }

    response = client.get("/api/v1/staging/projects/proj-stuck")
    assert response.status_code == 200
    data = response.json()
    assert data["project"]["status"] == "pending", (
        "Reconcile path NEVER produces 'failed'. The pre-fix bug derived "
        "'failed' from all-failed rooms; the new behavior is 'pending'."
    )


def test_get_project_persists_status_change_with_single_writeback(
    client, mock_staging_deps
):
    """When status transitions from processing → pending (terminal job +
    pending rooms), exactly one writeback to storage occurs. We're not
    fanning out a writeback per derivation pass."""
    mock_container = mock_staging_deps["container"]
    job_store = mock_staging_deps["job_store"]

    proj = _build_stale_processing_project(
        rooms_status="pending",
        current_project_job_id="proj-stuck:project:project:rev1",
    )
    mock_container.read_item.return_value = proj
    job_store.get_job.return_value = {
        "id": "proj-stuck:project:project:rev1",
        "project_id": "proj-stuck",
        "status": "succeeded",
    }

    response = client.get("/api/v1/staging/projects/proj-stuck")
    assert response.status_code == 200

    # Exactly one writeback. The reconcile path mutated rooms (variations
    # cleanup) AND the status-from-jobs path mutated status — but the
    # endpoint coalesces them into a single update_project call.
    assert mock_container.replace_item.call_count <= 1


def test_list_projects_does_not_flip_queued_projects_to_failed(
    client, mock_staging_deps
):
    """List endpoint applies the same derivation path. A queued project
    in the list response must surface 'processing', not 'failed'."""
    mock_container = mock_staging_deps["container"]
    job_store = mock_staging_deps["job_store"]

    proj = _build_stale_processing_project(
        rooms_status="pending",
        current_project_job_id="proj-stuck:project:project:rev1",
    )

    def _query(query=None, **kwargs):
        if query and "SELECT VALUE COUNT(1)" in query:
            return [1]
        return [proj]

    mock_container.query_items = _query
    job_store.get_job.return_value = {
        "id": "proj-stuck:project:project:rev1",
        "project_id": "proj-stuck",
        "status": "pending",
    }

    response = client.get("/api/v1/staging/projects?limit=10")
    assert response.status_code == 200
    data = response.json()
    assert len(data["projects"]) == 1
    assert data["projects"][0]["status"] == "processing"


def test_reset_project_force_resets_status_off_processing(client, mock_staging_deps):
    """The /reset endpoint is the user's manual escape hatch for stuck
    projects without a tracked job id. compute_project_status_from_jobs
    short-circuits in that case (returns None), so reset_project applies
    _derive_status_from_rooms directly to ensure the project doesn't stay
    stuck in 'processing'."""
    mock_container = mock_staging_deps["container"]

    proj = _build_stale_processing_project(
        rooms_status="pending",
        current_project_job_id=None,  # No job id; compute returns None.
    )
    mock_container.read_item.return_value = proj

    response = client.post("/api/v1/staging/projects/proj-stuck/reset")
    assert response.status_code == 200
    data = response.json()
    assert data["project"]["status"] == "pending", (
        "Reset must derive from rooms when there's no job id; otherwise "
        "the project stays stuck in 'processing'."
    )


def test_reset_project_force_with_terminal_job_derives_from_rooms(
    client, mock_staging_deps
):
    """When the job has reached a terminal state, reset still produces
    a sensible status from rooms (and never 'failed' from the reconcile
    path)."""
    mock_container = mock_staging_deps["container"]
    job_store = mock_staging_deps["job_store"]

    proj = _build_stale_processing_project(
        rooms_status="failed",
        current_project_job_id="proj-stuck:project:project:rev1",
    )
    mock_container.read_item.return_value = proj
    job_store.get_job.return_value = {
        "id": "proj-stuck:project:project:rev1",
        "project_id": "proj-stuck",
        "status": "failed",
    }

    response = client.post("/api/v1/staging/projects/proj-stuck/reset")
    assert response.status_code == 200
    data = response.json()
    assert data["project"]["status"] == "pending"
