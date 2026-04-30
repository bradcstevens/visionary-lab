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