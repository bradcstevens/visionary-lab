"""Regression tests for frontend–backend API bug fixes."""
import io
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi import UploadFile


@pytest.fixture
def mock_staging_deps():
    """Mock all staging dependencies."""
    with patch("backend.core.staging_storage.CosmosClient") as mock_cosmos, \
         patch("backend.core.staging_storage.DefaultAzureCredential") as mock_cred, \
         patch("backend.api.endpoints.staging.get_staging_pipeline") as mock_pipeline_fn:

        # Mock CosmosClient chain
        mock_client = MagicMock()
        mock_cosmos.return_value = mock_client
        mock_db = MagicMock()
        mock_client.get_database_client.return_value = mock_db
        mock_container = MagicMock()
        mock_db.create_container_if_not_exists.return_value = mock_container

        # Mock credential
        mock_cred.return_value = MagicMock()

        # Mock pipeline
        mock_pipeline = MagicMock()
        mock_pipeline_fn.return_value = mock_pipeline

        yield {"container": mock_container, "pipeline": mock_pipeline}


# --- Task 1: Upload rooms accepts 'images' field name ---

def test_upload_rooms_accepts_images_field(client, mock_staging_deps):
    """Backend upload_rooms endpoint accepts 'images' as the file field name."""
    mock_container = mock_staging_deps["container"]
    mock_container.read_item.return_value = {
        "id": "proj-123",
        "name": "Test",
        "prompt": "Test",
        "status": "uploading",
        "rooms": [],
        "settings": {"variations_per_room": 2, "model": "gpt-image-2", "quality": "high", "size": "auto"},
    }
    mock_container.upsert_item.return_value = None

    with patch("backend.api.endpoints.staging.AzureBlobStorageService") as mock_blob_cls:
        mock_blob = AsyncMock()
        mock_blob_cls.return_value = mock_blob
        mock_blob.upload_asset.return_value = {"url": "https://test.blob.core.windows.net/img.png"}

        response = client.post(
            "/api/v1/staging/projects/proj-123/rooms",
            files=[("images", ("test.png", io.BytesIO(b"fake-png"), "image/png"))],
            data={"labels": '["Backyard East"]'},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["rooms_added"] == 1
    assert data["rooms"][0]["label"] == "Backyard East"


# --- Task 2: Create project with nested settings ---

def test_create_project_with_nested_settings(client, mock_staging_deps):
    """Backend expects settings as a nested object, not flat fields."""
    mock_container = mock_staging_deps["container"]
    mock_container.create_item.return_value = {
        "id": "proj-456",
        "name": "Backyard Fence Line",
        "prompt": "Add layered privacy screen",
        "status": "uploading",
        "rooms": [],
        "settings": {"variations_per_room": 3, "model": "gpt-image-2", "quality": "high", "size": "auto"},
        "created_at": "2026-04-26T00:00:00Z",
        "updated_at": "2026-04-26T00:00:00Z",
        "doc_type": "staging_project",
    }

    response = client.post("/api/v1/staging/projects", json={
        "name": "Backyard Fence Line",
        "prompt": "Add layered privacy screen",
        "settings": {
            "variations_per_room": 3,
            "model": "gpt-image-2",
            "quality": "high",
            "size": "auto",
        },
    })

    assert response.status_code == 201
    data = response.json()
    assert data["project"]["settings"]["variations_per_room"] == 3


# --- Task 3: Generate endpoint without /stream suffix ---

def test_generate_endpoint_exists_without_stream_suffix(client, mock_staging_deps):
    """The generate endpoint is /generate, not /generate/stream."""
    mock_container = mock_staging_deps["container"]
    mock_container.read_item.return_value = {
        "id": "proj-789",
        "name": "Test",
        "prompt": "Test",
        "status": "uploading",
        "rooms": [
            {
                "id": "room-1",
                "label": "Room 1",
                "original_image_url": "https://test.blob.core.windows.net/img.png",
                "status": "pending",
                "variations": [{"id": "v-1", "status": "pending"}],
            }
        ],
        "settings": {"variations_per_room": 1, "model": "gpt-image-2", "quality": "high", "size": "auto"},
    }

    with patch("backend.api.endpoints.staging.get_staging_pipeline") as mock_pipeline_fn:
        mock_pipeline = MagicMock()
        mock_pipeline_fn.return_value = mock_pipeline

        async def fake_generate(project):
            yield {"type": "project_completed", "status": "completed"}

        mock_pipeline.generate_project = fake_generate

        response = client.post("/api/v1/staging/projects/proj-789/generate")
        assert response.status_code == 200

        response_404 = client.post("/api/v1/staging/projects/proj-789/generate/stream")
        assert response_404.status_code in (404, 405)
