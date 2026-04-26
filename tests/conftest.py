"""Shared test fixtures for the Visionary Lab backend.

All tests run without real Azure credentials by mocking external services.
"""

import os
import sys
import pytest
from unittest.mock import MagicMock, patch

# Set environment variables BEFORE importing anything from the backend,
# so that Settings() picks up test values instead of requiring real creds.
os.environ["MODEL_PROVIDER"] = "azure"
os.environ["AI_FOUNDRY_ENDPOINT"] = "https://test-foundry.cognitiveservices.azure.com/"
os.environ["LLM_DEPLOYMENT"] = "gpt-5-4"
os.environ["IMAGEGEN_DEPLOYMENT"] = "gpt-image-2"
os.environ["IMAGEGEN_1_MINI_DEPLOYMENT"] = "gpt-image-1-mini"
os.environ["FLUX_KONTEXT_DEPLOYMENT"] = "flux-kontext-pro"
os.environ["SORA_DEPLOYMENT"] = "sora-2"
os.environ["AZURE_STORAGE_ACCOUNT_NAME"] = "teststorage"
os.environ["AZURE_BLOB_SERVICE_URL"] = "https://teststorage.blob.core.windows.net/"
os.environ["AZURE_BLOB_IMAGE_CONTAINER"] = "images"
os.environ["AZURE_BLOB_VIDEO_CONTAINER"] = "videos"
os.environ["AZURE_COSMOS_DB_ENDPOINT"] = "https://test.documents.azure.com:443/"

# --- Module-level mocks ---
# backend.core.__init__ creates real Azure SDK clients at import time
# (DefaultAzureCredential, AzureOpenAI, BlobServiceClient, Sora, etc.).
# These hang in tests because they try to reach real Azure endpoints.
# We patch them here so ANY test that transitively imports backend.core
# gets mocks instead of real network calls.
_mock_credential = MagicMock()
_mock_token_provider = MagicMock(return_value="mock-token")

# Patch azure.identity before backend.core can import it
_identity_patch = patch("azure.identity.DefaultAzureCredential", return_value=_mock_credential)
_token_patch = patch("azure.identity.get_bearer_token_provider", return_value=_mock_token_provider)
_blob_patch = patch("azure.storage.blob.BlobServiceClient", return_value=MagicMock())
_openai_patch = patch("openai.AzureOpenAI", return_value=MagicMock())
_async_openai_patch = patch("openai.AsyncAzureOpenAI", return_value=MagicMock())

_identity_patch.start()
_token_patch.start()
_blob_patch.start()
_openai_patch.start()
_async_openai_patch.start()

from fastapi.testclient import TestClient


@pytest.fixture(scope="session")
def mock_azure_storage():
    """Mock Azure Blob Storage so no real connection is made."""
    mock_service = MagicMock()
    mock_container = MagicMock()
    mock_service.get_container_client.return_value = mock_container
    mock_container.exists.return_value = True

    with patch(
        "backend.core.azure_storage.BlobServiceClient",
        return_value=mock_service,
    ):
        yield mock_service


@pytest.fixture(scope="session")
def mock_cosmos():
    """Mock Cosmos DB client."""
    with patch("backend.core.cosmos_client.CosmosClient") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_db = MagicMock()
        mock_client.get_database_client.return_value = mock_db
        mock_container = MagicMock()
        mock_db.get_container_client.return_value = mock_container
        yield mock_container


@pytest.fixture
def mock_staging_storage():
    """Mock StagingStorageService for staging endpoint tests."""
    with patch("backend.core.staging_storage.CosmosClient") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_db = MagicMock()
        mock_client.get_database_client.return_value = mock_db
        mock_container = MagicMock()
        mock_db.create_container_if_not_exists.return_value = mock_container
        yield mock_container


@pytest.fixture(scope="session")
def app(mock_azure_storage, mock_cosmos):
    """Create a FastAPI test application with mocked external services."""
    from backend.main import app as _app
    return _app


@pytest.fixture(scope="session")
def client(app):
    """HTTP test client for the FastAPI application."""
    return TestClient(app)


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
