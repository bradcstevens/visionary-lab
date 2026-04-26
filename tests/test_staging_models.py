"""Tests for staging project Pydantic models."""
import pytest
from pydantic import ValidationError


def test_create_project_request_valid():
    from backend.models.staging import CreateProjectRequest
    req = CreateProjectRequest(name="My Project", prompt="Modern minimalist")
    assert req.name == "My Project"
    assert req.prompt == "Modern minimalist"
    assert req.settings.variations_per_room == 5
    assert req.settings.model == "gpt-image-2"


def test_create_project_request_missing_name():
    from backend.models.staging import CreateProjectRequest
    with pytest.raises(ValidationError):
        CreateProjectRequest(prompt="Modern minimalist")


def test_create_project_request_missing_prompt():
    from backend.models.staging import CreateProjectRequest
    with pytest.raises(ValidationError):
        CreateProjectRequest(name="My Project")


def test_staging_settings_defaults():
    from backend.models.staging import StagingSettings
    s = StagingSettings()
    assert s.variations_per_room == 5
    assert s.model == "gpt-image-2"
    assert s.quality == "high"
    assert s.size == "auto"


def test_staging_settings_custom():
    from backend.models.staging import StagingSettings
    s = StagingSettings(variations_per_room=3, model="gpt-image-2", quality="auto")
    assert s.variations_per_room == 3


def test_staging_settings_validates_variations():
    from backend.models.staging import StagingSettings
    with pytest.raises(ValidationError):
        StagingSettings(variations_per_room=0)
    with pytest.raises(ValidationError):
        StagingSettings(variations_per_room=11)


def test_variation_model():
    from backend.models.staging import Variation
    v = Variation(id="abc")
    assert v.status == "pending"
    assert v.image_url is None
    assert v.error is None


def test_room_model():
    from backend.models.staging import Room
    r = Room(id="abc", label="Living Room", original_image_url="https://example.com/img.png")
    assert r.status == "pending"
    assert r.variations == []


def test_staging_project_model():
    from backend.models.staging import StagingProject, StagingSettings
    p = StagingProject(
        id="proj-1",
        name="Test Project",
        prompt="Modern style",
        settings=StagingSettings(),
    )
    assert p.status == "uploading"
    assert p.rooms == []
    assert p.folder_path is None


def test_staging_project_status_values():
    from backend.models.staging import StagingProject, StagingSettings, ProjectStatus
    for status in ["uploading", "processing", "completed", "failed"]:
        p = StagingProject(
            id="proj-1", name="Test", prompt="Test",
            settings=StagingSettings(), status=status,
        )
        assert p.status == status
