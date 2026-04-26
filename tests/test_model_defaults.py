"""Tests for the post-migration default model identifiers.

These guard against regressions where the codebase reverts to legacy slugs
(`gpt-5.4`, `gpt-image-2`, etc.) that the deployed Foundry resource may no
longer have.
"""

import pytest
from pydantic import ValidationError


def test_default_image_model_is_gpt_image_2():
    """Settings.DEFAULT_IMAGE_MODEL points at the new gpt-image-2 slug."""
    from backend.core.config import settings

    assert settings.DEFAULT_IMAGE_MODEL == "gpt-image-2"


def test_image_generation_request_default_model_is_gpt_image_2():
    """Pydantic Field default reflects the new model."""
    from backend.models.images import ImageGenerationRequest

    req = ImageGenerationRequest(prompt="hello")
    assert req.model == "gpt-image-2"


@pytest.mark.parametrize(
    "model",
    [
        "gpt-image-2",
        "gpt-image-1-mini",
        "flux-kontext-pro",
    ],
)
def test_image_generation_request_accepts_supported_models(model):
    from backend.models.images import ImageGenerationRequest

    req = ImageGenerationRequest(prompt="hello", model=model)
    assert req.model == model


def test_image_generation_request_rejects_unknown_model():
    from backend.models.images import ImageGenerationRequest

    with pytest.raises(ValidationError):
        ImageGenerationRequest(prompt="hello", model="totally-fake-model")


def test_gpt_image_client_slug_map_includes_gpt_image_2(monkeypatch):
    """The internal model→deployment lookup honors the new gpt-image-2 slug."""
    from backend.core import gpt_image as gpt_image_module
    from backend.core.config import settings

    monkeypatch.setattr(settings, "IMAGEGEN_DEPLOYMENT", "vislab-gpt-image-2", raising=False)
    monkeypatch.setattr(settings, "IMAGEGEN_1_MINI_DEPLOYMENT", "vislab-mini", raising=False)
    monkeypatch.setattr(settings, "FLUX_KONTEXT_DEPLOYMENT", "vislab-flux", raising=False)
    monkeypatch.setattr(settings, "IMAGEGEN_15_DEPLOYMENT", "", raising=False)

    # Build an instance bypassing __init__ so we don't need real Azure creds
    client = gpt_image_module.GPTImageClient.__new__(gpt_image_module.GPTImageClient)

    assert client._get_deployment_for_model("gpt-image-2") == "vislab-gpt-image-2"
    assert client._get_deployment_for_model("gpt-image-1-mini") == "vislab-mini"
    assert client._get_deployment_for_model("flux-kontext-pro") == "vislab-flux"
