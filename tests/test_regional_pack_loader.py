"""Tests for ``backend/core/regional_pack_loader.py`` — issue 017.

Covers the AC bullets:

* ``RegionalPackLoader`` returns a pack by ``(domain, region)``;
  unknown key returns empty.
* Castle Rock landscaping pack ships as JSON in the repo.
* Domain classifier maps a free-text user description to a domain id.
* Wizard step config consumes the pack to surface quick-reply chips
  (asserted indirectly: chip keys overlap the section registry).
* Unit test: pack lookup hit/miss; classifier returns expected domain
  for landscaping prompt.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.core.brief_section_registry import section_ids
from backend.core.regional_pack_loader import (
    DomainClassifier,
    RegionalPack,
    RegionalPackLoader,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def packs_dir(tmp_path: Path) -> Path:
    """Empty packs directory for isolation from the shipped Castle Rock pack."""
    d = tmp_path / "packs"
    d.mkdir()
    return d


def _write_pack(packs_dir: Path, domain: str, region: str, body: dict) -> Path:
    p = packs_dir / domain / f"{region}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(body), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# RegionalPackLoader.get — happy / miss / malformed
# ---------------------------------------------------------------------------


def test_get_returns_pack_with_chips(packs_dir: Path) -> None:
    _write_pack(packs_dir, "landscaping", "castle-rock", {
        "domain": "landscaping",
        "region": "castle-rock",
        "display_name": "Castle Rock",
        "description": "Front Range",
        "chips": {"object_palette": ["Spruce", "Pine"]},
    })
    loader = RegionalPackLoader(packs_dir=packs_dir)

    pack = loader.get("landscaping", "castle-rock")

    assert pack.domain == "landscaping"
    assert pack.region == "castle-rock"
    assert pack.display_name == "Castle Rock"
    assert pack.description == "Front Range"
    assert pack.chips == {"object_palette": ["Spruce", "Pine"]}
    assert pack.is_empty is False


def test_get_unknown_domain_returns_empty_pack(packs_dir: Path) -> None:
    loader = RegionalPackLoader(packs_dir=packs_dir)

    pack = loader.get("does-not-exist", "anywhere")

    assert isinstance(pack, RegionalPack)
    assert pack.domain == "does-not-exist"
    assert pack.region == "anywhere"
    assert dict(pack.chips) == {}
    assert pack.is_empty is True


def test_get_unknown_region_for_known_domain_returns_empty_pack(packs_dir: Path) -> None:
    _write_pack(packs_dir, "landscaping", "castle-rock", {"chips": {"a": ["x"]}})
    loader = RegionalPackLoader(packs_dir=packs_dir)

    pack = loader.get("landscaping", "atlantis")

    assert pack.is_empty is True
    assert pack.domain == "landscaping"
    assert pack.region == "atlantis"


def test_get_caches_result(packs_dir: Path) -> None:
    pack_path = _write_pack(packs_dir, "landscaping", "castle-rock", {
        "chips": {"object_palette": ["Spruce"]},
    })
    loader = RegionalPackLoader(packs_dir=packs_dir)

    first = loader.get("landscaping", "castle-rock")
    # Mutating the file after first read must NOT change the cached pack.
    pack_path.write_text(json.dumps({"chips": {"object_palette": ["Replaced"]}}), encoding="utf-8")
    second = loader.get("landscaping", "castle-rock")

    assert first is second
    assert second.chips == {"object_palette": ["Spruce"]}


def test_get_with_missing_packs_dir_returns_empty(tmp_path: Path) -> None:
    missing = tmp_path / "no-such-dir"
    loader = RegionalPackLoader(packs_dir=missing)

    pack = loader.get("landscaping", "castle-rock")

    assert pack.is_empty is True


def test_get_handles_malformed_json_without_raising(packs_dir: Path) -> None:
    pack_path = packs_dir / "landscaping" / "castle-rock.json"
    pack_path.parent.mkdir(parents=True, exist_ok=True)
    pack_path.write_text("{not valid json", encoding="utf-8")
    loader = RegionalPackLoader(packs_dir=packs_dir)

    pack = loader.get("landscaping", "castle-rock")

    assert pack.is_empty is True
    assert pack.domain == "landscaping"


def test_get_handles_non_dict_root(packs_dir: Path) -> None:
    _write_pack(packs_dir, "landscaping", "castle-rock", {})  # create dir
    (packs_dir / "landscaping" / "castle-rock.json").write_text("[1,2,3]", encoding="utf-8")
    loader = RegionalPackLoader(packs_dir=packs_dir)

    pack = loader.get("landscaping", "castle-rock")

    assert pack.is_empty is True


def test_get_drops_non_string_section_keys_and_non_list_values(packs_dir: Path) -> None:
    _write_pack(packs_dir, "landscaping", "castle-rock", {
        "chips": {
            "object_palette": ["Spruce"],
            "regional_constraints": "not-a-list",
        },
    })
    loader = RegionalPackLoader(packs_dir=packs_dir)

    pack = loader.get("landscaping", "castle-rock")

    assert "object_palette" in pack.chips
    assert "regional_constraints" not in pack.chips


def test_get_stringifies_chip_values(packs_dir: Path) -> None:
    _write_pack(packs_dir, "landscaping", "castle-rock", {
        "chips": {"object_palette": ["Spruce", 42, None]},
    })
    loader = RegionalPackLoader(packs_dir=packs_dir)

    pack = loader.get("landscaping", "castle-rock")

    assert pack.chips["object_palette"] == ["Spruce", "42", "None"]


def test_list_available_returns_all_packs(packs_dir: Path) -> None:
    _write_pack(packs_dir, "landscaping", "castle-rock", {})
    _write_pack(packs_dir, "landscaping", "denver", {})
    _write_pack(packs_dir, "interior", "default", {})
    loader = RegionalPackLoader(packs_dir=packs_dir)

    available = sorted(loader.list_available())

    assert available == [
        ("interior", "default"),
        ("landscaping", "castle-rock"),
        ("landscaping", "denver"),
    ]


def test_list_available_with_no_packs_dir_returns_empty(tmp_path: Path) -> None:
    loader = RegionalPackLoader(packs_dir=tmp_path / "missing")
    assert loader.list_available() == []


# ---------------------------------------------------------------------------
# Shipped Castle Rock pack
# ---------------------------------------------------------------------------


def test_shipped_castle_rock_pack_loads_and_has_known_sections() -> None:
    """Default-constructed loader resolves the pack that ships in-repo."""
    loader = RegionalPackLoader()

    pack = loader.get("landscaping", "castle-rock")

    assert pack.is_empty is False
    assert pack.domain == "landscaping"
    assert pack.region == "castle-rock"
    assert pack.display_name  # non-empty


def test_shipped_castle_rock_pack_chips_keyed_by_registry_section_ids() -> None:
    """Wizard step config consumes the pack to surface chips per section.

    The pack's chip keys must be a subset of the canonical section
    registry so the wizard can slot them by id.
    """
    loader = RegionalPackLoader()
    pack = loader.get("landscaping", "castle-rock")

    registry_ids = set(section_ids())
    assert set(pack.chips.keys()).issubset(registry_ids), (
        f"unknown section keys: {set(pack.chips.keys()) - registry_ids}"
    )
    # And at least one section must carry chips — otherwise the wizard
    # has nothing to surface.
    assert any(len(v) > 0 for v in pack.chips.values())


def test_shipped_castle_rock_pack_includes_signature_species() -> None:
    """Defends against accidental wholesale replacement of pack content."""
    loader = RegionalPackLoader()
    pack = loader.get("landscaping", "castle-rock")

    palette = pack.chips.get("object_palette", [])
    assert any("Baby Blue Eyes Spruce" in chip for chip in palette)


# ---------------------------------------------------------------------------
# DomainClassifier
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", [
    "I want to redesign my backyard landscaping",
    "Help me pick plants for my front yard",
    "Replace the rock beds along my fence with shrubs and evergreens",
    "Garden redesign with drought-tolerant perennials",
    "Add a hedge of trees behind the patio",
])
def test_classifier_returns_landscaping_for_landscaping_prompts(text: str) -> None:
    classifier = DomainClassifier()
    assert classifier.classify(text) == "landscaping"


@pytest.mark.parametrize("text", [
    "",
    "   ",
    "a totally unrelated query about quantum mechanics",
    "weather forecast for tomorrow",
])
def test_classifier_returns_none_for_unrecognised_or_empty_input(text: str) -> None:
    classifier = DomainClassifier()
    assert classifier.classify(text) is None


def test_classifier_is_case_insensitive() -> None:
    classifier = DomainClassifier()
    assert classifier.classify("BACKYARD LANDSCAPING") == "landscaping"


def test_classifier_handles_punctuation_around_keywords() -> None:
    classifier = DomainClassifier()
    assert classifier.classify("My garden, please!") == "landscaping"


def test_classifier_only_matches_whole_words_not_substrings() -> None:
    """``"plant"`` must NOT fire on words like ``"plantar"``.

    Pinned because a substring-based classifier would mis-route a
    medical or footwear query into landscaping.
    """
    classifier = DomainClassifier()
    assert classifier.classify("severe plantar fasciitis pain") is None


def test_classifier_accepts_custom_keyword_map() -> None:
    classifier = DomainClassifier(keywords={"interior": ("kitchen", "sofa")})
    assert classifier.classify("redesign my kitchen") == "interior"
    assert classifier.classify("backyard garden") is None
