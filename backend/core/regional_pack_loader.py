"""RegionalPackLoader — load JSON content packs keyed by ``(domain, region)``.

Issue 017 of the image-pipeline-and-project-ux-overhaul PRD. Packs ship
as code (JSON files committed to the repo under
``backend/data/regional_packs/<domain>/<region>.json``) — there is no
user-upload path. The wizard's first step uses ``DomainClassifier`` to
infer the domain from a free-text user description, then the loader
resolves a pack and the wizard surfaces the pack's chips as quick-reply
options on subsequent steps (keyed by ``BriefSectionRegistry`` section
ids so the wizard step config and the chips line up by construction).

Two public entry points:

* ``RegionalPackLoader.get(domain, region)`` — always returns a
  ``RegionalPack``. Unknown ``(domain, region)`` returns an empty pack
  (``chips == {}``) so callers can iterate without null-checks (per AC
  bullet "unknown key returns empty"). The empty pack carries the
  requested ``domain`` / ``region`` so the caller can still tell what
  was asked for.
* ``DomainClassifier.classify(text)`` — keyword-based, returns the
  matching domain id (e.g. ``"landscaping"``) or ``None``. The
  wizard's step-1 form should treat ``None`` as "fall back to the
  generic chip-less flow". The classifier is intentionally simple —
  the PRD pins this as a "simple domain classifier", and an LLM-backed
  classifier would add a wizard-blocking network round-trip on every
  keystroke.

Pack JSON shape::

    {
      "domain": "landscaping",
      "region": "castle-rock",
      "display_name": "Castle Rock, Colorado — Landscaping",
      "description": "...",
      "chips": {
        "<section_id>": ["chip 1", "chip 2", ...],
        ...
      }
    }

``section_id`` keys SHOULD be members of
``brief_section_registry.section_ids()`` — unknown ids are tolerated on
load (the loader does not validate against the registry so a future
section addition does not break older packs) but the wizard will only
render chips it knows how to slot into a step.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)


# Default packs directory — sibling to ``backend/core``. Tests inject
# their own ``packs_dir`` so they don't depend on the shipped pack
# content drifting.
_DEFAULT_PACKS_DIR = Path(__file__).resolve().parent.parent / "data" / "regional_packs"


@dataclass(frozen=True)
class RegionalPack:
    """A loaded content pack for one ``(domain, region)`` combination.

    ``chips`` is a sparse map keyed by ``BriefSectionRegistry`` section
    ids. Values are ordered lists — the wizard renders them in list
    order so packs control chip presentation without a separate sort
    contract. Empty chip lists are valid (a pack might want to declare
    coverage of a section without surfacing chips for it).
    """

    domain: str
    region: str
    display_name: str = ""
    description: str = ""
    chips: Mapping[str, List[str]] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        """``True`` when the pack carries no chips for any section."""
        return not any(self.chips.values()) if self.chips else True


class RegionalPackLoader:
    """Resolves ``RegionalPack`` instances from disk.

    Packs are loaded lazily on first access and cached in-process so
    repeated lookups don't re-read JSON. The loader is a deep module
    around the small surface area of "(domain, region) → pack" so the
    wizard does not need to know the on-disk layout.

    Construction takes an optional ``packs_dir`` so tests can point at
    a fixture directory. Production callers use the default
    (``backend/data/regional_packs``).
    """

    def __init__(self, packs_dir: Optional[Path] = None) -> None:
        self._packs_dir: Path = Path(packs_dir) if packs_dir is not None else _DEFAULT_PACKS_DIR
        self._cache: Dict[Tuple[str, str], RegionalPack] = {}

    def get(self, domain: str, region: str) -> RegionalPack:
        """Return the pack for ``(domain, region)``.

        Always returns a ``RegionalPack``. Unknown / missing /
        unreadable / malformed packs return an empty pack carrying the
        requested ``domain`` / ``region`` so callers iterating over
        ``pack.chips`` never have to null-check. All failure modes are
        logged at WARNING; the loader never raises.
        """
        key = (domain, region)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        pack = self._load(domain, region)
        self._cache[key] = pack
        return pack

    def list_available(self) -> List[Tuple[str, str]]:
        """Return all ``(domain, region)`` pairs with a pack on disk.

        Used by admin tooling and tests. Order is filesystem-defined;
        callers that need a stable order should sort.
        """
        if not self._packs_dir.is_dir():
            return []
        out: List[Tuple[str, str]] = []
        for domain_dir in self._packs_dir.iterdir():
            if not domain_dir.is_dir():
                continue
            for pack_file in domain_dir.glob("*.json"):
                out.append((domain_dir.name, pack_file.stem))
        return out

    def _load(self, domain: str, region: str) -> RegionalPack:
        pack_path = self._packs_dir / domain / f"{region}.json"
        empty = RegionalPack(domain=domain, region=region)
        if not pack_path.is_file():
            return empty
        try:
            raw = json.loads(pack_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "regional_pack.load_failed domain=%s region=%s path=%s err=%s",
                domain, region, pack_path, exc,
            )
            return empty
        if not isinstance(raw, dict):
            logger.warning(
                "regional_pack.invalid_root_type domain=%s region=%s type=%s",
                domain, region, type(raw).__name__,
            )
            return empty
        chips_raw = raw.get("chips") or {}
        chips: Dict[str, List[str]] = {}
        if isinstance(chips_raw, dict):
            for section_id, values in chips_raw.items():
                if not isinstance(section_id, str) or not isinstance(values, list):
                    continue
                # Stringify defensively; chip values are user-facing labels.
                chips[section_id] = [str(v) for v in values]
        return RegionalPack(
            domain=str(raw.get("domain", domain)),
            region=str(raw.get("region", region)),
            display_name=str(raw.get("display_name", "") or ""),
            description=str(raw.get("description", "") or ""),
            chips=chips,
        )


# ----------------------------------------------------------------------
# Domain classifier
# ----------------------------------------------------------------------

# Keyword → domain id. Each domain lists tokens that, when found as a
# whole word in the lowercased input, vote for that domain. The domain
# with the most votes wins; ties are broken by registration order
# (dict insertion order, Python 3.7+).
_DOMAIN_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "landscaping": (
        "landscape", "landscaping", "yard", "backyard", "frontyard",
        "garden", "gardening", "plant", "plants", "planting",
        "lawn", "turf", "grass", "tree", "trees", "shrub", "shrubs",
        "hedge", "mulch", "patio", "pergola", "fence", "fencing",
        "xeriscape", "paver", "pavers", "flowerbed", "perennial",
        "perennials", "evergreen", "evergreens",
    ),
}


class DomainClassifier:
    """Maps a free-text description to a domain id.

    Pure-python keyword voting — no LLM, no network, deterministic.
    The PRD pins this as a "simple domain classifier"; the wizard
    layers an LLM-driven follow-up on top so the classifier only has
    to be right enough to seed step 1.

    ``classify`` returns ``None`` when no domain hits its threshold so
    the wizard can fall back to a generic chip-less flow rather than
    confidently mis-classifying a never-seen-before domain.
    """

    # Minimum keyword-match count to claim a classification. ``1`` is
    # generous on purpose — most user descriptions in the wizard's
    # first step are a single sentence.
    MIN_VOTES: int = 1

    def __init__(self, keywords: Optional[Mapping[str, Tuple[str, ...]]] = None) -> None:
        self._keywords: Mapping[str, Tuple[str, ...]] = keywords or _DOMAIN_KEYWORDS

    def classify(self, text: str) -> Optional[str]:
        """Return the matching domain id, or ``None`` if no match."""
        if not text or not text.strip():
            return None
        tokens = _tokenise(text)
        if not tokens:
            return None
        best_domain: Optional[str] = None
        best_score = 0
        for domain, words in self._keywords.items():
            score = sum(1 for w in words if w in tokens)
            if score > best_score:
                best_score = score
                best_domain = domain
        if best_score < self.MIN_VOTES:
            return None
        return best_domain


def _tokenise(text: str) -> set:
    """Lowercase + split on non-alphanumeric. Returns a set for O(1)
    membership testing in the keyword loop."""
    out: set = set()
    buf: List[str] = []
    for ch in text.lower():
        if ch.isalnum():
            buf.append(ch)
        else:
            if buf:
                out.add("".join(buf))
                buf = []
    if buf:
        out.add("".join(buf))
    return out
