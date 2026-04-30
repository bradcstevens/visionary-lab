"""BriefGeneratorService — synthesizes conversation into a structured Design Brief."""
import json
import logging
import uuid
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from pydantic import ValidationError

from backend.core.brief_resolver import (
    reconcile_overrides_by_name,
    resolve_objects_for_image,
)
from backend.core.config import settings
from backend.core.retry import call_with_retry
from backend.models.design_brief import (
    ChatMessage,
    DesignBrief,
    ImageAnalysis,
    ImageObjectOverride,
    ObjectEntry,
    PlacementGuide,
    ReconcileSummary,
)

logger = logging.getLogger(__name__)

BRIEF_GENERATION_PROMPT = """You are a design assistant. Synthesize the conversation below into a structured Design Brief.

IMAGE ANALYSES:
{analyses_text}

CONVERSATION:
{conversation_text}

Extract and organize all design decisions into this exact JSON structure:
{{
  "global_instructions": "Overall description of what to add and the style direction",
  "object_palette": [
    {{
      "name": "Common name (e.g. 'Vanderwolf Pine', 'Adirondack chair', 'Pendant lamp')",
      "description": "Optional detail (botanical name, model number, etc.) or null",
      "category": "one of: plant, tree, rock, furniture, lighting, hardscape, decor, other",
      "default_quantity": 1,
      "size": "free-form size description",
      "placement": "where to put it",
      "visual_notes": "key visual characteristics for image generation or null"
    }}
  ],
  "placement_guide": {{
    "back_row": "Tall objects description",
    "middle_row": "Mid-height description or null",
    "front_row": "Low objects description or null",
    "accent_areas": "Special areas or null"
  }},
  "per_image_notes": {{}},
  "per_image_objects": {{}},
  "preserve_elements": ["list of things to keep unchanged"]
}}

Be specific about visual characteristics — these will be used to generate images.
Choose the best fitting category per object; if uncertain, use "other".

PER-IMAGE OBJECT OVERRIDES (per_image_objects):

`per_image_objects` MUST be an empty object `{{}}` UNLESS the conversation explicitly differentiates
quantities or placement of specific objects between specific images. Do NOT populate this field
unless the user said something like "eight in the side yard, three in the front yard" or "no
lavender in the patio image". When in doubt, leave it as `{{}}`.

When you DO populate it, the shape is:

  "per_image_objects": {{
    "<room_id>": [
      {{ "object_name": "<exact name from object_palette>",
         "quantity": <int >= 0>,
         "placement": "<override placement or null>",
         "enabled": <true | false> }}
    ]
  }}

Rules:
- `<room_id>` MUST be one of the room IDs from IMAGE ANALYSES above. Any other key will be discarded.
- `object_name` MUST exactly match the `name` field of an entry in `object_palette` (case-insensitive,
  whitespace-trimmed). Do NOT invent names not in the palette — those entries will be discarded.
- Only include overrides for the specific images and objects the user differentiated. Do not echo
  defaults that match the palette."""

BRIEF_TO_PROMPTS_TEMPLATE = """You are an image editing prompt writer. Given a Design Brief and an image description, generate {n} distinct prompts for an image editing model.

DESIGN BRIEF:
Global: {global_instructions}
Objects: {object_summary}
Placement: {placement_summary}
Preserve: {preserve_summary}

IMAGE DESCRIPTION: {image_description}
{per_image_note}

Generate {n} variation prompts. Each should:
- ADD the specified objects to the scene described above
- Reference specific objects with their visual characteristics
- Respect the placement guide (back row, middle, front)
- NOT remove or change elements listed in preserve
- Vary the interpretation: different arrangements, densities, or seasonal looks

Return a JSON object with a "prompts" key containing an array of {n} strings. Example: {{"prompts": ["prompt 1", "prompt 2"]}}"""


def _normalize_name(name: Any) -> str:
    """Canonical name for case-insensitive whitespace-trimmed matching.

    Mirrors ``brief_resolver._normalize_name``; defined separately here
    to avoid coupling the LLM-parse path to a private helper in another
    module.
    """
    if not isinstance(name, str):
        return ""
    return name.strip().lower()


def _parse_llm_per_image_objects(
    raw: Any,
    name_to_id: Dict[str, str],
    valid_room_ids: set,
) -> Dict[str, List[ImageObjectOverride]]:
    """Convert an LLM-emitted ``per_image_objects`` raw dict into the
    typed ``Dict[str, List[ImageObjectOverride]]`` shape.

    Each row is wrapped in a narrow try/except so a single malformed
    entry doesn't blow up the entire brief generation. Rows are dropped
    individually for any of:

    * Non-dict shape.
    * Missing or non-int ``quantity``.
    * Missing ``object_name``.
    * ``object_name`` not in ``name_to_id`` (unknown name OR ambiguous
      duplicate name in the palette).
    * Pydantic ``ValidationError`` from the typed model constructor
      (covers placement coercion failures, etc.).

    Whole rooms are dropped if their ``room_id`` isn't in
    ``valid_room_ids``. Per the per-image-object-quantities PRD issue 004.
    """
    result: Dict[str, List[ImageObjectOverride]] = {}
    if not isinstance(raw, dict):
        return result

    for room_id, room_overrides in raw.items():
        if room_id not in valid_room_ids:
            continue
        if not isinstance(room_overrides, list):
            continue
        room_typed: List[ImageObjectOverride] = []
        for entry in room_overrides:
            if not isinstance(entry, dict):
                continue
            try:
                normalized = _normalize_name(entry.get("object_name"))
                if not normalized:
                    continue
                object_id = name_to_id.get(normalized)
                if object_id is None:
                    continue
                # Build the typed model. ``quantity`` is required and
                # validated as int >= 0 by the model itself; non-int
                # / missing values raise ValidationError, which we
                # convert into a per-row drop here.
                override = ImageObjectOverride(
                    object_id=object_id,
                    quantity=entry["quantity"],
                    placement=entry.get("placement"),
                    enabled=entry.get("enabled", True),
                )
                room_typed.append(override)
            except (KeyError, TypeError, ValueError, ValidationError) as e:
                logger.debug(
                    "Dropped malformed LLM per_image_objects entry for %s: %s",
                    room_id, e,
                )
                continue
        if room_typed:
            result[room_id] = room_typed

    return result


class BriefGeneratorService:
    """Generates structured Design Briefs from conversations and converts them to prompts."""

    def __init__(self, async_llm_client, llm_deployment: str):
        self.async_llm_client = async_llm_client
        self.llm_deployment = llm_deployment

    async def generate_brief(
        self,
        conversation_history: List[ChatMessage],
        image_analyses: List[ImageAnalysis],
        previous_brief: Optional[DesignBrief] = None,
    ) -> Tuple[DesignBrief, ReconcileSummary]:
        """Synthesize the chat into a typed ``DesignBrief``.

        Returns a tuple of ``(brief, ReconcileSummary)``. When
        ``previous_brief`` is supplied (the wizard's regenerate flow),
        per-image overrides from the prior brief are carried forward by
        case-insensitive whitespace-trimmed name match against the new
        palette and the summary reports counts. When no
        ``previous_brief`` is given, the summary is zero-zero.

        Issue 004 of the per-image-object-quantities PRD.
        """
        analyses_text = "\n".join(
            f"- {a.room_id}: {a.description} (features: {', '.join(a.features)})"
            for a in image_analyses
        )
        conversation_text = "\n".join(
            f"{msg.role.upper()}: {msg.content}" for msg in conversation_history
        )

        system_content = BRIEF_GENERATION_PROMPT.format(
            analyses_text=analyses_text,
            conversation_text=conversation_text,
        )

        valid_room_ids = {a.room_id for a in image_analyses}

        for attempt in range(3):
            response = await self.async_llm_client.chat.completions.create(
                model=self.llm_deployment,
                messages=[{"role": "system", "content": system_content}],
                temperature=0.3,
                response_format={"type": "json_object"},
            )
            try:
                parsed = json.loads(response.choices[0].message.content)
                # Per the per-image-object-quantities PRD, generate_brief
                # explicitly assigns a UUID to each new palette entry.
                palette_entries: List[ObjectEntry] = []
                for raw_obj in parsed.get("object_palette", []):
                    if not isinstance(raw_obj, dict):
                        continue
                    raw_obj.setdefault("id", str(uuid.uuid4()))
                    palette_entries.append(ObjectEntry(**raw_obj))

                # Build name→id map for substituting LLM-emitted overrides.
                # Issue 004: LLM emits ``object_name`` tags (NOT
                # ``object_id``); we substitute to UUID at parse time.
                # Skip palette entries whose normalized name is duplicated
                # — the match would be ambiguous so we drop any override
                # referring to them.
                normalized_name_counts = Counter(
                    _normalize_name(e.name) for e in palette_entries
                )
                name_to_id: Dict[str, str] = {}
                for e in palette_entries:
                    normalized = _normalize_name(e.name)
                    if normalized_name_counts[normalized] > 1:
                        continue
                    name_to_id[normalized] = e.id

                # Walk LLM-emitted per_image_objects with narrow
                # try/except so a single bad row doesn't fail the whole
                # brief.
                per_image_objects = _parse_llm_per_image_objects(
                    raw=parsed.get("per_image_objects", {}),
                    name_to_id=name_to_id,
                    valid_room_ids=valid_room_ids,
                )

                brief = DesignBrief(
                    global_instructions=parsed.get("global_instructions", ""),
                    object_palette=palette_entries,
                    placement_guide=PlacementGuide(**parsed.get("placement_guide", {})),
                    per_image_notes=parsed.get("per_image_notes", {}),
                    per_image_objects=per_image_objects,
                    preserve_elements=parsed.get("preserve_elements", []),
                )

                # Issue 004: regenerate flow carries forward prior
                # per-image overrides by name. The reconcile call is a
                # no-op when there are no prior overrides — counts are
                # zero — so the public contract stays uniform whether
                # previous_brief is supplied or not.
                if previous_brief is not None:
                    brief, summary = reconcile_overrides_by_name(
                        previous_brief, brief
                    )
                else:
                    summary = ReconcileSummary(carried_forward=0, dropped=0)

                # Final filter: drop any room_ids that survived
                # reconciliation but aren't in the current image_analyses
                # (e.g., the user removed an image before regenerate).
                # Per rubber-duck non-blocking suggestion.
                filtered_per_image = {
                    rid: overrides
                    for rid, overrides in brief.per_image_objects.items()
                    if rid in valid_room_ids
                }
                if filtered_per_image != brief.per_image_objects:
                    brief = brief.copy(
                        update={"per_image_objects": filtered_per_image}
                    )

                return brief, summary
            except (json.JSONDecodeError, KeyError, TypeError, ValidationError) as e:
                logger.warning(f"Brief generation attempt {attempt + 1} failed: {e}")
                continue

        raise RuntimeError("Failed to generate Design Brief after 3 attempts")

    async def brief_to_prompts(
        self,
        brief: DesignBrief,
        image_analyses: List[ImageAnalysis],
        n_variations: int = 5,
    ) -> Dict[str, List[str]]:
        # Project-wide context that doesn't depend on per-image overrides.
        placement_summary = f"Back: {brief.placement_guide.back_row}"
        if brief.placement_guide.middle_row:
            placement_summary += f" | Middle: {brief.placement_guide.middle_row}"
        if brief.placement_guide.front_row:
            placement_summary += f" | Front: {brief.placement_guide.front_row}"
        preserve_summary = ", ".join(brief.preserve_elements) if brief.preserve_elements else "None specified"

        result: Dict[str, List[str]] = {}

        for analysis in image_analyses:
            # Per-image object_summary: resolver merges palette + per-image
            # overrides for THIS room_id, so two rooms with different
            # override maps produce different prompts. Issue 003 of the
            # per-image-object-quantities PRD.
            resolved_objects = resolve_objects_for_image(brief, room_id=analysis.room_id)
            object_summary = "; ".join(
                f"{ro.quantity}x {ro.name} ({ro.size}, {ro.placement})"
                + (f" — {ro.visual_notes}" if ro.visual_notes else "")
                for ro in resolved_objects
            )

            per_image_note = ""
            if analysis.room_id in brief.per_image_notes:
                per_image_note = f"SPECIAL NOTE FOR THIS IMAGE: {brief.per_image_notes[analysis.room_id]}"

            system_content = BRIEF_TO_PROMPTS_TEMPLATE.format(
                n=n_variations,
                global_instructions=brief.global_instructions,
                object_summary=object_summary,
                placement_summary=placement_summary,
                preserve_summary=preserve_summary,
                image_description=analysis.description,
                per_image_note=per_image_note,
            )

            for attempt in range(3):
                response = await call_with_retry(
                    lambda: self.async_llm_client.chat.completions.create(
                        model=self.llm_deployment,
                        messages=[{"role": "system", "content": system_content}],
                        temperature=0.8,
                        response_format={"type": "json_object"},
                    ),
                    semaphore=None,
                    model=self.llm_deployment,
                    attempts=settings.IMAGE_GEN_RETRY_ATTEMPTS,
                    base_delay=settings.IMAGE_GEN_RETRY_BASE_DELAY,
                    max_total_wait=settings.IMAGE_GEN_RETRY_MAX_TOTAL_WAIT,
                )
                try:
                    content = response.choices[0].message.content
                    parsed = json.loads(content)
                    prompts = None
                    if isinstance(parsed, list):
                        prompts = parsed
                    elif isinstance(parsed, dict):
                        # Try common key names the LLM might use
                        for key in ("prompts", "variations", "results", "data"):
                            if key in parsed and isinstance(parsed[key], list):
                                prompts = parsed[key]
                                break
                        # Fallback: grab the first list value in the dict
                        if prompts is None:
                            for v in parsed.values():
                                if isinstance(v, list):
                                    prompts = v
                                    break
                        # Last resort: if all values are non-empty strings,
                        # treat them as prompts (LLM returned {"1": "...", "2": "..."})
                        if prompts is None:
                            str_vals = [v for v in parsed.values() if isinstance(v, str) and len(v) > 20]
                            if len(str_vals) >= n_variations:
                                prompts = str_vals
                    if prompts is not None:
                        result[analysis.room_id] = [str(p) for p in prompts[:n_variations]]
                        break
                    logger.warning(
                        "Prompt generation attempt %d for %s returned unexpected structure: %s",
                        attempt + 1, analysis.room_id, type(parsed).__name__,
                    )
                except (json.JSONDecodeError, KeyError, IndexError) as e:
                    logger.warning(f"Prompt generation attempt {attempt + 1} for {analysis.room_id} failed: {e}")
                    continue
            else:
                logger.warning(f"All prompt generation attempts failed for image {analysis.room_id}, using global instructions as fallback")
                result[analysis.room_id] = [brief.global_instructions] * n_variations

        return result
