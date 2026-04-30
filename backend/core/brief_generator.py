"""BriefGeneratorService — synthesizes conversation into a structured Design Brief."""
import json
import logging
import uuid
from typing import Dict, List

from pydantic import ValidationError

from backend.core.brief_resolver import resolve_objects_for_image
from backend.core.config import settings
from backend.core.retry import call_with_retry
from backend.models.design_brief import (
    ChatMessage,
    DesignBrief,
    ImageAnalysis,
    ObjectEntry,
    PlacementGuide,
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
  "preserve_elements": ["list of things to keep unchanged"]
}}

Be specific about visual characteristics — these will be used to generate images.
Choose the best fitting category per object; if uncertain, use "other"."""

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


class BriefGeneratorService:
    """Generates structured Design Briefs from conversations and converts them to prompts."""

    def __init__(self, async_llm_client, llm_deployment: str):
        self.async_llm_client = async_llm_client
        self.llm_deployment = llm_deployment

    async def generate_brief(
        self,
        conversation_history: List[ChatMessage],
        image_analyses: List[ImageAnalysis],
    ) -> DesignBrief:
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
                # explicitly assigns a UUID to each new palette entry. The
                # ObjectEntry default_factory would also do this implicitly,
                # but explicit assignment makes it clear in the call site
                # that ids originate here, not somewhere downstream.
                palette_entries = []
                for raw_obj in parsed.get("object_palette", []):
                    if not isinstance(raw_obj, dict):
                        continue
                    raw_obj.setdefault("id", str(uuid.uuid4()))
                    palette_entries.append(ObjectEntry(**raw_obj))
                return DesignBrief(
                    global_instructions=parsed.get("global_instructions", ""),
                    object_palette=palette_entries,
                    placement_guide=PlacementGuide(**parsed.get("placement_guide", {})),
                    per_image_notes=parsed.get("per_image_notes", {}),
                    preserve_elements=parsed.get("preserve_elements", []),
                )
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
        # Single project-wide object summary in this slice. Issue 003 will
        # introduce per-image summaries by moving this call inside the loop
        # and threading the room_id through resolve_objects_for_image.
        sentinel_room_id = image_analyses[0].room_id if image_analyses else "__project__"
        resolved_objects = resolve_objects_for_image(brief, room_id=sentinel_room_id)
        object_summary = "; ".join(
            f"{ro.quantity}x {ro.name} ({ro.size}, {ro.placement})"
            + (f" — {ro.visual_notes}" if ro.visual_notes else "")
            for ro in resolved_objects
        )
        placement_summary = f"Back: {brief.placement_guide.back_row}"
        if brief.placement_guide.middle_row:
            placement_summary += f" | Middle: {brief.placement_guide.middle_row}"
        if brief.placement_guide.front_row:
            placement_summary += f" | Front: {brief.placement_guide.front_row}"
        preserve_summary = ", ".join(brief.preserve_elements) if brief.preserve_elements else "None specified"

        result: Dict[str, List[str]] = {}

        for analysis in image_analyses:
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
