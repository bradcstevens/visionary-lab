"""DesignChatService — conversational AI for the Design Session."""
import json
import logging
from typing import List, Optional

from backend.models.design_brief import ChatMessage, ChatResponse, ImageAnalysis

logger = logging.getLogger(__name__)

DESIGN_CHAT_SYSTEM_PROMPT = """You are a landscape and interior design assistant helping users plan visual changes to their spaces.

IMAGE ANALYSES:
{analyses_text}

{focused_image_section}

Your job is to have a natural conversation to understand what the user wants to visualize. Ask about:
- What they want to add (plants, furniture, structures, etc.)
- Specific species, materials, or styles
- Where to place items (which areas, which images)
- Quantities and sizes
- What existing elements to preserve unchanged
- Any seasonal or style preferences

After gathering enough detail (typically 3-5 substantive exchanges), set ready_for_brief to true.

ALWAYS respond with valid JSON matching this schema:
{{"reply": "your message", "ready_for_brief": false, "suggested_actions": ["action_key1", "action_key2"]}}

suggested_actions are short keys like: specify_species, choose_density, set_height_preference, define_placement, choose_style, add_more_areas, generate_brief"""


class DesignChatService:
    """Handles conversational AI for the Design Session step."""

    def __init__(
        self,
        async_llm_client,
        llm_deployment: str,
        image_analyses: List[ImageAnalysis],
    ):
        self.async_llm_client = async_llm_client
        self.llm_deployment = llm_deployment
        self.image_analyses = image_analyses

    def _build_analyses_text(self) -> str:
        parts = []
        for a in self.image_analyses:
            parts.append(
                f"- Image '{a.room_id}': {a.description} "
                f"(features: {', '.join(a.features)}; zones: {', '.join(a.zones)})"
            )
        return "\n".join(parts) if parts else "No images analyzed yet."

    def _build_focused_section(self, focused_image_id: Optional[str]) -> str:
        if not focused_image_id:
            return ""
        for a in self.image_analyses:
            if a.room_id == focused_image_id:
                return (
                    f"\nFOCUSED IMAGE: The user is currently looking at image '{a.room_id}'.\n"
                    f"Description: {a.description}\n"
                    f"Features: {', '.join(a.features)}\n"
                    f"Zones: {', '.join(a.zones)}\n"
                    f"Tailor your response to this specific image."
                )
        return ""

    async def chat(
        self,
        message: str,
        conversation_history: List[ChatMessage],
        focused_image_id: Optional[str] = None,
    ) -> ChatResponse:
        system_content = DESIGN_CHAT_SYSTEM_PROMPT.format(
            analyses_text=self._build_analyses_text(),
            focused_image_section=self._build_focused_section(focused_image_id),
        )

        messages = [{"role": "system", "content": system_content}]
        for msg in conversation_history:
            messages.append({"role": msg.role, "content": msg.content})
        messages.append({"role": "user", "content": message})

        for attempt in range(3):
            response = await self.async_llm_client.chat.completions.create(
                model=self.llm_deployment,
                messages=messages,
                temperature=0.7,
                response_format={"type": "json_object"},
            )
            try:
                content = response.choices[0].message.content
                parsed = json.loads(content)
                return ChatResponse(
                    reply=parsed.get("reply", content),
                    ready_for_brief=parsed.get("ready_for_brief", False),
                    suggested_actions=parsed.get("suggested_actions", []),
                )
            except (json.JSONDecodeError, KeyError):
                logger.warning(f"Chat attempt {attempt + 1} returned invalid JSON, retrying")
                continue

        return ChatResponse(reply="I'm having trouble processing that. Could you rephrase?")
