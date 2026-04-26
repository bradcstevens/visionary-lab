# AI Design Questionnaire & Bug Fix Spec

> Combined redesign of the Visionary Lab Projects wizard: fixes 3 critical frontend–backend API mismatches and adds a full-featured, image-aware AI Design Session that replaces the flat text prompt with an interactive split-panel conversation and structured Design Brief editor.

## Problem

The current Projects wizard has three blocking bugs and a shallow UX:

1. **Upload API mismatch** — frontend sends `room_files`/`room_names`, backend expects `images`/`labels`. Uploads fail silently.
2. **Create project schema mismatch** — frontend sends flat fields (`style`, `variations_per_room`), backend expects nested `settings` object. Project creation sends malformed data.
3. **Streaming URL mismatch** — frontend calls `/generate/stream` and `/regenerate/stream`, backend routes are `/generate` and `/regenerate`. SSE streaming never connects.
4. **Shallow prompt input** — users type a single free-text prompt with no guidance. The AI gets minimal context and generates generic results. For complex scenarios like backyard landscaping with specific plant species, placement, and layering requirements, a free-text box is inadequate.

## Solution

Replace the 4-step linear wizard with a 5-step flow that adds an AI Design Session (split-panel chat) and a structured Design Brief editor:

| Step | Name | What happens |
|------|------|-------------|
| 1 | Name | Project name (unchanged) |
| 2 | Upload | Multi-image upload with editable labels (bug-fixed) |
| 3 | AI Design Session | Split-panel: image thumbnails left, AI chat right |
| 4 | Design Brief Editor | AI-generated structured form, fully editable |
| 5 | Generate | Review summary + launch (bug-fixed) |

## Architecture

### New Backend Endpoints

#### `POST /api/v1/staging/projects/{id}/analyze`

Triggers AI analysis of all uploaded images. Called automatically when entering Step 3.

**Response:**
```json
{
  "analyses": [
    {
      "room_id": "uuid",
      "description": "Backyard view showing wooden fence, turf area, existing low shrubs",
      "features": ["fence", "turf", "shrubs"],
      "zones": ["fence_line", "open_turf", "patio_edge"]
    }
  ]
}
```

**Implementation:** Reuses the existing `StagingPipeline.analyze_room()` method, called in parallel for all rooms via `asyncio.gather`. The system message is updated to detect outdoor/indoor context and identify spatial zones suitable for object placement.

#### `POST /api/v1/staging/projects/{id}/chat`

Conversational endpoint for the AI Design Session. Maintains conversation context and decides when enough information has been gathered.

**Request:**
```json
{
  "message": "I want trees along the fence",
  "conversation_history": [
    { "role": "assistant", "content": "I've analyzed your 14 photos..." },
    { "role": "user", "content": "I want trees along the fence" }
  ],
  "focused_image_id": "room-123"
}
```

**Response:**
```json
{
  "reply": "Great choice! What species are you considering?",
  "ready_for_brief": false,
  "suggested_actions": ["specify_species", "choose_density", "set_height_preference"]
}
```

**Implementation:** New `DesignChatService` class wraps the async LLM client. The system prompt includes:
- Image analyses (from the `/analyze` call)
- Conversation history
- If `focused_image_id` is set, that image's analysis is highlighted
- Instructions to ask about: placement, species, quantities, what to preserve, seasonal preferences
- Instructions to set `ready_for_brief: true` after 3-5 substantive exchanges covering the key topics

#### `POST /api/v1/staging/projects/{id}/brief`

Generates a structured Design Brief from the conversation history.

- `POST` — AI generates a new brief from conversation + image analyses
- `PUT` — saves user edits (request body is a `DesignBrief` object)

**POST Response:**
```json
{
  "brief": {
    "global_instructions": "Add layered evergreen privacy screen along fence line...",
    "plant_palette": [
      {
        "species": "Vanderwolf's Pyramid Limber Pine",
        "botanical_name": "Pinus flexilis 'Vanderwolf's Pyramid'",
        "quantity": 3,
        "size": "8-10 ft",
        "placement": "back row along east fence",
        "visual_notes": "Silvery-blue twisted needles, narrow pyramid form"
      }
    ],
    "placement_guide": {
      "back_row": "Tall trees: Limber Pine, Columnar Norway Spruce",
      "middle_row": "Mid-height shrubs",
      "front_row": "Low groundcover",
      "accent_areas": "Climbing jasmine on pergola posts"
    },
    "per_image_notes": {},
    "preserve_elements": ["existing patio", "fire pit", "pergola structure"],
    "settings": {
      "variations_per_room": 5,
      "model": "gpt-image-2",
      "quality": "high",
      "size": "auto"
    }
  }
}
```

**Implementation:** New `BriefGeneratorService` takes conversation history + image analyses and prompts the LLM to produce structured JSON matching the `DesignBrief` schema. The prompt includes the full conversation and explicit instructions to extract plant species, quantities, placement, and preservation instructions.

### Bug Fixes to Existing Endpoints

1. **Upload rooms** — Frontend `stagingApi.ts` `uploadRooms()` updated: `room_files` → `images`, `room_names` → `labels` (as JSON string).
2. **Create project** — Frontend `stagingApi.ts` `createProject()` updated: wrap `variations_per_room`, `model`, `quality`, `size` in a `settings` object. Remove the `style` field (not in backend schema).
3. **Streaming URLs** — Frontend `stagingApi.ts` `streamGeneration()` and `streamRoomRegeneration()` updated: remove `/stream` suffix from URLs.
4. **Lint error** — Remove unused `getStatusColor` from `frontend/components/staging/RoomGroup.tsx`.

### New Data Models

```python
class ImageAnalysis(BaseModel):
    room_id: str
    description: str
    features: List[str]
    zones: List[str]

class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    focused_image_id: Optional[str] = None
    timestamp: Optional[datetime] = None

class ChatRequest(BaseModel):
    message: str
    conversation_history: List[ChatMessage] = Field(default_factory=list)
    focused_image_id: Optional[str] = None

class ChatResponse(BaseModel):
    reply: str
    ready_for_brief: bool = False
    suggested_actions: List[str] = Field(default_factory=list)

class PlantEntry(BaseModel):
    species: str
    botanical_name: Optional[str] = None
    quantity: int = 1
    size: str = ""
    placement: str = ""
    visual_notes: Optional[str] = None

class PlacementGuide(BaseModel):
    back_row: str = ""
    middle_row: Optional[str] = None
    front_row: Optional[str] = None
    accent_areas: Optional[str] = None

class DesignBrief(BaseModel):
    global_instructions: str
    plant_palette: List[PlantEntry] = Field(default_factory=list)
    placement_guide: PlacementGuide = Field(default_factory=PlacementGuide)
    per_image_notes: Dict[str, str] = Field(default_factory=dict)
    preserve_elements: List[str] = Field(default_factory=list)
    settings: StagingSettings = Field(default_factory=StagingSettings)
```

### Prompt Adaptation Update

The existing `StagingPipeline.adapt_prompt()` method currently takes a flat text prompt. It will be updated to accept an optional `DesignBrief` object. When a brief is provided, the adaptation template changes to produce targeted prompts that reference specific plant species, visual characteristics, placement instructions, and per-image notes.

The `PROMPT_ADAPTATION_TEMPLATE` will be updated to handle both indoor and outdoor contexts. The current template says "virtual staging assistant" and references "room" and "furniture." The updated template will detect outdoor context from the image analysis and switch to landscaping-appropriate language.

### New Backend Services

**`DesignChatService`** (`backend/core/design_chat.py`)
- `__init__(async_llm_client, llm_deployment, image_analyses)` 
- `chat(message, conversation_history, focused_image_id) → ChatResponse`
- System prompt includes image analyses, conversation history, and guidance on what information to gather

**`BriefGeneratorService`** (`backend/core/brief_generator.py`)
- `__init__(async_llm_client, llm_deployment)`
- `generate_brief(conversation_history, image_analyses) → DesignBrief`
- `brief_to_prompts(brief, image_analyses) → Dict[str, List[str]]` — converts brief into per-image adapted prompts (key = `room_id`, value = list of variation prompt strings)

### New Frontend Components

**`ImageGalleryPanel`** (`frontend/components/staging/ImageGalleryPanel.tsx`)
- Left panel of the split view
- Groups thumbnails by AI-detected category from the `/analyze` response (images sharing common `features` are grouped, e.g., all images with "fence" feature → "Fence Line" group)
- Click-to-focus: clicking a thumbnail sets `focusedImageId` in state
- Shows focus indicator (blue border + eye icon) on selected image
- Collapsible groups with image count badges

**`DesignChat`** (`frontend/components/staging/DesignChat.tsx`)
- Right panel of the split view
- Renders chat messages (AI with avatar, user right-aligned)
- Streams AI responses using fetch + ReadableStream
- Shows `QuickReplyChips` below AI messages with `suggested_actions`
- "Generate Design Brief" CTA button when `ready_for_brief` is true
- Sends `focused_image_id` with each message
- Shows "Focused on: filename.png" badge in input area

**`QuickReplyChips`** (`frontend/components/staging/QuickReplyChips.tsx`)
- Renders AI-suggested clickable pills below messages
- Maps `suggested_actions` to human-readable labels
- Clicking a chip sends a pre-formatted message to the chat

**`DesignBriefEditor`** (`frontend/components/staging/DesignBriefEditor.tsx`)
- Full-page editor for Step 4
- Sections: Global Instructions (textarea), Plant Palette (table), Placement Guide (rows), Per-Image Notes (click thumbnail to open modal), Preserve Elements (tag input), Settings (model/quality/size/variations)
- All fields editable inline
- "Save & Continue" sends PUT to `/brief` endpoint

**`PlantPaletteTable`** (`frontend/components/staging/PlantPaletteTable.tsx`)
- Editable table with columns: Species, Botanical Name, Qty, Size, Placement, Visual Notes
- Add row button, delete row button per row
- Inline editing (click cell to edit)

**`GenerationSummary`** (`frontend/components/staging/GenerationSummary.tsx`)
- Step 5 review card
- Shows: project name, total images, variations per image, total variations, estimated time
- Collapsible brief summary section
- "Generate Project" launch button

### Mobile Responsive Strategy

On viewports below 768px, the split-panel in Step 3 becomes a tabbed interface:
- "Photos" tab shows the `ImageGalleryPanel`
- "Chat" tab shows the `DesignChat`
- A floating pill badge on the Chat tab shows the currently focused image name
- Tab switching preserves all state

## Testing Strategy

### Layer 1: Bug Fix Regression Tests (4 tests)

| Test | What it verifies |
|------|-----------------|
| `test_upload_rooms_field_names` | Endpoint accepts `images` field, returns rooms |
| `test_create_project_with_nested_settings` | `settings.variations_per_room` nesting works |
| `test_generate_endpoint_no_stream_suffix` | `/generate` returns SSE, not 404 |
| `test_roomgroup_no_unused_vars` | Lint clean after fixing `getStatusColor` |

### Layer 2: New Feature Unit Tests (7 tests)

| Test | What it verifies |
|------|-----------------|
| `test_design_brief_model_validation` | DesignBrief Pydantic model, required fields, plant palette schema |
| `test_chat_endpoint_returns_reply` | Mock LLM, verify response shape, `ready_for_brief` flag |
| `test_analyze_endpoint_returns_image_analyses` | Mock analyzer, verify per-image analysis structure |
| `test_brief_generation_from_conversation` | Mock LLM, verify structured brief output from chat history |
| `test_brief_to_prompt_adaptation` | Verify DesignBrief produces richer prompts than flat text |
| `test_chat_with_focused_image` | Verify `focused_image_id` injects image context into LLM call |
| `test_outdoor_prompt_template` | Verify prompt template adapts language for outdoor/landscaping |

### Layer 3: Backyard Landscaping Scenario Tests (6 tests)

Uses actual test data from `tests/projects/backyard-landscaping/` (14 images + BACKYARD.md plant details + landscaping-prompts.md).

| Test | What it verifies |
|------|-----------------|
| `test_backyard_project_creation` | Create project, upload 14 images, verify all rooms created |
| `test_backyard_image_analysis` | Analyze test images, verify fence/patio/pergola features detected |
| `test_backyard_chat_plant_selection` | Simulate conversation selecting specific plants from BACKYARD.md |
| `test_backyard_brief_includes_plant_details` | Verify brief has visual details (silvery-blue needles, pyramid form) |
| `test_backyard_adapted_prompts_are_specific` | Verify adapted prompts reference specific plants, placement, scene features |
| `test_backyard_per_image_notes` | Verify per-image notes for pergola vs fence generate different prompts |

### Layer 4: Frontend E2E Tests (6 tests)

| Test | What it verifies |
|------|-----------------|
| `test_wizard_5_step_flow` | Walk through all 5 steps, verify progression and back navigation |
| `test_split_panel_renders` | Verify ImageGalleryPanel and DesignChat render at Step 3 |
| `test_image_focus_click` | Click thumbnail, verify focus indicator and chat context badge |
| `test_design_brief_editor_fields` | Verify plant palette table, placement guide, per-image notes |
| `test_quick_reply_chips` | Verify AI suggestion chips render and send correct messages |
| `test_mobile_tabbed_fallback` | At narrow viewport, verify tabs replace split panel |

**Totals:** 23 new tests + 29 existing = 52 tests

## Files Changed

### Backend — Modified
- `backend/api/endpoints/staging.py` — add analyze, chat, brief endpoints
- `backend/core/staging_pipeline.py` — update `adapt_prompt` to accept DesignBrief, update prompt template for outdoor context
- `backend/models/staging.py` — add DesignBrief, PlantEntry, PlacementGuide, ImageAnalysis, ChatMessage, ChatRequest, ChatResponse models

### Backend — New
- `backend/core/design_chat.py` — DesignChatService
- `backend/core/brief_generator.py` — BriefGeneratorService

### Frontend — Modified
- `frontend/services/stagingApi.ts` — fix field names, fix schema, fix streaming URLs, add new API calls
- `frontend/components/staging/NewProjectWizard.tsx` — redesign to 5-step flow with new steps
- `frontend/components/staging/RoomGroup.tsx` — remove unused `getStatusColor`

### Frontend — New
- `frontend/components/staging/ImageGalleryPanel.tsx`
- `frontend/components/staging/DesignChat.tsx`
- `frontend/components/staging/QuickReplyChips.tsx`
- `frontend/components/staging/DesignBriefEditor.tsx`
- `frontend/components/staging/PlantPaletteTable.tsx`
- `frontend/components/staging/GenerationSummary.tsx`

### Tests — New
- `tests/test_bug_fixes.py` — 4 bug fix regression tests
- `tests/test_design_chat.py` — 3 chat + analyze unit tests
- `tests/test_design_brief.py` — 4 brief + adaptation unit tests
- `tests/test_backyard_scenario.py` — 6 scenario tests using test data
- `frontend/tests/e2e/ai-design-session.spec.ts` — 6 Playwright E2E tests
