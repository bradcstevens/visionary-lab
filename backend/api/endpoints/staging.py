"""FastAPI endpoints for virtual staging projects."""
import json
import logging
import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from backend.core.azure_storage import AzureBlobStorageService
from backend.core.config import settings
from backend.core.staging_storage import StagingStorageService
from backend.models.design_brief import (
    ChatRequest,
    ChatResponse,
    DesignBrief,
    GenerateBriefRequest,
    ImageAnalysis,
)
from backend.models.staging import (
    CreateProjectRequest,
    ItemStatus,
    ProjectListResponse,
    ProjectResponse,
    Room,
    StagingProject,
    UploadRoomsResponse,
    Variation,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def get_staging_storage() -> StagingStorageService:
    return StagingStorageService()


def get_image_analyzer():
    from backend.core import async_llm_client
    from backend.core.analyze import ImageAnalyzer
    return ImageAnalyzer(
        openai_client=None,
        model=settings.LLM_DEPLOYMENT,
        async_openai_client=async_llm_client,
    )


def get_staging_pipeline():
    from backend.core import async_llm_client
    from backend.core.analyze import ImageAnalyzer
    from backend.core.image_pipeline import ImagePipelineService
    from backend.core.staging_pipeline import StagingPipeline

    analyzer = ImageAnalyzer(
        openai_client=None,
        model=settings.LLM_DEPLOYMENT,
        async_openai_client=async_llm_client,
    )
    image_pipeline = ImagePipelineService()
    blob_service = AzureBlobStorageService()
    storage = StagingStorageService()

    return StagingPipeline(
        async_llm_client=async_llm_client,
        llm_deployment=settings.LLM_DEPLOYMENT,
        image_analyzer=analyzer,
        image_pipeline=image_pipeline,
        storage_service=storage,
        blob_service=blob_service,
    )


@router.post("/projects", response_model=ProjectResponse, status_code=201)
async def create_project(
    request: CreateProjectRequest,
    storage: StagingStorageService = Depends(get_staging_storage),
):
    project_data = {
        "id": str(uuid.uuid4()),
        "name": request.name,
        "prompt": request.prompt,
        "status": "uploading",
        "rooms": [],
        "settings": request.settings.dict(),
        "folder_path": f"staging/{request.name.lower().replace(' ', '-')}",
    }
    created = storage.create_project(project_data)
    return ProjectResponse(project=StagingProject(**{k: v for k, v in created.items() if k != "doc_type" and not k.startswith("_")}))


@router.get("/projects", response_model=ProjectListResponse)
async def list_projects(
    limit: int = 50,
    offset: int = 0,
    storage: StagingStorageService = Depends(get_staging_storage),
):
    projects_raw = storage.list_projects(limit=limit, offset=offset)
    total = storage.count_projects()
    projects = []
    for p in projects_raw:
        clean = {k: v for k, v in p.items() if k != "doc_type" and not k.startswith("_")}
        projects.append(StagingProject(**clean))
    return ProjectListResponse(projects=projects, total=total)


@router.get("/projects/{project_id}", response_model=ProjectResponse)
async def get_project(
    project_id: str,
    storage: StagingStorageService = Depends(get_staging_storage),
):
    project = storage.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    clean = {k: v for k, v in project.items() if k != "doc_type" and not k.startswith("_")}
    return ProjectResponse(project=StagingProject(**clean))


@router.delete("/projects/{project_id}")
async def delete_project(
    project_id: str,
    storage: StagingStorageService = Depends(get_staging_storage),
):
    if not storage.delete_project(project_id):
        raise HTTPException(status_code=404, detail="Project not found")
    return {"status": "deleted", "project_id": project_id}


@router.post("/projects/{project_id}/rooms", response_model=UploadRoomsResponse)
async def upload_rooms(
    project_id: str,
    images: List[UploadFile] = File(...),
    labels: Optional[str] = Form(None),
    storage: StagingStorageService = Depends(get_staging_storage),
):
    project_data = storage.get_project(project_id)
    if not project_data:
        raise HTTPException(status_code=404, detail="Project not found")

    if settings.STAGING_MAX_ROOMS_PER_PROJECT > 0 and len(images) > settings.STAGING_MAX_ROOMS_PER_PROJECT:
        raise HTTPException(status_code=400, detail=f"Maximum {settings.STAGING_MAX_ROOMS_PER_PROJECT} rooms per project")

    label_list = json.loads(labels) if labels else []
    blob_service = AzureBlobStorageService()
    new_rooms = []

    for idx, image_file in enumerate(images):
        room_id = str(uuid.uuid4())
        label = label_list[idx] if idx < len(label_list) else f"Room {len(project_data.get('rooms', [])) + idx + 1}"

        upload_result = await blob_service.upload_asset(
            file=image_file,
            asset_type="image",
            folder_path=f"staging/{project_id}/originals",
        )

        n_vars = project_data.get("settings", {}).get("variations_per_room", 5)
        variations = [Variation(id=str(uuid.uuid4())).dict() for _ in range(n_vars)]

        room = Room(
            id=room_id,
            label=label,
            original_image_url=upload_result["url"],
            variations=variations,
        )
        new_rooms.append(room)

    existing_rooms = project_data.get("rooms", [])
    existing_rooms.extend([r.dict() for r in new_rooms])
    storage.update_project(project_id, {"rooms": existing_rooms})

    return UploadRoomsResponse(project_id=project_id, rooms_added=len(new_rooms), rooms=new_rooms)


def _sse_event(event_type: str, data: dict) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


@router.post("/projects/{project_id}/generate")
async def generate_project(
    project_id: str,
    storage: StagingStorageService = Depends(get_staging_storage),
    pipeline=Depends(get_staging_pipeline),
):
    project_data = storage.get_project(project_id)
    if not project_data:
        raise HTTPException(status_code=404, detail="Project not found")

    clean = {k: v for k, v in project_data.items() if k != "doc_type" and not k.startswith("_")}
    project = StagingProject(**clean)

    if not project.rooms:
        raise HTTPException(status_code=400, detail="No rooms uploaded yet")

    async def event_stream():
        async for event in pipeline.generate_project(project):
            yield _sse_event(event["type"], event)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/projects/{project_id}/rooms/{room_id}/regenerate")
async def regenerate_room(
    project_id: str,
    room_id: str,
    storage: StagingStorageService = Depends(get_staging_storage),
    pipeline=Depends(get_staging_pipeline),
):
    project_data = storage.get_project(project_id)
    if not project_data:
        raise HTTPException(status_code=404, detail="Project not found")

    clean = {k: v for k, v in project_data.items() if k != "doc_type" and not k.startswith("_")}
    project = StagingProject(**clean)

    room = next((r for r in project.rooms if r.id == room_id), None)
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")

    room.status = ItemStatus.PENDING
    for v in room.variations:
        v.status = ItemStatus.PENDING
        v.image_url = None
        v.error = None

    async def event_stream():
        async for event in pipeline.process_room(project, room, brief_prompts=None):
            yield _sse_event(event["type"], event)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/projects/{project_id}/analyze")
async def analyze_project_images(
    project_id: str,
    storage: StagingStorageService = Depends(get_staging_storage),
):
    """Analyze all uploaded images in the project."""
    import asyncio
    import base64

    project_data = storage.get_project(project_id)
    if not project_data:
        raise HTTPException(status_code=404, detail="Project not found")

    rooms = project_data.get("rooms", [])
    if not rooms:
        raise HTTPException(status_code=400, detail="No images uploaded yet")

    analyzer = get_image_analyzer()
    blob_service = AzureBlobStorageService()

    system_msg = (
        "Describe this scene in detail for a design assistant. Include: "
        "existing structures, plants, materials, colors, spatial layout, "
        "and identifiable zones where items could be added. "
        "Return JSON with keys: description (string), features (list of strings), "
        "zones (list of strings identifying areas suitable for placing new objects)."
    )

    async def analyze_one(room: dict) -> dict:
        url = room["original_image_url"]
        blob_name = "/".join(url.split("/")[-2:])
        image_bytes = await blob_service.get_asset_content(
            blob_name=blob_name,
            container_name=settings.AZURE_BLOB_IMAGE_CONTAINER,
        )
        image_b64 = base64.b64encode(image_bytes).decode("utf-8") if isinstance(image_bytes, bytes) else image_bytes
        result = await analyzer.async_image_chat(image_base64=image_b64, system_message=system_msg)
        return {
            "room_id": room["id"],
            "description": result.get("description", ""),
            "features": result.get("features", []),
            "zones": result.get("zones", []),
        }

    analyses = await asyncio.gather(*[analyze_one(r) for r in rooms], return_exceptions=True)
    valid_analyses = [a for a in analyses if isinstance(a, dict)]

    storage.update_project(project_id, {"analyses": valid_analyses})

    return {"analyses": valid_analyses}


@router.post("/projects/{project_id}/chat", response_model=ChatResponse)
async def chat_with_project(
    project_id: str,
    request: ChatRequest,
    storage: StagingStorageService = Depends(get_staging_storage),
):
    """Conversational AI endpoint for the Design Session."""
    from backend.core import async_llm_client
    from backend.core.design_chat import DesignChatService

    project_data = storage.get_project(project_id)
    if not project_data:
        raise HTTPException(status_code=404, detail="Project not found")

    raw_analyses = project_data.get("analyses", [])
    analyses = [ImageAnalysis(**a) for a in raw_analyses]

    service = DesignChatService(
        async_llm_client=async_llm_client,
        llm_deployment=settings.LLM_DEPLOYMENT,
        image_analyses=analyses,
    )

    return await service.chat(
        message=request.message,
        conversation_history=request.conversation_history,
        focused_image_id=request.focused_image_id,
    )


@router.post("/projects/{project_id}/brief")
async def generate_brief(
    project_id: str,
    request: GenerateBriefRequest = None,
    storage: StagingStorageService = Depends(get_staging_storage),
):
    """Generate a structured Design Brief from the conversation."""
    from backend.core import async_llm_client
    from backend.core.brief_generator import BriefGeneratorService

    project_data = storage.get_project(project_id)
    if not project_data:
        raise HTTPException(status_code=404, detail="Project not found")

    raw_analyses = project_data.get("analyses", [])
    analyses = [ImageAnalysis(**a) for a in raw_analyses]

    conversation_history = []
    if request and request.conversation_history:
        conversation_history = request.conversation_history

    service = BriefGeneratorService(
        async_llm_client=async_llm_client,
        llm_deployment=settings.LLM_DEPLOYMENT,
    )

    brief = await service.generate_brief(
        conversation_history=conversation_history,
        image_analyses=analyses,
    )

    brief_dict = brief.dict()
    storage.update_project(project_id, {"design_brief": brief_dict})

    return {"brief": brief_dict}


@router.put("/projects/{project_id}/brief")
async def update_brief(
    project_id: str,
    brief: DesignBrief,
    storage: StagingStorageService = Depends(get_staging_storage),
):
    """Save user edits to the Design Brief."""
    project_data = storage.get_project(project_id)
    if not project_data:
        raise HTTPException(status_code=404, detail="Project not found")

    brief_dict = brief.dict()
    storage.update_project(project_id, {"design_brief": brief_dict})

    return {"brief": brief_dict}