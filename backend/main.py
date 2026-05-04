# Configure logging first, before other imports
from .core.logging_config import setup_logging
setup_logging()

from contextlib import asynccontextmanager  # noqa: E402

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
import os  # noqa: E402
import uvicorn  # noqa: E402
from .core.config import settings  # noqa: E402
from .core.embedded_worker import EmbeddedWorker  # noqa: E402
from .core.worker_factory import build_worker  # noqa: E402
from .api.endpoints import images, metadata_router, videos, gallery, env, staging  # noqa: E402


# Create directories if they don't exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.IMAGE_DIR, exist_ok=True)
os.makedirs(settings.VIDEO_DIR, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Auto-start an in-process JobWorker for development.

    The embedded worker checks ``AUTO_START_WORKER`` and ``ROLE`` —
    when running locally (``uv run fastapi dev``) the policy starts a
    worker so the queue drains naturally without a second process.
    Production sets ``AUTO_START_WORKER=False`` on the API container
    via ``infra/modules/containerApp.bicep``, so the policy short-
    circuits and only the dedicated worker container processes jobs.

    The worker is stashed on ``app.state`` (not a module-level global)
    so each app instance owns its own lifecycle — important for
    repeated TestClient lifespan usage in tests.
    """
    embedded = EmbeddedWorker()
    app.state.embedded_worker = embedded
    await embedded.start(build_worker)
    try:
        yield
    finally:
        await embedded.stop()


app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan,
)

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this with proper origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Include routers
app.include_router(images.router, prefix=f"{settings.API_V1_STR}/images", tags=["images"])
app.include_router(videos.router, prefix=f"{settings.API_V1_STR}/videos", tags=["videos"])
app.include_router(gallery.router, prefix=f"{settings.API_V1_STR}/gallery", tags=["gallery"])
app.include_router(metadata_router.router, prefix=f"{settings.API_V1_STR}/metadata", tags=["metadata"])
app.include_router(env.router, prefix=f"{settings.API_V1_STR}", tags=["env"])
app.include_router(staging.router, prefix=f"{settings.API_V1_STR}/staging", tags=["staging"])


@app.get("/")
def read_root():
    return {"message": "Welcome to AI Content Lab API"}


@app.get(f"{settings.API_V1_STR}/health")
def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
