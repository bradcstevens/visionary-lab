from pydantic_settings import BaseSettings
from typing import Optional
from pydantic import Extra, Field, validator


class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Visionary Lab API"

    # Model Provider Configuration
    MODEL_PROVIDER: str = "azure"  # Can be 'azure' or 'openai'

    # AI Foundry endpoint (unified for all AI services via managed identity)
    AI_FOUNDRY_ENDPOINT: Optional[str] = None

    # Model deployment names
    LLM_DEPLOYMENT: Optional[str] = None
    IMAGEGEN_DEPLOYMENT: Optional[str] = None
    IMAGEGEN_15_DEPLOYMENT: Optional[str] = None
    IMAGEGEN_1_MINI_DEPLOYMENT: Optional[str] = None
    FLUX_KONTEXT_DEPLOYMENT: Optional[str] = None
    SORA_DEPLOYMENT: Optional[str] = None
    DEFAULT_IMAGE_MODEL: str = "gpt-image-2"

    # OpenAI API for direct OpenAI usage (non-Azure)
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_ORG_ID: Optional[str] = None
    OPENAI_ORG_VERIFIED: bool = False
    GPT_IMAGE_MAX_TOKENS: int = 150000

    # Azure Blob Storage Settings (managed identity — no keys)
    AZURE_BLOB_SERVICE_URL: Optional[str] = None
    AZURE_STORAGE_ACCOUNT_NAME: Optional[str] = None
    CDN_BLOB_URL: Optional[str] = None

    # Container names
    AZURE_BLOB_IMAGE_CONTAINER: str = "images"
    AZURE_BLOB_VIDEO_CONTAINER: str = "videos"

    # CORS Configuration
    CORS_ALLOWED_ORIGINS: str = Field(
        default="*",
        description="Comma-separated list of allowed CORS origins, or * for all origins"
    )

    # Azure Cosmos DB Settings (managed identity — no keys)
    AZURE_COSMOS_DB_ENDPOINT: Optional[str] = None
    AZURE_COSMOS_DB_KEY: Optional[str] = None
    AZURE_COSMOS_DB_ID: str = "visionarylab"
    AZURE_COSMOS_CONTAINER_ID: str = "metadata"

    # Staging feature
    STAGING_MAX_ROOMS_PER_PROJECT: int = 0  # 0 = unlimited
    STAGING_MAX_VARIATIONS: int = 10
    STAGING_CONCURRENT_ROOMS: int = 10
    STAGING_COSMOS_CONTAINER_ID: str = "staging-projects"
    STAGING_STALE_PROCESSING_MINUTES: int = 5  # Consider processing stale after this many minutes

    # Retry for image-gen and LLM API calls (typed allowlist + full jitter
    # exponential backoff + cumulative-wait cap). See backend/core/retry.py.
    IMAGE_GEN_RETRY_ATTEMPTS: int = 5
    IMAGE_GEN_RETRY_BASE_DELAY: float = 2.0  # seconds; full jitter doubles each attempt
    IMAGE_GEN_RETRY_MAX_TOTAL_WAIT: float = 120.0  # seconds; cumulative wait cap

    # Global concurrency cap for Azure image generate / edit calls. Every
    # image-call code path (staging pipeline, single-variation regen, ad-hoc
    # generate, ad-hoc edit, ad-hoc edit upload) shares this cap via the
    # module-level semaphore in backend/core/image_pipeline.py. This is the
    # rate-limit bound. See parallel-processing PRD § Global image-call cap.
    IMAGE_GEN_MAX_CONCURRENT: int = 3

    # Issue 004 of image-pipeline-and-project-ux-overhaul PRD: enables
    # the persistent async-queue REST surface (``POST /jobs/regenerate``,
    # ``GET /jobs``, ``DELETE /jobs/{id}``) and, in a future slice, the
    # pipeline cutover from inline-await to enqueue. PRD § Feature flags:
    # default true in dev/staging; production flips via azd env var
    # after smoke test.
    FEATURE_ASYNC_QUEUE: bool = True

    # Azure OpenAI API Version
    AOAI_API_VERSION: str = "2025-04-01-preview"

    # File storage paths
    UPLOAD_DIR: str = "./static/uploads"
    IMAGE_DIR: str = "./static/images"
    VIDEO_DIR: str = "./static/videos"

    # Logging Configuration
    LOG_LEVEL: str = "INFO"


    # GPT-Image-1 Default Settings
    GPT_IMAGE_DEFAULT_SIZE: str = "1024x1024"
    GPT_IMAGE_DEFAULT_QUALITY: str = "high"
    GPT_IMAGE_DEFAULT_FORMAT: str = "PNG"
    GPT_IMAGE_ALLOW_TRANSPARENT: bool = True
    GPT_IMAGE_MAX_FILE_SIZE_MB: int = 25

    @validator('CORS_ALLOWED_ORIGINS')
    def validate_cors_origins(cls, v):
        """Validate CORS origins configuration to prevent Azure InvalidXmlNodeValue errors"""
        if v == "*":
            return v
        
        origins = [origin.strip() for origin in v.split(",") if origin.strip()]
        
        if "*" in origins and len(origins) > 1:
            raise ValueError(
                "Cannot mix wildcard '*' with specific origins in CORS configuration. "
                "Use either '*' alone for all origins, or specify individual origins without '*'."
            )
        
        for origin in origins:
            if origin != "*" and not (origin.startswith("http://") or origin.startswith("https://")):
                raise ValueError(f"Invalid origin format: {origin}. Origins must start with http:// or https://")
        
        return v

    class Config:
        env_file = "../.env"
        case_sensitive = True
        extra = Extra.allow


settings = Settings()
