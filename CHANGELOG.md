## Visionary Lab Changelog

<a name="0.2.0"></a>
# 0.2.0 (2026-04-29)

*Features*
* Parallel processing for staging projects: rooms now generate concurrently via an `asyncio.Queue` worker pool gated by `STAGING_CONCURRENT_ROOMS` (default 3). A 5-room project completes in ~2 minutes vs ~5 minutes sequentially.
* Automatic retry with exponential backoff on Azure 429 rate-limit responses (`IMAGE_GEN_RETRY_ATTEMPTS=3`, `IMAGE_GEN_RETRY_BASE_DELAY=2.0`); honours the `Retry-After` header when present.

*Reliability*
* Worker tasks are cancelled cleanly on consumer disconnect (no orphaned room generation continuing after the SSE stream closes).
* Image edit retries no longer leak file handles when the API call fails mid-stream.

---

<a name="0.1.0"></a>
# 0.1.0 (2024-04-28)

*Features*
* Upload, view, and manage images in folders using Azure Blob Storage
* Generate images using Azure OpenAI or OpenAI GPT-Image-1
* Organize images in a gallery with folder support
* Move, delete, and update metadata for images
* Secure direct access to images via SAS tokens
* Modern Next.js frontend with settings and status pages
* FastAPI backend with modular endpoints and async support

---

This project is an AI-powered content lab for generating, storing, and managing images. It provides a modern web interface for uploading, organizing, and analyzing images, with secure Azure Blob Storage integration and support for advanced image generation models.
