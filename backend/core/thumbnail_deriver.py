"""ThumbnailDeriver — pure-transform plus blob-upload helper for sibling
``thumb.webp`` (512px max-edge, q70) and ``md.webp`` (1024px max-edge, q80)
variants of every generated variation image.

Issue 010 of the image-pipeline-and-project-ux-overhaul PRD. Deep module:
the public surface is three methods, the rest is internal.

Sibling-naming contract (pinned by tests):
    staging/proj/variations/room-0/abc.png ->
        staging/proj/variations/room-0/abc.thumb.webp
        staging/proj/variations/room-0/abc.md.webp
"""
from __future__ import annotations

import asyncio
import io
import logging
import os
from typing import Tuple

from azure.storage.blob import ContentSettings
from PIL import Image

logger = logging.getLogger(__name__)


THUMB_MAX_EDGE = 512
THUMB_QUALITY = 70
MD_MAX_EDGE = 1024
MD_QUALITY = 80
WEBP_CONTENT_TYPE = "image/webp"


class ThumbnailDeriver:
    """Produce ``thumb.webp`` + ``md.webp`` siblings for a source image.

    ``derive`` is a pure transform (bytes in, bytes out) so it is trivially
    unit-testable. ``derive_and_upload`` composes it with a blob upload via
    the injected ``blob_service`` (an ``AzureBlobStorageService`` instance).
    """

    def __init__(self, blob_service):
        self.blob_service = blob_service

    # -- pure transform -------------------------------------------------

    @staticmethod
    def derive(image_bytes: bytes) -> Tuple[bytes, bytes]:
        """Return ``(thumb_webp_bytes, md_webp_bytes)`` for the given source.

        Pure: no I/O, no network, safe to call from any context (and from
        a worker thread via ``asyncio.to_thread``).
        """
        thumb = ThumbnailDeriver._resize_to_webp(
            image_bytes, max_edge=THUMB_MAX_EDGE, quality=THUMB_QUALITY
        )
        md = ThumbnailDeriver._resize_to_webp(
            image_bytes, max_edge=MD_MAX_EDGE, quality=MD_QUALITY
        )
        return thumb, md

    @staticmethod
    def _resize_to_webp(image_bytes: bytes, *, max_edge: int, quality: int) -> bytes:
        with Image.open(io.BytesIO(image_bytes)) as img:
            # WebP supports RGBA, but normalising to RGB matches the rest of
            # the pipeline (PNG sources from gpt-image-1 are opaque) and keeps
            # output sizes small. Flatten alpha onto a white canvas if present
            # so the visual result matches what the user sees in the app.
            if img.mode in ("RGBA", "LA"):
                bg = Image.new("RGB", img.size, (255, 255, 255))
                bg.paste(img, mask=img.split()[-1])
                img = bg
            elif img.mode != "RGB":
                img = img.convert("RGB")
            # Image.thumbnail mutates in place, preserves aspect ratio, and
            # NEVER upscales — small sources stay at their native size.
            img.thumbnail(
                (max_edge, max_edge), Image.Resampling.LANCZOS
            )
            buf = io.BytesIO()
            img.save(buf, format="WEBP", quality=quality, method=4)
            return buf.getvalue()

    # -- naming ---------------------------------------------------------

    @staticmethod
    def sibling_blob_names(original_blob_name: str) -> Tuple[str, str]:
        """Compute the two sibling blob names for a source blob.

        Strips the trailing extension from the file portion only — a ``.``
        in a parent directory must NOT be treated as the file extension.
        Files with no extension get the suffix appended directly.
        """
        dirname, basename = os.path.split(original_blob_name)
        stem, ext = os.path.splitext(basename)
        # If there's no extension, splitext returns ('abc', '').
        new_thumb = f"{stem}.thumb.webp"
        new_md = f"{stem}.md.webp"
        if dirname:
            return f"{dirname}/{new_thumb}", f"{dirname}/{new_md}"
        return new_thumb, new_md

    @staticmethod
    def _blob_name_from_url(blob_url: str) -> str:
        """Extract the blob name (path under the container) from a full URL.

        Mirrors ``StagingPipeline._extract_blob_name`` so this module stays
        self-contained.
        """
        parts = blob_url.split("/")
        try:
            net_idx = next(i for i, p in enumerate(parts) if p.endswith(".net"))
            return "/".join(parts[net_idx + 2:])  # skip container name
        except (StopIteration, IndexError):
            for container in ("images", "videos"):
                if f"/{container}/" in blob_url:
                    return blob_url.split(f"/{container}/")[1]
            return "/".join(parts[-2:])

    # -- upload ---------------------------------------------------------

    async def derive_and_upload(
        self,
        *,
        image_bytes: bytes,
        original_blob_name: str,
        container_name: str,
    ) -> Tuple[str, str]:
        """Derive both variants and upload them next to ``original_blob_name``.

        Returns ``(thumb_url, md_url)``. Raises whatever the SDK raises on
        upload failure — callers decide whether a missing thumbnail should
        fail the parent job (issue 010 wires it best-effort: log + skip).
        """
        thumb_bytes, md_bytes = await asyncio.to_thread(self.derive, image_bytes)
        thumb_name, md_name = self.sibling_blob_names(original_blob_name)
        thumb_url = await asyncio.to_thread(
            self._upload, thumb_name, thumb_bytes, container_name
        )
        md_url = await asyncio.to_thread(
            self._upload, md_name, md_bytes, container_name
        )
        return thumb_url, md_url

    async def derive_and_upload_from_url(
        self,
        *,
        image_bytes: bytes,
        original_url: str,
        container_name: str,
    ) -> Tuple[str, str]:
        """Convenience wrapper: take a full blob URL, extract the name, upload."""
        return await self.derive_and_upload(
            image_bytes=image_bytes,
            original_blob_name=self._blob_name_from_url(original_url),
            container_name=container_name,
        )

    def _upload(self, blob_name: str, data: bytes, container_name: str) -> str:
        container_client = self.blob_service.blob_service_client.get_container_client(
            container_name
        )
        blob_client = container_client.get_blob_client(blob_name)
        blob_client.upload_blob(
            data=data,
            content_settings=ContentSettings(content_type=WEBP_CONTENT_TYPE),
            overwrite=True,
        )
        return blob_client.url
