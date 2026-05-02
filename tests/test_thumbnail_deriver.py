"""Tests for ThumbnailDeriver — pure transform + sibling-blob upload."""
import io
from unittest.mock import MagicMock

import pytest
from PIL import Image

from backend.core.thumbnail_deriver import ThumbnailDeriver


def _png_bytes(width: int, height: int, color=(255, 0, 0)) -> bytes:
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestDerivePureTransform:
    def test_derive_returns_two_webp_byte_strings(self):
        src = _png_bytes(2000, 1500)
        thumb, md = ThumbnailDeriver.derive(src)
        assert isinstance(thumb, bytes) and isinstance(md, bytes)
        assert len(thumb) > 0 and len(md) > 0
        with Image.open(io.BytesIO(thumb)) as img:
            assert img.format == "WEBP"
        with Image.open(io.BytesIO(md)) as img:
            assert img.format == "WEBP"

    def test_thumb_max_edge_is_512(self):
        src = _png_bytes(2000, 1500)
        thumb, _ = ThumbnailDeriver.derive(src)
        with Image.open(io.BytesIO(thumb)) as img:
            assert max(img.size) == 512
            # aspect ratio preserved (2000:1500 = 4:3 -> 512:384)
            assert img.size == (512, 384)

    def test_md_max_edge_is_1024(self):
        src = _png_bytes(2000, 1500)
        _, md = ThumbnailDeriver.derive(src)
        with Image.open(io.BytesIO(md)) as img:
            assert max(img.size) == 1024
            assert img.size == (1024, 768)

    def test_portrait_aspect_ratio_preserved(self):
        src = _png_bytes(800, 1600)
        thumb, md = ThumbnailDeriver.derive(src)
        with Image.open(io.BytesIO(thumb)) as img:
            assert img.size == (256, 512)
        with Image.open(io.BytesIO(md)) as img:
            assert img.size == (512, 1024)

    def test_source_smaller_than_max_edge_not_upscaled(self):
        src = _png_bytes(300, 200)
        thumb, md = ThumbnailDeriver.derive(src)
        with Image.open(io.BytesIO(thumb)) as img:
            assert img.size == (300, 200)
        with Image.open(io.BytesIO(md)) as img:
            assert img.size == (300, 200)

    def test_rgba_source_handled(self):
        img = Image.new("RGBA", (1000, 1000), (255, 0, 0, 128))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        thumb, md = ThumbnailDeriver.derive(buf.getvalue())
        with Image.open(io.BytesIO(thumb)) as t:
            assert t.format == "WEBP"
            assert t.size == (512, 512)


class TestSiblingBlobNames:
    def test_basic_png(self):
        thumb, md = ThumbnailDeriver.sibling_blob_names(
            "staging/proj/variations/room-0/abc.png"
        )
        assert thumb == "staging/proj/variations/room-0/abc.thumb.webp"
        assert md == "staging/proj/variations/room-0/abc.md.webp"

    def test_multi_dot_filename(self):
        thumb, md = ThumbnailDeriver.sibling_blob_names(
            "staging/proj/variations/room-0/img.v2.png"
        )
        assert thumb == "staging/proj/variations/room-0/img.v2.thumb.webp"
        assert md == "staging/proj/variations/room-0/img.v2.md.webp"

    def test_no_extension(self):
        thumb, md = ThumbnailDeriver.sibling_blob_names("staging/proj/abc")
        assert thumb == "staging/proj/abc.thumb.webp"
        assert md == "staging/proj/abc.md.webp"

    def test_dot_in_directory_only(self):
        # A '.' in a parent path component must NOT be treated as the file ext.
        thumb, md = ThumbnailDeriver.sibling_blob_names("staging/v1.0/proj/abc")
        assert thumb == "staging/v1.0/proj/abc.thumb.webp"
        assert md == "staging/v1.0/proj/abc.md.webp"


class TestDeriveAndUpload:
    @pytest.mark.asyncio
    async def test_uploads_both_variants_and_returns_urls(self):
        src = _png_bytes(2000, 1500)

        # Mock blob_service surface used by the deriver.
        thumb_blob = MagicMock()
        thumb_blob.url = (
            "https://acct.blob.core.windows.net/images/"
            "staging/proj/variations/room-0/abc.thumb.webp"
        )
        md_blob = MagicMock()
        md_blob.url = (
            "https://acct.blob.core.windows.net/images/"
            "staging/proj/variations/room-0/abc.md.webp"
        )

        def _get_blob_client(name):
            return thumb_blob if "thumb" in name else md_blob

        container_client = MagicMock()
        container_client.get_blob_client.side_effect = _get_blob_client

        blob_service = MagicMock()
        blob_service.blob_service_client.get_container_client.return_value = (
            container_client
        )

        deriver = ThumbnailDeriver(blob_service=blob_service)
        thumb_url, md_url = await deriver.derive_and_upload(
            image_bytes=src,
            original_blob_name="staging/proj/variations/room-0/abc.png",
            container_name="images",
        )

        assert thumb_url.endswith(".thumb.webp")
        assert md_url.endswith(".md.webp")
        # Both blobs uploaded once, content-type webp.
        assert thumb_blob.upload_blob.call_count == 1
        assert md_blob.upload_blob.call_count == 1
        for blob in (thumb_blob, md_blob):
            kwargs = blob.upload_blob.call_args.kwargs
            assert kwargs.get("overwrite") is True
            assert kwargs["content_settings"].content_type == "image/webp"

    @pytest.mark.asyncio
    async def test_extracts_blob_name_from_url(self):
        src = _png_bytes(800, 600)

        captured = []

        def _get_blob_client(name):
            captured.append(name)
            m = MagicMock()
            m.url = f"https://acct.blob.core.windows.net/images/{name}"
            return m

        container_client = MagicMock()
        container_client.get_blob_client.side_effect = _get_blob_client
        blob_service = MagicMock()
        blob_service.blob_service_client.get_container_client.return_value = (
            container_client
        )

        deriver = ThumbnailDeriver(blob_service=blob_service)
        await deriver.derive_and_upload_from_url(
            image_bytes=src,
            original_url=(
                "https://acct.blob.core.windows.net/images/"
                "staging/proj/variations/room-0/abc.png"
            ),
            container_name="images",
        )
        assert any(n.endswith(".thumb.webp") for n in captured)
        assert any(n.endswith(".md.webp") for n in captured)
        assert all(n.startswith("staging/proj/variations/room-0/abc.") for n in captured)


class TestPipelineIntegration:
    """Issue 010 AC: staging_pipeline.py invokes the deriver at the tail of
    each variation job before marking succeeded; Variation gains thumb_url,
    md_url, revision."""

    @pytest.mark.asyncio
    async def test_process_room_populates_thumb_md_revision(self):
        from unittest.mock import AsyncMock, MagicMock
        from backend.core.staging_pipeline import StagingPipeline
        from backend.models.images import (
            ImageGenerationResponse, ImageSaveResponse,
            ImagePipelineResponse, PipelineStepResult,
        )
        from backend.models.staging import (
            ItemStatus, Room, StagingProject, StagingSettings, Variation,
        )

        rooms = [Room(
            id="room-0", label="R1",
            original_image_url="https://acct.blob.core.windows.net/images/staging/p/originals/o.png",
            variations=[Variation(id="v-0")],
        )]
        project = StagingProject(
            id="p", name="P", prompt="x",
            settings=StagingSettings(variations_per_room=1), rooms=rooms,
        )
        saved_url = "https://acct.blob.core.windows.net/images/staging/p/variations/room-0/g.png"
        gen = ImageGenerationResponse(success=True, message="ok",
            imgen_model_response={"data": [{"b64_json": "AAAA"}]})
        save = ImageSaveResponse(success=True, message="ok",
            saved_images=[{"url": saved_url, "blob_name": "staging/p/variations/room-0/g.png",
                           "container": "images", "file_id": "f", "size": 1, "content_type": "image/png"}],
            total_saved=1)
        pipe_resp = ImagePipelineResponse(success=True, message="ok",
            steps=[PipelineStepResult(step="edit", success=True),
                   PipelineStepResult(step="save", success=True)],
            generation=gen, save=save)

        mock_pipeline = AsyncMock()
        mock_pipeline.process_pipeline.return_value = pipe_resp

        mock_blob = MagicMock()
        mock_blob.get_asset_content.return_value = (_png_bytes(800, 600), "image/png")
        # Wire blob_service_client surface used by ThumbnailDeriver._upload.
        thumb_client = MagicMock(); thumb_client.url = "https://acct.blob.core.windows.net/images/staging/p/variations/room-0/g.thumb.webp"
        md_client = MagicMock(); md_client.url = "https://acct.blob.core.windows.net/images/staging/p/variations/room-0/g.md.webp"
        cont_client = MagicMock()
        cont_client.get_blob_client.side_effect = lambda n: thumb_client if "thumb" in n else md_client
        mock_blob.blob_service_client.get_container_client.return_value = cont_client

        mock_storage = MagicMock(); mock_storage.update_project = MagicMock()
        mock_analyzer = AsyncMock()
        mock_analyzer.async_image_chat.return_value = {"description": "x", "features": []}
        mock_llm = AsyncMock()
        mock_llm.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content='["adapted"]'))])

        staging = StagingPipeline(
            async_llm_client=mock_llm, llm_deployment="gpt-4o",
            image_analyzer=mock_analyzer, image_pipeline=mock_pipeline,
            storage_service=mock_storage, blob_service=mock_blob,
        )

        async for _ in staging.process_room(project, rooms[0]):
            pass

        v = rooms[0].variations[0]
        assert v.status == ItemStatus.COMPLETED
        assert v.image_url == saved_url
        assert v.thumb_url and v.thumb_url.endswith(".thumb.webp")
        assert v.md_url and v.md_url.endswith(".md.webp")
        assert v.revision == 1

    @pytest.mark.asyncio
    async def test_derive_failure_does_not_fail_variation(self):
        """Best-effort: if deriver raises, the variation still completes."""
        from unittest.mock import AsyncMock, MagicMock
        from backend.core.staging_pipeline import StagingPipeline
        from backend.models.images import (
            ImageGenerationResponse, ImageSaveResponse,
            ImagePipelineResponse, PipelineStepResult,
        )
        from backend.models.staging import (
            ItemStatus, Room, StagingProject, StagingSettings, Variation,
        )

        rooms = [Room(
            id="room-0", label="R1",
            original_image_url="https://acct.blob.core.windows.net/images/staging/p/originals/o.png",
            variations=[Variation(id="v-0")],
        )]
        project = StagingProject(id="p", name="P", prompt="x",
            settings=StagingSettings(variations_per_room=1), rooms=rooms)
        saved_url = "https://acct.blob.core.windows.net/images/staging/p/variations/room-0/g.png"
        gen = ImageGenerationResponse(success=True, message="ok",
            imgen_model_response={"data": [{"b64_json": "AAAA"}]})
        save = ImageSaveResponse(success=True, message="ok",
            saved_images=[{"url": saved_url, "blob_name": "x", "container": "images",
                           "file_id": "f", "size": 1, "content_type": "image/png"}],
            total_saved=1)
        pipe_resp = ImagePipelineResponse(success=True, message="ok",
            steps=[PipelineStepResult(step="edit", success=True),
                   PipelineStepResult(step="save", success=True)],
            generation=gen, save=save)

        mock_pipeline = AsyncMock(); mock_pipeline.process_pipeline.return_value = pipe_resp
        mock_blob = MagicMock()
        mock_blob.get_asset_content.return_value = (_png_bytes(800, 600), "image/png")
        # Make the deriver's upload step fail so we exercise the
        # best-effort branch without breaking the source-image read.
        mock_blob.blob_service_client.get_container_client.side_effect = (
            RuntimeError("simulated upload failure")
        )

        mock_storage = MagicMock()
        mock_analyzer = AsyncMock()
        mock_analyzer.async_image_chat.return_value = {"description": "x", "features": []}
        mock_llm = AsyncMock()
        mock_llm.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content='["adapted"]'))])

        staging = StagingPipeline(
            async_llm_client=mock_llm, llm_deployment="gpt-4o",
            image_analyzer=mock_analyzer, image_pipeline=mock_pipeline,
            storage_service=mock_storage, blob_service=mock_blob,
        )
        async for _ in staging.process_room(project, rooms[0]):
            pass

        v = rooms[0].variations[0]
        assert v.status == ItemStatus.COMPLETED
        assert v.image_url == saved_url
        assert v.thumb_url is None
        assert v.md_url is None
        assert v.revision == 1
