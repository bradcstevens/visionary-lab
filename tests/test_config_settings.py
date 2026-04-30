from backend.core.config import Settings


def test_retry_settings_have_defaults():
    s = Settings()
    assert s.IMAGE_GEN_RETRY_ATTEMPTS == 5
    assert s.IMAGE_GEN_RETRY_BASE_DELAY == 2.0
    assert s.IMAGE_GEN_RETRY_MAX_TOTAL_WAIT == 120.0


def test_image_gen_max_concurrent_default():
    """Global image-call cap default is 3 (parallel-processing PRD)."""
    s = Settings()
    assert s.IMAGE_GEN_MAX_CONCURRENT == 3
