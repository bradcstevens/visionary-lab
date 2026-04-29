"""Centralized logging configuration for the backend application.

This module provides a single point of logging configuration. Call setup_logging()
once at application startup before importing other modules.
"""

import logging

from backend.core.config import settings


def setup_logging() -> None:
    """Configure logging for the entire application.
    
    Call this function once at startup, before other modules are imported,
    to ensure all loggers inherit this configuration.
    
    The log level is controlled by the LOG_LEVEL environment variable,
    defaulting to INFO if not specified.
    """
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Suppress verbose Azure SDK logging
    logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(
        logging.WARNING
    )

    # Prevent uvicorn loggers from propagating to the root logger,
    # which would cause every uvicorn message to be printed twice
    # (once by uvicorn's own handler and again by the root handler
    # installed via basicConfig above).
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logging.getLogger(name).propagate = False
