"""Shared retry utility for image-gen and LLM call sites.

This module provides ``call_with_retry`` — a single entry point used by every
image-generation and LLM call path that needs typed retry, full-jitter
exponential backoff, and a cumulative-wait cap. It is the deep module that
later parallel-processing slices (global image semaphore, hybrid parallelism)
depend on.

Behavior contract (see ``issues/parallel-processing/003-shared-retry-utility.md``):

* The util invokes ``coro_fn()`` itself on each attempt — the caller passes a
  closure / lambda, not a pre-awaited coroutine, so the util controls the
  attempt loop.
* When ``semaphore`` is supplied, the util acquires it once *before* the first
  attempt and releases it *after* the final outcome. The slot is held across
  retry sleeps as documented backpressure (not a bug).
* Retry is restricted to a typed allowlist: rate-limit (429), connection
  errors / request timeouts, and HTTP 5xx in {500, 502, 503, 504}. Anything
  else (4xx other than 429, content-policy errors, ``ValueError``, etc.)
  raises immediately.
* ``asyncio.CancelledError`` and other ``BaseException`` subclasses propagate
  immediately and are never retried.
* Backoff:
    - Exponential path: ``random.uniform(0, base_delay * 2 ** attempt)``
      (full jitter — concurrent failures don't retry in lockstep).
    - Rate-limit path: honors ``Retry-After`` (numeric seconds) when present,
      with a small uniform jitter on top; otherwise the exponential path.
* Cumulative wait cap: if the *next* attempt's projected sleep would push
  cumulative wait over ``max_total_wait``, the underlying error is raised
  instead of sleeping.
* Each retry emits one structured ``INFO`` log line with fields
  ``event=retry``, ``reason``, ``attempt`` (1-indexed), ``delay_seconds``,
  and ``model``. Fields appear both in the formatted message (the contract)
  and on the log record via ``extra=`` (best-effort, for log-aggregation
  consumers).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import random
from typing import Any, Awaitable, Callable, Optional

import httpx
import openai
import requests

logger = logging.getLogger(__name__)

_RETRYABLE_5XX: frozenset[int] = frozenset({500, 502, 503, 504})


def _categorize_error(exc: Exception) -> Optional[str]:
    """Return the retry category for ``exc``, or ``None`` if not retryable.

    Categories:
        ``rate_limit``    — 429 from any client
        ``timeout``       — request/read/write timeouts from any client
        ``connection``    — network-level connection failures from any client
        ``server_error``  — HTTP 5xx in the configured allowlist

    Order matters: rate_limit is checked before generic status_error so that a
    429 (which inherits from APIStatusError in openai) is categorized as
    rate_limit.
    """
    # OpenAI SDK exceptions
    if isinstance(exc, openai.RateLimitError):
        return "rate_limit"
    if isinstance(exc, openai.APITimeoutError):
        return "timeout"
    if isinstance(exc, openai.APIConnectionError):
        return "connection"
    if isinstance(exc, openai.APIStatusError):
        status = getattr(exc, "status_code", None)
        if status in _RETRYABLE_5XX:
            return "server_error"
        return None

    # httpx exceptions (defensive — openai usually wraps these, but raw httpx
    # can leak from non-openai paths or custom client wiring).
    if isinstance(exc, httpx.TimeoutException):
        return "timeout"
    if isinstance(exc, httpx.ConnectError):
        return "connection"

    # requests exceptions (Azure image-edit path uses requests directly).
    # NOTE: requests.HTTPError inherits from requests.RequestException, NOT
    # from ConnectionError/Timeout, so the order below works.
    if isinstance(exc, requests.HTTPError):
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None) if response is not None else None
        if status == 429:
            return "rate_limit"
        if status in _RETRYABLE_5XX:
            return "server_error"
        return None
    if isinstance(exc, requests.Timeout):
        return "timeout"
    if isinstance(exc, requests.ConnectionError):
        return "connection"

    return None


def _retry_after_seconds(exc: Exception) -> Optional[float]:
    """Extract a numeric ``Retry-After`` header value from ``exc``, if present.

    Supports openai.RateLimitError (.response.headers) and
    requests.HTTPError(429) (.response.headers). Only the integer/float seconds
    form is honored — HTTP-date form is not parsed (rare in practice and not
    required by the issue contract).
    """
    response = getattr(exc, "response", None)
    if response is None:
        return None
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    raw = headers.get("retry-after") if hasattr(headers, "get") else None
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if value >= 0 else None


@contextlib.asynccontextmanager
async def _maybe_acquire(semaphore: Optional[asyncio.Semaphore]):
    """Acquire ``semaphore`` for the duration of the ``async with`` block.

    If ``semaphore`` is None, this is a no-op context manager.
    """
    if semaphore is None:
        yield
    else:
        async with semaphore:
            yield


async def call_with_retry(
    coro_fn: Callable[[], Awaitable[Any]],
    *,
    semaphore: Optional[asyncio.Semaphore],
    model: str,
    attempts: int,
    base_delay: float,
    max_total_wait: float,
) -> Any:
    """Invoke ``coro_fn()`` with typed retries, full-jitter backoff, wait cap.

    Args:
        coro_fn: Zero-arg async callable. Called freshly on each attempt — the
            caller passes a closure, not a pre-awaited coroutine.
        semaphore: Optional ``asyncio.Semaphore``-compatible async context
            manager. When provided, acquired once before the first attempt and
            released once after the final outcome. ``None`` is acceptable for
            call sites that don't need rate-limit backpressure (LLM callers).
        model: Identifier of the model being called. Used only for log lines.
        attempts: Maximum number of attempts (must be >= 1). E.g. ``5`` means
            the call is tried up to 5 times.
        base_delay: Base for exponential backoff in seconds.
        max_total_wait: Cumulative-wait cap in seconds. If the next attempt's
            projected sleep would push total wait over this cap, the underlying
            error is raised instead of sleeping.

    Returns:
        Whatever ``coro_fn()`` returns on success.

    Raises:
        ValueError: when ``attempts < 1``.
        BaseException: the underlying error from ``coro_fn`` after retries are
            exhausted, or if the error is not in the retry allowlist, or if
            the cumulative-wait cap is reached.
    """
    if attempts < 1:
        raise ValueError(f"attempts must be >= 1, got {attempts}")

    async with _maybe_acquire(semaphore):
        cumulative_wait = 0.0

        for attempt in range(attempts):
            try:
                return await coro_fn()
            except Exception as exc:  # noqa: BLE001 — typed allowlist below
                category = _categorize_error(exc)
                if category is None:
                    raise
                # Last attempt: out of budget, propagate.
                if attempt >= attempts - 1:
                    raise

                # Compute next delay with full jitter.
                if category == "rate_limit":
                    retry_after = _retry_after_seconds(exc)
                    if retry_after is not None:
                        # Add a small uniform jitter on top of Retry-After to
                        # decorrelate concurrent retriers.
                        jitter_max = min(retry_after * 0.1, 1.0)
                        delay = retry_after + random.uniform(0, jitter_max)
                    else:
                        delay = random.uniform(0, base_delay * (2 ** attempt))
                else:
                    delay = random.uniform(0, base_delay * (2 ** attempt))

                # Cumulative-wait cap: if the projected sleep would push total
                # over the cap, raise the underlying error rather than sleep.
                if cumulative_wait + delay > max_total_wait:
                    raise

                logger.info(
                    "retry event=retry reason=%s attempt=%d "
                    "delay_seconds=%.3f model=%s",
                    category,
                    attempt + 1,
                    delay,
                    model,
                    extra={
                        "event": "retry",
                        "reason": category,
                        "attempt": attempt + 1,
                        "delay_seconds": delay,
                        "model": model,
                    },
                )
                await asyncio.sleep(delay)
                cumulative_wait += delay

        # Unreachable: the final-attempt branch above always raises.
        raise RuntimeError("call_with_retry: control reached end of loop")  # pragma: no cover
