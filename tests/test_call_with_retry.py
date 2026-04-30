"""Tests for the shared retry utility in backend/core/retry.py."""
import asyncio
import logging
from unittest.mock import AsyncMock, patch

import httpx
import openai
import pytest
import requests
from httpx import Request, Response
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    BadRequestError,
    RateLimitError,
)

from backend.core.retry import call_with_retry


# ---------------------------------------------------------------------------
# Error fixtures
# ---------------------------------------------------------------------------


def _openai_request_kwargs(method: str = "POST", url: str = "https://example.com"):
    """Build a minimal httpx.Request kwarg dict for openai exception construction."""
    return {"method": method, "url": url}


def _openai_rate_limit_error(retry_after: str | None = None) -> RateLimitError:
    headers = {"retry-after": retry_after} if retry_after else {}
    response = Response(
        status_code=429,
        headers=headers,
        request=Request(**_openai_request_kwargs()),
    )
    return RateLimitError(message="429", response=response, body=None)


def _openai_status_error(status_code: int) -> APIStatusError:
    """Build a generic openai.APIStatusError with the given status code."""
    response = Response(
        status_code=status_code,
        request=Request(**_openai_request_kwargs()),
    )
    if status_code == 400:
        return BadRequestError(message=f"{status_code}", response=response, body=None)
    return APIStatusError(message=f"{status_code}", response=response, body=None)


def _openai_connection_error() -> APIConnectionError:
    return APIConnectionError(request=Request(**_openai_request_kwargs()))


def _openai_timeout_error() -> APITimeoutError:
    return APITimeoutError(request=Request(**_openai_request_kwargs()))


def _httpx_connect_error() -> httpx.ConnectError:
    return httpx.ConnectError("conn refused")


def _httpx_timeout_error() -> httpx.ReadTimeout:
    return httpx.ReadTimeout("timed out")


def _requests_http_error(status_code: int) -> requests.HTTPError:
    response = requests.Response()
    response.status_code = status_code
    err = requests.HTTPError(f"{status_code} error", response=response)
    return err


def _requests_connection_error() -> requests.ConnectionError:
    return requests.ConnectionError("conn refused")


def _requests_timeout_error() -> requests.Timeout:
    return requests.Timeout("timed out")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class SentinelSemaphore:
    """Async context manager that records __aenter__/__aexit__ calls."""

    def __init__(self) -> None:
        self.enter_count = 0
        self.exit_count = 0
        self.held = False

    async def __aenter__(self) -> "SentinelSemaphore":
        self.enter_count += 1
        self.held = True
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        self.exit_count += 1
        self.held = False
        return False


def _make_failing_then_succeeding(failures: list[BaseException], success_value=object()):
    """Return a coro_fn factory that raises each failure once, then returns success."""
    calls: dict[str, int] = {"count": 0}

    async def coro():
        calls["count"] += 1
        idx = calls["count"] - 1
        if idx < len(failures):
            raise failures[idx]
        return success_value

    return coro, calls, success_value


# ---------------------------------------------------------------------------
# Retry-eligible error classes — should retry
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "exc_factory,expected_reason",
    [
        (lambda: _openai_rate_limit_error(retry_after="0.01"), "rate_limit"),
        (lambda: _openai_rate_limit_error(retry_after=None), "rate_limit"),
        (_openai_connection_error, "connection"),
        (_openai_timeout_error, "timeout"),
        (lambda: _openai_status_error(500), "server_error"),
        (lambda: _openai_status_error(502), "server_error"),
        (lambda: _openai_status_error(503), "server_error"),
        (lambda: _openai_status_error(504), "server_error"),
        (_httpx_connect_error, "connection"),
        (_httpx_timeout_error, "timeout"),
        (_requests_connection_error, "connection"),
        (_requests_timeout_error, "timeout"),
        (lambda: _requests_http_error(429), "rate_limit"),
        (lambda: _requests_http_error(500), "server_error"),
        (lambda: _requests_http_error(502), "server_error"),
        (lambda: _requests_http_error(503), "server_error"),
        (lambda: _requests_http_error(504), "server_error"),
    ],
)
@pytest.mark.asyncio
async def test_retries_on_eligible_errors(exc_factory, expected_reason):
    """Each retry-eligible error class triggers one retry, then succeeds."""
    exc = exc_factory()
    coro_fn, calls, sentinel = _make_failing_then_succeeding([exc])

    with patch("backend.core.retry.asyncio.sleep", new_callable=AsyncMock):
        result = await call_with_retry(
            coro_fn,
            semaphore=None,
            model="test-model",
            attempts=5,
            base_delay=0.5,
            max_total_wait=120.0,
        )

    assert result is sentinel
    assert calls["count"] == 2  # one failure + one success


# ---------------------------------------------------------------------------
# Non-retry-eligible error classes — should raise immediately, no sleep
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "exc_factory",
    [
        lambda: _openai_status_error(400),  # BadRequestError
        lambda: _openai_status_error(401),
        lambda: _openai_status_error(403),
        lambda: _openai_status_error(404),
        lambda: _requests_http_error(400),
        lambda: _requests_http_error(403),
        lambda: ValueError("bad input"),
        lambda: TypeError("bad type"),
        lambda: RuntimeError("oops"),
    ],
)
@pytest.mark.asyncio
async def test_does_not_retry_on_ineligible_errors(exc_factory):
    """Non-retry-eligible errors raise on the first occurrence with no sleep."""
    exc = exc_factory()
    call_count = {"n": 0}

    async def coro():
        call_count["n"] += 1
        raise exc

    sleep_mock = AsyncMock()
    with patch("backend.core.retry.asyncio.sleep", sleep_mock):
        with pytest.raises(type(exc)):
            await call_with_retry(
                coro,
                semaphore=None,
                model="test-model",
                attempts=5,
                base_delay=0.5,
                max_total_wait=120.0,
            )

    assert call_count["n"] == 1
    assert sleep_mock.await_count == 0


# ---------------------------------------------------------------------------
# Cancellation propagates
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancelled_error_propagates_immediately():
    """asyncio.CancelledError must propagate without any retry."""
    call_count = {"n": 0}

    async def coro():
        call_count["n"] += 1
        raise asyncio.CancelledError()

    sleep_mock = AsyncMock()
    with patch("backend.core.retry.asyncio.sleep", sleep_mock):
        with pytest.raises(asyncio.CancelledError):
            await call_with_retry(
                coro,
                semaphore=None,
                model="test-model",
                attempts=5,
                base_delay=0.5,
                max_total_wait=120.0,
            )

    assert call_count["n"] == 1
    assert sleep_mock.await_count == 0


# ---------------------------------------------------------------------------
# Jitter: delay falls within documented bound
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_jitter_delay_bound_per_attempt():
    """For non-rate-limit retries, delay must satisfy 0 <= delay <= base_delay * 2 ** attempt."""
    base_delay = 1.5
    delays_observed: list[float] = []

    async def fake_sleep(d: float):
        delays_observed.append(d)

    async def always_fail():
        raise _openai_connection_error()

    with patch("backend.core.retry.asyncio.sleep", side_effect=fake_sleep):
        with pytest.raises(APIConnectionError):
            await call_with_retry(
                always_fail,
                semaphore=None,
                model="test-model",
                attempts=4,
                base_delay=base_delay,
                max_total_wait=10_000.0,  # well above any single jittered sleep
            )

    # 4 attempts → 3 retries → 3 sleeps (attempts 0, 1, 2)
    assert len(delays_observed) == 3
    for attempt_idx, delay in enumerate(delays_observed):
        upper_bound = base_delay * (2 ** attempt_idx)
        assert 0.0 <= delay <= upper_bound, (
            f"attempt {attempt_idx} delay {delay} not in [0, {upper_bound}]"
        )


# ---------------------------------------------------------------------------
# Total-wait cap honored
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_total_wait_cap_honored():
    """If next sleep would push cumulative wait over the cap, raise the underlying error."""
    base_delay = 8.0
    max_total_wait = 10.0

    sleep_calls: list[float] = []

    async def fake_sleep(d: float):
        sleep_calls.append(d)

    async def always_fail():
        raise _openai_connection_error()

    # Force jitter to upper bound for determinism: random.uniform(0, X) -> X
    def fake_uniform(low, high):
        return high

    with patch("backend.core.retry.random.uniform", side_effect=fake_uniform):
        with patch("backend.core.retry.asyncio.sleep", side_effect=fake_sleep):
            with pytest.raises(APIConnectionError):
                await call_with_retry(
                    always_fail,
                    semaphore=None,
                    model="test-model",
                    attempts=5,
                    base_delay=base_delay,
                    max_total_wait=max_total_wait,
                )

    # attempt 0: delay = 8 (cumulative=8 <= 10 → sleep)
    # attempt 1: delay = 16 → cumulative+delay = 24 > 10 → raise (no sleep)
    assert sleep_calls == [8.0]


# ---------------------------------------------------------------------------
# Retry-After header honored on rate-limit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retry_after_header_honored():
    """When Retry-After is present, sleep is at least that value."""
    sleep_calls: list[float] = []

    async def fake_sleep(d: float):
        sleep_calls.append(d)

    exc = _openai_rate_limit_error(retry_after="0.5")
    coro_fn, _calls, _ = _make_failing_then_succeeding([exc])

    with patch("backend.core.retry.asyncio.sleep", side_effect=fake_sleep):
        await call_with_retry(
            coro_fn,
            semaphore=None,
            model="test-model",
            attempts=5,
            base_delay=2.0,
            max_total_wait=120.0,
        )

    assert len(sleep_calls) == 1
    # Retry-After=0.5 plus small uniform jitter (≤ min(0.5*0.1, 1.0) = 0.05)
    assert 0.5 <= sleep_calls[0] <= 0.5 + 0.05 + 1e-9


# ---------------------------------------------------------------------------
# Structured log emitted on each retry
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_structured_log_line_emitted_per_retry(caplog):
    """Each retry emits one INFO log with event/reason/attempt/delay_seconds/model fields."""
    caplog.set_level(logging.INFO, logger="backend.core.retry")

    exc = _openai_connection_error()
    coro_fn, _calls, _ = _make_failing_then_succeeding([exc, exc])

    with patch("backend.core.retry.asyncio.sleep", new_callable=AsyncMock):
        await call_with_retry(
            coro_fn,
            semaphore=None,
            model="my-model",
            attempts=5,
            base_delay=0.1,
            max_total_wait=120.0,
        )

    retry_records = [r for r in caplog.records if getattr(r, "event", None) == "retry"]
    assert len(retry_records) == 2

    # First retry: attempt=1
    assert retry_records[0].reason == "connection"
    assert retry_records[0].attempt == 1
    assert retry_records[0].model == "my-model"
    assert isinstance(retry_records[0].delay_seconds, float)

    # Second retry: attempt=2
    assert retry_records[1].attempt == 2

    # Message itself contains key/value pairs (don't depend on extra= alone)
    assert "event=retry" in retry_records[0].getMessage()
    assert "reason=connection" in retry_records[0].getMessage()
    assert "model=my-model" in retry_records[0].getMessage()


# ---------------------------------------------------------------------------
# Semaphore acquire/release semantics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_semaphore_acquired_before_first_attempt_released_after_success():
    """Sentinel records exactly one __aenter__ before the first call and one __aexit__ after success."""
    sentinel = SentinelSemaphore()
    enter_seen_during_call = {"value": False}

    async def coro():
        enter_seen_during_call["value"] = sentinel.held
        return "ok"

    result = await call_with_retry(
        coro,
        semaphore=sentinel,
        model="test-model",
        attempts=5,
        base_delay=0.1,
        max_total_wait=120.0,
    )

    assert result == "ok"
    assert sentinel.enter_count == 1
    assert sentinel.exit_count == 1
    assert enter_seen_during_call["value"] is True
    assert sentinel.held is False  # released after return


@pytest.mark.asyncio
async def test_semaphore_held_across_retry_sleeps_and_released_on_terminal_failure():
    """Slot must be acquired once at start, held across sleeps, and released after final failure."""
    sentinel = SentinelSemaphore()
    held_during_call: list[bool] = []

    async def always_fail():
        held_during_call.append(sentinel.held)
        raise _openai_connection_error()

    with patch("backend.core.retry.asyncio.sleep", new_callable=AsyncMock):
        with pytest.raises(APIConnectionError):
            await call_with_retry(
                always_fail,
                semaphore=sentinel,
                model="test-model",
                attempts=3,
                base_delay=0.1,
                max_total_wait=120.0,
            )

    assert sentinel.enter_count == 1
    assert sentinel.exit_count == 1
    assert sentinel.held is False  # released after terminal failure
    # Held during every attempt
    assert held_during_call == [True, True, True]


@pytest.mark.asyncio
async def test_semaphore_released_on_non_retryable_error():
    """Slot must be released even when a non-retryable error short-circuits the retry loop."""
    sentinel = SentinelSemaphore()

    async def coro():
        raise ValueError("nope")

    with pytest.raises(ValueError):
        await call_with_retry(
            coro,
            semaphore=sentinel,
            model="test-model",
            attempts=5,
            base_delay=0.1,
            max_total_wait=120.0,
        )

    assert sentinel.enter_count == 1
    assert sentinel.exit_count == 1
    assert sentinel.held is False


@pytest.mark.asyncio
async def test_semaphore_none_works_cleanly():
    """semaphore=None → no slot acquired, normal flow proceeds."""
    async def coro():
        return "value"

    result = await call_with_retry(
        coro,
        semaphore=None,
        model="test-model",
        attempts=5,
        base_delay=0.1,
        max_total_wait=120.0,
    )
    assert result == "value"


# ---------------------------------------------------------------------------
# Real asyncio.Semaphore concurrency bound
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_real_asyncio_semaphore_bounds_concurrency():
    """A real asyncio.Semaphore(2) caps concurrent in-flight calls at 2."""
    sem = asyncio.Semaphore(2)
    in_flight = {"count": 0, "max": 0}
    gate = asyncio.Event()

    async def coro():
        in_flight["count"] += 1
        in_flight["max"] = max(in_flight["max"], in_flight["count"])
        await gate.wait()
        in_flight["count"] -= 1
        return "ok"

    async def runner():
        return await call_with_retry(
            coro,
            semaphore=sem,
            model="m",
            attempts=1,
            base_delay=0.1,
            max_total_wait=10.0,
        )

    tasks = [asyncio.create_task(runner()) for _ in range(5)]
    # Wait for the cap to be saturated
    for _ in range(50):
        await asyncio.sleep(0)
        if in_flight["count"] == 2:
            break
    assert in_flight["count"] == 2
    gate.set()
    results = await asyncio.gather(*tasks)
    assert all(r == "ok" for r in results)
    assert in_flight["max"] == 2


# ---------------------------------------------------------------------------
# Exhausted retries propagate the underlying error
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_exhausted_retries_propagate_underlying_error():
    """After `attempts` failures, the most recent retry-eligible error is raised."""
    async def always_fail():
        raise _openai_connection_error()

    with patch("backend.core.retry.asyncio.sleep", new_callable=AsyncMock):
        with pytest.raises(APIConnectionError):
            await call_with_retry(
                always_fail,
                semaphore=None,
                model="m",
                attempts=3,
                base_delay=0.1,
                max_total_wait=120.0,
            )


# ---------------------------------------------------------------------------
# attempts < 1 rejected
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_zero_attempts_rejected():
    """attempts=0 is invalid and raises ValueError before any call."""
    called = {"n": 0}

    async def coro():
        called["n"] += 1

    with pytest.raises(ValueError):
        await call_with_retry(
            coro,
            semaphore=None,
            model="m",
            attempts=0,
            base_delay=0.1,
            max_total_wait=120.0,
        )

    assert called["n"] == 0


# ---------------------------------------------------------------------------
# coro_fn invoked freshly each attempt (not pre-awaited by caller)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_coro_fn_invoked_fresh_each_attempt():
    """call_with_retry must invoke coro_fn() each attempt, not reuse a single coroutine."""
    invocations = {"count": 0}

    async def coro():
        invocations["count"] += 1
        if invocations["count"] < 3:
            raise _openai_connection_error()
        return "ok"

    with patch("backend.core.retry.asyncio.sleep", new_callable=AsyncMock):
        result = await call_with_retry(
            coro,
            semaphore=None,
            model="m",
            attempts=5,
            base_delay=0.1,
            max_total_wait=120.0,
        )
    assert result == "ok"
    assert invocations["count"] == 3
