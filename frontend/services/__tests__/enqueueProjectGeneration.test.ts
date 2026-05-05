/**
 * Tests for ``enqueueProjectGeneration`` (issue 008 of the
 * project-generation-async-queue-cutover PRD).
 *
 * The helper is the typed call site for the new ``POST /jobs/generate``
 * endpoint added in issue 006. Tests pin three contracts:
 *
 * 1. Happy-path shape — returns ``{ job_id }`` parsed from the JSON
 *    body, posts to the correct URL with the correct body
 *    (``{ regenerate_all }`` mapped from camelCase ``regenerateAll``).
 * 2. Error surfacing — non-2xx responses reject with a descriptive
 *    Error containing the status + body, matching the convention used
 *    by every other helper in stagingApi.ts.
 * 3. **180s abort timeout via AbortController**, surfacing as a
 *    recognizable ``EnqueueGenerationTimeoutError`` (NOT a generic
 *    AbortError) so issue 011 can render a "couldn't reach generation;
 *    try again" message against the typed error.
 *
 * Tests do NOT cover the HTTP shape — that's covered by the backend
 * ``test_staging_endpoints_generate_jobs.py`` suite. The frontend
 * helper is a thin wrapper; the contract here is the wrapper logic
 * (URL composition, body shape, abort plumbing, error mapping).
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  EnqueueGenerationFailedError,
  EnqueueGenerationTimeoutError,
  enqueueProjectGeneration,
  mintIdempotencyKey,
  API_BASE_URL,
} from "../stagingApi";

describe("enqueueProjectGeneration", () => {
  let originalFetch: typeof globalThis.fetch;

  beforeEach(() => {
    originalFetch = globalThis.fetch;
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  // --------------------------------------------------------------------
  // Happy path + URL/body contract
  // --------------------------------------------------------------------

  it("returns { job_id } parsed from the 202 JSON body on success", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ job_id: "proj-1:__project__:__project__:abc" }), {
        status: 202,
        headers: { "Content-Type": "application/json" },
      }),
    );

    const result = await enqueueProjectGeneration("proj-1");

    expect(result).toEqual({ job_id: "proj-1:__project__:__project__:abc" });
  });

  it("POSTs to /staging/projects/{id}/jobs/generate", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ job_id: "j1" }), { status: 202 }),
    );
    globalThis.fetch = fetchMock;

    await enqueueProjectGeneration("project-xyz");

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [calledUrl, init] = fetchMock.mock.calls[0];
    expect(calledUrl).toBe(
      `${API_BASE_URL}/staging/projects/project-xyz/jobs/generate`,
    );
    expect(init?.method).toBe("POST");
    expect(init?.headers).toMatchObject({ "Content-Type": "application/json" });
  });

  it("sends body { regenerate_all: false } when no options provided", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ job_id: "j1" }), { status: 202 }),
    );
    globalThis.fetch = fetchMock;

    await enqueueProjectGeneration("p1");

    const init = fetchMock.mock.calls[0][1];
    const body = JSON.parse(init?.body as string);
    expect(body).toEqual({ regenerate_all: false });
  });

  it("sends body { regenerate_all: true } when regenerateAll=true", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ job_id: "j1" }), { status: 202 }),
    );
    globalThis.fetch = fetchMock;

    await enqueueProjectGeneration("p1", { regenerateAll: true });

    const init = fetchMock.mock.calls[0][1];
    const body = JSON.parse(init?.body as string);
    expect(body).toEqual({ regenerate_all: true });
  });

  it("coerces undefined regenerateAll to false (no truthy garbage forwarded)", async () => {
    // The endpoint uses StrictBool, so the helper MUST coerce its
    // optional camelCase regenerateAll input to a strict literal
    // boolean. Forwarding undefined would be rejected with 422.
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ job_id: "j1" }), { status: 202 }),
    );
    globalThis.fetch = fetchMock;

    await enqueueProjectGeneration("p1", {});

    const init = fetchMock.mock.calls[0][1];
    const body = JSON.parse(init?.body as string);
    expect(body).toEqual({ regenerate_all: false });
    expect(typeof body.regenerate_all).toBe("boolean");
  });

  // --------------------------------------------------------------------
  // Error surfacing
  // --------------------------------------------------------------------

  it("rejects with a descriptive Error on non-2xx (status + body included)", async () => {
    // Use mockImplementation so each call gets a fresh Response (the
    // body stream is single-use; reusing a Response across two
    // enqueueProjectGeneration calls would yield an empty body on
    // the second .text() read).
    globalThis.fetch = vi.fn().mockImplementation(() =>
      Promise.resolve(
        new Response("No rooms uploaded yet", {
          status: 400,
          headers: { "Content-Type": "text/plain" },
        }),
      ),
    );

    // Assert against ONE call so we can pin both status + body in
    // the same rejected error message (single regex that requires
    // both substrings to appear in the thrown Error.message).
    await expect(
      enqueueProjectGeneration("p1"),
    ).rejects.toThrow(/400.*No rooms uploaded yet/);
  });

  it("rejects with status info on a 502 (queue down) response", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response("Queue enqueue failed: queue down", {
        status: 502,
      }),
    );

    await expect(
      enqueueProjectGeneration("p1"),
    ).rejects.toThrow(/502/);
  });

  // --------------------------------------------------------------------
  // 180s abort timeout
  // --------------------------------------------------------------------

  it("aborts the fetch at 180s and rejects with EnqueueGenerationTimeoutError", async () => {
    vi.useFakeTimers();

    // Mock fetch to return a promise that ONLY rejects when the
    // signal is aborted (the production fetch behaviour). This
    // mirrors how DOM fetch surfaces an AbortError when its
    // AbortSignal fires.
    let rejectFn: ((reason: unknown) => void) | undefined;
    const fetchMock = vi.fn().mockImplementation(
      (_url: string, init: RequestInit | undefined) => {
        return new Promise((_resolve, reject) => {
          rejectFn = reject;
          init?.signal?.addEventListener("abort", () => {
            const abortErr = new DOMException(
              "The operation was aborted",
              "AbortError",
            );
            reject(abortErr);
          });
        });
      },
    );
    globalThis.fetch = fetchMock;

    const promise = enqueueProjectGeneration("p1");
    // Attach a no-op rejection handler immediately so vitest doesn't
    // flag the rejection as unhandled while we advance fake timers.
    promise.catch(() => {});

    // Just before the 180s mark — should NOT have aborted yet.
    await vi.advanceTimersByTimeAsync(179_000);
    expect(rejectFn).toBeDefined(); // fetch is still pending

    // Cross the 180s boundary; the controller should fire abort and
    // the promise should reject with our typed timeout error.
    await vi.advanceTimersByTimeAsync(2_000);

    await expect(promise).rejects.toBeInstanceOf(EnqueueGenerationTimeoutError);
    await expect(promise).rejects.toMatchObject({
      name: "EnqueueGenerationTimeoutError",
    });
  });

  it("does NOT abort when the response arrives in under 180s", async () => {
    vi.useFakeTimers();

    let resolveFetch: ((res: Response) => void) | undefined;
    globalThis.fetch = vi.fn().mockImplementation(() => {
      return new Promise<Response>((resolve) => {
        resolveFetch = resolve;
      });
    });

    const promise = enqueueProjectGeneration("p1");

    // 30s in — still pending; resolve the fetch.
    await vi.advanceTimersByTimeAsync(30_000);
    resolveFetch!(
      new Response(JSON.stringify({ job_id: "j1" }), { status: 202 }),
    );

    const result = await promise;
    expect(result).toEqual({ job_id: "j1" });

    // Advance well beyond 180s — the timer must have been cleared,
    // so no abort fires (and no late rejection bubbles up).
    await vi.advanceTimersByTimeAsync(200_000);
  });

  it("EnqueueGenerationTimeoutError has the expected typed shape", () => {
    // Issue 011 will catch and discriminate on this error specifically,
    // so the structural shape (name + instanceof) must be stable.
    const err = new EnqueueGenerationTimeoutError("test");
    expect(err).toBeInstanceOf(Error);
    expect(err.name).toBe("EnqueueGenerationTimeoutError");
    expect(err.message).toBe("test");
  });

  // --------------------------------------------------------------------
  // Issue 002: Idempotency-Key header + structured error shape
  // --------------------------------------------------------------------

  it("mints a fresh Idempotency-Key per call and sends it as a header", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({ job_id: "j1", already_in_flight: false }),
        { status: 202 },
      ),
    );
    globalThis.fetch = fetchMock;

    await enqueueProjectGeneration("p1");

    const headers = fetchMock.mock.calls[0][1]?.headers as Record<string, string>;
    expect(headers).toHaveProperty("Idempotency-Key");
    const key = headers["Idempotency-Key"];
    // Backend regex: /^[A-Za-z0-9_-]{1,128}$/. crypto.randomUUID()
    // returns 36-char lower-case hex with hyphens, which matches.
    expect(key).toMatch(/^[A-Za-z0-9_-]{1,128}$/);
    expect(key.length).toBeGreaterThanOrEqual(8);
  });

  it("uses the caller-supplied idempotencyKey when provided", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({ job_id: "j1", already_in_flight: false }),
        { status: 202 },
      ),
    );
    globalThis.fetch = fetchMock;

    await enqueueProjectGeneration("p1", {
      idempotencyKey: "test-key-12345",
    });

    const headers = fetchMock.mock.calls[0][1]?.headers as Record<string, string>;
    expect(headers["Idempotency-Key"]).toBe("test-key-12345");
  });

  it("returns { job_id, already_in_flight } from the new response shape", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({ job_id: "j-abc", already_in_flight: true }),
        { status: 200 },
      ),
    );

    const result = await enqueueProjectGeneration("p1");
    expect(result.job_id).toBe("j-abc");
    expect(result.already_in_flight).toBe(true);
  });

  it("throws EnqueueGenerationFailedError on a structured error body", async () => {
    // Issue 002: backend returns
    // ``{ error_kind, user_message, detail }`` for classified
    // failures. The helper parses this into a typed error so the
    // recovery banner can render kind-specific messaging.
    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          error_kind: "QUEUE_PERMISSION",
          user_message: "Worker can't reach the queue.",
          detail: { type: "ClientAuthenticationError", message: "denied" },
        }),
        {
          status: 502,
          headers: { "Content-Type": "application/json" },
        },
      ),
    );

    let caught: unknown;
    try {
      await enqueueProjectGeneration("p1");
    } catch (e) {
      caught = e;
    }
    expect(caught).toBeInstanceOf(EnqueueGenerationFailedError);
    const err = caught as EnqueueGenerationFailedError;
    expect(err.errorKind).toBe("QUEUE_PERMISSION");
    expect(err.userMessage).toBe("Worker can't reach the queue.");
    expect(err.httpStatus).toBe(502);
    expect(err.detail).toEqual({
      type: "ClientAuthenticationError",
      message: "denied",
    });
    // ``message`` falls through to ``userMessage`` so existing
    // ``error.message`` callers still see something useful.
    expect(err.message).toBe("Worker can't reach the queue.");
  });

  it("falls back to generic Error when the error body is not parseable", async () => {
    // 502 from a load balancer can return raw HTML — the helper must
    // fall back to a generic Error rather than throwing on JSON.parse.
    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response("<html>504 gateway timeout</html>", {
        status: 504,
        headers: { "Content-Type": "text/html" },
      }),
    );

    await expect(enqueueProjectGeneration("p1")).rejects.toThrow(/504/);
  });

  it("falls back to generic Error when JSON body lacks error_kind", async () => {
    // Defensive: a JSON body without error_kind is not the issue 002
    // shape (could be a legacy or third-party error). Fall back so
    // the generic Error path still catches it.
    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ detail: "Something went wrong" }), {
        status: 500,
        headers: { "Content-Type": "application/json" },
      }),
    );

    let caught: unknown;
    try {
      await enqueueProjectGeneration("p1");
    } catch (e) {
      caught = e;
    }
    expect(caught).not.toBeInstanceOf(EnqueueGenerationFailedError);
    expect((caught as Error).message).toMatch(/500/);
  });

  it("EnqueueGenerationFailedError has the expected typed shape", () => {
    const err = new EnqueueGenerationFailedError({
      errorKind: "BRIEF_FAILED",
      userMessage: "We couldn't draft the brief.",
      httpStatus: 502,
      detail: { type: "BriefCompositionFailed", message: "LLM down" },
    });
    expect(err).toBeInstanceOf(Error);
    expect(err.name).toBe("EnqueueGenerationFailedError");
    expect(err.errorKind).toBe("BRIEF_FAILED");
    expect(err.userMessage).toBe("We couldn't draft the brief.");
    expect(err.httpStatus).toBe(502);
    expect(err.detail).toEqual({
      type: "BriefCompositionFailed",
      message: "LLM down",
    });
  });

  it("mintIdempotencyKey produces backend-regex-compatible keys", () => {
    // Backend regex: ^[A-Za-z0-9_-]{1,128}$
    for (let i = 0; i < 10; i++) {
      const key = mintIdempotencyKey();
      expect(key).toMatch(/^[A-Za-z0-9_-]{1,128}$/);
    }
  });

  it("two calls mint distinct idempotency keys (no global collision)", async () => {
    const fetchMock = vi.fn().mockImplementation(() =>
      Promise.resolve(
        new Response(
          JSON.stringify({ job_id: "j1", already_in_flight: false }),
          { status: 202 },
        ),
      ),
    );
    globalThis.fetch = fetchMock;

    await enqueueProjectGeneration("p1");
    await enqueueProjectGeneration("p1");

    const k1 = (fetchMock.mock.calls[0][1]?.headers as Record<string, string>)[
      "Idempotency-Key"
    ];
    const k2 = (fetchMock.mock.calls[1][1]?.headers as Record<string, string>)[
      "Idempotency-Key"
    ];
    expect(k1).not.toBe(k2);
  });
});
