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
  EnqueueGenerationTimeoutError,
  enqueueProjectGeneration,
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
});
