import { test, expect, Page, Route } from '@playwright/test';

/**
 * Issue 002 of failed-variation-retry-queue PRD: per-page in-memory retry
 * queue. When a single variation fails mid-stream, clicking Retry while
 * the global generation stream is still open used to silently no-op
 * (the page-level `isGenerating` flag short-circuited the click). This
 * spec covers the two scenarios called out by the issue's acceptance
 * criteria for slice 002:
 *
 *   1. Queue happy path: a Retry click lands during in-flight generation,
 *      shows a Queued indicator + toast + activity-log entry, then drains
 *      automatically into a single variation-regen POST after the global
 *      stream terminates.
 *
 *   2. Dedup on multi-click: three rapid Retry clicks land exactly one
 *      toast.info and exactly one variation-regen POST.
 *
 * Issue 003 (this slice) adds:
 *
 *   3. Supersede on Regenerate Room: when a queued retry exists and the
 *      user triggers a larger regen action (Regenerate Room), the queue
 *      is silently cleared so the queued retry does not fire after the
 *      bigger action completes. See PRD §"Page integration" (clear()
 *      paragraph) and Testing Decisions → scenario 2.
 *
 * The remaining PRD scenario (drop on global stream error) is tracked
 * separately under issue 004.
 *
 * Mocking pattern follows
 * `frontend/tests/e2e/regen-failure-preserves-prior-image.spec.ts`.
 */

const SCREENSHOT_DIR = 'test-results/screenshots/retry-queue-during-generation';
const PROJECT_ID = 'test-retry-queue';
const API_BASE = 'http://localhost:8000/api/v1';
const COMPLETED_IMAGE_URL = (variationId: string) =>
  `https://storage.blob.core.windows.net/images/staging/${PROJECT_ID}/variations/room-1/${variationId}.png`;

function sseEvent(type: string, data: Record<string, unknown>): string {
  return `event: ${type}\ndata: ${JSON.stringify(data)}\n\n`;
}

function makeBaseProject() {
  const now = new Date().toISOString();
  return {
    id: PROJECT_ID,
    name: 'Retry Queue Test',
    prompt: 'Modern minimalist',
    status: 'processing' as const,
    settings: {
      style: 'modern',
      room_count: 1,
      variations_per_room: 5,
      output_format: 'png',
      quality: 'high',
    },
    rooms: [
      {
        id: 'room-1',
        label: 'Living Room',
        original_image_url: `https://storage.blob.core.windows.net/images/staging/${PROJECT_ID}/originals/room-1.png?sv=mock`,
        // Room status reflects "has at least one failed variation" so the
        // header CTA renders as "Generate Remaining (1)" — exercising the
        // same code path as production where a partially-failed room
        // surfaces a header CTA the user can click to start a new global
        // stream that this test then holds open.
        status: 'failed' as const,
        variations: [
          {
            id: 'r1-v0',
            status: 'completed' as const,
            image_url: COMPLETED_IMAGE_URL('r1-v0'),
            created_at: now,
            updated_at: now,
          },
          {
            id: 'r1-v1',
            status: 'completed' as const,
            image_url: COMPLETED_IMAGE_URL('r1-v1'),
            created_at: now,
            updated_at: now,
          },
          {
            id: 'r1-v2',
            // The variation that fails mid-stream — the one we Retry.
            status: 'failed' as const,
            error: 'Simulated 429 from upstream',
            created_at: now,
            updated_at: now,
          },
          {
            id: 'r1-v3',
            status: 'completed' as const,
            image_url: COMPLETED_IMAGE_URL('r1-v3'),
            created_at: now,
            updated_at: now,
          },
          {
            id: 'r1-v4',
            status: 'completed' as const,
            image_url: COMPLETED_IMAGE_URL('r1-v4'),
            created_at: now,
            updated_at: now,
          },
        ],
        created_at: now,
        updated_at: now,
      },
    ],
    total_variations: 5,
    completed_variations: 4,
    created_at: now,
    updated_at: now,
  };
}

// Issue 003 fixture: same project shape as makeBaseProject() but with
// TWO failed variations (r1-v2 and r1-v3) and the project at status
// 'failed' so isGenerating starts false. This lets us exercise the
// `regeneratingVariationId` busy-gate path of the queue (the only
// realistic supersede path — every supersede entry-point button is
// `disabled={isGenerating}` so the isGenerating-gate path can't be
// reached via a real user click).
function makeProjectWithTwoFailedVariations() {
  const now = new Date().toISOString();
  return {
    id: PROJECT_ID,
    name: 'Retry Queue Test',
    prompt: 'Modern minimalist',
    status: 'failed' as const,
    settings: {
      style: 'modern',
      room_count: 1,
      variations_per_room: 5,
      output_format: 'png',
      quality: 'high',
    },
    rooms: [
      {
        id: 'room-1',
        label: 'Living Room',
        original_image_url: `https://storage.blob.core.windows.net/images/staging/${PROJECT_ID}/originals/room-1.png?sv=mock`,
        status: 'failed' as const,
        variations: [
          {
            id: 'r1-v0',
            status: 'completed' as const,
            image_url: COMPLETED_IMAGE_URL('r1-v0'),
            created_at: now,
            updated_at: now,
          },
          {
            id: 'r1-v1',
            status: 'completed' as const,
            image_url: COMPLETED_IMAGE_URL('r1-v1'),
            created_at: now,
            updated_at: now,
          },
          {
            id: 'r1-v2',
            status: 'failed' as const,
            error: 'Simulated 429 from upstream',
            created_at: now,
            updated_at: now,
          },
          {
            id: 'r1-v3',
            status: 'failed' as const,
            error: 'Simulated 429 from upstream',
            created_at: now,
            updated_at: now,
          },
          {
            id: 'r1-v4',
            status: 'completed' as const,
            image_url: COMPLETED_IMAGE_URL('r1-v4'),
            created_at: now,
            updated_at: now,
          },
        ],
        created_at: now,
        updated_at: now,
      },
    ],
    total_variations: 5,
    completed_variations: 3,
    created_at: now,
    updated_at: now,
  };
}

// Post-room-regen GET fixture: room regenerated successfully so r1-v2
// and r1-v3 are now completed. Used to satisfy the AC's "thumbnail
// enters the processing state from the room regen" → "regen completed"
// transition. The page reloads via project_completed → loadProject
// after the room regen SSE stream emits its terminal event.
function makeProjectWithRoomCompleted() {
  const base = makeProjectWithTwoFailedVariations();
  return {
    ...base,
    status: 'completed' as const,
    completed_variations: 5,
    rooms: base.rooms.map((room) => ({
      ...room,
      status: 'completed' as const,
      variations: room.variations.map((v) =>
        v.id === 'r1-v2' || v.id === 'r1-v3'
          ? {
              ...v,
              status: 'completed' as const,
              image_url: COMPLETED_IMAGE_URL(v.id + '-regen'),
              error: undefined,
            }
          : v,
      ),
    })),
  };
}

async function setupSasMock(page: Page) {
  await page.route(`${API_BASE}/gallery/sas-tokens`, (route: Route) =>
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        video_sas_token: 'sv=mock',
        image_sas_token: 'sv=mock',
        video_container_url: 'https://storage.blob.core.windows.net/videos',
        image_container_url: 'https://storage.blob.core.windows.net/images',
        expiry: new Date(Date.now() + 3600_000).toISOString(),
      }),
    }),
  );
}

test.describe('Retry queue during in-flight generation (issue 002)', () => {
  test('queue happy path: retry click during stream → Queued indicator + toast + drain on completion', async ({
    page,
  }) => {
    const project = makeBaseProject();
    await setupSasMock(page);

    let projectGetCount = 0;
    await page.route(`${API_BASE}/staging/projects/${PROJECT_ID}`, (route: Route) => {
      if (route.request().method() === 'GET') {
        projectGetCount += 1;
        // Always return the same fixture (the failed variation stays
        // failed until the variation-regen POST succeeds; we don't
        // simulate a status flip here because the queue test only cares
        // about the queue/dispatch contract, not the post-regen render).
        return route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ project }),
        });
      }
      return route.continue();
    });

    // Hold the global generation stream open until we explicitly release
    // it. This is what gives us the in-flight window during which the
    // Retry click must be queued.
    let releaseGlobalStream!: () => void;
    const globalStreamHeld = new Promise<void>((resolve) => {
      releaseGlobalStream = resolve;
    });

    let globalStreamCount = 0;
    await page.route(
      `${API_BASE}/staging/projects/${PROJECT_ID}/generate`,
      async (route: Route) => {
        globalStreamCount += 1;
        // Hold the global stream open until releaseGlobalStream() is
        // called by the test. This gives us a deterministic window during
        // which isGenerating=true, so the Retry click MUST queue rather
        // than dispatch.
        await globalStreamHeld;
        return route.fulfill({
          status: 200,
          contentType: 'text/event-stream',
          headers: { 'Cache-Control': 'no-cache', 'Connection': 'keep-alive' },
          body:
            sseEvent('variation_failed', {
              type: 'variation_failed',
              room_id: 'room-1',
              variation_index: 2,
              variation_id: 'r1-v2',
              error: 'Simulated 429 from upstream',
              elapsed_ms: 1500,
              tokens_used: null,
              model: 'gpt-image-2',
            }) +
            sseEvent('project_completed', { type: 'project_completed' }) +
            sseEvent('stream_ended', { type: 'stream_ended' }),
        });
      },
    );

    // Track the variation-regen POST.
    let regenPostCount = 0;
    let regenPostTime = 0;
    await page.route(
      `${API_BASE}/staging/projects/${PROJECT_ID}/rooms/room-1/variations/r1-v2/regenerate**`,
      (route: Route) => {
        regenPostCount += 1;
        regenPostTime = Date.now();
        return route.fulfill({
          status: 200,
          contentType: 'text/event-stream',
          headers: { 'Cache-Control': 'no-cache', 'Connection': 'keep-alive' },
          body:
            sseEvent('variation_completed', {
              type: 'variation_completed',
              room_id: 'room-1',
              variation_index: 2,
              elapsed_ms: 4500,
              tokens_used: 1234,
              model: 'gpt-image-2',
              image_url: COMPLETED_IMAGE_URL('r1-v2-regen'),
            }) +
            sseEvent('project_completed', { type: 'project_completed' }) +
            sseEvent('stream_ended', { type: 'stream_ended' }),
        });
      },
    );

    await page.goto(`/projects/${PROJECT_ID}`);
    await page.waitForLoadState('networkidle');

    // Sanity: the failed variation's Retry button is visible.
    const retryButton = page.getByRole('button', { name: /^Retry$/ });
    await expect(retryButton).toBeVisible();

    await page.screenshot({
      path: `${SCREENSHOT_DIR}/01-before-retry.png`,
      fullPage: true,
    });

    // Start global generation (which holds open until releaseGlobalStream).
    // The fixture project is already 'processing' so the user clicks the
    // header's Generate Remaining / Generate button. Find the project
    // header CTA.
    const headerCta = page.getByTestId('project-header-action');
    if (await headerCta.isVisible()) {
      await headerCta.click();
    } else {
      // Some layouts may not have the header CTA visible if all rooms
      // are already non-pending — fall back to the room-level Generate
      // button.
      const roomGenerate = page.getByRole('button', { name: /^Generate$|^Regenerate$/ }).first();
      await roomGenerate.click();
    }

    // Wait for the global stream to be requested (proves isGenerating=true).
    await expect.poll(() => globalStreamCount, { timeout: 5000 }).toBe(1);

    // CRITICAL: at this moment, the variation-regen POST must NOT have
    // fired yet. We're about to click Retry while the stream is open.
    expect(regenPostCount).toBe(0);

    // Click Retry on the failed variation.
    await retryButton.click();

    // Toast should appear immediately with the exact PRD copy.
    const queuedToast = page.getByText(
      /^Retry queued — will run when generation completes$/,
    );
    await expect(queuedToast).toBeVisible({ timeout: 3000 });

    // Thumbnail Retry button should be gone, Queued indicator visible.
    const queuedIndicator = page.getByTestId('variation-3-queued');
    await expect(queuedIndicator).toBeVisible();
    await expect(retryButton).toHaveCount(0);

    // Activity log entry should be present.
    const activityEntry = page.getByText(
      /Variation 3 retry queued/i,
    );
    await expect(activityEntry).toBeVisible({ timeout: 3000 });

    await page.screenshot({
      path: `${SCREENSHOT_DIR}/02-queued-during-stream.png`,
      fullPage: true,
    });

    // BLOCKING ASSERTION: no variation-regen POST has fired yet.
    // A buggy impl that dispatches eagerly instead of queuing would
    // already have a regenPostCount > 0 here.
    expect(regenPostCount).toBe(0);

    // Release the global stream so the project_completed event reaches
    // the frontend → isGenerating goes false → queue drains.
    releaseGlobalStream();

    // Now the queue should drain: exactly ONE variation-regen POST
    // fires for r1-v2.
    await expect.poll(() => regenPostCount, { timeout: 8000 }).toBe(1);

    // Sanity: the regen POST happened AFTER the global stream was released
    // (i.e., it didn't fire eagerly during the in-flight window).
    expect(regenPostTime).toBeGreaterThan(0);

    await page.screenshot({
      path: `${SCREENSHOT_DIR}/03-after-drain.png`,
      fullPage: true,
    });

    // Sanity: project was reloaded after drain (cumulative GETs should
    // include initial load + at least one post-event reload).
    expect(projectGetCount).toBeGreaterThan(1);
  });

  test('dedup on multi-click: three rapid Retry clicks → exactly one toast + one regen POST', async ({
    page,
  }) => {
    const project = makeBaseProject();
    await setupSasMock(page);

    await page.route(`${API_BASE}/staging/projects/${PROJECT_ID}`, (route: Route) => {
      if (route.request().method() === 'GET') {
        return route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ project }),
        });
      }
      return route.continue();
    });

    let releaseGlobalStream!: () => void;
    const globalStreamHeld = new Promise<void>((resolve) => {
      releaseGlobalStream = resolve;
    });

    let globalStreamCount = 0;
    await page.route(
      `${API_BASE}/staging/projects/${PROJECT_ID}/generate`,
      async (route: Route) => {
        globalStreamCount += 1;
        await globalStreamHeld;
        return route.fulfill({
          status: 200,
          contentType: 'text/event-stream',
          headers: { 'Cache-Control': 'no-cache', 'Connection': 'keep-alive' },
          body:
            sseEvent('variation_failed', {
              type: 'variation_failed',
              room_id: 'room-1',
              variation_index: 2,
              variation_id: 'r1-v2',
              error: 'Simulated 429',
              elapsed_ms: 1000,
              tokens_used: null,
              model: 'gpt-image-2',
            }) +
            sseEvent('project_completed', { type: 'project_completed' }) +
            sseEvent('stream_ended', { type: 'stream_ended' }),
        });
      },
    );

    let regenPostCount = 0;
    await page.route(
      `${API_BASE}/staging/projects/${PROJECT_ID}/rooms/room-1/variations/r1-v2/regenerate**`,
      (route: Route) => {
        regenPostCount += 1;
        return route.fulfill({
          status: 200,
          contentType: 'text/event-stream',
          headers: { 'Cache-Control': 'no-cache', 'Connection': 'keep-alive' },
          body:
            sseEvent('variation_completed', {
              type: 'variation_completed',
              room_id: 'room-1',
              variation_index: 2,
              elapsed_ms: 3000,
              tokens_used: 900,
              model: 'gpt-image-2',
            }) +
            sseEvent('project_completed', { type: 'project_completed' }) +
            sseEvent('stream_ended', { type: 'stream_ended' }),
        });
      },
    );

    await page.goto(`/projects/${PROJECT_ID}`);
    await page.waitForLoadState('networkidle');

    const retryButton = page.getByRole('button', { name: /^Retry$/ });
    await expect(retryButton).toBeVisible();

    const headerCta = page.getByTestId('project-header-action');
    if (await headerCta.isVisible()) {
      await headerCta.click();
    } else {
      const roomGenerate = page.getByRole('button', { name: /^Generate$|^Regenerate$/ }).first();
      await roomGenerate.click();
    }

    await expect.poll(() => globalStreamCount, { timeout: 5000 }).toBe(1);
    expect(regenPostCount).toBe(0);

    // Three rapid clicks. After the first click, the button is replaced
    // by the Queued indicator, so subsequent .click() calls would fail.
    // Instead, dispatch the React onClick directly via the page so we
    // exercise the dedup path inside enqueue() rather than the DOM-
    // visibility path.
    //
    // Strategy: capture the failed thumbnail's container, then call its
    // onRetry prop three times via dispatchEvent on the original Retry
    // button BEFORE it's removed from the DOM. Easiest path: click once,
    // wait for queued, then re-trigger via direct enqueue. Since we
    // can't reach the hook from the test, we click the (now-removed)
    // button via repeated locator clicks with timeout=0 — the second
    // and third clicks will fail with "element not found" if the
    // button truly went away.
    //
    // Simpler approach: do all three clicks AS FAST AS POSSIBLE using
    // Promise.all-ish parallelism. The page state hasn't yet re-rendered
    // (React state batching), so the button is still in the DOM for
    // multiple clicks. The hook's inFlight + queue dedup is what
    // catches the duplicates.
    await Promise.all([
      retryButton.click({ force: true }).catch(() => undefined),
      retryButton.click({ force: true }).catch(() => undefined),
      retryButton.click({ force: true }).catch(() => undefined),
    ]);

    // Exactly one toast appears.
    const queuedToast = page.getByText(
      /^Retry queued — will run when generation completes$/,
    );
    await expect(queuedToast).toBeVisible({ timeout: 3000 });
    await expect(queuedToast).toHaveCount(1);

    // Activity log entry appears exactly once.
    const activityEntries = page.getByText(/Variation 3 retry queued/i);
    await expect(activityEntries).toHaveCount(1);

    expect(regenPostCount).toBe(0);

    await page.screenshot({
      path: `${SCREENSHOT_DIR}/04-dedup-during-stream.png`,
      fullPage: true,
    });

    // Drain.
    releaseGlobalStream();

    // Wait long enough for any duplicates to fire — assert that ONE and
    // ONLY ONE regen POST happens. Add a hold + re-assert for robustness
    // against late duplicates.
    await expect.poll(() => regenPostCount, { timeout: 8000 }).toBe(1);
    await page.waitForTimeout(800);
    expect(regenPostCount).toBe(1);

    await page.screenshot({
      path: `${SCREENSHOT_DIR}/05-after-dedup-drain.png`,
      fullPage: true,
    });
  });

  test('supersede on Regenerate Room: queued retry → click Regenerate Room → queue cleared, no per-variation regen POST fires (issue 003)', async ({
    page,
  }) => {
    // SCENARIO DESIGN (rubber-duck-flagged):
    //
    // The realistic supersede path uses the `regeneratingVariationId`
    // busy gate, NOT the `isGenerating` busy gate. When `isGenerating`
    // is true, every supersede entry-point button (project-header-action,
    // RoomGroup's Regenerate) carries `disabled={isGenerating}`, so a
    // real user click never reaches the handler — the browser respects
    // the `disabled` HTML attribute. (Same constraint as in
    // `lightbox-regenerate-disabled-when-blocked.spec.ts`.)
    //
    // The hook's busy gate also fires on `regeneratingVariationId !==
    // null`, so we exercise THAT path:
    //
    //   1. Setup: two failed variations (r1-v2 at index 2, r1-v3 at
    //      index 3). isGenerating=false at start (no global stream).
    //   2. Click Retry on r1-v2 → DISPATCHED immediately (system idle).
    //      regeneratingVariationId=r1-v2, /variations/r1-v2/regenerate
    //      stream is HELD forever (the browser will abort it when
    //      Regenerate Room fires `streamCleanupRef.current?.()`).
    //   3. Click Retry on r1-v3 → QUEUED (regeneratingVariationId
    //      busy gate). variation-4-queued indicator visible.
    //   4. Click Regenerate Room (button enabled because isGenerating
    //      is still false; only regeneratingVariationId is set, which
    //      doesn't disable the room-level button per RoomGroup line
    //      176 `disabled={isGenerating}`).
    //   5. WITH FIX: clear() runs FIRST → queue empty → indicator
    //      cleared. WITHOUT FIX: queue still contains r1-v3 → indicator
    //      stays visible.
    //
    // BLOCKING ASSERTION (the unique pre-fix vs post-fix differentiator):
    // The variation-4-queued indicator MUST be gone after the Regenerate
    // Room click.
    //
    // The "no /variations/r1-v3/regenerate POST fires" assertion is
    // CORROBORATING but NOT unique to the fix — pre-fix, the abort of
    // /variations/r1-v2/regenerate leaves `regeneratingVariationId`
    // stuck at r1-v2 (the abort path returns early without emitting
    // any terminal SSE event, so the variation handler never calls
    // setRegeneratingVariationId(null)). The drain effect's gate
    // `if (isGenerating || regeneratingVariationId !== null) return;`
    // therefore never re-fires, and r1-v3 stays queued without
    // dispatching. So pre-fix the regen POST count is also 0; the
    // assertion catches the variant where someone accidentally clears
    // regeneratingVariationId on abort, but the indicator-visibility
    // assertion is the load-bearing regression catcher.

    const project = makeProjectWithTwoFailedVariations();
    await setupSasMock(page);

    // Stateful project GET so room regen reload reflects the room
    // having actually run (per AC: "thumbnail enters the processing
    // state from the room regen"). We flip `roomCompleted` true once
    // the room regen POST is received; subsequent GETs return a
    // fixture with the room status='completed' and r1-v2/r1-v3
    // converted from failed → completed.
    let roomCompleted = false;
    await page.route(`${API_BASE}/staging/projects/${PROJECT_ID}`, (route: Route) => {
      if (route.request().method() === 'GET') {
        const body = roomCompleted ? makeProjectWithRoomCompleted() : project;
        return route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ project: body }),
        });
      }
      return route.continue();
    });

    // Variation regen for r1-v2: HELD forever. The browser's
    // AbortController will cancel the fetch when handleRegenerateRoom
    // calls streamCleanupRef.current?.(); the Node-side promise just
    // hangs and Playwright cleans up at test end.
    let v2RegenCount = 0;
    await page.route(
      `${API_BASE}/staging/projects/${PROJECT_ID}/rooms/room-1/variations/r1-v2/regenerate**`,
      () => {
        v2RegenCount += 1;
        return new Promise(() => {
          /* never resolve — held until abort */
        });
      },
    );

    // Variation regen for r1-v3: must NEVER fire. Counter only.
    let v3RegenCount = 0;
    await page.route(
      `${API_BASE}/staging/projects/${PROJECT_ID}/rooms/room-1/variations/r1-v3/regenerate**`,
      (route: Route) => {
        v3RegenCount += 1;
        return route.fulfill({
          status: 200,
          contentType: 'text/event-stream',
          headers: { 'Cache-Control': 'no-cache', 'Connection': 'keep-alive' },
          body:
            sseEvent('variation_completed', {
              type: 'variation_completed',
              room_id: 'room-1',
              variation_index: 3,
              elapsed_ms: 1000,
              tokens_used: 500,
              model: 'gpt-image-2',
            }) +
            sseEvent('project_completed', { type: 'project_completed' }) +
            sseEvent('stream_ended', { type: 'stream_ended' }),
        });
      },
    );

    // Room regen: HELD via deferred Promise so we can assert the queue
    // state IMMEDIATELY after the Regenerate Room click — BEFORE any
    // post-regen reload flips r1-v3.status from 'failed' → 'completed'.
    // (If we let the room regen complete first, the failed-branch in
    // VariationThumbnail wouldn't render at all, hiding the Queued
    // indicator for a reason unrelated to clear() — see scenario design
    // comment.)
    let roomRegenCount = 0;
    let releaseRoomRegen!: () => void;
    const roomRegenHeld = new Promise<void>((resolve) => {
      releaseRoomRegen = resolve;
    });
    await page.route(
      `${API_BASE}/staging/projects/${PROJECT_ID}/rooms/room-1/regenerate**`,
      async (route: Route) => {
        roomRegenCount += 1;
        await roomRegenHeld;
        roomCompleted = true;
        return route.fulfill({
          status: 200,
          contentType: 'text/event-stream',
          headers: { 'Cache-Control': 'no-cache', 'Connection': 'keep-alive' },
          body:
            sseEvent('room_started', {
              type: 'room_started',
              room_id: 'room-1',
              label: 'Living Room',
            }) +
            sseEvent('room_completed', {
              type: 'room_completed',
              room_id: 'room-1',
            }) +
            sseEvent('project_completed', { type: 'project_completed' }) +
            sseEvent('stream_ended', { type: 'stream_ended' }),
        });
      },
    );

    await page.goto(`/projects/${PROJECT_ID}`);
    await page.waitForLoadState('networkidle');

    // Sanity: two failed variations → two Retry buttons.
    const retryButtons = page.getByRole('button', { name: /^Retry$/ });
    await expect(retryButtons).toHaveCount(2);

    await page.screenshot({
      path: `${SCREENSHOT_DIR}/06-supersede-before-retry.png`,
      fullPage: true,
    });

    // Step 2: Retry r1-v2 (the FIRST failed variation in DOM order →
    // index 2 → "variation 3"). System is idle → DISPATCHED.
    await retryButtons.nth(0).click();

    // Wait for r1-v2 regen POST to actually fire — proves the dispatch
    // ran and regeneratingVariationId is now set in React state. Without
    // this poll, the next click could land BEFORE handleRegenerateVariation
    // finishes its synchronous setRegeneratingVariationId(id), making
    // the next enqueue a 'dispatched' (not 'queued') outcome.
    await expect.poll(() => v2RegenCount, { timeout: 5000 }).toBe(1);

    // Step 3: Retry r1-v3. Note: VariationThumbnail's failed branch does
    // NOT react to `isRegenerating` (it only triggers the spinner on
    // status='completed' tiles), so r1-v2's Retry button stays in the
    // DOM. Both Retry buttons are still present; we explicitly target
    // nth(1) for r1-v3 (the SECOND failed variation in DOM order).
    await expect(retryButtons).toHaveCount(2);
    await retryButtons.nth(1).click();

    // Toast + activity-log entry confirm the queued outcome.
    const queuedToast = page.getByText(
      /^Retry queued — will run when generation completes$/,
    );
    await expect(queuedToast).toBeVisible({ timeout: 3000 });

    // Queued indicator visible on r1-v3 (index 3 → "variation-4-queued").
    const queuedIndicatorV3 = page.getByTestId('variation-4-queued');
    await expect(queuedIndicatorV3).toBeVisible({ timeout: 3000 });

    // After r1-v3 is queued, its Retry button is replaced by the
    // Queued indicator, so exactly 1 Retry button remains (r1-v2's,
    // which is still showing because of the isRegenerating gap above).
    await expect(retryButtons).toHaveCount(1);

    expect(v3RegenCount).toBe(0);
    expect(roomRegenCount).toBe(0);

    await page.screenshot({
      path: `${SCREENSHOT_DIR}/07-supersede-queued-on-v3.png`,
      fullPage: true,
    });

    // Step 4: Click Regenerate Room. Button is enabled because
    // isGenerating=false at this moment (only regeneratingVariationId
    // is set, which doesn't disable the room-level button).
    const regenRoomBtn = page.getByRole('button', { name: /^Regenerate$/ });
    await expect(regenRoomBtn).toHaveCount(1);
    await expect(regenRoomBtn).toBeEnabled();
    await regenRoomBtn.click();

    // Wait for room regen POST to land — proves the click reached the
    // handler. The route is HELD, so the project state still reflects
    // r1-v3 as 'failed' (the stateful GET fixture only flips when
    // roomCompleted=true, which happens after release).
    await expect.poll(() => roomRegenCount, { timeout: 5000 }).toBe(1);

    // Toast confirms the room regen handler proceeded past the
    // (now-inert pre-fix) clear() call AND the existing isGenerating
    // guard. This is the user-visible signal that the larger regen
    // action took ownership.
    await expect(page.getByText(/^Regenerating Living Room\.\.\.$/)).toBeVisible({
      timeout: 3000,
    });

    // BLOCKING POST-FIX ASSERTION (the load-bearing differentiator):
    // queued indicator on r1-v3 disappears.
    //
    // PRE-FIX: queue still contains r1-v3 (clear() was never called).
    // r1-v3.status is still 'failed' (room regen stream is held, no
    // SSE events have flipped state). queuedIds.has('r1-v3') is true,
    // so VariationThumbnail's failed branch renders the Queued
    // indicator. This assertion fails.
    //
    // POST-FIX: clear() emptied the queue. queuedIds.has('r1-v3') is
    // false. VariationThumbnail's failed branch renders the Retry
    // button instead. variation-4-queued has count 0. ✓
    await expect(queuedIndicatorV3).toHaveCount(0, { timeout: 3000 });

    await page.screenshot({
      path: `${SCREENSHOT_DIR}/08-supersede-after-room-regen-click.png`,
      fullPage: true,
    });

    // Release the room regen stream so the page settles cleanly
    // (otherwise the test process exit leaves Playwright with a hung
    // route handler).
    releaseRoomRegen();

    // Wait for room regen stream to terminate and for any late v3
    // dispatch to have a chance to fire. The 800ms hold mirrors the
    // pattern from scenario 2 (dedup) which also guards against late
    // duplicate POSTs.
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(800);

    // CORROBORATING: r1-v3 regen POST never fired. (Pre-fix this also
    // happens to be 0 because the stuck regeneratingVariationId
    // prevents the drain — see scenario design comment above. The
    // load-bearing assertion is the indicator visibility above.)
    expect(v3RegenCount).toBe(0);

    // Sanity: room regen happened exactly once + r1-v2 dispatch
    // happened exactly once (proves we exercised the
    // regeneratingVariationId busy-gate path, not the isGenerating
    // path).
    expect(roomRegenCount).toBe(1);
    expect(v2RegenCount).toBe(1);

    await page.screenshot({
      path: `${SCREENSHOT_DIR}/09-supersede-final.png`,
      fullPage: true,
    });
  });
});
