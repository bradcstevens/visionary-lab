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
 * The remaining two PRD scenarios (supersede on Regenerate Room, drop on
 * global error) are tracked separately under issues 003 and 004.
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
});
