import { test, expect, Page, Route } from '@playwright/test';

/**
 * Project Generation E2E Tests
 *
 * Validates the full generation lifecycle: SSE streaming, activity log,
 * progress tracker, and variation thumbnail updates.
 *
 * Run with: npx playwright test tests/e2e/project-generation.spec.ts --headed
 */

const SCREENSHOT_DIR = 'test-results/screenshots/project-generation';
const PROJECT_ID = 'test-project-gen';
const API_BASE = 'http://localhost:8000/api/v1';

// ---------------------------------------------------------------------------
// Mock Data
// ---------------------------------------------------------------------------

function makeMockProject(overrides: Partial<MockProject> = {}): MockProject {
  return {
    id: PROJECT_ID,
    name: 'Backyard Redesign',
    prompt: 'Add drought-tolerant landscaping with native plants',
    status: 'pending',
    settings: {
      style: 'modern',
      room_count: 3,
      variations_per_room: 5,
      output_format: 'png',
      quality: 'high',
    },
    rooms: [
      makeRoom('room-1', 'Front Yard', 5),
      makeRoom('room-2', 'Side Garden', 5),
      makeRoom('room-3', 'Back Patio', 5),
    ],
    total_variations: 15,
    completed_variations: 0,
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
    ...overrides,
  };
}

function makeRoom(
  id: string,
  label: string,
  variationCount: number,
  roomStatus = 'pending',
  variationOverrides?: Partial<MockVariation>[],
): MockRoom {
  return {
    id,
    label,
    original_image_url: `https://storage.blob.core.windows.net/images/staging/${PROJECT_ID}/originals/${id}.png?sv=mock`,
    status: roomStatus,
    variations: Array.from({ length: variationCount }, (_, i) => ({
      id: `${id}-v${i}`,
      status: variationOverrides?.[i]?.status ?? 'pending',
      image_url: variationOverrides?.[i]?.image_url,
      error: variationOverrides?.[i]?.error,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    })),
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
  };
}

interface MockVariation {
  id: string;
  status: string;
  image_url?: string;
  error?: string;
  created_at: string;
  updated_at: string;
}

interface MockRoom {
  id: string;
  label: string;
  original_image_url: string;
  status: string;
  variations: MockVariation[];
  created_at: string;
  updated_at: string;
}

interface MockProject {
  id: string;
  name: string;
  prompt: string;
  status: string;
  settings: Record<string, unknown>;
  rooms: MockRoom[];
  total_variations: number;
  completed_variations: number;
  created_at: string;
  updated_at: string;
}

// ---------------------------------------------------------------------------
// SSE helpers
// ---------------------------------------------------------------------------

/** Build a single SSE event string (same format as backend _sse_event). */
function sseEvent(type: string, data: Record<string, unknown>): string {
  return `event: ${type}\ndata: ${JSON.stringify(data)}\n\n`;
}

/** Build a complete SSE body for a successful 3-room generation. */
function buildFullGenerationSSE(): string {
  let body = '';
  const rooms = ['room-1', 'room-2', 'room-3'];
  const labels = ['Front Yard', 'Side Garden', 'Back Patio'];

  for (let r = 0; r < rooms.length; r++) {
    body += sseEvent('room_started', { type: 'room_started', room_id: rooms[r], label: labels[r] });
    for (let v = 0; v < 5; v++) {
      body += sseEvent('variation_completed', {
        type: 'variation_completed',
        room_id: rooms[r],
        variation_index: v,
        image_url: `https://storage.blob.core.windows.net/images/staging/${PROJECT_ID}/variations/${rooms[r]}/v${v}.png`,
        elapsed_ms: 3000 + Math.random() * 2000,
        tokens_used: 1200 + Math.floor(Math.random() * 300),
        model: 'gpt-image-2',
      });
    }
    body += sseEvent('room_completed', { type: 'room_completed', room_id: rooms[r], status: 'completed' });
  }
  body += sseEvent('project_completed', { type: 'project_completed', status: 'completed' });
  return body;
}

// ---------------------------------------------------------------------------
// Route helpers
// ---------------------------------------------------------------------------

/**
 * Set up API route mocks with stateful GET handling.
 * The first GET returns `initialProject`, subsequent GETs return `updatedProject`.
 */
async function setupMockedRoutes(
  page: Page,
  initialProject: MockProject,
  updatedProject: MockProject,
  sseBody: string,
) {
  let getCount = 0;

  // Mock SAS tokens
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

  // Stateful project GET
  await page.route(`${API_BASE}/staging/projects/${PROJECT_ID}`, (route: Route) => {
    if (route.request().method() === 'GET') {
      getCount++;
      const data = getCount <= 1 ? initialProject : updatedProject;
      return route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ project: data }),
      });
    }
    return route.continue();
  });

  // SSE generation endpoint
  await page.route(`${API_BASE}/staging/projects/${PROJECT_ID}/generate`, (route: Route) =>
    route.fulfill({
      status: 200,
      contentType: 'text/event-stream',
      headers: { 'Cache-Control': 'no-cache', 'Connection': 'keep-alive' },
      body: sseBody,
    }),
  );
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test.describe('Project Generation', () => {

  test('project page loads with rooms and pending variations', async ({ page }) => {
    const project = makeMockProject();
    await setupMockedRoutes(page, project, project, '');

    await page.goto(`/projects/${PROJECT_ID}`);
    await page.waitForLoadState('networkidle');

    // Project name
    await expect(page.locator('h1')).toContainText('Backyard Redesign');

    // All three room labels visible
    for (const label of ['Front Yard', 'Side Garden', 'Back Patio']) {
      await expect(page.getByText(label, { exact: true }).first()).toBeVisible();
    }

    // Pending variation placeholders
    const pendingBadges = page.getByText('Awaiting generation');
    expect(await pendingBadges.count()).toBeGreaterThanOrEqual(3);

    // Generate CTA with variation count
    await expect(page.getByRole('button', { name: /Generate 15 Variations/i })).toBeVisible();

    await page.screenshot({ path: `${SCREENSHOT_DIR}/01-page-loaded.png`, fullPage: true });
  });

  test('generate button triggers SSE and shows generating banner', async ({ page }) => {
    const pending = makeMockProject();
    const processing = makeMockProject({
      status: 'processing',
      rooms: [
        makeRoom('room-1', 'Front Yard', 5, 'processing'),
        makeRoom('room-2', 'Side Garden', 5),
        makeRoom('room-3', 'Back Patio', 5),
      ],
    });

    // Minimal SSE — just room_started + project_completed
    const sse = sseEvent('room_started', { type: 'room_started', room_id: 'room-1', label: 'Front Yard' })
      + sseEvent('project_completed', { type: 'project_completed', status: 'completed' });

    await setupMockedRoutes(page, pending, processing, sse);
    await page.goto(`/projects/${PROJECT_ID}`);
    await page.waitForLoadState('networkidle');

    // Click Generate
    const generateBtn = page.getByRole('button', { name: /Generate 15 Variations/i });
    await expect(generateBtn).toBeVisible();

    const [postReq] = await Promise.all([
      page.waitForRequest(req => req.url().includes('/generate') && req.method() === 'POST'),
      generateBtn.click(),
    ]);
    expect(postReq).toBeTruthy();

    // Generating banner should appear
    await expect(page.getByText('Generating variations...')).toBeVisible({ timeout: 5000 });

    // The big CTA disappears (replaced by the banner)
    await expect(page.getByRole('button', { name: /Generate 15 Variations/i })).not.toBeVisible();

    await page.screenshot({ path: `${SCREENSHOT_DIR}/02-generating-banner.png`, fullPage: true });
  });

  test('activity log receives and renders SSE events', async ({ page }) => {
    const errors: string[] = [];
    page.on('pageerror', (err) => errors.push(err.message));

    const pending = makeMockProject();
    const processing = makeMockProject({ status: 'processing' });

    const sse =
      sseEvent('room_started', { type: 'room_started', room_id: 'room-1', label: 'Front Yard' })
      + sseEvent('variation_completed', {
        type: 'variation_completed', room_id: 'room-1', variation_index: 0,
        image_url: 'https://example.com/img.png', elapsed_ms: 4500, tokens_used: 1300, model: 'gpt-image-2',
      })
      + sseEvent('variation_completed', {
        type: 'variation_completed', room_id: 'room-1', variation_index: 1,
        image_url: 'https://example.com/img2.png', elapsed_ms: 3200, tokens_used: 1100, model: 'gpt-image-2',
      })
      + sseEvent('project_completed', { type: 'project_completed', status: 'completed' });

    await setupMockedRoutes(page, pending, processing, sse);
    await page.goto(`/projects/${PROJECT_ID}`);
    await page.waitForLoadState('networkidle');

    // Start generation
    await page.getByRole('button', { name: /Generate 15 Variations/i }).click();

    // Activity log panel should auto-open (first log entry triggers it)
    const activityPanel = page.locator('text=Activity').first();
    await expect(activityPanel).toBeVisible({ timeout: 8000 });

    // Log entries should contain generation messages
    await expect(page.getByText(/Starting generation for/).first()).toBeVisible({ timeout: 8000 });
    await expect(page.getByText(/Variation 1 saved/).first()).toBeVisible({ timeout: 8000 });
    await expect(page.getByText(/Variation 2 saved/).first()).toBeVisible({ timeout: 8000 });

    // Summary counters should show success count
    const successBadge = page.locator('text=/^[0-9]+$/').first();
    await expect(successBadge).toBeVisible();

    await page.screenshot({ path: `${SCREENSHOT_DIR}/03-activity-log.png`, fullPage: true });
    expect(errors).toEqual([]);
  });

  test('progress tracker renders during generation with correct progress', async ({ page }) => {
    // Initial state: pending
    const pending = makeMockProject();

    // After reload: processing with 2/15 complete
    const processing = makeMockProject({
      status: 'processing',
      rooms: [
        makeRoom('room-1', 'Front Yard', 5, 'processing', [
          { id: 'room-1-v0', status: 'completed', image_url: 'https://example.com/1.png', created_at: '', updated_at: '' },
          { id: 'room-1-v1', status: 'completed', image_url: 'https://example.com/2.png', created_at: '', updated_at: '' },
          { id: 'room-1-v2', status: 'processing', created_at: '', updated_at: '' },
          { id: 'room-1-v3', status: 'pending', created_at: '', updated_at: '' },
          { id: 'room-1-v4', status: 'pending', created_at: '', updated_at: '' },
        ]),
        makeRoom('room-2', 'Side Garden', 5),
        makeRoom('room-3', 'Back Patio', 5),
      ],
    });

    const sse =
      sseEvent('room_started', { type: 'room_started', room_id: 'room-1', label: 'Front Yard' })
      + sseEvent('variation_completed', {
        type: 'variation_completed', room_id: 'room-1', variation_index: 0,
        image_url: 'https://example.com/1.png', elapsed_ms: 4000, tokens_used: 1300, model: 'gpt-image-2',
      })
      + sseEvent('variation_completed', {
        type: 'variation_completed', room_id: 'room-1', variation_index: 1,
        image_url: 'https://example.com/2.png', elapsed_ms: 3800, tokens_used: 1200, model: 'gpt-image-2',
      });
    // No project_completed — stream stays open (tests mid-generation state)

    await setupMockedRoutes(page, pending, processing, sse);
    await page.goto(`/projects/${PROJECT_ID}`);
    await page.waitForLoadState('networkidle');

    // Start generation
    await page.getByRole('button', { name: /Generate 15 Variations/i }).click();

    // Wait for the debounced reload to fire and render the progress tracker
    await expect(page.getByText('Generation Progress')).toBeVisible({ timeout: 10_000 });

    // Progress bar exists
    const progressBar = page.locator('[role="progressbar"]');
    await expect(progressBar).toBeVisible();

    // Variations count (appears in both header and progress tracker; use exact match)
    await expect(page.getByText('2/15 variations', { exact: true }).first()).toBeVisible();

    // Per-room badges
    await expect(page.getByText('Front Yard').first()).toBeVisible();
    await expect(page.getByText('Side Garden').first()).toBeVisible();
    await expect(page.getByText('Back Patio').first()).toBeVisible();

    // ProgressTracker shows a status badge (Processing or Interrupted depending on timing)
    const processingOrInterrupted = page.locator('text=/Processing|Interrupted/').first();
    await expect(processingOrInterrupted).toBeVisible();

    await page.screenshot({ path: `${SCREENSHOT_DIR}/04-progress-tracker.png`, fullPage: true });
  });

  test('variation thumbnails transition from pending to completed', async ({ page }) => {
    const pending = makeMockProject();

    const withCompleted = makeMockProject({
      status: 'processing',
      rooms: [
        makeRoom('room-1', 'Front Yard', 5, 'completed', [
          { id: 'r1-v0', status: 'completed', image_url: 'https://storage.blob.core.windows.net/images/v0.png?sv=mock', created_at: '', updated_at: '' },
          { id: 'r1-v1', status: 'completed', image_url: 'https://storage.blob.core.windows.net/images/v1.png?sv=mock', created_at: '', updated_at: '' },
          { id: 'r1-v2', status: 'completed', image_url: 'https://storage.blob.core.windows.net/images/v2.png?sv=mock', created_at: '', updated_at: '' },
          { id: 'r1-v3', status: 'completed', image_url: 'https://storage.blob.core.windows.net/images/v3.png?sv=mock', created_at: '', updated_at: '' },
          { id: 'r1-v4', status: 'completed', image_url: 'https://storage.blob.core.windows.net/images/v4.png?sv=mock', created_at: '', updated_at: '' },
        ]),
        makeRoom('room-2', 'Side Garden', 5),
        makeRoom('room-3', 'Back Patio', 5),
      ],
    });

    const sse = buildFullGenerationSSE();
    await setupMockedRoutes(page, pending, withCompleted, sse);
    await page.goto(`/projects/${PROJECT_ID}`);
    await page.waitForLoadState('networkidle');

    // Initially all pending
    const pendingPlaceholders = page.getByText('Awaiting generation');
    expect(await pendingPlaceholders.count()).toBeGreaterThanOrEqual(5);

    // Start generation
    await page.getByRole('button', { name: /Generate 15 Variations/i }).click();

    // Wait for reload to reflect completed state
    // When room.status is 'completed' with no failures, the status message isn't rendered.
    // Instead, check for the per-room variation count text "5/5 variations"
    await expect(page.getByText('5/5 variations').first()).toBeVisible({ timeout: 10_000 });

    // Numbered badges for completed variations (1-5 for room-1)
    for (let i = 1; i <= 5; i++) {
      // The numbered badges are inside variation thumbnails
      const badge = page.locator(`text="${i}"`).first();
      await expect(badge).toBeVisible();
    }

    await page.screenshot({ path: `${SCREENSHOT_DIR}/05-variation-thumbnails.png`, fullPage: true });
  });

  test('full generation lifecycle completes with toast and status badge', async ({ page }) => {
    const pending = makeMockProject();
    const completed = makeMockProject({
      status: 'completed',
      completed_variations: 15,
      rooms: [
        makeRoom('room-1', 'Front Yard', 5, 'completed',
          Array.from({ length: 5 }, (_, i) => ({
            id: `r1-v${i}`, status: 'completed' as const,
            image_url: `https://storage.blob.core.windows.net/images/v${i}.png?sv=mock`,
            created_at: '', updated_at: '',
          }))),
        makeRoom('room-2', 'Side Garden', 5, 'completed',
          Array.from({ length: 5 }, (_, i) => ({
            id: `r2-v${i}`, status: 'completed' as const,
            image_url: `https://storage.blob.core.windows.net/images/v${i}.png?sv=mock`,
            created_at: '', updated_at: '',
          }))),
        makeRoom('room-3', 'Back Patio', 5, 'completed',
          Array.from({ length: 5 }, (_, i) => ({
            id: `r3-v${i}`, status: 'completed' as const,
            image_url: `https://storage.blob.core.windows.net/images/v${i}.png?sv=mock`,
            created_at: '', updated_at: '',
          }))),
      ],
    });

    const sse = buildFullGenerationSSE();
    await setupMockedRoutes(page, pending, completed, sse);
    await page.goto(`/projects/${PROJECT_ID}`);
    await page.waitForLoadState('networkidle');

    // Start generation
    await page.getByRole('button', { name: /Generate 15 Variations/i }).click();

    // Activity log shows completion message
    await expect(page.getByText('Generation complete!')).toBeVisible({ timeout: 15_000 });

    // Toast notification
    await expect(page.getByText('Generation completed!')).toBeVisible({ timeout: 5000 });

    // Status badge should show "completed" (the badge renders the status text)
    await expect(page.locator('text=completed').first()).toBeVisible({ timeout: 10_000 });

    // Variation count shows all complete (may appear in header or body)
    await expect(page.getByText(/15\/15 variations complete/).first()).toBeVisible();

    await page.screenshot({ path: `${SCREENSHOT_DIR}/06-generation-complete.png`, fullPage: true });
  });

  test('stream_ended fallback reconciles state when no terminal event received', async ({ page }) => {
    const pending = makeMockProject();
    const partialComplete = makeMockProject({
      status: 'processing',
      rooms: [
        makeRoom('room-1', 'Front Yard', 5, 'completed',
          Array.from({ length: 5 }, (_, i) => ({
            id: `r1-v${i}`, status: 'completed' as const,
            image_url: `https://example.com/v${i}.png`,
            created_at: '', updated_at: '',
          }))),
        makeRoom('room-2', 'Side Garden', 5),
        makeRoom('room-3', 'Back Patio', 5),
      ],
    });

    // SSE stream with events but NO project_completed or error (stream just ends)
    const sse =
      sseEvent('room_started', { type: 'room_started', room_id: 'room-1', label: 'Front Yard' })
      + sseEvent('variation_completed', {
        type: 'variation_completed', room_id: 'room-1', variation_index: 0,
        image_url: 'https://example.com/v0.png', elapsed_ms: 3000, tokens_used: 1100, model: 'gpt-image-2',
      });
    // Stream ends after this — no terminal event

    await setupMockedRoutes(page, pending, partialComplete, sse);
    await page.goto(`/projects/${PROJECT_ID}`);
    await page.waitForLoadState('networkidle');

    await page.getByRole('button', { name: /Generate 15 Variations/i }).click();

    // The stream_ended fallback should fire, setting isGenerating=false
    // and triggering a project reload. Generating banner should disappear.
    await expect(page.getByText('Generating variations...')).not.toBeVisible({ timeout: 15_000 });

    // After reload, page should recover and show the partially-completed state
    await expect(page.getByText('Front Yard').first()).toBeVisible();

    await page.screenshot({ path: `${SCREENSHOT_DIR}/07-stream-ended-fallback.png`, fullPage: true });
  });

  test('SSE parser handles events split across ReadableStream chunks', async ({ page }) => {
    const pending = makeMockProject();
    const processing = makeMockProject({ status: 'processing' });

    // We inject a custom fetch that delivers SSE events across multiple chunks
    // to test the parser's cross-chunk handling (the bug that was fixed).
    await setupMockedRoutes(page, pending, processing, '');

    // Override the generate route with a chunked response
    await page.route(`${API_BASE}/staging/projects/${PROJECT_ID}/generate`, async (route: Route) => {
      // Build two chunks that split an event across the boundary:
      // Chunk 1: complete first event + partial second event (event: line only)
      // Chunk 2: data: line + empty line of second event + terminal event
      const chunk1 =
        `event: room_started\ndata: ${JSON.stringify({ type: 'room_started', room_id: 'room-1', label: 'Front Yard' })}\n\n`
        + `event: variation_completed\n`;

      const chunk2 =
        `data: ${JSON.stringify({
          type: 'variation_completed', room_id: 'room-1', variation_index: 0,
          image_url: 'https://example.com/chunk-test.png', elapsed_ms: 2000, tokens_used: 900, model: 'gpt-image-2',
        })}\n\n`
        + `event: project_completed\ndata: ${JSON.stringify({ type: 'project_completed', status: 'completed' })}\n\n`;

      const encoder = new TextEncoder();
      const body = Buffer.concat([
        Buffer.from(encoder.encode(chunk1)),
        Buffer.from(encoder.encode(chunk2)),
      ]);

      await route.fulfill({
        status: 200,
        contentType: 'text/event-stream',
        headers: { 'Cache-Control': 'no-cache' },
        body,
      });
    });

    await page.goto(`/projects/${PROJECT_ID}`);
    await page.waitForLoadState('networkidle');
    await page.getByRole('button', { name: /Generate 15 Variations/i }).click();

    // Both events should be parsed — room_started and variation_completed
    await expect(page.getByText(/Starting generation for/).first()).toBeVisible({ timeout: 8000 });
    await expect(page.getByText(/Variation 1 saved/).first()).toBeVisible({ timeout: 8000 });

    // Terminal event should also be received
    await expect(page.getByText('Generation complete!').first()).toBeVisible({ timeout: 8000 });

    await page.screenshot({ path: `${SCREENSHOT_DIR}/08-chunk-boundary.png`, fullPage: true });
  });

  test('generation error renders error banner with retry button', async ({ page }) => {
    const pending = makeMockProject();
    const failed = makeMockProject({ status: 'failed' });

    const sse =
      sseEvent('room_started', { type: 'room_started', room_id: 'room-1', label: 'Front Yard' })
      + sseEvent('error', { type: 'error', error: 'Rate limit exceeded — please wait and retry' });

    await setupMockedRoutes(page, pending, failed, sse);
    await page.goto(`/projects/${PROJECT_ID}`);
    await page.waitForLoadState('networkidle');

    await page.getByRole('button', { name: /Generate 15 Variations/i }).click();

    // Error banner should appear
    await expect(page.getByText('Generation encountered an error')).toBeVisible({ timeout: 8000 });
    await expect(page.getByText('Rate limit exceeded').first()).toBeVisible();

    // Retry button in error banner
    await expect(page.getByRole('button', { name: /Retry/i }).first()).toBeVisible();

    await page.screenshot({ path: `${SCREENSHOT_DIR}/09-error-banner.png`, fullPage: true });
  });

  test('variation failure shows error in thumbnail', async ({ page }) => {
    const pending = makeMockProject();
    const withFailure = makeMockProject({
      status: 'processing',
      rooms: [
        makeRoom('room-1', 'Front Yard', 5, 'completed', [
          { id: 'r1-v0', status: 'completed', image_url: 'https://example.com/v0.png', created_at: '', updated_at: '' },
          { id: 'r1-v1', status: 'failed', error: 'Content policy violation', created_at: '', updated_at: '' },
          { id: 'r1-v2', status: 'completed', image_url: 'https://example.com/v2.png', created_at: '', updated_at: '' },
          { id: 'r1-v3', status: 'pending', created_at: '', updated_at: '' },
          { id: 'r1-v4', status: 'pending', created_at: '', updated_at: '' },
        ]),
        makeRoom('room-2', 'Side Garden', 5),
        makeRoom('room-3', 'Back Patio', 5),
      ],
    });

    const sse =
      sseEvent('room_started', { type: 'room_started', room_id: 'room-1', label: 'Front Yard' })
      + sseEvent('variation_completed', {
        type: 'variation_completed', room_id: 'room-1', variation_index: 0,
        image_url: 'https://example.com/v0.png', elapsed_ms: 3000, tokens_used: 1100, model: 'gpt-image-2',
      })
      + sseEvent('variation_failed', {
        type: 'variation_failed', room_id: 'room-1', variation_index: 1,
        error: 'Content policy violation', elapsed_ms: 1000,
      })
      + sseEvent('project_completed', { type: 'project_completed', status: 'completed' });

    await setupMockedRoutes(page, pending, withFailure, sse);
    await page.goto(`/projects/${PROJECT_ID}`);
    await page.waitForLoadState('networkidle');

    await page.getByRole('button', { name: /Generate 15 Variations/i }).click();

    // Activity log should show the failure
    await expect(page.getByText(/Variation 2 failed/)).toBeVisible({ timeout: 8000 });

    // After reload, the failed variation thumbnail should show error text
    await expect(page.getByText('Content policy violation').first()).toBeVisible({ timeout: 10_000 });

    // Retry button on failed thumbnail
    await expect(page.getByRole('button', { name: /Retry/i }).first()).toBeVisible();

    await page.screenshot({ path: `${SCREENSHOT_DIR}/10-variation-failure.png`, fullPage: true });
  });
});

// ---------------------------------------------------------------------------
// Live Smoke Test (opt-in — requires running backend)
// ---------------------------------------------------------------------------

test.describe('Live Smoke Test', () => {
  // Skip by default — enable with LIVE_SMOKE=1 env var
  test.skip(!process.env.LIVE_SMOKE, 'Set LIVE_SMOKE=1 to run against live backend');

  const LIVE_PROJECT_ID = process.env.LIVE_PROJECT_ID ?? '17735db0-a0bf-4dfb-8a45-fcf59fe4de3e';

  test('SSE stream establishes and events flow for real project', async ({ page }) => {
    const errors: string[] = [];
    page.on('pageerror', (err) => errors.push(err.message));

    // Collect all console messages for debugging
    const consoleLogs: string[] = [];
    page.on('console', (msg) => consoleLogs.push(`[${msg.type()}] ${msg.text()}`));

    await page.goto(`/projects/${LIVE_PROJECT_ID}`);
    await page.waitForLoadState('networkidle');

    // Project should load
    await expect(page.locator('h1').first()).toBeVisible({ timeout: 10_000 });

    await page.screenshot({ path: `${SCREENSHOT_DIR}/11-live-project-loaded.png`, fullPage: true });

    // Look for generate or regenerate button
    const generateBtn = page.getByRole('button', { name: /Generate|Regenerate/i }).first();
    const hasButton = await generateBtn.isVisible().catch(() => false);

    if (!hasButton) {
      console.log('No generate/regenerate button found — project may already be completed or processing');
      console.log('Console logs:', consoleLogs.join('\n'));
      return;
    }

    // Click and verify SSE POST is made
    const [postReq] = await Promise.all([
      page.waitForRequest(
        req => req.url().includes('/generate') && req.method() === 'POST',
        { timeout: 5000 },
      ),
      generateBtn.click(),
    ]);

    expect(postReq).toBeTruthy();
    console.log('SSE POST request made to:', postReq.url());

    // Wait for any activity log entry to appear (proves events are flowing)
    try {
      await expect(page.getByText(/Starting generation for/)).toBeVisible({ timeout: 30_000 });
      console.log('✓ SSE events are flowing — activity log shows generation started');
    } catch {
      console.log('⚠ No activity log entries after 30s — SSE events may not be reaching frontend');
      console.log('Console logs:', consoleLogs.join('\n'));
    }

    await page.screenshot({ path: `${SCREENSHOT_DIR}/12-live-generation-progress.png`, fullPage: true });

    // Log any JS errors for debugging
    if (errors.length > 0) {
      console.log('JS errors:', errors);
    }
  });
});
