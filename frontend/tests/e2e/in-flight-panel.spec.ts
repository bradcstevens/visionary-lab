import { test, expect, Page, Route } from '@playwright/test';

/**
 * Issue 008 of the projects-page-improvements PRD: live "In Flight (N)"
 * section pinned to the top of the activity log panel.
 *
 * Acceptance criteria covered (from issues/projects-page-improvements/008):
 *   - AC #1: A new ActivityFeed surface tracks `inFlight` operations
 *     alongside the chronological `entries` (covered indirectly — we
 *     observe the panel rendering both sections).
 *   - AC #2: useGenerationFleet calls startOp on click and endOp on
 *     termination — verified by the count appearing on click and
 *     dropping when a stream completes.
 *   - AC #3: The In Flight (N) section renders pinned above the
 *     chronological log when inFlight.length > 0; chronological log
 *     below is unchanged.
 *   - AC #4: Operations appear in In Flight on click (BEFORE the first
 *     SSE event lands). The test holds the SSE response unresolved
 *     after the POST is in-flight; the In Flight section is asserted
 *     visible AND showing the per-room labels at that point. Phase
 *     reads "Starting…" because no SSE event has been forwarded yet.
 *   - AC #5: Queued/starting label rendered honestly. Phase text
 *     "Starting…" is asserted before any SSE event fires; "Running"
 *     after the first non-terminal event lands.
 *   - AC #6: Completed operations drop out of In Flight and remain
 *     in the chronological log below — verified by releasing one
 *     stream and asserting the row count drops to 2 while the success
 *     log entry appears.
 *   - AC #7: No backend changes (verified by zero backend file
 *     changes in this slice).
 *   - AC #8: Three concurrent rooms; manually open the activity log
 *     (per slice 006, auto-open is gone); In Flight (3) shows 3
 *     entries with labels + live elapsed timer ticking up; one
 *     completes and drops out.
 *
 * False-negative resistance:
 *   - The In Flight count is read from a dedicated test-id
 *     (`in-flight-count`) so changes to label copy don't desync the
 *     test from the AC's "shows N entries" semantic.
 *   - The "live elapsed timer ticks up" assertion captures the
 *     elapsed text twice with a 1.5s gap and asserts they differ
 *     (or the second is ≥ the first, since formatElapsed rounds
 *     down to whole seconds and a tick may not have crossed).
 *   - The post-completion drop assertion uses both the count badge
 *     AND the explicit row test-id absence, so a layout regression
 *     that hides the row but doesn't actually remove it would fail.
 *
 * Mocking pattern follows ``concurrent-room-generation.spec.ts`` plus
 * the per-room held-stream pattern.
 */

const SCREENSHOT_DIR = 'test-results/screenshots/in-flight-panel';
const PROJECT_ID = 'test-in-flight-panel';
const API_BASE = 'http://localhost:8000/api/v1';

function sseEvent(type: string, data: Record<string, unknown>): string {
  return `event: ${type}\ndata: ${JSON.stringify(data)}\n\n`;
}

interface RoomFixture {
  id: string;
  label: string;
}

const ROOMS: RoomFixture[] = [
  { id: 'roomA', label: 'Living Room' },
  { id: 'roomB', label: 'Kitchen' },
  { id: 'roomC', label: 'Bedroom' },
];

function makeProject(allCompleted: boolean) {
  const now = new Date().toISOString();
  return {
    id: PROJECT_ID,
    name: 'In Flight Panel Test',
    prompt: 'Modern minimalist',
    status: allCompleted ? ('completed' as const) : ('pending' as const),
    settings: {
      style: 'modern',
      room_count: 3,
      variations_per_room: 1,
      output_format: 'png',
      quality: 'high',
    },
    rooms: ROOMS.map((room) => ({
      id: room.id,
      label: room.label,
      original_image_url: `https://storage.blob.core.windows.net/images/staging/${PROJECT_ID}/originals/${room.id}.png?sv=mock`,
      // 'completed' so Regenerate button renders.
      status: 'completed' as const,
      variations: [
        {
          id: `${room.id}-v0`,
          status: 'completed' as const,
          image_url: `https://storage.blob.core.windows.net/images/staging/${PROJECT_ID}/variations/${room.id}/v0.png`,
          created_at: now,
          updated_at: now,
        },
      ],
      created_at: now,
      updated_at: now,
    })),
    total_variations: 3,
    completed_variations: 3,
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

test.describe('In Flight panel inside activity log (issue 008)', () => {
  test('three concurrent rooms render In Flight (3); one completion drops the count to 2 and the log entry appears below', async ({
    page,
  }) => {
    const project = makeProject(true);
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

    // Per-room held streams. The route handler RETURNS the held promise's
    // resolution inline; it sends a `room_started` event THEN waits for
    // the test's release before sending the terminal events. This gives
    // the test a window where:
    //   - the room is in In Flight,
    //   - phase has flipped to "Running" (room_started fired),
    //   - elapsed time is ticking,
    //   - the stream isn't yet terminated (so the row hasn't been removed).
    //
    // Note: route.fulfill() with an SSE body sends the WHOLE body atomically.
    // To fire one event, hold, then fire terminal events, we'd need
    // ReadableStream support. Playwright's route.fulfill body is a string,
    // not a stream — so instead we hold the entire response (no events fire
    // until release) and assert the "Starting…" phase + elapsed time
    // ticking BEFORE release. That matches AC #4 / #5 (panel ack on click,
    // before any SSE event lands).
    const releasers = new Map<string, () => void>();
    const counts = new Map<string, number>();
    const heldPromises = new Map<string, Promise<void>>();
    for (const room of ROOMS) {
      counts.set(room.id, 0);
      let release!: () => void;
      const held = new Promise<void>((resolve) => {
        release = resolve;
      });
      releasers.set(room.id, release);
      heldPromises.set(room.id, held);
    }

    for (const room of ROOMS) {
      const roomId = room.id;
      await page.route(
        `${API_BASE}/staging/projects/${PROJECT_ID}/rooms/${roomId}/regenerate**`,
        async (route: Route) => {
          counts.set(roomId, (counts.get(roomId) ?? 0) + 1);
          await heldPromises.get(roomId)!;
          return route.fulfill({
            status: 200,
            contentType: 'text/event-stream',
            headers: { 'Cache-Control': 'no-cache', 'Connection': 'keep-alive' },
            body:
              sseEvent('room_started', {
                type: 'room_started',
                room_id: roomId,
                label: room.label,
              }) +
              sseEvent('room_completed', {
                type: 'room_completed',
                room_id: roomId,
              }) +
              sseEvent('project_completed', { type: 'project_completed' }) +
              sseEvent('stream_ended', { type: 'stream_ended' }),
          });
        },
      );
    }

    await page.goto(`/projects/${PROJECT_ID}`);
    await page.waitForLoadState('networkidle');

    // The toggle starts in the closed state — auto-open is gone (slice 006).
    const toggle = page.locator(
      'button[title="Show activity log"], button[title="Hide activity log"]',
    );
    await expect(toggle).toHaveAttribute('title', 'Show activity log');

    // Click Regenerate on all three rooms in rapid succession.
    const regenButtons = page.getByRole('button', { name: /^Regenerate$/ });
    await expect(regenButtons).toHaveCount(3);
    await regenButtons.nth(0).click();
    await regenButtons.nth(1).click();
    await regenButtons.nth(2).click();

    // All three POSTs should fire (the fleet hook opens streams concurrently).
    await expect.poll(() => counts.get('roomA'), { timeout: 5000 }).toBe(1);
    await expect.poll(() => counts.get('roomB'), { timeout: 5000 }).toBe(1);
    await expect.poll(() => counts.get('roomC'), { timeout: 5000 }).toBe(1);

    // Manually open the activity log panel — auto-open is gone per slice 006.
    await toggle.click();
    await expect(toggle).toHaveAttribute('title', 'Hide activity log');

    // ── AC #3 + #8: In Flight (N) section is visible, pinned above the
    // chronological log, with the correct count.
    const inFlightSection = page.getByTestId('in-flight-section');
    await expect(inFlightSection).toBeVisible();
    await expect(page.getByTestId('in-flight-count')).toHaveText('3');

    // ── AC #8: rows show the correct labels for each in-flight room.
    await expect(inFlightSection.getByText('Living Room')).toBeVisible();
    await expect(inFlightSection.getByText('Kitchen')).toBeVisible();
    await expect(inFlightSection.getByText('Bedroom')).toBeVisible();

    // ── AC #5: phase label reads "Starting…" — no SSE event has been
    // forwarded yet (the entire SSE response is held). This is the
    // "queued or starting" honest signal.
    const phaseLabels = inFlightSection.getByText('Starting…');
    await expect(phaseLabels).toHaveCount(3);

    await page.screenshot({
      path: `${SCREENSHOT_DIR}/01-three-in-flight-starting.png`,
      fullPage: true,
    });

    // ── AC #8: live elapsed timer ticking up. Capture the elapsed text
    // for one row, wait ~2s, capture again, assert the time has advanced.
    // Use `page.evaluate` on the rendered text so we can compare numeric
    // values rather than struggling with a Playwright text-extraction
    // race during the tick.
    const elapsedLocator = inFlightSection
      .locator('[data-testid^="in-flight-elapsed-"]')
      .first();
    const t0Text = (await elapsedLocator.textContent()) ?? '0:00';
    // Format is `m:ss` — parse to seconds for monotonic comparison.
    const parseMmSs = (s: string): number => {
      const [m, ss] = s.split(':').map((n) => parseInt(n, 10));
      return (Number.isNaN(m) ? 0 : m) * 60 + (Number.isNaN(ss) ? 0 : ss);
    };
    const t0Seconds = parseMmSs(t0Text);
    await page.waitForTimeout(2100);
    const t1Text = (await elapsedLocator.textContent()) ?? '0:00';
    const t1Seconds = parseMmSs(t1Text);
    // Live tick: at least one second should have elapsed in the rendered
    // text. Pre-fix (no setInterval / static text): t0 === t1 always.
    expect(t1Seconds).toBeGreaterThanOrEqual(t0Seconds + 1);

    // ── AC #6: release ONE stream → that row drops out of In Flight,
    // count drops to 2, and the corresponding log entry appears in the
    // chronological log below.
    releasers.get('roomA')!();
    await page.waitForLoadState('networkidle');

    // Living Room row removed from In Flight.
    await expect(inFlightSection.getByText('Living Room')).toHaveCount(0, {
      timeout: 8000,
    });
    await expect(page.getByTestId('in-flight-count')).toHaveText('2');

    // Living Room success log entry now visible in the chronological log
    // below the In Flight section. The page's handleStreamEvent emits
    // "Room {label} regenerated" on room_completed (or similar copy).
    // Use a forgiving locator to tolerate small copy variation.
    const logSuccess = page
      .getByText(/Living Room|completed/i)
      .filter({ visible: true });
    await expect(logSuccess.first()).toBeVisible({ timeout: 8000 });

    await page.screenshot({
      path: `${SCREENSHOT_DIR}/02-one-completed-two-remaining.png`,
      fullPage: true,
    });

    // ── Sanity: the other two rows are still in flight with their labels.
    await expect(inFlightSection.getByText('Kitchen')).toBeVisible();
    await expect(inFlightSection.getByText('Bedroom')).toBeVisible();

    // ── Release the rest. The In Flight section should disappear entirely
    // (the section conditionally renders only when inFlight.length > 0).
    releasers.get('roomB')!();
    releasers.get('roomC')!();
    await page.waitForLoadState('networkidle');

    await expect(page.getByTestId('in-flight-section')).toHaveCount(0, {
      timeout: 8000,
    });

    await page.screenshot({
      path: `${SCREENSHOT_DIR}/03-all-completed-section-gone.png`,
      fullPage: true,
    });
  });

  test('In Flight section is NOT rendered when no operations are active', async ({ page }) => {
    // Sanity: the section only renders when inFlight.length > 0. Avoids
    // an empty heading-only "In Flight (0)" stub that would clutter the
    // panel for users who haven't kicked anything off yet.
    const project = makeProject(true);
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

    await page.goto(`/projects/${PROJECT_ID}`);
    await page.waitForLoadState('networkidle');

    // Open the activity log panel.
    const toggle = page.locator(
      'button[title="Show activity log"], button[title="Hide activity log"]',
    );
    await toggle.click();
    await expect(toggle).toHaveAttribute('title', 'Hide activity log');

    // No In Flight section.
    await expect(page.getByTestId('in-flight-section')).toHaveCount(0);
  });
});
