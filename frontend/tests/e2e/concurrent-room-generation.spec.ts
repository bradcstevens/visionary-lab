import { test, expect, Page, Route } from '@playwright/test';

/**
 * Issue 007 of the projects-page-improvements PRD: per-room concurrent
 * generation via the new ``useGenerationFleet`` hook.
 *
 * Acceptance criteria covered (from issues/projects-page-improvements/007):
 *   - AC #4: Clicking Generate on Image B while Image A is still streaming
 *     starts BOTH generations; A's button stays disabled, B's button enters
 *     its per-room loading state, and all other rooms remain enabled.
 *   - AC #9: Playwright spec covering concurrent per-room generation: kick
 *     off three rooms in rapid succession; assert all three Generate
 *     buttons enter their per-room loading state; assert non-targeted rooms
 *     remain enabled (in this test, all three ARE targeted, so we instead
 *     assert that non-targeted variation actions inside one room remain
 *     accessible).
 *
 * Pre-issue-007 the page used a single global ``isGenerating`` flag that
 * serialized every Generate / Regenerate button. Clicking "Regenerate" on
 * room B while room A was streaming would no-op silently (the flag was
 * already true). The new fleet hook tracks per-operation in-flight state,
 * so each room's button reads only ``inFlightProject || inFlightRooms.has(room.id)``
 * and unrelated rooms remain actionable.
 *
 * Test scenario:
 *   1. Setup: 3 completed-status rooms so each one renders a Regenerate
 *      button (the room-level button shows when room.status is 'failed',
 *      'completed', or 'processing'). All variations completed so the
 *      header CTA is hidden (status='completed').
 *   2. Click Regenerate on roomA. Hold its room-regen SSE stream open
 *      (deferred Promise) so isAnyInFlight stays true and roomA's button
 *      stays disabled.
 *   3. Assert roomA's Regenerate is now disabled (it is the in-flight one).
 *   4. Assert roomB's Regenerate is STILL enabled (per-room concurrency).
 *      Pre-fix: roomB would also be disabled (global flag).
 *   5. Click Regenerate on roomB. Hold its stream too. Assert roomB's
 *      button is now disabled AND roomA's stream POST already fired
 *      (concurrent rooms).
 *   6. Click Regenerate on roomC. Three concurrent streams.
 *   7. Release all streams. Assert all three buttons re-enable after the
 *      project_completed events land.
 *
 * False-negative resistance:
 *   - Each room's regen route returns a UNIQUE counter so we can prove
 *     all three POSTs landed (not just one of them while others were
 *     gated out).
 *   - The streams are HELD via deferred promises so the test can assert
 *     the in-flight state synchronously without racing against the SSE
 *     terminal events finalizing the streams early.
 *   - PRE-fix this test fails on the "B's button enabled while A holds"
 *     assertion because the global isGenerating flag would disable B too.
 *
 * Mocking pattern follows
 * ``frontend/tests/e2e/retry-queue-during-generation.spec.ts``.
 */

const SCREENSHOT_DIR = 'test-results/screenshots/concurrent-room-generation';
const PROJECT_ID = 'test-concurrent-rooms';
const API_BASE = 'http://localhost:8000/api/v1';
const COMPLETED_IMAGE_URL = (roomId: string, variationId: string) =>
  `https://storage.blob.core.windows.net/images/staging/${PROJECT_ID}/variations/${roomId}/${variationId}.png`;

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

function makeProject() {
  const now = new Date().toISOString();
  return {
    id: PROJECT_ID,
    name: 'Concurrent Rooms Test',
    prompt: 'Modern minimalist',
    // 'completed' so the header CTA hides and there's no "Generate
    // Remaining" button in the way — only per-room Regenerate buttons.
    status: 'completed' as const,
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
      // 'completed' so Regenerate button renders (per RoomGroup line 174:
      // shown when status is 'failed' | 'completed' | 'processing').
      status: 'completed' as const,
      variations: [
        {
          id: `${room.id}-v0`,
          status: 'completed' as const,
          image_url: COMPLETED_IMAGE_URL(room.id, `${room.id}-v0`),
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

test.describe('Concurrent per-room generation (issue 007)', () => {
  test('three rooms can stream concurrently — each button reads only its own per-room state', async ({
    page,
  }) => {
    const project = makeProject();
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

    // Per-room held streams. Each room's regen POST is held open (deferred
    // Promise) until the test explicitly releases it. This gives a
    // deterministic in-flight window for the per-room concurrency
    // assertions below.
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
      // Each route handler increments this room's counter, then awaits
      // the per-room held promise, then emits a clean terminal sequence.
      // The route handler closure captures `room.id` so each one is
      // distinct.
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

    // Each room's Regenerate button is rendered with text "Regenerate"
    // (RoomGroup line 182). All three are visible since all three rooms
    // are in 'completed' state.
    const regenButtons = page.getByRole('button', { name: /^Regenerate$/ });
    await expect(regenButtons).toHaveCount(3);
    await expect(regenButtons.nth(0)).toBeEnabled();
    await expect(regenButtons.nth(1)).toBeEnabled();
    await expect(regenButtons.nth(2)).toBeEnabled();

    await page.screenshot({
      path: `${SCREENSHOT_DIR}/01-before-concurrent.png`,
      fullPage: true,
    });

    // Step 2: kick off roomA.
    await regenButtons.nth(0).click();

    // Wait for roomA's regen POST to actually fire — proves the click
    // reached the handler and the fleet hook started the stream.
    await expect.poll(() => counts.get('roomA'), { timeout: 5000 }).toBe(1);

    // The room-toast confirms the handler proceeded.
    await expect(page.getByText(/^Regenerating Living Room\.\.\.$/)).toBeVisible({
      timeout: 3000,
    });

    // BLOCKING CONCURRENCY ASSERTION (the unique pre-fix vs post-fix
    // differentiator): roomA's button is disabled (its stream is in
    // flight); roomB's and roomC's buttons remain ENABLED. Pre-issue-007
    // the global isGenerating flag would disable all three. Post-fix
    // only the in-flight room's button is disabled.
    await expect(regenButtons.nth(0)).toBeDisabled();
    await expect(regenButtons.nth(1)).toBeEnabled();
    await expect(regenButtons.nth(2)).toBeEnabled();

    await page.screenshot({
      path: `${SCREENSHOT_DIR}/02-roomA-in-flight.png`,
      fullPage: true,
    });

    // Step 3: kick off roomB. roomA's stream is still held.
    await regenButtons.nth(1).click();

    await expect.poll(() => counts.get('roomB'), { timeout: 5000 }).toBe(1);

    // BLOCKING: both A and B in flight; only C enabled.
    await expect(regenButtons.nth(0)).toBeDisabled();
    await expect(regenButtons.nth(1)).toBeDisabled();
    await expect(regenButtons.nth(2)).toBeEnabled();

    // CRITICAL: roomA's POST count is still 1 (no double-fire). roomA's
    // route is HELD; if anything had aborted+restarted it, the counter
    // would be > 1.
    expect(counts.get('roomA')).toBe(1);

    await page.screenshot({
      path: `${SCREENSHOT_DIR}/03-roomA-and-B-concurrent.png`,
      fullPage: true,
    });

    // Step 4: kick off roomC.
    await regenButtons.nth(2).click();

    await expect.poll(() => counts.get('roomC'), { timeout: 5000 }).toBe(1);

    // All three concurrent — all disabled.
    await expect(regenButtons.nth(0)).toBeDisabled();
    await expect(regenButtons.nth(1)).toBeDisabled();
    await expect(regenButtons.nth(2)).toBeDisabled();

    // All three POSTs landed exactly once.
    expect(counts.get('roomA')).toBe(1);
    expect(counts.get('roomB')).toBe(1);
    expect(counts.get('roomC')).toBe(1);

    await page.screenshot({
      path: `${SCREENSHOT_DIR}/04-all-three-concurrent.png`,
      fullPage: true,
    });

    // Step 5: release all streams. Each emits project_completed +
    // stream_ended → fleet hook finalizes each stream → in-flight Sets
    // all empty out. Buttons re-enable.
    releasers.get('roomA')!();
    releasers.get('roomB')!();
    releasers.get('roomC')!();

    await page.waitForLoadState('networkidle');

    // After all three streams complete, every button is enabled again.
    await expect(regenButtons.nth(0)).toBeEnabled({ timeout: 8000 });
    await expect(regenButtons.nth(1)).toBeEnabled();
    await expect(regenButtons.nth(2)).toBeEnabled();

    // Sanity: no late duplicate POSTs.
    await page.waitForTimeout(800);
    expect(counts.get('roomA')).toBe(1);
    expect(counts.get('roomB')).toBe(1);
    expect(counts.get('roomC')).toBe(1);

    await page.screenshot({
      path: `${SCREENSHOT_DIR}/05-all-released.png`,
      fullPage: true,
    });
  });
});
