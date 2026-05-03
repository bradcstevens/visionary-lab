import { test, expect, Page, Route } from '@playwright/test';

/**
 * Single recovery-banner regression — issue 004 of the
 * `projects-page-stalled-stream-error-cleanup` PRD.
 *
 * The PRD's headlining bug: stalled-project landings used to paint
 * up to three overlapping banners (red destructive "Generation
 * encountered an error", amber lost-op, amber stale-processing)
 * because three independent code paths owned three banner blocks.
 * Issue 002 collapsed those into ONE classifier-driven banner with
 * `data-testid="recovery-banner"` and `data-recovery-kind`. This
 * spec pins the invariant: there is exactly ONE banner in the DOM
 * on a stalled-project landing, so any future change reintroducing
 * the three-banner stack fails CI loudly and independently of
 * banner copy.
 *
 * The stalled condition is seeded by intercepting the project fetch
 * with a Playwright route handler that returns a fixture project
 * with `status: 'processing'`, zero in-flight ops, and zero
 * progress. With those inputs `getRecoveryState` returns
 * `{ kind: 'interrupted' }` (precedence rule 3 — see
 * `frontend/utils/recovery-state.ts`), so the visible banner's
 * `data-recovery-kind` MUST equal `interrupted`.
 *
 * No two-minute watchdog wait, no new `tests/projects/`
 * fixture-loader plumbing — the spec is fully self-contained via
 * `page.route` interception, matching the pattern in
 * `project-status-badge.spec.ts`.
 */

const SCREENSHOT_DIR = 'test-results/screenshots/recovery-banner-single';
const PROJECT_ID = 'test-recovery-banner-single';
const API_BASE = 'http://localhost:8000/api/v1';

type RoomStatus = 'pending' | 'processing' | 'completed' | 'failed';

interface MockVariation {
  id: string;
  status: string;
  image_url?: string;
  created_at: string;
  updated_at: string;
}

interface MockRoom {
  id: string;
  label: string;
  original_image_url: string;
  status: RoomStatus;
  variations: MockVariation[];
  created_at: string;
  updated_at: string;
}

function makeStalledProject() {
  const now = new Date().toISOString();
  // Five rooms all in `processing` status with zero completed
  // variations: a project the backend believes is mid-run, but with
  // no live SSE stream backing it (the page loads cold, so
  // `isAnyInFlight` is false and `projectLostOps` is empty). This is
  // exactly the combination that should resolve to
  // `{ kind: 'interrupted' }`.
  const rooms: MockRoom[] = Array.from({ length: 5 }, (_, i) => {
    const idx = i + 1;
    return {
      id: `room-${idx}`,
      label: `Room ${idx}`,
      original_image_url: `https://storage.blob.core.windows.net/images/staging/${PROJECT_ID}/originals/room-${idx}.png?sv=mock`,
      status: 'processing' as RoomStatus,
      variations: Array.from({ length: 5 }, (_, v) => ({
        id: `r${idx}-v${v}`,
        status: 'pending',
        created_at: now,
        updated_at: now,
      })),
      created_at: now,
      updated_at: now,
    };
  });

  return {
    id: PROJECT_ID,
    name: 'Stalled Project',
    prompt: 'Modern minimalist',
    status: 'processing' as const,
    settings: {
      style: 'modern',
      room_count: rooms.length,
      variations_per_room: 5,
      output_format: 'png',
      quality: 'high',
    },
    rooms,
    total_variations: rooms.length * 5,
    completed_variations: 0,
    created_at: now,
    updated_at: now,
  };
}

async function setupSasTokenMock(page: Page) {
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

async function setupProjectGet(
  page: Page,
  project: ReturnType<typeof makeStalledProject>,
) {
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
}

test.describe('Single recovery banner on stalled landing (issue 004 — projects-page-stalled-stream-error-cleanup)', () => {
  test('stalled project landing renders exactly one recovery banner with kind="interrupted"', async ({
    page,
  }) => {
    await setupSasTokenMock(page);
    await setupProjectGet(page, makeStalledProject());

    await page.goto(`/projects/${PROJECT_ID}`);
    await page.waitForLoadState('networkidle');

    // Wait for the banner to be present — the page renders it
    // synchronously once the project fetch resolves and
    // `getRecoveryState` returns `interrupted`.
    const banners = page.getByTestId('recovery-banner');
    await expect(banners.first()).toBeVisible();

    // The headlining invariant: exactly ONE banner. Future changes
    // that reintroduce the three-banner stack would surface here
    // independently of any banner copy.
    await expect(banners).toHaveCount(1);

    // The banner's discriminator MUST match the seeded condition
    // (zero-in-flight + processing → interrupted, NOT error and NOT
    // stream-lost).
    await expect(banners.first()).toHaveAttribute(
      'data-recovery-kind',
      'interrupted',
    );

    await page.screenshot({
      path: `${SCREENSHOT_DIR}/01-stalled-shows-single-interrupted-banner.png`,
      fullPage: true,
    });
  });
});
