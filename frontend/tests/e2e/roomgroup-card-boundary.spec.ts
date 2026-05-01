import { test, expect, Page, Route } from '@playwright/test';

/**
 * Issue 005 of the projects-page-improvements PRD: each `RoomGroup` is
 * wrapped in the existing Card primitive so the per-room Generate /
 * Regenerate button is visibly enclosed in the same container as the
 * images it acts on. A small "Image N of M" label is added next to
 * the room title for positional context.
 *
 * Acceptance criteria covered (issues/projects-page-improvements/005):
 *   - AC #1: RoomGroup is wrapped in the existing Card primitive (we
 *     assert one `[data-testid="room-card-${id}"]` per room AND that
 *     Card has the `data-slot="card"` attribute from the primitive).
 *   - AC #2 / #3: existing internal layout — title row with Regenerate
 *     button, status badge, "N/M variations" counter, image grid — is
 *     preserved unchanged INSIDE each Card.
 *   - AC #4: a small "Image N of M" label appears next to the room
 *     title, reflecting this room's position in `project.rooms`.
 *   - AC #5: no behavioral change — verified implicitly by the
 *     adjacent specs (concurrent-room-generation, retry-queue-during-
 *     generation, regen-button-a11y, etc.) all continuing to pass.
 *   - AC #6: Playwright spec asserts each rendered room is enclosed in
 *     a single visible container element AND the Regenerate button is
 *     a descendant of the same container as the room's image grid
 *     (no shared parent across rooms; no nested room cards).
 *
 * Load-bearing differentiator (false-negative resistance):
 *   PRE-FIX, no `[data-testid^="room-card-"]` element exists on the
 *   project detail page (the outer wrapper was a plain
 *   `<div className="space-y-4">`). The very first assertion —
 *   `await expect(roomCards).toHaveCount(3)` — is the unique pre-fix
 *   vs post-fix differentiator. Pre-fix the count is 0 → fail.
 *   Post-fix the count is 3 → pass.
 *
 *   Each per-card assertion is then scoped via `.locator()` so the
 *   "button is a descendant of the same container as the images"
 *   AC is asserted by ACTUAL DOM containment, not by accidental
 *   global counts.
 *
 * No-shared-parent assertion strategy (rubber-duck-recommended):
 *   Rather than asserting the Cards are direct siblings under
 *   `.space-y-12` (which would FAIL on a valid implementation
 *   because each room is wrapped in a per-room
 *   `<div className="space-y-3">` for lost-op banners — see
 *   `app/projects/[id]/page.tsx` around line 1118), the test asserts
 *   the per-room contract directly:
 *     - exactly one `room-card-${id}` per room,
 *     - each card contains its own room's content,
 *     - zero nested `room-card-*` elements (no Card contains another
 *       Card → no Card spans two rooms).
 *
 * Mocking pattern follows `frontend/tests/e2e/concurrent-room-
 * generation.spec.ts`.
 */

const PROJECT_ID = 'test-roomgroup-card-boundary';
const API_BASE = 'http://localhost:8000/api/v1';
const COMPLETED_IMAGE_URL = (roomId: string, variationId: string) =>
  `https://storage.blob.core.windows.net/images/staging/${PROJECT_ID}/variations/${roomId}/${variationId}.png`;

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
    name: 'Card Boundary Test',
    prompt: 'Modern minimalist',
    // 'completed' so the header CTA hides and there's no "Generate
    // Remaining" button in the way; only per-room Regenerate buttons
    // render inside each RoomGroup.
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
      // 'completed' so the room-level Regenerate button renders (per
      // RoomGroup: shown when status is 'failed' | 'completed' | 'processing').
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

test.describe('RoomGroup Card boundary (issue 005)', () => {
  test('each room is wrapped in a Card; Regenerate button shares the same Card as the images', async ({
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

    await page.goto(`/projects/${PROJECT_ID}`);
    await page.waitForLoadState('networkidle');

    // ─────────────────────────────────────────────────────────────
    // LOAD-BEARING DIFFERENTIATOR (AC #1, #6).
    // ─────────────────────────────────────────────────────────────
    // Pre-fix: the room outer wrapper was a plain <div className=
    // "space-y-4"> with no data-testid. `toHaveCount(3)` would be 0
    // and this assertion would fail. Post-fix: each RoomGroup renders
    // a Card with `data-testid="room-card-${room.id}"`, so the count
    // is exactly 3.
    const roomCards = page.locator('[data-testid^="room-card-"]');
    await expect(roomCards).toHaveCount(3);

    // ─────────────────────────────────────────────────────────────
    // AC #1: each Card uses the existing Card primitive (which sets
    // data-slot="card"). This pins the implementation to the shared
    // primitive rather than a hand-rolled <div> with the testid.
    // ─────────────────────────────────────────────────────────────
    for (const room of ROOMS) {
      const card = page.locator(`[data-testid="room-card-${room.id}"]`);
      await expect(card).toHaveAttribute('data-slot', 'card');
    }

    // ─────────────────────────────────────────────────────────────
    // AC #6: no shared parent across rooms — verified by asserting
    // zero nested room cards. If any Card contained another Card,
    // it would mean one container was spanning two rooms (or a room
    // was double-rendered).
    // ─────────────────────────────────────────────────────────────
    const nestedRoomCards = page.locator(
      '[data-testid^="room-card-"] [data-testid^="room-card-"]',
    );
    await expect(nestedRoomCards).toHaveCount(0);

    // ─────────────────────────────────────────────────────────────
    // Per-room assertions: every existing piece of internal layout
    // is preserved inside the Card AND the Regenerate button is a
    // descendant of the same Card as that room's image (AC #2, #3,
    // #5, #6).
    // ─────────────────────────────────────────────────────────────
    for (let i = 0; i < ROOMS.length; i++) {
      const room = ROOMS[i];
      const card = page.locator(`[data-testid="room-card-${room.id}"]`);

      // Title row: room label inside the card.
      await expect(card.getByRole('heading', { name: room.label })).toBeVisible();

      // AC #4: "Image N of M" label inside the card. M = ROOMS.length;
      // N is 1-based to match the user-facing position. The testid
      // pins the exact element so the assertion stays sharp even if
      // copy is tweaked later.
      const positionLabel = card.locator(`[data-testid="room-position-${room.id}"]`);
      await expect(positionLabel).toBeVisible();
      await expect(positionLabel).toHaveText(`Image ${i + 1} of ${ROOMS.length}`);

      // AC #3: existing status badge ('completed' for this fixture)
      // remains inside the card.
      await expect(card.getByText('completed', { exact: true })).toBeVisible();

      // AC #3: existing "N/M variations" counter remains inside the
      // card (1/1 for each room in this fixture).
      await expect(card.getByText('1/1 variations')).toBeVisible();

      // AC #6 (load-bearing): the Regenerate button is a descendant
      // of THIS room's Card. Scoping the locator to `card.locator(...)`
      // means the assertion only passes if the button sits inside the
      // Card boundary — the exact "no shared parent" / "Generate
      // sits within the same container as the images" contract.
      const regenButton = card.getByRole('button', { name: /^Regenerate$/ });
      await expect(regenButton).toBeVisible();
      await expect(regenButton).toBeEnabled();

      // AC #6: the room's images are also inside the SAME Card.
      // We use the room-specific original-image alt text (from
      // RoomGroup line 260: alt={`${room.label} original`}) so the
      // selector cannot accidentally match an unrelated <img>.
      const originalImage = card.getByRole('img', {
        name: `${room.label} original`,
      });
      await expect(originalImage).toBeVisible();

      // The variation thumbnail's <img> alt comes from
      // VariationThumbnail line 65: `alt={\`Variation ${index + 1}\`}`.
      // Each fixture room has exactly one variation so we expect
      // "Variation 1". Asserting the variation image is inside the
      // SAME Card pins the "image grid lives in the same container
      // as the Generate button" AC.
      const variationImage = card.getByRole('img', { name: 'Variation 1' });
      await expect(variationImage).toBeVisible();
    }

    // Sanity screenshot for the visual-regression note in the PRD's
    // "Testing Decisions → What is NOT tested" section: we don't
    // automate visual diffing, but we do leave a single full-page
    // screenshot artifact for human review.
    await page.screenshot({
      path: 'test-results/screenshots/roomgroup-card-boundary/01-cards-rendered.png',
      fullPage: true,
    });
  });

  test('only one room renders only one Card; the "Image N of M" label reflects the singleton', async ({
    page,
  }) => {
    // Edge case: a project with a single room should still render
    // exactly one Card and the position label should read "Image 1
    // of 1". Pins the M-of-M edge case (off-by-one regression
    // protection on roomIndex / totalRooms wiring in page.tsx).
    const single = makeProject();
    single.rooms = single.rooms.slice(0, 1);
    single.settings.room_count = 1;
    single.total_variations = 1;
    single.completed_variations = 1;

    await setupSasMock(page);
    await page.route(`${API_BASE}/staging/projects/${PROJECT_ID}`, (route: Route) => {
      if (route.request().method() === 'GET') {
        return route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ project: single }),
        });
      }
      return route.continue();
    });

    await page.goto(`/projects/${PROJECT_ID}`);
    await page.waitForLoadState('networkidle');

    const roomCards = page.locator('[data-testid^="room-card-"]');
    await expect(roomCards).toHaveCount(1);

    const card = page.locator('[data-testid="room-card-roomA"]');
    await expect(card.locator('[data-testid="room-position-roomA"]')).toHaveText(
      'Image 1 of 1',
    );
  });
});
