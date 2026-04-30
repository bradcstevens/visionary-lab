/**
 * Regenerate brief preserves per-image-object overrides — issue 004 of the
 * per-image-object-quantities-design PRD.
 *
 * Drives the full wizard happy-path through to step 4, edits a quantity
 * override (Lavender qty=8 in room-1), goes BACK to step 3, sends another
 * chat message, and clicks Generate Design Brief a second time. The
 * second `/brief` POST should:
 *   1. Carry `previous_brief` in the request body (so the backend can
 *      reconcile-by-name).
 *   2. Receive a backend response whose new palette renames "Pine" to
 *      "Pine Tree" — Pine should drop, Lavender should survive (different
 *      UUID, same normalized name).
 *
 * The spec asserts at the wizard surface:
 *   - The qty=8 override is carried forward onto the new Lavender row.
 *   - A non-blocking toast appears with carried_forward=1 / dropped=1
 *     copy (broadened wording per the rubber-duck review).
 */
import { test, expect, Page } from '@playwright/test';
import { join } from 'node:path';

const FIXTURES = join(__dirname, 'fixtures');

const PROJECT_ID = 'reg-preserve-' + Date.now();
const OBJ_LAV_OLD = 'obj-lavender-old';
const OBJ_PINE_OLD = 'obj-pine-old';
const OBJ_LAV_NEW = 'obj-lavender-new';
const OBJ_PINETREE_NEW = 'obj-pinetree-new';

interface CapturedBriefRequest {
  callIndex: number;
  hasPreviousBrief: boolean;
  previousBriefOverrideQty?: number;
}

async function mockStagingApi(page: Page) {
  const captured: CapturedBriefRequest[] = [];

  await page.route('**/api/v1/staging/projects', async (route, request) => {
    if (request.method() === 'POST') {
      await route.fulfill({
        status: 201,
        contentType: 'application/json',
        body: JSON.stringify({
          project: {
            id: PROJECT_ID,
            name: 'Regen Preserves Test',
            prompt: '',
            status: 'uploading',
            rooms: [],
            settings: { variations_per_room: 1, model: 'gpt-image-2', quality: 'high', size: 'auto' },
            created_at: new Date().toISOString(),
          },
        }),
      });
    } else {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ projects: [], total: 0 }),
      });
    }
  });

  await page.route('**/api/v1/staging/projects/*/rooms', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        project_id: PROJECT_ID,
        rooms_added: 1,
        rooms: [
          {
            id: 'room-1',
            label: 'Backyard',
            original_image_url: 'https://example.com/r1.png',
            status: 'pending',
            variations: [{ id: 'v-1', status: 'pending' }],
          },
        ],
      }),
    });
  });

  await page.route('**/api/v1/staging/projects/*/analyze', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        analyses: [
          { room_id: 'room-1', description: 'Backyard with fence', features: ['fence'], zones: ['fence line'] },
        ],
        failed_count: 0,
      }),
    });
  });

  // Chat is always "ready for brief" so the user only needs ONE message
  // to unlock the button on each visit to step 3.
  let chatCount = 0;
  await page.route('**/api/v1/staging/projects/*/chat', async (route) => {
    chatCount++;
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        reply: `Reply #${chatCount}`,
        ready_for_brief: chatCount >= 1,
        suggested_actions: ['generate_brief'],
      }),
    });
  });

  // First /brief POST returns a 2-object palette: Lavender + Pine.
  // Second /brief POST returns a 2-object palette where Lavender SURVIVES
  // (different UUID, same name) but Pine is RENAMED to "Pine Tree" —
  // reconcile drops the prev Pine override, carries forward the prev
  // Lavender override (qty=8) onto OBJ_LAV_NEW. PUT /brief is allowed
  // through the same handler (used implicitly by step 4 → 5; not under
  // test here but the wizard needs a 200 to advance).
  let briefCallIndex = 0;
  await page.route('**/api/v1/staging/projects/*/brief', async (route, request) => {
    if (request.method() === 'PUT') {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ brief: bodyOf(request) }),
      });
      return;
    }
    if (request.method() !== 'POST') {
      await route.fallback();
      return;
    }
    briefCallIndex++;
    const bodyJson = (request.postDataJSON?.() ?? {}) as Record<string, unknown>;
    const prev = bodyJson.previous_brief as
      | { per_image_objects?: Record<string, Array<{ object_id: string; quantity: number }>> }
      | undefined;
    const lavOverride = prev?.per_image_objects?.['room-1']?.find(
      (o) => o.object_id === OBJ_LAV_OLD,
    );
    captured.push({
      callIndex: briefCallIndex,
      hasPreviousBrief: prev != null,
      previousBriefOverrideQty: lavOverride?.quantity,
    });

    if (briefCallIndex === 1) {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          brief: {
            global_instructions: 'Add lush greenery',
            object_palette: [
              { id: OBJ_LAV_OLD, name: 'Lavender', description: 'Lavandula', category: 'plant', default_quantity: 3, size: '2 ft', placement: 'front row' },
              { id: OBJ_PINE_OLD, name: 'Pine', description: 'Pinus', category: 'tree', default_quantity: 2, size: '8 ft', placement: 'back row' },
            ],
            placement_guide: { back_row: 'Tall trees', front_row: 'Low groundcover' },
            per_image_notes: {},
            per_image_objects: {},
            preserve_elements: ['fence'],
            settings: { variations_per_room: 1, model: 'gpt-image-2', quality: 'high', size: 'auto' },
          },
          reconciliation_summary: { carried_forward: 0, dropped: 0 },
        }),
      });
      return;
    }

    // Second call — emulate the backend running its own reconcile.
    const carriedOverrides: Array<{ object_id: string; quantity: number; placement: string | null; enabled: boolean }> = [];
    let droppedCount = 0;
    if (lavOverride) {
      // Lavender survived (same name, new UUID).
      carriedOverrides.push({ object_id: OBJ_LAV_NEW, quantity: lavOverride.quantity, placement: null, enabled: true });
    }
    const pineOverride = prev?.per_image_objects?.['room-1']?.find(
      (o) => o.object_id === OBJ_PINE_OLD,
    );
    if (pineOverride) {
      droppedCount++;
    }

    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        brief: {
          global_instructions: 'Add lush greenery (revised)',
          object_palette: [
            { id: OBJ_LAV_NEW, name: 'Lavender', description: 'Lavandula', category: 'plant', default_quantity: 3, size: '2 ft', placement: 'front row' },
            { id: OBJ_PINETREE_NEW, name: 'Pine Tree', description: 'Pinus', category: 'tree', default_quantity: 2, size: '8 ft', placement: 'back row' },
          ],
          placement_guide: { back_row: 'Tall trees', front_row: 'Low groundcover' },
          per_image_notes: {},
          per_image_objects: carriedOverrides.length > 0 ? { 'room-1': carriedOverrides } : {},
          preserve_elements: ['fence'],
          settings: { variations_per_room: 1, model: 'gpt-image-2', quality: 'high', size: 'auto' },
        },
        reconciliation_summary: {
          carried_forward: carriedOverrides.length,
          dropped: droppedCount,
        },
      }),
    });
  });

  await page.route('**/gallery/sas-tokens', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ imageSasToken: 'sv=mock', videoSasToken: 'sv=mock' }),
    });
  });

  return { captured };
}

function bodyOf(request: { postDataJSON?: () => unknown; postData?: () => string | null }) {
  const j = request.postDataJSON?.();
  if (j != null) return j;
  const raw = request.postData?.();
  return raw ? JSON.parse(raw) : {};
}

test.describe('Regenerate brief preserves per-image overrides', () => {
  test('after editing qty=8 + going back + chatting + regenerate, override carries forward and toast surfaces dropped count', async ({
    page,
  }) => {
    const { captured } = await mockStagingApi(page);

    // ---- Step 1: name ----------------------------------------------------
    await page.goto('/projects/new');
    await page.waitForLoadState('domcontentloaded');
    await page.getByPlaceholder('e.g., Backyard Fence Line').fill('Regen Preserves');
    await page.getByRole('button', { name: 'Next', exact: true }).click();

    // ---- Step 2: upload --------------------------------------------------
    const fileInput = page.locator('#room-upload');
    await fileInput.setInputFiles([join(FIXTURES, 'test-room-1.png')]);
    await page.getByRole('button', { name: 'Next', exact: true }).click();

    // ---- Step 3: chat (1st pass) ----------------------------------------
    await expect(page.getByText("Here's what I see")).toBeVisible({ timeout: 15000 });
    let chatInput = page.locator('input[placeholder*="visualize"], input[placeholder*="go ahead"]').first();
    await chatInput.fill('Tropical fence line plants');
    await chatInput.press('Enter');
    await expect(page.getByText('Reply #1')).toBeVisible({ timeout: 5000 });

    // ---- Step 3 → 4: first brief generation -----------------------------
    await page.getByRole('button', { name: 'Generate Design Brief', exact: true }).click();
    await expect(page.getByText('Object Palette')).toBeVisible({ timeout: 10000 });

    // ---- Step 4: edit qty=8 on Lavender / room-1 + qty=5 on Pine ------
    await page.getByTestId('tab-image-room-1').click();
    const qtyOld = page.getByTestId(`qty-input-${OBJ_LAV_OLD}`);
    await expect(qtyOld).toBeVisible();
    await qtyOld.fill('8');
    await qtyOld.blur();
    // A second override on Pine — this one will be DROPPED on regenerate
    // because the new palette renames Pine to "Pine Tree". The toast
    // surfaces the drop count to the user.
    const qtyOldPine = page.getByTestId(`qty-input-${OBJ_PINE_OLD}`);
    await qtyOldPine.fill('5');
    await qtyOldPine.blur();
    await expect(page.getByTestId('override-indicator').first()).toBeVisible();
    await expect(qtyOld).toHaveValue('8');

    // ---- Back to step 3, chat one more time -----------------------------
    await page.getByRole('button', { name: 'Back' }).click();
    chatInput = page.locator('input[placeholder*="visualize"], input[placeholder*="go ahead"]').first();
    await chatInput.fill('Make it more lush');
    await chatInput.press('Enter');
    await expect(page.getByText('Reply #2')).toBeVisible({ timeout: 5000 });

    // ---- Regenerate (2nd brief call) ------------------------------------
    await page.getByRole('button', { name: 'Generate Design Brief', exact: true }).click();

    // Toast (sonner) surfaces the broadened "could not be matched in the
    // regenerated palette" copy when ``dropped > 0``. Asserted FIRST,
    // before any further navigation, because sonner toasts auto-dismiss
    // on a short timer and the rest of the assertions can outrun it.
    await expect(page.getByText(/could not be matched in the regenerated palette/i)).toBeVisible({
      timeout: 10000,
    });

    await expect(page.getByText('Object Palette')).toBeVisible({ timeout: 10000 });

    // The second /brief request carried previous_brief with our qty=8
    // override on the OLD Lavender id.
    expect(captured).toHaveLength(2);
    expect(captured[0].hasPreviousBrief).toBe(false);
    expect(captured[1].hasPreviousBrief).toBe(true);
    expect(captured[1].previousBriefOverrideQty).toBe(8);

    // The new palette now has OBJ_LAV_NEW + OBJ_PINETREE_NEW.
    await page.getByTestId('tab-image-room-1').click();
    const qtyNew = page.getByTestId(`qty-input-${OBJ_LAV_NEW}`);
    await expect(qtyNew).toBeVisible();
    // Carried forward: qty=8 should now appear on the NEW Lavender row.
    await expect(qtyNew).toHaveValue('8');
    await expect(page.getByTestId('override-indicator').first()).toBeVisible();

    // The renamed Pine ("Pine Tree") row exists and is at its palette
    // default — its prev override was dropped.
    const qtyPineTree = page.getByTestId(`qty-input-${OBJ_PINETREE_NEW}`);
    await expect(qtyPineTree).toBeVisible();
    await expect(qtyPineTree).toHaveValue('2');
  });
});
