/**
 * Per-image quantity / placement / skip override flow.
 *
 * Issue 003 of the per-image-object-quantities-design PRD. Drives the wizard
 * happy path through to step 4, then exercises the BriefEditorTabs + the
 * PerImageObjectTable component against the mocked brief.
 *
 * The end-to-end "generate AND assert per-room prompts" coverage lives in
 * the backend tests (tests/test_brief_generator.py); this spec is the UI
 * counterpart focused on observable component behavior at the brief-editor
 * surface.
 */
import { test, expect, Page } from '@playwright/test';
import { join } from 'node:path';

const FIXTURES = join(__dirname, 'fixtures');

/** Mock staging API; returns a brief with TWO objects so the per-image table
 *  has more than one row to exercise. */
async function mockStagingApi(page: Page) {
  const projectId = 'pio-' + Date.now();
  const objId1 = 'obj-lavender-1';
  const objId2 = 'obj-pine-1';

  await page.route('**/api/v1/staging/projects', async (route, request) => {
    if (request.method() === 'POST') {
      const body = request.postDataJSON?.() ?? {};
      await route.fulfill({
        status: 201,
        contentType: 'application/json',
        body: JSON.stringify({
          project: {
            id: projectId,
            name: body.name ?? 'Per-Image Test',
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
    await new Promise((r) => setTimeout(r, 100));
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        project_id: projectId,
        rooms_added: 2,
        rooms: [
          { id: 'room-1', label: 'Backyard', original_image_url: 'https://example.com/r1.png', status: 'pending', variations: [{ id: 'v-1', status: 'pending' }] },
          { id: 'room-2', label: 'Patio', original_image_url: 'https://example.com/r2.png', status: 'pending', variations: [{ id: 'v-2', status: 'pending' }] },
        ],
      }),
    });
  });

  await page.route('**/api/v1/staging/projects/*/analyze', async (route) => {
    await new Promise((r) => setTimeout(r, 100));
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        analyses: [
          { room_id: 'room-1', description: 'Backyard with fence', features: ['fence'], zones: ['fence line'] },
          { room_id: 'room-2', description: 'Stone patio', features: ['patio'], zones: ['patio border'] },
        ],
        failed_count: 0,
      }),
    });
  });

  let chatCount = 0;
  await page.route('**/api/v1/staging/projects/*/chat', async (route) => {
    chatCount++;
    await new Promise((r) => setTimeout(r, 100));
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        reply: chatCount === 1 ? 'What style?' : 'Ready for brief?',
        ready_for_brief: chatCount >= 2,
        suggested_actions: chatCount >= 2 ? ['generate_brief'] : ['choose_style'],
      }),
    });
  });

  await page.route('**/api/v1/staging/projects/*/brief', async (route, request) => {
    if (request.method() === 'POST') {
      await new Promise((r) => setTimeout(r, 100));
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          brief: {
            global_instructions: 'Add lush greenery',
            object_palette: [
              { id: objId1, name: 'Lavender', description: 'Lavandula', category: 'plant', default_quantity: 3, size: '2 ft', placement: 'front row' },
              { id: objId2, name: 'Pine', description: 'Pinus', category: 'tree', default_quantity: 2, size: '8 ft', placement: 'back row' },
            ],
            placement_guide: { back_row: 'Tall grasses', front_row: 'Low groundcover' },
            per_image_notes: {},
            per_image_objects: {},
            preserve_elements: ['fence'],
            settings: { variations_per_room: 1, model: 'gpt-image-2', quality: 'high', size: 'auto' },
          },
        }),
      });
    }
  });

  await page.route('**/gallery/sas-tokens', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ imageSasToken: 'sv=mock', videoSasToken: 'sv=mock' }),
    });
  });

  return { projectId, objId1, objId2 };
}

async function navigateToBriefEditor(page: Page) {
  const ids = await mockStagingApi(page);
  await page.goto('/projects/new');
  await page.waitForLoadState('domcontentloaded');

  await page.getByPlaceholder('e.g., Backyard Fence Line').fill('Per-Image Quantities');
  await page.getByRole('button', { name: 'Next', exact: true }).click();

  const fileInput = page.locator('#room-upload');
  await fileInput.setInputFiles([join(FIXTURES, 'test-room-1.png'), join(FIXTURES, 'test-room-2.png')]);
  await page.getByRole('button', { name: 'Next', exact: true }).click();

  await expect(page.getByText("Here's what I see")).toBeVisible({ timeout: 15000 });

  // Build a 2-message conversation to unlock the brief generator.
  let chatInput = page.locator('input[placeholder*="visualize"]');
  await chatInput.fill('Tropical fence line plants');
  await chatInput.press('Enter');
  await expect(page.getByText('What style?')).toBeVisible({ timeout: 5000 });

  chatInput = page.locator('input[placeholder*="visualize"], input[placeholder*="go ahead"]').first();
  await chatInput.fill('Natural prairie style');
  await chatInput.press('Enter');
  await expect(page.getByText('Ready for brief?')).toBeVisible({ timeout: 5000 });

  await page.getByRole('button', { name: 'Generate Design Brief', exact: true }).click();
  await expect(page.getByText('Object Palette')).toBeVisible({ timeout: 10000 });
  return ids;
}

test.describe('Brief Editor — Per-Image Object Overrides', () => {
  test('renders Default Palette tab + one tab per uploaded image', async ({ page }) => {
    await navigateToBriefEditor(page);

    await expect(page.getByTestId('tab-default-palette')).toBeVisible();
    await expect(page.getByTestId('tab-image-room-1')).toBeVisible();
    await expect(page.getByTestId('tab-image-room-2')).toBeVisible();
  });

  test('switching to image tab shows per-image object table with palette defaults', async ({ page }) => {
    const { objId1 } = await navigateToBriefEditor(page);
    await page.getByTestId('tab-image-room-1').click();

    // Each palette object becomes a row pre-filled with its default_quantity.
    const qty1 = page.getByTestId(`qty-input-${objId1}`);
    await expect(qty1).toBeVisible();
    await expect(qty1).toHaveValue('3');

    // No override badge yet.
    await expect(page.getByTestId('override-indicator')).toHaveCount(0);
  });

  test('editing quantity creates an override entry and shows the override indicator', async ({ page }) => {
    const { objId1 } = await navigateToBriefEditor(page);
    await page.getByTestId('tab-image-room-1').click();

    const qty1 = page.getByTestId(`qty-input-${objId1}`);
    await qty1.fill('10');
    await qty1.blur();

    // Override badge appears for this row.
    await expect(page.getByTestId('override-indicator')).toBeVisible();
    await expect(qty1).toHaveValue('10');
  });

  test('typing the same value as palette default does NOT create a sticky override entry', async ({ page }) => {
    const { objId1 } = await navigateToBriefEditor(page);
    await page.getByTestId('tab-image-room-1').click();

    const qty1 = page.getByTestId(`qty-input-${objId1}`);
    // Replace 3 with 3 — same as default. Override should be auto-pruned.
    await qty1.fill('3');
    await qty1.blur();

    await expect(page.getByTestId('override-indicator')).toHaveCount(0);
  });

  test('"Use Default" removes the override entry', async ({ page }) => {
    const { objId1 } = await navigateToBriefEditor(page);
    await page.getByTestId('tab-image-room-1').click();

    const qty1 = page.getByTestId(`qty-input-${objId1}`);
    await qty1.fill('7');
    await qty1.blur();
    await expect(page.getByTestId('override-indicator')).toBeVisible();

    await page.getByTestId(`use-default-btn-${objId1}`).click();
    await expect(page.getByTestId('override-indicator')).toHaveCount(0);
    // Quantity input snaps back to palette default.
    await expect(qty1).toHaveValue('3');
  });

  test('"Skip" disables qty/placement inputs and keeps the override indicator', async ({ page }) => {
    const { objId1 } = await navigateToBriefEditor(page);
    await page.getByTestId('tab-image-room-1').click();

    await page.getByTestId(`skip-btn-${objId1}`).click();

    await expect(page.getByTestId(`qty-input-${objId1}`)).toBeDisabled();
    await expect(page.getByTestId(`placement-input-${objId1}`)).toBeDisabled();
    await expect(page.getByTestId('override-indicator')).toBeVisible();

    // "Use Default" recovers from skip.
    await page.getByTestId(`use-default-btn-${objId1}`).click();
    await expect(page.getByTestId(`qty-input-${objId1}`)).toBeEnabled();
    await expect(page.getByTestId('override-indicator')).toHaveCount(0);
  });

  test('clearing placement back to empty after typing a value reverts to inherit (no empty-string override persisted)', async ({ page }) => {
    const { objId1 } = await navigateToBriefEditor(page);
    await page.getByTestId('tab-image-room-1').click();

    const placement = page.getByTestId(`placement-input-${objId1}`);
    // Type a real placement override.
    await placement.fill('back row, near pergola');
    await placement.blur();
    await expect(page.getByTestId('override-indicator')).toBeVisible();

    // Clear back to empty — override should be pruned (default-equivalent).
    await placement.fill('');
    await placement.blur();
    await expect(page.getByTestId('override-indicator')).toHaveCount(0);
    // Input snaps back to the palette's placement value.
    await expect(placement).toHaveValue('front row');
  });

  test('per-image quantity override does NOT leak into other image tabs', async ({ page }) => {
    const { objId1 } = await navigateToBriefEditor(page);

    await page.getByTestId('tab-image-room-1').click();
    const qty1A = page.getByTestId(`qty-input-${objId1}`);
    await qty1A.fill('15');
    await qty1A.blur();
    await expect(page.getByTestId('override-indicator')).toBeVisible();

    // Switch to image tab 2: the same row should still show palette default,
    // not the override applied to room-1.
    await page.getByTestId('tab-image-room-2').click();
    const qty1B = page.getByTestId(`qty-input-${objId1}`);
    await expect(qty1B).toHaveValue('3');
    await expect(page.getByTestId('override-indicator')).toHaveCount(0);
  });

  test('per-image note textarea persists across tab switches and is bound per-image', async ({ page }) => {
    await navigateToBriefEditor(page);

    await page.getByTestId('tab-image-room-1').click();
    const note1 = page.getByTestId('per-image-note-room-1');
    await note1.fill('Heavy on red blooms');

    // Switch tabs.
    await page.getByTestId('tab-image-room-2').click();
    const note2 = page.getByTestId('per-image-note-room-2');
    await expect(note2).toHaveValue('');
    await note2.fill('Cool greens only');

    // Switch back: room-1 note still there.
    await page.getByTestId('tab-image-room-1').click();
    await expect(page.getByTestId('per-image-note-room-1')).toHaveValue('Heavy on red blooms');
  });

  test('deleting an ObjectEntry from the palette prunes its override entries from every image', async ({ page }) => {
    const { objId1 } = await navigateToBriefEditor(page);

    // First, set an override on room-1 for Lavender (objId1).
    await page.getByTestId('tab-image-room-1').click();
    await page.getByTestId(`qty-input-${objId1}`).fill('11');
    await page.getByTestId(`qty-input-${objId1}`).blur();
    await expect(page.getByTestId('override-indicator')).toBeVisible();

    // Go back to Default Palette and delete the Lavender row.
    await page.getByTestId('tab-default-palette').click();
    await page.getByTestId(`palette-delete-${objId1}`).click();

    // Re-visit room-1 tab — Lavender's row should be gone (its override too).
    await page.getByTestId('tab-image-room-1').click();
    await expect(page.getByTestId(`qty-input-${objId1}`)).toHaveCount(0);
    // No override indicator anywhere (the only override was on Lavender).
    await expect(page.getByTestId('override-indicator')).toHaveCount(0);
  });

  test('empty palette renders empty-state on per-image tab', async ({ page }) => {
    const { objId1, objId2 } = await navigateToBriefEditor(page);

    // Delete BOTH palette rows (Lavender + Pine).
    await page.getByTestId('tab-default-palette').click();
    await page.getByTestId(`palette-delete-${objId1}`).click();
    await page.getByTestId(`palette-delete-${objId2}`).click();

    await page.getByTestId('tab-image-room-1').click();
    await expect(page.getByText(/No objects in the palette yet/i)).toBeVisible();
  });
});
