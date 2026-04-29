import { test, expect, Page } from '@playwright/test';
import { join } from 'node:path';

const FIXTURES = join(__dirname, 'fixtures');

// --- Mock helpers ---

/** Intercept all staging API calls with realistic mock responses. */
async function mockStagingApi(page: Page) {
  const projectId = 'test-project-' + Date.now();

  // POST /staging/projects — create project
  await page.route('**/api/v1/staging/projects', async (route, request) => {
    if (request.method() === 'POST') {
      const body = request.postDataJSON?.() ?? {};
      await route.fulfill({
        status: 201,
        contentType: 'application/json',
        body: JSON.stringify({
          project: {
            id: projectId,
            name: body.name ?? 'Test Project',
            prompt: body.prompt ?? '',
            status: 'uploading',
            rooms: [],
            settings: { variations_per_room: 1, model: 'gpt-image-2', quality: 'high', size: 'auto' },
            created_at: new Date().toISOString(),
          },
        }),
      });
    } else {
      // GET /staging/projects — list
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ projects: [], total: 0 }),
      });
    }
  });

  // POST /staging/projects/:id/rooms — upload rooms
  await page.route('**/api/v1/staging/projects/*/rooms', async (route) => {
    await new Promise((r) => setTimeout(r, 500)); // simulate upload delay
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        project_id: projectId,
        rooms_added: 2,
        rooms: [
          { id: 'room-1', label: 'Backyard', original_image_url: 'https://example.com/room1.png', status: 'pending', variations: [{ id: 'v-1', status: 'pending' }] },
          { id: 'room-2', label: 'Patio', original_image_url: 'https://example.com/room2.png', status: 'pending', variations: [{ id: 'v-2', status: 'pending' }] },
        ],
      }),
    });
  });

  // POST /staging/projects/:id/analyze — analyze images
  await page.route('**/api/v1/staging/projects/*/analyze', async (route) => {
    await new Promise((r) => setTimeout(r, 800)); // simulate analysis delay
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        analyses: [
          { room_id: 'room-1', description: 'A backyard with a wooden fence and grass lawn', features: ['Outdoor space', 'Fence', 'Lawn'], zones: ['Along fence line', 'Center of lawn'] },
          { room_id: 'room-2', description: 'A stone patio with seating area', features: ['Patio', 'Seating', 'Stone pavers'], zones: ['Patio border', 'Around seating'] },
        ],
        failed_count: 0,
      }),
    });
  });

  // POST /staging/projects/:id/chat — design chat
  let chatCount = 0;
  await page.route('**/api/v1/staging/projects/*/chat', async (route) => {
    chatCount++;
    await new Promise((r) => setTimeout(r, 300));
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        reply: chatCount === 1
          ? "Great! I can help you visualize some landscaping ideas. What style are you going for?"
          : "That sounds wonderful! I have a clear picture of what you want. Would you like me to generate a design brief?",
        ready_for_brief: chatCount >= 2,
        suggested_actions: chatCount >= 2 ? ['generate_brief'] : ['choose_style', 'specify_species'],
      }),
    });
  });

  // POST /staging/projects/:id/brief — generate brief
  await page.route('**/api/v1/staging/projects/*/brief', async (route, request) => {
    if (request.method() === 'POST') {
      await new Promise((r) => setTimeout(r, 1000)); // simulate brief generation
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          brief: {
            global_instructions: 'Add lush greenery along the fence line with layered heights',
            plant_palette: [{ species: 'Japanese Maple', quantity: 1, size: 'medium', placement: 'Corner accent', botanical_name: 'Acer palmatum' }],
            placement_guide: { back_row: 'Tall ornamental grasses', front_row: 'Low ground cover', middle_row: 'Medium shrubs' },
            per_image_notes: {},
            preserve_elements: ['Existing fence', 'Patio pavers'],
            settings: { variations_per_room: 1, model: 'gpt-image-2', quality: 'high', size: 'auto' },
          },
        }),
      });
    }
  });

  // GET /staging/projects/:id — get project detail
  // Use a glob pattern that matches the project ID path but not sub-paths
  await page.route(`**/staging/projects/${projectId}`, async (route, request) => {
    if (request.method() === 'GET') {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          project: {
            id: projectId,
            name: 'Backyard Landscaping',
            prompt: 'Lush greenery along fence',
            status: 'completed',
            rooms: [
              {
                id: 'room-1', label: 'Backyard', original_image_url: 'https://example.com/room1.png', status: 'completed',
                variations: [{ id: 'v-1', status: 'completed', image_url: 'https://example.com/v1.png' }],
              },
            ],
            settings: { variations_per_room: 1, model: 'gpt-image-2', quality: 'high', size: 'auto' },
            created_at: new Date().toISOString(),
          },
        }),
      });
    } else {
      await route.continue();
    }
  });

  // SAS token endpoint (on the backend API, not Next.js)
  await page.route('**/gallery/sas-tokens', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ imageSasToken: 'sv=mock', videoSasToken: 'sv=mock' }),
    });
  });

  return projectId;
}


// ===== TEST SUITE 1: Wizard Step Navigation & Stepper =====

test.describe('New Project Wizard — Navigation & Stepper', () => {
  test('step 1: name input, stepper shows labels, Next button advances', async ({ page }) => {
    await page.goto('/projects/new');
    await page.waitForLoadState('domcontentloaded');

    // Verify step 1 is active
    await expect(page.getByText('Choose a name for your project')).toBeVisible();
    await expect(page.getByText('New Project').first()).toBeVisible();

    // Verify stepper shows step labels
    await expect(page.getByText('Name', { exact: true })).toBeVisible();
    await expect(page.getByText('Upload', { exact: true })).toBeVisible();

    // Next button should be disabled without a name
    const nextBtn = page.getByRole('button', { name: 'Next', exact: true });
    await expect(nextBtn).toBeDisabled();

    // Type a project name
    await page.getByPlaceholder('e.g., Backyard Fence Line').fill('Test Backyard Project');

    // Next button should be enabled now
    await expect(nextBtn).toBeEnabled();

    // Click Next
    await nextBtn.click();

    // Should be on step 2
    await expect(page.getByText('Upload baseline photos')).toBeVisible();
    await expect(page.getByText('Click to upload or drag and drop')).toBeVisible();

    await page.screenshot({ path: 'test-results/screenshots/wizard/step2-upload.png', fullPage: true });
  });

  test('step 1: Enter key advances to step 2', async ({ page }) => {
    await page.goto('/projects/new');
    await page.waitForLoadState('domcontentloaded');

    const input = page.getByPlaceholder('e.g., Backyard Fence Line');
    await input.fill('Enter Key Test');
    await input.press('Enter');

    // Should be on step 2
    await expect(page.getByText('Upload baseline photos')).toBeVisible();
  });
});


// ===== TEST SUITE 2: Upload Experience (Step 2) =====

test.describe('New Project Wizard — Upload Experience', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/projects/new');
    await page.waitForLoadState('domcontentloaded');
    // Navigate to step 2
    await page.getByPlaceholder('e.g., Backyard Fence Line').fill('Upload Test');
    await page.getByRole('button', { name: 'Next', exact: true }).click();
    await expect(page.getByText('Click to upload or drag and drop')).toBeVisible();
  });

  test('file input uploads and shows photo grid', async ({ page }) => {
    // Upload files using the file chooser
    const fileInput = page.locator('#room-upload');
    await fileInput.setInputFiles([
      join(FIXTURES, 'test-room-1.png'),
      join(FIXTURES, 'test-room-2.png'),
    ]);

    // Photos should appear in grid
    await expect(page.getByText('Photos (2)')).toBeVisible();

    // Photo name inputs should be visible
    const nameInputs = page.locator('input[placeholder="Label this photo"]');
    await expect(nameInputs).toHaveCount(2);

    // "Add more" button should be visible
    await expect(page.getByRole('button', { name: 'Add more' })).toBeVisible();

    // Next button should be enabled
    await expect(page.getByRole('button', { name: 'Next', exact: true })).toBeEnabled();

    await page.screenshot({ path: 'test-results/screenshots/wizard/step2-photos-uploaded.png', fullPage: true });
  });

  test('photo labels are editable', async ({ page }) => {
    const fileInput = page.locator('#room-upload');
    await fileInput.setInputFiles([join(FIXTURES, 'test-room-1.png')]);

    await expect(page.getByText('Photos (1)')).toBeVisible();

    // Edit the label
    const labelInput = page.locator('input[placeholder="Label this photo"]');
    await labelInput.clear();
    await labelInput.fill('My Backyard');
    await expect(labelInput).toHaveValue('My Backyard');
  });

  test('remove button removes a photo', async ({ page }) => {
    const fileInput = page.locator('#room-upload');
    await fileInput.setInputFiles([
      join(FIXTURES, 'test-room-1.png'),
      join(FIXTURES, 'test-room-2.png'),
    ]);

    await expect(page.getByText('Photos (2)')).toBeVisible();

    // Hover over first photo to reveal remove button, then click it
    const removeBtn = page.locator('[data-slot="card"] .group button[variant="destructive"], [data-slot="card"] .group button.bg-destructive').first();
    // If that doesn't match, try the destructive button within the card content
    const photosSection = page.locator('.grid').filter({ has: page.locator('img') });
    const firstPhoto = photosSection.locator('.group').first();
    await firstPhoto.hover();
    await firstPhoto.locator('button').click();

    // Should now show 1 photo
    await expect(page.getByText('Photos (1)')).toBeVisible();
  });

  test('Next button is disabled without photos', async ({ page }) => {
    const nextBtn = page.getByRole('button', { name: 'Next', exact: true });
    await expect(nextBtn).toBeDisabled();
  });
});


// ===== TEST SUITE 3: Step 2→3 Transition with Progress =====

test.describe('New Project Wizard — Non-blocking Step Transition', () => {
  test('step 2→3 advances immediately and shows progress phases', async ({ page }) => {
    await mockStagingApi(page);
    await page.goto('/projects/new');
    await page.waitForLoadState('domcontentloaded');

    // Step 1: Name
    await page.getByPlaceholder('e.g., Backyard Fence Line').fill('Progress Test Project');
    await page.getByRole('button', { name: 'Next', exact: true }).click();

    // Step 2: Upload
    const fileInput = page.locator('#room-upload');
    await fileInput.setInputFiles([
      join(FIXTURES, 'test-room-1.png'),
      join(FIXTURES, 'test-room-2.png'),
    ]);
    await expect(page.getByText('Photos (2)')).toBeVisible();

    // Click Next — should advance to step 3 IMMEDIATELY (not block)
    await page.getByRole('button', { name: 'Next', exact: true }).click();

    // Should see progress phase text (one of creating/uploading/analyzing)
    // The step 3 view should be visible with progress indicators
    const progressVisible = await Promise.race([
      page.getByText('Creating project...').waitFor({ timeout: 3000 }).then(() => 'creating'),
      page.getByText(/Uploading \d+ photo/).waitFor({ timeout: 3000 }).then(() => 'uploading'),
      page.getByText('Analyzing your photos...').waitFor({ timeout: 3000 }).then(() => 'analyzing'),
    ]).catch(() => null);

    expect(progressVisible).toBeTruthy();

    await page.screenshot({ path: 'test-results/screenshots/wizard/step3-progress.png', fullPage: true });

    // Stepper should show step 2 with spinner (in-progress indicator)
    // Wait for the phase to complete — chat should eventually appear
    await expect(page.getByText("Here's what I see")).toBeVisible({ timeout: 15000 });

    await page.screenshot({ path: 'test-results/screenshots/wizard/step3-chat-ready.png', fullPage: true });
  });

  test('step 3 error state shows retry and back buttons', async ({ page }) => {
    // Mock API with failure
    await page.route('**/api/v1/staging/projects', async (route, request) => {
      if (request.method() === 'POST') {
        await route.fulfill({ status: 500, contentType: 'application/json', body: JSON.stringify({ detail: 'Service unavailable' }) });
      }
    });

    await page.goto('/projects/new');
    await page.waitForLoadState('domcontentloaded');

    await page.getByPlaceholder('e.g., Backyard Fence Line').fill('Error Test');
    await page.getByRole('button', { name: 'Next', exact: true }).click();

    const fileInput = page.locator('#room-upload');
    await fileInput.setInputFiles([join(FIXTURES, 'test-room-1.png')]);
    await page.getByRole('button', { name: 'Next', exact: true }).click();

    // Should show error state with retry options
    await expect(page.getByText('Something went wrong')).toBeVisible({ timeout: 10000 });
    await expect(page.getByRole('button', { name: 'Back to photos' })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Try again' })).toBeVisible();

    await page.screenshot({ path: 'test-results/screenshots/wizard/step3-error-state.png', fullPage: true });
  });
});


// ===== TEST SUITE 4: Design Chat — Brief Generation UX =====

test.describe('Design Chat — Brief Generation', () => {
  /** Helper: navigate through wizard to step 3 with chat ready */
  async function navigateToChatReady(page: Page) {
    await mockStagingApi(page);
    await page.goto('/projects/new');
    await page.waitForLoadState('domcontentloaded');

    await page.getByPlaceholder('e.g., Backyard Fence Line').fill('Chat Test Project');
    await page.getByRole('button', { name: 'Next', exact: true }).click();

    const fileInput = page.locator('#room-upload');
    await fileInput.setInputFiles([join(FIXTURES, 'test-room-1.png'), join(FIXTURES, 'test-room-2.png')]);
    await page.getByRole('button', { name: 'Next', exact: true }).click();

    // Wait for prep to complete and chat to be ready
    await expect(page.getByText("Here's what I see")).toBeVisible({ timeout: 15000 });
  }

  test('chat shows initial AI message and input is enabled', async ({ page }) => {
    await navigateToChatReady(page);

    // AI initial message should be visible
    await expect(page.getByText('A backyard with a wooden fence')).toBeVisible();

    // Chat input should be enabled
    const chatInput = page.locator('input[placeholder*="visualize"]');
    await expect(chatInput).toBeEnabled();

    await page.screenshot({ path: 'test-results/screenshots/wizard/step3-chat-initial.png', fullPage: true });
  });

  test('sending messages enables Generate Design Brief button', async ({ page }) => {
    await navigateToChatReady(page);

    // Brief button should NOT be visible before conversation
    await expect(page.getByRole('button', { name: 'Generate Design Brief', exact: true })).not.toBeVisible();

    // Send first message
    let chatInput = page.locator('input[placeholder*="visualize"]');
    await chatInput.fill('I want tropical plants along the fence');
    await chatInput.press('Enter');

    // Wait for AI response
    await expect(page.getByText('What style are you going for')).toBeVisible({ timeout: 5000 });

    // After first exchange, placeholder may still be the original or may have changed
    // Re-locate the chat input by finding the one near the Send button
    chatInput = page.locator('input[placeholder*="visualize"], input[placeholder*="go ahead"]').first();
    await chatInput.fill('Modern minimalist with ornamental grasses');
    await chatInput.press('Enter');

    // Wait for AI response
    await expect(page.getByText('Would you like me to generate')).toBeVisible({ timeout: 5000 });

    // Brief button should now be visible (canGenerateBrief = true after 2 exchanges)
    await expect(page.getByRole('button', { name: 'Generate Design Brief', exact: true })).toBeVisible();

    // Quick reply chip should also show "Generate Design Brief"
    await expect(page.getByText('📋 Generate Design Brief')).toBeVisible();

    // Placeholder should hint about proceed intent
    await expect(page.locator('input[placeholder*="go ahead"]')).toBeVisible();

    await page.screenshot({ path: 'test-results/screenshots/wizard/step3-brief-button-visible.png', fullPage: true });
  });

  test('clicking Generate Design Brief shows loading overlay then advances to step 4', async ({ page }) => {
    await navigateToChatReady(page);

    // Build conversation (2 exchanges)
    let chatInput = page.locator('input[placeholder*="visualize"]');
    await chatInput.fill('Tropical fence line plants');
    await chatInput.press('Enter');
    await expect(page.getByText('What style are you going for')).toBeVisible({ timeout: 5000 });

    chatInput = page.locator('input[placeholder*="visualize"], input[placeholder*="go ahead"]').first();
    await chatInput.fill('Lush and green, lots of texture');
    await chatInput.press('Enter');
    await expect(page.getByText('Would you like me to generate')).toBeVisible({ timeout: 5000 });

    // Click Generate Design Brief button
    await page.getByRole('button', { name: 'Generate Design Brief', exact: true }).click();

    // Loading overlay should appear
    await expect(page.getByText('Creating your Design Brief...')).toBeVisible({ timeout: 3000 });
    await expect(page.getByText('Analyzing the conversation')).toBeVisible();

    await page.screenshot({ path: 'test-results/screenshots/wizard/step3-brief-loading.png', fullPage: true });

    // Should advance to step 4 after brief is generated
    await expect(page.getByText('Global Instructions')).toBeVisible({ timeout: 10000 });
    await expect(page.getByText('Plant Palette')).toBeVisible();

    await page.screenshot({ path: 'test-results/screenshots/wizard/step4-brief-editor.png', fullPage: true });
  });

  test('typing "go ahead" triggers brief generation directly', async ({ page }) => {
    await navigateToChatReady(page);

    // Build the minimum conversation
    let chatInput = page.locator('input[placeholder*="visualize"]');
    await chatInput.fill('Add ornamental grasses along the fence');
    await chatInput.press('Enter');
    await expect(page.getByText('What style are you going for')).toBeVisible({ timeout: 5000 });

    chatInput = page.locator('input[placeholder*="visualize"], input[placeholder*="go ahead"]').first();
    await chatInput.fill('Natural prairie style');
    await chatInput.press('Enter');
    await expect(page.getByText('Would you like me to generate')).toBeVisible({ timeout: 5000 });

    // Type proceed intent — should trigger brief generation without AI round-trip
    chatInput = page.locator('input[placeholder*="go ahead"]');
    await chatInput.fill("Sounds good, let's go ahead");
    await chatInput.press('Enter');

    // Loading overlay should appear (brief generation triggered directly)
    await expect(page.getByText('Creating your Design Brief...')).toBeVisible({ timeout: 3000 });

    // Should advance to step 4
    await expect(page.getByText('Global Instructions')).toBeVisible({ timeout: 10000 });
  });
});


// ===== TEST SUITE 5: Project Detail — Status Display =====

test.describe('Project Detail — Status Display', () => {
  test('completed project shows correct status badges and variation counts', async ({ page }) => {
    const projectId = await mockStagingApi(page);

    // Navigate and wait for the project API response to be received
    const [response] = await Promise.all([
      page.waitForResponse(resp => resp.url().includes(`/staging/projects/${projectId}`) && resp.status() === 200),
      page.goto(`/projects/${projectId}`),
    ]);

    // Now wait for React to render the actual content
    await expect(page.getByText('Backyard Landscaping')).toBeVisible({ timeout: 10000 });

    // Project header should show completed status badge
    await expect(page.getByText('completed').first()).toBeVisible();

    // Variation count should show
    await expect(page.getByText('1/1 variations complete')).toBeVisible();

    // Room group should be visible with correct status
    await expect(page.getByText('Backyard', { exact: true }).first()).toBeVisible();
    // Room has "completed" badge and "1/1 variations" count (no status message needed when fully complete)
    await expect(page.getByText('1/1 variations complete')).toBeVisible();

    await page.screenshot({ path: 'test-results/screenshots/project/detail-completed.png', fullPage: true });
  });
});
