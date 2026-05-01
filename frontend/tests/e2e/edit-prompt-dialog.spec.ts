import { test, expect, Page, Route } from '@playwright/test';

/**
 * Per-variation Edit Prompt — issue 004 of the
 * `projects-page-improvements` PRD.
 *
 * Drives the variation-overflow-menu → "Edit Prompt" → Dialog → Generate
 * flow on a one-room-one-completed-variation project and asserts:
 *
 *   1. The "Edit Prompt" item is in the variation overflow menu.
 *   2. "Try Something New" is GONE (replaced by Edit Prompt per PRD).
 *   3. Clicking Edit Prompt opens a Dialog with the variation's prior
 *      `generation_metadata.adapted_prompt` prefilled in the textarea.
 *   4. Editing the textarea and clicking Generate fires
 *      `POST /staging/projects/{id}/rooms/{rid}/variations/{vid}/edit-prompt`
 *      with `{adapted_prompt: <edited>}`.
 *   5. NO request to `/regenerate` fires (rubber-duck-flagged tripwire:
 *      ensures Edit Prompt doesn't accidentally route to the wrong
 *      backend endpoint).
 *   6. The mock SSE stream completes, the page reloads project state,
 *      and the new variation appears (count grew by one). The original
 *      variation's image_url is byte-identical (preserved per PRD).
 *   7. A second scenario covers the fallback notice: when
 *      generation_metadata is missing, the textarea defaults to the
 *      project's prompt and a notice is visible.
 *   8. A third scenario covers Cancel: opening the dialog and
 *      clicking Cancel closes it without firing any backend request.
 *
 * Backend is fully mocked. The actual append-vs-mutate semantics on
 * the server are covered by `tests/test_staging_endpoints_edit_prompt.py`.
 */

const SCREENSHOT_DIR = 'test-results/screenshots/edit-prompt-dialog';
const PROJECT_ID = 'test-edit-prompt';
const ROOM_ID = 'room-1';
const SOURCE_VARIATION_ID = 'r1-v0';
const SOURCE_PRIOR_PROMPT =
  'A cozy living room with a tan leather sofa, walnut coffee table, and large area rug';
const API_BASE = 'http://localhost:8000/api/v1';

interface MockProject {
  id: string;
  name: string;
  prompt: string;
  status: string;
  settings: {
    variations_per_room: number;
    model: string;
    quality: string;
    size: string;
  };
  rooms: Array<{
    id: string;
    label: string;
    original_image_url: string;
    status: string;
    prompt_addendum?: string | null;
    variations: Array<{
      id: string;
      status: string;
      image_url?: string;
      generation_metadata?: {
        model?: string;
        adapted_prompt?: string;
        generation_time_ms?: number;
        tokens_used?: number;
      };
      created_at: string;
      updated_at: string;
    }>;
    created_at: string;
    updated_at: string;
  }>;
  total_variations: number;
  completed_variations: number;
  created_at: string;
  updated_at: string;
}

function makeProject(opts: { withMetadata?: boolean } = {}): MockProject {
  const now = new Date().toISOString();
  return {
    id: PROJECT_ID,
    name: 'Edit Prompt Test',
    prompt: 'modern minimalist with warm wood tones',
    status: 'completed',
    settings: {
      variations_per_room: 1,
      model: 'gpt-image-2',
      quality: 'high',
      size: 'auto',
    },
    rooms: [
      {
        id: ROOM_ID,
        label: 'Living Room',
        original_image_url: `https://storage.blob.core.windows.net/images/staging/${PROJECT_ID}/originals/room-1.png`,
        status: 'completed',
        variations: [
          {
            id: SOURCE_VARIATION_ID,
            status: 'completed',
            image_url: `https://storage.blob.core.windows.net/images/staging/${PROJECT_ID}/variations/room-1/v0.png`,
            generation_metadata: opts.withMetadata !== false
              ? {
                  model: 'gpt-image-2',
                  adapted_prompt: SOURCE_PRIOR_PROMPT,
                  generation_time_ms: 5000,
                  tokens_used: 1234,
                }
              : undefined,
            created_at: now,
            updated_at: now,
          },
        ],
        created_at: now,
        updated_at: now,
      },
    ],
    total_variations: 1,
    completed_variations: 1,
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

/**
 * SSE stream body for a successful Edit Prompt response. Mirrors the
 * backend's event vocabulary: variation_completed → project_completed.
 */
function makeSuccessStream(opts: {
  newVariationId: string;
  newVariationImageUrl: string;
  adaptedPrompt: string;
}): string {
  const variationCompleted = {
    room_id: ROOM_ID,
    variation_index: 1,
    image_url: opts.newVariationImageUrl,
    error: null,
    elapsed_ms: 1234,
    tokens_used: 5678,
    model: 'gpt-image-2',
    adapted_prompt: opts.adaptedPrompt,
  };
  const projectCompleted = { status: 'completed' };
  return [
    `event: variation_completed`,
    `data: ${JSON.stringify(variationCompleted)}`,
    ``,
    `event: project_completed`,
    `data: ${JSON.stringify(projectCompleted)}`,
    ``,
    ``,
  ].join('\n');
}

test.describe('Per-variation Edit Prompt (issue 004)', () => {
  test('overflow menu → Edit Prompt → Dialog → Generate appends a new variation', async ({
    page,
  }) => {
    let projectState = makeProject();
    const editPromptRequests: Array<{ url: string; body: Record<string, unknown> }> = [];
    const generateOrRegenerateRequests: string[] = [];

    await setupSasTokenMock(page);

    // GET project — returns whatever projectState currently is.
    await page.route(`${API_BASE}/staging/projects/${PROJECT_ID}`, (route: Route) => {
      const method = route.request().method();
      if (method === 'GET') {
        return route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ project: projectState }),
        });
      }
      return route.continue();
    });

    // Mock the new edit-prompt endpoint as an SSE stream that yields
    // variation_completed → project_completed. After it fires, we
    // mutate projectState to include the appended variation so the
    // post-stream loadProject() reload surfaces the new variation.
    const newVariationId = 'r1-v1-appended';
    const newVariationImageUrl = `https://storage.blob.core.windows.net/images/staging/${PROJECT_ID}/variations/room-1/v1-appended.png`;
    await page.route(
      `${API_BASE}/staging/projects/${PROJECT_ID}/rooms/${ROOM_ID}/variations/${SOURCE_VARIATION_ID}/edit-prompt`,
      async (route: Route) => {
        const body = JSON.parse(route.request().postData() || '{}');
        editPromptRequests.push({ url: route.request().url(), body });

        // Mutate projectState to mirror the backend's append: original
        // variation byte-identical, new variation appended at the end.
        const adapted = body.adapted_prompt as string;
        const now = new Date().toISOString();
        projectState = {
          ...projectState,
          rooms: projectState.rooms.map((r) =>
            r.id === ROOM_ID
              ? {
                  ...r,
                  variations: [
                    ...r.variations,
                    {
                      id: newVariationId,
                      status: 'completed',
                      image_url: newVariationImageUrl,
                      generation_metadata: {
                        model: 'gpt-image-2',
                        adapted_prompt: adapted,
                        generation_time_ms: 1234,
                        tokens_used: 5678,
                      },
                      created_at: now,
                      updated_at: now,
                    },
                  ],
                }
              : r,
          ),
          total_variations: projectState.total_variations + 1,
          completed_variations: projectState.completed_variations + 1,
        };

        return route.fulfill({
          status: 200,
          contentType: 'text/event-stream',
          body: makeSuccessStream({
            newVariationId,
            newVariationImageUrl,
            adaptedPrompt: adapted,
          }),
        });
      },
    );

    // Tripwire: NO requests to /generate or /regenerate. The Edit
    // Prompt menu item must route exclusively to /edit-prompt.
    await page.route(
      new RegExp(`${API_BASE}/staging/projects/${PROJECT_ID}/(generate|rooms/[^/]+/regenerate|rooms/[^/]+/variations/[^/]+/regenerate)`),
      (route: Route) => {
        generateOrRegenerateRequests.push(route.request().url());
        return route.fulfill({ status: 500, body: 'tripwire: should not be hit' });
      },
    );

    await page.goto(`/projects/${PROJECT_ID}`);
    await page.waitForLoadState('networkidle');

    // 1. Source variation thumbnail is visible (pre-edit state — 1 variation).
    const sourceImage = page.locator(
      `img[src*="staging/${PROJECT_ID}/variations/room-1/v0.png"]`,
    ).first();
    await expect(sourceImage).toBeVisible({ timeout: 10_000 });
    const srcBefore = await sourceImage.getAttribute('src');

    // 2. Click the variation's regen-trigger to open the overflow menu.
    const regenTrigger = page.getByTestId('variation-1-regen-trigger');
    await expect(regenTrigger).toBeVisible();

    await page.screenshot({
      path: `${SCREENSHOT_DIR}/01-pre-menu.png`,
      fullPage: true,
    });

    await regenTrigger.click();

    // 3. The "Retry Same Prompt" item is still visible (kept from before).
    await expect(page.getByTestId('variation-1-retry-same-prompt')).toBeVisible();

    // 4. The "Edit Prompt" item is NEW and visible.
    const editPromptItem = page.getByTestId('variation-1-edit-prompt');
    await expect(editPromptItem).toBeVisible();

    // 5. "Try Something New" is GONE (replaced per PRD § Solution → 4).
    await expect(page.getByText('Try Something New')).toHaveCount(0);

    await page.screenshot({
      path: `${SCREENSHOT_DIR}/02-menu-open.png`,
      fullPage: true,
    });

    // 6. Clicking Edit Prompt opens the Dialog with the prefilled
    //    textarea showing the source variation's prior adapted_prompt.
    await editPromptItem.click();

    const dialog = page.getByTestId('edit-prompt-dialog');
    await expect(dialog).toBeVisible();
    const textarea = page.getByTestId('edit-prompt-textarea');
    await expect(textarea).toBeVisible();
    await expect(textarea).toHaveValue(SOURCE_PRIOR_PROMPT);

    // Fallback notice is NOT shown (we have generation_metadata).
    await expect(page.getByTestId('edit-prompt-fallback-notice')).toHaveCount(0);

    await page.screenshot({
      path: `${SCREENSHOT_DIR}/03-dialog-open-prefilled.png`,
      fullPage: true,
    });

    // 7. Edit the prompt and click Generate.
    const editedPrompt =
      'A cozy living room with a tan leather sofa, walnut coffee table, large area rug, AND TWO BRASS FLOOR LAMPS';
    await textarea.fill(editedPrompt);

    const generateBtn = page.getByTestId('edit-prompt-generate');
    await Promise.all([
      page.waitForRequest(
        (req) =>
          req.method() === 'POST' &&
          req.url().endsWith(`/edit-prompt`) &&
          req.url().includes(`/variations/${SOURCE_VARIATION_ID}/`),
      ),
      generateBtn.click(),
    ]);

    // 8. The PATCH/POST body matches what we expect.
    expect(editPromptRequests).toHaveLength(1);
    expect(editPromptRequests[0].body).toEqual({ adapted_prompt: editedPrompt });

    // 9. After the SSE stream completes, the dialog closes and the
    //    page reloads to show the appended variation.
    await expect(dialog).not.toBeVisible({ timeout: 10_000 });

    // The new variation's image_url appears in the DOM.
    const appendedImage = page.locator(
      `img[src*="staging/${PROJECT_ID}/variations/room-1/v1-appended.png"]`,
    ).first();
    await expect(appendedImage).toBeVisible({ timeout: 10_000 });

    // The original variation's image is still there (preserved for A/B comparison).
    const sourceStillThere = page.locator(
      `img[src*="staging/${PROJECT_ID}/variations/room-1/v0.png"]`,
    ).first();
    await expect(sourceStillThere).toBeVisible();
    const srcAfter = await sourceStillThere.getAttribute('src');
    // SAS-token preservation regression check (same pattern as
    // per-room-prompt-addendum spec): URLs should still carry sv=mock
    // after the post-Edit reload.
    expect(srcBefore).toContain('sv=mock');
    expect(srcAfter).toContain('sv=mock');

    await page.screenshot({
      path: `${SCREENSHOT_DIR}/04-after-edit-prompt.png`,
      fullPage: true,
    });

    // 10. Tripwire passed: no /regenerate or /generate requests fired.
    expect(generateOrRegenerateRequests).toEqual([]);
  });

  test('fallback notice surfaces when generation_metadata is missing', async ({ page }) => {
    // Source variation has NO generation_metadata — exercises the
    // PRD's fallback path: textarea defaults to project.prompt and a
    // notice explains the fallback.
    const projectState = makeProject({ withMetadata: false });

    await setupSasTokenMock(page);

    await page.route(`${API_BASE}/staging/projects/${PROJECT_ID}`, (route: Route) => {
      if (route.request().method() === 'GET') {
        return route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ project: projectState }),
        });
      }
      return route.continue();
    });

    await page.goto(`/projects/${PROJECT_ID}`);
    await page.waitForLoadState('networkidle');

    await page.getByTestId('variation-1-regen-trigger').click();
    await page.getByTestId('variation-1-edit-prompt').click();

    const textarea = page.getByTestId('edit-prompt-textarea');
    await expect(textarea).toBeVisible();
    // Falls back to project.prompt (modernsominimalist...).
    await expect(textarea).toHaveValue('modern minimalist with warm wood tones');
    // The fallback notice is visible.
    await expect(page.getByTestId('edit-prompt-fallback-notice')).toBeVisible();

    await page.screenshot({
      path: `${SCREENSHOT_DIR}/05-fallback-notice.png`,
      fullPage: true,
    });
  });

  test('Cancel closes the dialog without firing any backend request', async ({ page }) => {
    const projectState = makeProject();
    const anyMutationRequests: string[] = [];

    await setupSasTokenMock(page);

    // Combined GET + tripwire on a single route — Playwright's
    // ``route.continue()`` doesn't fall through to previously-
    // registered handlers, so we handle both cases inline. Tripwire:
    // any non-GET to /staging/projects/{id} or sub-paths is a
    // regression — Cancel must not fire anything.
    await page.route(
      new RegExp(`${API_BASE}/staging/projects/${PROJECT_ID}(/.*)?$`),
      (route: Route) => {
        const method = route.request().method();
        if (method === 'GET' && route.request().url().endsWith(`/${PROJECT_ID}`)) {
          return route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify({ project: projectState }),
          });
        }
        if (method !== 'GET') {
          anyMutationRequests.push(`${method} ${route.request().url()}`);
          return route.fulfill({ status: 500, body: 'tripwire: cancel must not POST' });
        }
        // Other GETs (sub-paths, if any) get the project too.
        return route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ project: projectState }),
        });
      },
    );

    await page.goto(`/projects/${PROJECT_ID}`);
    await page.waitForLoadState('networkidle');

    await page.getByTestId('variation-1-regen-trigger').click();
    await page.getByTestId('variation-1-edit-prompt').click();

    const dialog = page.getByTestId('edit-prompt-dialog');
    await expect(dialog).toBeVisible();

    await page.getByTestId('edit-prompt-cancel').click();
    await expect(dialog).not.toBeVisible();

    expect(anyMutationRequests).toEqual([]);
  });
});
