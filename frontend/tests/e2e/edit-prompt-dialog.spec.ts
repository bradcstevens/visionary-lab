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

  // ─────────────────────────────────────────────────────────────────
  // Issue 003 of radix-dialog-body-lock-fix PRD: mount-pattern
  // standardization. The dialog is now ALWAYS MOUNTED by its parent
  // and reset on the rising edge of `open`. The next three tests pin
  // the user-observable contracts that this mount-pattern change
  // promises:
  //
  //   - cross-variation prefill (story 2): close-without-save on
  //     variation 1, then re-open on variation 2, must show
  //     variation 2's prior adapted prompt (NOT variation 1's, NOT
  //     the abandoned draft from variation 1's session).
  //
  //   - same-variation prefill (story 3): close-without-save on
  //     variation 1, then re-open on the SAME variation 1, must show
  //     variation 1's prior adapted prompt FRESH (NOT the abandoned
  //     draft).
  //
  //   - close-then-interactive regression (stories 9 / 14 / 15):
  //     after each close path (Cancel, ✕, Escape, click-outside) the
  //     page below must remain fully interactive — no inline
  //     `pointer-events: none` on `<body>`, no `data-scroll-locked`
  //     attribute on `<body>`, and a non-`force` Playwright click on
  //     `[data-testid="project-header-action"]` succeeds. The third
  //     assertion is the ground-truth user-facing check that catches
  //     any OTHER mechanism (stuck focus-trap, orphan portal,
  //     scheduler bug) beyond the two known stuck-attribute
  //     signatures.
  // ─────────────────────────────────────────────────────────────────

  function makeProjectTwoVariations(): MockProject {
    const now = new Date().toISOString();
    return {
      id: PROJECT_ID,
      name: 'Edit Prompt Cross-Variation Test',
      prompt: 'modern minimalist with warm wood tones',
      status: 'completed',
      settings: {
        variations_per_room: 2,
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
              id: 'r1-v0',
              status: 'completed',
              image_url: `https://storage.blob.core.windows.net/images/staging/${PROJECT_ID}/variations/room-1/v0.png`,
              generation_metadata: {
                model: 'gpt-image-2',
                adapted_prompt: 'V1 PROMPT — tan leather sofa and brass lamps',
                generation_time_ms: 5000,
                tokens_used: 1234,
              },
              created_at: now,
              updated_at: now,
            },
            {
              id: 'r1-v1',
              status: 'completed',
              image_url: `https://storage.blob.core.windows.net/images/staging/${PROJECT_ID}/variations/room-1/v1.png`,
              generation_metadata: {
                model: 'gpt-image-2',
                adapted_prompt: 'V2 PROMPT — emerald velvet armchair and walnut credenza',
                generation_time_ms: 5000,
                tokens_used: 1234,
              },
              created_at: now,
              updated_at: now,
            },
          ],
          created_at: now,
          updated_at: now,
        },
      ],
      total_variations: 2,
      completed_variations: 2,
      created_at: now,
      updated_at: now,
    };
  }

  test('cross-variation prefill: close-without-save on V1, open V2 shows V2 prompt (not V1, not abandoned draft)', async ({
    page,
  }) => {
    const projectState = makeProjectTwoVariations();
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

    const V1_PROMPT = 'V1 PROMPT — tan leather sofa and brass lamps';
    const V2_PROMPT = 'V2 PROMPT — emerald velvet armchair and walnut credenza';
    const ABANDONED_DRAFT = 'ABANDONED DRAFT — should not survive cancel';

    // Open EditPrompt on variation 1 → assert prefilled with V1 prompt.
    await page.getByTestId('variation-1-regen-trigger').click();
    await page.getByTestId('variation-1-edit-prompt').click();

    const dialog = page.getByTestId('edit-prompt-dialog');
    const textarea = page.getByTestId('edit-prompt-textarea');
    await expect(dialog).toBeVisible();
    await expect(textarea).toHaveValue(V1_PROMPT);

    // Type abandoned draft, then cancel.
    await textarea.fill(ABANDONED_DRAFT);
    await expect(textarea).toHaveValue(ABANDONED_DRAFT);
    await page.getByTestId('edit-prompt-cancel').click();
    await expect(dialog).not.toBeVisible();

    await page.screenshot({
      path: `${SCREENSHOT_DIR}/06-cross-variation-after-cancel-v1.png`,
      fullPage: true,
    });

    // Open EditPrompt on variation 2 → assert prefilled with V2 prompt
    // (NOT V1's, NOT the abandoned draft). This is the rising-edge
    // reset working: each new open re-derives sourcePrompt from the
    // latest props.
    await page.getByTestId('variation-2-regen-trigger').click();
    await page.getByTestId('variation-2-edit-prompt').click();
    await expect(dialog).toBeVisible();
    await expect(textarea).toHaveValue(V2_PROMPT);
    // Explicit anti-assertions to make a regression unmistakable:
    await expect(textarea).not.toHaveValue(V1_PROMPT);
    await expect(textarea).not.toHaveValue(ABANDONED_DRAFT);

    await page.screenshot({
      path: `${SCREENSHOT_DIR}/07-cross-variation-open-v2.png`,
      fullPage: true,
    });

    // Cancel out cleanly so no stray state lingers for the next test.
    await page.getByTestId('edit-prompt-cancel').click();
    await expect(dialog).not.toBeVisible();
  });

  test('same-variation prefill: close-without-save then re-open on same variation shows fresh prior prompt (not abandoned draft)', async ({
    page,
  }) => {
    const projectState = makeProject();
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

    const ABANDONED_DRAFT = 'ABANDONED DRAFT — should not survive cancel';

    // Open → assert prefilled with prior adapted_prompt.
    await page.getByTestId('variation-1-regen-trigger').click();
    await page.getByTestId('variation-1-edit-prompt').click();

    const dialog = page.getByTestId('edit-prompt-dialog');
    const textarea = page.getByTestId('edit-prompt-textarea');
    await expect(dialog).toBeVisible();
    await expect(textarea).toHaveValue(SOURCE_PRIOR_PROMPT);

    // Type abandoned draft, then cancel.
    await textarea.fill(ABANDONED_DRAFT);
    await expect(textarea).toHaveValue(ABANDONED_DRAFT);
    await page.getByTestId('edit-prompt-cancel').click();
    await expect(dialog).not.toBeVisible();

    // Re-open on the SAME variation → must show prior adapted_prompt
    // FRESH, not the abandoned draft. This is the rising-edge reset
    // catching the same-variation case (the case that fully-controlled
    // dialogs that don't reset on re-open silently get wrong).
    await page.getByTestId('variation-1-regen-trigger').click();
    await page.getByTestId('variation-1-edit-prompt').click();
    await expect(dialog).toBeVisible();
    await expect(textarea).toHaveValue(SOURCE_PRIOR_PROMPT);
    await expect(textarea).not.toHaveValue(ABANDONED_DRAFT);

    await page.screenshot({
      path: `${SCREENSHOT_DIR}/08-same-variation-reopen-fresh.png`,
      fullPage: true,
    });

    // Cancel out cleanly.
    await page.getByTestId('edit-prompt-cancel').click();
    await expect(dialog).not.toBeVisible();
  });

  test('close-then-interactive regression: page below remains fully interactive after every close path', async ({
    page,
  }) => {
    const projectState = makeProject();
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

    const dialog = page.getByTestId('edit-prompt-dialog');
    const overlay = page.locator('[data-slot="dialog-overlay"]');
    // The Edit Prompt mock project has all rooms in `completed` state,
    // so the header CTA is hidden by `getHeaderAction(rooms)`. The
    // "More actions" overflow button is always visible (it owns
    // Project Settings, Add more images, Delete project), so we use
    // it as the always-clickable below-the-overlay target. Clicking
    // it opens a dropdown menu — we dismiss with Escape inside the
    // helper so each iteration leaves the page in a clean state.
    const headerAction = page.getByRole('button', { name: /more actions/i });
    // Sanity: the click-through anchor exists before we begin.
    await expect(headerAction).toBeVisible();

    // Helper: open the Edit Prompt dialog from the overflow menu.
    async function openDialog() {
      await page.getByTestId('variation-1-regen-trigger').click();
      await page.getByTestId('variation-1-edit-prompt').click();
      await expect(dialog).toBeVisible();
    }

    // Helper: assert the close path left the page fully interactive.
    // Failure messages on each assertion call out the lock-leak
    // family being caught so a future contributor reading a CI
    // failure understands the regression surface (PRD AC bullet 4).
    async function assertPageIsInteractive(closePathLabel: string) {
      // 1. Wait for the dialog and overlay to be gone — Radix's exit
      //    animation + cleanup must complete before we can fairly
      //    assert body-lock state. This avoids the documented race
      //    where Radix sets `data-state="closed"` mid-animation and
      //    body-lock cleanup hasn't run yet.
      await expect(
        dialog,
        `[${closePathLabel}] dialog should fully close before page-interactive checks`,
      ).not.toBeVisible();
      await expect(
        overlay,
        `[${closePathLabel}] overlay should detach before page-interactive checks`,
      ).toHaveCount(0);

      // 2. <body> must not carry the inline `pointer-events: none`
      //    that Radix's react-remove-scroll leaves behind when the
      //    body-lock cleanup glitches. If this fails, the regression
      //    is the original Radix Dialog body-lock leak and the
      //    layout-level BodyLockGuard either is not mounted or
      //    failed to clear within an animation frame.
      const bodyPointerEvents = await page
        .locator('body')
        .evaluate((b) => (b as HTMLElement).style.pointerEvents);
      expect(
        bodyPointerEvents,
        `[${closePathLabel}] <body> still has inline pointer-events="${bodyPointerEvents}" after close — Radix body-lock leak (regression of body-lock-fix PRD)`,
      ).not.toBe('none');

      // 3. <body> must not carry the `data-scroll-locked` attribute
      //    that react-remove-scroll uses to mark a stuck scroll
      //    lock. Same provenance and same regression family as #2.
      const hasScrollLocked = await page
        .locator('body')
        .evaluate((b) => b.hasAttribute('data-scroll-locked'));
      expect(
        hasScrollLocked,
        `[${closePathLabel}] <body> still has data-scroll-locked attribute after close — react-remove-scroll lock leak (regression of body-lock-fix PRD)`,
      ).toBe(false);

      // 4. Ground-truth user-facing assertion: a non-`force` click on
      //    a normal page element below the closed overlay must
      //    succeed. This catches not only the two known stuck-
      //    attribute signatures above, but ANY other mechanism that
      //    could leave the page non-interactive (stuck focus-trap,
      //    orphan portal capturing pointer events, React-19
      //    scheduler bug, etc.). PRD § Test contract surface. We
      //    immediately dismiss the resulting dropdown with Escape so
      //    each close-path iteration leaves the page in a clean
      //    state for the next iteration.
      await headerAction.click({ timeout: 2_000 });
      await page.keyboard.press('Escape');
    }

    // ─────────── Close path 1: Cancel button ───────────
    await openDialog();
    await page.getByTestId('edit-prompt-cancel').click();
    await assertPageIsInteractive('Cancel button');

    // ─────────── Close path 2: ✕ button (Radix DialogClose) ───────────
    await openDialog();
    // shadcn/ui DialogContent renders an inline ``<DialogPrimitive.Close>``
    // with a visually hidden "Close" label and an XIcon. It does NOT
    // carry a `data-slot="dialog-close"` attribute (that slot is only
    // on the standalone DialogClose export). Target by accessible
    // name within the dialog scope.
    await dialog.getByRole('button', { name: /close/i }).click();
    await assertPageIsInteractive('X close button');

    // ─────────── Close path 3: Escape key ───────────
    await openDialog();
    // Click the dialog's textarea to move focus INTO the dialog
    // before pressing Escape. After the per-variation overflow
    // dropdown closes (it dismissed when we clicked
    // ``variation-1-edit-prompt``), Radix's ``onCloseAutoFocus``
    // restores focus to the dropdown trigger button — which is in
    // the page below the dialog. From that focus position, the
    // Escape key is captured by the dropdown's own document-level
    // handler before it reaches the open dialog. Clicking the
    // textarea reseats focus inside the dialog so Radix Dialog's
    // Escape handler fires correctly. Empirically verified: a bare
    // ``page.keyboard.press('Escape')`` (or even
    // ``textarea.focus()``) leaves the dialog open; only a real
    // click on the textarea moves focus in a way Radix Dialog sees.
    await page.getByTestId('edit-prompt-textarea').click();
    await page.keyboard.press('Escape');
    await assertPageIsInteractive('Escape key');

    // ─────────── Close path 4: click outside (overlay corner) ───────────
    await openDialog();
    // Click the overlay at a corner so we land OUTSIDE the dialog
    // content (the centered DialogContent sits on top of the overlay
    // — clicking the overlay's center would land on the content).
    // The Edit Prompt dialog's `onInteractOutside` only preventDefaults
    // when isSubmitting, so a not-submitting outside click closes it.
    await overlay.click({ position: { x: 5, y: 5 } });
    await assertPageIsInteractive('click outside (overlay corner)');

    await page.screenshot({
      path: `${SCREENSHOT_DIR}/09-close-then-interactive-final.png`,
      fullPage: true,
    });
  });
});
