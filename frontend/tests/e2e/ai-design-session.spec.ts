import { test, expect } from '@playwright/test';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';

const STAGING_DIR = join(__dirname, '..', '..', 'components', 'staging');

test('NewProjectWizard has 5 steps', async () => {
  const source = readFileSync(join(STAGING_DIR, 'NewProjectWizard.tsx'), 'utf-8');
  expect(source).toContain('AI Design Session');
  expect(source).toContain('Design Brief');
  expect(source).toContain('Generate');
  const stepMatches = source.match(/number:\s*\d/g);
  expect(stepMatches?.length).toBeGreaterThanOrEqual(5);
});

test('ImageGalleryPanel groups images by feature', async () => {
  const source = readFileSync(join(STAGING_DIR, 'ImageGalleryPanel.tsx'), 'utf-8');
  expect(source).toContain('groupImages');
  expect(source).toContain('focusedImageId');
  expect(source).toContain('onFocusImage');
});

test('DesignChat supports focused image context', async () => {
  const source = readFileSync(join(STAGING_DIR, 'DesignChat.tsx'), 'utf-8');
  expect(source).toContain('focusedImageId');
  expect(source).toContain('chatWithProject');
  expect(source).toContain('onReadyForBrief');
});

test('DesignBriefEditor wraps BriefEditorTabs and renders palette/placement controls', async () => {
  const source = readFileSync(join(STAGING_DIR, 'DesignBriefEditor.tsx'), 'utf-8');
  expect(source).toContain('BriefEditorTabs');
  expect(source).toContain('ObjectPaletteTable');
  expect(source).toContain('PerImageObjectTable');
  expect(source).toContain('placement_guide');
  expect(source).toContain('preserve_elements');
  expect(source).toContain('global_instructions');
});

test('QuickReplyChips maps action keys to labels', async () => {
  const source = readFileSync(join(STAGING_DIR, 'QuickReplyChips.tsx'), 'utf-8');
  expect(source).toContain('specify_species');
  expect(source).toContain('generate_brief');
  expect(source).toContain('ACTION_LABELS');
});

test('stagingApi includes new Design Session endpoints', async () => {
  const source = readFileSync(
    join(__dirname, '..', '..', 'services', 'stagingApi.ts'),
    'utf-8',
  );
  expect(source).toContain('analyzeImages');
  expect(source).toContain('chatWithProject');
  expect(source).toContain('generateBrief');
  expect(source).toContain('updateBrief');
  // Bug fixes should be applied
  expect(source).not.toContain("'room_files'");
  expect(source).not.toContain('/generate/stream');
});
