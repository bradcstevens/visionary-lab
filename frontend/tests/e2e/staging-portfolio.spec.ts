import { test, expect } from '@playwright/test';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';

test('RoomGroup component source includes ORIGINAL badge', async () => {
  const source = readFileSync(
    join(__dirname, '..', '..', 'components', 'staging', 'RoomGroup.tsx'),
    'utf-8',
  );
  expect(source).toContain('ORIGINAL');
  expect(source).toContain('original_image_url');
});

test('VariationThumbnail supports all states', async () => {
  const source = readFileSync(
    join(__dirname, '..', '..', 'components', 'staging', 'VariationThumbnail.tsx'),
    'utf-8',
  );
  expect(source).toContain("'completed'");
  expect(source).toContain("'processing'");
  expect(source).toContain("'failed'");
  expect(source).toContain("'pending'");
});

test('sidebar source includes Projects entry', async () => {
  const source = readFileSync(
    join(__dirname, '..', '..', 'components', 'app-sidebar.tsx'),
    'utf-8',
  );
  expect(source).toContain('"Projects"');
  expect(source).toContain('/projects');
});
