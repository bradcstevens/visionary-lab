import { describe, it, expect } from 'vitest';
import {
  getRecoveryState,
  type GetRecoveryStateInput,
} from '../recovery-state';
import type { LostOp } from '@/hooks/useGenerationFleet';

/**
 * Issue 002 of the projects-page-stalled-stream-error-cleanup PRD.
 *
 * Truth-table coverage of the `getRecoveryState` classifier. The four
 * arms have a strict precedence order (`error` > `stream-lost` >
 * `interrupted` > `none`); the cases below fire each arm in isolation
 * AND in every two-way / three-way conflict so a future edit that
 * accidentally swaps the precedence cannot slip through.
 */

const projectLostOp = (id: string): LostOp => ({
  id,
  kind: 'project',
  projectId: 'project-1',
  lostAt: new Date('2026-05-02T00:00:00Z'),
});

const baseInput = (
  overrides: Partial<GetRecoveryStateInput> = {},
): GetRecoveryStateInput => ({
  projectStatus: 'pending',
  isAnyInFlight: false,
  projectLostOps: [],
  generationError: null,
  ...overrides,
});

describe('getRecoveryState — single-arm cases', () => {
  it('returns `none` when nothing is wrong', () => {
    expect(getRecoveryState(baseInput())).toEqual({ kind: 'none' });
  });

  it('returns `none` while a live stream is in flight (processing + isAnyInFlight)', () => {
    expect(
      getRecoveryState(
        baseInput({ projectStatus: 'processing', isAnyInFlight: true }),
      ),
    ).toEqual({ kind: 'none' });
  });

  it('returns `interrupted` when project is processing but no stream is live', () => {
    expect(
      getRecoveryState(
        baseInput({ projectStatus: 'processing', isAnyInFlight: false }),
      ),
    ).toEqual({ kind: 'interrupted' });
  });

  it('returns `stream-lost` with the first lost op id when projectLostOps is non-empty', () => {
    expect(
      getRecoveryState(
        baseInput({ projectLostOps: [projectLostOp('op-A')] }),
      ),
    ).toEqual({ kind: 'stream-lost', lostOpId: 'op-A' });
  });

  it('returns `error` with the parsed statusCode/detail forwarded', () => {
    expect(
      getRecoveryState(
        baseInput({
          generationError: {
            statusCode: 500,
            detail: 'Server boom',
            raw: 'Failed to generate: 500 Server boom',
          },
        }),
      ),
    ).toEqual({ kind: 'error', statusCode: 500, detail: 'Server boom' });
  });
});

describe('getRecoveryState — precedence', () => {
  it('error beats stream-lost (lost-op present alongside a real error)', () => {
    const result = getRecoveryState(
      baseInput({
        generationError: {
          statusCode: 502,
          detail: 'Bad gateway',
          raw: 'r',
        },
        projectLostOps: [projectLostOp('op-A')],
      }),
    );
    expect(result).toEqual({
      kind: 'error',
      statusCode: 502,
      detail: 'Bad gateway',
    });
  });

  it('stream-lost beats interrupted (lost-op present on a stalled processing project)', () => {
    const result = getRecoveryState(
      baseInput({
        projectStatus: 'processing',
        isAnyInFlight: false,
        projectLostOps: [projectLostOp('op-A')],
      }),
    );
    expect(result).toEqual({ kind: 'stream-lost', lostOpId: 'op-A' });
  });

  it('error beats every other arm (error + lost-op + interrupted condition all true)', () => {
    const result = getRecoveryState(
      baseInput({
        projectStatus: 'processing',
        isAnyInFlight: false,
        projectLostOps: [projectLostOp('op-A')],
        generationError: {
          statusCode: 500,
          detail: 'boom',
          raw: 'r',
        },
      }),
    );
    expect(result.kind).toBe('error');
  });

  it('stream-lost picks FIFO when multiple lost ops exist (defensive tiebreaker)', () => {
    const result = getRecoveryState(
      baseInput({
        projectLostOps: [projectLostOp('first'), projectLostOp('second')],
      }),
    );
    expect(result).toEqual({ kind: 'stream-lost', lostOpId: 'first' });
  });
});

describe('getRecoveryState — non-recovery project statuses', () => {
  it.each(['uploading', 'pending', 'completed', 'failed'] as const)(
    'returns `none` when projectStatus is %s and no lost ops / error',
    (status) => {
      expect(
        getRecoveryState(baseInput({ projectStatus: status })),
      ).toEqual({ kind: 'none' });
    },
  );

  it('does NOT return `interrupted` for a non-processing stalled project (e.g. failed)', () => {
    expect(
      getRecoveryState(
        baseInput({ projectStatus: 'failed', isAnyInFlight: false }),
      ),
    ).toEqual({ kind: 'none' });
  });
});

describe('getRecoveryState — error arm forwards optional fields verbatim', () => {
  it('omits statusCode and detail when neither was parseable', () => {
    expect(
      getRecoveryState(
        baseInput({
          generationError: { raw: 'opaque error' },
        }),
      ),
    ).toEqual({ kind: 'error', statusCode: undefined, detail: undefined });
  });
});
