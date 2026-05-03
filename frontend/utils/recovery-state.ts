/**
 * Issue 002 of the projects-page-stalled-stream-error-cleanup PRD.
 *
 * Pure classifier that maps the project detail page's recovery-relevant
 * inputs to exactly one of four discriminated states. The banner block,
 * the header status badge, the header CTA visibility, and the
 * `RoomStatusPill` stalled treatment all derive from this single
 * function so they cannot drift out of sync.
 *
 * Pure: no React, no IO, no `useMemo`. Co-located with the existing
 * `staging-header.ts` helper and exhaustively tested as a truth table.
 *
 * Precedence rules (top wins; exactly one arm fires per call):
 *
 *   1. `generationError != null`
 *        → `error` (carries `statusCode` / `detail` for the destructive
 *          banner; raw error string is preserved by the caller for the
 *          collapsible "Full error" pre-formatted block).
 *   2. `projectLostOps.length > 0`
 *        → `stream-lost` using the FIRST lost op (FIFO). The watchdog
 *          dedupes to ≤1 project-scope lost op at a time; FIFO is the
 *          defensive tiebreaker.
 *   3. `projectStatus === 'processing' && !isAnyInFlight && projectLostOps.length === 0`
 *        → `interrupted` (a previous run didn't finish and there is no
 *          live stream to recover via).
 *   4. Otherwise → `none`.
 *
 * Note: rule (1) wins over (2) on purpose. With the synthetic-watchdog
 * guard in place (issue 001), `generationError` is set ONLY by real
 * server-sent errors, so when both arms could fire the `error` arm
 * surfaces the more specific signal.
 */

import type { LostOp } from '@/hooks/useGenerationFleet';
import type { StagingProject } from '@/services/stagingApi';

export type ProjectStatus = StagingProject['status'];

export type RecoveryState =
  | { kind: 'none' }
  | { kind: 'stream-lost'; lostOpId: string }
  | { kind: 'interrupted' }
  | { kind: 'error'; statusCode?: number; detail?: string };

export interface GetRecoveryStateInput {
  projectStatus: ProjectStatus;
  isAnyInFlight: boolean;
  projectLostOps: ReadonlyArray<LostOp>;
  generationError:
    | { statusCode?: number; detail?: string; raw: string }
    | null;
}

export function getRecoveryState(
  input: GetRecoveryStateInput,
): RecoveryState {
  const { projectStatus, isAnyInFlight, projectLostOps, generationError } =
    input;

  if (generationError != null) {
    return {
      kind: 'error',
      statusCode: generationError.statusCode,
      detail: generationError.detail,
    };
  }

  if (projectLostOps.length > 0) {
    return { kind: 'stream-lost', lostOpId: projectLostOps[0].id };
  }

  if (projectStatus === 'processing' && !isAnyInFlight) {
    return { kind: 'interrupted' };
  }

  return { kind: 'none' };
}
