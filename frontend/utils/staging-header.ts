/**
 * Pure helper that decides which CTA the project detail page header
 * should render, given the project's room statuses.
 *
 * Issue 002 of the `per-room-generation-control` PRD replaces the old
 * `allPending ? Generate : Regenerate All` ternary in `ProjectDetailPage`
 * with a 3-arm switch on this helper's tagged-union return value:
 *
 *   - `{ kind: 'generate' }`              every room is `pending`
 *                                         (the all-or-nothing first-time
 *                                         generation case)
 *
 *   - `{ kind: 'generate-remaining', count }`
 *                                         at least one room is in a
 *                                         non-completed state — the
 *                                         label tells the user exactly
 *                                         how many `pending + failed`
 *                                         rooms will be processed
 *
 *   - `{ kind: 'hidden' }`                every room is `completed`
 *                                         (or rooms is empty), so the
 *                                         header CTA is suppressed —
 *                                         per-row Regenerate handles
 *                                         single-room redo
 *
 * Pure: no React, no IO, no `useMemo`. Exhaustively reasoned about via
 * the rendered header button label (`header-generate-remaining-label`
 * Playwright spec).
 *
 * Edge cases & rationale:
 *
 *  - Empty `rooms` array. `Array#every` is vacuously true on an empty
 *    array, which would otherwise classify "no rooms" as "all
 *    completed" → hidden. That's the same answer this helper returns
 *    explicitly, so no special-case is required, but the explicit
 *    early-return below makes the intent obvious to future readers.
 *
 *  - Rooms in `processing`. The PRD's state table only enumerates
 *    pending / failed / completed. A `processing` room appears
 *    transiently during an active stream, when the header button is
 *    already disabled-and-spinning via `isGenerating`. Such rooms are
 *    treated as "remaining" (not counted toward `count`, but they
 *    prevent the `all-completed → hidden` arm from firing), which
 *    keeps the spinner / disabled CTA visible until the stream
 *    actually finishes and rooms flip to `completed` or `failed`.
 *
 *  - The `count` field is `pending + failed` ONLY — `processing` rooms
 *    are excluded from the count because the user already knows
 *    they're being processed (the spinner says so). The number is what
 *    will be processed by the bulk POST `/projects/{id}/generate`,
 *    which the backend filters to `pending` and `failed` rooms.
 */

import type { Room } from '@/services/stagingApi';

export type HeaderAction =
  | { kind: 'generate' }
  | { kind: 'generate-remaining'; count: number }
  | { kind: 'hidden' };

export function getHeaderAction(rooms: ReadonlyArray<Pick<Room, 'status'>>): HeaderAction {
  if (rooms.length === 0) {
    return { kind: 'hidden' };
  }
  if (rooms.every((r) => r.status === 'completed')) {
    return { kind: 'hidden' };
  }
  if (rooms.every((r) => r.status === 'pending')) {
    return { kind: 'generate' };
  }
  const count = rooms.reduce(
    (n, r) => (r.status === 'pending' || r.status === 'failed' ? n + 1 : n),
    0,
  );
  return { kind: 'generate-remaining', count };
}
