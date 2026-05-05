/**
 * Issue 004 of the active-and-queued-jobs-ux-redesign PRD.
 *
 * Pure mapping from the backend's ``ErrorKind`` (issue 002) to the
 * recovery banner copy. The mapping is the single source of truth
 * for "what does the user see when an enqueue fails?" — both the
 * preflight error path (``EnqueueGenerationFailedError`` thrown by
 * ``enqueueProjectGeneration``) and the worker terminal-failure
 * path (``ProjectJob.error_kind`` written by the worker) feed the
 * same UI surface and resolve through this module.
 *
 * Design choices:
 *
 * - Stable plain-object return shape (NOT a class instance) so it
 *   serializes cleanly through React props, plays nicely with
 *   ``vi.mock`` factories, and survives a build-time
 *   ``JSON.stringify`` should we ever need to pre-render.
 * - Defensive ``UNKNOWN`` fallback is wired as the default arm of
 *   the lookup. A backend that ships a 6th ErrorKind ahead of a
 *   matching frontend ship will degrade gracefully to the generic
 *   "try again" copy rather than rendering raw enum tokens like
 *   "FOO_FAILED" in the banner.
 * - The QUEUE_PERMISSION user message names the missing Azure
 *   role verbatim ("Storage Queue Data Message Sender"). The user
 *   reading the banner is — by deployment topology — the developer
 *   running the dev environment, NOT a customer; the explicit role
 *   name is intentional and lets them self-serve a fix without
 *   pulling up Azure docs.
 *
 * NOTE: the five canonical values are defined in
 * ``backend/core/job_errors.py``. A drift between the two is a
 * test failure on the backend side (``test_job_errors.py`` enum
 * value list) AND on the frontend side (the table in this
 * module's ``__tests__/error-kind-copy.test.ts``).
 */

export interface ErrorKindCopy {
  /** Banner header text. Short, banner-suitable, never the raw enum. */
  friendlyTitle: string;
  /** Body paragraph in the banner. May be 1–2 sentences. */
  userMessage: string;
  /**
   * Whether to render the banner's Retry button. False for failures
   * that only an admin can fix (RBAC) so the user doesn't burn time
   * clicking Retry against an upstream problem.
   */
  retryable: boolean;
  /**
   * Whether to surface a "contact your administrator" subline.
   * Currently only true for ``QUEUE_PERMISSION`` (the most
   * obviously admin-fixable failure).
   */
  showAdminContact: boolean;
}

const FALLBACK_COPY: ErrorKindCopy = {
  friendlyTitle: "Generation didn't start",
  userMessage:
    "Something went wrong while starting generation. Try again, and contact support if the problem persists.",
  retryable: true,
  showAdminContact: false,
};

const COPY_TABLE: Record<string, ErrorKindCopy> = {
  BRIEF_FAILED: {
    friendlyTitle: "Couldn't draft the design brief",
    userMessage:
      "We couldn't compose the design brief for this project. Try again — this is usually transient.",
    retryable: true,
    showAdminContact: false,
  },
  STORE_FAILED: {
    friendlyTitle: "Couldn't save the generation request",
    userMessage:
      "Saving the generation request to the database failed. Try again — this is usually transient.",
    retryable: true,
    showAdminContact: false,
  },
  QUEUE_PERMISSION: {
    friendlyTitle: "Worker can't reach the queue",
    userMessage:
      "The worker doesn't have permission to enqueue messages. Grant the deployment identity the 'Storage Queue Data Message Sender' role on the Storage account, then try again.",
    retryable: false,
    showAdminContact: true,
  },
  UNAVAILABLE: {
    friendlyTitle: "Generation service is temporarily unavailable",
    userMessage:
      "We couldn't reach the generation queue. This is usually transient — try again in a moment.",
    retryable: true,
    showAdminContact: false,
  },
  UNKNOWN: {
    friendlyTitle: "Generation didn't start",
    userMessage:
      "Couldn't start generation. Try again, and contact support if the problem persists.",
    retryable: true,
    showAdminContact: false,
  },
};

/**
 * Resolve recovery-banner copy for an ``ErrorKind`` value.
 *
 * Unknown / empty / future-backend kinds fall back to ``UNKNOWN``
 * copy — the recovery banner stays renderable and the user gets
 * actionable guidance even if the backend ships a new kind ahead
 * of a frontend update.
 */
export function getErrorKindCopy(kind: string): ErrorKindCopy {
  if (kind && Object.prototype.hasOwnProperty.call(COPY_TABLE, kind)) {
    return COPY_TABLE[kind];
  }
  return FALLBACK_COPY;
}
