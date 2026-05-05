/**
 * Issue 004 of the active-and-queued-jobs-ux-redesign PRD: deep
 * module that maps the backend's ``ErrorKind`` (issue 002) values
 * to user-visible recovery copy.
 *
 * Pure mapping function: no React, no IO, no env reads. Tests pin
 * the table-driven contract for all five canonical enum values plus
 * the unknown-kind fallback.
 *
 * The five canonical values are defined in
 * ``backend/core/job_errors.py``:
 *   - BRIEF_FAILED, STORE_FAILED, QUEUE_PERMISSION, UNAVAILABLE,
 *     UNKNOWN.
 *
 * PRD AC pins:
 *   - QUEUE_PERMISSION user message names the specific Azure role
 *     ("Storage Queue Data Message Sender") so the developer
 *     receiving the message can self-serve a fix.
 *   - The mapping returns a ``retryable`` flag the recovery banner
 *     can use to decide whether to show a Retry button at all
 *     (some failures, like an absent role assignment, are not
 *     retryable from the user's side — only an admin can fix
 *     them).
 *   - The mapping returns a ``showAdminContact`` flag the recovery
 *     banner can use to surface a "contact your administrator"
 *     subline.
 */
import { describe, expect, it } from "vitest";

import { getErrorKindCopy } from "../error-kind-copy";

describe("getErrorKindCopy — five canonical ErrorKind values", () => {
  it("BRIEF_FAILED returns retryable user-visible copy", () => {
    const copy = getErrorKindCopy("BRIEF_FAILED");
    expect(copy.userMessage).toBeTruthy();
    // Brief composition is the LLM call inside the producer. The
    // user can usually retry — there's no admin-side fix.
    expect(copy.retryable).toBe(true);
    expect(copy.showAdminContact).toBe(false);
    // Friendly title is short, banner-suitable, NOT the raw enum
    // string.
    expect(copy.friendlyTitle).toBeTruthy();
    expect(copy.friendlyTitle).not.toMatch(/BRIEF_FAILED/);
  });

  it("STORE_FAILED returns retryable user-visible copy", () => {
    const copy = getErrorKindCopy("STORE_FAILED");
    expect(copy.userMessage).toBeTruthy();
    // Cosmos write transient — retry usually works.
    expect(copy.retryable).toBe(true);
    expect(copy.showAdminContact).toBe(false);
    expect(copy.friendlyTitle).toBeTruthy();
    expect(copy.friendlyTitle).not.toMatch(/STORE_FAILED/);
  });

  it("QUEUE_PERMISSION names the specific Azure role and contacts admin (NOT user-retryable)", () => {
    const copy = getErrorKindCopy("QUEUE_PERMISSION");
    // PRD AC: developer-targeted message MUST name the specific
    // Azure role. This is the most-pinned contract in slice 5.
    expect(copy.userMessage).toMatch(/Storage Queue Data Message Sender/);
    // RBAC is admin-fixable, not user-retryable. Hiding the Retry
    // button avoids the user pointlessly clicking it 10 times.
    expect(copy.retryable).toBe(false);
    expect(copy.showAdminContact).toBe(true);
    expect(copy.friendlyTitle).toBeTruthy();
  });

  it("UNAVAILABLE returns retryable transient-failure copy", () => {
    const copy = getErrorKindCopy("UNAVAILABLE");
    expect(copy.userMessage).toBeTruthy();
    // Generic Azure transport failure — by definition retryable.
    expect(copy.retryable).toBe(true);
    expect(copy.showAdminContact).toBe(false);
    expect(copy.friendlyTitle).toBeTruthy();
  });

  it("UNKNOWN returns the generic 'try again' fallback", () => {
    const copy = getErrorKindCopy("UNKNOWN");
    // PRD AC: the unknown-kind fallback is "Couldn't start
    // generation, try again" — match key tokens (case insensitive)
    // rather than exact copy so the implementation can paraphrase.
    expect(copy.userMessage).toMatch(/(try again|retry)/i);
    // Default to retryable since we don't know what failed.
    expect(copy.retryable).toBe(true);
    expect(copy.showAdminContact).toBe(false);
    expect(copy.friendlyTitle).toBeTruthy();
  });
});

describe("getErrorKindCopy — unknown-kind fallback", () => {
  it("returns the same shape as UNKNOWN for an unrecognized string", () => {
    // A backend that adds a 6th ErrorKind we don't know about must
    // not crash the UI — the fallback delivers safe generic copy.
    const copy = getErrorKindCopy("SOMETHING_NEW_FROM_BACKEND");
    expect(copy.userMessage).toBeTruthy();
    expect(copy.userMessage).toMatch(/(try again|retry)/i);
    expect(copy.retryable).toBe(true);
    expect(copy.showAdminContact).toBe(false);
    expect(copy.friendlyTitle).toBeTruthy();
  });

  it("handles empty string by falling back to UNKNOWN-equivalent copy", () => {
    const copy = getErrorKindCopy("");
    expect(copy.userMessage).toMatch(/(try again|retry)/i);
    expect(copy.retryable).toBe(true);
  });

  it("returns identical structural shape across every input (table-shape invariant)", () => {
    const inputs = [
      "BRIEF_FAILED",
      "STORE_FAILED",
      "QUEUE_PERMISSION",
      "UNAVAILABLE",
      "UNKNOWN",
      "INVENTED_KIND",
    ];
    for (const k of inputs) {
      const copy = getErrorKindCopy(k);
      expect(typeof copy.friendlyTitle).toBe("string");
      expect(typeof copy.userMessage).toBe("string");
      expect(typeof copy.retryable).toBe("boolean");
      expect(typeof copy.showAdminContact).toBe("boolean");
      // No empty strings on any field — every entry must produce
      // banner-renderable copy.
      expect(copy.friendlyTitle.length).toBeGreaterThan(0);
      expect(copy.userMessage.length).toBeGreaterThan(0);
    }
  });
});
