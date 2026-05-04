import { describe, it, expect, vi, afterEach } from "vitest";
import { render, cleanup, screen, fireEvent } from "@testing-library/react";
import { ProjectGenerationBanner } from "../ProjectGenerationBanner";

afterEach(() => {
  cleanup();
});

// ---------------------------------------------------------------------------
// Issue 010 of project-generation-async-queue-cutover PRD.
//
// Pure presentational component bound to the inFlightProjectGeneration slice.
// The slice owns when the banner is mounted; this component owns:
//   - phase-string → friendly label derivation
//   - progress display
//   - belt-and-suspenders null return on terminal status
//   - a11y wiring (role=status, accessible cancel button)
// ---------------------------------------------------------------------------

describe("ProjectGenerationBanner — rendering", () => {
  it("renders phase label and progress percentage", () => {
    render(
      <ProjectGenerationBanner
        progress={47}
        phase="room_started"
        status="running"
        onCancel={() => {}}
      />,
    );
    // Banner is in the DOM.
    expect(screen.getByTestId("project-generation-banner")).toBeTruthy();
    // Friendly label rendered (NOT the raw phase string).
    expect(screen.getByText(/Generating/i)).toBeTruthy();
    // Progress percentage rendered (the AC requires "numeric percentage
    // and / or progress bar"; we ship both — test the numeric part).
    expect(screen.getByText(/47\s*%/)).toBeTruthy();
  });

  it("data-testid='project-generation-banner' is present (stable selector for issue 011)", () => {
    render(
      <ProjectGenerationBanner
        progress={0}
        phase="queued"
        status="pending"
        onCancel={() => {}}
      />,
    );
    const banner = screen.getByTestId("project-generation-banner");
    expect(banner).toBeTruthy();
    // data-status mirrors the prop so e2e specs can wait on it.
    expect(banner.getAttribute("data-status")).toBe("pending");
  });

  it("renders without crashing at 0% progress (boundary)", () => {
    render(
      <ProjectGenerationBanner
        progress={0}
        phase="queued"
        status="pending"
        onCancel={() => {}}
      />,
    );
    expect(screen.getByText(/0\s*%/)).toBeTruthy();
  });

  it("renders without crashing at 100% progress (boundary)", () => {
    // Caller would normally pair 100% with a terminal status, but the
    // banner's null-return path triggers on STATUS, not progress, so
    // 100% + status='running' must still render (the worker writes
    // progress=100 + phase=finalizing BEFORE flipping status).
    render(
      <ProjectGenerationBanner
        progress={100}
        phase="finalizing"
        status="running"
        onCancel={() => {}}
      />,
    );
    expect(screen.getByText(/100\s*%/)).toBeTruthy();
    expect(screen.getByText(/Finalizing/i)).toBeTruthy();
  });
});

describe("ProjectGenerationBanner — phase label derivation", () => {
  // The label-derivation rules belong to this component (per issue spec).
  // Backend pipeline emits these phase strings (see staging_pipeline.py +
  // job_worker.py):
  //   - "queued"           → job created, not yet picked up by worker
  //   - "room_started"     → a room began processing
  //   - "room_completed"   → a room finished (worker continues to next)
  //   - "room_failed"      → a room failed (pipeline continues)
  //   - "variation_failed" → an individual variation failed (continues)
  //   - "finalizing"       → worker post-pipeline cleanup
  //   - unknown / future   → forward-compat: title-case the raw value
  it.each([
    ["queued", /Queued/i],
    ["room_started", /Generating/i],
    ["room_completed", /Generating/i],
    ["room_failed", /Generating/i],
    ["variation_failed", /Generating/i],
    ["finalizing", /Finalizing/i],
  ])("maps phase %s to friendly label", (phase, expectedLabel) => {
    render(
      <ProjectGenerationBanner
        progress={50}
        phase={phase}
        status="running"
        onCancel={() => {}}
      />,
    );
    expect(screen.getByTestId("project-generation-banner").textContent).toMatch(
      expectedLabel,
    );
  });

  it("forward-compat: unknown phase strings are rendered title-cased (snake_case → spaces)", () => {
    render(
      <ProjectGenerationBanner
        progress={10}
        phase="composing_brief"
        status="running"
        onCancel={() => {}}
      />,
    );
    // Future-proof: a backend-side new phase string ("composing_brief",
    // "uploading", etc.) appears in the banner with light formatting
    // rather than producing a confusing literal "composing_brief".
    expect(screen.getByTestId("project-generation-banner").textContent).toMatch(
      /Composing Brief/,
    );
  });

  it("falls back to a generic 'Generating' label when phase is empty string", () => {
    render(
      <ProjectGenerationBanner
        progress={20}
        phase=""
        status="running"
        onCancel={() => {}}
      />,
    );
    expect(screen.getByTestId("project-generation-banner").textContent).toMatch(
      /Generating/i,
    );
  });
});

describe("ProjectGenerationBanner — terminal status null-return", () => {
  // Belt-and-suspenders: the slice already nulls on terminal status, but
  // a stale render between change-feed events could leave the banner
  // briefly mounted with a terminal status. Returning null defensively
  // prevents the "Cancel" button being clickable on a finished job.
  it.each(["succeeded", "failed", "cancelled"])(
    "returns null when status is %s",
    (status) => {
      const { container } = render(
        <ProjectGenerationBanner
          progress={100}
          phase="finalizing"
          status={status}
          onCancel={() => {}}
        />,
      );
      expect(container.firstChild).toBeNull();
      expect(screen.queryByTestId("project-generation-banner")).toBeNull();
    },
  );
});

describe("ProjectGenerationBanner — cancel control", () => {
  it("clicking the Cancel button invokes onCancel", () => {
    const onCancel = vi.fn();
    render(
      <ProjectGenerationBanner
        progress={30}
        phase="room_started"
        status="running"
        onCancel={onCancel}
      />,
    );
    const button = screen.getByRole("button", { name: /cancel/i });
    fireEvent.click(button);
    expect(onCancel).toHaveBeenCalledTimes(1);
  });

  it("cancel button is a real <button> element with an accessible name", () => {
    render(
      <ProjectGenerationBanner
        progress={30}
        phase="room_started"
        status="running"
        onCancel={() => {}}
      />,
    );
    const button = screen.getByRole("button", { name: /cancel/i });
    // Real <button>, not a <div role="button"> — keyboard-focusable by
    // default; matches the AC ("Cancel button is a real <button>").
    expect(button.tagName).toBe("BUTTON");
    // Accessible name resolves to "Cancel" (or contains it). The AC
    // requires an accessible name; we don't pin specific copy beyond
    // "must contain Cancel".
    const accessibleName =
      button.getAttribute("aria-label") || button.textContent || "";
    expect(accessibleName).toMatch(/cancel/i);
  });

  it("cancel button is disabled and does NOT invoke onCancel when cancelling=true", () => {
    const onCancel = vi.fn();
    render(
      <ProjectGenerationBanner
        progress={60}
        phase="room_started"
        status="running"
        onCancel={onCancel}
        cancelling={true}
      />,
    );
    const button = screen.getByRole("button", { name: /cancel/i });
    expect(button.hasAttribute("disabled")).toBe(true);
    fireEvent.click(button);
    // disabled <button> doesn't fire click handlers natively, but pin
    // the contract anyway so a future "fix" using a div+onClick can't
    // silently break it.
    expect(onCancel).not.toHaveBeenCalled();
  });

  it("cancel button is enabled when cancelling is undefined (default)", () => {
    render(
      <ProjectGenerationBanner
        progress={60}
        phase="room_started"
        status="running"
        onCancel={() => {}}
      />,
    );
    const button = screen.getByRole("button", { name: /cancel/i });
    expect(button.hasAttribute("disabled")).toBe(false);
  });

  it("cancel button is enabled when cancelling=false (explicit)", () => {
    render(
      <ProjectGenerationBanner
        progress={60}
        phase="room_started"
        status="running"
        onCancel={() => {}}
        cancelling={false}
      />,
    );
    const button = screen.getByRole("button", { name: /cancel/i });
    expect(button.hasAttribute("disabled")).toBe(false);
  });
});

describe("ProjectGenerationBanner — accessibility", () => {
  it("banner is a live region (role='status')", () => {
    render(
      <ProjectGenerationBanner
        progress={50}
        phase="room_started"
        status="running"
        onCancel={() => {}}
      />,
    );
    // Screen readers announce updates without users having to inspect
    // the DOM. role="status" is implicitly aria-live="polite".
    const status = screen.getByRole("status");
    expect(status).toBeTruthy();
    // The banner IS the status region (or contains it).
    const banner = screen.getByTestId("project-generation-banner");
    expect(
      banner === status || banner.contains(status) || status.contains(banner),
    ).toBe(true);
  });

  it("progress is exposed via aria-valuenow on the progress region", () => {
    render(
      <ProjectGenerationBanner
        progress={73}
        phase="room_started"
        status="running"
        onCancel={() => {}}
      />,
    );
    // Radix Progress primitive exposes aria-valuenow; the test confirms
    // we're routing the prop through it correctly (NOT just rendering
    // the percentage as plain text and forgetting the bar).
    const progress = screen.getByRole("progressbar");
    expect(progress.getAttribute("aria-valuenow")).toBe("73");
  });
});
