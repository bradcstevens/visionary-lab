import { describe, it, expect, afterEach, vi } from "vitest";
import { render, cleanup, screen, fireEvent } from "@testing-library/react";
import { StalenessSubline } from "../StalenessSubline";
import type { StalenessState } from "@/lib/job-staleness";

const make = (
  kind: StalenessState["kind"],
  secondsAgo: number,
  jobId = "j1",
): StalenessState => ({ jobId, kind, secondsAgo });

describe("StalenessSubline", () => {
  afterEach(() => cleanup());

  it("renders fallback when staleness is null (fresh)", () => {
    render(
      <StalenessSubline
        staleness={null}
        cancelling={false}
        onCancelAllClick={() => {}}
        fallback={<p data-testid="counter">3/4 variations complete</p>}
      />,
    );
    expect(screen.getByTestId("counter")).toBeTruthy();
    expect(screen.queryByTestId("staleness-subline")).toBeNull();
  });

  it("renders fallback when staleness.kind === 'fresh'", () => {
    render(
      <StalenessSubline
        staleness={make("fresh", 5)}
        cancelling={false}
        onCancelAllClick={() => {}}
        fallback={<p data-testid="counter">3/4 variations complete</p>}
      />,
    );
    expect(screen.getByTestId("counter")).toBeTruthy();
    expect(screen.queryByTestId("staleness-subline")).toBeNull();
  });

  it("renders soft-pending message without cancel button", () => {
    render(
      <StalenessSubline
        staleness={make("soft-pending", 60)}
        cancelling={false}
        onCancelAllClick={() => {}}
      />,
    );
    expect(screen.getByTestId("staleness-subline").getAttribute("data-state")).toBe(
      "soft-pending",
    );
    expect(screen.getByText(/Waiting for worker to pick up your job/)).toBeTruthy();
    expect(screen.queryByTestId("cancel-queued-jobs-button")).toBeNull();
  });

  it("renders soft-running message with secondsAgo spliced in", () => {
    render(
      <StalenessSubline
        staleness={make("soft-running", 47)}
        cancelling={false}
        onCancelAllClick={() => {}}
      />,
    );
    expect(screen.getByText(/Generation paused/)).toBeTruthy();
    expect(screen.getByText(/47s ago/)).toBeTruthy();
    expect(screen.queryByTestId("cancel-queued-jobs-button")).toBeNull();
  });

  it("renders hard-pending with verbatim PRD copy + cancel button", () => {
    render(
      <StalenessSubline
        staleness={make("hard-pending", 130)}
        cancelling={false}
        onCancelAllClick={() => {}}
      />,
    );
    expect(
      screen.getByText(
        "Worker may be unavailable. Try cancelling and starting again.",
      ),
    ).toBeTruthy();
    expect(screen.getByTestId("cancel-queued-jobs-button")).toBeTruthy();
  });

  it("renders hard-running with verbatim PRD copy + cancel button", () => {
    render(
      <StalenessSubline
        staleness={make("hard-running", 200)}
        cancelling={false}
        onCancelAllClick={() => {}}
      />,
    );
    expect(
      screen.getByText(
        "Worker stopped responding. Cancel to free the queue and retry.",
      ),
    ).toBeTruthy();
    expect(screen.getByTestId("cancel-queued-jobs-button")).toBeTruthy();
  });

  it("invokes onCancelAllClick exactly once when the cancel button is clicked", () => {
    const onClick = vi.fn();
    render(
      <StalenessSubline
        staleness={make("hard-pending", 130)}
        cancelling={false}
        onCancelAllClick={onClick}
      />,
    );
    fireEvent.click(screen.getByTestId("cancel-queued-jobs-button"));
    expect(onClick).toHaveBeenCalledTimes(1);
  });

  it("renders 'Cancelling…' fixed copy when cancelling=true regardless of staleness", () => {
    render(
      <StalenessSubline
        staleness={make("hard-running", 130)}
        cancelling={true}
        onCancelAllClick={() => {}}
      />,
    );
    expect(screen.getByTestId("staleness-subline").getAttribute("data-state")).toBe(
      "cancelling",
    );
    expect(screen.getByText(/Cancelling/)).toBeTruthy();
    // Cancel button is suppressed during cancellation.
    expect(screen.queryByTestId("cancel-queued-jobs-button")).toBeNull();
  });

  it("'Cancelling…' subline renders even when staleness is null (suppression-after-timeout overlap)", () => {
    render(
      <StalenessSubline
        staleness={null}
        cancelling={true}
        onCancelAllClick={() => {}}
      />,
    );
    expect(screen.getByText(/Cancelling/)).toBeTruthy();
  });
});
