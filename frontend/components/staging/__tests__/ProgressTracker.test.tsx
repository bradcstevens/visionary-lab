import { describe, it, expect, afterEach } from "vitest";
import { render, cleanup, screen } from "@testing-library/react";
import { ProgressTracker } from "../ProgressTracker";
import type { ProjectJob } from "@/context/jobs-context";

afterEach(() => cleanup());

function makeJob(overrides: Partial<ProjectJob> = {}): ProjectJob {
  return {
    id: "j1",
    project_id: "p1",
    room_id: "r1",
    variation_id: "v1",
    revision: 0,
    kind: "regenerate_variation",
    status: "running",
    progress: 0.5,
    phase: "generating",
    updated_at: "2026-05-02T00:00:00.000Z",
    ...overrides,
  };
}

describe("ProgressTracker — per-image", () => {
  it("renders nothing when job is null/undefined", () => {
    const { container } = render(<ProgressTracker kind="per-image" job={null} />);
    expect(container.firstChild).toBeNull();
  });

  it("renders nothing when job is in a terminal state", () => {
    const { container: c1 } = render(
      <ProgressTracker kind="per-image" job={makeJob({ status: "succeeded" })} />,
    );
    expect(c1.firstChild).toBeNull();
    cleanup();
    const { container: c2 } = render(
      <ProgressTracker kind="per-image" job={makeJob({ status: "failed" })} />,
    );
    expect(c2.firstChild).toBeNull();
    cleanup();
    const { container: c3 } = render(
      <ProgressTracker kind="per-image" job={makeJob({ status: "cancelled" })} />,
    );
    expect(c3.firstChild).toBeNull();
  });

  it("renders queued state with indeterminate bar (status=pending, phase=queued)", () => {
    render(
      <ProgressTracker
        kind="per-image"
        job={makeJob({ status: "pending", phase: "queued", progress: 0 })}
      />,
    );
    const bar = screen.getByTestId("per-image-progress");
    expect(bar.getAttribute("data-phase")).toBe("queued");
    expect(screen.getByTestId("per-image-progress-indeterminate")).toBeTruthy();
    expect(bar.getAttribute("aria-busy")).toBe("true");
    // No determinate value while queued
    expect(bar.getAttribute("aria-valuenow")).toBeNull();
  });

  it("renders running state with determinate bar fed by job.progress", () => {
    render(
      <ProgressTracker
        kind="per-image"
        job={makeJob({ status: "running", phase: "generating", progress: 0.42 })}
      />,
    );
    const bar = screen.getByTestId("per-image-progress");
    expect(bar.getAttribute("data-phase")).toBe("running");
    const fill = screen.getByTestId("per-image-progress-determinate") as HTMLElement;
    expect(fill.style.width).toBe("42%");
    expect(bar.getAttribute("aria-valuenow")).toBe("42");
  });

  it("clamps progress >1 to 100 and <0 to 0", () => {
    render(
      <ProgressTracker
        kind="per-image"
        job={makeJob({ status: "running", phase: "generating", progress: 1.5 })}
      />,
    );
    const fill = screen.getByTestId("per-image-progress-determinate") as HTMLElement;
    expect(fill.style.width).toBe("100%");
    cleanup();
    render(
      <ProgressTracker
        kind="per-image"
        job={makeJob({ status: "running", phase: "generating", progress: -0.2 })}
      />,
    );
    const fill2 = screen.getByTestId("per-image-progress-determinate") as HTMLElement;
    expect(fill2.style.width).toBe("0%");
  });

  it("treats running status as non-queued even if phase is missing", () => {
    render(
      <ProgressTracker
        kind="per-image"
        job={makeJob({ status: "running", phase: null, progress: 0.1 })}
      />,
    );
    const bar = screen.getByTestId("per-image-progress");
    expect(bar.getAttribute("data-phase")).toBe("running");
  });

  it("treats pending+phase=generating as running (worker advanced phase before status)", () => {
    render(
      <ProgressTracker
        kind="per-image"
        job={makeJob({ status: "pending", phase: "generating", progress: 0.2 })}
      />,
    );
    expect(screen.getByTestId("per-image-progress").getAttribute("data-phase")).toBe(
      "running",
    );
  });
});

describe("ProgressTracker — per-project", () => {
  it("renders nothing when there are no jobs", () => {
    const { container } = render(
      <ProgressTracker kind="per-project" jobs={[]} />,
    );
    expect(container.firstChild).toBeNull();
  });

  it("renders nothing when all jobs are terminal", () => {
    const { container } = render(
      <ProgressTracker
        kind="per-project"
        jobs={[
          makeJob({ id: "a", status: "succeeded" }),
          makeJob({ id: "b", status: "failed" }),
          makeJob({ id: "c", status: "cancelled" }),
        ]}
      />,
    );
    expect(container.firstChild).toBeNull();
  });

  it("renders aggregate bar averaging progress of active jobs only", () => {
    render(
      <ProgressTracker
        kind="per-project"
        jobs={[
          makeJob({ id: "a", status: "running", progress: 0.4 }),
          makeJob({ id: "b", status: "running", progress: 0.8 }),
          // terminal — must NOT lift the average
          makeJob({ id: "c", status: "succeeded", progress: 1.0 }),
        ]}
      />,
    );
    expect(screen.getByTestId("per-project-progress")).toBeTruthy();
    const bar = screen.getByTestId("per-project-progress-bar");
    // average of 0.4 and 0.8 = 0.6 → 60%
    expect(bar.getAttribute("aria-label")).toContain("60%");
  });

  it("counts running vs queued separately", () => {
    render(
      <ProgressTracker
        kind="per-project"
        jobs={[
          makeJob({ id: "a", status: "running", phase: "generating", progress: 0.5 }),
          makeJob({ id: "b", status: "pending", phase: "queued", progress: 0 }),
          makeJob({ id: "c", status: "pending", phase: "queued", progress: 0 }),
        ]}
      />,
    );
    const counts = screen.getByTestId("per-project-progress-counts");
    expect(counts.textContent).toContain("1 running");
    expect(counts.textContent).toContain("2 queued");
  });

  it("hides as soon as the last active job reaches a terminal state", () => {
    const { rerender, container } = render(
      <ProgressTracker
        kind="per-project"
        jobs={[makeJob({ id: "a", status: "running", progress: 0.5 })]}
      />,
    );
    expect(screen.getByTestId("per-project-progress")).toBeTruthy();
    rerender(
      <ProgressTracker
        kind="per-project"
        jobs={[makeJob({ id: "a", status: "succeeded", progress: 1.0 })]}
      />,
    );
    expect(container.firstChild).toBeNull();
  });
});

describe("ProgressTracker — legacy summary mode (back-compat)", () => {
  it("returns null when project status is not 'processing'", () => {
    const { container } = render(
      <ProgressTracker
        project={{
          id: "p1",
          name: "n",
          status: "completed",
          rooms: [],
          prompt: "",
        } as never}
        isGenerating={false}
      />,
    );
    expect(container.firstChild).toBeNull();
  });

  it("renders the summary card when project status is 'processing'", () => {
    render(
      <ProgressTracker
        project={{
          id: "p1",
          name: "n",
          status: "processing",
          prompt: "",
          rooms: [
            {
              id: "r1",
              label: "Living Room",
              status: "processing",
              variations: [
                { id: "v1", status: "completed" },
                { id: "v2", status: "processing" },
              ],
            },
          ],
        } as never}
        isGenerating={true}
      />,
    );
    expect(screen.getByText(/Generation Progress/i)).toBeTruthy();
    expect(screen.getByText("1/2 variations")).toBeTruthy();
  });
});
