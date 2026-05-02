import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent, waitFor } from "@testing-library/react";
import { ProjectRoomsManager } from "../ProjectRoomsManager";
import type { StagingProject } from "@/services/stagingApi";
import * as stagingApi from "@/services/stagingApi";
import * as sonner from "sonner";

/**
 * Vitest unit coverage for `ProjectRoomsManager` — issue 004 of the
 * project-settings-completeness PRD.
 *
 * The component owns the inline rename UX on the Project Settings sheet.
 * The Playwright integration spec at
 * `frontend/tests/e2e/project-settings-sheet.spec.ts` covers the wiring
 * across the sheet + page + backend; these tests exercise the
 * component's pure observable behavior at the React tree level:
 *
 *   1. Rendering behavior (rooms list, edit affordance, fallback when
 *      empty).
 *   2. Edit-mode lifecycle (pencil reveals input pre-filled with
 *      current label; Cancel discards no API call; trimmed/no-op /
 *      empty / unchanged labels disable Save).
 *   3. Save lifecycle (server-confirmed: `updateRoom` then
 *      `onProjectUpdate`; rapid double-click does not double-fire).
 *   4. Error path (toast surfaces; row reverts to view mode).
 *   5. `disabled` prop forwarding.
 *
 * The component is intentionally a deep module with the narrow
 * `{ project, onProjectUpdate, disabled }` interface — the tests
 * assert that interface and never reach into private state, so future
 * implementation refactors that preserve the contract don't need to
 * update these tests.
 */

const STORAGE_HOST = "https://acct.blob.core.windows.net/images";

function makeProject(overrides: Partial<StagingProject> = {}): StagingProject {
  return {
    id: "proj-rooms",
    name: "Rooms Project",
    prompt: "modern minimalist",
    status: "completed",
    settings: {
      variations_per_room: 5,
      model: "gpt-image-2",
      quality: "high",
      size: "auto",
    },
    rooms: [
      {
        id: "room-A",
        label: "Living Room",
        original_image_url: `${STORAGE_HOST}/staging/proj/originals/a.png`,
        original_thumbnail_url: `${STORAGE_HOST}/staging/proj/originals/a-thumb.png`,
        status: "completed",
        prompt_addendum: null,
        variations: [],
      },
      {
        id: "room-B",
        label: "Kitchen",
        original_image_url: `${STORAGE_HOST}/staging/proj/originals/b.png`,
        original_thumbnail_url: `${STORAGE_HOST}/staging/proj/originals/b-thumb.png`,
        status: "completed",
        prompt_addendum: "B's existing addendum",
        variations: [],
      },
    ],
    created_at: "2026-05-01T00:00:00Z",
    updated_at: "2026-05-01T00:00:00Z",
    ...overrides,
  } as StagingProject;
}

let updateRoomSpy: ReturnType<typeof vi.spyOn>;
let toastErrorSpy: ReturnType<typeof vi.spyOn>;

beforeEach(() => {
  updateRoomSpy = vi.spyOn(stagingApi, "updateRoom");
  toastErrorSpy = vi.spyOn(sonner.toast, "error").mockImplementation(() => "");
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("ProjectRoomsManager — rendering", () => {
  it("renders a row per room with the current label", () => {
    const project = makeProject();
    render(
      <ProjectRoomsManager project={project} onProjectUpdate={() => {}} disabled={false} />,
    );
    expect(
      screen.getByTestId("project-rooms-manager-label-room-A").textContent,
    ).toBe("Living Room");
    expect(
      screen.getByTestId("project-rooms-manager-label-room-B").textContent,
    ).toBe("Kitchen");
    // Edit affordance per row.
    expect(screen.getByTestId("project-rooms-manager-edit-room-A")).toBeTruthy();
    expect(screen.getByTestId("project-rooms-manager-edit-room-B")).toBeTruthy();
  });

  it("renders an empty-state hint when project has no rooms", () => {
    const project = makeProject({ rooms: [] });
    render(
      <ProjectRoomsManager project={project} onProjectUpdate={() => {}} disabled={false} />,
    );
    expect(screen.getByTestId("project-rooms-manager-empty")).toBeTruthy();
  });
});

describe("ProjectRoomsManager — edit-mode lifecycle", () => {
  it("clicking pencil reveals input prefilled with current label and hides the static label", () => {
    const project = makeProject();
    render(
      <ProjectRoomsManager project={project} onProjectUpdate={() => {}} disabled={false} />,
    );

    fireEvent.click(screen.getByTestId("project-rooms-manager-edit-room-A"));

    const input = screen.getByTestId("project-rooms-manager-input-room-A") as HTMLInputElement;
    expect(input.value).toBe("Living Room");
    // Static label gone for the editing row.
    expect(screen.queryByTestId("project-rooms-manager-label-room-A")).toBeNull();
    // The other row is still in view mode (only one row is editable
    // at a time — keeps the implementation simple and the UX clear).
    expect(screen.getByTestId("project-rooms-manager-label-room-B")).toBeTruthy();
  });

  it("Cancel discards the local edit, returns to view mode, and never calls updateRoom", () => {
    const onProjectUpdate = vi.fn();
    const project = makeProject();
    render(
      <ProjectRoomsManager
        project={project}
        onProjectUpdate={onProjectUpdate}
        disabled={false}
      />,
    );

    fireEvent.click(screen.getByTestId("project-rooms-manager-edit-room-A"));
    const input = screen.getByTestId("project-rooms-manager-input-room-A") as HTMLInputElement;
    fireEvent.change(input, { target: { value: "Den" } });
    fireEvent.click(screen.getByTestId("project-rooms-manager-cancel-room-A"));

    expect(updateRoomSpy).not.toHaveBeenCalled();
    expect(onProjectUpdate).not.toHaveBeenCalled();
    // Back in view mode with the original (un-renamed) label.
    expect(
      screen.getByTestId("project-rooms-manager-label-room-A").textContent,
    ).toBe("Living Room");
  });

  it("Save is disabled when the trimmed draft is empty", () => {
    const project = makeProject();
    render(
      <ProjectRoomsManager project={project} onProjectUpdate={() => {}} disabled={false} />,
    );

    fireEvent.click(screen.getByTestId("project-rooms-manager-edit-room-A"));
    const input = screen.getByTestId("project-rooms-manager-input-room-A") as HTMLInputElement;
    fireEvent.change(input, { target: { value: "   " } });

    const saveBtn = screen.getByTestId("project-rooms-manager-save-room-A") as HTMLButtonElement;
    expect(saveBtn.disabled).toBe(true);
  });

  it("Save is disabled when trimmed draft equals the current label (no-op rename)", () => {
    // Rubber-duck finding: a trimmed `" Living Room "` edit when the
    // persisted label is `"Living Room"` should be a no-op rather
    // than a wasted round-trip.
    const project = makeProject();
    render(
      <ProjectRoomsManager project={project} onProjectUpdate={() => {}} disabled={false} />,
    );

    fireEvent.click(screen.getByTestId("project-rooms-manager-edit-room-A"));
    const input = screen.getByTestId("project-rooms-manager-input-room-A") as HTMLInputElement;
    fireEvent.change(input, { target: { value: "  Living Room  " } });

    const saveBtn = screen.getByTestId("project-rooms-manager-save-room-A") as HTMLButtonElement;
    expect(saveBtn.disabled).toBe(true);
  });
});

describe("ProjectRoomsManager — save lifecycle", () => {
  it("Save calls updateRoom with the trimmed label, then onProjectUpdate with the response", async () => {
    const project = makeProject();
    const updatedFromServer = makeProject({
      rooms: [
        { ...project.rooms[0], label: "Master Bedroom" },
        project.rooms[1],
      ],
    });
    updateRoomSpy.mockResolvedValueOnce(updatedFromServer);
    const onProjectUpdate = vi.fn();

    render(
      <ProjectRoomsManager
        project={project}
        onProjectUpdate={onProjectUpdate}
        disabled={false}
      />,
    );

    fireEvent.click(screen.getByTestId("project-rooms-manager-edit-room-A"));
    fireEvent.change(screen.getByTestId("project-rooms-manager-input-room-A"), {
      target: { value: "  Master Bedroom  " },
    });
    fireEvent.click(screen.getByTestId("project-rooms-manager-save-room-A"));

    await waitFor(() => {
      expect(updateRoomSpy).toHaveBeenCalledWith("proj-rooms", "room-A", {
        label: "Master Bedroom",
      });
    });
    await waitFor(() => {
      expect(onProjectUpdate).toHaveBeenCalledWith(updatedFromServer);
    });
  });

  it("rapid double-click on Save does not fire updateRoom twice (per-row pending guard)", async () => {
    // Rubber-duck blind spot: without per-row pending state, a fast
    // user could trigger two PATCH requests for the same rename.
    const project = makeProject();
    let resolveFirstCall: (value: StagingProject) => void = () => {};
    const firstCallPromise = new Promise<StagingProject>((resolve) => {
      resolveFirstCall = resolve;
    });
    updateRoomSpy.mockReturnValueOnce(firstCallPromise);

    render(
      <ProjectRoomsManager
        project={project}
        onProjectUpdate={() => {}}
        disabled={false}
      />,
    );

    fireEvent.click(screen.getByTestId("project-rooms-manager-edit-room-A"));
    fireEvent.change(screen.getByTestId("project-rooms-manager-input-room-A"), {
      target: { value: "Den" },
    });
    const saveBtn = screen.getByTestId("project-rooms-manager-save-room-A");
    // Fire two clicks back-to-back BEFORE the first PATCH resolves.
    fireEvent.click(saveBtn);
    fireEvent.click(saveBtn);
    fireEvent.click(saveBtn);

    // Only one updateRoom call landed despite three clicks.
    expect(updateRoomSpy).toHaveBeenCalledTimes(1);

    // Drain the pending promise so React doesn't warn about an
    // unresolved act() boundary.
    resolveFirstCall(makeProject({ rooms: [{ ...project.rooms[0], label: "Den" }, project.rooms[1]] }));
    await waitFor(() => {
      // Still only one call after the resolve flushes.
      expect(updateRoomSpy).toHaveBeenCalledTimes(1);
    });
  });
});

describe("ProjectRoomsManager — error path", () => {
  it("on updateRoom failure, toasts the error and reverts to view mode with the original label", async () => {
    const project = makeProject();
    const failure = new Error("Network blew up");
    updateRoomSpy.mockRejectedValueOnce(failure);
    const onProjectUpdate = vi.fn();

    render(
      <ProjectRoomsManager
        project={project}
        onProjectUpdate={onProjectUpdate}
        disabled={false}
      />,
    );

    fireEvent.click(screen.getByTestId("project-rooms-manager-edit-room-A"));
    fireEvent.change(screen.getByTestId("project-rooms-manager-input-room-A"), {
      target: { value: "Den" },
    });
    fireEvent.click(screen.getByTestId("project-rooms-manager-save-room-A"));

    await waitFor(() => {
      expect(toastErrorSpy).toHaveBeenCalledWith("Network blew up");
    });
    // onProjectUpdate is never called because the server didn't ack.
    expect(onProjectUpdate).not.toHaveBeenCalled();
    // Row reverts to view mode with the ORIGINAL label — `project`
    // hasn't been replaced (server didn't update), so the static
    // label re-renders with "Living Room".
    await waitFor(() => {
      expect(
        screen.getByTestId("project-rooms-manager-label-room-A").textContent,
      ).toBe("Living Room");
    });
  });
});

describe("ProjectRoomsManager — disabled prop", () => {
  it("when disabled=true, every per-row pencil button is disabled", () => {
    const project = makeProject();
    render(
      <ProjectRoomsManager project={project} onProjectUpdate={() => {}} disabled={true} />,
    );

    expect(
      (screen.getByTestId("project-rooms-manager-edit-room-A") as HTMLButtonElement).disabled,
    ).toBe(true);
    expect(
      (screen.getByTestId("project-rooms-manager-edit-room-B") as HTMLButtonElement).disabled,
    ).toBe(true);
  });

  it("when disabled=true and the user is somehow in edit mode, Save and the input are disabled", () => {
    // Defense-in-depth: the edit affordance is disabled when
    // `disabled=true`, but if the prop flips mid-edit (e.g., a
    // generation kicks off after the user enters edit mode), the
    // input + save controls must respect the new disabled value.
    const project = makeProject();
    const { rerender } = render(
      <ProjectRoomsManager project={project} onProjectUpdate={() => {}} disabled={false} />,
    );
    fireEvent.click(screen.getByTestId("project-rooms-manager-edit-room-A"));
    fireEvent.change(screen.getByTestId("project-rooms-manager-input-room-A"), {
      target: { value: "Den" },
    });

    // Simulate the parent flipping disabled to true mid-edit.
    rerender(
      <ProjectRoomsManager project={project} onProjectUpdate={() => {}} disabled={true} />,
    );

    expect(
      (screen.getByTestId("project-rooms-manager-input-room-A") as HTMLInputElement).disabled,
    ).toBe(true);
    expect(
      (screen.getByTestId("project-rooms-manager-save-room-A") as HTMLButtonElement).disabled,
    ).toBe(true);
    expect(
      (screen.getByTestId("project-rooms-manager-cancel-room-A") as HTMLButtonElement).disabled,
    ).toBe(true);
  });
});
