import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { render, cleanup, screen, fireEvent, act } from "@testing-library/react";
import { StorageImage } from "../StorageImage";

// sasTokenService is imported by StorageImage; mock it so tests don't
// hit the real /api/sas-token endpoint and we can assert the
// invalidate/refresh flow drives a fresh URL.
vi.mock("@/services/sas-token", () => ({
  sasTokenService: {
    invalidate: vi.fn(),
    getTokens: vi.fn().mockResolvedValue({ imageSasToken: "sv=fresh" }),
  },
}));

import { sasTokenService } from "@/services/sas-token";

describe("StorageImage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    (sasTokenService.getTokens as ReturnType<typeof vi.fn>).mockResolvedValue({
      imageSasToken: "sv=fresh",
    });
  });

  afterEach(() => cleanup());

  describe("variant URL resolution", () => {
    it("variant=thumb selects thumbUrl", () => {
      render(
        <StorageImage
          variant="thumb"
          thumbUrl="https://x/foo.thumb.webp?sas=1"
          mdUrl="https://x/foo.md.webp"
          originalUrl="https://x/foo.png"
          alt="t"
        />
      );
      const img = screen.getByAltText("t") as HTMLImageElement;
      expect(img.src).toContain(".thumb.");
      expect(img.src).not.toContain(".md.");
      expect(img.src).not.toContain("foo.png");
    });

    it("variant=md selects mdUrl", () => {
      render(
        <StorageImage
          variant="md"
          thumbUrl="https://x/foo.thumb.webp"
          mdUrl="https://x/foo.md.webp?sas=1"
          originalUrl="https://x/foo.png"
          alt="t"
        />
      );
      expect((screen.getByAltText("t") as HTMLImageElement).src).toContain(".md.");
    });

    it("variant=original selects originalUrl", () => {
      render(
        <StorageImage
          variant="original"
          thumbUrl="https://x/foo.thumb.webp"
          mdUrl="https://x/foo.md.webp"
          originalUrl="https://x/foo.png?sas=1"
          alt="t"
        />
      );
      expect((screen.getByAltText("t") as HTMLImageElement).src).toContain("foo.png");
    });

    it("legacy src prop still works when variant is omitted", () => {
      render(<StorageImage src="https://x/legacy.png?sas=1" alt="t" />);
      expect((screen.getByAltText("t") as HTMLImageElement).src).toContain("legacy.png");
    });

    it("never renders <img src='undefined'> when variant URL is missing", () => {
      const { container } = render(
        <StorageImage variant="thumb" thumbUrl={undefined} alt="t" />
      );
      expect(container.querySelector("img")).toBeNull();
    });
  });

  describe("skeleton state", () => {
    it("renders skeleton when chosen variant URL is missing", () => {
      render(<StorageImage variant="thumb" thumbUrl={undefined} alt="t" />);
      expect(screen.getByTestId("storage-image-skeleton")).toBeTruthy();
      expect(screen.queryByTestId("storage-image-error")).toBeNull();
    });

    it("renders skeleton overlay while img is loading", () => {
      render(<StorageImage variant="thumb" thumbUrl="https://x/a.thumb.webp" alt="t" />);
      // The loading overlay sits behind the <img>; both render until onLoad fires.
      expect(screen.getByTestId("storage-image-skeleton")).toBeTruthy();
      expect(screen.getByAltText("t")).toBeTruthy();
    });

    it("removes skeleton overlay after onLoad fires", () => {
      render(<StorageImage variant="thumb" thumbUrl="https://x/a.thumb.webp" alt="t" />);
      const img = screen.getByAltText("t") as HTMLImageElement;
      fireEvent.load(img);
      expect(screen.queryByTestId("storage-image-skeleton")).toBeNull();
    });

    it("renders skeleton (not error) when URL missing AND jobStatus pending", () => {
      render(
        <StorageImage variant="thumb" thumbUrl={undefined} jobStatus="pending" alt="t" />
      );
      expect(screen.getByTestId("storage-image-skeleton")).toBeTruthy();
    });

    it("renders error (not skeleton) when URL missing AND jobStatus failed", () => {
      render(
        <StorageImage variant="thumb" thumbUrl={undefined} jobStatus="failed" alt="t" />
      );
      expect(screen.getByTestId("storage-image-error")).toBeTruthy();
      expect(screen.queryByTestId("storage-image-skeleton")).toBeNull();
    });
  });

  describe("error + retry flow", () => {
    it("first onError triggers SAS-token refresh and rewrites src", async () => {
      render(
        <StorageImage variant="thumb" thumbUrl="https://x/a.thumb.webp?stale=1" alt="t" />
      );
      const img = screen.getByAltText("t") as HTMLImageElement;
      await act(async () => {
        fireEvent.error(img);
        await Promise.resolve();
      });
      expect(sasTokenService.invalidate).toHaveBeenCalledTimes(1);
      expect(sasTokenService.getTokens).toHaveBeenCalledTimes(1);
      // Re-rendered <img> has the fresh sas appended; no error UI yet.
      const refreshed = screen.getByAltText("t") as HTMLImageElement;
      expect(refreshed.src).toContain("sv=fresh");
      expect(screen.queryByTestId("storage-image-error")).toBeNull();
    });

    it("second onError surfaces the manual retry button", async () => {
      render(
        <StorageImage variant="thumb" thumbUrl="https://x/a.thumb.webp" alt="t" />
      );
      const img = screen.getByAltText("t") as HTMLImageElement;
      await act(async () => {
        fireEvent.error(img); // triggers auto SAS refresh
        await Promise.resolve();
      });
      const refreshed = screen.getByAltText("t") as HTMLImageElement;
      await act(async () => {
        fireEvent.error(refreshed); // second failure
        await Promise.resolve();
      });
      expect(screen.getByTestId("storage-image-error")).toBeTruthy();
      expect(screen.getByTestId("storage-image-retry")).toBeTruthy();
    });

    it("clicking retry invalidates SAS, remounts <img>, and re-attempts", async () => {
      render(
        <StorageImage variant="thumb" thumbUrl="https://x/a.thumb.webp" alt="t" />
      );
      // Drive into error state
      const img1 = screen.getByAltText("t") as HTMLImageElement;
      await act(async () => { fireEvent.error(img1); await Promise.resolve(); });
      const img2 = screen.getByAltText("t") as HTMLImageElement;
      await act(async () => { fireEvent.error(img2); await Promise.resolve(); });

      vi.clearAllMocks();
      (sasTokenService.getTokens as ReturnType<typeof vi.fn>).mockResolvedValue({
        imageSasToken: "sv=second",
      });

      await act(async () => {
        fireEvent.click(screen.getByTestId("storage-image-retry"));
        await Promise.resolve();
        await Promise.resolve();
      });

      expect(sasTokenService.invalidate).toHaveBeenCalledTimes(1);
      expect(sasTokenService.getTokens).toHaveBeenCalledTimes(1);
      // Error UI cleared; <img> back with the second-generation SAS.
      expect(screen.queryByTestId("storage-image-error")).toBeNull();
      const refreshed = screen.getByAltText("t") as HTMLImageElement;
      expect(refreshed.src).toContain("sv=second");
    });

    it("retry button stops propagation so clicks don't bubble to parent onClick", async () => {
      const parentClick = vi.fn();
      render(
        <div onClick={parentClick}>
          <StorageImage variant="thumb" thumbUrl="https://x/a.thumb.webp" alt="t" />
        </div>
      );
      // Drive to error
      const img1 = screen.getByAltText("t") as HTMLImageElement;
      await act(async () => { fireEvent.error(img1); await Promise.resolve(); });
      const img2 = screen.getByAltText("t") as HTMLImageElement;
      await act(async () => { fireEvent.error(img2); await Promise.resolve(); });
      await act(async () => {
        fireEvent.click(screen.getByTestId("storage-image-retry"));
        await Promise.resolve();
      });
      expect(parentClick).not.toHaveBeenCalled();
    });
  });

  describe("src reset on prop change", () => {
    it("resets to loading skeleton when resolved URL changes", () => {
      const { rerender } = render(
        <StorageImage variant="thumb" thumbUrl="https://x/a.thumb.webp" alt="t" />
      );
      const img = screen.getByAltText("t") as HTMLImageElement;
      fireEvent.load(img);
      expect(screen.queryByTestId("storage-image-skeleton")).toBeNull();

      rerender(<StorageImage variant="thumb" thumbUrl="https://x/b.thumb.webp" alt="t" />);
      // New URL → skeleton overlay visible again until the new img loads.
      expect(screen.getByTestId("storage-image-skeleton")).toBeTruthy();
      const img2 = screen.getByAltText("t") as HTMLImageElement;
      expect(img2.src).toContain("b.thumb");
    });
  });
});
