"use client"

import { Undo2, Ban } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import type {
  ImageObjectOverride,
  ObjectEntry,
} from "@/services/stagingApi";

interface PerImageObjectTableProps {
  palette: ObjectEntry[];
  overrides: ImageObjectOverride[];
  onChange: (overrides: ImageObjectOverride[]) => void;
}

const COLUMN_TEMPLATE =
  "grid-cols-[1.4fr_80px_1.2fr_80px_80px_60px]";

// "Default-equivalent" override: same quantity as palette default, no placement
// override (null), and enabled=true. We drop entries that match this state so
// per_image_objects stays sparse.
function isDefaultEquivalent(
  ovr: ImageObjectOverride,
  palette: ObjectEntry,
): boolean {
  return (
    ovr.quantity === palette.default_quantity &&
    ovr.placement === null &&
    ovr.enabled === true
  );
}

// Replace or append an override; if the resulting entry is default-equivalent,
// drop it instead.
function upsertOrPrune(
  overrides: ImageObjectOverride[],
  next: ImageObjectOverride,
  palette: ObjectEntry,
): ImageObjectOverride[] {
  const filtered = overrides.filter((o) => o.object_id !== next.object_id);
  if (isDefaultEquivalent(next, palette)) {
    return filtered;
  }
  return [...filtered, next];
}

function removeForId(
  overrides: ImageObjectOverride[],
  objectId: string,
): ImageObjectOverride[] {
  return overrides.filter((o) => o.object_id !== objectId);
}

export function PerImageObjectTable({
  palette,
  overrides,
  onChange,
}: PerImageObjectTableProps) {
  if (palette.length === 0) {
    return (
      <div className="text-sm text-muted-foreground py-6 text-center border border-dashed rounded-md">
        No objects in the palette yet — add objects in the Default Palette tab
        before adjusting per-image overrides.
      </div>
    );
  }

  const overrideById = new Map(overrides.map((o) => [o.object_id, o]));

  const handleQuantityChange = (palette_entry: ObjectEntry, raw: string) => {
    // Critique catch: parseInt(...) || default would silently turn 0 into
    // the palette default. Use Number.parseInt + Number.isNaN so a literal
    // 0 is preserved (0 IS a valid skip signal).
    const parsed = Number.parseInt(raw, 10);
    const nextQuantity = Number.isNaN(parsed) ? palette_entry.default_quantity : Math.max(0, parsed);
    const existing = overrideById.get(palette_entry.id);
    const nextOverride: ImageObjectOverride = {
      object_id: palette_entry.id,
      quantity: nextQuantity,
      // Preserve any existing placement override; a new override gets null.
      placement: existing?.placement ?? null,
      // Editing quantity to a positive value implies "not skipped" — keep
      // the canonical state model consistent so we can't end up with
      // {enabled: false, quantity: 5}.
      enabled: nextQuantity === 0 ? (existing?.enabled ?? true) : true,
    };
    onChange(upsertOrPrune(overrides, nextOverride, palette_entry));
  };

  const handlePlacementChange = (palette_entry: ObjectEntry, raw: string) => {
    const trimmed = raw.trim();
    const nextPlacement = trimmed === "" ? null : raw;
    const existing = overrideById.get(palette_entry.id);
    const nextOverride: ImageObjectOverride = {
      object_id: palette_entry.id,
      quantity: existing?.quantity ?? palette_entry.default_quantity,
      placement: nextPlacement,
      enabled: existing?.enabled ?? true,
    };
    onChange(upsertOrPrune(overrides, nextOverride, palette_entry));
  };

  const handleSkip = (palette_entry: ObjectEntry) => {
    // Canonical "skip" state: enabled=false AND quantity=0. Either flag
    // alone would skip per the resolver, but storing both keeps the
    // intent unambiguous.
    const nextOverride: ImageObjectOverride = {
      object_id: palette_entry.id,
      quantity: 0,
      placement: null,
      enabled: false,
    };
    // Skip is always a non-default state — never default-equivalent.
    onChange([
      ...overrides.filter((o) => o.object_id !== palette_entry.id),
      nextOverride,
    ]);
  };

  const handleUseDefault = (palette_entry: ObjectEntry) => {
    onChange(removeForId(overrides, palette_entry.id));
  };

  return (
    <div className="space-y-3">
      <div
        className={`grid ${COLUMN_TEMPLATE} gap-2 text-xs font-medium text-muted-foreground`}
      >
        <div>Object</div>
        <div>Qty</div>
        <div>Placement</div>
        <div></div>
        <div></div>
        <div></div>
      </div>

      {palette.map((entry) => {
        const ovr = overrideById.get(entry.id);
        const effectiveQuantity = ovr?.quantity ?? entry.default_quantity;
        const effectivePlacement = ovr?.placement ?? entry.placement;
        const isSkipped = ovr !== undefined && (!ovr.enabled || ovr.quantity === 0);
        const hasOverride = ovr !== undefined;

        return (
          <div key={entry.id} className={`grid ${COLUMN_TEMPLATE} gap-2 items-center`}>
            <div className="flex items-center gap-2 min-w-0">
              <span className="text-sm truncate" title={entry.name}>
                {entry.name || <em className="text-muted-foreground">unnamed</em>}
              </span>
              {hasOverride && (
                <Badge
                  variant="secondary"
                  className="text-[10px] h-4 px-1.5 shrink-0"
                  data-testid="override-indicator"
                >
                  override
                </Badge>
              )}
            </div>
            <Input
              type="number"
              value={effectiveQuantity}
              onChange={(e) => handleQuantityChange(entry, e.target.value)}
              min={0}
              disabled={isSkipped}
              className="text-sm h-8"
              data-testid={`qty-input-${entry.id}`}
            />
            <Input
              value={effectivePlacement}
              onChange={(e) => handlePlacementChange(entry, e.target.value)}
              placeholder="(inherit palette)"
              disabled={isSkipped}
              className="text-sm h-8"
              data-testid={`placement-input-${entry.id}`}
            />
            <Button
              size="sm"
              variant={isSkipped ? "default" : "ghost"}
              onClick={() => handleSkip(entry)}
              className="h-8 text-xs"
              title="Skip in this image"
              data-testid={`skip-btn-${entry.id}`}
            >
              <Ban className="h-3.5 w-3.5 mr-1" /> Skip
            </Button>
            <Button
              size="sm"
              variant="ghost"
              onClick={() => handleUseDefault(entry)}
              disabled={!hasOverride}
              className="h-8 text-xs"
              title="Reset to palette defaults"
              data-testid={`use-default-btn-${entry.id}`}
            >
              <Undo2 className="h-3.5 w-3.5 mr-1" /> Default
            </Button>
            <div></div>
          </div>
        );
      })}
    </div>
  );
}
