"use client"

import { Plus, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import type { ObjectCategory, ObjectEntry } from "@/services/stagingApi";

interface ObjectPaletteTableProps {
  objects: ObjectEntry[];
  onChange: (objects: ObjectEntry[]) => void;
}

const CATEGORY_OPTIONS: { value: ObjectCategory; label: string }[] = [
  { value: "plant", label: "Plant" },
  { value: "tree", label: "Tree" },
  { value: "rock", label: "Rock" },
  { value: "furniture", label: "Furniture" },
  { value: "lighting", label: "Lighting" },
  { value: "hardscape", label: "Hardscape" },
  { value: "decor", label: "Decor" },
  { value: "other", label: "Other" },
];

// 8 columns: Name | Description | Category | Default Qty | Size | Placement | Visual Notes | Trash
const COLUMN_TEMPLATE =
  "grid-cols-[1.2fr_1fr_120px_80px_1fr_1fr_1fr_40px]";

function makeBlankEntry(): ObjectEntry {
  return {
    // Defer to backend for the canonical UUID; UI generates a transient id
    // for new local rows so React's `key=` and the controlled inputs work
    // before save. crypto.randomUUID is widely available in modern browsers.
    id:
      typeof crypto !== "undefined" && "randomUUID" in crypto
        ? crypto.randomUUID()
        : `tmp-${Math.random().toString(36).slice(2, 11)}`,
    name: "",
    description: "",
    category: "other",
    default_quantity: 1,
    size: "",
    placement: "",
    visual_notes: "",
  };
}

export function ObjectPaletteTable({ objects, onChange }: ObjectPaletteTableProps) {
  const updateEntry = <K extends keyof ObjectEntry>(
    index: number,
    field: K,
    value: ObjectEntry[K],
  ) => {
    const updated = [...objects];
    updated[index] = { ...updated[index], [field]: value };
    onChange(updated);
  };

  const addEntry = () => {
    onChange([...objects, makeBlankEntry()]);
  };

  const removeEntry = (index: number) => {
    onChange(objects.filter((_, i) => i !== index));
  };

  return (
    <div className="space-y-3">
      <div
        className={`grid ${COLUMN_TEMPLATE} gap-2 text-xs font-medium text-muted-foreground`}
      >
        <div>Name</div>
        <div>Description</div>
        <div>Category</div>
        <div>Default Qty</div>
        <div>Size</div>
        <div>Placement</div>
        <div>Visual Notes</div>
        <div></div>
      </div>

      {objects.map((entry, idx) => (
        <div key={entry.id ?? idx} className={`grid ${COLUMN_TEMPLATE} gap-2`}>
          <Input
            value={entry.name}
            onChange={(e) => updateEntry(idx, "name", e.target.value)}
            placeholder="Object name"
            className="text-sm h-8"
          />
          <Input
            value={entry.description ?? ""}
            onChange={(e) => updateEntry(idx, "description", e.target.value)}
            placeholder="Detail (optional)"
            className="text-sm h-8"
          />
          <Select
            value={entry.category}
            onValueChange={(value) =>
              updateEntry(idx, "category", value as ObjectCategory)
            }
          >
            <SelectTrigger className="text-sm h-8">
              <SelectValue placeholder="Category" />
            </SelectTrigger>
            <SelectContent>
              {CATEGORY_OPTIONS.map((opt) => (
                <SelectItem key={opt.value} value={opt.value}>
                  {opt.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Input
            type="number"
            value={entry.default_quantity}
            onChange={(e) =>
              updateEntry(idx, "default_quantity", parseInt(e.target.value) || 1)
            }
            min={0}
            className="text-sm h-8"
          />
          <Input
            value={entry.size}
            onChange={(e) => updateEntry(idx, "size", e.target.value)}
            placeholder="e.g. 8-10 ft"
            className="text-sm h-8"
          />
          <Input
            value={entry.placement}
            onChange={(e) => updateEntry(idx, "placement", e.target.value)}
            placeholder="e.g. back row"
            className="text-sm h-8"
          />
          <Input
            value={entry.visual_notes ?? ""}
            onChange={(e) => updateEntry(idx, "visual_notes", e.target.value)}
            placeholder="Visual cue"
            className="text-sm h-8"
          />
          <Button
            size="sm"
            variant="ghost"
            onClick={() => removeEntry(idx)}
            className="h-8 w-8 p-0"
          >
            <Trash2 className="h-3.5 w-3.5 text-destructive" />
          </Button>
        </div>
      ))}

      <Button size="sm" variant="outline" onClick={addEntry} className="w-full">
        <Plus className="h-3.5 w-3.5 mr-1" /> Add Object
      </Button>
    </div>
  );
}
