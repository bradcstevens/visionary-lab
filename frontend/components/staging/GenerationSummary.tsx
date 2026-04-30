"use client"

import { Badge } from "@/components/ui/badge";
import type { DesignBrief } from "@/services/stagingApi";

interface GenerationSummaryProps {
  projectName: string;
  imageCount: number;
  brief: DesignBrief;
}

export function GenerationSummary({ projectName, imageCount, brief }: GenerationSummaryProps) {
  const totalVariations = imageCount * brief.settings.variations_per_room;

  return (
    <div className="space-y-6 max-w-2xl">
      <div className="space-y-4">
        <div>
          <span className="text-sm font-medium text-muted-foreground">Project</span>
          <p className="text-lg font-semibold">{projectName}</p>
        </div>

        <div className="grid grid-cols-3 gap-4">
          <div className="p-4 bg-muted/50 rounded-lg text-center">
            <div className="text-2xl font-bold">{imageCount}</div>
            <div className="text-xs text-muted-foreground">Images</div>
          </div>
          <div className="p-4 bg-muted/50 rounded-lg text-center">
            <div className="text-2xl font-bold">{brief.settings.variations_per_room}</div>
            <div className="text-xs text-muted-foreground">Per Image</div>
          </div>
          <div className="p-4 bg-muted/50 rounded-lg text-center">
            <div className="text-2xl font-bold">{totalVariations}</div>
            <div className="text-xs text-muted-foreground">Total Variations</div>
          </div>
        </div>
      </div>

      <div className="space-y-2">
        <span className="text-sm font-medium text-muted-foreground">Design Direction</span>
        <p className="text-sm leading-relaxed">{brief.global_instructions}</p>
      </div>

      {brief.object_palette.length > 0 && (
        <div className="space-y-2">
          <span className="text-sm font-medium text-muted-foreground">Objects ({brief.object_palette.length})</span>
          <div className="flex flex-wrap gap-1.5">
            {brief.object_palette.map((obj) => (
              <Badge key={obj.id} variant="secondary" className="text-xs">
                {obj.default_quantity}× {obj.name}
              </Badge>
            ))}
          </div>
        </div>
      )}

      {brief.preserve_elements.length > 0 && (
        <div className="space-y-2">
          <span className="text-sm font-medium text-muted-foreground">Preserving</span>
          <div className="flex flex-wrap gap-1.5">
            {brief.preserve_elements.map((el, i) => (
              <Badge key={i} variant="outline" className="text-xs">{el}</Badge>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
