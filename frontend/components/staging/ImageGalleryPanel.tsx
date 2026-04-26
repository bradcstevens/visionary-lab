"use client"

import { useState } from "react";
import { ChevronDown, ChevronRight, Eye } from "lucide-react";
import type { ImageAnalysisResult } from "@/services/stagingApi";

interface ImageItem {
  id: string;
  label: string;
  url: string;
}

interface ImageGalleryPanelProps {
  images: ImageItem[];
  analyses: ImageAnalysisResult[];
  focusedImageId: string | null;
  onFocusImage: (imageId: string | null) => void;
  perImageNotes: Record<string, string>;
}

interface ImageGroup {
  name: string;
  images: ImageItem[];
}

function groupImages(images: ImageItem[], analyses: ImageAnalysisResult[]): ImageGroup[] {
  const analysisMap = new Map(analyses.map(a => [a.room_id, a]));
  const groups = new Map<string, ImageItem[]>();

  for (const img of images) {
    const analysis = analysisMap.get(img.id);
    const primaryFeature = analysis?.features[0] ?? "Other";
    const groupName = primaryFeature.charAt(0).toUpperCase() + primaryFeature.slice(1);
    if (!groups.has(groupName)) groups.set(groupName, []);
    groups.get(groupName)!.push(img);
  }

  return Array.from(groups.entries()).map(([name, imgs]) => ({ name, images: imgs }));
}

export function ImageGalleryPanel({
  images,
  analyses,
  focusedImageId,
  onFocusImage,
  perImageNotes,
}: ImageGalleryPanelProps) {
  const groups = groupImages(images, analyses);
  const [collapsed, setCollapsed] = useState<Set<string>>(new Set());

  const toggleGroup = (name: string) => {
    setCollapsed(prev => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });
  };

  return (
    <div className="h-full overflow-y-auto p-4 space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="font-semibold text-sm">Your Photos</h3>
        <span className="text-xs text-muted-foreground">{images.length} images</span>
      </div>

      {groups.map(group => (
        <div key={group.name}>
          <button
            onClick={() => toggleGroup(group.name)}
            className="flex items-center gap-1 text-xs font-semibold text-primary uppercase tracking-wide mb-2 w-full"
          >
            {collapsed.has(group.name) ? <ChevronRight className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
            {group.name} ({group.images.length})
          </button>

          {!collapsed.has(group.name) && (
            <div className="grid grid-cols-3 gap-1.5">
              {group.images.map(img => (
                <button
                  key={img.id}
                  onClick={() => onFocusImage(focusedImageId === img.id ? null : img.id)}
                  className={`relative aspect-video rounded overflow-hidden border-2 transition-colors ${
                    focusedImageId === img.id ? "border-primary" : "border-transparent hover:border-muted-foreground/30"
                  }`}
                >
                  <img src={img.url} alt={img.label} className="w-full h-full object-cover" />
                  {focusedImageId === img.id && (
                    <div className="absolute top-0 right-0 bg-primary rounded-bl p-0.5">
                      <Eye className="h-3 w-3 text-primary-foreground" />
                    </div>
                  )}
                  {perImageNotes[img.id] && (
                    <div className="absolute bottom-0 left-0 bg-yellow-500/80 rounded-tr px-1">
                      <span className="text-[9px] text-black font-medium">NOTE</span>
                    </div>
                  )}
                </button>
              ))}
            </div>
          )}
        </div>
      ))}

      <p className="text-[10px] text-muted-foreground text-center pt-2">
        Click any image to focus the conversation on that area
      </p>
    </div>
  );
}
