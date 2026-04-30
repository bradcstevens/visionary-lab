"use client"

import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

export interface BriefEditorTabsImage {
  id: string;
  label: string;
  url: string;
}

interface BriefEditorTabsProps {
  images: BriefEditorTabsImage[];
  defaultTabContent: React.ReactNode;
  // Render function so the parent owns the per-image render (it has the
  // brief state). Keeps this component leaf / presentational.
  renderImageTabContent: (image: BriefEditorTabsImage) => React.ReactNode;
}

export const DEFAULT_TAB_VALUE = "__default__";

export function BriefEditorTabs({
  images,
  defaultTabContent,
  renderImageTabContent,
}: BriefEditorTabsProps) {
  return (
    <Tabs defaultValue={DEFAULT_TAB_VALUE} className="w-full">
      <TabsList className="w-full justify-start h-auto flex-wrap gap-1 p-1">
        <TabsTrigger
          value={DEFAULT_TAB_VALUE}
          className="px-3 py-1.5 text-sm"
          data-testid="tab-default-palette"
        >
          Default Palette
        </TabsTrigger>
        {images.map((img) => (
          <TabsTrigger
            key={img.id}
            value={img.id}
            className="px-2 py-1 gap-2 text-sm"
            data-testid={`tab-image-${img.id}`}
          >
            <span className="relative h-6 w-9 shrink-0 overflow-hidden rounded border border-border/60">
              {/* Uploaded URLs from the wizard are short-lived blob/data
                  URLs served from the local backend. next/image's runtime
                  sanity checks reject blob: protocol, so use plain <img>. */}
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={img.url}
                alt={img.label}
                className="h-full w-full object-cover"
              />
            </span>
            <span className="truncate max-w-[140px]">{img.label}</span>
          </TabsTrigger>
        ))}
      </TabsList>
      <TabsContent value={DEFAULT_TAB_VALUE} className="pt-4">
        {defaultTabContent}
      </TabsContent>
      {images.map((img) => (
        <TabsContent key={img.id} value={img.id} className="pt-4">
          {renderImageTabContent(img)}
        </TabsContent>
      ))}
    </Tabs>
  );
}
