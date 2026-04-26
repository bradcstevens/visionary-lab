"use client"

import { useState, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Upload, X, ChevronRight, ChevronLeft, Loader2 } from "lucide-react";
import {
  createProject, uploadRooms, analyzeImages, generateBrief, updateBrief,
  ChatMessage, DesignBrief, ImageAnalysisResult, StagingProject,
} from "@/services/stagingApi";
import { ImageGalleryPanel } from "./ImageGalleryPanel";
import { DesignChat } from "./DesignChat";
import { DesignBriefEditor } from "./DesignBriefEditor";
import { GenerationSummary } from "./GenerationSummary";
import { toast } from "sonner";

interface NewProjectWizardProps {
  onComplete: (project: StagingProject) => void;
  onCancel: () => void;
}

interface RoomFile {
  file: File;
  name: string;
  preview: string;
}

const STEPS = [
  { number: 1, title: "Name", description: "Choose a name for your project" },
  { number: 2, title: "Upload", description: "Upload baseline photos" },
  { number: 3, title: "AI Design Session", description: "Describe your vision" },
  { number: 4, title: "Design Brief", description: "Review and edit the plan" },
  { number: 5, title: "Generate", description: "Review and launch" },
];

export function NewProjectWizard({ onComplete, onCancel }: NewProjectWizardProps) {
  const [currentStep, setCurrentStep] = useState(1);
  const [isLoading, setIsLoading] = useState(false);

  // Step 1 state
  const [projectName, setProjectName] = useState("");

  // Step 2 state
  const [roomFiles, setRoomFiles] = useState<RoomFile[]>([]);

  // Draft project (created after Step 2)
  const [projectId, setProjectId] = useState<string | null>(null);
  const [uploadedRooms, setUploadedRooms] = useState<Array<{ id: string; label: string; url: string }>>([]);

  // Step 3 state
  const [analyses, setAnalyses] = useState<ImageAnalysisResult[]>([]);
  const [conversationHistory, setConversationHistory] = useState<ChatMessage[]>([]);
  const [focusedImageId, setFocusedImageId] = useState<string | null>(null);
  const [initialAiMessage, setInitialAiMessage] = useState("");

  // Step 4 state
  const [designBrief, setDesignBrief] = useState<DesignBrief | null>(null);

  const handleFileChange = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files || []);
    const newRoomFiles = files.map(file => ({
      file,
      name: file.name.replace(/\.[^/.]+$/, ""),
      preview: URL.createObjectURL(file),
    }));
    setRoomFiles(prev => [...prev, ...newRoomFiles]);
  }, []);

  const removeFile = (index: number) => {
    setRoomFiles(prev => {
      const updated = [...prev];
      URL.revokeObjectURL(updated[index].preview);
      updated.splice(index, 1);
      return updated;
    });
  };

  const updateRoomName = (index: number, name: string) => {
    setRoomFiles(prev => {
      const updated = [...prev];
      updated[index] = { ...updated[index], name };
      return updated;
    });
  };

  const canProceed = (step: number) => {
    switch (step) {
      case 1: return projectName.trim().length > 0;
      case 2: return roomFiles.length > 0;
      case 3: return conversationHistory.length >= 2;
      case 4: return designBrief !== null;
      case 5: return true;
      default: return false;
    }
  };

  const transitionToDesignSession = async () => {
    setIsLoading(true);
    try {
      const project = await createProject({
        name: projectName,
        prompt: "Draft — pending AI Design Session",
      });
      setProjectId(project.id);

      const roomData = roomFiles.map(rf => ({ file: rf.file, name: rf.name }));
      await uploadRooms(project.id, roomData);
      toast.success("Photos uploaded");

      const analysisResults = await analyzeImages(project.id);
      setAnalyses(analysisResults);

      const featureSummary = analysisResults
        .map(a => `• ${a.description}`)
        .join("\n");
      setInitialAiMessage(
        `I've analyzed your ${analysisResults.length} photos. Here's what I see:\n\n${featureSummary}\n\nWhat would you like to visualize in these spaces?`
      );

      setUploadedRooms(analysisResults.map((a, i) => ({
        id: a.room_id,
        label: roomFiles[i]?.name ?? `Room ${i + 1}`,
        url: roomFiles[i]?.preview ?? "",
      })));

      setCurrentStep(3);
    } catch (error) {
      console.error("Failed to set up design session:", error);
      toast.error(error instanceof Error ? error.message : "Setup failed");
    } finally {
      setIsLoading(false);
    }
  };

  const transitionToBriefEditor = async () => {
    if (!projectId) return;
    setIsLoading(true);
    try {
      const brief = await generateBrief(projectId, conversationHistory);
      setDesignBrief(brief);
      setCurrentStep(4);
    } catch (error) {
      console.error("Failed to generate brief:", error);
      toast.error("Failed to generate Design Brief");
    } finally {
      setIsLoading(false);
    }
  };

  const transitionToGenerate = async () => {
    if (!projectId || !designBrief) return;
    setIsLoading(true);
    try {
      await updateBrief(projectId, designBrief);
      setCurrentStep(5);
    } catch (error) {
      console.error("Failed to save brief:", error);
      toast.error("Failed to save Design Brief");
    } finally {
      setIsLoading(false);
    }
  };

  const handleGenerate = () => {
    if (!projectId) return;
    toast.success("Generation started! Redirecting to project...");
    onComplete({ id: projectId, name: projectName } as StagingProject);
  };

  const nextStep = () => {
    if (!canProceed(currentStep)) return;
    if (currentStep === 2) {
      transitionToDesignSession();
      return;
    }
    if (currentStep === 4) {
      transitionToGenerate();
      return;
    }
    setCurrentStep(prev => Math.min(5, prev + 1));
  };

  const prevStep = () => setCurrentStep(prev => Math.max(1, prev - 1));

  const focusedLabel = focusedImageId
    ? uploadedRooms.find(r => r.id === focusedImageId)?.label ?? null
    : null;

  const imageLabels = Object.fromEntries(uploadedRooms.map(r => [r.id, r.label]));

  const renderStep = () => {
    switch (currentStep) {
      case 1:
        return (
          <div className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="project-name">Project Name</Label>
              <Input
                id="project-name"
                value={projectName}
                onChange={(e) => setProjectName(e.target.value)}
                placeholder="e.g., Backyard Fence Line — Spring 2026"
                className="text-base"
              />
            </div>
          </div>
        );

      case 2:
        return (
          <div className="space-y-4">
            <div className="border-2 border-dashed border-muted-foreground/25 rounded-lg p-6 text-center">
              <input type="file" id="room-upload" multiple accept="image/*" onChange={handleFileChange} className="hidden" />
              <label htmlFor="room-upload" className="cursor-pointer flex flex-col items-center gap-2">
                <Upload className="h-8 w-8 text-muted-foreground" />
                <div className="text-sm"><span className="font-medium">Click to upload</span> or drag and drop</div>
                <div className="text-xs text-muted-foreground">PNG, JPG, JPEG — no limit on images</div>
              </label>
            </div>
            {roomFiles.length > 0 && (
              <div className="space-y-3">
                <Label>Uploaded Photos ({roomFiles.length})</Label>
                <div className="grid grid-cols-3 gap-3">
                  {roomFiles.map((rf, index) => (
                    <div key={index} className="space-y-2">
                      <div className="relative aspect-video">
                        <img src={rf.preview} alt={rf.name} className="w-full h-full object-cover rounded-md border" />
                        <Button size="sm" variant="destructive" className="absolute -top-2 -right-2 h-6 w-6 rounded-full p-0" onClick={() => removeFile(index)}>
                          <X className="h-3 w-3" />
                        </Button>
                      </div>
                      <Input value={rf.name} onChange={(e) => updateRoomName(index, e.target.value)} placeholder="Label" className="text-sm" />
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        );

      case 3:
        return (
          <div className="flex gap-0 -mx-6 -mb-6 h-[500px] border-t">
            <div className="w-[40%] border-r bg-muted/30 hidden md:block">
              <ImageGalleryPanel
                images={uploadedRooms}
                analyses={analyses}
                focusedImageId={focusedImageId}
                onFocusImage={setFocusedImageId}
                perImageNotes={designBrief?.per_image_notes ?? {}}
              />
            </div>
            <div className="flex-1">
              <DesignChat
                projectId={projectId!}
                focusedImageId={focusedImageId}
                focusedImageLabel={focusedLabel}
                onClearFocus={() => setFocusedImageId(null)}
                onReadyForBrief={transitionToBriefEditor}
                initialMessage={initialAiMessage}
                conversationHistory={conversationHistory}
                onHistoryUpdate={setConversationHistory}
              />
            </div>
          </div>
        );

      case 4:
        return designBrief ? (
          <DesignBriefEditor brief={designBrief} onChange={setDesignBrief} imageLabels={imageLabels} />
        ) : (
          <div className="flex items-center justify-center h-32">
            <Loader2 className="h-5 w-5 animate-spin" />
          </div>
        );

      case 5:
        return designBrief ? (
          <GenerationSummary projectName={projectName} imageCount={roomFiles.length} brief={designBrief} />
        ) : null;

      default:
        return null;
    }
  };

  return (
    <Card className="w-full max-w-4xl mx-auto">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle>New Project</CardTitle>
            <p className="text-sm text-muted-foreground mt-1">{STEPS[currentStep - 1]?.description}</p>
          </div>
          <Badge variant="outline" className="text-xs">Step {currentStep} of 5</Badge>
        </div>
        <div className="flex items-center gap-2 pt-4">
          {STEPS.map((step) => (
            <div key={step.number} className="flex items-center gap-2">
              <div className={`w-8 h-8 rounded-full flex items-center justify-center text-xs font-medium ${
                currentStep >= step.number ? "bg-primary text-primary-foreground" : "bg-muted text-muted-foreground"
              }`}>{step.number}</div>
              {step.number < 5 && <div className={`w-6 h-px ${currentStep > step.number ? "bg-primary" : "bg-muted"}`} />}
            </div>
          ))}
        </div>
      </CardHeader>

      <CardContent>{renderStep()}</CardContent>

      <CardFooter className="flex items-center justify-between">
        <div className="flex gap-2">
          <Button variant="outline" onClick={onCancel}>Cancel</Button>
          {currentStep > 1 && currentStep !== 3 && (
            <Button variant="ghost" onClick={prevStep} disabled={isLoading}>
              <ChevronLeft className="h-4 w-4 mr-1" /> Back
            </Button>
          )}
        </div>
        <div className="flex gap-2">
          {currentStep === 5 ? (
            <Button onClick={handleGenerate} disabled={isLoading} className="min-w-[140px]">
              {isLoading ? <><Loader2 className="h-4 w-4 mr-2 animate-spin" />Generating...</> : "Generate Project"}
            </Button>
          ) : currentStep === 3 ? null : (
            <Button onClick={nextStep} disabled={!canProceed(currentStep) || isLoading}>
              {isLoading ? <Loader2 className="h-4 w-4 mr-2 animate-spin" /> : null}
              {currentStep === 2 ? "Upload & Analyze" : currentStep === 4 ? "Save & Continue" : "Next"}
              {!isLoading && <ChevronRight className="h-4 w-4 ml-1" />}
            </Button>
          )}
        </div>
      </CardFooter>
    </Card>
  );
}