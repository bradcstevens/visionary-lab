"use client"

import { useState, useCallback, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Upload, X, ChevronRight, ChevronLeft, Loader2, Check, AlertTriangle } from "lucide-react";
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

type PrepPhase = 'idle' | 'creating' | 'uploading' | 'analyzing' | 'ready' | 'error';

const STEPS = [
  { number: 1, title: "Name", description: "Choose a name for your project" },
  { number: 2, title: "Upload", description: "Upload baseline photos" },
  { number: 3, title: "Design", description: "Describe your vision with AI" },
  { number: 4, title: "Brief", description: "Review and edit the plan" },
  { number: 5, title: "Generate", description: "Review and launch" },
];

export function NewProjectWizard({ onComplete, onCancel }: NewProjectWizardProps) {
  const [currentStep, setCurrentStep] = useState(1);
  const [isLoading, setIsLoading] = useState(false);
  const [isGeneratingBrief, setIsGeneratingBrief] = useState(false);

  // Step 1 state
  const [projectName, setProjectName] = useState("");

  // Step 2 state
  const [roomFiles, setRoomFiles] = useState<RoomFile[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const dragCountRef = useRef(0);

  // Background preparation state (step 2→3 transition)
  const [prepPhase, setPrepPhase] = useState<PrepPhase>('idle');
  const [prepError, setPrepError] = useState<string | null>(null);
  const prepRunIdRef = useRef(0);

  // Draft project (created during prep)
  const [projectId, setProjectId] = useState<string | null>(null);
  const [uploadedRooms, setUploadedRooms] = useState<Array<{ id: string; label: string; url: string }>>([]);

  // Step 3 state
  const [analyses, setAnalyses] = useState<ImageAnalysisResult[]>([]);
  const [conversationHistory, setConversationHistory] = useState<ChatMessage[]>([]);
  const [focusedImageId, setFocusedImageId] = useState<string | null>(null);
  const [initialAiMessage, setInitialAiMessage] = useState("");

  // Step 4 state
  const [designBrief, setDesignBrief] = useState<DesignBrief | null>(null);

  // --- File handling with drag-and-drop ---

  const addFiles = useCallback((files: File[]) => {
    const imageFiles = files.filter(f => f.type.startsWith('image/'));
    const rejected = files.length - imageFiles.length;
    if (rejected > 0) {
      toast.warning(`${rejected} non-image file${rejected !== 1 ? 's' : ''} skipped — only PNG, JPG, JPEG accepted`);
    }
    if (imageFiles.length === 0) return;
    const newRoomFiles = imageFiles.map(file => ({
      file,
      name: file.name.replace(/\.[^/.]+$/, ""),
      preview: URL.createObjectURL(file),
    }));
    setRoomFiles(prev => [...prev, ...newRoomFiles]);
  }, []);

  const handleFileChange = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    addFiles(Array.from(event.target.files || []));
    // Reset input so re-selecting the same files works
    event.target.value = "";
  }, [addFiles]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const handleDragEnter = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    dragCountRef.current++;
    if (dragCountRef.current === 1) setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    dragCountRef.current--;
    if (dragCountRef.current === 0) setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    dragCountRef.current = 0;
    setIsDragging(false);
    addFiles(Array.from(e.dataTransfer.files));
  }, [addFiles]);

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

  // --- Step transitions ---

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

  /** Runs in background after step 2→3 advance. Updates prepPhase as it progresses. */
  const startDesignSessionPrep = useCallback(async () => {
    const runId = ++prepRunIdRef.current;
    setPrepError(null);

    try {
      // Phase 1: Create project (reuse if already created from a previous attempt)
      let currentProjectId = projectId;
      if (!currentProjectId) {
        setPrepPhase('creating');
        const project = await createProject({
          name: projectName,
          prompt: "Draft — pending AI Design Session",
        });
        if (runId !== prepRunIdRef.current) return;
        currentProjectId = project.id;
        setProjectId(currentProjectId);
      }

      // Phase 2: Upload room images
      setPrepPhase('uploading');
      const roomData = roomFiles.map(rf => ({ file: rf.file, name: rf.name }));
      await uploadRooms(currentProjectId, roomData);
      if (runId !== prepRunIdRef.current) return;
      toast.success(`${roomFiles.length} photo${roomFiles.length !== 1 ? 's' : ''} uploaded`);

      // Phase 3: AI analysis
      setPrepPhase('analyzing');
      const analysisResults = await analyzeImages(currentProjectId);
      if (runId !== prepRunIdRef.current) return;
      setAnalyses(analysisResults);

      const featureSummary = analysisResults
        .map(a => `• ${a.description}`)
        .join("\n");
      setInitialAiMessage(
        `I've analyzed your ${analysisResults.length} photo${analysisResults.length !== 1 ? 's' : ''}. Here's what I see:\n\n${featureSummary}\n\nWhat would you like to visualize in these spaces?`
      );

      setUploadedRooms(analysisResults.map((a, i) => ({
        id: a.room_id,
        label: roomFiles[i]?.name ?? `Room ${i + 1}`,
        url: roomFiles[i]?.preview ?? "",
      })));

      setPrepPhase('ready');
    } catch (error) {
      if (runId !== prepRunIdRef.current) return;
      console.error("Design session prep failed:", error);
      setPrepPhase('error');
      setPrepError(error instanceof Error ? error.message : "Setup failed — please try again");
    }
  }, [projectId, projectName, roomFiles]);

  const transitionToBriefEditor = async () => {
    if (!projectId) return;
    setIsGeneratingBrief(true);
    try {
      // Issue 004 of the per-image-object-quantities PRD: when this
      // transition is reached as a regenerate (designBrief is non-null
      // because the user already saw step 4 once), pass the current brief
      // back so per-image quantity / placement / skip overrides can be
      // carried forward by case-insensitive name match.
      const { brief, reconciliation_summary } = await generateBrief(
        projectId,
        conversationHistory,
        designBrief ?? undefined,
      );
      setDesignBrief(brief);
      if (reconciliation_summary.dropped > 0) {
        toast.info(
          `Carried forward ${reconciliation_summary.carried_forward} per-image quantity overrides; ${reconciliation_summary.dropped} were dropped because their objects could not be matched in the regenerated palette.`,
        );
      }
      setCurrentStep(4);
    } catch (error) {
      console.error("Failed to generate brief:", error);
      toast.error("Failed to generate Design Brief");
    } finally {
      setIsGeneratingBrief(false);
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
      // Advance to step 3 immediately; prep runs in the background
      setCurrentStep(3);
      startDesignSessionPrep();
      return;
    }
    if (currentStep === 4) {
      transitionToGenerate();
      return;
    }
    setCurrentStep(prev => Math.min(5, prev + 1));
  };

  const prevStep = () => {
    if (currentStep === 3) {
      // Going back from the design session — reset prep so user can modify files
      setPrepPhase('idle');
      setPrepError(null);
      setProjectId(null);
      prepRunIdRef.current++;
    }
    setCurrentStep(prev => Math.max(1, prev - 1));
  };

  const focusedLabel = focusedImageId
    ? uploadedRooms.find(r => r.id === focusedImageId)?.label ?? null
    : null;

  // Issue 003 of the per-image-object-quantities PRD: the brief editor now
  // wraps a tab strip with one tab per uploaded image, each rendering a
  // small thumbnail. Passing the full {id, label, url} triple here.
  const briefEditorImages = uploadedRooms.map(r => ({ id: r.id, label: r.label, url: r.url }));

  /** Step 2 only shows a checkmark once background prep is fully ready. */
  const isStepComplete = (stepNumber: number) => {
    if (stepNumber === 2) return currentStep > 2 && prepPhase === 'ready';
    return currentStep > stepNumber;
  };

  const isStepInProgress = (stepNumber: number) => {
    return stepNumber === 2 && currentStep === 3 && prepPhase !== 'ready' && prepPhase !== 'error' && prepPhase !== 'idle';
  };

  // --- Render helpers ---

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
                onKeyDown={(e) => e.key === "Enter" && canProceed(1) && nextStep()}
                placeholder="e.g., Backyard Fence Line — Spring 2026"
                className="text-base"
                autoFocus
              />
            </div>
          </div>
        );

      case 2:
        return (
          <div className="space-y-4">
            {/* Drop zone with real drag-and-drop support */}
            <div
              className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
                isDragging
                  ? 'border-primary bg-primary/5'
                  : 'border-muted-foreground/25 hover:border-muted-foreground/40'
              }`}
              onDragOver={handleDragOver}
              onDragEnter={handleDragEnter}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
            >
              <input type="file" id="room-upload" multiple accept="image/*" onChange={handleFileChange} className="hidden" />
              <label htmlFor="room-upload" className="cursor-pointer flex flex-col items-center gap-3">
                <div className={`w-12 h-12 rounded-full flex items-center justify-center transition-colors ${
                  isDragging ? 'bg-primary/10' : 'bg-muted'
                }`}>
                  <Upload className={`h-6 w-6 ${isDragging ? 'text-primary' : 'text-muted-foreground'}`} />
                </div>
                <div>
                  <span className="text-sm font-medium">
                    {isDragging ? 'Drop photos here' : 'Click to upload or drag and drop'}
                  </span>
                  <p className="text-xs text-muted-foreground mt-1">PNG, JPG, JPEG — upload as many as you need</p>
                </div>
              </label>
            </div>

            {/* Uploaded photo grid with labels */}
            {roomFiles.length > 0 && (
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <Label>Photos ({roomFiles.length})</Label>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="text-xs h-7"
                    onClick={() => document.getElementById('room-upload')?.click()}
                  >
                    <Upload className="h-3 w-3 mr-1" />
                    Add more
                  </Button>
                </div>
                <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3">
                  {roomFiles.map((rf, index) => (
                    <div key={index} className="space-y-1.5 group">
                      <div className="relative aspect-[4/3] rounded-lg overflow-hidden border bg-muted">
                        <img src={rf.preview} alt={rf.name} className="w-full h-full object-cover" />
                        <Button
                          size="sm"
                          variant="destructive"
                          className="absolute top-1.5 right-1.5 h-6 w-6 rounded-full p-0 opacity-0 group-hover:opacity-100 transition-opacity"
                          onClick={() => removeFile(index)}
                        >
                          <X className="h-3 w-3" />
                        </Button>
                      </div>
                      <Input
                        value={rf.name}
                        onChange={(e) => updateRoomName(index, e.target.value)}
                        placeholder="Label this photo"
                        className="text-xs h-7"
                      />
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        );

      case 3:
        // While prep is running (or failed), show progress view instead of chat
        if (prepPhase !== 'ready') {
          return (
            <div className="flex flex-col items-center justify-center py-16 space-y-6">
              {prepPhase === 'error' ? (
                <>
                  <div className="w-14 h-14 rounded-full bg-destructive/10 flex items-center justify-center">
                    <AlertTriangle className="h-7 w-7 text-destructive" />
                  </div>
                  <div className="text-center space-y-2 max-w-md">
                    <h3 className="font-semibold">Something went wrong</h3>
                    <p className="text-sm text-muted-foreground">{prepError}</p>
                  </div>
                  <div className="flex gap-2">
                    <Button variant="outline" onClick={prevStep}>Back to photos</Button>
                    <Button onClick={() => startDesignSessionPrep()}>Try again</Button>
                  </div>
                </>
              ) : (
                <>
                  <div className="w-14 h-14 rounded-full bg-primary/10 flex items-center justify-center">
                    <Loader2 className="h-7 w-7 animate-spin text-primary" />
                  </div>
                  <div className="text-center space-y-1.5">
                    <h3 className="font-semibold">
                      {prepPhase === 'creating' && 'Creating project...'}
                      {prepPhase === 'uploading' && `Uploading ${roomFiles.length} photo${roomFiles.length !== 1 ? 's' : ''}...`}
                      {prepPhase === 'analyzing' && 'Analyzing your photos...'}
                      {prepPhase === 'idle' && 'Preparing...'}
                    </h3>
                    <p className="text-sm text-muted-foreground max-w-sm">
                      {prepPhase === 'creating' && 'Setting up your workspace'}
                      {prepPhase === 'uploading' && 'Saving your photos to the cloud'}
                      {prepPhase === 'analyzing' && 'AI is identifying features, materials, and zones — this takes a moment'}
                      {prepPhase === 'idle' && 'Getting things ready'}
                    </p>
                  </div>
                  {/* Phase progress indicator */}
                  <div className="flex items-center gap-2">
                    {(['creating', 'uploading', 'analyzing'] as const).map((phase) => {
                      const phaseOrder: PrepPhase[] = ['creating', 'uploading', 'analyzing'];
                      const currentIdx = phaseOrder.indexOf(prepPhase as typeof phase);
                      const phaseIdx = phaseOrder.indexOf(phase);
                      const isDone = phaseIdx < currentIdx;
                      const isCurrent = phaseIdx === currentIdx;
                      return (
                        <div
                          key={phase}
                          className={`h-1.5 rounded-full transition-all duration-500 ${
                            isDone ? 'w-8 bg-primary' : isCurrent ? 'w-8 bg-primary animate-pulse' : 'w-8 bg-muted'
                          }`}
                        />
                      );
                    })}
                  </div>
                  <p className="text-xs text-muted-foreground">Keep this tab open while we prepare your design session</p>
                </>
              )}
            </div>
          );
        }

        // Prep is ready — render the full design chat
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
                isGeneratingBrief={isGeneratingBrief}
                canGenerateBrief={conversationHistory.length >= 2}
                initialMessage={initialAiMessage}
                conversationHistory={conversationHistory}
                onHistoryUpdate={setConversationHistory}
              />
            </div>
          </div>
        );

      case 4:
        return designBrief ? (
          <DesignBriefEditor brief={designBrief} onChange={setDesignBrief} images={briefEditorImages} />
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

        {/* Stepper with labels, checkmarks, and in-progress state */}
        <div className="flex items-center gap-1 pt-4">
          {STEPS.map((step, i) => {
            const completed = isStepComplete(step.number);
            const active = currentStep === step.number;
            const inProgress = isStepInProgress(step.number);
            return (
              <div key={step.number} className="flex items-center gap-1">
                <div className="flex items-center gap-1.5">
                  <div className={`w-7 h-7 rounded-full flex items-center justify-center text-xs font-medium transition-all ${
                    completed
                      ? "bg-primary text-primary-foreground"
                      : active
                        ? "bg-primary text-primary-foreground ring-2 ring-primary/20 ring-offset-2 ring-offset-background"
                        : inProgress
                          ? "bg-primary/20 text-primary"
                          : "bg-muted text-muted-foreground"
                  }`}>
                    {completed ? (
                      <Check className="h-3.5 w-3.5" />
                    ) : inProgress ? (
                      <Loader2 className="h-3.5 w-3.5 animate-spin" />
                    ) : (
                      step.number
                    )}
                  </div>
                  <span className={`text-xs hidden sm:inline transition-colors ${
                    active ? "font-medium text-foreground" : "text-muted-foreground"
                  }`}>{step.title}</span>
                </div>
                {i < STEPS.length - 1 && (
                  <div className={`w-4 lg:w-8 h-px mx-0.5 transition-colors ${completed ? "bg-primary" : "bg-muted"}`} />
                )}
              </div>
            );
          })}
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
              {currentStep === 4 ? "Save & Continue" : "Next"}
              {!isLoading && <ChevronRight className="h-4 w-4 ml-1" />}
            </Button>
          )}
        </div>
      </CardFooter>
    </Card>
  );
}