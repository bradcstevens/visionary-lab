"use client"

import { useState, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Textarea } from "@/components/ui/textarea";
import { Upload, X, ChevronRight, ChevronLeft, Loader2 } from "lucide-react";
import { createProject, uploadRooms, CreateProjectRequest, StagingProject } from "@/services/stagingApi";
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

export function NewProjectWizard({ onComplete, onCancel }: NewProjectWizardProps) {
  const [currentStep, setCurrentStep] = useState(1);
  const [isLoading, setIsLoading] = useState(false);
  const [formData, setFormData] = useState({
    name: '',
    prompt: '',
    style: 'modern luxury',
    variations_per_room: 5,
  });
  const [roomFiles, setRoomFiles] = useState<RoomFile[]>([]);

  const steps = [
    { number: 1, title: 'Project Name', description: 'Choose a name for your project' },
    { number: 2, title: 'Upload Rooms', description: 'Upload room images to stage' },
    { number: 3, title: 'Styling Prompt', description: 'Describe your desired styling' },
    { number: 4, title: 'Confirm', description: 'Review and generate' },
  ];

  const handleFileChange = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files || []);
    
    const newRoomFiles = files.map(file => ({
      file,
      name: file.name.replace(/\.[^/.]+$/, ''), // Remove extension for default name
      preview: URL.createObjectURL(file),
    }));

    setRoomFiles(prev => [...prev, ...newRoomFiles]);
  }, [roomFiles.length]);

  const removeFile = (index: number) => {
    setRoomFiles(prev => {
      const updated = [...prev];
      URL.revokeObjectURL(updated[index].preview); // Clean up object URL
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

  const canProceedFromStep = (step: number) => {
    switch (step) {
      case 1:
        return formData.name.trim().length > 0;
      case 2:
        return roomFiles.length > 0;
      case 3:
        return formData.prompt.trim().length > 0;
      case 4:
        return true;
      default:
        return false;
    }
  };

  const nextStep = () => {
    if (canProceedFromStep(currentStep)) {
      setCurrentStep(prev => Math.min(4, prev + 1));
    }
  };

  const prevStep = () => {
    setCurrentStep(prev => Math.max(1, prev - 1));
  };

  const handleSubmit = async () => {
    if (!canProceedFromStep(4)) return;

    setIsLoading(true);
    try {
      // Create the project
      const projectRequest: CreateProjectRequest = {
        name: formData.name,
        prompt: formData.prompt,
        style: formData.style,
        variations_per_room: formData.variations_per_room,
      };

      const project = await createProject(projectRequest);
      toast.success('Project created successfully');

      // Upload rooms
      const roomData = roomFiles.map(roomFile => ({
        file: roomFile.file,
        name: roomFile.name,
      }));

      await uploadRooms(project.id, roomData);
      toast.success('Rooms uploaded successfully');

      onComplete(project);
    } catch (error) {
      console.error('Failed to create project:', error);
      toast.error(error instanceof Error ? error.message : 'Failed to create project');
    } finally {
      setIsLoading(false);
    }
  };

  const renderStep = () => {
    switch (currentStep) {
      case 1:
        return (
          <div className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="project-name">Project Name</Label>
              <Input
                id="project-name"
                value={formData.name}
                onChange={(e) => setFormData(prev => ({ ...prev, name: e.target.value }))}
                placeholder="e.g., Living Room Redesign"
                className="text-base"
              />
            </div>
          </div>
        );

      case 2:
        return (
          <div className="space-y-4">
            {/* File Upload Area */}
            <div className="border-2 border-dashed border-muted-foreground/25 rounded-lg p-6 text-center">
              <input
                type="file"
                id="room-upload"
                multiple
                accept="image/*"
                onChange={handleFileChange}
                className="hidden"
              />
              <label 
                htmlFor="room-upload" 
                className="cursor-pointer flex flex-col items-center gap-2"
              >
                <Upload className="h-8 w-8 text-muted-foreground" />
                <div className="text-sm">
                  <span className="font-medium">Click to upload</span> or drag and drop
                </div>
                <div className="text-xs text-muted-foreground">
                  PNG, JPG, JPEG — no limit on images
                </div>
              </label>
            </div>

            {/* File Preview Grid */}
            {roomFiles.length > 0 && (
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <Label>Uploaded Rooms ({roomFiles.length})</Label>
                </div>
                <div className="grid grid-cols-2 gap-3">
                  {roomFiles.map((roomFile, index) => (
                    <div key={index} className="space-y-2">
                      <div className="relative aspect-video">
                        <img
                          src={roomFile.preview}
                          alt={roomFile.name}
                          className="w-full h-full object-cover rounded-md border"
                        />
                        <Button
                          size="sm"
                          variant="destructive"
                          className="absolute -top-2 -right-2 h-6 w-6 rounded-full p-0"
                          onClick={() => removeFile(index)}
                        >
                          <X className="h-3 w-3" />
                        </Button>
                      </div>
                      <Input
                        value={roomFile.name}
                        onChange={(e) => updateRoomName(index, e.target.value)}
                        placeholder="Room name"
                        className="text-sm"
                      />
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        );

      case 3:
        return (
          <div className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="prompt">Styling Direction</Label>
              <Textarea
                id="prompt"
                value={formData.prompt}
                onChange={(e) => setFormData(prev => ({ ...prev, prompt: e.target.value }))}
                placeholder="Describe the style you want to achieve... e.g., modern minimalist with warm lighting, neutral colors, and contemporary furniture"
                rows={4}
                className="text-base resize-none"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="style">Style Category</Label>
              <Input
                id="style"
                value={formData.style}
                onChange={(e) => setFormData(prev => ({ ...prev, style: e.target.value }))}
                placeholder="e.g., modern luxury, rustic farmhouse, industrial"
              />
            </div>
          </div>
        );

      case 4:
        return (
          <div className="space-y-6">
            <div className="space-y-4">
              <div>
                <Label className="text-sm font-medium text-muted-foreground">Project Name</Label>
                <p className="text-base">{formData.name}</p>
              </div>
              
              <div>
                <Label className="text-sm font-medium text-muted-foreground">Rooms ({roomFiles.length})</Label>
                <div className="flex flex-wrap gap-2 mt-1">
                  {roomFiles.map((roomFile, index) => (
                    <Badge key={index} variant="secondary" className="text-xs">
                      {roomFile.name}
                    </Badge>
                  ))}
                </div>
              </div>

              <div>
                <Label className="text-sm font-medium text-muted-foreground">Styling Direction</Label>
                <p className="text-sm text-muted-foreground mt-1 leading-relaxed">
                  {formData.prompt}
                </p>
              </div>

              <div>
                <Label className="text-sm font-medium text-muted-foreground">Variations per Room</Label>
                <p className="text-base">{formData.variations_per_room}</p>
              </div>
            </div>

            <div className="p-4 bg-muted/50 rounded-lg">
              <div className="text-sm font-medium mb-2">Generation Summary</div>
              <div className="text-sm text-muted-foreground">
                This will generate <strong>{roomFiles.length * formData.variations_per_room}</strong> total variations 
                across <strong>{roomFiles.length}</strong> rooms.
              </div>
            </div>
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <Card className="w-full max-w-2xl mx-auto">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle>New Staging Project</CardTitle>
            <p className="text-sm text-muted-foreground mt-1">
              {steps[currentStep - 1]?.description}
            </p>
          </div>
          <Badge variant="outline" className="text-xs">
            Step {currentStep} of 4
          </Badge>
        </div>

        {/* Progress Steps */}
        <div className="flex items-center gap-2 pt-4">
          {steps.map((step) => (
            <div key={step.number} className="flex items-center gap-2">
              <div className={`
                w-8 h-8 rounded-full flex items-center justify-center text-xs font-medium
                ${currentStep >= step.number 
                  ? 'bg-primary text-primary-foreground' 
                  : 'bg-muted text-muted-foreground'
                }
              `}>
                {step.number}
              </div>
              {step.number < 4 && (
                <div className={`w-8 h-px ${currentStep > step.number ? 'bg-primary' : 'bg-muted'}`} />
              )}
            </div>
          ))}
        </div>
      </CardHeader>

      <CardContent>
        {renderStep()}
      </CardContent>

      <CardFooter className="flex items-center justify-between">
        <div className="flex gap-2">
          <Button
            variant="outline"
            onClick={onCancel}
          >
            Cancel
          </Button>
          
          {currentStep > 1 && (
            <Button
              variant="ghost"
              onClick={prevStep}
              disabled={isLoading}
            >
              <ChevronLeft className="h-4 w-4 mr-1" />
              Back
            </Button>
          )}
        </div>

        <div className="flex gap-2">
          {currentStep < 4 ? (
            <Button
              onClick={nextStep}
              disabled={!canProceedFromStep(currentStep) || isLoading}
            >
              Next
              <ChevronRight className="h-4 w-4 ml-1" />
            </Button>
          ) : (
            <Button
              onClick={handleSubmit}
              disabled={!canProceedFromStep(4) || isLoading}
              className="min-w-[120px]"
            >
              {isLoading ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Creating...
                </>
              ) : (
                'Generate Project'
              )}
            </Button>
          )}
        </div>
      </CardFooter>
    </Card>
  );
}