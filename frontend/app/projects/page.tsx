"use client"

import { useState, useEffect, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import Link from "next/link";
import { Plus, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ProjectCard } from "@/components/staging/ProjectCard";
import { listProjects, deleteProject, StagingProject } from "@/services/stagingApi";
import { sasTokenService } from "@/services/sas-token";
import { toast } from "sonner";

function ProjectsList() {
  const [projects, setProjects] = useState<StagingProject[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const searchParams = useSearchParams();

  useEffect(() => {
    loadProjects();
  }, []);

  const loadProjects = async () => {
    try {
      setIsLoading(true);
      const data = await listProjects();
      const projectList = Array.isArray(data) ? data : [];

      // Resolve blob URLs with SAS tokens
      try {
        const tokens = await sasTokenService.getTokens();
        for (const project of projectList) {
          for (const room of project.rooms) {
            if (room.original_image_url && !room.original_image_url.includes('?')) {
              room.original_image_url = `${room.original_image_url}?${tokens.imageSasToken}`;
            }
          }
        }
      } catch (sasError) {
        console.warn('Failed to get SAS tokens for project list:', sasError);
      }

      setProjects(projectList);
    } catch (error) {
      console.error('Failed to load projects:', error);
      toast.error('Failed to load projects');
    } finally {
      setIsLoading(false);
    }
  };

  const handleDeleteProject = async (projectId: string) => {
    const project = projects.find(p => p.id === projectId);
    if (!confirm(`Delete "${project?.name ?? 'this project'}" and all its Azure artifacts? This cannot be undone.`)) return;
    try {
      await deleteProject(projectId);
      toast.success('Project deleted');
      loadProjects();
    } catch (error) {
      toast.error(error instanceof Error ? error.message : 'Failed to delete project');
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="flex items-center gap-2 text-muted-foreground">
          <Loader2 className="h-5 w-5 animate-spin" />
          Loading projects...
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Virtual Staging Projects</h1>
          <p className="text-muted-foreground mt-1">
            Create and manage your virtual staging projects
          </p>
        </div>
        <Button asChild>
          <Link href="/projects/new">
            <Plus className="h-4 w-4 mr-2" />
            New Project
          </Link>
        </Button>
      </div>

      {/* Projects Grid */}
      {projects.length === 0 ? (
        <div className="text-center py-12">
          <div className="space-y-3">
            <h3 className="text-xl font-semibold">No projects yet</h3>
            <p className="text-muted-foreground">
              Get started by creating your first virtual staging project
            </p>
            <Button asChild>
              <Link href="/projects/new">
                <Plus className="h-4 w-4 mr-2" />
                Create First Project
              </Link>
            </Button>
          </div>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
          {projects.map((project) => (
            <ProjectCard key={project.id} project={project} onDelete={handleDeleteProject} />
          ))}
        </div>
      )}
    </div>
  );
}

export default function ProjectsPage() {
  return (
    <div className="container mx-auto px-4 py-8">
      <Suspense fallback={
        <div className="flex items-center justify-center h-64">
          <div className="flex items-center gap-2 text-muted-foreground">
            <Loader2 className="h-5 w-5 animate-spin" />
            Loading...
          </div>
        </div>
      }>
        <ProjectsList />
      </Suspense>
    </div>
  );
}