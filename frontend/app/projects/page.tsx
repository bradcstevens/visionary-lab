"use client"

import { useState, useEffect, useRef, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import Link from "next/link";
import { Plus, Loader2, RefreshCw, ServerCrash, WifiOff, ShieldAlert, ChevronDown } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ProjectCard } from "@/components/staging/ProjectCard";
import { listProjects, deleteProject, StagingProject } from "@/services/stagingApi";
import { sasTokenService } from "@/services/sas-token";
import { toast } from "sonner";
import { parseApiError, ParsedError } from "@/utils/error-utils";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";

function ErrorIcon({ statusCode }: { statusCode: number | null }) {
  const className = "h-10 w-10 text-destructive/70";
  if (statusCode === 401 || statusCode === 403) return <ShieldAlert className={className} />;
  if (statusCode && statusCode >= 500) return <ServerCrash className={className} />;
  if (!statusCode) return <WifiOff className={className} />;
  return <ServerCrash className={className} />;
}

function ProjectsList() {
  const [projects, setProjects] = useState<StagingProject[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isRetrying, setIsRetrying] = useState(false);
  const [loadError, setLoadError] = useState<ParsedError | null>(null);
  const searchParams = useSearchParams();
  const latestLoadIdRef = useRef(0);

  useEffect(() => {
    loadProjects();
  }, []);

  const loadProjects = async ({ blocking = true }: { blocking?: boolean } = {}) => {
    const loadId = ++latestLoadIdRef.current;
    try {
      if (blocking) setIsLoading(true);
      setLoadError(null);

      // Fetch project list and SAS tokens in parallel — the list call is
      // the slow path (Cosmos query + per-project rooms) and the token call
      // is independent. Running them sequentially adds ~one round trip to
      // every projects-page load.
      const [data, tokensResult] = await Promise.all([
        listProjects(),
        sasTokenService.getTokens().catch((sasError) => {
          console.warn('Failed to get SAS tokens for project list:', sasError);
          toast.warning('Image previews may not load — storage access token unavailable', {
            id: 'sas-token-warning',
            duration: 8000,
          });
          return null;
        }),
      ]);

      // Discard stale responses from overlapping fetches.
      if (loadId !== latestLoadIdRef.current) return;

      const projectList = Array.isArray(data) ? data : [];

      // Resolve blob URLs with SAS tokens (only if we got them).
      if (tokensResult) {
        for (const project of projectList) {
          for (const room of project.rooms) {
            if (room.original_image_url && !room.original_image_url.includes('?')) {
              room.original_image_url = `${room.original_image_url}?${tokensResult.imageSasToken}`;
            }
          }
        }
      }

      setProjects(projectList);
    } catch (error) {
      if (loadId !== latestLoadIdRef.current) return;
      console.error('Failed to load projects:', error);
      setLoadError(parseApiError(error));
    } finally {
      // Always reset on the latest call regardless of `blocking`. This
      // guarantees a stranded ``isLoading=true`` from a prior blocking
      // call cannot persist after a later non-blocking call wins the
      // ``latestLoadIdRef`` race.
      if (loadId === latestLoadIdRef.current) {
        setIsLoading(false);
      }
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

      {/* Error State */}
      {loadError && (
        <div className="relative overflow-hidden rounded-xl border border-destructive/20 bg-destructive/[0.03]">
          <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top_right,_var(--tw-gradient-stops))] from-destructive/[0.08] via-transparent to-transparent pointer-events-none" />
          <div className="relative px-6 py-8 flex flex-col items-center text-center gap-4 max-w-lg mx-auto">
            <div className="rounded-full bg-destructive/10 p-3">
              <ErrorIcon statusCode={loadError.statusCode} />
            </div>
            <div className="space-y-1.5">
              <h3 className="text-lg font-semibold tracking-tight">
                {loadError.statusCode
                  ? `${loadError.statusCode} — ${loadError.title}`
                  : loadError.title}
              </h3>
              <p className="text-sm text-muted-foreground leading-relaxed">
                Could not load your projects. Check that the backend is running and try again.
              </p>
            </div>

            {loadError.detail && (
              <Collapsible className="w-full max-w-md">
                <CollapsibleTrigger className="group inline-flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors cursor-pointer">
                  <ChevronDown className="h-3 w-3 transition-transform group-data-[state=open]:rotate-180" />
                  Technical details
                </CollapsibleTrigger>
                <CollapsibleContent>
                  <pre className="mt-2 rounded-lg bg-muted/60 border border-border/50 px-4 py-3 text-xs text-muted-foreground text-left font-mono whitespace-pre-wrap break-all overflow-hidden max-h-40 overflow-y-auto">
                    {loadError.detail}{loadError.isTruncated && "…"}
                  </pre>
                </CollapsibleContent>
              </Collapsible>
            )}

            <Button
              onClick={() => {
                setIsRetrying(true);
                loadProjects({ blocking: false }).finally(() => setIsRetrying(false));
              }}
              disabled={isRetrying}
              variant="outline"
              size="sm"
              className="mt-1"
            >
              {isRetrying ? (
                <Loader2 className="h-3.5 w-3.5 mr-1.5 animate-spin" />
              ) : (
                <RefreshCw className="h-3.5 w-3.5 mr-1.5" />
              )}
              Retry
            </Button>
          </div>
        </div>
      )}

      {/* Projects Grid */}
      {!loadError && projects.length === 0 ? (
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
      ) : !loadError ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
          {projects.map((project) => (
            <ProjectCard key={project.id} project={project} onDelete={handleDeleteProject} />
          ))}
        </div>
      ) : null}
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