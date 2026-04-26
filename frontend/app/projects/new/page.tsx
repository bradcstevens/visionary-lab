"use client"

import { useState } from "react";
import { useRouter } from "next/navigation";
import { NewProjectWizard } from "@/components/staging/NewProjectWizard";
import { StagingProject } from "@/services/stagingApi";

export default function NewProjectPage() {
  const router = useRouter();

  const handleComplete = (project: StagingProject) => {
    router.push(`/projects/${project.id}`);
  };

  const handleCancel = () => {
    router.push('/projects');
  };

  return (
    <div className="container mx-auto px-4 py-8">
      <NewProjectWizard onComplete={handleComplete} onCancel={handleCancel} />
    </div>
  );
}