// src/components/layout/VerticalToolbar.tsx
import React, { useCallback } from "react";
import {
  Upload,
  ScanSearch,
  Target,
  Bot,
  Pencil,
  Box,
  Download,
  type LucideIcon,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { useUiStore, type WorkflowStep } from "@/store/uiStore";

interface ToolConfig {
  readonly id: WorkflowStep;
  readonly icon: LucideIcon;
  readonly label: string;
  readonly tooltip: string;
}

const TOOLS: readonly ToolConfig[] = [
  { id: "upload",  icon: Upload,     label: "Upload",  tooltip: "Upload DICOM Study" },
  { id: "scout",   icon: ScanSearch, label: "Scout",   tooltip: "Fast HU-Threshold Scout" },
  { id: "select",  icon: Target,     label: "Select",  tooltip: "Select ROI & Anatomy" },
  { id: "segment", icon: Bot,        label: "AI Seg",  tooltip: "AI Segmentation (TotalSegmentator)" },
  { id: "refine",  icon: Pencil,     label: "Refine",  tooltip: "Interactive Refinement (SAM-Med3D)" },
  { id: "mesh",    icon: Box,        label: "Mesh",    tooltip: "Generate Watertight Mesh" },
  { id: "export",  icon: Download,   label: "Export",  tooltip: "Export STL / OBJ / 3MF" },
] as const;

interface SidebarIconProps {
  readonly icon: LucideIcon;
  readonly label: string;
  readonly tooltip: string;
  readonly isActive: boolean;
  readonly isCompleted: boolean;
  readonly onClick: () => void;
}

const SidebarIcon: React.FC<SidebarIconProps> = React.memo(
  ({ icon: Icon, label, tooltip, isActive, isCompleted, onClick }) => (
    <button
      type="button"
      title={tooltip}
      aria-label={tooltip}
      aria-pressed={isActive}
      onClick={onClick}
      className={cn(
        "flex flex-col items-center justify-center w-14 h-14 gap-1 tech-transition",
        "hover:bg-primary/10 rounded-none cursor-pointer select-none",
        isActive && "bg-primary/20 border-r-2 border-primary",
        !isActive && isCompleted && "opacity-60",
      )}
    >
      <Icon
        size={18}
        className={cn(
          "shrink-0",
          isActive ? "text-primary" : "text-muted-foreground",
          isCompleted && !isActive && "text-primary/50",
        )}
      />
      <span
        className={cn(
          "font-tech text-[9px] tracking-wider uppercase",
          isActive ? "text-primary" : "text-muted-foreground/60",
        )}
      >
        {label}
      </span>
    </button>
  ),
);
SidebarIcon.displayName = "SidebarIcon";

const VerticalToolbar: React.FC = () => {
  const activeStep = useUiStore((s) => s.activeStep);
  const completedSteps = useUiStore((s) => s.completedSteps);
  const setActiveStep = useUiStore((s) => s.setActiveStep);

  const handleClick = useCallback(
    (step: WorkflowStep) => () => setActiveStep(step),
    [setActiveStep],
  );

  return (
    <nav
      className="flex flex-col w-14 h-full tech-glass border-r border-border/50 shrink-0"
      aria-label="Workflow steps"
    >
      {TOOLS.map((tool) => (
        <SidebarIcon
          key={tool.id}
          icon={tool.icon}
          label={tool.label}
          tooltip={tool.tooltip}
          isActive={activeStep === tool.id}
          isCompleted={completedSteps.has(tool.id)}
          onClick={handleClick(tool.id)}
        />
      ))}
    </nav>
  );
};

export default VerticalToolbar;
