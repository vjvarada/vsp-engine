// src/components/sidebar/RefinePanel.tsx
import React from "react";
import { Pencil, MousePointer, Square } from "lucide-react";
import { cn } from "@/lib/utils";
import { useUiStore } from "@/store/uiStore";

type RefineToolId = "point3d" | "bbox2d";

interface RefineToolConfig {
  readonly id: RefineToolId;
  readonly icon: React.ReactNode;
  readonly label: string;
  readonly description: string;
}

const REFINE_TOOLS: readonly RefineToolConfig[] = [
  {
    id: "point3d",
    icon: <MousePointer size={14} />,
    label: "3D Point (SAM-Med3D)",
    description: "Shift-click in 3D viewport  SAM-Med3D-turbo refines mask (~2 s)",
  },
  {
    id: "bbox2d",
    icon: <Square size={14} />,
    label: "2D Box (MedSAM)",
    description: "Draw bbox on MPR slice  MedSAM per-slice refinement",
  },
] as const;

const RefinePanel: React.FC = () => {
  const [activeTool, setActiveTool] = React.useState<RefineToolId | null>(null);
  const setRefineMode = useUiStore((s) => s.setRefineMode);
  const markComplete = useUiStore((s) => s.markStepComplete);
  const setActiveStep = useUiStore((s) => s.setActiveStep);

  const handleSelectTool = (id: RefineToolId) => {
    const next = activeTool === id ? null : id;
    setActiveTool(next);
    setRefineMode(next !== null);
  };

  return (
    <div className="p-4 space-y-4">
      <p className="text-[11px] text-muted-foreground/70 font-mono leading-relaxed">
        Optional: Fix any mislabeled regions. This step can be skipped.
      </p>

      <div className="space-y-2">
        {REFINE_TOOLS.map((tool) => (
          <button
            key={tool.id}
            type="button"
            aria-pressed={activeTool === tool.id}
            onClick={() => handleSelectTool(tool.id)}
            className={cn(
              "w-full flex items-start gap-3 p-3 rounded border tech-transition text-left",
              activeTool === tool.id
                ? "border-primary/50 bg-primary/10"
                : "border-border/30 hover:border-primary/30 hover:bg-primary/5",
            )}
          >
            <span className={cn("mt-0.5 shrink-0", activeTool === tool.id ? "text-primary" : "text-muted-foreground")}>
              {tool.icon}
            </span>
            <div className="space-y-0.5">
              <p className={cn("font-tech text-[10px] uppercase tracking-wider", activeTool === tool.id ? "text-primary" : "text-foreground/80")}>
                {tool.label}
              </p>
              <p className="text-[10px] text-muted-foreground/60 font-mono leading-relaxed">
                {tool.description}
              </p>
            </div>
          </button>
        ))}
      </div>

      {activeTool === "point3d" && (
        <div className="p-3 rounded bg-primary/5 border border-primary/20">
          <p className="text-[10px] text-primary/80 font-mono">Shift-click on a structure in the 3D viewport to refine its mask.</p>
        </div>
      )}

      {activeTool === "bbox2d" && (
        <div className="p-3 rounded bg-primary/5 border border-primary/20">
          <p className="text-[10px] text-primary/80 font-mono">Draw a bounding box on an MPR slice to refine that structure.</p>
        </div>
      )}

      <button
        type="button"
        onClick={() => { setRefineMode(false); markComplete("refine"); setActiveStep("mesh"); }}
        className="w-full py-1.5 rounded bg-primary/20 hover:bg-primary/30 text-primary font-tech text-[10px] uppercase tracking-wider tech-transition"
      >
        Done Refining  Generate Mesh
      </button>
    </div>
  );
};

export default RefinePanel;
