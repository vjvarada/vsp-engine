// src/components/layout/ContextOptionsPanel.tsx
// Collapsible 320px context panel with step header, step content, and mini-map.
import React, { useCallback } from "react";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { cn } from "@/lib/utils";
import { useUiStore, type WorkflowStep } from "@/store/uiStore";

interface ContextOptionsPanelProps {
  readonly children: React.ReactNode;
  readonly stepTitle: string;
  readonly stepIcon: React.ReactNode;
}

const STEP_MINI_ICONS: ReadonlyArray<{ id: WorkflowStep; label: string }> = [
  { id: "upload",  label: "" },
  { id: "scout",   label: "" },
  { id: "select",  label: "" },
  { id: "segment", label: "" },
  { id: "refine",  label: "" },
  { id: "mesh",    label: "" },
  { id: "export",  label: "" },
] as const;

const ContextOptionsPanel: React.FC<ContextOptionsPanelProps> = ({
  children,
  stepTitle,
  stepIcon,
}) => {
  const isOpen = useUiStore((s) => s.isContextPanelOpen);
  const toggleOpen = useUiStore((s) => s.toggleContextPanel);
  const activeStep = useUiStore((s) => s.activeStep);
  const completedSteps = useUiStore((s) => s.completedSteps);
  const setActiveStep = useUiStore((s) => s.setActiveStep);

  const handleMiniClick = useCallback(
    (step: WorkflowStep) => () => setActiveStep(step),
    [setActiveStep],
  );

  return (
    <div
      className={cn(
        "flex flex-col h-full tech-glass border-r border-border/50 tech-transition shrink-0 overflow-hidden",
        isOpen ? "w-[320px]" : "w-12",
      )}
    >
      {/* Collapse toggle */}
      <button
        type="button"
        aria-label={isOpen ? "Collapse context panel" : "Expand context panel"}
        onClick={toggleOpen}
        className="flex items-center justify-center h-8 w-full hover:bg-primary/10 tech-transition border-b border-border/30 shrink-0"
      >
        {isOpen ? (
          <ChevronLeft size={14} className="text-muted-foreground" />
        ) : (
          <ChevronRight size={14} className="text-muted-foreground" />
        )}
      </button>

      {isOpen && (
        <>
          {/* Step header */}
          <div className="flex items-center gap-3 px-4 py-3 border-b border-border/30 shrink-0">
            <div className="text-primary text-lg shrink-0">{stepIcon}</div>
            <div>
              <p className="font-tech text-xs text-primary tracking-widest uppercase">{stepTitle}</p>
            </div>
          </div>

          {/* Step content — scrollable */}
          <div className="flex-1 overflow-y-auto overflow-x-hidden min-h-0">
            {children}
          </div>

          {/* Mini-map step row */}
          <div className="flex items-center justify-center gap-0.5 px-2 py-2 border-t border-border/30 shrink-0">
            {STEP_MINI_ICONS.map(({ id, label }) => (
              <button
                key={id}
                type="button"
                aria-label={`Go to ${id} step`}
                title={id}
                onClick={handleMiniClick(id)}
                className={cn(
                  "flex items-center justify-center w-8 h-8 rounded text-sm tech-transition",
                  "hover:bg-primary/10",
                  activeStep === id && "bg-primary/20",
                  completedSteps.has(id) && activeStep !== id && "opacity-50",
                )}
              >
                {label}
              </button>
            ))}
          </div>
        </>
      )}
    </div>
  );
};

export default ContextOptionsPanel;
