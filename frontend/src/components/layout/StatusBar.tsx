// src/components/layout/StatusBar.tsx
import React from "react";
import { cn } from "@/lib/utils";
import { useUiStore } from "@/store/uiStore";
import { useSegmentStore } from "@/store/segmentStore";
import { useMeshStore } from "@/store/meshStore";

interface StatusBarProps {
  readonly className?: string;
}

const StatusBar: React.FC<StatusBarProps> = ({ className }) => {
  const viewportMode = useUiStore((s) => s.viewportMode);
  const segStatus = useSegmentStore((s) => s.taskStatus);
  const segPercent = useSegmentStore((s) => s.progressPercent);
  const segStep = useSegmentStore((s) => s.progressStep);
  const meshStatus = useMeshStore((s) => s.taskStatus);
  const meshPercent = useMeshStore((s) => s.progressPercent);

  const isRunning = segStatus === "running" || meshStatus === "running";
  const percent = segStatus === "running" ? segPercent : meshPercent;
  const step = segStatus === "running" ? segStep : "mesh generation";

  return (
    <footer
      className={cn(
        "flex h-7 items-center justify-between px-4 tech-glass border-t border-border/50 z-50",
        className,
      )}
    >
      {/* Left: status message */}
      <div className="flex items-center gap-3 text-[11px] font-mono text-muted-foreground">
        <span
          className={cn(
            "w-1.5 h-1.5 rounded-full",
            isRunning ? "bg-primary animate-pulse" : "bg-muted-foreground/40",
          )}
          aria-hidden
        />
        {isRunning ? (
          <span className="text-foreground/70">{step}</span>
        ) : (
          <span>Ready</span>
        )}
      </div>

      {/* Center: progress bar (only shown when running) */}
      {isRunning && (
        <div
          className="flex-1 mx-4 max-w-xs h-1 bg-muted rounded-full overflow-hidden"
          role="progressbar"
          aria-valuenow={percent}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-label="Task progress"
        >
          <div
            className="h-full bg-primary tech-transition rounded-full"
            style={{ width: `${percent}%` }}
          />
        </div>
      )}

      {/* Right: renderer info */}
      <div className="flex items-center gap-3 text-[11px] font-mono text-muted-foreground">
        <span>WebGL 2.0</span>
        <span className="opacity-30"></span>
        <span>{viewportMode.toUpperCase()}</span>
      </div>
    </footer>
  );
};

export default StatusBar;
