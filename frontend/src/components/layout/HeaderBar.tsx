// src/components/layout/HeaderBar.tsx
import React from "react";
import { cn } from "@/lib/utils";
import { useUploadStore } from "@/store/uploadStore";

interface HeaderBarProps {
  readonly className?: string;
}

const HeaderBar: React.FC<HeaderBarProps> = ({ className }) => {
  const study = useUploadStore((s) => s.study);
  const studyLabel = study !== null ? `Study ${study.studyId.slice(0, 8)}` : "No study loaded";

  return (
    <header
      className={cn(
        "flex h-10 items-center justify-between px-4 tech-glass border-b border-border/50 z-50",
        className,
      )}
    >
      {/* Left: brand */}
      <div className="flex items-center gap-2">
        <span className="font-tech text-primary text-xs tracking-widest uppercase select-none">
          VSP Engine
        </span>
        <span className="text-muted-foreground text-xs opacity-50">|</span>
        <span className="text-muted-foreground text-xs font-mono">{studyLabel}</span>
      </div>

      {/* Center: disclaimer */}
      <div className="absolute left-1/2 -translate-x-1/2">
        <span className="text-accent font-tech text-[10px] tracking-wider uppercase opacity-70">
          For Planning Purposes Only — Not For Diagnostic Use
        </span>
      </div>

      {/* Right: version badge */}
      <div className="flex items-center gap-2">
        <span className="text-muted-foreground text-[10px] font-mono opacity-50">v0.1.0</span>
      </div>
    </header>
  );
};

export default HeaderBar;
