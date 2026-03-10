// src/components/layout/PropertiesPanel.tsx
// Collapsible 280px right panel with accordion sections for study/masks/roi/mesh.
import React from "react";
import { ChevronRight, ChevronLeft } from "lucide-react";
import { cn } from "@/lib/utils";
import { useUiStore } from "@/store/uiStore";
import { useUploadStore } from "@/store/uploadStore";
import { useScoutStore } from "@/store/scoutStore";
import { useMeshStore } from "@/store/meshStore";

const PropertiesPanel: React.FC = () => {
  const isOpen = useUiStore((s) => s.isPropertiesPanelOpen);
  const toggleOpen = useUiStore((s) => s.togglePropertiesPanel);
  const study = useUploadStore((s) => s.study);
  const roi = useScoutStore((s) => s.roi);
  const meshResult = useMeshStore((s) => s.meshResult);

  return (
    <div
      className={cn(
        "flex flex-col h-full tech-glass border-l border-border/50 tech-transition shrink-0 overflow-hidden",
        isOpen ? "w-[280px]" : "w-12",
      )}
    >
      {/* Collapse toggle */}
      <button
        type="button"
        aria-label={isOpen ? "Collapse properties panel" : "Expand properties panel"}
        onClick={toggleOpen}
        className="flex items-center justify-center h-8 w-full hover:bg-primary/10 tech-transition border-b border-border/30 shrink-0"
      >
        {isOpen ? (
          <ChevronRight size={14} className="text-muted-foreground" />
        ) : (
          <ChevronLeft size={14} className="text-muted-foreground" />
        )}
      </button>

      {isOpen && (
        <div className="flex-1 overflow-y-auto min-h-0 p-3 space-y-3">
          {/* Study section */}
          <section className="space-y-1.5">
            <h3 className="font-tech text-[10px] text-muted-foreground uppercase tracking-widest">Study</h3>
            {study !== null ? (
              <div className="space-y-1 text-[11px] font-mono text-foreground/80">
                <PropRow label="ID" value={study.studyId.slice(0, 12)} />
                <PropRow label="Modality" value={study.modality} />
                <PropRow
                  label="Spacing"
                  value={study.voxelSpacing.map((v) => v.toFixed(2)).join("  ")}
                />
                <PropRow
                  label="Dims"
                  value={study.dimensions.join("  ")}
                />
              </div>
            ) : (
              <p className="text-[11px] text-muted-foreground/50 font-mono">No study loaded</p>
            )}
          </section>

          {/* ROI section */}
          {roi !== null && (
            <section className="space-y-1.5 border-t border-border/30 pt-3">
              <h3 className="font-tech text-[10px] text-muted-foreground uppercase tracking-widest">ROI (mm)</h3>
              <div className="space-y-1 text-[11px] font-mono text-foreground/80">
                <PropRow label="X" value={`${roi.xMin.toFixed(0)} – ${roi.xMax.toFixed(0)}`} />
                <PropRow label="Y" value={`${roi.yMin.toFixed(0)} – ${roi.yMax.toFixed(0)}`} />
                <PropRow label="Z" value={`${roi.zMin.toFixed(0)} – ${roi.zMax.toFixed(0)}`} />
              </div>
            </section>
          )}

          {/* Mesh section */}
          {meshResult !== null && (
            <section className="space-y-1.5 border-t border-border/30 pt-3">
              <h3 className="font-tech text-[10px] text-muted-foreground uppercase tracking-widest">Mesh</h3>
              <div className="space-y-1 text-[11px] font-mono text-foreground/80">
                <PropRow label="Faces" value={meshResult.faceCount.toLocaleString()} />
                <PropRow
                  label="Watertight"
                  value={meshResult.isWatertight ? " Yes" : " No"}
                />
              </div>
            </section>
          )}
        </div>
      )}
    </div>
  );
};

interface PropRowProps {
  readonly label: string;
  readonly value: string;
}

const PropRow: React.FC<PropRowProps> = ({ label, value }) => (
  <div className="flex justify-between items-baseline">
    <span className="text-muted-foreground/70">{label}</span>
    <span className="text-foreground/80 tabular-nums">{value}</span>
  </div>
);

export default PropertiesPanel;
