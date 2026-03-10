// src/components/sidebar/AISegPanel.tsx
import React, { useCallback } from "react";
import { Bot, Loader2, AlertTriangle, Eye, EyeOff } from "lucide-react";
import { cn } from "@/lib/utils";
import { useSegmentStore } from "@/store/segmentStore";
import { useScoutStore } from "@/store/scoutStore";
import { useUploadStore } from "@/store/uploadStore";
import { useUiStore } from "@/store/uiStore";
import { triggerSegment, fetchSegLabels, subscribeToJob } from "@/services/api";
import { logger } from "@/lib/logger";
import type { SegLabel } from "@/types";

//  LabelRow (extracted sub-component) 

interface LabelRowProps {
  readonly label: SegLabel;
  readonly onToggleVisibility: (id: number) => void;
}

const LabelRow: React.FC<LabelRowProps> = React.memo(({ label, onToggleVisibility }) => {
  const handleToggle = useCallback(() => onToggleVisibility(label.labelId), [label.labelId, onToggleVisibility]);

  return (
    <div className="flex items-center gap-2.5 px-3 py-1.5 rounded hover:bg-muted/20 tech-transition">
      <span
        className="w-2.5 h-2.5 rounded-full shrink-0"
        style={{ backgroundColor: label.color }}
        aria-hidden
      />
      <span className="flex-1 text-[11px] font-mono text-foreground/80 truncate">{label.name}</span>
      <button
        type="button"
        aria-label={label.isVisible ? `Hide ${label.name}` : `Show ${label.name}`}
        onClick={handleToggle}
        className="shrink-0 hover:text-primary tech-transition"
      >
        {label.isVisible ? (
          <Eye size={12} className="text-primary/70" />
        ) : (
          <EyeOff size={12} className="text-muted-foreground/40" />
        )}
      </button>
    </div>
  );
});
LabelRow.displayName = "LabelRow";

//  AISegPanel 

const AISegPanel: React.FC = () => {
  const study = useUploadStore((s) => s.study);
  const roi = useScoutStore((s) => s.roi);
  const selectedIslandIds = useScoutStore((s) => s.selectedIslandIds);

  const labels = useSegmentStore((s) => s.labels);
  const taskStatus = useSegmentStore((s) => s.taskStatus);
  const progressPercent = useSegmentStore((s) => s.progressPercent);
  const progressStep = useSegmentStore((s) => s.progressStep);
  const errorMessage = useSegmentStore((s) => s.errorMessage);
  const taskId = useSegmentStore((s) => s.taskId);

  const setLabels = useSegmentStore((s) => s.setLabels);
  const setTaskId = useSegmentStore((s) => s.setTaskId);
  const setProgress = useSegmentStore((s) => s.setProgress);
  const setError = useSegmentStore((s) => s.setError);
  const toggleLabelVisibility = useSegmentStore((s) => s.toggleLabelVisibility);
  const markComplete = useUiStore((s) => s.markStepComplete);
  const setActiveStep = useUiStore((s) => s.setActiveStep);

  const isRunning = taskStatus === "running";
  const isDone = taskStatus === "done";
  const canRun = study !== null && roi !== null && !isRunning;

  const handleRunSegment = useCallback(async () => {
    if (!canRun || study === null || roi === null) return;

    const hintLabels = Array.from(selectedIslandIds);

    try {
      const { taskId: id } = await triggerSegment(study.studyId, roi, hintLabels);
      setTaskId(id);
      logger.info("Segment task started", id);

      await new Promise<void>((resolve, reject) => {
        const unsub = subscribeToJob(
          id,
          (status) => {
            if (status.state === "PROGRESS") {
              setProgress(status.percent ?? 0, status.step ?? "segmenting");
            } else if (status.state === "SUCCESS") {
              unsub();
              resolve();
            } else if (status.state === "FAILURE") {
              unsub();
              reject(new Error(status.error ?? "Segmentation failed"));
            }
          },
          (err) => {
            logger.error("Segment SSE error", err);
            reject(new Error("Connection lost"));
          },
        );
      });

      const data = await fetchSegLabels(study.studyId);
      setLabels(data);
      markComplete("segment");
      logger.info("Segment complete", data.length, "labels");
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Segmentation failed";
      setError(msg);
      logger.error("Segment error", err);
    }
  }, [canRun, study, roi, selectedIslandIds, setTaskId, setProgress, setLabels, setError, markComplete]);

  return (
    <div className="p-4 space-y-4">
      {roi === null && (
        <div className="flex items-center gap-2 p-3 rounded bg-accent/10 border border-accent/30">
          <AlertTriangle size={13} className="text-accent shrink-0" />
          <p className="text-[11px] text-muted-foreground/80 font-mono">Confirm ROI in Select step first.</p>
        </div>
      )}

      {/* Run button */}
      <button
        type="button"
        disabled={!canRun}
        onClick={() => void handleRunSegment()}
        className={cn(
          "w-full flex items-center justify-center gap-2 py-2 rounded font-tech text-[10px] uppercase tracking-wider tech-transition",
          canRun
            ? "bg-primary/20 hover:bg-primary/30 text-primary"
            : "bg-muted/20 text-muted-foreground/40 cursor-not-allowed",
        )}
      >
        {isRunning ? (
          <Loader2 size={13} className="animate-spin" aria-hidden />
        ) : (
          <Bot size={13} aria-hidden />
        )}
        {isRunning ? "Segmenting" : "Run TotalSegmentator"}
      </button>

      {/* Progress */}
      {isRunning && (
        <div className="space-y-1.5">
          <div className="flex justify-between text-[10px] font-mono text-muted-foreground">
            <span>{progressStep}</span>
            <span>{progressPercent}%</span>
          </div>
          <div
            className="h-1 bg-muted/30 rounded-full overflow-hidden"
            role="progressbar"
            aria-valuenow={progressPercent}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-label="Segmentation progress"
          >
            <div className="h-full bg-primary tech-transition" style={{ width: `${progressPercent}%` }} />
          </div>
        </div>
      )}

      {/* Error */}
      {taskStatus === "error" && errorMessage !== null && (
        <div className="flex items-start gap-2 p-3 rounded bg-destructive/10 border border-destructive/30">
          <AlertTriangle size={13} className="text-destructive shrink-0 mt-0.5" />
          <div>
            <p className="text-[11px] text-destructive font-mono">{errorMessage}</p>
            <button type="button" onClick={() => void handleRunSegment()} className="text-[10px] text-primary hover:underline mt-1">Retry</button>
          </div>
        </div>
      )}

      {/* Label list */}
      {labels.length > 0 && (
        <>
          <p className="font-tech text-[10px] text-muted-foreground uppercase tracking-widest">
            {labels.length} Structures
          </p>
          <div className="space-y-0.5 max-h-64 overflow-y-auto">
            {labels.map((label) => (
              <LabelRow key={label.labelId} label={label} onToggleVisibility={toggleLabelVisibility} />
            ))}
          </div>
          <button
            type="button"
            onClick={() => { markComplete("segment"); setActiveStep("refine"); }}
            className="w-full py-1.5 rounded bg-primary/20 hover:bg-primary/30 text-primary font-tech text-[10px] uppercase tracking-wider tech-transition"
          >
            Proceed to Refine 
          </button>
        </>
      )}
    </div>
  );
};

export default AISegPanel;
