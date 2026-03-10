// src/components/sidebar/MeshPanel.tsx
import React, { useCallback } from "react";
import { Box, CheckCircle2, Loader2, AlertTriangle, XCircle } from "lucide-react";
import { cn } from "@/lib/utils";
import { useMeshStore } from "@/store/meshStore";
import { useSegmentStore } from "@/store/segmentStore";
import { useScoutStore } from "@/store/scoutStore";
import { useUploadStore } from "@/store/uploadStore";
import { useUiStore } from "@/store/uiStore";
import { triggerMesh, fetchMeshResult, subscribeToJob } from "@/services/api";
import { logger } from "@/lib/logger";
import type { MeshQuality } from "@/types";

const QUALITY_OPTIONS: ReadonlyArray<{ id: MeshQuality; label: string; desc: string }> = [
  { id: "preview",  label: "Preview",     desc: "~30k faces, fast" },
  { id: "standard", label: "Standard",    desc: "~150k faces, print quality" },
  { id: "high",     label: "High Detail", desc: "~500k faces, surgical precision" },
] as const;

const MeshPanel: React.FC = () => {
  const study = useUploadStore((s) => s.study);
  const roi = useScoutStore((s) => s.roi);
  const labels = useSegmentStore((s) => s.labels);

  const meshResult = useMeshStore((s) => s.meshResult);
  const quality = useMeshStore((s) => s.quality);
  const taskStatus = useMeshStore((s) => s.taskStatus);
  const progressPercent = useMeshStore((s) => s.progressPercent);
  const errorMessage = useMeshStore((s) => s.errorMessage);

  const setQuality = useMeshStore((s) => s.setQuality);
  const setTaskId = useMeshStore((s) => s.setTaskId);
  const setProgress = useMeshStore((s) => s.setProgress);
  const setMeshResult = useMeshStore((s) => s.setMeshResult);
  const setError = useMeshStore((s) => s.setError);
  const markComplete = useUiStore((s) => s.markStepComplete);
  const setActiveStep = useUiStore((s) => s.setActiveStep);

  const isRunning = taskStatus === "running";
  const visibleLabelIds = labels.filter((l) => l.isVisible).map((l) => l.labelId);
  const canRun = study !== null && roi !== null && visibleLabelIds.length > 0 && !isRunning;

  const handleGenerateMesh = useCallback(async () => {
    if (!canRun || study === null || roi === null) return;

    try {
      const { taskId } = await triggerMesh(study.studyId, visibleLabelIds, quality, roi);
      setTaskId(taskId);
      logger.info("Mesh task started", taskId);

      await new Promise<void>((resolve, reject) => {
        const unsub = subscribeToJob(
          taskId,
          (status) => {
            if (status.state === "PROGRESS") setProgress(status.percent ?? 0);
            else if (status.state === "SUCCESS") { unsub(); resolve(); }
            else if (status.state === "FAILURE") { unsub(); reject(new Error(status.error ?? "Mesh failed")); }
          },
          (err) => { logger.error("Mesh SSE", err); reject(new Error("Connection lost")); },
        );
      });

      const result = await fetchMeshResult(study.studyId);
      setMeshResult(result);
      markComplete("mesh");
      logger.info("Mesh complete", result.faceCount, "faces", result.isWatertight ? "watertight" : "NOT watertight");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Mesh generation failed");
      logger.error("Mesh error", err);
    }
  }, [canRun, study, roi, visibleLabelIds, quality, setTaskId, setProgress, setMeshResult, setError, markComplete]);

  return (
    <div className="p-4 space-y-4">
      {/* Quality selector */}
      <div className="space-y-1.5">
        <p className="font-tech text-[10px] text-muted-foreground uppercase tracking-widest">Quality</p>
        <div className="space-y-1">
          {QUALITY_OPTIONS.map((opt) => (
            <button
              key={opt.id}
              type="button"
              aria-pressed={quality === opt.id}
              onClick={() => setQuality(opt.id)}
              disabled={isRunning}
              className={cn(
                "w-full flex items-center justify-between px-3 py-2 rounded border tech-transition",
                quality === opt.id ? "border-primary/50 bg-primary/10" : "border-border/30 hover:border-primary/30",
                isRunning && "opacity-40 cursor-not-allowed",
              )}
            >
              <span className={cn("font-tech text-[10px] uppercase tracking-wider", quality === opt.id ? "text-primary" : "text-foreground/70")}>{opt.label}</span>
              <span className="text-[10px] text-muted-foreground/60 font-mono">{opt.desc}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Generate button */}
      <button
        type="button"
        disabled={!canRun}
        onClick={() => void handleGenerateMesh()}
        className={cn(
          "w-full flex items-center justify-center gap-2 py-2 rounded font-tech text-[10px] uppercase tracking-wider tech-transition",
          canRun ? "bg-primary/20 hover:bg-primary/30 text-primary" : "bg-muted/20 text-muted-foreground/40 cursor-not-allowed",
        )}
      >
        {isRunning ? <Loader2 size={13} className="animate-spin" aria-hidden /> : <Box size={13} aria-hidden />}
        {isRunning ? "Generating" : "Generate Mesh"}
      </button>

      {/* Progress */}
      {isRunning && (
        <div className="h-1 bg-muted/30 rounded-full overflow-hidden" role="progressbar" aria-valuenow={progressPercent} aria-valuemin={0} aria-valuemax={100} aria-label="Mesh generation progress">
          <div className="h-full bg-primary tech-transition" style={{ width: `${progressPercent}%` }} />
        </div>
      )}

      {/* Error */}
      {taskStatus === "error" && errorMessage !== null && (
        <div className="flex items-start gap-2 p-3 rounded bg-destructive/10 border border-destructive/30">
          <AlertTriangle size={13} className="text-destructive shrink-0 mt-0.5" />
          <div>
            <p className="text-[11px] text-destructive font-mono">{errorMessage}</p>
            <button type="button" onClick={() => void handleGenerateMesh()} className="text-[10px] text-primary hover:underline mt-1">Retry</button>
          </div>
        </div>
      )}

      {/* Mesh result */}
      {meshResult !== null && (
        <div className="p-3 rounded bg-muted/20 space-y-2">
          <div className="flex items-center gap-2">
            {meshResult.isWatertight ? (
              <CheckCircle2 size={14} className="text-green-400" aria-hidden />
            ) : (
              <XCircle size={14} className="text-destructive" aria-hidden />
            )}
            <span className={cn("font-tech text-[10px] uppercase tracking-wider", meshResult.isWatertight ? "text-green-400" : "text-destructive")}>
              {meshResult.isWatertight ? "Watertight " : "Not Watertight"}
            </span>
          </div>
          <p className="text-[11px] font-mono text-muted-foreground">{meshResult.faceCount.toLocaleString()} faces</p>
          {meshResult.isWatertight && (
            <button
              type="button"
              onClick={() => { markComplete("mesh"); setActiveStep("export"); }}
              className="w-full py-1.5 rounded bg-primary/20 hover:bg-primary/30 text-primary font-tech text-[10px] uppercase tracking-wider tech-transition"
            >
              Proceed to Export 
            </button>
          )}
        </div>
      )}
    </div>
  );
};

export default MeshPanel;
