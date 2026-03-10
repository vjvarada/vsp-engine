// src/components/sidebar/ExportPanel.tsx
import React, { useCallback } from "react";
import { Download, Loader2, AlertTriangle } from "lucide-react";
import { cn } from "@/lib/utils";
import { useExportStore } from "@/store/exportStore";
import { useSegmentStore } from "@/store/segmentStore";
import { useMeshStore } from "@/store/meshStore";
import { useUploadStore } from "@/store/uploadStore";
import { triggerExport } from "@/services/api";
import { logger } from "@/lib/logger";
import type { ExportFormat } from "@/types";

const FORMAT_OPTIONS: ReadonlyArray<{ id: ExportFormat; label: string }> = [
  { id: "stl", label: "STL" },
  { id: "obj", label: "OBJ" },
  { id: "3mf", label: "3MF" },
] as const;

const SCALE_OPTIONS = [
  { value: 1,   label: "1:1 (actual size)" },
  { value: 2,   label: "2 (double)" },
  { value: 0.5, label: "0.5 (half)" },
] as const;

const ExportPanel: React.FC = () => {
  const study = useUploadStore((s) => s.study);
  const meshResult = useMeshStore((s) => s.meshResult);
  const segLabels = useSegmentStore((s) => s.labels);

  const labels = useExportStore((s) => s.labels);
  const format = useExportStore((s) => s.format);
  const scaleFactor = useExportStore((s) => s.scaleFactor);
  const downloadUrl = useExportStore((s) => s.downloadUrl);
  const taskStatus = useExportStore((s) => s.taskStatus);
  const errorMessage = useExportStore((s) => s.errorMessage);

  const setLabels = useExportStore((s) => s.setLabels);
  const toggleLabel = useExportStore((s) => s.toggleLabel);
  const setLabelUnion = useExportStore((s) => s.setLabelUnion);
  const setFormat = useExportStore((s) => s.setFormat);
  const setScaleFactor = useExportStore((s) => s.setScaleFactor);
  const setDownloadUrl = useExportStore((s) => s.setDownloadUrl);
  const setTaskStatus = useExportStore((s) => s.setTaskStatus);
  const setError = useExportStore((s) => s.setError);

  // Initialize export labels from seg labels when they change
  React.useEffect(() => {
    if (segLabels.length > 0 && labels.length === 0) {
      setLabels(
        segLabels.map((l) => ({
          labelId: l.labelId,
          name: l.name,
          included: l.isVisible,
          union: "combined" as const,
        })),
      );
    }
  }, [segLabels, labels.length, setLabels]);

  const isRunning = taskStatus === "running";
  const canExport = study !== null && meshResult?.isWatertight === true && !isRunning;

  const handleExport = useCallback(async () => {
    if (!canExport || study === null) return;
    setTaskStatus("running");

    try {
      const { downloadUrl: url } = await triggerExport(study.studyId, labels, format, scaleFactor);
      setDownloadUrl(url);
      logger.info("Export ready", url);

      // Auto-trigger download
      const a = document.createElement("a");
      a.href = url;
      a.download = `vsp_export_${study.studyId.slice(0, 8)}.${format}`;
      a.click();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Export failed");
      logger.error("Export error", err);
    }
  }, [canExport, study, labels, format, scaleFactor, setTaskStatus, setDownloadUrl, setError]);

  const includedCount = labels.filter((l) => l.included).length;

  return (
    <div className="p-4 space-y-4">
      {meshResult === null && (
        <div className="flex items-center gap-2 p-3 rounded bg-accent/10 border border-accent/30">
          <AlertTriangle size={13} className="text-accent shrink-0" />
          <p className="text-[11px] text-muted-foreground/80 font-mono">Generate a watertight mesh first.</p>
        </div>
      )}

      {/* Structure selection */}
      {labels.length > 0 && (
        <div className="space-y-1.5">
          <p className="font-tech text-[10px] text-muted-foreground uppercase tracking-widest">Structures ({includedCount}/{labels.length})</p>
          <div className="space-y-0.5 max-h-40 overflow-y-auto">
            {labels.map((label) => (
              <div key={label.labelId} className="flex items-center gap-2 px-2 py-1.5 rounded hover:bg-muted/20 tech-transition">
                <input
                  type="checkbox"
                  id={`export-label-${label.labelId}`}
                  checked={label.included}
                  onChange={() => toggleLabel(label.labelId)}
                  aria-label={`Include ${label.name} in export`}
                  className="accent-primary"
                />
                <label htmlFor={`export-label-${label.labelId}`} className="flex-1 text-[11px] font-mono text-foreground/80 cursor-pointer">{label.name}</label>
                <div className="flex gap-1.5">
                  {(["combined", "separate"] as const).map((opt) => (
                    <button
                      key={opt}
                      type="button"
                      aria-pressed={label.union === opt}
                      disabled={!label.included}
                      onClick={() => setLabelUnion(label.labelId, opt)}
                      className={cn(
                        "text-[9px] font-tech uppercase tracking-wide px-1.5 py-0.5 rounded tech-transition",
                        label.union === opt && label.included ? "bg-primary/20 text-primary" : "text-muted-foreground/40",
                        !label.included && "opacity-30 cursor-not-allowed",
                      )}
                    >
                      {opt === "combined" ? "" : ""}
                    </button>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Format selector */}
      <div className="space-y-1.5">
        <p className="font-tech text-[10px] text-muted-foreground uppercase tracking-widest">Format</p>
        <div className="flex gap-2">
          {FORMAT_OPTIONS.map((opt) => (
            <button
              key={opt.id}
              type="button"
              aria-pressed={format === opt.id}
              onClick={() => setFormat(opt.id)}
              className={cn(
                "flex-1 py-1.5 rounded font-tech text-[10px] uppercase tracking-wider border tech-transition",
                format === opt.id ? "border-primary/50 bg-primary/15 text-primary" : "border-border/30 text-muted-foreground/60 hover:border-primary/30",
              )}
            >
              {opt.label}
            </button>
          ))}
        </div>
      </div>

      {/* Scale selector */}
      <div className="space-y-1.5">
        <p className="font-tech text-[10px] text-muted-foreground uppercase tracking-widest">Scale</p>
        <div className="space-y-1">
          {SCALE_OPTIONS.map((opt) => (
            <button
              key={opt.value}
              type="button"
              aria-pressed={scaleFactor === opt.value}
              onClick={() => setScaleFactor(opt.value)}
              className={cn(
                "w-full flex items-center justify-between px-3 py-1.5 rounded border text-left tech-transition",
                scaleFactor === opt.value ? "border-primary/50 bg-primary/10" : "border-border/20 hover:border-primary/30",
              )}
            >
              <span className={cn("font-tech text-[10px] uppercase tracking-wide", scaleFactor === opt.value ? "text-primary" : "text-foreground/60")}>{opt.label}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Error */}
      {taskStatus === "error" && errorMessage !== null && (
        <div className="flex items-start gap-2 p-3 rounded bg-destructive/10 border border-destructive/30">
          <AlertTriangle size={13} className="text-destructive shrink-0 mt-0.5" />
          <p className="text-[11px] text-destructive font-mono">{errorMessage}</p>
        </div>
      )}

      {/* Export button (only if watertight) */}
      <button
        type="button"
        disabled={!canExport || includedCount === 0}
        onClick={() => void handleExport()}
        className={cn(
          "w-full flex items-center justify-center gap-2 py-2 rounded font-tech text-[10px] uppercase tracking-wider tech-transition",
          canExport && includedCount > 0
            ? "bg-primary/20 hover:bg-primary/30 text-primary"
            : "bg-muted/20 text-muted-foreground/40 cursor-not-allowed",
        )}
      >
        {isRunning ? <Loader2 size={13} className="animate-spin" aria-hidden /> : <Download size={13} aria-hidden />}
        {isRunning ? "Exporting" : `Export ${includedCount} Structure${includedCount !== 1 ? "s" : ""}`}
      </button>

      {downloadUrl !== null && !isRunning && (
        <div className="p-3 rounded bg-green-500/10 border border-green-500/30">
          <p className="text-[10px] font-tech text-green-400 uppercase tracking-wider">Export ready!</p>
          <p className="text-[9px] text-muted-foreground/60 font-mono mt-1">FOR PLANNING PURPOSES ONLY</p>
        </div>
      )}
    </div>
  );
};

export default ExportPanel;
