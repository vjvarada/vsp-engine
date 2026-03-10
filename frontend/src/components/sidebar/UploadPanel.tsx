// src/components/sidebar/UploadPanel.tsx
import React, { useCallback, useRef } from "react";
import { Upload, AlertTriangle, CheckCircle2, Loader2, FolderOpen } from "lucide-react";
import { cn } from "@/lib/utils";
import { useUploadStore } from "@/store/uploadStore";
import { useUiStore } from "@/store/uiStore";
import { finalizeUpload, createSession } from "@/services/api";
import { logger } from "@/lib/logger";
import { useDicomConvert } from "@/hooks/useDicomConvert";
import type { Study } from "@/types";

const ACCEPTED_TYPES = ".dcm,.nii,.nii.gz,.nrrd,.zip";
const LOCAL_ACCEPTED_TYPES = ".nii,.nii.gz,.nrrd,.dcm,.zip";

const UploadPanel: React.FC = () => {
  const study = useUploadStore((s) => s.study);
  const taskStatus = useUploadStore((s) => s.taskStatus);
  const errorMessage = useUploadStore((s) => s.errorMessage);
  const uploadProgress = useUploadStore((s) => s.uploadProgress);
  const qcWarnings = useUploadStore((s) => s.qcWarnings);
  const hasDismissedWarnings = useUploadStore((s) => s.hasDismissedWarnings);

  const setTaskStatus = useUploadStore((s) => s.setTaskStatus);
  const setError = useUploadStore((s) => s.setError);
  const setUploadProgress = useUploadStore((s) => s.setUploadProgress);
  const setStudy = useUploadStore((s) => s.setStudy);
  const setLocalFileName = useUploadStore((s) => s.setLocalFileName);
  const setQcWarnings = useUploadStore((s) => s.setQcWarnings);
  const dismissWarnings = useUploadStore((s) => s.dismissWarnings);
  const markComplete = useUiStore((s) => s.markStepComplete);
  const setActiveStep = useUiStore((s) => s.setActiveStep);

  const { convertStatus, convertProgress, convertError, convertZip } = useDicomConvert();

  const fileInputRef = useRef<HTMLInputElement>(null);
  const localFileInputRef = useRef<HTMLInputElement>(null);
  const isDraggingRef = useRef(false);
  const [isDragOver, setIsDragOver] = React.useState(false);

  const handleFiles = useCallback(
    async (files: FileList) => {
      if (files.length === 0) return;
      const file = files[0];
      if (file === undefined) return;

      setTaskStatus("running");
      setUploadProgress(0);

      try {
        // Ensure session exists
        await createSession();
        setUploadProgress(20);

        // Upload the file to the backend
        const result = await finalizeUpload(file);
        setUploadProgress(100);
        setStudy(result);
        setQcWarnings(result.qcWarnings);
        markComplete("upload");
        setActiveStep("scout");
        logger.info("Upload complete", result.studyId);
      } catch (err) {
        const msg = err instanceof Error ? err.message : "Upload failed";
        setError(msg);
        logger.error("Upload failed", err);
      }
    },
    [setTaskStatus, setUploadProgress, setStudy, setQcWarnings, setError, markComplete, setActiveStep],
  );

  /** Load a NIfTI/NRRD directly as a blob URL (no backend, no conversion). */
  const handleNiftiFile = useCallback(
    (file: File) => {
      const prevUrl = useUploadStore.getState().study?.niftiUrl;
      if (prevUrl?.startsWith("blob:")) URL.revokeObjectURL(prevUrl);

      const blobUrl = URL.createObjectURL(file);
      const mockStudy: Study = {
        studyId: crypto.randomUUID(),
        niftiUrl: blobUrl,
        voxelSpacing: [1, 1, 1],
        dimensions: [0, 0, 0],
        modality: "unknown",
        qcWarnings: [],
      };
      setLocalFileName(file.name);
      setStudy(mockStudy);
      markComplete("upload");
      setActiveStep("scout");
      logger.info("Local NIfTI loaded (no backend)", file.name);
    },
    [setStudy, setLocalFileName, markComplete, setActiveStep],
  );

  /** Route ZIP → dcm2niix WASM, NIfTI/NRRD → direct blob URL. */
  const handleLocalFile = useCallback(
    async (file: File) => {
      if (file.name.toLowerCase().endsWith(".zip")) {
        const result = await convertZip(file);
        if (result === null) return; // error state already set in hook
        const mockStudy: Study = {
          studyId: crypto.randomUUID(),
          niftiUrl: result.blobUrl,
          voxelSpacing: [1, 1, 1],
          dimensions: [0, 0, 0],
          modality: "CT",
          qcWarnings:
            result.seriesCount > 1
              ? [`${result.seriesCount} series found — largest volume selected`]
              : [],
        };
        setLocalFileName(result.fileName);
        setStudy(mockStudy);
        markComplete("upload");
        setActiveStep("scout");
      } else {
        handleNiftiFile(file);
      }
    },
    [convertZip, handleNiftiFile, setStudy, setLocalFileName, markComplete, setActiveStep],
  );

  const handleLocalInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file !== undefined) void handleLocalFile(file);
    },
    [handleLocalFile],
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragOver(false);
      void handleFiles(e.dataTransfer.files);
    },
    [handleFiles],
  );

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback(() => setIsDragOver(false), []);

  const handleInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      if (e.target.files !== null) void handleFiles(e.target.files);
    },
    [handleFiles],
  );

  const isRunning = taskStatus === "running";
  const isDone = taskStatus === "done";

  return (
    <div className="p-4 space-y-4">
      {/* Drop zone */}
      <div
        role="button"
        tabIndex={0}
        aria-label="Drop DICOM files here or click to browse"
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onClick={() => !isRunning && fileInputRef.current?.click()}
        onKeyDown={(e) => e.key === "Enter" && fileInputRef.current?.click()}
        className={cn(
          "flex flex-col items-center justify-center gap-3 p-6 rounded-lg border-2 border-dashed tech-transition cursor-pointer",
          isDragOver ? "border-primary bg-primary/10" : "border-border hover:border-primary/50 hover:bg-primary/5",
          isRunning && "pointer-events-none opacity-60",
        )}
      >
        {isRunning ? (
          <Loader2 size={28} className="text-primary animate-spin" />
        ) : isDone ? (
          <CheckCircle2 size={28} className="text-green-400" />
        ) : (
          <Upload size={28} className="text-muted-foreground" />
        )}
        <div className="text-center space-y-1">
          <p className="text-xs font-tech text-foreground/80">
            {isDone ? "Study loaded" : "Drop DICOM / NIfTI / ZIP"}
          </p>
          <p className="text-[10px] text-muted-foreground/60">
            .dcm  .nii.gz  .nrrd  .zip
          </p>
        </div>
      </div>

      <input
        ref={fileInputRef}
        type="file"
        accept={ACCEPTED_TYPES}
        multiple
        onChange={handleInputChange}
        className="hidden"
        aria-label="File upload input"
      />

      {/* ── Local preview (no backend) ───────────────────────────── */}
      <div className="flex items-center gap-2 py-1">
        <div className="flex-1 h-px bg-border" />
        <span className="text-[9px] font-tech text-muted-foreground/50 uppercase tracking-widest">or</span>
        <div className="flex-1 h-px bg-border" />
      </div>

      <button
        type="button"
        disabled={isRunning}
        onClick={() => localFileInputRef.current?.click()}
        className={cn(
          "w-full flex items-center justify-center gap-2 py-2 rounded border border-dashed border-border/60",
          "hover:border-primary/50 hover:bg-primary/5 tech-transition text-muted-foreground/70 hover:text-foreground",
          isRunning && "pointer-events-none opacity-40",
        )}
        aria-label="Load local NIfTI or DICOM file without backend"
      >
        <FolderOpen size={13} aria-hidden />
        <span className="text-[10px] font-tech">Load Local File</span>
        <span className="text-[9px] text-muted-foreground/50">(no backend)</span>
      </button>

      <input
        ref={localFileInputRef}
        type="file"
        accept={LOCAL_ACCEPTED_TYPES}
        onChange={handleLocalInputChange}
        className="hidden"
        aria-label="Local file input (no backend)"
      />

      {/* DICOM ZIP conversion progress */}
      {(convertStatus === "extracting" || convertStatus === "converting") && (
        <div className="space-y-1.5">
          <div
            className="h-1 bg-muted rounded-full overflow-hidden"
            role="progressbar"
            aria-valuenow={convertProgress}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-label="DICOM conversion progress"
          >
            <div
              className="h-full bg-primary tech-transition"
              style={{ width: `${convertProgress}%` }}
            />
          </div>
          <p className="text-[10px] font-tech text-muted-foreground/70 text-center">
            {convertStatus === "extracting" ? "Extracting ZIP\u2026" : `Converting DICOM \u2014 ${convertProgress}%`}
          </p>
        </div>
      )}

      {/* DICOM conversion error */}
      {convertStatus === "error" && convertError !== null && (
        <div className="flex items-start gap-2 p-3 rounded-md bg-destructive/10 border border-destructive/30">
          <AlertTriangle size={14} className="text-destructive shrink-0 mt-0.5" />
          <p className="text-xs text-destructive font-mono">{convertError}</p>
        </div>
      )}

      {/* Upload progress */}
      {isRunning && (
        <div
          className="h-1 bg-muted rounded-full overflow-hidden"
          role="progressbar"
          aria-valuenow={uploadProgress}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-label="Upload progress"
        >
          <div
            className="h-full bg-primary tech-transition"
            style={{ width: `${uploadProgress}%` }}
          />
        </div>
      )}

      {/* Error message */}
      {taskStatus === "error" && errorMessage !== null && (
        <div className="flex items-start gap-2 p-3 rounded-md bg-destructive/10 border border-destructive/30">
          <AlertTriangle size={14} className="text-destructive shrink-0 mt-0.5" />
          <div className="space-y-1">
            <p className="text-xs text-destructive font-mono">{errorMessage}</p>
            <button
              type="button"
              onClick={() => void handleFiles(new DataTransfer().files)}
              className="text-[10px] text-primary hover:underline"
            >
              Retry
            </button>
          </div>
        </div>
      )}

      {/* QC warnings */}
      {qcWarnings.length > 0 && !hasDismissedWarnings && (
        <div className="p-3 rounded-md bg-accent/10 border border-accent/30 space-y-2">
          <div className="flex items-center gap-2">
            <AlertTriangle size={12} className="text-accent shrink-0" />
            <p className="text-[10px] font-tech text-accent uppercase tracking-wider">Scan QC Warnings</p>
          </div>
          <ul className="space-y-1">
            {qcWarnings.map((w, i) => (
              <li key={i} className="text-[11px] text-foreground/70 font-mono">{w}</li>
            ))}
          </ul>
          <button
            type="button"
            onClick={dismissWarnings}
            className="text-[10px] text-muted-foreground hover:text-foreground tech-transition"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* Study info */}
      {isDone && study !== null && (
        <div className="p-3 rounded-md bg-muted/30 space-y-1.5">
          <p className="font-tech text-[10px] text-muted-foreground uppercase tracking-widest">Study Loaded</p>
          <p className="text-xs font-mono text-primary">{study.modality}  {study.dimensions.join("")}</p>
          <button
            type="button"
            onClick={() => setActiveStep("scout")}
            className="w-full mt-2 py-1.5 rounded bg-primary/20 hover:bg-primary/30 text-primary font-tech text-[10px] uppercase tracking-wider tech-transition"
          >
            Proceed to Scout 
          </button>
        </div>
      )}
    </div>
  );
};

export default UploadPanel;
