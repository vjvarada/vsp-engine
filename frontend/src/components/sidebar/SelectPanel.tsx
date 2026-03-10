// src/components/sidebar/SelectPanel.tsx
// 3 dual-range sliders (X/Y/Z) + 6 numeric mm inputs, all synced to scoutStore.roi.
import React, { useCallback } from "react";
import { Lock, Unlock, AlertTriangle } from "lucide-react";
import { cn } from "@/lib/utils";
import { useScoutStore } from "@/store/scoutStore";
import { useUiStore } from "@/store/uiStore";
import type { Roi } from "@/types";

const MIN_ROI_SIZE_MM = 10;

//  AxisSlider (extracted per-axis sub-component) 

interface AxisSliderProps {
  readonly axis: "X" | "Y" | "Z";
  readonly minKey: keyof Roi;
  readonly maxKey: keyof Roi;
  readonly min: number;
  readonly max: number;
  readonly valueMin: number;
  readonly valueMax: number;
  readonly color: string;
  readonly isLocked: boolean;
  readonly onChange: (key: keyof Roi, value: number) => void;
}

const AxisSlider: React.FC<AxisSliderProps> = React.memo(
  ({ axis, minKey, maxKey, min, max, valueMin, valueMax, color, isLocked, onChange }) => {
    const range = max - min || 1;

    const handleMinChange = useCallback(
      (e: React.ChangeEvent<HTMLInputElement>) => {
        const v = Number(e.target.value);
        const clamped = Math.min(v, valueMax - MIN_ROI_SIZE_MM);
        onChange(minKey, clamped);
      },
      [minKey, onChange, valueMax],
    );

    const handleMaxChange = useCallback(
      (e: React.ChangeEvent<HTMLInputElement>) => {
        const v = Number(e.target.value);
        const clamped = Math.max(v, valueMin + MIN_ROI_SIZE_MM);
        onChange(maxKey, clamped);
      },
      [maxKey, onChange, valueMin],
    );

    const minPct = ((valueMin - min) / range) * 100;
    const maxPct = ((valueMax - min) / range) * 100;

    return (
      <div className="space-y-1.5">
        <div className="flex items-center justify-between">
          <span className="font-tech text-[10px] uppercase tracking-widest" style={{ color }}>
            {axis}
          </span>
          <div className="flex items-center gap-2 text-[10px] font-mono text-muted-foreground">
            <input
              type="number"
              value={Math.round(valueMin)}
              disabled={isLocked}
              onChange={handleMinChange}
              aria-label={`${axis} minimum mm`}
              aria-valuemin={min}
              aria-valuemax={valueMax - MIN_ROI_SIZE_MM}
              aria-valuenow={valueMin}
              className="w-14 bg-muted/30 border border-border/40 rounded px-1.5 py-0.5 text-right tabular-nums disabled:opacity-40"
            />
            <span className="text-muted-foreground/40">–</span>
            <input
              type="number"
              value={Math.round(valueMax)}
              disabled={isLocked}
              onChange={handleMaxChange}
              aria-label={`${axis} maximum mm`}
              aria-valuemin={valueMin + MIN_ROI_SIZE_MM}
              aria-valuemax={max}
              aria-valuenow={valueMax}
              className="w-14 bg-muted/30 border border-border/40 rounded px-1.5 py-0.5 text-right tabular-nums disabled:opacity-40"
            />
            <span className="text-muted-foreground/50">mm</span>
          </div>
        </div>

        {/* Visual range track */}
        <div className="relative h-2 bg-muted/30 rounded-full">
          <div
            className="absolute h-full rounded-full opacity-60"
            style={{
              backgroundColor: color,
              left: `${minPct}%`,
              width: `${maxPct - minPct}%`,
            }}
            aria-hidden
          />
        </div>
      </div>
    );
  },
);
AxisSlider.displayName = "AxisSlider";

//  SelectPanel 

const SelectPanel: React.FC = () => {
  const roi = useScoutStore((s) => s.roi);
  const isLocked = useScoutStore((s) => s.isRoiLocked);
  const islands = useScoutStore((s) => s.islands);
  const selectedIslandIds = useScoutStore((s) => s.selectedIslandIds);

  const updateRoiAxis = useScoutStore((s) => s.updateRoiAxis);
  const lockRoi = useScoutStore((s) => s.lockRoi);
  const unlockRoi = useScoutStore((s) => s.unlockRoi);
  const setRoi = useScoutStore((s) => s.setRoi);
  const markComplete = useUiStore((s) => s.markStepComplete);
  const setActiveStep = useUiStore((s) => s.setActiveStep);

  // Compute suggested ROI from selected islands
  const computeRoi = useCallback(() => {
    const selected = islands.filter((i) => selectedIslandIds.has(i.islandId));
    if (selected.length === 0) return;

    const PADDING = 0.1;
    const xMin = Math.min(...selected.map((i) => i.aabb.xMin));
    const xMax = Math.max(...selected.map((i) => i.aabb.xMax));
    const yMin = Math.min(...selected.map((i) => i.aabb.yMin));
    const yMax = Math.max(...selected.map((i) => i.aabb.yMax));
    const zMin = Math.min(...selected.map((i) => i.aabb.zMin));
    const zMax = Math.max(...selected.map((i) => i.aabb.zMax));

    const dx = (xMax - xMin) * PADDING;
    const dy = (yMax - yMin) * PADDING;
    const dz = (zMax - zMin) * PADDING;

    setRoi({
      xMin: xMin - dx, xMax: xMax + dx,
      yMin: yMin - dy, yMax: yMax + dy,
      zMin: zMin - dz, zMax: zMax + dz,
    });
  }, [islands, selectedIslandIds, setRoi]);

  // Auto-compute ROI when panel opens if none exists
  React.useEffect(() => {
    if (roi === null && selectedIslandIds.size > 0) computeRoi();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const handleConfirm = useCallback(() => {
    lockRoi();
    markComplete("select");
    setActiveStep("segment");
  }, [lockRoi, markComplete, setActiveStep]);

  if (islands.length === 0) {
    return (
      <div className="p-4">
        <p className="text-[11px] text-muted-foreground/60 font-mono">
          Run Scout first to generate bone islands.
        </p>
      </div>
    );
  }

  return (
    <div className="p-4 space-y-4">
      {/* Auto-compute button */}
      <button
        type="button"
        disabled={selectedIslandIds.size === 0 || isLocked}
        onClick={computeRoi}
        className={cn(
          "w-full py-1.5 rounded font-tech text-[10px] uppercase tracking-wider tech-transition",
          selectedIslandIds.size > 0 && !isLocked
            ? "bg-muted/30 hover:bg-muted/50 text-foreground/70"
            : "bg-muted/10 text-muted-foreground/30 cursor-not-allowed",
        )}
      >
        Auto-Compute ROI from Selection
      </button>

      {roi !== null ? (
        <>
          {/* 3 axis sliders */}
          <div className="space-y-3">
            <AxisSlider
              axis="X" minKey="xMin" maxKey="xMax"
              min={-500} max={500}
              valueMin={roi.xMin} valueMax={roi.xMax}
              color="#ef4444" isLocked={isLocked}
              onChange={updateRoiAxis}
            />
            <AxisSlider
              axis="Y" minKey="yMin" maxKey="yMax"
              min={-500} max={500}
              valueMin={roi.yMin} valueMax={roi.yMax}
              color="#22c55e" isLocked={isLocked}
              onChange={updateRoiAxis}
            />
            <AxisSlider
              axis="Z" minKey="zMin" maxKey="zMax"
              min={-500} max={1000}
              valueMin={roi.zMin} valueMax={roi.zMax}
              color="#3b82f6" isLocked={isLocked}
              onChange={updateRoiAxis}
            />
          </div>

          {/* Lock / Confirm */}
          <button
            type="button"
            onClick={isLocked ? unlockRoi : handleConfirm}
            className={cn(
              "w-full flex items-center justify-center gap-2 py-2 rounded font-tech text-[10px] uppercase tracking-wider tech-transition",
              isLocked
                ? "bg-accent/20 hover:bg-accent/30 text-accent"
                : "bg-primary/20 hover:bg-primary/30 text-primary",
            )}
          >
            {isLocked ? <Unlock size={12} aria-hidden /> : <Lock size={12} aria-hidden />}
            {isLocked ? "Edit ROI" : "Confirm ROI & Run AI "}
          </button>
        </>
      ) : (
        <div className="flex items-center gap-2 p-3 rounded bg-muted/20">
          <AlertTriangle size={13} className="text-accent shrink-0" />
          <p className="text-[11px] text-muted-foreground/70 font-mono">
            Select islands to auto-compute ROI.
          </p>
        </div>
      )}
    </div>
  );
};

export default SelectPanel;
