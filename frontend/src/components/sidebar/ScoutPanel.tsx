// src/components/sidebar/ScoutPanel.tsx
// Scout step: NiiVue-based HU thresholding for bone visualization.
// Uses NiiVue's native volume rendering — no backend needed for this step.
import React, { useCallback, useEffect } from "react";
import { ScanSearch, Bone, RotateCcw, Layers } from "lucide-react";
import { cn } from "@/lib/utils";
import { useNiivueStore } from "@/store/niivueStore";
import { useUploadStore } from "@/store/uploadStore";
import { useUiStore } from "@/store/uiStore";
import { logger } from "@/lib/logger";

// ── HU presets ──────────────────────────────────────────────
const HU_PRESETS = {
  bone:       { min: 200, max: 2000, label: "Bone" },
  softTissue: { min: -100, max: 300, label: "Soft Tissue" },
  lung:       { min: -1000, max: -200, label: "Lung" },
  full:       { min: -1024, max: 3071, label: "Full Range" },
} as const;

type PresetKey = keyof typeof HU_PRESETS;

// ── View mode labels ────────────────────────────────────────
const VIEW_MODES = [
  { key: "multi", label: "Multi" },
  { key: "3d", label: "3D" },
  { key: "axial", label: "Axial" },
  { key: "coronal", label: "Coronal" },
  { key: "sagittal", label: "Sagittal" },
] as const;

// ── NiiVue SLICE_TYPE constants ─────────────────────────────
// NiiVue uses numeric constants for setSliceType:
// SLICE_TYPE.AXIAL=0, CORONAL=1, SAGITTAL=2, MULTIPLANAR=3, RENDER=4
const NIIVUE_SLICE_TYPE = {
  axial: 0,
  coronal: 1,
  sagittal: 2,
  multiplanar: 3,
  render: 4,
} as const;

// ── ScoutPanel ──────────────────────────────────────────────

const ScoutPanel: React.FC = () => {
  const study = useUploadStore((s) => s.study);
  const nv = useNiivueStore((s) => s.nv);
  const huMin = useNiivueStore((s) => s.huMin);
  const huMax = useNiivueStore((s) => s.huMax);
  const opacity = useNiivueStore((s) => s.opacity);
  const is3dRender = useNiivueStore((s) => s.is3dRender);
  const activeViewMode = useNiivueStore((s) => s.activeViewMode);
  const setHuRange = useNiivueStore((s) => s.setHuRange);
  const setOpacity = useNiivueStore((s) => s.setOpacity);
  const setIs3dRender = useNiivueStore((s) => s.setIs3dRender);
  const setActiveViewMode = useNiivueStore((s) => s.setActiveViewMode);

  const markComplete = useUiStore((s) => s.markStepComplete);
  const setActiveStep = useUiStore((s) => s.setActiveStep);

  const hasVolume = study !== null && nv !== null;

  // Apply HU threshold to NiiVue volume
  const applyThreshold = useCallback(() => {
    if (nv === null) return;
    const vol = nv.volumes?.[0];
    if (vol === undefined) return;

    // Set colormap range (window/level) to the HU range
    vol.cal_min = huMin;
    vol.cal_max = huMax;
    vol.opacity = opacity;
    nv.updateGLVolume();
    logger.debug("NiiVue threshold applied", huMin, huMax, opacity);
  }, [nv, huMin, huMax, opacity]);

  // Apply threshold whenever values change
  useEffect(() => {
    applyThreshold();
  }, [applyThreshold]);

  // Switch to 3D render mode (used by the quick bone-view button)
  const handleToggle3d = useCallback(() => {
    if (nv === null) return;

    if (is3dRender) {
      nv.setSliceType(NIIVUE_SLICE_TYPE.multiplanar);
      setActiveViewMode("multi");
    } else {
      nv.setSliceType(NIIVUE_SLICE_TYPE.render);
      setActiveViewMode("3d");
      applyThreshold();
    }
  }, [nv, is3dRender, setActiveViewMode, applyThreshold]);

  // Lookup table for view mode → NiiVue slice type
  const VIEW_MODE_TO_SLICE: Record<string, number> = {
    multi: NIIVUE_SLICE_TYPE.multiplanar,
    "3d": NIIVUE_SLICE_TYPE.render,
    axial: NIIVUE_SLICE_TYPE.axial,
    coronal: NIIVUE_SLICE_TYPE.coronal,
    sagittal: NIIVUE_SLICE_TYPE.sagittal,
  };

  // Set a specific view mode
  const handleViewMode = useCallback((mode: string) => {
    if (nv === null) return;

    const sliceType = VIEW_MODE_TO_SLICE[mode] ?? NIIVUE_SLICE_TYPE.multiplanar;
    nv.setSliceType(sliceType);
    setActiveViewMode(mode as typeof activeViewMode);
  }, [nv, setActiveViewMode]);  // eslint-disable-line react-hooks/exhaustive-deps

  // Apply a preset
  const handlePreset = useCallback((key: PresetKey) => {
    const preset = HU_PRESETS[key];
    setHuRange(preset.min, preset.max);
  }, [setHuRange]);

  // Proceed to next step
  const handleProceed = useCallback(() => {
    markComplete("scout");
    setActiveStep("select");
  }, [markComplete, setActiveStep]);

  return (
    <div className="p-4 space-y-4">
      {/* Section: View Mode */}
      <div className="space-y-2">
        <p className="font-tech text-[10px] text-muted-foreground uppercase tracking-widest">
          View Mode
        </p>
        <div className="grid grid-cols-5 gap-1">
          {VIEW_MODES.map(({ key, label }) => (
            <button
              key={key}
              type="button"
              disabled={!hasVolume}
              onClick={() => handleViewMode(key)}
              className={cn(
                "py-1.5 rounded text-[10px] font-tech uppercase tracking-wider tech-transition",
                activeViewMode === key
                  ? "bg-primary/20 text-primary border border-primary/40"
                  : "bg-muted/20 text-muted-foreground/60 hover:bg-muted/30 border border-transparent",
                !hasVolume && "opacity-40 cursor-not-allowed",
              )}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      {/* Section: HU Presets */}
      <div className="space-y-2">
        <p className="font-tech text-[10px] text-muted-foreground uppercase tracking-widest">
          HU Presets
        </p>
        <div className="grid grid-cols-2 gap-1">
          {(Object.entries(HU_PRESETS) as [PresetKey, { min: number; max: number; label: string }][]).map(
            ([key, preset]) => (
              <button
                key={key}
                type="button"
                disabled={!hasVolume}
                onClick={() => handlePreset(key)}
                className={cn(
                  "flex items-center justify-center gap-1.5 py-1.5 rounded text-[10px] font-tech tech-transition",
                  huMin === preset.min && huMax === preset.max
                    ? "bg-primary/20 text-primary border border-primary/40"
                    : "bg-muted/20 text-muted-foreground/70 hover:bg-muted/30 border border-transparent",
                  !hasVolume && "opacity-40 cursor-not-allowed",
                )}
              >
                {key === "bone" && <Bone size={11} aria-hidden />}
                {key === "full" && <Layers size={11} aria-hidden />}
                {preset.label}
              </button>
            ),
          )}
        </div>
      </div>

      {/* Section: HU Range Sliders */}
      <div className="space-y-3">
        <p className="font-tech text-[10px] text-muted-foreground uppercase tracking-widest">
          HU Threshold
        </p>

        {/* Min HU */}
        <div className="space-y-1">
          <div className="flex items-center justify-between">
            <label htmlFor="hu-min" className="text-[10px] text-muted-foreground/70 font-mono">Min HU</label>
            <span className="text-[10px] text-primary font-mono tabular-nums">{huMin}</span>
          </div>
          <input
            id="hu-min"
            type="range"
            min={-1024}
            max={3071}
            step={10}
            value={huMin}
            disabled={!hasVolume}
            onChange={(e) => setHuRange(Number(e.target.value), huMax)}
            className="w-full h-1 accent-primary"
            aria-label="Minimum HU threshold"
            aria-valuemin={-1024}
            aria-valuemax={3071}
            aria-valuenow={huMin}
          />
        </div>

        {/* Max HU */}
        <div className="space-y-1">
          <div className="flex items-center justify-between">
            <label htmlFor="hu-max" className="text-[10px] text-muted-foreground/70 font-mono">Max HU</label>
            <span className="text-[10px] text-primary font-mono tabular-nums">{huMax}</span>
          </div>
          <input
            id="hu-max"
            type="range"
            min={-1024}
            max={3071}
            step={10}
            value={huMax}
            disabled={!hasVolume}
            onChange={(e) => setHuRange(huMin, Number(e.target.value))}
            className="w-full h-1 accent-primary"
            aria-label="Maximum HU threshold"
            aria-valuemin={-1024}
            aria-valuemax={3071}
            aria-valuenow={huMax}
          />
        </div>

        {/* Opacity */}
        <div className="space-y-1">
          <div className="flex items-center justify-between">
            <label htmlFor="vol-opacity" className="text-[10px] text-muted-foreground/70 font-mono">Opacity</label>
            <span className="text-[10px] text-primary font-mono tabular-nums">{(opacity * 100).toFixed(0)}%</span>
          </div>
          <input
            id="vol-opacity"
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={opacity}
            disabled={!hasVolume}
            onChange={(e) => setOpacity(Number(e.target.value))}
            className="w-full h-1 accent-primary"
            aria-label="Volume opacity"
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={Math.round(opacity * 100)}
          />
        </div>
      </div>

      {/* Quick 3D bone view button */}
      <button
        type="button"
        disabled={!hasVolume}
        onClick={() => {
          handlePreset("bone");
          if (!is3dRender) handleToggle3d();
        }}
        className={cn(
          "w-full flex items-center justify-center gap-2 py-2 rounded font-tech text-[10px] uppercase tracking-wider tech-transition",
          hasVolume
            ? "bg-primary/20 hover:bg-primary/30 text-primary"
            : "bg-muted/20 text-muted-foreground/40 cursor-not-allowed",
        )}
      >
        <Bone size={13} aria-hidden />
        Show 3D Bone View
      </button>

      {/* Proceed */}
      {hasVolume && (
        <button
          type="button"
          onClick={handleProceed}
          className="w-full py-1.5 rounded bg-primary/20 hover:bg-primary/30 text-primary font-tech text-[10px] uppercase tracking-wider tech-transition"
        >
          Proceed to Select ROI →
        </button>
      )}
    </div>
  );
};

export default ScoutPanel;
