// src/store/planReportStore.ts
// Captures viewport screenshot, measurements, surgeon notes for plan report.
import { create } from "zustand";
import type { Measurement } from "@/types";

interface PlanReportState {
  screenshotDataUrl: string | null;
  measurements: readonly Measurement[];
  surgeonNotes: string;
  exportManifest: readonly string[];
}

interface PlanReportActions {
  setScreenshot: (dataUrl: string) => void;
  addMeasurement: (m: Measurement) => void;
  removeMeasurement: (id: string) => void;
  setSurgeonNotes: (notes: string) => void;
  setExportManifest: (files: readonly string[]) => void;
  reset: () => void;
}

const INITIAL_STATE: PlanReportState = {
  screenshotDataUrl: null,
  measurements: [],
  surgeonNotes: "",
  exportManifest: [],
};

export const usePlanReportStore = create<PlanReportState & PlanReportActions>((set, get) => ({
  ...INITIAL_STATE,
  setScreenshot: (screenshotDataUrl) => set({ screenshotDataUrl }),
  addMeasurement: (m) => set({ measurements: [...get().measurements, m] }),
  removeMeasurement: (id) =>
    set({ measurements: get().measurements.filter((m) => m.id !== id) }),
  setSurgeonNotes: (surgeonNotes) => set({ surgeonNotes }),
  setExportManifest: (exportManifest) => set({ exportManifest }),
  reset: () => set(INITIAL_STATE),
}));
