// src/store/niivueStore.ts
// Shares the NiiVue instance between NiiVueCanvas and sidebar panels.
// Panels use the ref to adjust thresholds, render mode, clip planes, etc.
import { create } from "zustand";
import type { Niivue } from "@niivue/niivue";
import type { TaskStatus } from "@/types";

interface NiivueState {
  /** The live NiiVue instance (set once the canvas attaches). */
  nv: Niivue | null;
  /** Current HU threshold range for bone rendering. */
  huMin: number;
  huMax: number;
  /** Opacity of the 3D volume render (0–1). */
  opacity: number;
  /** Whether we are showing 3D render vs MPR slices. */
  is3dRender: boolean;
  /** Currently active view mode key. */
  activeViewMode: "multi" | "3d" | "axial" | "coronal" | "sagittal";
  taskStatus: TaskStatus;
  errorMessage: string | null;
}

interface NiivueActions {
  setNv: (nv: Niivue | null) => void;
  setHuRange: (min: number, max: number) => void;
  setOpacity: (opacity: number) => void;
  setIs3dRender: (is3d: boolean) => void;
  setActiveViewMode: (mode: NiivueState["activeViewMode"]) => void;
  setTaskStatus: (status: TaskStatus) => void;
  setError: (message: string | null) => void;
}

// Default bone window: 200–2000 HU
const INITIAL_STATE: NiivueState = {
  nv: null,
  huMin: 200,
  huMax: 2000,
  opacity: 1.0,
  is3dRender: false,
  activeViewMode: "multi",
  taskStatus: "idle",
  errorMessage: null,
};

export const useNiivueStore = create<NiivueState & NiivueActions>((set) => ({
  ...INITIAL_STATE,

  setNv: (nv) => set({ nv }),
  setHuRange: (huMin, huMax) => set({ huMin, huMax }),
  setOpacity: (opacity) => set({ opacity }),
  setIs3dRender: (is3dRender) => set({ is3dRender }),
  setActiveViewMode: (activeViewMode) => set({ activeViewMode, is3dRender: activeViewMode === "3d" }),
  setTaskStatus: (taskStatus) => set({ taskStatus }),
  setError: (message) => set({ errorMessage: message, taskStatus: "error" }),
}));
