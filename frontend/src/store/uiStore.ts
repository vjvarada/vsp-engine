// src/store/uiStore.ts
// Global UI state — active workflow step, panel collapse, viewport mode.
import { create } from "zustand";

export type WorkflowStep = "upload" | "scout" | "select" | "segment" | "refine" | "mesh" | "export";
export type ViewportMode = "3d" | "axial" | "coronal" | "sagittal" | "quad";

interface UiState {
  activeStep: WorkflowStep;
  completedSteps: ReadonlySet<WorkflowStep>;
  isContextPanelOpen: boolean;
  isPropertiesPanelOpen: boolean;
  viewportMode: ViewportMode;
  isRefineMode: boolean;
}

interface UiActions {
  setActiveStep: (step: WorkflowStep) => void;
  markStepComplete: (step: WorkflowStep) => void;
  toggleContextPanel: () => void;
  togglePropertiesPanel: () => void;
  setViewportMode: (mode: ViewportMode) => void;
  setRefineMode: (active: boolean) => void;
}

const STEP_ORDER: readonly WorkflowStep[] = [
  "upload", "scout", "select", "segment", "refine", "mesh", "export",
] as const;

export { STEP_ORDER };

const INITIAL_STATE: UiState = {
  activeStep: "upload",
  completedSteps: new Set(),
  isContextPanelOpen: true,
  isPropertiesPanelOpen: true,
  viewportMode: "3d",
  isRefineMode: false,
};

export const useUiStore = create<UiState & UiActions>((set, get) => ({
  ...INITIAL_STATE,

  setActiveStep: (activeStep) => set({ activeStep }),

  markStepComplete: (step) => {
    const next = new Set(get().completedSteps);
    next.add(step);
    set({ completedSteps: next });
  },

  toggleContextPanel: () =>
    set((s) => ({ isContextPanelOpen: !s.isContextPanelOpen })),

  togglePropertiesPanel: () =>
    set((s) => ({ isPropertiesPanelOpen: !s.isPropertiesPanelOpen })),

  setViewportMode: (viewportMode) => set({ viewportMode }),
  setRefineMode: (isRefineMode) => set({ isRefineMode }),
}));
