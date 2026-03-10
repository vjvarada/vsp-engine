// src/store/segmentStore.ts
import { create } from "zustand";
import type { SegLabel, TaskStatus } from "@/types";

interface SegmentState {
  labels: readonly SegLabel[];
  taskStatus: TaskStatus;
  taskId: string | null;
  progressPercent: number;
  progressStep: string;
  errorMessage: string | null;
}

interface SegmentActions {
  setLabels: (labels: readonly SegLabel[]) => void;
  toggleLabelVisibility: (labelId: number) => void;
  setTaskId: (taskId: string) => void;
  setTaskStatus: (status: TaskStatus) => void;
  setProgress: (percent: number, step: string) => void;
  setError: (message: string | null) => void;
  reset: () => void;
}

const INITIAL_STATE: SegmentState = {
  labels: [],
  taskStatus: "idle",
  taskId: null,
  progressPercent: 0,
  progressStep: "",
  errorMessage: null,
};

export const useSegmentStore = create<SegmentState & SegmentActions>((set, get) => ({
  ...INITIAL_STATE,

  setLabels: (labels) => set({ labels, taskStatus: "done" }),

  toggleLabelVisibility: (labelId) => {
    const labels = get().labels.map((l) =>
      l.labelId === labelId ? { ...l, isVisible: !l.isVisible } : l,
    );
    set({ labels });
  },

  setTaskId: (taskId) => set({ taskId, taskStatus: "running" }),
  setTaskStatus: (taskStatus) => set({ taskStatus }),
  setProgress: (progressPercent, progressStep) => set({ progressPercent, progressStep }),
  setError: (message) => set({ errorMessage: message, taskStatus: "error" }),
  reset: () => set(INITIAL_STATE),
}));
