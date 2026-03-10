// src/store/exportStore.ts
import { create } from "zustand";
import type { ExportFormat, ExportLabelConfig, TaskStatus } from "@/types";

interface ExportState {
  labels: readonly ExportLabelConfig[];
  format: ExportFormat;
  scaleFactor: number;
  downloadUrl: string | null;
  taskStatus: TaskStatus;
  errorMessage: string | null;
}

interface ExportActions {
  setLabels: (labels: readonly ExportLabelConfig[]) => void;
  toggleLabel: (labelId: number) => void;
  setLabelUnion: (labelId: number, union: "combined" | "separate") => void;
  setFormat: (format: ExportFormat) => void;
  setScaleFactor: (factor: number) => void;
  setDownloadUrl: (url: string) => void;
  setTaskStatus: (status: TaskStatus) => void;
  setError: (message: string | null) => void;
  reset: () => void;
}

const INITIAL_STATE: ExportState = {
  labels: [],
  format: "stl",
  scaleFactor: 1,
  downloadUrl: null,
  taskStatus: "idle",
  errorMessage: null,
};

export const useExportStore = create<ExportState & ExportActions>((set, get) => ({
  ...INITIAL_STATE,

  setLabels: (labels) => set({ labels }),

  toggleLabel: (labelId) => {
    const labels = get().labels.map((l) =>
      l.labelId === labelId ? { ...l, included: !l.included } : l,
    );
    set({ labels });
  },

  setLabelUnion: (labelId, union) => {
    const labels = get().labels.map((l) =>
      l.labelId === labelId ? { ...l, union } : l,
    );
    set({ labels });
  },

  setFormat: (format) => set({ format }),
  setScaleFactor: (scaleFactor) => set({ scaleFactor }),
  setDownloadUrl: (downloadUrl) => set({ downloadUrl, taskStatus: "done" }),
  setTaskStatus: (taskStatus) => set({ taskStatus }),
  setError: (message) => set({ errorMessage: message, taskStatus: "error" }),
  reset: () => set(INITIAL_STATE),
}));
