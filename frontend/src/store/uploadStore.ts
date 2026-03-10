// src/store/uploadStore.ts
import { create } from "zustand";
import type { Study, TaskStatus } from "@/types";

interface UploadState {
  study: Study | null;
  /** Original filename of a locally-loaded volume (used by NiiVue for format detection on blob URLs). */
  localFileName: string | null;
  taskStatus: TaskStatus;
  errorMessage: string | null;
  uploadProgress: number;
  qcWarnings: readonly string[];
  hasDismissedWarnings: boolean;
}

interface UploadActions {
  setStudy: (study: Study) => void;
  setLocalFileName: (name: string | null) => void;
  setTaskStatus: (status: TaskStatus) => void;
  setError: (message: string | null) => void;
  setUploadProgress: (percent: number) => void;
  setQcWarnings: (warnings: readonly string[]) => void;
  dismissWarnings: () => void;
  reset: () => void;
}

const INITIAL_STATE: UploadState = {
  study: null,
  localFileName: null,
  taskStatus: "idle",
  errorMessage: null,
  uploadProgress: 0,
  qcWarnings: [],
  hasDismissedWarnings: false,
};

export const useUploadStore = create<UploadState & UploadActions>((set) => ({
  ...INITIAL_STATE,
  setStudy: (study) => set({ study, taskStatus: "done" }),
  setLocalFileName: (localFileName) => set({ localFileName }),
  setTaskStatus: (status) => set({ taskStatus: status }),
  setError: (message) => set({ errorMessage: message, taskStatus: "error" }),
  setUploadProgress: (percent) => set({ uploadProgress: percent }),
  setQcWarnings: (warnings) => set({ qcWarnings: warnings }),
  dismissWarnings: () => set({ hasDismissedWarnings: true }),
  reset: () => set(INITIAL_STATE),
}));
