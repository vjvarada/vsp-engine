// src/store/meshStore.ts
import { create } from "zustand";
import type { MeshQuality, MeshResult, TaskStatus } from "@/types";

interface MeshState {
  meshResult: MeshResult | null;
  quality: MeshQuality;
  taskStatus: TaskStatus;
  taskId: string | null;
  progressPercent: number;
  errorMessage: string | null;
}

interface MeshActions {
  setMeshResult: (result: MeshResult) => void;
  setQuality: (quality: MeshQuality) => void;
  setTaskId: (taskId: string) => void;
  setTaskStatus: (status: TaskStatus) => void;
  setProgress: (percent: number) => void;
  setError: (message: string | null) => void;
  reset: () => void;
}

const INITIAL_STATE: MeshState = {
  meshResult: null,
  quality: "standard",
  taskStatus: "idle",
  taskId: null,
  progressPercent: 0,
  errorMessage: null,
};

export const useMeshStore = create<MeshState & MeshActions>((set) => ({
  ...INITIAL_STATE,
  setMeshResult: (meshResult) => set({ meshResult, taskStatus: "done" }),
  setQuality: (quality) => set({ quality }),
  setTaskId: (taskId) => set({ taskId, taskStatus: "running" }),
  setTaskStatus: (taskStatus) => set({ taskStatus }),
  setProgress: (progressPercent) => set({ progressPercent }),
  setError: (message) => set({ errorMessage: message, taskStatus: "error" }),
  reset: () => set(INITIAL_STATE),
}));
