// src/store/scoutStore.ts
import { create } from "zustand";
import type { IslandMeta, Roi, TaskStatus } from "@/types";

interface ScoutState {
  islands: readonly IslandMeta[];
  selectedIslandIds: ReadonlySet<string>;
  roi: Roi | null;
  isRoiLocked: boolean;
  taskStatus: TaskStatus;
  errorMessage: string | null;
}

interface ScoutActions {
  setIslands: (islands: readonly IslandMeta[]) => void;
  toggleIsland: (islandId: string) => void;
  selectAll: () => void;
  clearSelection: () => void;
  setRoi: (roi: Roi) => void;
  updateRoiAxis: (key: keyof Roi, value: number) => void;
  lockRoi: () => void;
  unlockRoi: () => void;
  setTaskStatus: (status: TaskStatus) => void;
  setError: (message: string | null) => void;
  reset: () => void;
}

const INITIAL_STATE: ScoutState = {
  islands: [],
  selectedIslandIds: new Set(),
  roi: null,
  isRoiLocked: false,
  taskStatus: "idle",
  errorMessage: null,
};

export const useScoutStore = create<ScoutState & ScoutActions>((set, get) => ({
  ...INITIAL_STATE,

  setIslands: (islands) => set({ islands }),

  toggleIsland: (islandId) => {
    const current = get().selectedIslandIds;
    const next = new Set(current);
    if (next.has(islandId)) {
      next.delete(islandId);
    } else {
      next.add(islandId);
    }
    set({ selectedIslandIds: next });
  },

  selectAll: () => {
    const all = new Set(get().islands.map((i) => i.islandId));
    set({ selectedIslandIds: all });
  },

  clearSelection: () => set({ selectedIslandIds: new Set() }),

  setRoi: (roi) => set({ roi }),

  updateRoiAxis: (key, value) => {
    const current = get().roi;
    if (current === null) return;
    set({ roi: { ...current, [key]: value } });
  },

  lockRoi: () => set({ isRoiLocked: true }),
  unlockRoi: () => set({ isRoiLocked: false }),

  setTaskStatus: (taskStatus) => set({ taskStatus }),
  setError: (message) => set({ errorMessage: message, taskStatus: "error" }),
  reset: () => set({ ...INITIAL_STATE, selectedIslandIds: new Set() }),
}));
