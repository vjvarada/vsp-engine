// src/store/implantStore.ts
// Stores implant template STL overlay transform (TransformControls — correct usage).
import { create } from "zustand";
import type * as THREE from "three";

interface ImplantState {
  implantUrl: string | null;
  position: readonly [number, number, number];
  rotation: readonly [number, number, number];
  scale: number;
  isVisible: boolean;
}

interface ImplantActions {
  setImplantUrl: (url: string | null) => void;
  setTransform: (
    position: readonly [number, number, number],
    rotation: readonly [number, number, number],
    scale: number,
  ) => void;
  setPositionFromMatrix: (matrix: THREE.Matrix4) => void;
  toggleVisibility: () => void;
  reset: () => void;
}

const INITIAL_STATE: ImplantState = {
  implantUrl: null,
  position: [0, 0, 0],
  rotation: [0, 0, 0],
  scale: 1,
  isVisible: true,
};

export const useImplantStore = create<ImplantState & ImplantActions>((set, get) => ({
  ...INITIAL_STATE,
  setImplantUrl: (implantUrl) => set({ implantUrl }),
  setTransform: (position, rotation, scale) => set({ position, rotation, scale }),
  setPositionFromMatrix: (matrix) => {
    const pos: readonly [number, number, number] = [
      matrix.elements[12] ?? 0,
      matrix.elements[13] ?? 0,
      matrix.elements[14] ?? 0,
    ];
    set({ position: pos });
  },
  toggleVisibility: () => set((s) => ({ isVisible: !s.isVisible })),
  reset: () => set(INITIAL_STATE),
}));
