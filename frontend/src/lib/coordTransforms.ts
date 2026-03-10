// src/lib/coordTransforms.ts
// EXPLICIT coordinate-system transforms for VSP Engine.
//
// Coordinate systems:
//   NIfTI RAS mm  — X=Right, Y=Anterior, Z=Superior (world mm)
//   R3F scene     — same as NIfTI RAS; scene units = mm, Y = Superior
//   NiiVue voxel  — (i, j, k) integer voxel indices
//   SAM-Med3D     — (z_norm, y_norm, x_norm) normalized [0,1] within 128 patch
//   Canvas pixel  — (px, py) in NiiVue canvas pixel space
//
// NEVER guess coordinate axis order — always use helpers from this file.

import type * as THREE from "three";

//  NIfTI voxel  RAS mm 

export interface NiftiHeader {
  /** Affine matrix row-major [4][4] */
  readonly affine: readonly [
    readonly [number, number, number, number],
    readonly [number, number, number, number],
    readonly [number, number, number, number],
    readonly [number, number, number, number],
  ];
}

/** Convert NIfTI voxel indices to RAS world-mm coordinates. */
export function voxelToRas(
  voxel: readonly [number, number, number],
  affine: NiftiHeader["affine"],
): [number, number, number] {
  const [i, j, k] = voxel;
  const x = affine[0][0] * i + affine[0][1] * j + affine[0][2] * k + affine[0][3];
  const y = affine[1][0] * i + affine[1][1] * j + affine[1][2] * k + affine[1][3];
  const z = affine[2][0] * i + affine[2][1] * j + affine[2][2] * k + affine[2][3];
  return [x, y, z];
}

/** Convert RAS world-mm to NIfTI voxel indices (nearest-neighbour). */
export function rasToVoxel(
  ras: readonly [number, number, number],
  affineInv: NiftiHeader["affine"],
): [number, number, number] {
  const [x, y, z] = ras;
  const i = affineInv[0][0] * x + affineInv[0][1] * y + affineInv[0][2] * z + affineInv[0][3];
  const j = affineInv[1][0] * x + affineInv[1][1] * y + affineInv[1][2] * z + affineInv[1][3];
  const k = affineInv[2][0] * x + affineInv[2][1] * y + affineInv[2][2] * z + affineInv[2][3];
  return [Math.round(i), Math.round(j), Math.round(k)];
}

//  RAS mm  R3F scene 
// R3F scene is 1:1 with RAS mm (scene units = mm, Y = Superior).
// THREE.Vector3 directly holds RAS mm values.

/** Convert RAS mm to a Three.js Vector3 (no-op — same coordinate system). */
export function rasToR3f(ras: readonly [number, number, number]): [number, number, number] {
  return [ras[0], ras[1], ras[2]];
}

/** Convert a Three.js Vector3 world position (R3F mm) to RAS mm. */
export function r3fToRas(vec: THREE.Vector3): [number, number, number] {
  return [vec.x, vec.y, vec.z];
}

//  RAS mm  SAM-Med3D point 
// SAM-Med3D expects (z_norm, y_norm, x_norm) within the 128 patch [0,1].

export interface SamPatch {
  /** Center of the 128 patch in voxel space */
  readonly centerVoxel: readonly [number, number, number];
  /** Side length of the patch in voxels */
  readonly patchSizeVoxels: 128;
  readonly affineInv: NiftiHeader["affine"];
}

/**
 * Convert an R3F viewport click (RAS mm) to SAM-Med3D normalized point.
 * Returns (z_norm, y_norm, x_norm) as required by the SAM-Med3D API.
 */
export function niftiVoxelToSamMed3dPoint(
  clickRas: readonly [number, number, number],
  patch: SamPatch,
): [number, number, number] {
  const vox = rasToVoxel(clickRas, patch.affineInv);
  const half = patch.patchSizeVoxels / 2;
  // patch spans [center - half, center + half) on each axis
  const zNorm = (vox[2] - (patch.centerVoxel[2] - half)) / patch.patchSizeVoxels;
  const yNorm = (vox[1] - (patch.centerVoxel[1] - half)) / patch.patchSizeVoxels;
  const xNorm = (vox[0] - (patch.centerVoxel[0] - half)) / patch.patchSizeVoxels;
  // clamp to [0, 1]
  return [
    Math.max(0, Math.min(1, zNorm)),
    Math.max(0, Math.min(1, yNorm)),
    Math.max(0, Math.min(1, xNorm)),
  ];
}

//  NiiVue canvas pixel helpers 
// NiiVue exposes mm2frac() to convert world mm to fractional position [0,1],
// then we multiply by canvas pixel dimensions to get pixel position.

/**
 * Convert a world-mm coordinate to canvas pixel position using NiiVue.
 * The NiiVue instance must be initialized (canvas is active).
 *
 * Usage:
 *   const canvasPx = rasMMToCanvasPixel(nv, zWorldMm, canvas.height);
 */
export function rasMMtoFrac(
  nv: { mm2frac: (mm: number[], volIdx: number, isDepthPicker: boolean) => number[] },
  rasXyz: readonly [number, number, number],
): [number, number, number] {
  const frac = nv.mm2frac([rasXyz[0], rasXyz[1], rasXyz[2]], 0, false);
  return [frac[0] ?? 0, frac[1] ?? 0, frac[2] ?? 0];
}
