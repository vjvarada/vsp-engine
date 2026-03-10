// src/types/index.ts
// Domain types for VSP Engine — all world-space coordinates in NIfTI RAS mm

export type TaskStatus = "idle" | "running" | "done" | "error";

export type TaskResult<T> =
  | { status: "idle" }
  | { status: "running"; percent: number }
  | { status: "done"; data: T }
  | { status: "error"; message: string };

//  Study 

export interface Study {
  readonly studyId: string;
  readonly niftiUrl: string;
  readonly voxelSpacing: readonly [number, number, number];
  readonly dimensions: readonly [number, number, number];
  readonly modality: "CT" | "MR" | "unknown";
  readonly qcWarnings: readonly string[];
}

//  Scout Islands 

export interface IslandMeta {
  readonly islandId: string;
  readonly centroidXyz: readonly [number, number, number]; // RAS mm
  readonly aabb: Aabb;                                      // RAS mm
  readonly plyUrl: string;
  readonly voxelCount: number;
}

export interface Aabb {
  readonly xMin: number;
  readonly xMax: number;
  readonly yMin: number;
  readonly yMax: number;
  readonly zMin: number;
  readonly zMax: number;
}

//  ROI 

export interface Roi {
  xMin: number;
  xMax: number;
  yMin: number;
  yMax: number;
  zMin: number;
  zMax: number;
}

//  Segmentation 

export interface SegLabel {
  readonly labelId: number;
  readonly name: string;         // e.g. "femur_left"
  readonly color: string;        // hex
  readonly meshUrl: string | null;
  isVisible: boolean;
}

//  Mesh 

export type MeshQuality = "preview" | "standard" | "high";

export interface MeshResult {
  readonly meshUrl: string;
  readonly faceCount: number;
  readonly isWatertight: boolean;
}

//  Export 

export type ExportFormat = "stl" | "obj" | "3mf";
export type ExportUnion = "combined" | "separate";

export interface ExportLabelConfig {
  readonly labelId: number;
  readonly name: string;
  included: boolean;
  union: ExportUnion;
}

//  Measurement 

export interface Measurement {
  readonly id: string;
  readonly type: "distance" | "angle";
  readonly value: number;       // mm or degrees
  readonly label: string;
}

//  API helpers 

export interface ApiError {
  readonly detail: string;
  readonly type: string;
}

export interface TaskResponse {
  readonly taskId: string;
}

export interface JobStatus {
  readonly state: "PENDING" | "PROGRESS" | "SUCCESS" | "FAILURE";
  readonly percent?: number;
  readonly step?: string;
  readonly result?: unknown;
  readonly error?: string;
}
