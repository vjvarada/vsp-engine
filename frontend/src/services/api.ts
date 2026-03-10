// src/services/api.ts
// Axios-based API client for VSP Engine backend.
// All endpoints are prefixed with /api (proxied to http://localhost:8000 in dev).

import axios, { type AxiosError } from "axios";
import type {
  Study,
  IslandMeta,
  Roi,
  SegLabel,
  MeshQuality,
  MeshResult,
  ExportFormat,
  ExportLabelConfig,
  TaskResponse,
  JobStatus,
  ApiError,
} from "@/types";
import { logger } from "@/lib/logger";

const BASE_URL = "/api";

const client = axios.create({
  baseURL: BASE_URL,
  withCredentials: true, // HttpOnly session cookie
  headers: { "Content-Type": "application/json" },
});

//  Response interceptor — log errors, re-throw typed 
client.interceptors.response.use(
  (r) => r,
  (err: AxiosError<ApiError>) => {
    const detail = err.response?.data?.detail ?? err.message;
    logger.error("API error", err.config?.url, detail);
    return Promise.reject(err);
  },
);

//  Auth 

/** Issue a new anonymous session (returns session JWT in cookie). */
export async function createSession(): Promise<{ sessionId: string }> {
  const { data } = await client.post<{ session_id: string }>("/auth/session");
  return { sessionId: data.session_id };
}

//  Upload 

/** Upload a file and finalize (triggers scan QC on backend). */
export async function finalizeUpload(file: File): Promise<Study> {
  const formData = new FormData();
  formData.append("file", file);

  const { data } = await client.post<{
    study_id: string;
    nifti_url: string;
    voxel_spacing: [number, number, number];
    dimensions: [number, number, number];
    modality: "CT" | "MR" | "unknown";
    qc_task_id: string | null;
    qc_warnings: string[];
  }>(`/upload/finalize`, formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });

  return {
    studyId: data.study_id,
    niftiUrl: data.nifti_url,
    voxelSpacing: data.voxel_spacing,
    dimensions: data.dimensions,
    modality: data.modality,
    qcWarnings: data.qc_warnings,
  };
}

//  Scout 

/** Trigger scout pass (MeshLib HU threshold  islands). */
export async function triggerScout(studyId: string): Promise<TaskResponse> {
  const { data } = await client.post<{ task_id: string }>("/scout/run", {
    study_id: studyId,
  });
  return { taskId: data.task_id };
}

/** Fetch scout islands after task completes. */
export async function fetchIslands(studyId: string): Promise<readonly IslandMeta[]> {
  const { data } = await client.get<Array<{
    island_id: string;
    centroid_xyz: [number, number, number];
    aabb: { x_min: number; x_max: number; y_min: number; y_max: number; z_min: number; z_max: number };
    ply_url: string;
    voxel_count: number;
  }>>(`/scout/islands/${studyId}`);

  return data.map((d) => ({
    islandId: d.island_id,
    centroidXyz: d.centroid_xyz,
    aabb: {
      xMin: d.aabb.x_min,
      xMax: d.aabb.x_max,
      yMin: d.aabb.y_min,
      yMax: d.aabb.y_max,
      zMin: d.aabb.z_min,
      zMax: d.aabb.z_max,
    },
    plyUrl: d.ply_url,
    voxelCount: d.voxel_count,
  }));
}

//  Segment 

/** Trigger TotalSegmentator on the cropped ROI. */
export async function triggerSegment(
  studyId: string,
  roi: Roi,
  hintLabels: readonly string[],
): Promise<TaskResponse> {
  const { data } = await client.post<{ task_id: string }>("/segment/run", {
    study_id: studyId,
    roi,
    hint_labels: hintLabels,
  });
  return { taskId: data.task_id };
}

/** Fetch AI segment labels after task completes. */
export async function fetchSegLabels(studyId: string): Promise<readonly SegLabel[]> {
  const { data } = await client.get<Array<{
    label_id: number;
    name: string;
    color: string;
    mesh_url: string | null;
  }>>(`/segment/labels/${studyId}`);

  return data.map((d) => ({
    labelId: d.label_id,
    name: d.name,
    color: d.color,
    meshUrl: d.mesh_url,
    isVisible: true,
  }));
}

//  Refine 

/** Trigger SAM-Med3D point refinement. */
export async function triggerRefinePoint(
  studyId: string,
  labelId: number,
  clickXyz: readonly [number, number, number],
): Promise<TaskResponse> {
  const { data } = await client.post<{ task_id: string }>("/refine/point", {
    study_id: studyId,
    label_id: labelId,
    click_xyz: clickXyz,
  });
  return { taskId: data.task_id };
}

/** Trigger MedSAM 2D bbox refinement. */
export async function triggerRefineBbox(
  studyId: string,
  labelId: number,
  sliceIdx: number,
  bbox: readonly [number, number, number, number],
  plane: "axial" | "coronal" | "sagittal",
): Promise<TaskResponse> {
  const { data } = await client.post<{ task_id: string }>("/refine/bbox", {
    study_id: studyId,
    label_id: labelId,
    slice_idx: sliceIdx,
    bbox,
    plane,
  });
  return { taskId: data.task_id };
}

//  Mesh 

/** Trigger mesh generation for selected labels. */
export async function triggerMesh(
  studyId: string,
  labelIds: readonly number[],
  quality: MeshQuality,
  roi: Roi,
): Promise<TaskResponse> {
  const { data } = await client.post<{ task_id: string }>("/mesh/generate", {
    study_id: studyId,
    label_ids: labelIds,
    quality,
    roi,
  });
  return { taskId: data.task_id };
}

/** Fetch mesh result after generation. */
export async function fetchMeshResult(studyId: string): Promise<MeshResult> {
  const { data } = await client.get<{
    mesh_url: string;
    face_count: number;
    is_watertight: boolean;
  }>(`/mesh/result/${studyId}`);
  return {
    meshUrl: data.mesh_url,
    faceCount: data.face_count,
    isWatertight: data.is_watertight,
  };
}

//  Export 

export async function triggerExport(
  studyId: string,
  labels: readonly ExportLabelConfig[],
  format: ExportFormat,
  scaleFactor: number,
): Promise<{ downloadUrl: string }> {
  const { data } = await client.post<{ download_url: string }>("/mesh/export", {
    study_id: studyId,
    labels: labels.map((l) => ({
      label_id: l.labelId,
      included: l.included,
      union: l.union === "combined",
    })),
    format,
    scale_factor: scaleFactor,
  });
  return { downloadUrl: data.download_url };
}

//  Jobs (SSE polling) 

/** Poll job status (fallback if SSE is unavailable). */
export async function pollJob(taskId: string): Promise<JobStatus> {
  const { data } = await client.get<JobStatus>(`/jobs/${taskId}`);
  return data;
}

/** Open an SSE connection for real-time task progress. */
export function subscribeToJob(
  taskId: string,
  onProgress: (status: JobStatus) => void,
  onError: (err: Event) => void,
): () => void {
  const url = `${BASE_URL}/jobs/${taskId}/stream`;
  const es = new EventSource(url, { withCredentials: true });

  es.onmessage = (evt: MessageEvent<string>) => {
    try {
      const status = JSON.parse(evt.data) as JobStatus;
      onProgress(status);
      if (status.state === "SUCCESS" || status.state === "FAILURE") {
        es.close();
      }
    } catch {
      logger.warn("Failed to parse SSE event", evt.data);
    }
  };

  es.onerror = (err) => {
    onError(err);
    es.close();
  };

  return () => es.close();
}

//  Feature flags 

export interface FeatureFlags {
  readonly appendicularBones: boolean;
  readonly medsam2: boolean;
}

export async function fetchFeatureFlags(): Promise<FeatureFlags> {
  const { data } = await client.get<{
    appendicular_bones: boolean;
    medsam2: boolean;
  }>("/config/features");
  return {
    appendicularBones: data.appendicular_bones,
    medsam2: data.medsam2,
  };
}
