// src/lib/utils.ts
import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}

/** Clamp a number to [min, max]. */
export function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

/** Format bytes as human-readable string. */
export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

/** Format a duration in seconds as mm:ss. */
export function formatDuration(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}

/** Generate a random hex color from a deterministic index. */
export function islandColor(index: number): string {
  const PALETTE = [
    "#22d3ee", "#f472b6", "#a78bfa", "#34d399", "#fb923c",
    "#60a5fa", "#facc15", "#f87171", "#4ade80", "#c084fc",
    "#38bdf8", "#fb7185", "#a3e635", "#fbbf24", "#2dd4bf",
    "#e879f9", "#818cf8", "#ec4899", "#10b981", "#f59e0b",
  ] as const;
  return PALETTE[index % PALETTE.length] ?? "#22d3ee";
}
