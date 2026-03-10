// src/lib/logger.ts
// Thin logger utility — NEVER use console.log/warn/error in production code.
// Use logger.debug/info/warn/error instead.

type LogLevel = "debug" | "info" | "warn" | "error";

const IS_DEV = import.meta.env.DEV;

const LEVEL_COLORS: Record<LogLevel, string> = {
  debug: "color: #6b7280",
  info: "color: #22d3ee",
  warn: "color: #fb923c",
  error: "color: #f87171",
} as const;

function log(level: LogLevel, ...args: unknown[]): void {
  if (level === "debug" && !IS_DEV) return;
  const style = LEVEL_COLORS[level];
  // eslint-disable-next-line no-console
  console[level](`%c[VSP ${level.toUpperCase()}]`, style, ...args);
}

export const logger = {
  debug: (...args: unknown[]) => log("debug", ...args),
  info: (...args: unknown[]) => log("info", ...args),
  warn: (...args: unknown[]) => log("warn", ...args),
  error: (...args: unknown[]) => log("error", ...args),
} as const;
