// src/hooks/useDicomConvert.ts
// In-browser DICOM-to-NIfTI conversion pipeline using @niivue/dcm2niix WASM
// and fflate for ZIP extraction. Runs entirely without a backend.
import { useState, useRef, useCallback } from "react";
import { unzip } from "fflate";
import { logger } from "@/lib/logger";

// ---------------------------------------------------------------------------
// Minimal types for @niivue/dcm2niix (no .d.ts shipped with the package)
// ---------------------------------------------------------------------------

interface Dcm2niixProcessor {
  z(level: string): Dcm2niixProcessor;
  b(value: string): Dcm2niixProcessor;
  f(value: string): Dcm2niixProcessor;
  run(): Promise<File[]>;
}

interface Dcm2niixInstance {
  init(): Promise<boolean>;
  input(files: File[]): Dcm2niixProcessor;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const SKIP_EXTENSIONS = new Set([
  ".txt", ".pdf", ".png", ".jpg", ".jpeg", ".xml",
  ".json", ".html", ".htm", ".csv", ".log",
]);

function isDicomLike(path: string): boolean {
  const lower = path.toLowerCase();
  if (lower.endsWith("/")) return false; // directory entry
  const dotIdx = lower.lastIndexOf(".");
  if (dotIdx === -1) return true; // no extension typically means DICOM
  const ext = lower.slice(dotIdx);
  return ext === ".dcm" || ext === ".ima" || !SKIP_EXTENSIONS.has(ext);
}

type DicomFile = File & { _webkitRelativePath: string };

function makeDicomFile(bytes: Uint8Array, zipPath: string): DicomFile {
  const basename = zipPath.split("/").pop() ?? zipPath;
  const file = new File([bytes], basename, { type: "application/octet-stream" });
  return Object.assign(file, { _webkitRelativePath: zipPath });
}

function unzipAsync(data: Uint8Array): Promise<Record<string, Uint8Array>> {
  return new Promise((resolve, reject) => {
    unzip(data, (err, result) => {
      if (err !== null) reject(err);
      else resolve(result);
    });
  });
}

async function loadDcm2niixClass(): Promise<new () => Dcm2niixInstance> {
  // Dynamic import avoids bundling issues with the WASM worker
  const mod = (await import("@niivue/dcm2niix")) as {
    Dcm2niix: new () => Dcm2niixInstance;
  };
  return mod.Dcm2niix;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

export type ConvertStatus = "idle" | "extracting" | "converting" | "done" | "error";

export interface ConvertResult {
  readonly blobUrl: string;
  readonly fileName: string;
  /** Number of NIfTI series returned by dcm2niix (may be > 1 for multi-echo/multi-series ZIPs) */
  readonly seriesCount: number;
}

interface UseDicomConvertReturn {
  readonly convertStatus: ConvertStatus;
  readonly convertProgress: number;
  readonly convertError: string | null;
  readonly convertZip: (zipFile: File) => Promise<ConvertResult | null>;
  readonly reset: () => void;
}

export function useDicomConvert(): UseDicomConvertReturn {
  const [convertStatus, setConvertStatus] = useState<ConvertStatus>("idle");
  const [convertProgress, setConvertProgress] = useState(0);
  const [convertError, setConvertError] = useState<string | null>(null);

  // Lazy-singleton: init WASM worker once per component lifecycle
  const instanceRef = useRef<Dcm2niixInstance | null>(null);
  const prevBlobRef = useRef<string | null>(null);

  const reset = useCallback(() => {
    const prev = prevBlobRef.current;
    if (prev !== null) {
      URL.revokeObjectURL(prev);
      prevBlobRef.current = null;
    }
    setConvertStatus("idle");
    setConvertProgress(0);
    setConvertError(null);
  }, []);

  const convertZip = useCallback(
    async (zipFile: File): Promise<ConvertResult | null> => {
      // Revoke any previous blob URL to free GPU-side texture memory
      const prev = prevBlobRef.current;
      if (prev !== null) {
        URL.revokeObjectURL(prev);
        prevBlobRef.current = null;
      }

      setConvertStatus("extracting");
      setConvertProgress(5);
      setConvertError(null);

      try {
        // 1  Read ZIP bytes (main thread, blocking but fast for <1GB)
        const buffer = await zipFile.arrayBuffer();
        setConvertProgress(10);
        logger.info("ZIP read into memory", zipFile.name, zipFile.size);

        // 2  Extract ZIP asynchronously (fflate uses a worker internally)
        const extracted = await unzipAsync(new Uint8Array(buffer));
        setConvertProgress(50);
        logger.info("ZIP extracted", Object.keys(extracted).length, "entries");

        // 3  Filter to DICOM-like files; skip empty / metadata entries
        const dicomFiles = Object.entries(extracted)
          .filter(([path, bytes]) => bytes.length > 128 && isDicomLike(path))
          .map(([path, bytes]) => makeDicomFile(bytes, path));

        if (dicomFiles.length === 0) {
          throw new Error("No DICOM files found in the ZIP archive.");
        }
        logger.info("DICOM files to convert", dicomFiles.length);

        // 4  Load and initialise dcm2niix WASM (lazy singleton)
        setConvertStatus("converting");
        setConvertProgress(55);

        if (instanceRef.current === null) {
          const Dcm2niix = await loadDcm2niixClass();
          const instance = new Dcm2niix();
          await instance.init();
          instanceRef.current = instance;
          logger.info("dcm2niix WASM worker initialised");
        }
        setConvertProgress(65);

        // 5  Convert: -z y (gzip), -b n (no BIDS sidecar), -f %p_%t_%s (series-based names)
        const resultFiles: File[] = await instanceRef.current
          .input(dicomFiles)
          .z("y")
          .b("n")
          .f("%p_%t_%s")
          .run();

        setConvertProgress(95);
        logger.info("dcm2niix produced", resultFiles.length, "files");

        // 6  Pick the largest .nii.gz (= main CT volume when multiple series present)
        const niftiFiles = resultFiles.filter(
          (f) => f.name.endsWith(".nii.gz") || f.name.endsWith(".nii"),
        );
        if (niftiFiles.length === 0) {
          throw new Error("dcm2niix produced no NIfTI files. The DICOM data may be unsupported.");
        }

        const largest = niftiFiles.reduce((a, b) => (a.size > b.size ? a : b));
        logger.info("Selected volume", largest.name, largest.size, "bytes");

        const blobUrl = URL.createObjectURL(largest);
        prevBlobRef.current = blobUrl;

        setConvertProgress(100);
        setConvertStatus("done");

        return { blobUrl, fileName: largest.name, seriesCount: niftiFiles.length };
      } catch (err) {
        const msg = err instanceof Error ? err.message : "Conversion failed";
        setConvertError(msg);
        setConvertStatus("error");
        logger.error("DICOM conversion failed", err);
        return null;
      }
    },
    [],
  );

  return { convertStatus, convertProgress, convertError, convertZip, reset };
}