// src/components/viewport/NiiVueCanvas.tsx
// Wraps NiiVue in a React component.
// NOTE: Do NOT call loseContext() in cleanup — it permanently kills the canvas
// WebGL context and prevents React 19 StrictMode from remounting correctly.
import React, { useEffect, useRef } from "react";
import { Niivue } from "@niivue/niivue";
import { logger } from "@/lib/logger";
import { useNiivueStore } from "@/store/niivueStore";

interface NiiVueCanvasProps {
  readonly niftiUrl: string | null;
  /**
   * Original filename hint (e.g. "skull.nii.gz") — required when niftiUrl is a
   * blob: URL so NiiVue can detect the format from the extension.
   */
  readonly volumeName?: string;
  readonly onNvReady?: (nv: Niivue) => void;
}

const NiiVueCanvas: React.FC<NiiVueCanvasProps> = ({ niftiUrl, volumeName, onNvReady }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const nvRef = useRef<Niivue | null>(null);
  const setNv = useNiivueStore((s) => s.setNv);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (canvas === null) return;

    let cancelled = false;

    const nv = new Niivue({
      show3Dcrosshair: true,
      backColor: [0.07, 0.08, 0.1, 1],
      crosshairColor: [0.13, 0.83, 0.94, 0.8],
    });
    nvRef.current = nv;

    void nv.attachToCanvas(canvas).then(() => {
      if (cancelled) return;
      logger.info("NiiVue attached to canvas");
      setNv(nv);
      onNvReady?.(nv);
    }).catch((err: unknown) => {
      if (cancelled) return;
      logger.error("NiiVue attachToCanvas failed", err);
    });

    return () => {
      cancelled = true;
      setNv(null);
      nvRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const nv = nvRef.current;
    if (nv === null || niftiUrl === null) return;

    const volumeEntry = volumeName !== undefined
      ? { url: niftiUrl, name: volumeName }
      : { url: niftiUrl };

    void (async () => {
      try {
        await nv.loadVolumes([volumeEntry]);
        logger.info("NiiVue volume loaded", volumeName ?? niftiUrl);
      } catch (err) {
        logger.error("NiiVue load failed", err);
      }
    })();
  }, [niftiUrl, volumeName]);

  return (
    <canvas
      ref={canvasRef}
      className="block w-full h-full"
      aria-label="Medical volume viewer — MPR slices"
    />
  );
};

export default NiiVueCanvas;