// src/components/viewport/ViewportScene.tsx
// Main viewport: NiiVue handles volume rendering + MPR natively.
// R3F canvas is only mounted when downstream steps need mesh overlays
// (island selection, crop box, implant positioning, measurements).
import React, { Suspense, useRef } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls, GizmoHelper, GizmoViewport, Stats } from "@react-three/drei";
import NiiVueCanvas from "./NiiVueCanvas";
import CropBox from "./CropBox";
import IslandMesh from "./IslandMesh";
import { useUploadStore } from "@/store/uploadStore";
import { useScoutStore } from "@/store/scoutStore";
import { useUiStore } from "@/store/uiStore";

const ViewportScene: React.FC = () => {
  const niftiUrl = useUploadStore((s) => s.study?.niftiUrl ?? null);
  const volumeName = useUploadStore((s) => s.localFileName ?? undefined);
  const islands = useScoutStore((s) => s.islands);
  const roi = useScoutStore((s) => s.roi);
  const activeStep = useUiStore((s) => s.activeStep);
  const nvReadyRef = useRef<import("@niivue/niivue").Niivue | null>(null);

  const showCropBox = (activeStep === "select" || activeStep === "segment") && roi !== null;
  const showIslands = islands.length > 0 && activeStep !== "upload";

  // Only mount R3F when there are meshes or gizmos to render.
  // NiiVue handles all volume rendering, MPR slices, and 3D bone views natively.
  const showR3fOverlay = showIslands || showCropBox;

  return (
    <div className="relative w-full h-full bg-[hsl(220_13%_5%)]">
      {/* NiiVue: volume rendering + MPR slices — always active */}
      <div className="absolute inset-0">
        <NiiVueCanvas
          niftiUrl={niftiUrl}
          volumeName={volumeName}
          onNvReady={(nv) => { nvReadyRef.current = nv; }}
        />
      </div>

      {/* R3F: mesh overlays — only mounted when needed (select/segment/refine/export) */}
      {showR3fOverlay && (
        <div className="absolute inset-0">
          <Canvas
            camera={{ position: [0, 0, 600], fov: 45, near: 1, far: 5000 }}
            gl={{ antialias: true, alpha: true }}
            aria-label="3D anatomy viewport"
            role="img"
          >
            <ambientLight intensity={0.4} />
            <directionalLight position={[200, 400, 300]} intensity={0.8} castShadow={false} />
            <directionalLight position={[-200, -200, 200]} intensity={0.3} />

            <OrbitControls makeDefault enableDamping dampingFactor={0.1} />

            <GizmoHelper alignment="bottom-right" margin={[60, 60]}>
              <GizmoViewport axisColors={["#ef4444", "#22c55e", "#3b82f6"]} labelColor="white" />
            </GizmoHelper>

            {showIslands && (
              <Suspense fallback={null}>
                {islands.map((island, idx) => (
                  <IslandMesh key={island.islandId} island={island} colorIndex={idx} />
                ))}
              </Suspense>
            )}

            {showCropBox && <CropBox />}

            {import.meta.env.DEV && <Stats />}
          </Canvas>
        </div>
      )}
    </div>
  );
};

export default ViewportScene;
