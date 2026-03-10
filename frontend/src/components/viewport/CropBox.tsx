// src/components/viewport/CropBox.tsx
// Custom 6-handle crop box for ROI selection.
// Each handle is constrained to exactly one axis.
// DO NOT use Drei TransformControls here — that moves the whole box.
import React, { useCallback, useRef } from "react";
import { ThreeEvent } from "@react-three/fiber";
import * as THREE from "three";
import { useScoutStore } from "@/store/scoutStore";
import { clamp } from "@/lib/utils";

const MIN_ROI_SIZE_MM = 10; // minimum 10mm per axis

//  Handle directions 

type HandleId = "xMin" | "xMax" | "yMin" | "yMax" | "zMin" | "zMax";

interface HandleConfig {
  readonly id: HandleId;
  readonly axis: 0 | 1 | 2;
  readonly direction: 1 | -1;
  readonly color: string;
  readonly ariaLabel: string;
}

const HANDLES: readonly HandleConfig[] = [
  { id: "xMin", axis: 0, direction: -1, color: "#ef4444", ariaLabel: "X minus face handle" },
  { id: "xMax", axis: 0, direction:  1, color: "#ef4444", ariaLabel: "X plus face handle" },
  { id: "yMin", axis: 1, direction: -1, color: "#22c55e", ariaLabel: "Y minus face handle" },
  { id: "yMax", axis: 1, direction:  1, color: "#22c55e", ariaLabel: "Y plus face handle" },
  { id: "zMin", axis: 2, direction: -1, color: "#3b82f6", ariaLabel: "Z minus face handle" },
  { id: "zMax", axis: 2, direction:  1, color: "#3b82f6", ariaLabel: "Z plus face handle" },
] as const;

//  Handle Component 

interface FaceHandleProps {
  readonly config: HandleConfig;
  readonly position: THREE.Vector3;
  readonly isLocked: boolean;
}

const FaceHandle: React.FC<FaceHandleProps> = React.memo(({ config, position, isLocked }) => {
  const updateRoiAxis = useScoutStore((s) => s.updateRoiAxis);
  const roi = useScoutStore((s) => s.roi);
  const isDraggingRef = useRef(false);
  const startMouseRef = useRef(0);
  const startValueRef = useRef(0);

  const handlePointerDown = useCallback(
    (e: ThreeEvent<PointerEvent>) => {
      if (isLocked || roi === null) return;
      e.stopPropagation();
      isDraggingRef.current = true;
      startMouseRef.current = e.clientX;
      startValueRef.current = roi[config.id];
      (e.target as HTMLElement).setPointerCapture?.(e.pointerId);
    },
    [config.id, isLocked, roi],
  );

  const handlePointerMove = useCallback(
    (e: ThreeEvent<PointerEvent>) => {
      if (!isDraggingRef.current || roi === null) return;
      const delta = (e.clientX - startMouseRef.current) * config.direction * 0.5;
      const raw = startValueRef.current + delta;

      // Clamp: min face must be < max face - MIN_ROI_SIZE_MM
      let clamped: number;
      if (config.id === "xMin") clamped = clamp(raw, -10000, roi.xMax - MIN_ROI_SIZE_MM);
      else if (config.id === "xMax") clamped = clamp(raw, roi.xMin + MIN_ROI_SIZE_MM, 10000);
      else if (config.id === "yMin") clamped = clamp(raw, -10000, roi.yMax - MIN_ROI_SIZE_MM);
      else if (config.id === "yMax") clamped = clamp(raw, roi.yMin + MIN_ROI_SIZE_MM, 10000);
      else if (config.id === "zMin") clamped = clamp(raw, -10000, roi.zMax - MIN_ROI_SIZE_MM);
      else clamped = clamp(raw, roi.zMin + MIN_ROI_SIZE_MM, 10000);

      updateRoiAxis(config.id, clamped);
    },
    [config.direction, config.id, roi, updateRoiAxis],
  );

  const handlePointerUp = useCallback(() => {
    isDraggingRef.current = false;
  }, []);

  return (
    <mesh
      position={position}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
      userData={{ ariaLabel: config.ariaLabel }}
    >
      <sphereGeometry args={[4, 8, 8]} />
      <meshStandardMaterial
        color={config.color}
        emissive={config.color}
        emissiveIntensity={isLocked ? 0.1 : 0.4}
        opacity={isLocked ? 0.4 : 0.9}
        transparent
      />
    </mesh>
  );
});
FaceHandle.displayName = "FaceHandle";

//  CropBox Component 

const CropBox: React.FC = () => {
  const roi = useScoutStore((s) => s.roi);
  const isLocked = useScoutStore((s) => s.isRoiLocked);

  if (roi === null) return null;

  const cx = (roi.xMin + roi.xMax) / 2;
  const cy = (roi.yMin + roi.yMax) / 2;
  const cz = (roi.zMin + roi.zMax) / 2;
  const sx = roi.xMax - roi.xMin;
  const sy = roi.yMax - roi.yMin;
  const sz = roi.zMax - roi.zMin;

  const handlePositions: Record<HandleId, THREE.Vector3> = {
    xMin: new THREE.Vector3(roi.xMin, cy, cz),
    xMax: new THREE.Vector3(roi.xMax, cy, cz),
    yMin: new THREE.Vector3(cx, roi.yMin, cz),
    yMax: new THREE.Vector3(cx, roi.yMax, cz),
    zMin: new THREE.Vector3(cx, cy, roi.zMin),
    zMax: new THREE.Vector3(cx, cy, roi.zMax),
  };

  return (
    <group>
      {/* Wireframe box */}
      <mesh position={[cx, cy, cz]}>
        <boxGeometry args={[sx, sy, sz]} />
        <meshBasicMaterial
          color={isLocked ? "#fb923c" : "#22d3ee"}
          wireframe
          opacity={0.6}
          transparent
        />
      </mesh>

      {/* 6 face handles */}
      {HANDLES.map((cfg) => {
        const pos = handlePositions[cfg.id];
        if (pos === undefined) return null;
        return (
          <FaceHandle
            key={cfg.id}
            config={cfg}
            position={pos}
            isLocked={isLocked}
          />
        );
      })}
    </group>
  );
};

export default CropBox;
