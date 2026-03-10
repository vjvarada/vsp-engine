// src/components/viewport/IslandMesh.tsx
// Renders a single scout island PLY mesh in R3F.
import React, { useEffect, useRef, useCallback } from "react";
import { ThreeEvent, useLoader } from "@react-three/fiber";
import * as THREE from "three";
import { PLYLoader } from "three/examples/jsm/loaders/PLYLoader.js";
import { useScoutStore } from "@/store/scoutStore";
import type { IslandMeta } from "@/types";
import { islandColor } from "@/lib/utils";

interface IslandMeshProps {
  readonly island: IslandMeta;
  readonly colorIndex: number;
}

const IslandMesh: React.FC<IslandMeshProps> = React.memo(({ island, colorIndex }) => {
  const geometry = useLoader(PLYLoader, island.plyUrl);
  const selectedIslandIds = useScoutStore((s) => s.selectedIslandIds);
  const toggleIsland = useScoutStore((s) => s.toggleIsland);

  const isSelected = selectedIslandIds.has(island.islandId);
  const color = islandColor(colorIndex);

  const meshRef = useRef<THREE.Mesh>(null);
  const materialRef = useRef<THREE.MeshStandardMaterial>(null);

  // Dispose geometry and material on unmount
  useEffect(() => {
    const geo = meshRef.current?.geometry;
    const mat = materialRef.current;
    return () => {
      geo?.dispose();
      mat?.dispose();
    };
  }, []);

  const handleClick = useCallback(
    (e: ThreeEvent<MouseEvent>) => {
      e.stopPropagation();
      toggleIsland(island.islandId);
    },
    [island.islandId, toggleIsland],
  );

  return (
    <mesh
      ref={meshRef}
      geometry={geometry}
      onClick={handleClick}
      userData={{ islandId: island.islandId }}
    >
      <meshStandardMaterial
        ref={materialRef}
        color={color}
        emissive={isSelected ? color : "#000000"}
        emissiveIntensity={isSelected ? 0.4 : 0}
        transparent
        opacity={isSelected ? 0.85 : 0.6}
        side={THREE.DoubleSide}
      />
    </mesh>
  );
});
IslandMesh.displayName = "IslandMesh";

export default IslandMesh;
