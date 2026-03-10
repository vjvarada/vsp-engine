from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
RasPoint = tuple[float, float, float]
VoxelPoint = tuple[int, int, int]
SamPoint = tuple[float, float, float]
def voxel_to_ras(voxel: VoxelPoint, affine: NDArray[np.float64]) -> RasPoint:
    """Convert voxel index (i,j,k) to NIfTI RAS mm using the volume affine."""
    v = np.array([*voxel, 1.0], dtype=np.float64)
    ras = affine @ v
    return (float(ras[0]), float(ras[1]), float(ras[2]))
def ras_to_voxel(ras: RasPoint, affine_inv: NDArray[np.float64]) -> VoxelPoint:
    """Convert NIfTI RAS mm to voxel index."""
    r = np.array([*ras, 1.0], dtype=np.float64)
    v = affine_inv @ r
    return (int(round(v[0])), int(round(v[1])), int(round(v[2])))
def nifti_voxel_to_sam_med3d_point(
    click_ras: RasPoint,
    patch_origin_ras: RasPoint,
    patch_size_mm: float = 128.0,
) -> SamPoint:
    """Convert a surgeon 3-D click (RAS mm) to SAM-Med3D normalised (z,y,x) format."""
    dx = click_ras[0] - patch_origin_ras[0]
    dy = click_ras[1] - patch_origin_ras[1]
    dz = click_ras[2] - patch_origin_ras[2]
    x_norm = float(np.clip(dx / patch_size_mm, 0.0, 1.0))
    y_norm = float(np.clip(dy / patch_size_mm, 0.0, 1.0))
    z_norm = float(np.clip(dz / patch_size_mm, 0.0, 1.0))
    return (z_norm, y_norm, x_norm)
