from __future__ import annotations
import logging
from dataclasses import dataclass
import numpy as np
import nibabel as nib
logger = logging.getLogger(__name__)
_MAX_SLICE_THICKNESS_MM = 3.0
@dataclass
class QcResult:
    passed: bool
    warnings: list[str]
def run_scan_qc(nifti_path: str) -> QcResult:
    """Check slice thickness, HU range, isotropy. Returns non-blocking warnings."""
    warnings: list[str] = []
    img = nib.load(nifti_path)
    header = img.header
    pixdim = header.get_zooms()
    slice_thickness = float(pixdim[2]) if len(pixdim) > 2 else 0.0
    if slice_thickness > _MAX_SLICE_THICKNESS_MM:
        warnings.append(f"Slice thickness {slice_thickness:.1f}mm > {_MAX_SLICE_THICKNESS_MM}mm — segmentation quality may be reduced")
    vox_sizes = [float(p) for p in pixdim[:3]]
    if max(vox_sizes) / (min(vox_sizes) + 1e-6) > 3.0:
        warnings.append(f"Non-isotropic voxels {vox_sizes} — AI segmentation may be less accurate")
    data = np.asarray(img.dataobj)
    hu_min = float(data.min())
    hu_max = float(data.max())
    if hu_min >= -100 or hu_max <= 100:
        warnings.append(f"Unusual HU range [{hu_min:.0f}, {hu_max:.0f}] — volume may not be a CT scan")
    logger.info("QC nifti=%s warnings=%d", nifti_path, len(warnings))
    return QcResult(passed=len(warnings) == 0, warnings=warnings)
