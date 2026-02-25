# VSP Engine — Ecosystem Research

> Comprehensive technical research into open-source projects, libraries, and tools
> relevant to medical image segmentation, volumetric visualization, and 3D mesh generation.
> Compiled 2026-02-25. Use this to inform architecture and implementation decisions.

---

## Table of Contents

1. [Landscape Overview](#1-landscape-overview)
2. [AI Segmentation Layer](#2-ai-segmentation-layer)
3. [Mesh Processing Layer](#3-mesh-processing-layer)
4. [Volume Visualization Layer](#4-volume-visualization-layer)
5. [Reference Applications](#5-reference-applications)
6. [Web DICOM Viewers](#6-web-dicom-viewers)
7. [Key Design Decisions Derived From Research](#7-key-design-decisions-derived-from-research)
8. [Workflow Patterns Learned](#8-workflow-patterns-learned)
9. [Critical Warnings and Pitfalls](#9-critical-warnings-and-pitfalls)
10. [Licensing Summary](#10-licensing-summary)

---

## 1. Landscape Overview

There are two distinct architectural approaches to "DICOM → 3D printable mesh":

| Approach | Examples | Pros | Cons |
|---|---|---|---|
| **Client-only (WASM/WebGPU)** | ct2print, brain2print, brainchop | No server cost, data privacy by default, zero server infra | Limited to lightweight models, GPU-dependent models need WebGPU | 
| **Server-side AI + Web frontend** | SlicerTotalSegmentator (desktop), OHIF + AI plugins | Full TotalSegmentator accuracy, 117 classes, GPU-accelerated where available | Infrastructure complexity, needs job queue |

**VSP Engine falls in the server-side category** — surgeons need the full 117-class TotalSegmentator
accuracy across all bone structures (appendicular_bones task includes tibia, fibula, radius, ulna, patella,
carpals, metacarpals, phalanges, tarsals, metatarsals). This cannot be done in-browser.

---

## 2. AI Segmentation Layer

### 2.1 TotalSegmentator (Primary)

**Repo:** `https://github.com/wasserth/TotalSegmentator`  
**Stars:** 2.5k | **License:** Apache-2.0 (free including commercial)  
**Version as of research:** v2.11.0 (Docker), PyPI up to date  

#### What it does
- Segments 117 anatomical structures in CT (task=`total`) or 50 in MR (task=`total_mr`)
- Based on nnU-Net — self-configuring U-Net; no manual tuning needed
- Outputs individual NIfTI mask per class OR combined multilabel NIfTI (`--ml` flag)

#### Key Subtasks for Surgical Planning
```
appendicular_bones  → patella, tibia, fibula, tarsal, metatarsal, phalanges_feet,
                       ulna, radius, carpal, metacarpal, phalanges_hand
vertebrae_body      → vertebral bodies + intervertebral discs
craniofacial_structures → mandible, teeth_lower, skull, sinus_maxillary
headneck_bones_vessels → thyroid cartilage, hyoid, carotid arteries
total               → includes humerus, scapula, clavicula, femur, hip, ribs, sternum, skull
```
> **Note:** `appendicular_bones` is a licensed subtask (free for non-commercial via website form).
> The default `total` task includes humerus, scapula, femur, hip, ribs, skull — sufficient for most orthopedic planning.

#### Python API Pattern
```python
import nibabel as nib
from totalsegmentator.python_api import totalsegmentator

# Input: NIfTI file path or nibabel image object
# Option 1: file paths
totalsegmentator(input_path, output_path, task="total", device="gpu")

# Option 2: nib image objects (preferred for Celery pipeline)
input_img = nib.load(input_path)
output_img = totalsegmentator(input_img, task="total", device="gpu", fast=False)
nib.save(output_img, output_path)

# ROI subset (much faster, saves memory — use when user selects specific bones)
totalsegmentator(input_img, task="total", roi_subset=["femur_left", "femur_right", "hip_left"])
```

#### Resource Requirements
| Mode | GPU Memory | Time (RTX 3090) |
|------|-----------|-----------------|
| Full resolution (1.5mm) | ~7GB | 60-90s |
| Fast (3mm via `--fast`) | ~4GB | 20-30s |
| CPU (`--fast`) | 8GB RAM | 40-50 min |
| CPU (full) | 16GB RAM | several hours |

#### Critical Gotchas Learned from SlicerTotalSegmentator
- **ITK loading error** → pin `SimpleITK==2.0.2` if `ITK ERROR: ITK only supports orthonormal direction cosines`
- **CUDA version must match PyTorch** — `cuNNN` where NNN = CUDA major+minor (e.g. CUDA 11.7 → `cu117`)
- **Memory** — split with `--force_split` only for large images; small images get field-of-view artifacts
- **Weights download** — first run downloads ~GB to `~/.totalsegmentator`; set `TOTALSEG_HOME_DIR` env var
- **Model weights** — pre-download in Docker build: `totalseg_download_weights -t total`
- **Input must be original HU values** — do NOT rescale Hounsfield units before passing to TotalSegmentator

#### What We Learn for VSP Engine
1. Use `roi_subset` when user selects specific bones in the UI → massive runtime reduction
2. First run needs model download → implement a "model ready" check endpoint in API
3. `--fast` flag should be the default for responsive UX; offer "HD" toggle for final export
4. Store weights in a Docker volume, not in the image
5. MR support exists via `total_mr` — expose modality auto-detection (`totalseg_get_modality`)
6. The multilabel output (`--ml`, saves one NIfTI) is better for our pipeline than per-class files

---

### 2.2 nnU-Net (TotalSegmentator Backbone)

**Repo:** `https://github.com/MIC-DKFZ/nnUNet`  
**Stars:** 8.1k | **License:** Apache-2.0  
**Note:** Do NOT use torch ≥ 2.9.0 — severe 3D conv performance regression (use ≤ 2.8.0)

#### Why It Matters for VSP
- TotalSegmentator v2 is built on nnU-Net v2
- nnU-Net auto-configures: 2D U-Net, 3D full-res U-Net, 3D cascade — picks best for dataset
- We do NOT train nnU-Net ourselves for phase 1, but could for custom structures in phase 2+
- The self-configuring strategy means we can fine-tune on our own labeled data later without re-architecture

---

### 2.3 MONAI (Secondary / Fine-tuning Framework)

**Repo:** `https://github.com/Project-MONAI/MONAI`  
**Stars:** 7.9k | **License:** Apache-2.0 | **Version:** 1.5.2  

#### What It Adds Beyond TotalSegmentator
- Flexible transforms pipeline for preprocessing (Spacing, Orientation, ScaleIntensity, CropForeground)
- Model Zoo bundles — pre-trained network configurations in JSON/YAML
- `from monai.bundle import ConfigParser` — load any model zoo bundle declaratively
- Evaluation metrics: Dice, Hausdorff, Surface Distance
- Good for custom fine-tuning on surgeon-specific structures

#### VSP Engine Usage Pattern
```python
# Use MONAI for preprocessing normalization before TotalSegmentator
from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, Orientationd, 
    Spacingd, ScaleIntensityRanged, CropForegroundd
)

# MONAI Model Zoo for specialty models (e.g. prostate, cardiac)
from monai.bundle import ConfigParser
parser = ConfigParser()
parser.read_config(f"{bundle_root}/configs/inference.json")
model = parser.get_parsed_content("network")
```

---

## 3. Mesh Processing Layer

### 3.1 MeshLib (Primary)

**Repo:** `https://github.com/MeshInspector/MeshLib`  
**Stars:** 741 | **Latest:** v3.1.1.92 (weekly releases)  
**Install:** `pip install meshlib`  
**License:** Non-commercial free / Commercial license required  

#### Key Python API Patterns

```python
import meshlib.mrmeshpy as mr

# === DICOM → VDB Volume (direct path, no pydicom needed) ===
dicoms = mr.VoxelsLoad.loadDicomsFolderTreeAsVdb("path/to/dicom/folder")
dicom_obj = dicoms[0]
if dicom_obj:
    vdb_volume = dicom_obj.value().vol
    # vdb_volume is mr.VdbVolume — ready for marching cubes

# === Marching Cubes: VDB Volume → Mesh ===
iso_value = 300  # HU threshold for bone (typical: 200-400 HU)
mesh = mr.marchingCubesFloat(vdb_volume, mr.MarchingCubesParams(iso_value))

# === Mesh Repair (CRITICAL before STL export) ===
mr.fixSelfIntersections(mesh)      # remove self-intersecting triangles
mr.fillHoles(mesh)                  # fill all holes
mr.removeSmallComponents(mesh, 100) # remove noise islands

# === Boolean ROI subtraction ===
box_mesh = mr.makeCube(mr.Box3f(mr.Vector3f(*min_xyz), mr.Vector3f(*max_xyz)))
result = mr.boolean(mesh, box_mesh, mr.BooleanOperation.Intersection)

# === Simplification ===
params = mr.DecimateSettings()
params.maxError = 0.5  # mm tolerance
mr.decimateMesh(result.mesh, params)

# === Watertight check ===
is_watertight = result.mesh.topology.isClosed()

# === Export STL ===
mr.saveMesh(result.mesh, "output.stl")
```

> **Critical:** MeshLib can directly load DICOM folders via `VoxelsLoad.loadDicomsFolderTreeAsVdb` — 
> this bypasses pydicom/SimpleITK for the mesh generation path. However, we still need
> pydicom for DICOM tag reading and anonymization.

#### MeshLib vs Alternatives
| Feature | MeshLib | PyMeshFix | trimesh | Open3D | VTK |
|---|---|---|---|---|---|
| Bool operations | ✅ 10x faster | ❌ | ❌ limited | ❌ | ✅ slow |
| Mesh healing | ✅ comprehensive | ✅ | partial | partial | ✅ |
| DICOM direct load | ✅ | ❌ | ❌ | ❌ | ✅ |
| GPU accel | ✅ CUDA | ❌ | ❌ | ✅ | ❌ |
| Python bindings | ✅ | ✅ | ✅ | ✅ | ✅ |
| License | Non-commercial free | MIT | MIT | MIT | BSD |

---

## 4. Volume Visualization Layer

### 4.1 NiiVue (Primary Volume Renderer)

**Repo:** `https://github.com/niivue/niivue`  
**Stars:** 417 | **Version:** 0.68.1 (weekly releases) | **License:** BSD-2-Clause  
**npm:** `@niivue/niivue`  

#### What It Provides
- WebGL2 volume rendering, MPR (multiplanar reconstruction), 3D surface rendering
- 30+ formats natively: NIfTI, NRRD, MIF, MGH, MRC, AFNI HEAD/BRIK
- DICOM support via `@niivue/dicom` plugin
- Overlays: segmentation masks, statistical maps
- Mesh formats: GIfTI, PLY, MZ3, FreeSurfer, BrainVoyager SRF
- Colormap API, clip planes, oblique reslicing

#### How brain2print/ct2print Use NiiVue (Key Patterns)
```typescript
import { Niivue } from "@niivue/niivue"

// Init (must be in useEffect with cleanup)
const nv = new Niivue({
  show3Dcroshair: true,
  backColor: [0.1, 0.1, 0.1, 1],
  crosshairColor: [0, 1, 0, 1],
})
nv.attachToCanvas(canvasRef.current)
await nv.loadVolumes([{ url: niftiUrl, colormap: "gray" }])

// Cleanup — CRITICAL to avoid GPU memory leaks
return () => nv.dispose()

// Load segmentation overlay
await nv.addVolume({ url: maskUrl, colormap: "warm", opacity: 0.5 })

// Connect NiiVue crosshair click to coordinate display
nv.onLocationChange = (data) => {
  console.log(`HU: ${data.values[0]}, xyz: ${data.mm}`)
}

// Load mesh (STL/PLY from backend)
await nv.loadMeshes([{ url: stlUrl, rgba255: [255, 165, 0, 255] }])
```

#### DICOM Loading Strategy
NiiVue does NOT natively parse DICOM. Two approaches:
1. **Server-side conversion (our approach):** Upload DICOM → `dcm2niix` converts to NIfTI → serve NIfTI URL to NiiVue
2. **Client-side plugin:** `@niivue/dicom` plugin for direct DICOM drag-and-drop (uses WASM)

Our server-side approach is preferred because:
- TotalSegmentator needs NIfTI anyway
- We can anonymize tags at the server boundary
- More control over file handling

---

### 4.2 ITK-Wasm (Secondary — For Lightweight Mesh Generation)

**Repo:** `https://github.com/InsightSoftwareConsortium/ITK-Wasm`  
**Stars:** 224 | **License:** Apache-2.0  
**Used by:** ct2print, brain2print for all voxel-to-mesh operations  

#### What It Provides
- ITK compiled to WebAssembly — runs in browser OR Node.js OR Python
- Voxel-to-mesh (marching cubes), mesh smoothing, mesh simplification
- All image I/O support of ITK
- Web worker pool for async processing

#### When to Use in VSP Engine
- **Don't use for main pipeline** — MeshLib on the backend is 10x faster and more capable
- **Optional client-side preview:** For an instant low-res preview mesh before the server returns the final mesh
- Pattern: `itkwasm` Python package as fallback if MeshLib unavailable

---

## 5. Reference Applications

### 5.1 ct2print (Closest Reference — Simple Version of VSP)

**Repo:** `https://github.com/niivue/ct2print`  
**Live:** https://ct2print.org  
**Stack:** Vite + vanilla JS + NiiVue + ITK-Wasm + niimath (WASM)  
**Architecture:** 100% client-side, zero server  

#### What It Does
1. User drag-drops NIfTI or DICOM files
2. NiiVue displays volume
3. User picks isosurface threshold (Otsu auto-suggested)
4. `Create Mesh` → ITK-Wasm marching cubes → PLY mesh
5. Optional hollow, smooth, simplify
6. `Save Mesh` → STL/OBJ download

#### What VSP Engine Does Differently
| Feature | ct2print | VSP Engine |
|---|---|---|
| Segmentation | HU threshold (user picks) | AI (TotalSegmentator 117 classes) |
| Processing location | Browser (WASM) | Server (CUDA GPU) |
| Bone specificity | All above threshold | Per-bone selection (femur, tibia, etc.) |
| ROI | None | Surgeon-defined bounding box gizmo |
| Export | STL (one blob) | Tagged STL with anatomy labels |
| UI | Single page drag-drop | Guided step workflow |
| Planning | None | Measurement tools, implant overlay |

#### Key Code Insights from ct2print
- Uses `Niivue.onLocationChange` to show HU value when user clicks volume (for threshold guidance)
- Otsu threshold: `nv.getOtsuThreshold()` — excellent default for bone isolation
- ITK-Wasm usage: `itkwasm-mesh-filters` npm package for smooth+simplify
- mesh format: uses PLY internally, converts to STL for download

---

### 5.2 brain2print (AI Segmentation Reference)

**Repo:** `https://github.com/niivue/brain2print`  
**Live:** https://brain2print.org  
**Stack:** Vite + vanilla JS + NiiVue + WebGPU (via brainchop) + ITK-Wasm  
**Published:** Nature Scientific Reports 2025 (DOI: 10.1038/s41598-025-00014-5)  

#### How AI Inference Works Client-Side
```
User drops T1 MRI → NiiVue loads → brainchop WebGPU inference → 
labelmap overlay on NiiVue → ITK-Wasm mesh extraction → 3D preview → STL export
```

- Uses `WebGPU` (browser API) for GPU inference — no server needed
- `brainchop-webworker.js` runs MeshNet model in Web Worker (non-blocking)
- Model: MeshNet — lightweight 3D convnet trained on MNI-conformant T1 MRIs
- **Limitation:** Only works on T1-weighted 1.5-3.0T MRI scans. CT → use ct2print

#### What VSP Engine Learns from brain2print
1. The NiiVue segmentation overlay pattern: load segmentation mask as secondary volume with colormap
2. After AI segmentation → show colorized bone overlay on the CT in the same view
3. "Hollowing" option is useful for 3D printing — reduces material use while maintaining structure
4. The settings dialog pattern: modal/popover with threshold, hollow, smooth, simplify toggles
5. Anatomical structure selection as checkboxes (brain2print: GM/WM/CSF → VSP: per-bone)

---

### 5.3 brainchop (In-Browser AI Reference)

**Repo:** `https://github.com/neuroneural/brainchop`  
**Stars:** 520 | **License:** MIT  

#### Architecture Highlights
- TensorFlow.js model inference in browser — Web Worker + main thread modes
- Models stored as tfjs format in `public/` — served as static files
- NiiVue v4 integrated for 2D/3D MRI display
- Self-contained — no backend server required
- Multi-step tiling for large 3D volumes (patches of 64×64×64)

#### Lesson for VSP
- Celery task progress is equivalent to brainchop's `self.update_state` in JS (Web Worker postMessage)
- The Web Worker pattern maps 1:1 to our SSE streaming pattern for progress updates
- brainchop uses `bwlabels.js` (connected components) to isolate largest cluster → we need this too (remove floating bone fragments)

---

### 5.4 3D Slicer + SlicerTotalSegmentator (Desktop Gold Standard)

**Repos:** Slicer/Slicer + lassoan/SlicerTotalSegmentator  
**Stars:** 2.3k + 242 | C++/Python/VTK/ITK  

#### Workflow That VSP Engine Replicates in Web
```
3D Slicer Workflow:
1. File → Add Data → Load DICOM folder
2. Cropping (optional, Crop Volume module) → reduces memory for large scans
3. TotalSegmentator module → Select task → Apply
4. Show 3D → renders all segments as colored meshes
5. Segment Editor → manual correction tools (paint, erase, scissors)
6. File → Export → Save Segmentation as STL
```

#### Key UI Patterns We Should Steal from 3D Slicer
- **Segmentation task selector** — dropdown of tasks (total, appendicular_bones, vertebrae, etc.)
- **"Fast" toggle** — lower resolution for preview quality
- **Show/Hide per-segment** — toggle visibility of each bone in 3D view
- **Opacity slider per segment** — transparency control for each bone overlay
- **Export selected segments** — checkbox list → export only selected structures

#### SlicerTotalSegmentator GPU Handling (Applies to Our Backend Too)
- Check GPU availability first; fall back to CPU with `--fast`
- Memory error pattern: crop image to ROI first, THEN segment (not after)
- First run downloads ~10GB model weights — handle this as an async progress state in our API
- Model weights cached in `~/.totalsegmentator` → map this to a Docker volume in our deployment

---

## 6. Web DICOM Viewers

### 6.1 OHIF Viewer (Production Reference)

**Repo:** `https://github.com/OHIF/Viewers`  
**Stars:** 4k | **License:** MIT | **Version:** 3.13.0-beta  

#### Architecture Insights
- Monorepo: `extensions/` + `modes/` + `platform/`
- Extension system: each "mode" is a set of extensions + layout config
- Core rendering: Cornerstone3D (WebGL, GPU-accelerated volume rendering)
- DICOMweb integration: WADO-RS, STOW-RS, QIDO-RS
- Features: Volume rendering, MPR, MIP, Segmentation labelmaps, RTSTRUCT

#### What OHIF Uses That Maps to VSP
```
OHIF extension type → VSP Engine equivalent
data-source extension → backend API (FastAPI DICOM Routes)
panel extension (side panel) → ContextOptionsPanel workflow steps
viewport extension → NiiVue canvas + R3F canvas
toolbar extension → VerticalToolbar tools
```

#### Why We Don't Use OHIF Directly
- Overkill for our focused surgical planning workflow
- Cornerstone3D does not integrate with React Three Fiber paradigm
- NiiVue is smaller, simpler, designed for exactly our use case
- OHIF is DICOMweb-server-centric; we want direct DICOM upload

---

## 7. Key Design Decisions Derived From Research

### 7.1 DICOM Ingest Pipeline

**Research Finding:** MeshLib can directly load DICOM (`VoxelsLoad.loadDicomsFolderTreeAsVdb`), but TotalSegmentator needs NIfTI. Therefore the pipeline is:

```
DICOM upload → pydicom (anonymize + metadata) → dcm2niix (DICOM→NIfTI) → 
TotalSegmentator (NIfTI→labelmap NIfTI) → serve NIfTI to NiiVue
```

Then separately for mesh:
```
Selected bones → binary mask NIfTI → MeshLib marching cubes → heal → simplify → STL
```

**Alternative path (MeshLib direct):** Could use MeshLib's DICOM loader for the mesh branch to skip the NIfTI conversion step, loading directly from DICOM to VDB → marching cubes. This is faster but we lose the TotalSegmentator-guided bone separation. Use it for the HU-threshold fallback mode.

### 7.2 Two-Mode Segmentation Strategy

```
Mode 1 (Default): AI-Guided
  DICOM → NIfTI → TotalSegmentator --fast → per-bone NIfTI masks → 
  user selects bones → MeshLib boolean on selected masks → STL

Mode 2 (HU Threshold Fallback — fast preview):
  DICOM → MeshLib VDB → marching cubes at HU ~300 → mesh → STL
  (same as ct2print, no AI needed, runs in <5s)
```

### 7.3 Otsu Threshold for Bone Isolation

**From ct2print:** Always suggest the Otsu threshold as default HU value for bone. CT bone HU typically 400-1800 HU, soft tissue 20-80 HU. Otsu threshold (typically 200-350 HU for bone) is an excellent automated starting point.

The `nv.getOtsuThreshold()` NiiVue API call returns this value — expose it in the UI as a "suggested threshold" hint.

### 7.4 ROI Crop First, Segment Second

**From SlicerTotalSegmentator troubleshooting:** Always crop to region of interest BEFORE running TotalSegmentator. This:
- Reduces memory by 4-8x
- Speeds up inference 2-4x
- Avoids `numpy.core._exceptions._ArrayMemoryError`

Implementation: In our Celery task, add a `--roi_subset` parameter based on what bones the surgeon selects in the UI. This activates TotalSegmentator's internal cropping logic.

### 7.5 Mesh Quality Pipeline (MeshLib Order)

Based on MeshLib documentation, the correct order for mesh repair before STL export:

```python
1. marchingCubesFloat()              # extract isosurface
2. fixSelfIntersections()            # remove bad geometry
3. removeSmallComponents(n)          # cull noise islands  
4. fillHoles()                       # close gaps
5. mr.relaxMesh()                    # light smoothing
6. decimateMesh(params)              # simplify for print
7. topology.isClosed()               # assert watertight — FAIL EXPORT if False
```

### 7.6 Progress Reporting Pattern

From brainchop Web Worker + ct2print architecture:

```python
# Celery task (matches brainchop Web Worker postMessage pattern)
@celery.task(bind=True)
def run_segmentation(self, study_id: str, task_name: str, roi_subset: list[str]):
    self.update_state(state="PROGRESS", meta={"percent": 0, "step": "loading"})
    # ... load DICOM ...
    self.update_state(state="PROGRESS", meta={"percent": 10, "step": "converting"})
    # ... dcm2niix ...
    self.update_state(state="PROGRESS", meta={"percent": 20, "step": "segmenting"})
    # ... totalsegmentator ...
    self.update_state(state="PROGRESS", meta={"percent": 90, "step": "extracting_mesh"})
    # ... meshlib ...
    self.update_state(state="SUCCESS", meta={"percent": 100, "stl_url": url})
```

Frontend polls `/api/jobs/{task_id}` every 1 second with TanStack Query `refetchInterval`.

---

## 8. Workflow Patterns Learned

### 8.1 The 3D Slicer Segmentation Workflow (Industry Standard)

```
LOAD → VIEW → SEGMENT → EDIT → MEASURE → EXPORT
  ↑        ↑       ↑        ↑       ↑          ↑
Upload   NiiVue  TotalSeg  Brush  Ruler      STL/OBJ
```

VSP Engine maps this to workflow steps directly:
| 3D Slicer Step | VSP Step | Panel Content |
|---|---|---|
| Add Data | 1. Upload | Drag-drop zone, DICOM metadata preview |
| Volume Render | 2. Volume View | NiiVue canvas, HU windowing controls |
| TotalSegmentator | 3. Segment | AI task selector, fast/HD toggle, progress bar |
| Segment Editor | 4. Review | Per-bone visibility, opacity, show/hide all |
| Markup ROI | 5. Crop / ROI | R3F bounding box gizmo (Drei Box + TransformControls) |
| Export | 6. Export | STL download with watermark, print settings |

### 8.2 Segmentation Overlay UX (from brain2print)

After AI segmentation completes:
1. Color each bone segment differently (use a medical colormap)
2. Show as semi-transparent overlay on the CT volume in NiiVue
3. Allow toggling each label on/off in the sidebar
4. Click on bone in 3D view → highlight in sidebar list (bidirectional selection)

Pattern: Load the multilabel NIfTI output (`--ml` flag) as a NiiVue overlay volume with opacity 0.5.

### 8.3 Mesh Preview Before Download (from ct2print)

ct2print shows the mesh interactively in the same NiiVue canvas before the user downloads STL.

In VSP Engine: Use React Three Fiber for the mesh preview (separate from NiiVue):
- NiiVue → volume/segmentation overlay view
- R3F Canvas → final 3D mesh preview with orbit controls, zoom, pan
- User can switch between views

### 8.4 Model Download on First Run

SlicerTotalSegmentator and TotalSegmentator both auto-download weights on first run.

**VSP Engine Pattern:**
1. Docker build runs `totalseg_download_weights -t total` → bakes weights into the image or volume
2. Startup health check: verify weights exist → set `segmentation_ready: true` in Redis
3. Frontend shows "AI models loading..." overlay if `segmentation_ready: false`
4. Only show the Segment step after `segmentation_ready: true`

---

## 9. Critical Warnings and Pitfalls

### From TotalSegmentator
- ⚠️ **PyTorch ≥ 2.9.0 breaks nnU-Net** — pin to `torch==2.8.0` or lower
- ⚠️ **SimpleITK version** — pin `SimpleITK==2.0.2` to avoid ITK orthonormal error
- ⚠️ **Input HU must be original** — do NOT rescale HU before inference
- ⚠️ **Patient orientation** — spine must be at bottom in axial view (standardize with `dcm2niix`)
- ⚠️ **appendicular_bones** task requires license — use default `total` or apply for free academic license

### From MeshLib
- ⚠️ **License** — non-commercial free; need paid license for commercial deployment. Check if non-commercial academic + planning-only classification applies
- ⚠️ **Marching cubes threshold** — HU 300 is typical for compact bone; trabecular bone at ~100-200 HU; adjust per scan
- ⚠️ **`topology.isClosed()` check is mandatory** before offering STL download — broken mesh = poor print

### From nnU-Net / Memory
- ⚠️ **Crop before segment** — always ROI crop first (see 7.4)
- ⚠️ **GPU required for prod** — 40-50 min on CPU = unacceptable UX; target CUDA GPU in deployment
- ⚠️ **`--force_split` artifacts** — only use for genuinely large full-body scans, not cropped ROIs

### From NiiVue
- ⚠️ **Always call `nv.dispose()`** on React component unmount — GPU memory leak otherwise
- ⚠️ **WebGL2 required** — check browser support; Safari requires flag in older iOS
- ⚠️ **DICOM drag-drop** — direct DICOM into NiiVue only works with the plugin; standard NiiVue needs NIfTI

### Medical / Safety
- ⚠️ **PHI stripping is non-negotiable** — strip patient name, DOB, MRN before any processing/storage
- ⚠️ **NOT a medical device** — TotalSegmentator authors explicitly state this; we must too
- ⚠️ **Watermark all exports** — "FOR PLANNING PURPOSES ONLY — NOT FOR DIAGNOSTIC USE"

---

## 10. Licensing Summary

| Library | License | Commercial Use |
|---|---|---|
| TotalSegmentator (total, total_mr, body tasks) | Apache-2.0 | ✅ Free |
| TotalSegmentator (appendicular_bones, heartchambers_highres, tissue_types) | Custom (Academic: free, Commercial: contact) | ⚠️ Need license |
| nnU-Net | Apache-2.0 | ✅ Free |
| MONAI | Apache-2.0 | ✅ Free |
| MeshLib | Non-commercial free / Commercial paid | ⚠️ Need commercial license |
| NiiVue | BSD-2-Clause | ✅ Free |
| ITK-Wasm | Apache-2.0 | ✅ Free |
| OHIF Viewer | MIT | ✅ Free |
| 3D Slicer | BSD-like | ✅ Free |
| pydicom | MIT | ✅ Free |
| SimpleITK | Apache-2.0 | ✅ Free |
| dcm2niix | BSD | ✅ Free |
| FastAPI | MIT | ✅ Free |
| Celery | BSD | ✅ Free |

> **Action Required:** For commercial deployment, obtain MeshLib commercial license before launch.
> For academic/research deployment: apply for free academic license at https://backend.totalsegmentator.com/license-academic/
> for the appendicular_bones task if needed.

---

## Appendix: Useful URLs

| Resource | URL |
|---|---|
| TotalSegmentator web demo | https://totalsegmentator.com/ |
| TotalSegmentator structure finder | https://backend.totalsegmentator.com/find-task/ |
| TotalSegmentator Python API source | https://github.com/wasserth/TotalSegmentator/blob/master/totalsegmentator/python_api.py |
| MeshLib Python docs | https://meshlib.io/documentation/Examples.html |
| MeshLib DICOM example | https://meshlib.io/documentation/ExampleDicomFiles.html |
| NiiVue docs + demos | https://niivue.com/docs/ |
| ct2print live demo | https://ct2print.org/ |
| brain2print live demo | https://brain2print.org/ |
| brainchop live demo | https://neuroneural.github.io/brainchop/ |
| OHIF live demo | https://viewer.ohif.org/ |
| MONAI Model Zoo | https://github.com/Project-MONAI/model-zoo |
| nnU-Net docs | https://github.com/MIC-DKFZ/nnUNet/tree/master/documentation |
| 3D Slicer docs | https://slicer.readthedocs.io/en/latest/ |
