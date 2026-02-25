# VSP Engine — Virtual Surgery Planning Platform

## Architecture, Technology Stack & Build Plan

> **Project Codename:** VSP Engine
> **Target Users:** Surgeons and clinical teams who need to go from a DICOM CT/MR scan to a patient-specific 3D-printable model with minimal friction.

---

## 1. Product Vision

A browser-based clinical tool that:

1. **Ingests** DICOM CT / MR studies (or NIfTI / NRRD equivalents)
2. **Segments** bone (and soft-tissue regions of interest) automatically using AI
3. **Lets the surgeon refine** the segmentation and crop the exact volume to print
4. **Exports** a watertight, print-ready STL / OBJ mesh — correctly scaled, shelled, and repaired

The interface must be operable by a non-technical user (surgeon, radiographer). Zero command-line interaction.

---

## 2. Reference Projects — Key Learnings

> **Full research notes in [RESEARCH.md](RESEARCH.md)** — includes Python API patterns, pitfalls, licensing, and workflow decisions derived from each project.

| Project                            | GitHub                                                                 | What We Learn                                                                                                                                                                                                                                                                                                                                                                                                                  |
| ---------------------------------- | ---------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **niivue/ct2print**                | `niivue/ct2print`                                                      | The closest-to-VSP reference. NiiVue + ITK-Wasm + Otsu threshold. Workflow: drag-drop NIfTI → show volume → isosurface settings dialog → mesh preview → download STL. Pattern: use `nv.getOtsuThreshold()` as default bone HU; hollow + smooth + simplify toggles. We differ by adding AI segmentation and server-side pipeline.                                                                                               |
| **niivue/brain2print**             | `niivue/brain2print`                                                   | AI in-browser version of ct2print. Uses WebGPU brainchop inference + ITK-Wasm mesh. Shows the segmentation-overlay-on-volume UX pattern: load labelmap as second NiiVue volume with 0.5 opacity. Scientific paper: Nature Scientific Reports 2025.                                                                                                                                                                             |
| **neuroneural/brainchop**          | `neuroneural/brainchop`                                                | In-browser MeshNet segmentation via TF.js. Shows Web Worker progress pattern (maps to our Celery SSE), connected components for largest-cluster isolation, per-tissue toggle checkboxes.                                                                                                                                                                                                                                       |
| **TotalSegmentator**               | `wasserth/TotalSegmentator`                                            | Primary AI engine. 117-class CT segmentation via nnU-Net. Python API: `totalsegmentator(input_img, task="total", device="gpu", fast=False)`. Key subtasks for bones: `total` (skull, vertebrae, femur, ribs, sternum, humerus, scapula, clavicula, hip), `appendicular_bones` (tibia, fibula, patella, carpals — requires license). Always use `--fast` for preview, HD for export. Pin `torch<=2.8.0` and `SimpleITK==2.0.2`. |
| **nnU-Net**                        | `MIC-DKFZ/nnUNet`                                                      | TotalSegmentator's backbone. Self-configuring 2D/3D U-Net. Do NOT use torch ≥ 2.9.0 (3D conv regression). Underlies all AI inference — no need to interact with it directly, but understand it for troubleshooting.                                                                                                                                                                                                            |
| **MONAI**                          | `Project-MONAI/MONAI`                                                  | Secondary AI framework. Used for custom transforms preprocessing pipeline, Model Zoo bundles, and evaluation metrics (Dice, Hausdorff). `from monai.bundle import ConfigParser` for loading pre-trained specialty models.                                                                                                                                                                                                      |
| **MeshLib**                        | `MeshInspector/MeshLib`                                                | Primary mesh engine. `pip install meshlib` → `import meshlib.mrmeshpy as mr`. Can load DICOM directly: `mr.VoxelsLoad.loadDicomsFolderTreeAsVdb(folder)`. Mesh pipeline: marchingCubes → fixSelfIntersections → fillHoles → removeSmallComponents → decimateMesh → assert `topology.isClosed()` before export. 10x faster booleans than VTK.                                                                                   |
| **lassoan/SlicerTotalSegmentator** | `lassoan/SlicerTotalSegmentator`                                       | 3D Slicer plugin wrapping TotalSegmentator. Shows production workflow patterns: crop-first-then-segment to avoid OOM, GPU check + fallback, model weight cache management, per-segment show/hide + opacity in 3D view.                                                                                                                                                                                                         |
| **OHIF/Viewers**                   | `OHIF/Viewers`                                                         | Production web medical viewer. Extension + Modes architecture over Cornerstone3D. Reference for DICOMweb integration design patterns. We use NiiVue instead of Cornerstone3D but learn the extension/mode architectural split.                                                                                                                                                                                                 |
| **RapidTool-Fixture**              | Local: `C:\Users\VijayRaghavVarada\Documents\Github\RapidTool-Fixture` | Direct UI source. React 19 + Vite + TypeScript. 4-column AppShell: `w-14` VerticalToolbar → 320px ContextOptionsPanel → flex-1 R3F viewport → 280px properties panel. `tech-glass`/`font-tech`/`tech-transition` CSS utilities, cyan `hsl(198 89% 50%)`, `@rapidtool/cad-ui` (StepProgress, SidebarIcon, ViewCube).                                                                                                            |

---

## 3. System Architecture

```

                        BROWSER (Client)


    React 19 + Vite + TypeScript


      Sidebar /            3D Viewport
      Panel UI       React Three Fiber (R3F) + Drei
      Shadcn/UI
      Tailwind        Volume Layer    Mesh Layer
                      (NiiVue canvas  Three.js geo
      Workflow         embedded /    STL / OBJ
      Steps:           WebGL2 ptr)   + seg masks
      1. Upload
      2. Segment
      3. Crop         Crop/ROI Gizmo  (Drei TransCtrl)
      4. Export


    State: Zustand     API Client: TanStack Query + Axios


                            HTTPS REST / WebSocket

                     BACKEND (Python)

  FastAPI

   /upload          (DICOM series  NIfTI conversion)
        pydicom  +  SimpleITK  +  dcm2niix

   /segment         (AI segmentation task  job ID)
        TotalSegmentator Python API  OR  MONAI Bundle
        Runs async via Celery + Redis

   /mesh/generate   (voxel mask  mesh)
        MeshLib Python: marching-cubes, boolean, heal, simplify

   /mesh/crop       (apply ROI bounding box + boolean subtract)
        MeshLib boolean operations

   /mesh/export     (watertight STL / OBJ / 3MF download)
        MeshLib  +  numpy-stl

   /jobs/{id}       (SSE / WebSocket progress streaming)

  Storage: MinIO or local filesystem (study DICOM  NIfTI  masks)

```

---

## 4. Technology Stack Details

### 4.1 Frontend

| Layer                  | Technology                                             | Rationale                                                                                                                                                            |
| ---------------------- | ------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Framework              | **React 19 + Vite + TypeScript**                       | Fast HMR, modern JSX transforms, strict typing                                                                                                                       |
| 3D Rendering           | **React Three Fiber (R3F) v9 + @react-three/drei**     | Declarative Three.js in React; Drei provides TransformControls, Box3Helper, GizmoHelper, Stats, etc.                                                                 |
| Volume Viewer          | **NiiVue** (embedded via `@niivue/niivue` npm package) | Proven WebGL2 medical volume renderer; DICOM / NIfTI / NRRD support; can coexist with R3F viewport                                                                   |
| UI Components          | **Shadcn/UI + Radix UI + Tailwind CSS v4**             | Accessible, unstyled primitives; carry over the `tech-glass`, `font-tech`, `tech-transition` CSS classes and cyan/dark design tokens directly from RapidTool-Fixture |
| State                  | **Zustand**                                            | Lightweight global store for scan metadata, segmentation masks, mesh state, viewport config                                                                          |
| Data Fetching          | **TanStack Query v5**                                  | Async job polling, cache, optimistic updates                                                                                                                         |
| DICOM Parsing (client) | **dcmjs + dicom-parser**                               | In-browser DICOM tag reading, series grouping before upload                                                                                                          |
| Mesh Preview           | **Three.js STLLoader + OBJLoader**                     | Load backend-generated mesh for interactive R3F preview                                                                                                              |
| Annotation / Drawing   | **NiiVue drawing API**                                 | Freehand mask refinements on MPR slices                                                                                                                              |
| Icons                  | **Lucide React**                                       | Clean medical-style icons                                                                                                                                            |

### 4.2 Backend

| Layer                    | Technology                           | Rationale                                                                                                                                                                                                                                                                                                                                                        |
| ------------------------ | ------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| API Server               | **FastAPI (Python 3.11+)**           | Async, OpenAPI auto-docs, easy streaming endpoints                                                                                                                                                                                                                                                                                                               |
| Task Queue               | **Celery + Redis**                   | Offload heavy AI/mesh jobs; SSE progress back to browser                                                                                                                                                                                                                                                                                                         |
| DICOM I/O                | **pydicom + SimpleITK + dcm2niix**   | Battle-tested DICOM→NIfTI pipeline. **Pin `SimpleITK==2.0.2`** (TotalSegmentator hard requirement)                                                                                                                                                                                                                                                               |
| AI Segmentation          | **TotalSegmentator v2**              | 117-class CT segmentation. Python API: `totalsegmentator(input_img, task="total", roi_subset=[...], fast=True, device="gpu")`. **Pin `torch<=2.8.0`** — nnU-Net has severe 3D conv regression in torch ≥ 2.9.0. `appendicular_bones` subtask requires commercial license; default `total` task covers femur, tibia, humerus, hip, ribs, sternum, skull for free. |
| AI Segmentation Fallback | **MeshLib VDB + Otsu threshold**     | Two-mode pipeline: if AI unavailable or user wants fast preview, load DICOM directly via `mr.VoxelsLoad.loadDicomsFolderTreeAsVdb()` and run marching cubes at Otsu-determined HU threshold. Completes in <5 s vs 30–300 s for AI.                                                                                                                               |
| AI Segmentation (custom) | **MONAI + nnU-Net**                  | For additional fine-grained tasks (craniofacial, teeth, appendicular bones) or fine-tuning on specific anatomy                                                                                                                                                                                                                                                   |
| Mesh Processing          | **MeshLib (Python)**                 | Primary heavy mesh ops: marching-cubes from voxel mask, boolean subtract ROI, mesh heal, simplify, offset for thickness, STL export                                                                                                                                                                                                                              |
| Auxiliary Mesh           | **numpy-stl, trimesh, open3d**       | Format conversions, normal repair, secondary operations                                                                                                                                                                                                                                                                                                          |
| Image Processing         | **scipy, nibabel, skimage**          | Morphological closing/opening on segmentation masks before meshing                                                                                                                                                                                                                                                                                               |
| Storage                  | **MinIO (S3-compatible) / local FS** | Study isolation, job artifact storage                                                                                                                                                                                                                                                                                                                            |
| GPU (optional)           | **NVIDIA CUDA + PyTorch**            | TotalSegmentator/MONAI inference; CPU fallback supported                                                                                                                                                                                                                                                                                                         |

### 4.3 Infrastructure

| Component        | Technology                              |
| ---------------- | --------------------------------------- |
| Containerization | Docker + Docker Compose                 |
| Reverse Proxy    | Nginx                                   |
| GPU Assignment   | NVIDIA Container Toolkit                |
| CI/CD            | GitHub Actions                          |
| Environment      | `.env`-based config, no secrets in code |

---

## 5. UI Layout — Directly Adapted from RapidTool-Fixture

### What RapidTool-Fixture actually is

RapidTool-Fixture is a **React 19 + Vite + TypeScript** web app (not PyQt — it lives in this same repo at `C:\Users\VijayRaghavVarada\Documents\Github\RapidTool-Fixture`). Its `AppShell` uses a **4-column flex layout** inside a full-height shell with a header bar and footer status bar:

1. **Vertical icon toolbar** (`w-14`, `VerticalToolbar`) — one `SidebarIcon` per workflow step, always visible
2. **Collapsible context panel** (320px → 48px, `ContextOptionsPanel`) — step header with large icon + title + `StepProgress` bar, step-specific controls rendered as children, and a step mini-map row of icon buttons at the very bottom
3. **Main R3F viewport** (`flex-1`, `ThreeDViewer`) — full `<Canvas>` with floating tips overlay in top-left corner
4. **Collapsible properties panel** (280px → 48px, `PartPropertiesAccordion`) — accordion sections for the currently selected object

All panels use `tech-glass` (backdrop-blur + semi-transparent border + bg-background/80), `font-tech`, and `tech-transition` Tailwind utilities. Primary accent is **cyan `hsl(198 89% 50%)`**, dark background **`220 13% 8%`**, secondary accent **orange `27 96% 61%`**.

### VSP Engine shell maps to this pattern

```
┌──────────────────────────────────────────────────────────────────────────┐
│  VSP Engine  [Patient-001 / Study]  [Front|Top|Iso btns]  [🌙] [👤]      │  ← header
├────┬─────────────────────────┬──────────────────────────┬────────────────┤
│    │                         │                          │                │
│ 🔼 │  CONTEXT PANEL (320px)  │   3D VIEWPORT (flex-1)   │  PROPERTIES    │
│    │  ┌─────────────────┐    │                          │  (280px)       │
│ 🧩 │  │ Step header     │    │  [NiiVue WebGL2 canvas]  │  ┌──────────┐  │
│    │  │ icon + title    │    │   volume + MPR slices    │  │ Study    │  │
│ 🎯 │  │ + StepProgress  │    │                          │  ├──────────┤  │
│    │  └─────────────────┘    │  overlaid with R3F       │  │ Masks    │  │
│ ✂️  │  ┌─────────────────┐    │  ┌────────────────────┐  │  ├──────────┤  │
│    │  │ Step content    │    │  │ ROI Bounding Box   │  │  │ ROI /    │  │
│ ⚙️  │  │ (per step)      │    │  │ (Drei Transform-   │  │  │ Crop     │  │
│    │  └─────────────────┘    │  │  Controls gizmo)   │  │  ├──────────┤  │
│ 💾 │  ───────────────────    │  └────────────────────┘  │  │ Mesh     │  │
│    │  [🔼 🧩 🎯 ✂️ ⚙️ 💾]    │                          │  └──────────┘  │
│    │  mini-map step row      │  [floating tips overlay] │                │
├────┴─────────────────────────┴──────────────────────────┴────────────────┤
│  Ready  •  WebGL 2.0  •  TotalSegmentator v2  •  ████████░░  87%         │  ← status bar
└──────────────────────────────────────────────────────────────────────────┘
```

### VSP Engine workflow steps (replacing Fixture steps)

| Icon | Step            | Replaces Fixture Step | Content                                                    |
| ---- | --------------- | --------------------- | ---------------------------------------------------------- |
| 🔼   | **Upload**      | Import                | DICOM drag-drop zone, series browser, NIfTI status         |
| 🧩   | **Volume View** | _(new)_               | NiiVue windowing, MPR axes, brightness/contrast/WL         |
| 🎯   | **Segment**     | Supports              | Run AI button, structure checkboxes, mask overlay colors   |
| ✂️   | **Crop / ROI**  | Baseplates            | 3D bounding box gizmo + XYZ numeric inputs                 |
| ⚙️   | **Mesh**        | Cavity                | Quality, shell thickness, simplification sliders           |
| 💾   | **Export**      | Export                | STL/OBJ/3MF format, scale selector, watertight check badge |

### CSS design tokens reused verbatim from RapidTool-Fixture

```css
/* Utility classes carried over from RapidTool-Fixture src/index.css */
.tech-glass {
  backdrop-filter: blur(8px);
  background: hsl(var(--background) / 0.8);
  border: 1px solid hsl(var(--border) / 0.5);
}
.font-tech {
  font-family: "RealityHyper", monospace;
  letter-spacing: 0.02em;
}
.tech-transition {
  transition: var(--transition-tech);
} /* cubic-bezier(0.25, 0.46, 0.45, 0.94) */

/* Color tokens (identical) */
--primary: 198 89% 50%; /* cyan — active step, primary buttons */
--background: 220 13% 8%; /* dark base */
--accent: 27 96% 61%; /* orange — warnings, skipped steps */
--border: 220 13% 18%; /* panel dividers */
```

---

## 6. Data Flow

```
DICOM Series (ZIP / folder drag-drop)


1. CLIENT: dcmjs groups into series, previews thumbnails


2. BACKEND /upload: pydicom validates + SimpleITK converts  NIfTI


3. BACKEND /segment (async Celery job):
      TotalSegmentator  NIfTI label masks (per-bone class)
      MONAI (optional refinement)


4. CLIENT: polling /jobs/{id}  receive mask NIfTI URLs
      NiiVue loads original + mask overlay  colored segmentation view


5. SURGEON: select structures, adjust ROI bounding box in 3D,
            optionally draw freehand corrections on slices


6. BACKEND /mesh/generate (Celery):
      MeshLib: load voxel mask  marching cubes
              morphology close  heal mesh  simplify
              boolean intersect with ROI box
              offset-shell for wall thickness
              make watertight


7. CLIENT: loads STL  R3F mesh preview  surgeon inspects


8. BACKEND /mesh/export: scaled STL / OBJ / 3MF download
```

---

## 7. AI Segmentation Strategy

### Primary: TotalSegmentator

- Install: `pip install TotalSegmentator`
- Python API:
  ```python
  from totalsegmentator.python_api import totalsegmentator
  output_nifti = totalsegmentator(input_nifti, task="total")
  # For appendicular bones (requires license):
  output_nifti = totalsegmentator(input_nifti, task="appendicular_bones")
  # For craniofacial:
  output_nifti = totalsegmentator(input_nifti, task="craniofacial_structures")
  ```
- Bone classes available in `total` task: skull, all vertebrae (C1–L5 + sacrum), ribs (1–12 bilateral), sternum, humerus, scapula, clavicula, femur, hip (pelvis), costal cartilages
- GPU: RTX 3090 ~60s for full-body 1.5mm. CPU with `--fast` flag ~8 min.

### Secondary: MONAI Model Zoo

- Used for anatomy where TotalSegmentator has lower accuracy or for MR input
- Bundle system: `monai.bundle.download(name="wholeBody_ct_segmentation")`
- Custom fine-tuning pipeline for hospital-specific data via MONAI's training workflows

### Threshold Fallback

- For quick previews: Hounsfield Unit-based threshold (bone ~400 HU) using scipy/skimage before AI finishes
- Marching cubes directly on thresholded NIfTI fast approximate mesh for immediate surgeon feedback

---

## 8. Mesh Processing Pipeline (MeshLib)

```python
import meshlib.mrmeshpy as mr

# 1. Load voxel mask (binary NIfTI  SimpleGrid)
vox = mr.loadVoxels("bone_mask.nrrd")

# 2. Marching cubes  mesh
mesh = mr.gridToMesh(vox, isoValue=0.5)

# 3. Heal mesh (fill holes, remove self-intersections)
mr.fixSelfIntersections(mesh)
mr.fillHoles(mesh)

# 4. Smooth
mr.relax(mesh, MeshRelaxParams())

# 5. Simplify (to ~50k faces for web preview)
mr.decimate(mesh, DecimateSettings(maxError=0.3))

# 6. Boolean intersect with ROI box
roi_mesh = mr.makeBox(mr.Box3f(min_pt, max_pt))
result = mr.boolean(mesh, roi_mesh, mr.BooleanOperation.Intersection)

# 7. Shell offset for wall thickness (for hollow surgical guides)
shelled = mr.offsetMesh(result, offset=1.5)  # 1.5mm wall

# 8. Export
mr.saveMesh(result, "output.stl")
```

---

## 9. File Structure

```
vsp-engine/
 frontend/                     # React + Vite + TypeScript
    src/
       app/                  # App shell, routing
       components/
          viewport/         # R3F Canvas, NiiVue integration
             VolumeViewer.tsx
             MeshViewer.tsx
             ROIBox.tsx    # Drei TransformControls crop box
             MPRSlices.tsx
          sidebar/          # Workflow panel steps
             UploadPanel.tsx
             SegmentPanel.tsx
             CropPanel.tsx
             MeshPanel.tsx
             ExportPanel.tsx
          ui/               # Shadcn/UI components
       store/                # Zustand slices
          studyStore.ts
          segmentationStore.ts
          meshStore.ts
       hooks/                # TanStack Query hooks
       lib/                  # dcmjs helpers, API client
       types/
    public/
    index.html
    vite.config.ts
    tailwind.config.ts
    package.json

 backend/                      # Python FastAPI
    app/
       main.py               # FastAPI app entry
       routers/
          upload.py
          segment.py
          mesh.py
          jobs.py
       services/
          dicom_service.py  # pydicom + SimpleITK
          segment_service.py # TotalSegmentator + MONAI
          mesh_service.py   # MeshLib operations
       tasks/                # Celery tasks
          segment_task.py
          mesh_task.py
       models/               # Pydantic schemas
       config.py
    requirements.txt
    Dockerfile

 docker/
    docker-compose.yml
    docker-compose.gpu.yml    # GPU override
    nginx.conf

 .env.example
 .github/
    workflows/
        ci.yml
 README.md
```

---

## 10. Build Plan — Phased Execution

### Phase 1: Foundation (Week 1–2)

- [ ] Monorepo setup: `pnpm workspace` + Python `uv` environment
- [ ] Frontend scaffold: Vite + React + TypeScript + Tailwind + Shadcn/UI init
- [ ] Backend scaffold: FastAPI + Celery + Redis docker-compose
- [ ] Basic DICOM upload endpoint: pydicom ingestion NIfTI conversion
- [ ] File storage service (local FS with MinIO abstraction)
- [ ] DICOM drop zone + series browser in frontend

### Phase 2: Volume Viewer (Week 2–3)

- [ ] Integrate NiiVue into the React viewport (canvas composition with R3F overlay)
- [ ] Load backend NIfTI via URL into NiiVue volume renderer
- [ ] MPR panel: axial / coronal / sagittal slice display
- [ ] Brightness/contrast/windowing controls (Shadcn sliders)
- [ ] Zustand store for viewer state

### Phase 3: AI Segmentation (Week 3–5)

- [ ] TotalSegmentator Celery task: `total` task for bones
- [ ] Job progress streaming via SSE frontend progress bar
- [ ] NiiVue overlay: load segmentation NIfTI as color mask on volume
- [ ] Structure selector UI: checkboxes for bone classes (ribs, vertebrae, femur, etc.)
- [ ] HU-threshold fallback mesh for immediate preview (< 5 seconds)
- [ ] MONAI integration for craniofacial / appendicular subtasks

### Phase 4: ROI Selection & Mesh Generation (Week 5–7)

- [ ] R3F ROI bounding box: Drei `TransformControls` + box wireframe gizmo
- [ ] Crop panel: numeric XYZ inputs synced with 3D gizmo
- [ ] Backend `/mesh/generate`: MeshLib marching cubes on selected label mask
- [ ] Mesh healing pipeline: fill holes, remove self-intersections, smooth
- [ ] Mesh simplification: quality slider decimation param
- [ ] STL/OBJ download from backend

### Phase 5: Mesh Preview & Refinement (Week 7–8)

- [ ] Frontend STL loader R3F mesh display with lighting / matcap shader
- [ ] Mesh quality inspector: face count, volume, watertight check
- [ ] Wall thickness / shell offset control (for surgical guides / implants)
- [ ] Scale selection (1:1, 1:2, 1:4) for export
- [ ] Manual mesh correction tools: smoothing brush, fill hole button

### Phase 6: Polish & Clinical UX (Week 8–10)

- [ ] Study management: patient list, past exports
- [ ] Error handling: corrupt DICOM, insufficient image quality warnings
- [ ] Accessibility: keyboard navigation, screen reader labels
- [ ] Performance: WebWorker for in-browser DICOM parsing, lazy R3F chunks
- [ ] Docs: user guide PDF for surgeons
- [ ] Docker Compose GPU single-command deploy
- [ ] Unit + integration tests (Vitest frontend, pytest backend)

---

## 11. Agent & Workspace Setup Recommendations

### VS Code Extensions to Install

```
ms-python.python
ms-python.vscode-pylance
ms-vscode.vscode-typescript-next
bradlc.vscode-tailwindcss
esbenp.prettier-vscode
dbaeumer.vscode-eslint
ms-azuretools.vscode-docker
eamodio.gitlens
GitHub.copilot
GitHub.copilot-chat
```

### Workspace Settings (`.vscode/settings.json`)

```json
{
  "editor.formatOnSave": true,
  "editor.defaultFormatter": "esbenp.prettier-vscode",
  "[python]": { "editor.defaultFormatter": "ms-python.black-formatter" },
  "python.analysis.typeCheckingMode": "strict",
  "typescript.preferences.importModuleSpecifier": "relative",
  "tailwindCSS.experimental.classRegex": [
    ["cva\\(([^)]*)\\)", "[\"'`]([^\"'`]*).*?[\"'`]"]
  ]
}
```

### Copilot Custom Instructions (`.github/copilot-instructions.md`)

- Always use TypeScript strict mode
- Follow React Three Fiber patterns (declarative JSX geometry)
- Use Zustand slices for state; avoid Redux
- MeshLib Python: always use `mr.` prefix (meshlib.mrmeshpy as mr)
- Medical data: NEVER log patient identifiers; anonymize at ingestion
- Celery tasks: always return progress updates via backend state store

### Environment Variables (`.env.example`)

```env
# Backend
REDIS_URL=redis://localhost:6379
STORAGE_PATH=./data/studies
TOTALSEG_HOME_DIR=./models/totalsegmentator
MONAI_BUNDLE_DIR=./models/monai
GPU_DEVICE=cuda:0  # or cpu

# Frontend
VITE_API_BASE_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000
```

---

## 12. Key Design Decisions & Rationale

| Decision                    | Choice                            | Why                                                                                                                                                                                                             |
| --------------------------- | --------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Volume rendering in browser | **NiiVue** (WebGL2, not pure R3F) | 10+ years of proven medical volume rendering; DICOM/NIfTI native; far superior to naive Three.js volume shaders                                                                                                 |
| 3D mesh interaction         | **React Three Fiber**             | Declarative, composable; Drei provides production-ready gizmos, transform controls, orbit controls                                                                                                              |
| UI library                  | **Shadcn/UI + Tailwind**          | Reuse RapidTool-Fixture's exact 4-column AppShell pattern, `tech-glass`/`font-tech`/`tech-transition` utilities, and cyan primary design tokens — gives VSP Engine a consistent, proven surgical-tool aesthetic |
| AI segmentation             | **TotalSegmentator first**        | 117 classes, clinical-grade accuracy, Apache-2.0 license, pip-installable, MONAI-compatible; GPU inference ~60s                                                                                                 |
| Mesh backend                | **MeshLib**                       | Explicitly designed for medical device / 3D printing use cases (dental/orthopedic customers); boolean + healing + offsetting all in one Python library                                                          |
| Task queue                  | **Celery + Redis**                | AI inference and mesh generation are CPU/GPU bound and take 30s–5min; must not block the API server                                                                                                             |
| State management            | **Zustand**                       | Lightweight, no boilerplate, plays well with R3F's state model                                                                                                                                                  |

---

## 13. Regulatory Considerations

> **This is a research/planning tool, not a certified medical device (yet).** TotalSegmentator explicitly states it is "not a medical device." Any clinical use requires additional validation.

- Label all outputs with: "FOR PLANNING PURPOSES ONLY — NOT FOR DIAGNOSTIC USE"
- Anonymize DICOM data at upload (remove patient PHI from tags)
- Audit trail: log all segmentation runs and mesh exports with timestamps
- Plan for FDA 510(k) / CE marking pathway if commercializing

---

## 14. Immediate Next Steps

1. **Initialize monorepo** with `pnpm` workspaces + Python `uv`
2. **Scaffold frontend** with `pnpm create vite frontend -- --template react-ts`
3. **Scaffold backend** with FastAPI + Celery + Redis docker-compose
4. **Install and verify** TotalSegmentator on a test CT NIfTI file
5. **Prototype NiiVue** integration inside a React functional component
6. **Wire up** first complete path: DICOM upload NIfTI NiiVue display
