# VSP Engine — Virtual Surgery Planning Platform

## Architecture, Technology Stack & Build Plan

> **Project Codename:** VSP Engine
> **Target Users:** Surgeons and clinical teams who need to go from a DICOM CT/MR scan to a patient-specific 3D-printable model with minimal friction.

---

## 1. Product Vision

A browser-based clinical tool that:

1. **Ingests** DICOM CT / MR studies (or NIfTI / NRRD equivalents)
2. **Scouts** immediately — a fast HU-threshold pass produces per-bone connected-component islands the surgeon can click (<10 s, no GPU needed)
3. **Focuses** — the surgeon selects the anatomy they need; the system auto-computes a tight ROI crop
4. **Segments accurately** — AI (TotalSegmentator) runs only on the small ROI crop (10–58× faster, more accurate, less GPU memory)
5. **Refines interactively** — surgeon can fix any mislabeled region with a single 3D click (SAM-Med3D-turbo) or 2D bounding box (MedSAM)
6. **Exports** a watertight, print-ready STL / OBJ mesh — correctly scaled, repaired, and annotated

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
| **VISTA3D-CT / VISTA3D-CTMR**      | `Project-MONAI/VISTA` — HuggingFace: `nvidia/NV-Segment-CT` / `nvidia/NV-Segment-CTMR` | NVIDIA CVPR 2025 foundation model. 127 CT classes (auto + interactive point prompts). VISTA3D-CTMR adds MRI support, 345 total classes. **⚠️ NVIDIA OneWay Noncommercial License — NOT for production.** Learn: same MONAI Bundle inference.json pattern works for both auto label_prompt and interactive `points`/`point_labels` API; ShapeKit postprocessing for connected-component filtering. |
| **bowang-lab/MedSAM**              | `bowang-lab/MedSAM`                                                    | SAM-ViT-B fine-tuned on 1.5M medical image-mask pairs (10 modalities). Nature Communications 2024. **Apache 2.0 ✅.** Input: 2D bounding box → binary mask. Use for any anatomy TotalSegmentator doesn't cover. LiteMedSAM variant runs on CPU. Pattern: surgeon draws bbox on NiiVue MPR slice → `[x1,y1,x2,y2]` + slice_idx → backend MedSAM inference → stack masks → 3D volume. |
| **bowang-lab/MedSAM2**             | `bowang-lab/MedSAM2`                                                   | SAM2 for 3D CT + video segmentation. arXiv April 2025. **Apache 2.0 ✅.** Treats CT slices as video → propagates from one marked slice to full 3D structure (RECIST marker workflow). ⚠️ Requires `torch==2.5.1` — isolate in separate Docker container. Efficient MedSAM2 variant for CPU inference. Best for tubular/elongated structures. |
| **uni-medical/SAM-Med3D**          | `uni-medical/SAM-Med3D`                                                | Fully 3D SAM. ECCV 2024 Oral. **Apache 2.0 ✅.** Trained on SA-Med3D-140K (143K 3D masks, 245 categories). SAM-Med3D-turbo checkpoint. 10–100× fewer prompts than 2D SAM. One-liner: `medim.create_model("SAM-Med3D", pretrained=True)`. Input: 3D point click from R3F viewport → binary 3D mask. Best choice for interactive intraoperative refinement. |
| **MrGiovanni/SuPreM**              | `MrGiovanni/SuPreM`                                                    | Supervised pre-training on AbdomenAtlas 1.1 (9,262 CTs, 25 classes). ICLR 2024 Oral. SwinUNETR/U-Net/SegResNet backbone checkpoints. Use as fine-tuning initialization for new bone segmentation tasks: converges 5–10× faster than random init. ⚠️ "Patents pending" — verify license before commercial use. |
| **uni-medical/STU-Net**            | `uni-medical/STU-Net`                                                  | Scalable U-Net (14M–1.4B params) built on nnU-Net v2. **Apache 2.0 ✅.** Pre-trained on TotalSegmentator atlas: 27 organs + **59 bones** + 10 muscles + 8 vessels. Won MICCAI 2023 ATLAS + SPPIN challenges. STU-Net-B (58M, 10 GB GPU) is the practical fine-tuning backbone for custom bone datasets. Fine-tune via `run_finetuning.py` with `-pretrained_weights`. |
| **MONAI Auto3DSeg**                | `Project-MONAI/MONAI` (auto3dseg module)                               | AutoML for 3D segmentation. AutoRunner ensembles DiNTS + SegResNet + SwinUNETR. **Apache 2.0 ✅.** Won MICCAI 2022 HECKTOR 1st place. Minimal input: `datalist.json` + `task.yaml`. Use when hospital provides >100 annotated scans for a new anatomy domain. CLI: `python -m monai.apps.auto3dseg AutoRunner run --input='./task.yaml'`. |

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         BROWSER  (Client)                                   │
│                                                                             │
│  React 19 + Vite + TypeScript  ·  Zustand  ·  TanStack Query               │
│                                                                             │
│  4-Column AppShell  (adapted from RapidTool-Fixture)                        │
│  ┌──────────┬─────────────────────┬───────────────────────┬──────────────┐  │
│  │Toolbar   │ Context Panel       │ 3D Viewport           │ Properties   │  │
│  │(icons)   │ (workflow steps)    │                       │ Panel        │  │
│  │📥 Upload │  UploadPanel        │  NiiVue WebGL2        │              │  │
│  │🔍 Scout  │  ScoutPanel         │  volume renderer      │  Study info  │  │
│  │🎯 Select │  SelectPanel        │  ─────────────────    │  Masks       │  │
│  │🤖 AI Seg │  AISegmentPanel     │  R3F mesh layer:      │  ROI/Crop    │  │
│  │✏️ Refine │  RefinePanel        │  · scout islands      │  Mesh        │  │
│  │⚙️ Mesh   │  MeshPanel          │  · AI label meshes    │  Implants    │  │
│  │💾 Export │  ExportPanel        │  · 6-handle CropBox   │              │  │
│  └──────────┴─────────────────────┴───────────────────────┴──────────────┘  │
│                                                                             │
│  session_id (JWT/signed cookie) bound to every request                     │
│  DICOM series picker → chunked upload → NIfTI (universal format)            │
└─────────────────────────────────────────────────────────────────────────────┘
                          HTTPS (TLS required in prod)
┌─────────────────────────────────────────────────────────────────────────────┐
│                         BACKEND  (Python FastAPI)                           │
│                                                                             │
│  /upload      pydicom → PHI strip → dcm2niix → NIfTI → scan_qc task         │
│  /scout       MeshLib  HU-threshold → connected-component islands  (CPU)    │
│  /segment     TotalSegmentator on cropped ROI volume  (gpu_queue)           │
│  /refine      SAM-Med3D-turbo 3D click  |  MedSAM 2D bbox  (gpu_queue)      │
│  /mesh        MeshLib Pass B: repair → watertight → multi/single export     │
│  /jobs/{id}   SSE progress stream per Celery task                           │
│                                                                             │
│  Celery workers:                                                            │
│    cpu_queue  (concurrency=4)  → scout, scan_qc, mesh generation            │
│    gpu_queue  (concurrency=1)  → segment, refine  (VRAM serialized)         │
│                                                                             │
│  Storage: MinIO (AES-256 SSE-S3)  |  Redis (task state + session cache)     │
│  Audit log: postgres audit_log table — every task + export timestamped      │
└─────────────────────────────────────────────────────────────────────────────┘
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
| AI Segmentation (Tier 2 — Auto)   | **TotalSegmentator v2**              | 117-class CT segmentation. Python API: `totalsegmentator(input_img, task="total", roi_subset=[...], fast=True, device="gpu")`. **Pin `torch<=2.8.0`** — nnU-Net has severe 3D conv regression in torch ≥ 2.9.0. `appendicular_bones` subtask requires commercial license; default `total` task covers femur, tibia, humerus, hip, ribs, sternum, skull for free. 2025 new subtasks: `craniofacial_structures` (mandible, teeth, skull, sinuses), `teeth` (FDI notation, CVPR 2025), `trunk_cavities`. |
| AI Segmentation (Tier 3 — Interactive 3D) | **SAM-Med3D-turbo** (Apache 2.0) | Surgeon clicks 3D point in R3F viewport → backend runs `medim.create_model("SAM-Med3D", pretrained=True)` → returns binary 3D mask. 10–100× fewer prompts than 2D SAM. Install: `pip install medim`. |
| AI Segmentation (Tier 4 — Interactive 2D) | **MedSAM / LiteMedSAM** (Apache 2.0) | Surgeon draws 2D bbox on NiiVue MPR slice → MedSAM inference on single slice → stack across z → 3D mask. Use for unknown structures not in TotalSegmentator's 117 classes. **MedSAM2** (requires `torch==2.5.1`) must run in a **separate Docker service** (`medsam2/`) and is accessed via internal HTTP call from the main backend — never imported in the same process as TotalSegmentator. API contract: `POST /medsam2/propagate {study_id, slice_idx, bbox}` → `{mask_nifti_url}`. GPU allocation: separate device (`GPU_DEVICE_MEDSAM2=cuda:1`) or time-shared with a VRAM guard. |
| AI Segmentation (Tier 5 — Custom Fine-tune) | **STU-Net-B + nnU-Net v2** (Apache 2.0) | Hospital provides annotated bone CTs → fine-tune STU-Net-B (58M params, 59-bone pre-trained) via `run_finetuning.py`. Alternatively: Auto3DSeg AutoRunner for ensemble training. MedNeXt-L k5 for <200 case datasets. |
| AI Segmentation Fallback (Tier 1) | **MeshLib VDB + Otsu threshold**     | Two-mode pipeline: if AI unavailable or user wants fast preview, load DICOM directly via `mr.VoxelsLoad.loadDicomsFolderTreeAsVdb()` and run marching cubes at Otsu-determined HU threshold. Completes in <5 s vs 30–300 s for AI. |
| AI Segmentation (custom bundles)  | **MONAI + nnU-Net + Auto3DSeg**      | For additional fine-grained tasks (craniofacial, teeth, appendicular bones) or fine-tuning on specific anatomy. Auto3DSeg AutoRunner for ensemble training: `python -m monai.apps.auto3dseg AutoRunner run --input='./task.yaml'`. |
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
│    │  │ Step content    │    │  │ ROI Box: 6-handle  │  │  │ ROI /    │  │
│    │  │ (per step)      │    │  │ CropBox, per-axis  │  │  │ Crop     │  │
│    │  └─────────────────┘    │  │  drag (X⁻X⁺Y⁻Y⁺Z⁻Z⁺)  │  ├──────────┤  │
│ 💾 │  ───────────────────    │  └────────────────────┘  │  │ Mesh     │  │
│    │  [🔼 🧩 🎯 ✂️ ⚙️ 💾]    │                          │  └──────────┘  │
│    │  mini-map step row      │  [floating tips overlay] │                │
├────┴─────────────────────────┴──────────────────────────┴────────────────┤
│  Ready  •  WebGL 2.0  •  TotalSegmentator v2  •  ████████░░  87%         │  ← status bar
└──────────────────────────────────────────────────────────────────────────┘
```

### VSP Engine workflow steps — Scout → Select → AI → Refine → Export

The workflow is designed so the surgeon sees *something* in under 10 seconds and always has interactive control before any slow AI job runs.

| Icon | Step            | Time      | What Happens                                                                                                    |
| ---- | --------------- | --------- | --------------------------------------------------------------------------------------------------------------- |
| 📥   | **Upload**      | 2–5 s     | DICOM drag-drop → dcm2niix NIfTI conversion in background; NiiVue starts rendering raw CT immediately            |
| 🔍   | **Scout**       | 5–10 s    | MeshLib HU threshold (~300 HU) → connected-component islands → per-island colored 3D mesh overlaid in R3F; surgeon can visually inspect all bone anatomy at once  |
| 🎯   | **Select**      | 0 s       | Surgeon **clicks bone islands** in the 3D view to select anatomy of interest; OR draws a manual 3D bounding box. System auto-computes tight ROI AABB. **6-handle crop box** (one draggable arrow per face: X⁻ X⁺ Y⁻ Y⁺ Z⁻ Z⁺) lets the surgeon fine-tune each axis independently. SelectPanel shows three dual-range sliders (one per axis) **and** the NiiVue MPR views each show a colored crop-plane line that moves in sync — axial slice shows Z⁻/Z⁺ lines, coronal shows Y⁻/Y⁺, sagittal shows X⁻/X⁺. |
| 🤖   | **AI Segment**  | 5–120 s   | TotalSegmentator runs on **cropped ROI volume only** → per-bone NIfTI labels; colored labels replace rough threshold islands; sidebar shows named anatomy checklist |
| ✏️   | **Refine**      | 2 s/click | Optional: surgeon clicks a misclassified region → SAM-Med3D-turbo 3D point → corrected mask; OR draws 2D bbox on MPR → MedSAM slice refinement |
| ⚙️   | **Mesh**        | 5–15 s    | MeshLib on selected AI masks → repair → simplify; quality/thickness/hollow sliders; watertight badge            |
| 💾   | **Export**      | instant   | Select which bones to include → STL/OBJ/3MF download with planning watermark                                    |

> **Why Scout first?** The threshold mesh gives the surgeon an immediate 3D anatomical map of the scan — they know exactly which bone island to click for AI to focus on. AI then runs on a crop 5–20× smaller than the full volume: faster (5–30 s vs 60–300 s), more accurate (no out-of-ROI confusion), and uses far less GPU memory. The rough mesh is a *navigation proxy*, not the final output.

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

## 6. Data Flow — Scout → Select → AI → Refine → Export

```
── PHASE 1: LOAD ──────────────────────────────────────── ~5–15 s ──

CLIENT (series picker):
         DICOM drag-drop (folder or ZIP)
         dcmjs → parse all DICOM files in a WebWorker
         → group by SeriesInstanceUID → display series grid:
             thumbnail (first-slice canvas), SeriesDescription,
             slice count, voxel spacing, modality (CT/MR)
         Surgeon selects the correct series (bone kernel, non-contrast CT)
         ↓ Validate: warn if slice thickness > 3 mm or modality ≠ CT

CLIENT (chunked upload):
         Selected series files → chunked upload via tus protocol
         Chunk size: 5 MB; show progress bar
         POST /upload/chunk {chunk_index, total_chunks, study_id}
         POST /upload/finalize {study_id} when all chunks received

CLIENT (auth):
         Every request includes session_id (JWT in Authorization header
         or HttpOnly signed cookie). Issued by /auth/session on first visit.
         Backend binds study to session_id — cross-session access returns 403.

BACKEND /upload/finalize:
         Assert session_id ownership of study_id
         pydicom → strip PHI (PatientName, PatientID, DOB, AccessionNumber,
                               InstitutionName, ReferringPhysicianName)
         dcm2niix → NIfTI (.nii.gz) with preserved voxel spacing + orientation
         Store: MinIO study/{id}/volume.nii.gz  (AES-256 SSE-S3)
         Enqueue scan_qc Celery task (cpu_queue)
         Return: {study_id, nifti_url, qc_task_id}

BACKEND /jobs/scan_qc (Celery cpu_queue, ~2 s):
         Check slice thickness (warn if > 3 mm)
         Check voxel spacing isotropy (warn if z-spacing > 2× xy-spacing)
         Check HU range (−1000 to 3000 expected for CT)
         Check volume coverage (number of slices, gap detection)
         Return: {warnings: [], ready: true}
         Frontend shows warning banner if any warnings — non-blocking

CLIENT:  NiiVue attaches to canvas, loads NIfTI URL
         → CT renders immediately; scan QC warnings shown in UploadPanel
         → surgeon acknowledges warnings, proceeds to Scout


── PHASE 2: SCOUT ─────────────────────────────────────── ~5–10 s ──

BACKEND /scout (fast Celery task, Tier 1 — cpu_queue):
         # NIfTI-primary path: volume.nii.gz is the universal format.
         # dcm2niix runs once at upload; all downstream tools (NiiVue,
         # TotalSegmentator, MeshLib, SAM-Med3D) consume the same NIfTI.
         # Never re-read DICOM during scout or segment phases.
         vox = mr.loadVoxels("volume.nii.gz")       ← NIfTI, already converted
         mesh = mr.marchingCubesFloat(vox.data, mr.MarchingCubesParams(300.0))  # bone HU ~300
         mr.removeSmallComponents(mesh, minVolumeVerts=50)  ← remove noise
         components = mr.getAllComponents(mesh)             ← per-island split
         # Each component in world-mm coordinates (NIfTI affine preserves spacing)
         Each island → separate .ply blob + centroid_xyz (world mm) + AABB (world mm) + voxel_count
         Return: [{island_id, centroid_xyz, aabb, ply_url, voxel_count}, …]
         # centroid_xyz is in NIfTI RAS world space — used later for TotalSegmentator task routing

CLIENT:  R3F loads each island PLY as a separate <mesh>
         Color per island (medical colormap, 20 colors cycling)
         Hover → highlight + show voxel_count tooltip
         Click island → toggle selected (yellow glow)
         Sidebar lists islands sorted by size with checkboxes


── PHASE 3: SELECT / ROI ──────────────────────────────── ~0 s ───

CLIENT:  Surgeon selects ≥1 islands (click in R3F or sidebar checkbox)
         System: union of selected island AABBs → tight crop bbox
                 + 10% padding each axis → proposed ROI box

         ROI box rendered as a custom <CropBox> R3F component:
           - Wireframe box outline in cyan
           - 6 face-center drag handles (one per face, arrow mesh):
               X⁻ (left)   X⁺ (right)
               Y⁻ (front)  Y⁺ (back)
               Z⁻ (bottom) Z⁺ (top)
           - Each handle constrained to its axis only (no off-axis drift)
           - Handles clamp: Xmin < Xmax, Ymin < Ymax, Zmin < Zmax
           - NOT Drei TransformControls (that moves the whole box; wrong here)

         SelectPanel (context panel) shows:
           Axis    | Face-      | Face+     | mm values
           ─────────────────────────────────────────────
           X (R→L) | ←  ──●──  |  ──●──  → | e.g. 45 mm … 312 mm
           Y (A→P) | ←  ──●──  |  ──●──  → | e.g. 80 mm … 290 mm
           Z (I→S) | ←  ──●──  |  ──●──  → | e.g. 120 mm … 480 mm

           Three dual-handle range sliders (Shadcn Slider, min/max thumbs)
           + numeric text inputs for each of the 6 values
           All inputs two-way synced with the 3D handles

         NiiVue MPR views show crop-plane lines in sync:
           Axial slice   (XY plane, scroll = Z) → horizontal dashed lines at Zmin, Zmax
           Coronal slice (XZ plane, scroll = Y) → horizontal dashed lines at Ymin, Ymax
           Sagittal slice(YZ plane, scroll = X) → horizontal dashed lines at Xmin, Xmax
           Lines drawn via a Canvas 2D overlay absolutely positioned over NiiVue canvas
           (NiiVue does not have a native annotation API for crop lines)

         "Confirm ROI" button → lock crop, pass 6 values to backend
         Locked state: handles dim, box turns orange, "Edit ROI" to reopen


── PHASE 4: AI SEGMENT ────────────────────────────────── 5–120 s ──

BACKEND /segment (Celery task, Tier 2):
         Crop NIfTI to ROI: SimpleITK.RegionOfInterest(nifti, roi_box)
         totalsegmentator(
           cropped_volume,
           task=auto_pick_task(roi_labels_hint),  ← total / craniofacial / teeth
           roi_subset=suggested_label_names,
           fast=False,
           device="gpu"
         )
         → multilabel NIfTI (each integer = one bone class)
         Store: study/{id}/seg_{task}.nii.gz
         self.update_state(PROGRESS, percent=…, step="segmenting")

CLIENT:  SSE progress → status bar updates
         On complete: NiiVue adds seg NIfTI as overlay (opacity=0.5, colormap="warm")
         Scout island meshes REPLACED by AI-labeled colored meshes in R3F
         Sidebar: named anatomy checklist (e.g. femur_left ✓, vertebra_L3 ✓)
         Each label has show/hide eye icon + color swatch


── PHASE 5: INTERACTIVE REFINE ──────────────────────────── optional ──

CLIENT (Tier 3 — SAM-Med3D):
         Surgeon shift-clicks a voxel in R3F viewport
         → RayCast → world (x,y,z) → voxel (z,y,x) passed to backend
         /refine/point: SAM-Med3D-turbo medim.create_model() → binary mask
         New mask overlaid; surgeon accepts or dismisses

CLIENT (Tier 4 — MedSAM):
         Surgeon draws 2D bbox on NiiVue MPR slice
         nv.onLocationChange gives slice index + image coords
         /refine/bbox: MedSAM per-slice → stack across z → 3D volume
         Useful for implants, hardware, rare pathology


── PHASE 6: MESH ─────────────────────────────────────────── 5–15 s ──

BACKEND /mesh/generate:
         For each selected label:
           binary_mask = extract_label(seg_nifti, label_id)
           scipy.ndimage.binary_closing(mask, iterations=2)  ← fill gaps
           SimpleGrid → MeshLib marchingCubesFloat(iso=0.5)
           mr.fixSelfIntersections(mesh)
           mr.fillHoles(mesh)
           mr.removeSmallComponents(mesh, 100)
           mr.relaxMesh(mesh)                                ← smooth
           mr.decimateMesh(mesh, maxError=0.2)               ← simplify
         Boolean union of all selected labels → combined mesh
         mr.boolean(mesh, roi_box_mesh, BooleanOperation.Intersection)
         assert topology.isClosed()                          ← REQUIRED
         Export: study/{id}/mesh_{timestamp}.stl

CLIENT:  STLLoader → R3F mesh preview (replace scout clouds)
         Orbit / pan / zoom; per-label color; show/hide toggles


── PHASE 7: EXPORT ───────────────────────────────────────── 2–10 s ──

CLIENT (ExportPanel — multi-structure tree):
         Each AI label is shown as a tree row:
           [☑] femur_left      [◉ combined | ○ separate]
           [☑] tibia_left      [◉ combined | ○ separate]
           [☐] fibula_left     (deselected — not included)
         Surgeon chooses per-label: merge into combined STL or export as
         individual file. Entire selection → POST /mesh/export

         Format selector: STL / OBJ / 3MF
         Scale selector: 1:1 / ×2 / ×0.5

BACKEND /mesh/export:
         If all selected labels → union=True: boolean union → single .stl
         If any label marked separate: union=False → ZIP of per-label .stl files
           each named {label_name}_{study_id}_{timestamp}.stl
         Embed "FOR PLANNING PURPOSES ONLY — NOT FOR DIAGNOSTIC USE" in STL header
         Log to audit_log: {session_id, study_id, labels, format, timestamp}
         Return: signed MinIO URL (15-min expiry)

CLIENT (case report):
         planReport Zustand slice captures:
           - viewport screenshot (gl.domElement.toDataURL()) at Export step
           - list of exported structures + file names
           - any measurements taken (caliper distances / angles)
           - surgeon notes (free text input in ExportPanel)
         "Download Report" → browser window.print() on styled report page
           (or pdfmake for programmatic PDF generation)
```

---

## 7. AI Segmentation Strategy — Scout → Select → Focus → Refine

### Design Principle: AI Runs Last, On the Smallest Possible Volume

Running TotalSegmentator on a full 512×512×800 CT when the surgeon only needs one femur wastes 80–90% of GPU time and increases error risk (structures outside ROI cause label confusion). The workflow front-loads a cheap threshold pass that lets the surgeon define the ROI *before* AI starts.

```
ROI crop ratio examples:
  Full chest CT (512×512×600)  → femur ROI (200×200×300) = 9.8% of volume  → 10× faster AI
  Full head CT  (512×512×400)  → mandible ROI (250×200×150) = 7.2% of volume → 14× faster AI
  Full spine    (512×512×800)  → L4-L5 ROI (150×150×100) = 1.7% of volume  → 58× faster AI
```

### Phase 2 — Scout (MeshLib HU Threshold + Connected Components)

```python
import meshlib.mrmeshpy as mr

# Fast path: load NIfTI voxels into MeshLib
vox = mr.loadVoxels("volume.nii.gz")                 # or fromSimpleGrid()
mesh = mr.marchingCubesFloat(vox.data, mr.MarchingCubesParams(300.0))  # HU 300
mr.removeSmallComponents(mesh, 50)                   # remove sub-50-voxel noise

# Split into connected components for per-island UX
components = mr.getAllComponents(mesh)               # returns list[Mesh]
# Each component → PLY blob → return to frontend for 3D display
for i, comp in enumerate(components):
    mr.saveMesh(comp, f"island_{i}.ply")
```

### Phase 4 — AI Segment (TotalSegmentator on Cropped Volume)

```python
import SimpleITK as sitk
from totalsegmentator.python_api import totalsegmentator

# Crop to ROI (surgeon-confirmed bounding box)
img = sitk.ReadImage("volume.nii.gz")
physical_origin = roi_min_xyz          # world coords from R3F gizmo
physical_size   = roi_max_xyz - roi_min_xyz
resampler = sitk.RegionOfInterestImageFilter()
resampler.SetRegionOfInterest(start_index, size_voxels)
cropped = resampler.Execute(img)
sitk.WriteImage(cropped, "cropped.nii.gz")

# Auto-pick task from ROI hint labels
task = pick_task(hint_labels)         # "total" / "craniofacial_structures" / "teeth"
output = totalsegmentator(
    "cropped.nii.gz",
    task=task,
    roi_subset=hint_label_names,      # skip classes outside this ROI
    fast=False,
    device="gpu",
    ml=True,                          # multilabel NIfTI output
)
```

**Task auto-selection logic:**
| Scout island centroids suggest… | → task |
|---|---|
| skull / mandible / maxilla region | `craniofacial_structures` |
| individual tooth crowns | `teeth` (FDI notation, CVPR 2025) |
| vertebrae column | `total` with `roi_subset=["vertebrae_*"]` |
| femur / tibia / fibula | `total` with `roi_subset=["femur", "tibia", "fibula"]` |
| full body or unknown | `total` (all 117 classes) |

### Phase 5 — Interactive Refinement

**Tier 3 — SAM-Med3D-turbo (3D point, ~2 s):**
```python
import medim, torch
model = medim.create_model("SAM-Med3D", pretrained=True)  # downloads turbo ckpt
# point from R3F shift-click → (z, y, x) voxel order
point = torch.tensor([[[z_norm, y_norm, x_norm]]])         # normalized [0,1]
label = torch.tensor([[1]])                                 # 1=positive
with torch.no_grad():
    mask = model(image_patch_tensor, point, label)          # (1,1,D,H,W)
```

**Tier 4 — MedSAM (2D box, ~0.5 s/slice):**
```python
# box from NiiVue slice interaction → bbox [x1, y1, x2, y2] + slice_idx
# MedSAM inference per slice → stack along z → 3D mask volume
from segment_anything import sam_model_registry
sam = sam_model_registry["vit_b"](checkpoint="medsam_vit_b.pth").eval()
# ... run per-slice, concatenate → final 3D mask
```

### Task Routing Summary

| Scenario | Tier | Tool | GPU | Time |
|---|---|---|---|---|
| Bone overview, rough print | 1 | MeshLib HU threshold | No | <5 s |
| Standard CT bone (known anatomy) | 2 | TotalSegmentator on ROI crop | Yes | 5–60 s |
| Unknown structure / rare pathology | 3 | SAM-Med3D-turbo 3D click | Yes | ~2 s |
| Implant / hardware / 2D box natural | 4 | MedSAM bbox on slice | No/Yes | ~1 s |
| Hospital has custom annotated data | 5 | STU-Net-B fine-tuned | Yes | offline train |

---

## 8. Mesh Processing Pipeline (MeshLib)

Two mesh processing passes at different quality levels:

### Pass A — Scout Mesh (fast, rough, interactive)
```python
import meshlib.mrmeshpy as mr

# From NIfTI voxels or DICOM folder:
vox = mr.loadVoxels("volume.nii.gz")
mesh = mr.marchingCubesFloat(vox.data, mr.MarchingCubesParams(300.0))  # HU 300
mr.removeSmallComponents(mesh, minVolumeVerts=50)   # remove tiny noise islands
components = mr.getAllComponents(mesh)              # split into islands
# Each component → .ply → client for interactive clicking
# Runtime: ~5–10 s on CPU. No GPU needed.
```

### Pass B — Final Export Mesh (accurate, watertight, print-ready)
```python
import meshlib.mrmeshpy as mr

# 1. Load binary mask from AI label (one per selected bone)
vox = mr.loadVoxels("label_femur_left.nii.gz")

# 2. Marching cubes on binary mask
mesh = mr.marchingCubesFloat(vox.data, mr.MarchingCubesParams(0.5))

# 3. Heal: MUST run in this order
mr.fixSelfIntersections(mesh)          # removes bad triangles
mr.fillHoles(mesh)                     # closes open boundaries
mr.removeSmallComponents(mesh, 100)    # cull noise fragments

# 4. Smooth (light — preserve anatomy detail)
params = mr.MeshRelaxParams()
params.iterations = 3
mr.relax(mesh, params)

# 5. Simplify (50k faces = good web preview; 200k = print quality)
da = mr.DecimateSettings()
da.maxError = 0.2
mr.decimateMesh(mesh, da)

# 6. Boolean intersect with ROI box (clean crop boundary)
box = mr.makeCube(mr.Box3f(
    mr.Vector3f(*roi_min_mm),
    mr.Vector3f(*roi_max_mm)
))
result = mr.boolean(mesh, box, mr.BooleanOperation.Intersection).mesh

# 7. Optional: shell for hollow surgical guide models
# shelled = mr.offsetMesh(result, 1.5)   ← 1.5mm wall

# 8. REQUIRED: assert watertight before export
assert result.topology.isClosed(), "Mesh not watertight — refuse STL export"

# 9. Export
mr.saveMesh(result, "output.stl")
```

### Mesh Quality Tiers Exposed in UI

| UI Setting    | decimateMesh maxError | Target faces  | Use case                          |
|---------------|-----------------------|---------------|-----------------------------------|
| Preview       | 0.5 mm                | ~30k          | Fast R3F viewport preview         |
| Standard      | 0.2 mm                | ~100–200k     | General 3D printing               |
| High Detail   | 0.05 mm               | ~500k+        | Surgical accuracy, fine features  |

---

## 9. ROI Crop Box — 6-Handle Component Design

The crop box needs **independent per-axis face handles**, not a whole-box transform. `Drei TransformControls` is the wrong tool here — it applies unified translate/rotate/scale to the object as a whole. We need 6 separate draggable arrows.

### R3F Component Architecture

```tsx
// components/viewport/CropBox.tsx
import { useRef, useState } from 'react'
import { ThreeEvent } from '@react-three/fiber'
import { Line } from '@react-three/drei'

interface CropBoxProps {
  min: [number, number, number]   // [Xmin, Ymin, Zmin] in world mm
  max: [number, number, number]   // [Xmax, Ymax, Zmax] in world mm
  onChange: (min: [number, number, number], max: [number, number, number]) => void
  locked?: boolean
}

// 6 face handles — each constrained to ONE axis, ONE direction
const HANDLES = [
  { axis: 0, side: 'min', dir: -1, color: '#ff4444', label: 'X⁻' },  // left face
  { axis: 0, side: 'max', dir: +1, color: '#ff4444', label: 'X⁺' },  // right face
  { axis: 1, side: 'min', dir: -1, color: '#44ff44', label: 'Y⁻' },  // front face
  { axis: 1, side: 'max', dir: +1, color: '#44ff44', label: 'Y⁺' },  // back face
  { axis: 2, side: 'min', dir: -1, color: '#4488ff', label: 'Z⁻' },  // bottom face
  { axis: 2, side: 'max', dir: +1, color: '#4488ff', label: 'Z⁺' },  // top face
] as const

// Each handle renders:
// - An arrow cone mesh at the face center
// - Drag gesture constrained to handle's axis via pointer capture
// - Clamp: Xmin < Xmax - MIN_SIZE (never invert the box)
// Color coding: X=red, Y=green, Z=blue (standard medical/3D convention)
```

### SelectPanel Sliders (Context Panel)

```tsx
// components/sidebar/SelectPanel.tsx
// Three dual-range sliders, one per axis
// Each slider has TWO thumbs: min-face and max-face

const axes = [
  { label: 'X  (Left ↔ Right)',     axis: 0, color: 'text-red-400'   },
  { label: 'Y  (Anterior ↔ Post.)', axis: 1, color: 'text-green-400' },
  { label: 'Z  (Inferior ↔ Sup.)',  axis: 2, color: 'text-blue-400'  },
]

// For each axis, render:
// <label>  X (Left ↔ Right)  [45 mm … 312 mm]
// <Slider min={volMin[0]} max={volMax[0]} value={[roi.min[0], roi.max[0]]}
//         onValueChange={([lo, hi]) => updateROI(0, lo, hi)} />
// Two number inputs below for precise mm entry

// Total: 6 drag handles in 3D + 6 numeric inputs + 3 dual-range sliders
// All synchronized via the same Zustand scoutStore.roi state slice
```

### NiiVue MPR Crop-Plane Lines

NiiVue has no native crop-plane annotation API. We draw lines using a **Canvas 2D overlay** that sits `position: absolute` on top of the NiiVue canvas:

```tsx
// components/viewport/MPRCropOverlay.tsx
// Receives roi bbox in world mm; NiiVue exposes mm→pixel via nv.mm2frac() + canvas dims

useEffect(() => {
  const ctx = overlayCanvas.current.getContext('2d')
  ctx.clearRect(0, 0, w, h)

  // Axial (XY plane) — scrolling changes Z → draw Zmin / Zmax as horizontal lines
  const zMinPx = nv.mm2frac([0,0, roiMin[2]])[2] * h
  const zMaxPx = nv.mm2frac([0,0, roiMax[2]])[2] * h
  drawDashedLine(ctx, 0, zMinPx, w, zMinPx, '#4488ff')  // Z⁻  blue
  drawDashedLine(ctx, 0, zMaxPx, w, zMaxPx, '#4488ff')  // Z⁺  blue

  // Coronal (XZ plane) — scrolling changes Y → draw Ymin / Ymax
  const yMinPx = nv.mm2frac([0, roiMin[1], 0])[1] * h
  const yMaxPx = nv.mm2frac([0, roiMax[1], 0])[1] * h
  drawDashedLine(ctx, 0, yMinPx, w, yMinPx, '#44ff44')  // Y⁻  green
  drawDashedLine(ctx, 0, yMaxPx, w, yMaxPx, '#44ff44')  // Y⁺  green

  // Sagittal (YZ plane) — scrolling changes X → draw Xmin / Xmax as vertical lines
  const xMinPx = nv.mm2frac([roiMin[0], 0, 0])[0] * w
  const xMaxPx = nv.mm2frac([roiMax[0], 0, 0])[0] * w
  drawDashedLine(ctx, xMinPx, 0, xMinPx, h, '#ff4444')  // X⁻  red
  drawDashedLine(ctx, xMaxPx, 0, xMaxPx, h, '#ff4444')  // X⁺  red
}, [roiMin, roiMax])
```

### Zustand State Slice

```ts
// store/scoutStore.ts
type TaskStatus = 'idle' | 'running' | 'done' | 'error'

interface ROIState {
  min: [number, number, number]   // [Xmin, Ymin, Zmin] in mm (NIfTI RAS world space)
  max: [number, number, number]   // [Xmax, Ymax, Zmax] in mm
  locked: boolean
}

interface ScoutState {
  taskStatus: TaskStatus
  errorMessage: string | null     // shown in ScoutPanel error banner on 'error'
  islands: IslandMeta[]
  selectedIds: Set<string>
  roi: ROIState
}

// All other task slices follow the same error pattern:
// segmentationStore: { taskStatus, errorMessage, labels, ... }
// meshStore:         { taskStatus, errorMessage, meshUrl, quality, ... }
// studyStore:        { taskStatus, errorMessage, study, qcWarnings, ... }

// Error UX rule: each panel shows its own inline error banner
// (not a global toast) with a "Retry" action that re-queues the same task.
// If AI segment fails, surgeon can still proceed to Phase 7 with scout meshes.
```

// Single source of truth — CropBox 3D handles, SelectPanel sliders,
// and numeric inputs all read/write the same scoutStore.roi slice

### Axis Color Convention (standard in medical imaging)

| Axis | Color  | MPR view affected       | Handle label  |
|------|--------|-------------------------|---------------|
| X    | Red    | Sagittal (YZ plane)     | X⁻ X⁺         |
| Y    | Green  | Coronal  (XZ plane)     | Y⁻ Y⁺         |
| Z    | Blue   | Axial    (XY plane)     | Z⁻ Z⁺         |

This matches the left-hand radiological convention used in DICOM, ITK, and NiiVue (RAS orientation).

---

## 10. File Structure

```
vsp-engine/
 frontend/                     # React + Vite + TypeScript
    src/
       app/                  # App shell, routing
       components/
          viewport/         # R3F Canvas, NiiVue integration
             VolumeViewer.tsx       # NiiVue WebGL2 canvas wrapper
             ScoutViewer.tsx        # R3F: per-island threshold meshes (clickable)
             ROIBox.tsx             # CropBox: 6 face-center drag handles (X⁻ X⁺ Y⁻ Y⁺ Z⁻ Z⁺), each constrained to its axis
             MPRCropOverlay.tsx     # Canvas 2D overlay: draws Xmin/Xmax/Ymin/Ymax/Zmin/Zmax dashed lines on NiiVue MPR
             AILabelViewer.tsx      # R3F: per-label AI mesh overlays
             MeshViewer.tsx         # R3F: final export-quality mesh
             MPRSlices.tsx          # NiiVue MPR slice host canvas
          sidebar/          # Workflow panel steps
             UploadPanel.tsx      # Phase 1: DICOM drag-drop + series browser
             ScoutPanel.tsx       # Phase 2: threshold mesh + island list
             SelectPanel.tsx      # Phase 3: 3 axis range sliders (X/Y/Z, dual-thumb) + 6 numeric mm inputs + Confirm ROI
             AISegmentPanel.tsx   # Phase 4: TotalSegmentator task picker + progress
             RefinePanel.tsx      # Phase 5: SAM-Med3D click / MedSAM bbox
             MeshPanel.tsx        # Phase 6: quality sliders + watertight badge
             ExportPanel.tsx      # Phase 7: format / scale / download
          ui/               # Shadcn/UI components
       store/                # Zustand slices
          studyStore.ts          # DICOM metadata, upload state, qcWarnings, session_id
          scoutStore.ts          # island list, selected islands, roi:{min:[x,y,z],max:[x,y,z],locked}, taskStatus+errorMessage
          segmentationStore.ts   # AI labels, per-label visibility/color, taskStatus+errorMessage
          meshStore.ts           # final mesh state, quality settings, taskStatus+errorMessage
          viewportStore.ts       # NiiVue state, R3F camera, crosshair
          planReportStore.ts     # viewport screenshots, measurements, surgeon notes, export manifest
          implantStore.ts        # uploaded implant template STL refs, per-implant transform matrix
       hooks/                # TanStack Query hooks
       lib/                  # dcmjs helpers, API client
          coordTransforms.ts  # EXPLICIT coordinate-system transforms (see Section 15)
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
          upload.py         # POST /upload  → NIfTI conversion
          scout.py          # POST /scout   → HU threshold connected components
          segment.py        # POST /segment → TotalSegmentator on ROI crop
          refine.py         # POST /refine/point  /refine/bbox
          mesh.py           # POST /mesh/generate  GET /mesh/export
          jobs.py           # GET /jobs/{id}  SSE progress
       services/
          dicom_service.py    # pydicom + SimpleITK + dcm2niix + PHI strip
          scan_qc_service.py  # slice-thickness, spacing, HU-range, gap checks
          scout_service.py    # MeshLib HU threshold + connected components (NIfTI input)
          segment_service.py  # TotalSegmentator + task routing
          refine_service.py   # SAM-Med3D-turbo + MedSAM (local); MedSAM2 via HTTP to medsam2 service
          mesh_service.py     # MeshLib Pass B: repair → watertight → union/separate export
          coord_service.py    # coordinate-system transform helpers (see Section 15)
          audit_service.py    # write to audit_log: task runs, exports, session events
          feature_flags.py    # appendicular_bones license check, MedSAM2 availability flag
       tasks/                 # Celery tasks
          scan_qc_task.py     # cpu_queue
          scout_task.py       # cpu_queue
          segment_task.py     # gpu_queue (concurrency=1)
          refine_task.py      # gpu_queue (concurrency=1)
          mesh_task.py        # cpu_queue
       models/                # Pydantic v2 schemas
       config.py
    requirements.txt
    Dockerfile

 docker/
    docker-compose.yml
    docker-compose.gpu.yml    # GPU override (nvidia runtime, gpu_queue worker)
    medsam2/                  # isolated service: torch==2.5.1, separate GPU device
       Dockerfile
       main.py                # FastAPI: POST /medsam2/propagate
    nginx.conf                # includes ssl_certificate block; client_max_body_size 1g

 .env.example
 .github/
    workflows/
        ci.yml
 README.md
```

---

## 10. Build Plan — Scout → Select → AI → Refine → Export

### Phase 1: Foundation (Week 1–2)

- [ ] Monorepo setup: `pnpm workspace` + Python `uv` environment
- [ ] Frontend scaffold: Vite + React + TypeScript + Tailwind + Shadcn/UI init
- [ ] Backend scaffold: FastAPI + Celery + Redis docker-compose
- [ ] Two Celery queues: `cpu_queue` (concurrency=4) and `gpu_queue` (concurrency=1, GPU workers only)
- [ ] Session auth: `/auth/session` issues signed JWT; all endpoints enforce session→study ownership
- [ ] DICOM series picker: WebWorker dcmjs parse → SeriesInstanceUID groups → thumbnail grid UI
- [ ] Chunked upload via tus protocol: `/upload/chunk` + `/upload/finalize` endpoints; Nginx `client_max_body_size 1g`
- [ ] PHI anonymization at finalize: PatientName, PatientID, DOB, AccessionNumber, InstitutionName stripped
- [ ] dcm2niix → NIfTI conversion; NIfTI is the universal format for all downstream tools
- [ ] Scan QC Celery task (cpu_queue): slice thickness, spacing, HU range, z-gap checks → warning list
- [ ] File storage service: MinIO with SSE-S3 encryption; local FS fallback for dev
- [ ] Audit log: `audit_log` Postgres table — session_id, action, study_id, timestamp (no PHI)
- [ ] Nginx config: TLS termination block (self-signed for dev, Let's Encrypt hook for prod)

### Phase 2: Volume Viewer (Week 2–3)

- [ ] Integrate NiiVue into the React viewport (canvas + R3F overlay)
- [ ] Load backend NIfTI via URL into NiiVue volume renderer
- [ ] MPR panel: axial / coronal / sagittal slice display
- [ ] Brightness/contrast/windowing controls (Shadcn sliders)
- [ ] Zustand stores: studyStore + viewportStore

### Phase 3: Scout Pass (Week 3–4)

- [ ] Backend `/scout` Celery task: MeshLib HU threshold (300 HU) → marching cubes on CPU
- [ ] `mr.getAllComponents()` → per-island PLY blobs + centroid + AABB + voxel count
- [ ] Return island list to frontend; SSE progress (fast, 5–10 s)
- [ ] Frontend: R3F renders each island as a separate colored mesh
- [ ] Island hover (highlight) + click (toggle select, yellow glow)
- [ ] ScoutPanel: sorted island checklist sidebar with voxel counts
- [ ] scoutStore: selected island IDs, computed ROI AABB

### Phase 4: ROI Selection (Week 4)

- [ ] ROI box auto-computed from union of selected island AABBs + 10% padding
- [ ] Custom `<CropBox>` R3F component: 6 face-center drag handles (X⁻ X⁺ Y⁻ Y⁺ Z⁻ Z⁺), each constrained to its own axis only; no Drei TransformControls
- [ ] Handles color-coded: X=red, Y=green, Z=blue (DICOM/ITK convention)
- [ ] Clamp logic: Xmin < Xmax - MIN_SIZE for each axis (prevent box inversion)
- [ ] SelectPanel: 3 dual-range sliders (one per axis, two thumbs each) + 6 numeric mm inputs
- [ ] All controls bidirectionally synced via `scoutStore.roi` Zustand slice
- [ ] `MPRCropOverlay.tsx`: Canvas 2D overlay draws 2 dashed crop lines per MPR plane (6 lines total)
- [ ] Crop lines update in real-time as handles/sliders drag (no "confirm" needed for line preview)
- [ ] "Confirm ROI" → lock, advance; "Edit ROI" to reopen when locked

### Phase 5: AI Segmentation on ROI Crop (Week 4–6)

- [ ] Backend `/segment` Celery task: SimpleITK ROI crop → TotalSegmentator
- [ ] Task auto-selection logic (island centroid hints → `total`/`craniofacial_structures`/`teeth`)
- [ ] `roi_subset` hint list reduces TotalSegmentator classes to relevant anatomy
- [ ] Multilabel NIfTI output → per-label PLY meshes returned to frontend
- [ ] SSE progress → status bar ("Segmenting: 47%")
- [ ] Frontend: scout island meshes fade out; AI label meshes fade in
- [ ] NiiVue: add seg NIfTI as overlay (opacity 0.5, colormap="warm")
- [ ] AISegmentPanel: named anatomy checklist, show/hide eye, color swatch per label

### Phase 6: Interactive Refinement (Week 6–7)

- [ ] SAM-Med3D-turbo backend service: `medim.create_model()` warm-up on startup
- [ ] Shift-click in R3F → rayCast → world XYZ → voxel (z,y,x) → `/refine/point`
- [ ] MedSAM backend: NiiVue slice bbox → `/refine/bbox` → stack masks → 3D mask
- [ ] RefinePanel: tool switcher (point vs box), accept/dismiss per-correction
- [ ] Corrected mask merges into segmentation store and R3F label mesh

### Phase 7: Mesh Generation & Export (Week 7–8)

- [ ] `/mesh/generate`: MeshLib Pass B on selected labels → per-label repair → boolean union
- [ ] Mesh quality UI: Preview / Standard / High Detail decimation presets
- [ ] Shell / hollow offset for surgical guide models
- [ ] Watertight badge (`topology.isClosed()`) — blocks download if False
- [ ] Multi-structure export tree in ExportPanel: per-label combine/separate toggle
- [ ] `/mesh/export`: `union=True` → single STL; `union=False` → ZIP of per-label STL files
- [ ] Measurement tools in MeshPanel: point-to-point caliper (two R3F rayCast picks → world distance mm); 3-point angle measurement; results shown in viewport labels and saved to planReportStore
- [ ] Export watermark: "FOR PLANNING PURPOSES ONLY — NOT FOR DIAGNOSTIC USE" in STL binary header comment
- [ ] Audit log entry for every export: session_id, label list, format, timestamp
- [ ] Case report: planReport Zustand slice → viewport screenshot + measurements + surgeon notes → "Download Report" PDF
- [ ] MeshPanel + ExportPanel

### Phase 8: Polish & Clinical UX (Week 8–10)

- [ ] Study management: patient list, past exports, session restore
- [ ] Robust error handling: each panel shows inline error banner + Retry; AI failure → fallback to scout meshes for export path
- [ ] AI weight pre-download on Docker build (`totalseg_download_weights`)
- [ ] `segmentation_ready` Redis flag → "AI loading…" overlay if weights not ready
- [ ] Scout works without AI ready — surgeon can proceed to Phase 4 while AI loads
- [ ] `appendicular_bones` license feature flag: `feature_flags.py` returns `{tibia: false, fibula: false}` if license absent; SelectPanel shows "License required" badge on affected anatomy; gracefully falls back to `total` task
- [ ] Implant overlay (basic): upload reference STL (implant template) → rendered in R3F alongside patient mesh; translate/rotate with `Drei TransformControls` (this is the correct use for TransformControls — on implant template, not on crop box); implant transform stored in implantStore
- [ ] MedSAM2 Docker service (`docker/medsam2/`): isolated `torch==2.5.1` environment; internal `POST /medsam2/propagate` API; GPU device assignment (`GPU_DEVICE_MEDSAM2`)
- [ ] Study data TTL: MinIO lifecycle policy — auto-delete study data after 30 days; configurable via env var `STUDY_TTL_DAYS`
- [ ] Performance: WebWorker for DICOM series parsing; lazy R3F asset loading; NiiVue canvas cleanup on step unmount
- [ ] Unit + integration tests (Vitest frontend, pytest backend)
- [ ] Docker Compose GPU single-command deploy; Nginx SSL block with self-signed-cert dev helper

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

## 13. Regulatory & Security Considerations

> **This is a research/planning tool, not a certified medical device (yet).** TotalSegmentator explicitly states it is "not a medical device." Any clinical use requires additional validation.

### Output Labeling
- All STL/OBJ/3MF exports: embed "FOR PLANNING PURPOSES ONLY — NOT FOR DIAGNOSTIC USE" in file metadata
- All report PDFs: watermark on every page

### PHI / HIPAA
- Anonymize at upload boundary: strip PatientName, PatientID, DOB, AccessionNumber, InstitutionName, ReferringPhysicianName
- Never log any DICOM tag values — audit log records actions only (session_id, study_id UUID, timestamp)
- Study data TTL: auto-delete from MinIO after configurable retention period (default 30 days)
- Session isolation: every API endpoint verifies `study.session_id == request.session_id`; 403 on mismatch

### Security
- **TLS required in production**: Nginx terminates HTTPS; all traffic over HTTP redirects to HTTPS
- **Encryption at rest**: MinIO SSE-S3 (AES-256) for all study NIfTI, segmentation masks, and mesh files
- **Session tokens**: HttpOnly signed cookie or short-lived JWT (24h expiry); refreshed on activity
- **Nginx limits**: `client_max_body_size 1g`; rate limiting on `/upload` and `/segment` endpoints

### Audit Trail
- `audit_log` Postgres table: `{id, session_id, action, study_id, labels, format, timestamp}`
- Actions logged: upload_complete, scout_run, segment_run, refine_click, mesh_generate, mesh_export
- Exportable as CSV for institutional compliance review

### License Gating
- `appendicular_bones` subtask (tibia, fibula, patella, carpals): requires TotalSegmentator commercial license
- `feature_flags.py` checks `TOTALSEG_LICENSE_KEY` env var at startup → sets `features.appendicular_bones = True/False`
- Frontend queries `/config/features` on load; SelectPanel gates affected anatomy with "License required" badge
- Graceful fallback: `total` task covers femur, tibia (less precise), hip — usable without license for most cases

### Regulatory Pathway
- Plan for FDA 510(k) / CE Class IIa pathway if commercializing
- Predicate device candidates: Materialise Mimics (FDA cleared), Synopsys Simpleware ScanFE
- Required for clinical use: IEC 62304 software lifecycle documentation, IEC 62366 usability testing, HIPAA BAA with hosting provider

---

## 14. Immediate Next Steps

1. **Initialize monorepo** with `pnpm` workspaces + Python `uv`
2. **Scaffold frontend** with `pnpm create vite frontend -- --template react-ts`
3. **Scaffold backend** with FastAPI + Celery + Redis docker-compose
4. **Configure two Celery queues** (`cpu_queue` + `gpu_queue`) in `config.py` and docker-compose
5. **Install and verify** TotalSegmentator on a test CT NIfTI file
6. **Prototype NiiVue** integration inside a React functional component
7. **Wire up** first complete path: DICOM series picker → chunked upload → NIfTI → scan_qc → NiiVue display
8. **Add session auth stub**: `/auth/session` endpoint + session_id binding on `/upload`

---

## 15. Coordinate System Reference

Every spatial boundary in VSP Engine crosses multiple coordinate frames. Bugs here cause mirrored meshes, wrong-axis crop lines, and misaligned SAM-Med3D clicks. This section is the single source of truth.

### Frame Definitions

| Frame | Origin | Axes | Used By |
|---|---|---|---|
| **DICOM Image** | Top-left of first slice | Row / Col / Slice (image space) | pydicom pixel arrays |
| **DICOM Patient (LPS)** | Center of scanner bore | L=Left, P=Posterior, S=Superior | DICOM Position/Orientation tags |
| **NIfTI RAS (world mm)** | per qform/sform affine | R=Right, A=Anterior, S=Superior | NiiVue, SimpleITK, TotalSegmentator, SAM-Med3D |
| **MeshLib world (mm)** | Matches NIfTI affine when loaded via `mr.loadVoxels()` | R/A/S world mm | Scout PLY meshes, CropBox min/max |
| **Three.js / R3F** | User defined; we set it = NIfTI RAS mm | Y-up right-hand | R3F viewport, CropBox handles, rayCast results |
| **NiiVue canvas pixel** | Top-left of NiiVue canvas | x=right, y=down (px) | MPRCropOverlay Canvas2D |
| **SAM-Med3D input** | Normalized [0,1] within the 128³ patch | d/h/w (z,y,x order) | `/refine/point` backend |

### Key Transforms

```
DICOM LPS  →  NIfTI RAS:   flip L→R and P→A  (R = −L, A = −P, S = S)
NIfTI RAS  →  R3F world:   direct (we set R3F scene units = mm, Y=S=Superior)
R3F world  →  SAM-Med3D:   (x_world, y_world, z_world) → crop to patch →
                             normalize to [0,1] → reorder to (z_norm, y_norm, x_norm)
NIfTI RAS mm → NiiVue canvas px:  nv.mm2frac(mmXYZ) → [0,1] fraction →
                             multiply by canvas width/height
```

### Implementation

```ts
// frontend/src/lib/coordTransforms.ts
// All world-space coordinates in VSP Engine use NIfTI RAS mm.
// R3F scene is initialized with this same frame (no additional transform needed
// if MeshLib PLY meshes are loaded directly from NIfTI voxels).

export function r3fWorldToNiftiVoxel(
  worldMm: [number, number, number],
  affineInv: number[][]  // 4×4 inverse of NIfTI qform affine
): [number, number, number] {
  // affineInv transforms RAS mm → voxel IJK
  const [i, j, k] = applyAffine(affineInv, worldMm)
  return [Math.round(i), Math.round(j), Math.round(k)]
}

export function niftiVoxelToSamMed3dPoint(
  voxelIJK: [number, number, number],
  cropOrigin: [number, number, number],  // voxel origin of ROI crop
  cropSize: [number, number, number]     // voxel size of ROI crop
): [number, number, number] {
  // SAM-Med3D expects normalized (z, y, x) within the 128³ patch
  const [i, j, k] = voxelIJK
  const [oi, oj, ok] = cropOrigin
  const [si, sj, sk] = cropSize
  return [(k - ok) / sk, (j - oj) / sj, (i - oi) / si]  // z,y,x norm
}
```

```python
# backend/app/services/coord_service.py
import numpy as np, SimpleITK as sitk

def world_mm_to_voxel_index(image: sitk.Image,
                             world_mm: tuple[float, float, float]) -> tuple[int, int, int]:
    """Convert NIfTI RAS world-space mm to voxel index (i,j,k).
    SimpleITK uses LPS internally; negate X and Y for RAS input.
    """
    lps = (-world_mm[0], -world_mm[1], world_mm[2])   # RAS → LPS
    idx = image.TransformPhysicalPointToIndex(lps)
    return tuple(idx)

def voxel_to_sam_med3d_point(voxel_ijk: tuple[int,int,int],
                              crop_origin: tuple[int,int,int],
                              crop_size: tuple[int,int,int]) -> tuple[float,float,float]:
    """Normalize voxel within ROI crop and reorder to SAM-Med3D (z,y,x)."""
    i, j, k = voxel_ijk
    oi, oj, ok = crop_origin
    si, sj, sk = crop_size
    return ((k-ok)/sk, (j-oj)/sj, (i-oi)/si)  # z_norm, y_norm, x_norm
```
