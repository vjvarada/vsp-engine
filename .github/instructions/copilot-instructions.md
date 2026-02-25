---
applyTo: "**"
---

# VSP Engine — GitHub Copilot Instructions

## Project Overview

VSP Engine is a virtual surgery planning web platform. It ingests DICOM CT/MR scans, runs AI bone segmentation (TotalSegmentator + MONAI), lets surgeons select a region of interest, and exports a watertight 3D-printable STL mesh. It is NOT a certified medical device — always label outputs as "FOR PLANNING PURPOSES ONLY."

> **Reference docs**: See [RESEARCH.md](../../RESEARCH.md) for validated API patterns, version pins, licensing notes, and workflow decisions derived from 19 open-source reference projects (TotalSegmentator, MeshLib, NiiVue, ct2print, brain2print, OHIF, VISTA3D, MedSAM, SAM-Med3D, SuPreM, STU-Net, MedNeXt, Auto3DSeg, etc.). See [ARCHITECTURE.md](../../ARCHITECTURE.md) for the full system design.

## Architecture

- **Frontend**: React 19 + Vite + TypeScript + React Three Fiber (R3F) + NiiVue + Shadcn/UI + Tailwind + Zustand
- **Backend**: Python 3.11+ FastAPI + Celery + Redis + TotalSegmentator + MONAI + MeshLib
- **UI Shell**: Directly adapted from `RapidTool-Fixture` (see `C:\Users\VijayRaghavVarada\Documents\Github\RapidTool-Fixture`) — 4-column AppShell layout, `tech-glass`/`font-tech`/`tech-transition` CSS utilities, cyan primary `hsl(198 89% 50%)`, `ContextOptionsPanel` with mini-map, `VerticalToolbar`

## Coding Standards

### TypeScript / React

- Always use strict TypeScript — no `any`, no implicit `any`
- Prefer functional components with hooks; no class components
- Use Zustand slices for global state — avoid prop drilling > 2 levels
- React Three Fiber: use declarative JSX geometry (`<mesh>`, `<boxGeometry>`) not imperative Three.js
- Import R3F types from `@react-three/fiber`, helpers from `@react-three/drei`
- **ROI Crop Box: do NOT use `Drei TransformControls` for the crop gizmo** — it moves the whole box. Implement a custom `<CropBox>` with 6 separate face-center drag handles, each constrained to one axis (X⁻ X⁺ Y⁻ Y⁺ Z⁻ Z⁺). Color X=red, Y=green, Z=blue (DICOM/ITK convention). Clamp each axis: `min < max - MIN_SIZE`.
- **ROI Panel: 3 dual-range sliders** (one per axis, two thumbs) + 6 numeric mm inputs, all synced to `scoutStore.roi` — not just a single XYZ triplet
- **MPR crop lines: Canvas 2D overlay** drawn on top of NiiVue canvas (position: absolute). Axial MPR shows Z⁻/Z⁺ blue lines; Coronal shows Y⁻/Y⁺ green; Sagittal shows X⁻/X⁺ red. Use `nv.mm2frac()` to convert world mm to canvas pixel positions.
- NiiVue: always wrap in a `useEffect` with cleanup to call `nv.dispose()` on unmount
- File naming: PascalCase for components, camelCase for hooks (useMyHook.ts), kebab-case for utils

### Python / Backend

- Python 3.11+ syntax — use `match`, `|` union types, `tomllib` etc.
- FastAPI: use `async def` for I/O routes, sync worker for CPU-bound (via `run_in_executor`)
- Always use Pydantic v2 models for request/response schemas
- MeshLib: `import meshlib.mrmeshpy as mr` — always use the `mr.` prefix
- TotalSegmentator: call via Python API `totalsegmentator(input_img, task="total")`; never shell out; always pin `torch<=2.8.0` and `SimpleITK==2.0.2` (nnU-Net 3D conv regression in torch ≥ 2.9.0)
- TotalSegmentator 2025 new subtasks: `craniofacial_structures` (mandible, maxillary/frontal sinuses, teeth_upper/teeth_lower, skull, head), `teeth` (individual FDI notation with pulp+canal — CVPR 2025 ToothFairy3), `trunk_cavities`. Use for maxillofacial and dental planning.
- Five-tier segmentation strategy: Tier 1 = MeshLib HU-threshold <5 s; Tier 2 = TotalSegmentator auto AI; Tier 3 = SAM-Med3D-turbo 3D point prompts; Tier 4 = MedSAM 2D bbox prompts; Tier 5 = STU-Net-B fine-tuned on hospital data
- **Scout → Select → AI → Refine → Export** is the canonical VSP Engine workflow: MeshLib HU threshold runs first (no GPU, 5–10 s) to generate per-island connected-component bone meshes; surgeon clicks islands to define ROI; TotalSegmentator runs only on the ROI crop (10–58× smaller volume); AI labels replace scout islands. Never run TotalSegmentator on the full volume when a surgeon-selected ROI is available.
- SAM-Med3D-turbo (Apache 2.0): `import medim; model = medim.create_model("SAM-Med3D", pretrained=True)` — surgeon 3D click in R3F viewport → binary 3D mask; input point in (z,y,x) format
- MedSAM (Apache 2.0): surgeon draws 2D bbox on NiiVue MPR slice → `[x1,y1,x2,y2]` + slice_idx → MedSAM inference → stack masks across z → 3D binary volume
- MedSAM2 (Apache 2.0): requires `torch==2.5.1` — **always run in a separate Docker service or conda env**, never in the same process as TotalSegmentator
- STU-Net-B (Apache 2.0, 58M params): best fine-tuning backbone for custom bone segmentation (pre-trained on 59 bones via TotalSegmentator atlas); fine-tune via nnU-Net v2 `run_finetuning.py -pretrained_weights`
- VISTA3D (NVIDIA OneWay Noncommercial): **DO NOT use in any production or clinical path** — see DO NOT section
- Always crop ROI before calling TotalSegmentator to avoid GPU OOM on large volumes
- After mesh generation always call `topology.isClosed()` before offering STL download
- Celery tasks: always update task state with `self.update_state(state="PROGRESS", meta={"percent": n})`
- Never log patient PHI; anonymize DICOM tags at ingestion boundary

### Medical Data / Safety

- All DICOM data must be anonymized at upload — strip patient name, DOB, MRN
- Add "FOR PLANNING PURPOSES ONLY — NOT FOR DIAGNOSTIC USE" watermark on all exports
- STL files must pass watertight check before download is offered to user

### File Structure

```
frontend/src/components/viewport/  — R3F + NiiVue components only
frontend/src/components/sidebar/   — Workflow step panels
frontend/src/store/                — Zustand slices
backend/app/routers/               — FastAPI route handlers
backend/app/services/              — Business logic (dicom, segment, mesh)
backend/app/tasks/                 — Celery async tasks
```

## Key Libraries — Quick Reference

| Library           | Import Pattern                                                              |
| ----------------- | --------------------------------------------------------------------------- |
| React Three Fiber | `import { Canvas, useFrame, useThree } from "@react-three/fiber"`           |
| Drei              | `import { TransformControls, OrbitControls, Box } from "@react-three/drei"` |
| NiiVue            | `import { Niivue } from "@niivue/niivue"`                                   |
| Zustand           | `import { create } from "zustand"`                                          |
| MeshLib Python    | `import meshlib.mrmeshpy as mr`                                             |
| FastAPI           | `from fastapi import FastAPI, UploadFile, BackgroundTasks`                  |
| TotalSegmentator  | `from totalsegmentator.python_api import totalsegmentator`                  |
| MONAI             | `from monai.bundle import ConfigParser`                                     |
| SAM-Med3D-turbo   | `import medim; model = medim.create_model("SAM-Med3D", pretrained=True)`    |
| MedSAM            | `from segment_anything import sam_model_registry` (medsam_vit_b.pth ckpt)  |
| STU-Net           | nnU-Net v2 trainer: `STUNetTrainer_base_ft` (fine-tune) or `STUNetTrainer_base` (scratch) |

## DO NOT

- Do not use Redux — use Zustand
- Do not use class components
- Do not shell out to subprocesses for TotalSegmentator/MONAI/SAM-Med3D/MedSAM — use Python APIs
- Do not store patient data beyond temporary session storage
- Do not skip mesh healing before STL export
- Do not use `console.log` in production code — use a logger utility
- **Do not use VISTA3D in any production, commercial, or clinical path** — NVIDIA OneWay Noncommercial License prohibits it
- **Do not run MedSAM2 in the same Python process or container as TotalSegmentator** — incompatible torch versions (`torch==2.5.1` vs `torch<=2.8.0`)
- Do not use SuPreM weights in production without verifying the license file ("patents pending" — not a standard OSS license)
- Do not offer STL download until `topology.isClosed()` returns `True`
