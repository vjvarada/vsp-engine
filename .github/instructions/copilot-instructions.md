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

#### Domain rules
- Always use strict TypeScript — no `any`, no implicit `any`. Use `unknown` + type guard instead.
- Prefer functional components with hooks; no class components
- Use Zustand slices for global state — avoid prop drilling > 2 levels
- React Three Fiber: use declarative JSX geometry (`<mesh>`, `<boxGeometry>`) not imperative Three.js
- Import R3F types from `@react-three/fiber`, helpers from `@react-three/drei`
- **ROI Crop Box: do NOT use `Drei TransformControls` for the crop gizmo** — it moves the whole box. Implement a custom `<CropBox>` with 6 separate face-center drag handles, each constrained to one axis (X⁻ X⁺ Y⁻ Y⁺ Z⁻ Z⁺). Color X=red, Y=green, Z=blue (DICOM/ITK convention). Clamp each axis: `min < max - MIN_SIZE`.
- **ROI Panel: 3 dual-range sliders** (one per axis, two thumbs) + 6 numeric mm inputs, all synced to `scoutStore.roi` — not just a single XYZ triplet
- **MPR crop lines: Canvas 2D overlay** drawn on top of NiiVue canvas (position: absolute). Axial MPR shows Z⁻/Z⁺ blue lines; Coronal shows Y⁻/Y⁺ green; Sagittal shows X⁻/X⁺ red. Use `nv.mm2frac()` to convert world mm to canvas pixel positions.
- NiiVue: always wrap in a `useEffect` with cleanup to call `nv.dispose()` on unmount

#### Naming
- Components: PascalCase (`ScoutPanel`, `CropBox`). Hooks: `usePrefix` camelCase (`useScoutIslands`). Event handlers: `handle` prefix (`handleIslandClick`). Booleans: `is`/`has`/`can` prefix (`isLocked`, `canExport`). Module constants: SCREAMING_SNAKE_CASE (`MIN_ROI_SIZE_MM`). Utility files: kebab-case (`coord-transforms.ts`).

#### TypeScript strictness
- `tsconfig.json`: `strict: true`, `exactOptionalPropertyTypes: true`, `noUncheckedIndexedAccess: true`, `noImplicitReturns: true`
- Use `satisfies` to validate config object shape without widening the type
- `interface` for public API contracts; `type` for unions, mapped types, intersections
- `import type { ... }` for all type-only imports
- Model task outcomes as a discriminated union: `type TaskResult<T> = {status:'idle'} | {status:'running';percent:number} | {status:'done';data:T} | {status:'error';message:string}`
- Zustand slices: `interface XState { ... } & interface XActions { ... }` — one interface for data, one for mutations

#### Cyclomatic complexity (max 5 branches per function)
- Guard clauses + early return instead of nested `if` blocks
- Replace `if/else` chains with `Record<K, V>` lookup tables: `const TASK_BY_REGION = { skull: 'craniofacial_structures', ... } as const`
- No nested ternaries — extract to named variable or function

#### Cognitive complexity (component max 120 lines, function max 40 lines)
- Extract sub-components for list items (`<IslandListItem>`, `<AxisSlider>`, `<LabelRow>`)
- Extract custom hook for any `useEffect`+state pair that exceeds 10 lines
- Compute all derived values in variables above the `return` — no filter/map/sort inline inside JSX
- No side effects in render path — all fetching lives in `useEffect` or TanStack Query

#### React patterns
- `React.memo`: only for components with stable-reference props that re-render frequently (individual R3F meshes, list rows)
- `useCallback`: only when passed to a `React.memo` child or in a `useEffect` dep array
- `useMemo`: only for computations measurably >1 ms — not for object/array literals
- `useRef`: for NiiVue instance, Three.js geometry/material, animation frame IDs
- Dispose R3F `geometry` + `material` in `useEffect` cleanup. Dispose NiiVue with `nv.dispose()`.
- No inline arrow functions as props on `React.memo` children — wrap with `useCallback` instead
- One component per file. No multi-export component files.

#### Accessibility
- All icon-only buttons: `aria-label` required
- All sliders: `aria-label`, `aria-valuemin`, `aria-valuemax`, `aria-valuenow`
- Progress bars: `role="progressbar"`, `aria-valuenow`, `aria-valuemin={0}`, `aria-valuemax={100}`
- R3F `<Canvas>`: `aria-label="3D [description] viewport"`, `role="img"`
- Color-coded information always paired with a text label — never color alone

#### Dead code hygiene
- No `console.log/warn/error` — import from `src/lib/logger.ts` (`logger.debug/info/warn/error`)
- No commented-out code blocks — use git history
- No unused imports (ESLint `@typescript-eslint/no-unused-vars` in CI)

### Python / Backend

#### Domain rules
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
- **Two Celery queues required**: `cpu_queue` (concurrency=4) for scout, scan_qc, mesh tasks; `gpu_queue` (concurrency=1) for segment + refine. Workers bind to named queues. Never run two GPU inference tasks concurrently.
- **NIfTI is the universal format**: dcm2niix converts DICOM→NIfTI once at upload. All subsequent pipeline stages (NiiVue, MeshLib, TotalSegmentator, SAM-Med3D) read from `volume.nii.gz`. Never re-read DICOM after upload.
- **MedSAM2 isolation**: MedSAM2 runs in `docker/medsam2/` (separate Dockerfile, `torch==2.5.1`, separate GPU device env var `GPU_DEVICE_MEDSAM2`). Main backend calls it via `POST /medsam2/propagate {study_id, slice_idx, bbox}` over internal network.
- **appendicular_bones feature gate**: check `TOTALSEG_LICENSE_KEY` env var in `feature_flags.py`; expose `/config/features` endpoint; SelectPanel shows "License required" badge if flag is False; fallback to `total` task.
- **Coordinate transforms**: all world-space coordinates in VSP Engine use NIfTI RAS mm. R3F scene units = mm, Y = Superior. Conversion helpers live in `frontend/src/lib/coordTransforms.ts` and `backend/app/services/coord_service.py`. SAM-Med3D expects (z_norm, y_norm, x_norm) within the 128³ patch — always use `niftiVoxelToSamMed3dPoint()`. Never guess coordinate order.
- **Scan QC task** (cpu_queue): runs after upload; checks slice thickness >3mm, z-gap, HU range, isotropy; returns non-blocking warnings to frontend UploadPanel.
- **Session auth**: every request carries `session_id` (JWT or HttpOnly signed cookie). Backend binds study to session_id; 403 if mismatch. Issue via `/auth/session`.
- **Chunked upload**: use tus protocol in frontend; `/upload/chunk` + `/upload/finalize` on backend; Nginx `client_max_body_size 1g`.
- **DICOM series picker**: dcmjs parses series in a WebWorker before upload; show thumbnail grid grouped by SeriesInstanceUID; surgeon selects the correct series.
- **Error states**: every Zustand slice has `taskStatus: 'idle' | 'running' | 'done' | 'error'` and `errorMessage: string | null`. Each panel shows its own inline error banner (not global toast) with a Retry action.
- **Multi-structure export**: ExportPanel shows per-label combine/separate toggle. `/mesh/export` supports `union=True` (single STL) and `union=False` (ZIP of per-label STL files).
- **Measurements**: MeshPanel exposes point-to-point caliper and 3-point angle tools via R3F rayCast. Results stored in `planReportStore`.
- **Implant overlay**: `Drei TransformControls` is the correct tool for moving implant template STL overlays — NOT for the crop box. Implant transform stored in `implantStore`.
- **planReportStore**: captures viewport screenshot (`gl.domElement.toDataURL()`), measurement results, surgeon notes, export manifest. "Download Report" → PDF.
- **Study data TTL**: MinIO lifecycle policy auto-deletes study files after `STUDY_TTL_DAYS` (default 30). Patient data must never persist beyond session.
- Never log patient PHI; anonymize DICOM tags at ingestion boundary

#### Naming
- Functions/variables: snake_case. Classes (Pydantic/dataclass): PascalCase. Module constants: SCREAMING_SNAKE_CASE. Private helpers: `_` prefix (`_apply_morphological_close`). Celery task names: `module.action` string (`"scout.run_scout_pass"`).

#### Type strictness
- `from __future__ import annotations` at the top of every module
- Complete type hints on all public functions — mypy `strict = true` enforced in CI
- `TypeAlias` for complex nested types: `IslandList: TypeAlias = list[IslandMeta]`
- `Protocol` for storage/model interfaces (enables DI and unit testing without real GPU/MinIO)
- Pydantic v2 response models: `model_config = ConfigDict(frozen=True)`

#### Module boundary rules (enforced by architecture, no exceptions)
- Routers → Services only. Tasks → Services only.
- Services must NOT import from routers or Celery tasks.
- Services must NOT accept FastAPI `Request`/`Response` objects.
- Routers contain zero business logic — parse request, call service, return response.
- Celery tasks contain zero business logic — serialize args, call service, update task state.

#### Cyclomatic complexity (max 8 per function — Ruff `mccabe`)
- Dispatch tables instead of `elif` chains: `_HINT_TO_TASK: dict[str, str] = { ... }` then `.get(hint, 'total')`
- Guard clauses + early return instead of nested `if` blocks

#### Error handling
- Typed domain exceptions defined in `app/exceptions.py` (`ScoutFailed`, `SegmentOOM`, `MeshNotWatertight`)
- FastAPI: one `@app.exception_handler` per exception type → HTTP status — no `raise HTTPException` inside services
- Celery tasks: always wrap service call with try/except → `self.update_state(state="FAILURE", meta={"error": str(exc), "type": type(exc).__name__})`; then re-raise
- Never swallow exceptions (`except Exception: pass` forbidden — Ruff `B001`)

#### Logging
- `logger = logging.getLogger(__name__)` in every module — never `print()`
- Log levels: DEBUG (mesh stats, voxel counts), INFO (task start/complete), WARNING (scan QC issues), ERROR (task failures)
- Never log PHI — log `study_id` UUID only

### Medical Data / Safety

- All DICOM data must be anonymized at upload — strip patient name, DOB, MRN, AccessionNumber, InstitutionName
- Add "FOR PLANNING PURPOSES ONLY — NOT FOR DIAGNOSTIC USE" watermark on all exports (STL binary header + report PDF)
- STL files must pass watertight check before download is offered to user
- All storage must use AES-256 encryption at rest (MinIO SSE-S3 or equivalent)
- Nginx must terminate TLS; all HTTP redirects to HTTPS in production
- Audit log: write to `audit_log` Postgres table for every upload, task run, and export (session_id, action, study_id, timestamp — no PHI values)

### File Structure

```
frontend/src/components/viewport/  — R3F + NiiVue components only
frontend/src/components/sidebar/   — Workflow step panels
frontend/src/store/                — Zustand slices (all include taskStatus + errorMessage)
frontend/src/lib/coordTransforms.ts — EXPLICIT coordinate-system transforms (RAS↔voxel↔R3F↔SAM)
backend/app/routers/               — FastAPI route handlers
backend/app/services/              — Business logic (dicom, scan_qc, scout, segment, mesh, coord, audit, feature_flags)
backend/app/tasks/                 — Celery tasks (cpu_queue: scout/scan_qc/mesh; gpu_queue: segment/refine)
docker/medsam2/                    — Isolated MedSAM2 service (torch==2.5.1, separate GPU)
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
- Do not store patient data beyond temporary session storage (enforce TTL via MinIO lifecycle)
- Do not skip mesh healing before STL export
- Do not use `console.log` in production code — use a logger utility
- **Do not use VISTA3D in any production, commercial, or clinical path** — NVIDIA OneWay Noncommercial License prohibits it
- **Do not run MedSAM2 in the same Python process or container as TotalSegmentator** — incompatible torch versions (`torch==2.5.1` vs `torch<=2.8.0`); use `docker/medsam2/` service with internal HTTP API
- Do not use SuPreM weights in production without verifying the license file ("patents pending" — not a standard OSS license)
- Do not offer STL download until `topology.isClosed()` returns `True`
- **Do not re-read DICOM files after upload** — dcm2niix runs once at upload; all pipeline stages use NIfTI
- **Do not use `Drei TransformControls` for the ROI crop box** — use the custom 6-handle `<CropBox>`. TransformControls is correct ONLY for implant template overlay positioning.
- Do not guess coordinate axis order — always use `coordTransforms.ts` / `coord_service.py` helpers
- Do not offer `appendicular_bones` anatomy in the UI without checking the `feature_flags.appendicular_bones` API response
- Do not skip error state in Zustand slices — every slice must have `taskStatus` and `errorMessage` fields
- Do not run two GPU inference Celery tasks concurrently — `gpu_queue` must have `concurrency=1`
