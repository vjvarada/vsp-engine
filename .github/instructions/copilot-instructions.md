---
applyTo: "**"
---

# VSP Engine — GitHub Copilot Instructions

## Project Overview

VSP Engine is a virtual surgery planning web platform. It ingests DICOM CT/MR scans, runs AI bone segmentation (TotalSegmentator + MONAI), lets surgeons select a region of interest, and exports a watertight 3D-printable STL mesh. It is NOT a certified medical device — always label outputs as "FOR PLANNING PURPOSES ONLY."

> **Reference docs**: See [RESEARCH.md](../../RESEARCH.md) for validated API patterns, version pins, licensing notes, and workflow decisions derived from 11 open-source reference projects (TotalSegmentator, MeshLib, NiiVue, ct2print, brain2print, OHIF, etc.). See [ARCHITECTURE.md](../../ARCHITECTURE.md) for the full system design.

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
- NiiVue: always wrap in a `useEffect` with cleanup to call `nv.dispose()` on unmount
- File naming: PascalCase for components, camelCase for hooks (useMyHook.ts), kebab-case for utils

### Python / Backend

- Python 3.11+ syntax — use `match`, `|` union types, `tomllib` etc.
- FastAPI: use `async def` for I/O routes, sync worker for CPU-bound (via `run_in_executor`)
- Always use Pydantic v2 models for request/response schemas
- MeshLib: `import meshlib.mrmeshpy as mr` — always use the `mr.` prefix
- TotalSegmentator: call via Python API `totalsegmentator(input_img, task="total")`; never shell out; always pin `torch<=2.8.0` and `SimpleITK==2.0.2` (nnU-Net 3D conv regression in torch ≥ 2.9.0)
- Two-mode segmentation: Mode 1 = AI (TotalSegmentator); Mode 2 = HU-threshold fallback via `mr.VoxelsLoad.loadDicomsFolderTreeAsVdb()` + marching cubes at Otsu HU, completes in <5 s
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

## DO NOT

- Do not use Redux — use Zustand
- Do not use class components
- Do not shell out to subprocesses for TotalSegmentator/MONAI — use Python APIs
- Do not store patient data beyond temporary session storage
- Do not skip mesh healing before STL export
- Do not use `console.log` in production code — use a logger utility
