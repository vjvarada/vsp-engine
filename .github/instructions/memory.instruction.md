---
applyTo: "**"
---

# Copilot Memory — VSP Engine

## Project Identity

- **Project**: VSP Engine (Virtual Surgery Planning Platform)
- **Owner**: Vijay Raghav Varada (FracktalWorks)
- **Workspace**: `c:\Users\VijayRaghavVarada\Documents\Github\VSP Engine`

## Current Status

- [x] Architecture document written: `ARCHITECTURE.md`
- [x] Workspace settings: `.vscode/settings.json`
- [x] Copilot instructions: `.github/instructions/copilot-instructions.md`
- [x] Environment template: `.env.example`
- [ ] Frontend scaffold (Vite + React + TypeScript + Tailwind + Shadcn)
- [ ] Backend scaffold (FastAPI + Celery + Redis)
- [ ] Docker Compose setup
- [ ] DICOM upload route
- [ ] NiiVue integration
- [ ] TotalSegmentator Celery task
- [ ] MeshLib mesh pipeline
- [ ] ROI bounding box gizmo (R3F + Drei)
- [ ] STL export

## Key Technology Choices

- Frontend: React 19 + Vite + TypeScript + React Three Fiber + NiiVue + Shadcn/UI + Tailwind + Zustand
- Backend: Python 3.11 + FastAPI + Celery + Redis + TotalSegmentator + MONAI + MeshLib
- UI layout inspiration: RapidTool-Fixture (React web app, 4-column AppShell: icon toolbar → collapsible context panel → R3F viewport → properties panel, `tech-glass`/`font-tech` design system, cyan primary)
  Research complete: See RESEARCH.md — 900+ line technical reference covering TotalSegmentator API patterns (incl. 2025 subtasks: craniofacial_structures, teeth FDI, trunk_cavities), MeshLib DICOM load, NiiVue integration, ct2print/brain2print workflows, OHIF, nnU-Net warnings (torch≤2.8.0, SimpleITK==2.0.2 pin), MeshLib mesh repair order, ROI-crop-first strategy.
  **Extended segmentation landscape research completed**: Sections 2.4–2.12 added covering VISTA3D (NVIDIA NonComm ⚠️ — NO production use), MedSAM (bbox, Apache 2.0), MedSAM2 (3D video, torch==2.5.1 separate env), SAM-Med3D-turbo (3D point, medim one-liner, Apache 2.0), SuPreM (patents pending ⚠️), STU-Net (59-bone pre-train, Apache 2.0), MedNeXt-L k5, Auto3DSeg. Five-tier strategy validated: Tier1=HU<5s, Tier2=TotalSegmentator AI, Tier3=SAM-Med3D point, Tier4=MedSAM bbox, Tier5=STU-Net fine-tune.

## User Preferences

- Prefers incremental implementation with working intermediate states
- Wants doctor-friendly UI (no terminals, drag-and-drop, visual feedback)
- Non-commercial / planning tool - not certified medical device
- Uses pnpm for frontend package management
