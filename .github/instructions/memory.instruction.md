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
  Research complete: See RESEARCH.md — 490-line technical reference covering TotalSegmentator API patterns, MeshLib DICOM load, NiiVue integration, ct2print/brain2print workflows, OHIF, nnU-Net warnings (torch≤2.8.0, SimpleITK==2.0.2 pin), MeshLib mesh repair order, ROI-crop-first strategy, two-mode segmentation (AI + HU threshold fallback), and licensing notes

## User Preferences

- Prefers incremental implementation with working intermediate states
- Wants doctor-friendly UI (no terminals, drag-and-drop, visual feedback)
- Non-commercial / planning tool - not certified medical device
- Uses pnpm for frontend package management
