# VSP Engine — Copilot Agent Workspace

You are the AI coding agent for VSP Engine, a virtual surgery planning platform.

## What you can do in this workspace

- **Read / write all source files** via the filesystem MCP server
- **Search the web** for up-to-date library docs via the fetch MCP server
- **Manage GitHub issues, PRs, and Actions** via the GitHub MCP server
- **Browse running frontend** via the Playwright MCP server

## Key rules when modifying code

1. Follow all rules in `.github/instructions/copilot-instructions.md`
2. Pin `torch<=2.8.0` and `SimpleITK==2.0.2` in any Python requirements changes
3. Never commit `.dcm`, `.nii`, `.nii.gz`, or any file under `data/` or `models/`
4. Always call `topology.isClosed()` before returning a mesh download URL
5. Strip DICOM PHI at the upload boundary — never log patient name, DOB, MRN
6. Use Zustand, not Redux, for frontend global state

## Where things live

| What | Where |
|---|---|
| Architecture | `ARCHITECTURE.md` |
| Research notes & API patterns | `RESEARCH.md` |
| Frontend components | `frontend/src/components/` |
| Zustand stores | `frontend/src/store/` |
| FastAPI routes | `backend/app/routers/` |
| Segmentation + mesh services | `backend/app/services/` |
| Celery tasks | `backend/app/tasks/` |

## Common tasks

- **Scaffold frontend**: `cd frontend && pnpm install && pnpm dev`
- **Run backend**: `cd backend && uvicorn app.main:app --reload`
- **Run tests**: `cd backend && pytest tests/ -v`
- **Type-check frontend**: `cd frontend && pnpm exec tsc --noEmit`
