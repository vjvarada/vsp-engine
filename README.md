# VSP Engine

> **FOR PLANNING PURPOSES ONLY — NOT FOR DIAGNOSTIC USE**

A virtual surgery planning web platform. Ingests DICOM CT/MR scans, runs AI bone segmentation (TotalSegmentator), lets surgeons crop a region of interest, and exports a watertight 3D-printable STL mesh.

---

## Architecture

| Layer | Technology |
|---|---|
| Frontend | React 19 + Vite + TypeScript + React Three Fiber + NiiVue + Shadcn/UI + Zustand |
| Backend | Python 3.11 + FastAPI + Celery + Redis + TotalSegmentator + MONAI + MeshLib |
| Infrastructure | Docker Compose + Nginx + NVIDIA Container Toolkit |

Full design: [ARCHITECTURE.md](ARCHITECTURE.md) | Research notes: [RESEARCH.md](RESEARCH.md)

---

## Workflow

```
Upload DICOM  View Volume (NiiVue)  AI Segment  Crop ROI  Generate Mesh  Export STL
```

**Two segmentation modes:**
- **AI mode** — TotalSegmentator (30 s GPU / 5 min CPU), 117-class labels, per-bone masks
- **Fast mode (<5 s)** — MeshLib DICOM direct load + marching cubes at Otsu HU threshold

---

## Quick Start (Development)

### Prerequisites

- Node.js  20 + pnpm
- Python 3.11+
- Docker + Docker Compose
- NVIDIA GPU + CUDA (optional; CPU fallback supported)

### 1. Clone & configure

```bash
git clone https://github.com/vjvarada/vsp-engine.git
cd vsp-engine
cp .env.example .env
# Edit .env — set SECRET_KEY, GPU_DEVICE, storage paths
```

### 2. Start backend services (Docker)

```bash
docker compose up -d redis
# With GPU:
docker compose up -d
# CPU-only:
GPU_DEVICE=cpu docker compose up -d
```

### 3. Start backend (local dev)

```bash
cd backend
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
# In a second terminal:
celery -A app.tasks worker --loglevel=info
```

### 4. Start frontend

```bash
cd frontend
pnpm install
pnpm dev
# Opens at http://localhost:5173
```

---

## Project Structure

```
vsp-engine/
 frontend/
    src/
        components/viewport/    # R3F + NiiVue canvas components
        components/sidebar/     # Workflow step panels (Upload, Segment, Crop, Mesh, Export)
        store/                  # Zustand slices
 backend/
    app/
        routers/                # FastAPI route handlers
        services/               # Business logic (dicom, segment, mesh)
        tasks/                  # Celery async tasks
 ARCHITECTURE.md
 RESEARCH.md
 docker-compose.yml
```

---

## Safety & Compliance

- DICOM tags anonymized at upload boundary (patient name, DOB, MRN stripped)
- All STL exports watermarked: **"FOR PLANNING PURPOSES ONLY — NOT FOR DIAGNOSTIC USE"**
- Mesh watertight check (`topology.isClosed()`) required before download is offered
- No patient data persisted beyond temporary session storage

---

## Key Version Pins

| Package | Pin | Reason |
|---|---|---|
| `torch` | `<=2.8.0` | nnU-Net 3D conv regression in  2.9.0 |
| `SimpleITK` | `==2.0.2` | TotalSegmentator hard requirement |
| `meshlib` | latest stable | MeshLib non-commercial license — see [RESEARCH.md](RESEARCH.md) |

---

## License

MIT  2025 vjvarada
