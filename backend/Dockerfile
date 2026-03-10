# VSP Engine — Backend Dockerfile
# FOR PLANNING PURPOSES ONLY — NOT FOR DIAGNOSTIC USE
#
# NOTE: torch is pinned to <=2.8.0 — do NOT upgrade.
# nnU-Net (used by TotalSegmentator) has a 3D conv regression in torch >=2.9.0.
# MedSAM2 MUST run in docker/medsam2/ (separate service, torch==2.5.1).

FROM python:3.11-slim as base

WORKDIR /app

# System dependencies for dcm2niix, MeshLib, and NIfTI processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    dcm2niix \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch BEFORE other packages (pinned <=2.8.0)
RUN pip install --no-cache-dir \
    "torch<=2.8.0" \
    torchvision \
    --index-url https://download.pytorch.org/whl/cu118

# Install project dependencies
COPY pyproject.toml /app/
RUN pip install --no-cache-dir \
    totalsegmentator \
    meshlib \
    && pip install --no-cache-dir -e ".[dev]"

COPY app/ /app/app/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]

