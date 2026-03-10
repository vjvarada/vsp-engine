# MedSAM2 Isolation Service
# MUST run in a separate container from the main backend —
# requires torch==2.5.1 which is incompatible with TotalSegmentator (torch<=2.8.0)
FROM python:3.11-slim

WORKDIR /app

# Install CUDA-compatible PyTorch 2.5.1 (incompatible with TotalSegmentator — isolation required)
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu118

# Install SAM2 and dependencies
RUN pip install --no-cache-dir \
    git+https://github.com/facebookresearch/segment-anything-2.git \
    fastapi \
    "uvicorn[standard]" \
    nibabel \
    numpy \
    "SimpleITK==2.0.2" \
    boto3 \
    pydantic \
    "pydantic-settings"

COPY medsam2_server.py /app/

EXPOSE 8001

CMD ["uvicorn", "medsam2_server:app", "--host", "0.0.0.0", "--port", "8001"]
