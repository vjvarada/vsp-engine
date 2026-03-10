from __future__ import annotations
import logging
from fastapi import FastAPI
from pydantic import BaseModel
app = FastAPI(title='MedSAM2 Isolation Service')
logger = logging.getLogger(__name__)
class PropagateRequest(BaseModel):
    study_id: str
    slice_idx: int
    bbox: list[float]
    plane: str = 'axial'
@app.post('/medsam2/propagate')
async def propagate(body: PropagateRequest):
    # TODO: load SAM2 model, run inference on slice, return mask
    logger.info('MedSAM2 study=%s slice=%d', body.study_id, body.slice_idx)
    return {'study_id': body.study_id, 'mask_key': f'studies/{body.study_id}/refine/bbox_{body.slice_idx}.nii.gz'}
