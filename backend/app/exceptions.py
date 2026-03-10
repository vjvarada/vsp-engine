from __future__ import annotations


class VspEngineError(Exception):
    "Base domain exception."


class StudyNotFoundError(VspEngineError):
    "Study does not exist or belongs to another session."


class UploadFailedError(VspEngineError):
    "DICOM upload or dcm2niix conversion failed."


class ScanQcError(VspEngineError):
    "Scan failed quality check (non-blocking; surfaced as warning)."


class ScoutFailedError(VspEngineError):
    "MeshLib HU-threshold scout pass failed."


class SegmentOomError(VspEngineError):
    "TotalSegmentator ran out of GPU memory."


class SegmentFailedError(VspEngineError):
    "AI segmentation task failed."


class RefineFailedError(VspEngineError):
    "SAM-Med3D or MedSAM refinement failed."


class MeshNotWatertightError(VspEngineError):
    "Generated mesh is not watertight — STL export blocked."


class MeshFailedError(VspEngineError):
    "Mesh generation task failed."


class ExportFailedError(VspEngineError):
    "STL/OBJ export generation failed."


class StorageError(VspEngineError):
    "MinIO storage operation failed."


class AuthError(VspEngineError):
    "Session authentication error."
