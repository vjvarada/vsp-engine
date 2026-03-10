from __future__ import annotations
import dataclasses
from functools import lru_cache
from app.config import get_settings
@dataclasses.dataclass(frozen=True)
class FeatureFlags:
    appendicular_bones: bool
    teeth: bool
    trunk_cavities: bool
@lru_cache
def get_feature_flags() -> FeatureFlags:
    settings = get_settings()
    has_license = bool(settings.totalseg_license_key)
    return FeatureFlags(
        appendicular_bones=has_license,
        teeth=has_license,
        trunk_cavities=True,
    )
