"""Sample Pydantic config classes used by ``valid_manifest.toml`` in tests.

These exist so the manifest's ``stages.*.config_class`` references resolve
inside the maldet repo's test environment (the real detector packages such
as ``elfrfdet`` live in separate repositories).

All classes follow the Phase 11e contract:
- ``BaseModel`` subclass
- ``model_config = ConfigDict(extra='forbid')``
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    n_estimators: int = 100


class EvaluateConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    threshold: float = 0.5


class PredictConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    batch_size: int = 32
