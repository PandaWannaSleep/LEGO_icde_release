from __future__ import annotations

import sys
from dataclasses import dataclass

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal


InferenceMode = Literal["single_step", "iterative"]


@dataclass(frozen=True)
class InferenceConfig:
    mode: InferenceMode = "single_step"
    max_iter: int = 10
    eps_adj: float = 4e-5
    return_final_adj: bool = False
    min_prediction_floor: float = 0.0

