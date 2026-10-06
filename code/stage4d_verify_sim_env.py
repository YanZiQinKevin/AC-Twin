"""Fail-loud verification for the frozen Stage 4D-D2 simulation environment."""

from __future__ import annotations

import importlib.util
import json
import platform
from pathlib import Path

import numpy as np
import torch
import metadrive
import scenarionet
import scipy
import sklearn


EXPECTED_METADRIVE = Path(r"D:\MetaDrive\metadrive").resolve()
EXPECTED_SCENARIONET = Path(r"D:\MetaDrive\scenarionet").resolve()


def _inside(module_file: str, expected_root: Path) -> bool:
    return Path(module_file).resolve().is_relative_to(expected_root)


assert platform.python_version_tuple()[:2] == ("3", "11")
assert torch.__version__ == "2.9.1+cu130"
assert torch.version.cuda == "13.0"
assert torch.cuda.is_available()
assert _inside(metadrive.__file__, EXPECTED_METADRIVE)
assert _inside(scenarionet.__file__, EXPECTED_SCENARIONET)
assert importlib.util.find_spec("tensorflow") is None

print(
    json.dumps(
        {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_build": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "metadrive_source": str(EXPECTED_METADRIVE),
            "scenarionet_source": str(EXPECTED_SCENARIONET),
            "tensorflow_present": False,
            "status": "SIM_ENV_VERIFY_PASS",
        },
        ensure_ascii=False,
        sort_keys=True,
    )
)
