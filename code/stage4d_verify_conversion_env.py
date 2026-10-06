"""Quick import verification for the Stage 4D-D2 conversion environment."""
import platform

import tensorflow as tf
import metadrive  # noqa: F401
import scenarionet  # noqa: F401
from scenarionet.converter.waymo.utils import convert_waymo_scenario  # noqa: F401

print("python", platform.python_version())
print("tf", tf.__version__)
print("converter import OK")
