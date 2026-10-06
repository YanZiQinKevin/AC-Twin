"""Print pinned package versions for the conversion environment."""
import importlib.metadata as md

for name in ["numpy", "protobuf", "tensorflow", "waymo-open-dataset-tf-2-11-0", "metadrive-simulator", "scenarionet"]:
    try:
        print(name, md.version(name))
    except md.PackageNotFoundError:
        print(name, "NOT_INSTALLED")
