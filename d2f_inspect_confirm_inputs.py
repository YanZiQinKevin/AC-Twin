"""Read-only inspection: D2F screen quantities npz keys vs POW extended-vector requirements."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code"))

import numpy as np  # noqa: E402

npz = ROOT / "experiments/stage4_reliability/stage4d_d2f/development/screen_group_quantities.npz"
with np.load(npz, allow_pickle=False) as z:
    keys = sorted(z.files)
    print(f"screen_group_quantities.npz ({len(keys)} keys):")
    for k in keys:
        print(f"  {k}: shape={z[k].shape} dtype={z[k].dtype}")

receipt = ROOT / "experiments/stage4_reliability/stage4d_d2f/development/screen_group_quantities_receipt.json"
import json
doc = json.loads(receipt.read_text(encoding="utf-8"))
print("\nreceipt keys:", sorted(doc.keys()))

d2e_npz = ROOT / "experiments/stage4_reliability/stage4d_d2e/development/power_validation_quantities.npz"
with np.load(d2e_npz, allow_pickle=False) as z:
    d2e_keys = sorted(z.files)
print(f"\nD2E power_validation_quantities.npz ({len(d2e_keys)} keys):")
for k in d2e_keys:
    print(f"  {k}: shape={z[k].shape} dtype={z[k].dtype}")
