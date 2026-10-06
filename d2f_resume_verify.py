"""D2F resume read-only anchor re-verification (plan section 0.1). Writes nothing."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code"))

from ac_twin.stage1_data import sha256_file  # noqa: E402
from ac_twin import stage4d_d2f_protocol as P  # noqa: E402

D = Path("experiments/stage4_reliability/stage4d_d2f")

bad = []
for rel, sha, _ in P.PLAN_LISTED_SOURCES:
    p = Path(rel) if rel.startswith("code/") else Path("experiments/stage4_reliability") / rel
    if not p.exists() or sha256_file(p) != sha:
        bad.append(rel)
print("plan-listed SHA mismatches:", bad if bad else "NONE (9/9 match)")

for name, key in [
    ("fresh_id_verification.json", "status"),
    ("pool_capacity_audit.json", "status"),
    ("pool_capacity_audit_post_smoke.json", "status"),
    ("power/power_correction_report.json", "status"),
]:
    doc = json.loads((D / name).read_text(encoding="utf-8"))
    print(f"{name}: {doc.get(key)}")

parity = json.loads((D / "development/scorer_parity_receipt.json").read_text(encoding="utf-8"))
print("scorer_parity passed:", parity["passed"])
print("path_a:", parity["path_a_composition_parity"])
print("path_b envelope:", parity["path_b_live_reinference_envelope"]["max_abs_diff"])

print("--- one-shot resources must all be ABSENT ---")
for label, path in [
    ("screen_manifest.json", D / "screen_manifest.json"),
    ("screen_protocol_lock.json", D / "screen_protocol_lock.json"),
    ("development_screen_used.lock", D / "development" / "development_screen_used.lock"),
    ("code_inventory.json", D / "code_inventory.json"),
    ("model_inventory.json", D / "model_inventory.json"),
    ("roles/development_screen", D / "roles" / "development_screen"),
    ("confirm dir", D / "confirm"),
]:
    print(f"{label}: {'EXISTS!!' if path.exists() else 'absent (ok)'}")
