"""Diagnostic: inspect ego v_s under FASTER/SLOWER CF branches in a role npz.

Engineering aid only (pre-contract-freeze). Distinguishes a controller bug from
scenario congestion by showing whether the ego's longitudinal speed responds to the
FASTER/SLOWER meta-action, and the factual speed level.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "code") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "code"))

import stage4d_metadrive_adapter as A

CH_S, CH_D, CH_VS = 2, 3, 4


def main(npz_path: str) -> None:
    with np.load(npz_path, allow_pickle=False) as z:
        g = {k: z[k] for k in z.files}
    cf = g["object_counterfactual_futures"]        # (G,10,20,5,8)
    fact = g["object_factual_future"]              # (G,20,5,8)
    cf_actions = g["cf_actions"]
    cf_durations = g["cf_durations"]
    print("plan table (idx: intent, dur):", [(i, int(cf_actions[i]), int(cf_durations[i])) for i in range(len(cf_actions))])
    G = cf.shape[0]
    faster_idx = [i for i in range(len(cf_actions)) if int(cf_actions[i]) == 3]
    slower_idx = [i for i in range(len(cf_actions)) if int(cf_actions[i]) == 4]
    print(f"groups={G} faster_plans={faster_idx} slower_plans={slower_idx}")
    print("factual ego v_s[0,-1] per group:", np.round(fact[:, [0, -1], 0, CH_VS], 2).tolist())
    print("factual ego s progress per group:", np.round(fact[:, -1, 0, CH_S] - fact[:, 0, 0, CH_S], 2).tolist())
    for label, idxs in (("FASTER", faster_idx), ("SLOWER", slower_idx)):
        for i in idxs:
            vs = cf[:, i, :, 0, CH_VS]             # (G,20)
            dv = vs[:, -1] - vs[:, 0]
            print(f"{label} plan{i} (dur {int(cf_durations[i])}): "
                  f"v_s[t0] median={np.median(vs[:,0]):.2f} "
                  f"v_s[tend] median={np.median(vs[:,-1]):.2f} "
                  f"dv median={np.median(dv):.2f} frac>|1|={np.mean(np.abs(dv)>=1.0):.2f}")
            for grp in range(min(G, 6)):
                print(f"    g{grp} v_s: {np.round(vs[grp, [0,3,7,11,15,19]],2).tolist()} dv={dv[grp]:.2f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         str(PROJECT_ROOT / "experiments/stage4_reliability/stage4d_external/roles/smoke/groups.npz"))
