#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from wm.dataset import load_twinworld_npz, mask_obs_for_model
from wm.env_utils import extract_ego_xy
from wm.metrics import compute_ade_fde, mean_and_se
from wm.models import MLPDynamics
from wm.utils import ensure_dir, save_json, set_global_seeds

K_LIST = [5, 10, 20, 30, 50]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate counterfactual rollout on Twin-World data.")
    p.add_argument("--twin_data", type=str, default="outputs/data/twinworld_merge.npz")
    p.add_argument("--ckpt", type=str, default="outputs/ckpts/mlp_action_conditioned.pt")
    p.add_argument("--ckpt_agnostic", type=str, default="")
    p.add_argument("--H", type=int, default=50)
    p.add_argument(
        "--struct_mode",
        type=str,
        default="none",
        choices=["none", "presence", "radius", "topk"],
        help="Structured mask mode applied to model inputs during rollout.",
    )
    p.add_argument("--radius", type=float, default=30.0, help="Used when struct_mode=radius.")
    p.add_argument("--topk", type=int, default=None, help="Used when struct_mode=topk.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", type=str, default="outputs/eval")
    p.add_argument("--out_name", type=str, default="eval_report.json")
    return p.parse_args()


def load_model(ckpt_path: str, device: torch.device) -> MLPDynamics:
    payload = torch.load(ckpt_path, map_location=device)
    model = MLPDynamics(
        obs_dim=int(payload["obs_dim"]),
        action_dim=int(payload["action_dim"]),
        action_agnostic=bool(payload["action_agnostic"]),
    ).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


def reshape_obs(flat_obs: np.ndarray, obs_shape: tuple[int, ...]) -> np.ndarray:
    if int(np.prod(obs_shape)) == int(flat_obs.shape[-1]):
        return flat_obs.reshape(obs_shape)
    return flat_obs


def rollout_predict(
    model: MLPDynamics,
    start_obs: np.ndarray,
    act_cf: int,
    act_future: np.ndarray,
    H: int,
    struct_mode: str,
    radius: float,
    topk: int | None,
    device: torch.device,
) -> np.ndarray:
    masked_start = mask_obs_for_model(start_obs, struct_mode=struct_mode, radius=radius, topk=topk)
    obs_t = torch.from_numpy(masked_start).float().unsqueeze(0).to(device)
    pred = []
    actions = [int(act_cf)] + [int(a) for a in np.asarray(act_future).tolist()]
    for a in actions[:H]:
        with torch.no_grad():
            act_t = torch.tensor([a], dtype=torch.long, device=device)
            nxt = model(obs_t, None if model.action_agnostic else act_t)
        nxt_np = nxt.squeeze(0).detach().cpu().numpy().astype(np.float32)
        pred.append(nxt_np)
        # Keep training/evaluation masking logic identical for autoregressive inputs.
        nxt_masked = mask_obs_for_model(nxt_np, struct_mode=struct_mode, radius=radius, topk=topk)
        obs_t = torch.from_numpy(nxt_masked).float().unsqueeze(0).to(device)
    return np.asarray(pred, dtype=np.float32)


def eval_one_model(
    model: MLPDynamics,
    model_name: str,
    idxs: np.ndarray,
    data: dict[str, Any],
    obs_shape: tuple[int, ...],
    position_mode: str,
    struct_mode: str,
    radius: float,
    topk: int | None,
    H: int,
    device: torch.device,
) -> dict[str, Any]:
    ades, fdes = [], []
    ade_at: dict[int, list[float]] = {k: [] for k in K_LIST}
    fde_at: dict[int, list[float]] = {k: [] for k in K_LIST}
    hint_used = []

    for i in idxs:
        prefix = np.asarray(data["obs_prefix"][i], dtype=np.float32)
        start_obs = prefix[-1]
        true_fut = np.asarray(data["fut_cf"][i], dtype=np.float32)
        cf_action = int(data["act_cf"][i])
        act_future = np.asarray(data["act_future"][i], dtype=np.int64)

        pred_fut = rollout_predict(
            model=model,
            start_obs=start_obs,
            act_cf=cf_action,
            act_future=act_future,
            H=H,
            struct_mode=struct_mode,
            radius=radius,
            topk=topk,
            device=device,
        )

        n = min(len(pred_fut), len(true_fut))
        if n == 0:
            continue
        pred_xy = []
        true_xy = []
        for t in range(n):
            pred_raw = reshape_obs(pred_fut[t], obs_shape)
            true_raw = reshape_obs(true_fut[t], obs_shape)
            p_xy, p_hint = extract_ego_xy(pred_raw, mode=position_mode)
            g_xy, g_hint = extract_ego_xy(true_raw, mode=position_mode)
            pred_xy.append(p_xy)
            true_xy.append(g_xy)
            hint_used.extend([p_hint, g_hint])

        pred_xy_arr = np.asarray(pred_xy)
        true_xy_arr = np.asarray(true_xy)
        ade, fde = compute_ade_fde(pred_xy_arr, true_xy_arr)
        ades.append(ade)
        fdes.append(fde)

        for k in K_LIST:
            kk = min(k, n)
            ade_k, fde_k = compute_ade_fde(pred_xy_arr[:kk], true_xy_arr[:kk])
            ade_at[k].append(ade_k)
            fde_at[k].append(fde_k)

    ade_mean, ade_se = mean_and_se(ades)
    fde_mean, fde_se = mean_and_se(fdes)
    ade_at_summary = {}
    fde_at_summary = {}
    for k in K_LIST:
        m_ade, se_ade = mean_and_se(ade_at[k])
        m_fde, se_fde = mean_and_se(fde_at[k])
        ade_at_summary[str(k)] = {"mean": m_ade, "se": se_ade}
        fde_at_summary[str(k)] = {"mean": m_fde, "se": se_fde}

    return {
        "model_name": model_name,
        "num_eval_samples": int(len(ades)),
        "CF-ADE_mean": ade_mean,
        "CF-ADE_se": ade_se,
        "CF-FDE_mean": fde_mean,
        "CF-FDE_se": fde_se,
        "ADE@K": ade_at_summary,
        "FDE@K": fde_at_summary,
        "position_extraction_used": sorted(list(set(hint_used))),
    }


def main() -> None:
    args = parse_args()
    set_global_seeds(args.seed)
    out_dir = ensure_dir(args.out_dir)
    out_path = out_dir / args.out_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_twinworld_npz(args.twin_data)
    obs_shape = tuple(np.asarray(data.get("obs_shape", np.array([-1]))).tolist())
    # Keep one fixed parsing mode for both models in comparison.
    position_mode = "kinematics_presence_xy"
    split = data.get("split", None)
    ep_ids = np.asarray(data["episode_ids"], dtype=np.int64)

    if split and "test" in split:
        test_eps = np.asarray(split["test"], dtype=np.int64)
        eval_mask = np.isin(ep_ids, test_eps)
        idxs = np.where(eval_mask)[0]
    else:
        idxs = np.arange(len(ep_ids))

    if len(idxs) == 0:
        raise RuntimeError("No test samples available in Twin-World dataset.")

    model_cf = load_model(args.ckpt, device)
    results = {
        "action_conditioned": eval_one_model(
            model=model_cf,
            model_name="action_conditioned",
            idxs=idxs,
            data=data,
            obs_shape=obs_shape,
            position_mode=position_mode,
            struct_mode=args.struct_mode,
            radius=args.radius,
            topk=args.topk,
            H=args.H,
            device=device,
        )
    }

    ckpt_agnostic = args.ckpt_agnostic
    if not ckpt_agnostic:
        sibling = os.path.join(os.path.dirname(args.ckpt), "mlp_action_agnostic.pt")
        if os.path.exists(sibling):
            ckpt_agnostic = sibling
    if ckpt_agnostic and os.path.exists(ckpt_agnostic):
        model_aa = load_model(ckpt_agnostic, device)
        results["action_agnostic"] = eval_one_model(
            model=model_aa,
            model_name="action_agnostic",
            idxs=idxs,
            data=data,
            obs_shape=obs_shape,
            position_mode=position_mode,
            struct_mode=args.struct_mode,
            radius=args.radius,
            topk=args.topk,
            H=args.H,
            device=device,
        )
    else:
        print("[eval_counterfactual] action_agnostic checkpoint not found, skipped.")

    assumption = (
        "Ego x,y use fixed kinematics_presence_xy parsing for both models; if unavailable, proxy may degrade."
    )
    report = {
        "twin_data": args.twin_data,
        "device": str(device),
        "num_test_samples": int(len(idxs)),
        "horizon": args.H,
        "struct_mask": {
            "mode": args.struct_mode,
            "radius": args.radius,
            "topk": args.topk,
        },
        "assumption": assumption,
        "position_mode": position_mode,
        "results": results,
        "comparison": {},
    }
    if "action_conditioned" in results and "action_agnostic" in results:
        ac = results["action_conditioned"]
        aa = results["action_agnostic"]
        report["comparison"] = {
            "CF-ADE_delta_agnostic_minus_conditioned": aa["CF-ADE_mean"] - ac["CF-ADE_mean"],
            "CF-FDE_delta_agnostic_minus_conditioned": aa["CF-FDE_mean"] - ac["CF-FDE_mean"],
        }

    save_json(out_path, report)
    print(f"[eval_counterfactual] Saved report: {out_path}")
    for k, v in results.items():
        print(
            f"[eval_counterfactual][{k}] "
            f"CF-ADE={v['CF-ADE_mean']:.6f}+/-{v['CF-ADE_se']:.6f}, "
            f"CF-FDE={v['CF-FDE_mean']:.6f}+/-{v['CF-FDE_se']:.6f}"
        )


if __name__ == "__main__":
    main()

