#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from wm.dataset import FactualTransitionDataset, episode_level_split, load_factual_npz
from wm.models import MLPDynamics
from wm.utils import ensure_dir, save_json, set_global_seeds


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train baseline one-step dynamics model(s).")
    p.add_argument("--data", type=str, default="outputs/data/factual_transitions.npz")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--train_ratio", type=float, default=0.7)
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument(
        "--struct_mode",
        type=str,
        default="none",
        choices=["none", "presence", "radius", "topk"],
        help="Structured mask mode applied to obs input.",
    )
    p.add_argument("--radius", type=float, default=30.0, help="Used when struct_mode=radius.")
    p.add_argument("--topk", type=int, default=None, help="Used when struct_mode=topk.")
    p.add_argument("--out_dir", type=str, default="outputs/ckpts")
    return p.parse_args()


def make_loader(
    obs: np.ndarray,
    act: np.ndarray,
    nxt: np.ndarray,
    batch_size: int,
    shuffle: bool,
    struct_mode: str,
    radius: float,
    topk: int | None,
):
    ds = FactualTransitionDataset(
        obs=obs,
        acts=act,
        next_obs=nxt,
        struct_mode=struct_mode,
        radius=radius,
        topk=topk,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def run_epoch(
    model: MLPDynamics,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> float:
    train = optimizer is not None
    model.train(train)
    losses = []
    mse = torch.nn.MSELoss()
    for obs, act, nxt in loader:
        obs, act, nxt = obs.to(device), act.to(device), nxt.to(device)
        pred = model(obs, act if not model.action_agnostic else None)
        loss = mse(pred, nxt)
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    return float(np.mean(losses)) if losses else float("nan")


def train_single_model(
    model_name: str,
    action_agnostic: bool,
    obs_train: np.ndarray,
    act_train: np.ndarray,
    nxt_train: np.ndarray,
    obs_val: np.ndarray,
    act_val: np.ndarray,
    nxt_val: np.ndarray,
    obs_dim: int,
    action_dim: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[MLPDynamics, list[dict[str, float]], int]:
    model = MLPDynamics(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
        action_agnostic=action_agnostic,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_loader = make_loader(
        obs_train,
        act_train,
        nxt_train,
        args.batch_size,
        shuffle=True,
        struct_mode=args.struct_mode,
        radius=args.radius,
        topk=args.topk,
    )
    val_loader = make_loader(
        obs_val,
        act_val,
        nxt_val,
        args.batch_size,
        shuffle=False,
        struct_mode=args.struct_mode,
        radius=args.radius,
        topk=args.topk,
    )

    curve: list[dict[str, float]] = []
    best_val = float("inf")
    best_epoch = -1
    best_state = None
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, opt, device)
        with torch.no_grad():
            val_loss = run_epoch(model, val_loader, None, device)
        curve.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(
            f"[train_baseline][{model_name}] epoch={epoch:03d} "
            f"train={train_loss:.6f} val={val_loss:.6f}"
        )
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, curve, best_epoch


def main() -> None:
    args = parse_args()
    set_global_seeds(args.seed)

    data = load_factual_npz(args.data)
    obs = np.asarray(data["obs"], dtype=np.float32)
    acts = np.asarray(data["acts"], dtype=np.int64)
    nxt = np.asarray(data["next_obs"], dtype=np.float32)
    ep_ids = np.asarray(data["episode_ids"], dtype=np.int64)

    split = episode_level_split(ep_ids, args.seed, args.train_ratio, args.val_ratio)
    train_mask = np.isin(ep_ids, split["train"])
    val_mask = np.isin(ep_ids, split["val"])
    test_mask = np.isin(ep_ids, split["test"])

    if train_mask.sum() == 0 or val_mask.sum() == 0:
        raise RuntimeError(
            "Train/val split is empty. Increase n_steps or number of episodes in factual data."
        )

    obs_train, act_train, nxt_train = obs[train_mask], acts[train_mask], nxt[train_mask]
    obs_val, act_val, nxt_val = obs[val_mask], acts[val_mask], nxt[val_mask]

    obs_dim = int(obs.shape[1])
    action_dim = int(acts.max()) + 1
    out_dir = ensure_dir(args.out_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(
        f"[train_baseline] device={device}, obs_dim={obs_dim}, action_dim={action_dim}, "
        f"train={len(obs_train)}, val={len(obs_val)}, test={int(test_mask.sum())}, "
        f"struct_mode={args.struct_mode}, radius={args.radius}, topk={args.topk}"
    )

    model_ac, curve_ac, best_ep_ac = train_single_model(
        model_name="action_conditioned",
        action_agnostic=False,
        obs_train=obs_train,
        act_train=act_train,
        nxt_train=nxt_train,
        obs_val=obs_val,
        act_val=act_val,
        nxt_val=nxt_val,
        obs_dim=obs_dim,
        action_dim=action_dim,
        args=args,
        device=device,
    )
    ckpt_ac = out_dir / "mlp_action_conditioned.pt"
    torch.save(
        {
            "model_state_dict": model_ac.state_dict(),
            "obs_dim": obs_dim,
            "action_dim": action_dim,
            "action_agnostic": False,
            "struct_mode": args.struct_mode,
            "radius": args.radius,
            "topk": args.topk,
        },
        ckpt_ac,
    )

    model_aa, curve_aa, best_ep_aa = train_single_model(
        model_name="action_agnostic",
        action_agnostic=True,
        obs_train=obs_train,
        act_train=act_train,
        nxt_train=nxt_train,
        obs_val=obs_val,
        act_val=act_val,
        nxt_val=nxt_val,
        obs_dim=obs_dim,
        action_dim=action_dim,
        args=args,
        device=device,
    )
    ckpt_aa = out_dir / "mlp_action_agnostic.pt"
    torch.save(
        {
            "model_state_dict": model_aa.state_dict(),
            "obs_dim": obs_dim,
            "action_dim": action_dim,
            "action_agnostic": True,
            "struct_mode": args.struct_mode,
            "radius": args.radius,
            "topk": args.topk,
        },
        ckpt_aa,
    )

    report = {
        "data_path": args.data,
        "seed": args.seed,
        "device": str(device),
        "struct_mask": {
            "mode": args.struct_mode,
            "radius": args.radius,
            "topk": args.topk,
        },
        "split_episode_ids": {k: v.tolist() for k, v in split.items()},
        "best_epoch": {
            "action_conditioned": best_ep_ac,
            "action_agnostic": best_ep_aa,
        },
        "curves": {
            "action_conditioned": curve_ac,
            "action_agnostic": curve_aa,
        },
        "checkpoints": {
            "action_conditioned": str(ckpt_ac),
            "action_agnostic": str(ckpt_aa),
        },
    }
    report_path = out_dir / "train_report.json"
    save_json(report_path, report)
    print(f"[train_baseline] Saved checkpoints: {ckpt_ac}, {ckpt_aa}")
    print(f"[train_baseline] Saved report: {report_path}")


if __name__ == "__main__":
    main()
