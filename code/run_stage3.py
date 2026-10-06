"""Stage 3 state-machine entry point.

Subcommands wire the Stage 3 development pipeline. Only short, read-only or
engineering-smoke operations run automatically; multi-seed formal training,
cross-fit, formal collection, full frozen inference, 10,000x bootstrap and the
single unlock are prepared here but started manually by the user in PowerShell.

The ``smoke`` command runs a 12-episode / 1-epoch pass through the response
predictor, the Direct/Dual/Oracle minimal forwards, the statistics and a smoke
joint lock, and may only ever report ``ENGINEERING_SMOKE_ONLY``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "code") not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from ac_twin.progress import ProgressReporter  # noqa: E402
from ac_twin import stage3_protocol as s3p  # noqa: E402
from ac_twin import stage3_statistics as s3s  # noqa: E402
from ac_twin.stage3_data import (  # noqa: E402
    compute_effect_eval_mask,
    compute_response_pos,
    compute_regime_labels,
    split_role_indices,
    validate_stage3_source,
)
from ac_twin.stage3_response import Stage3ResponsePredictor  # noqa: E402
from ac_twin.stage3_routing import Condition, Stage3Router  # noqa: E402

SMOKE_STATUS = "ENGINEERING_SMOKE_ONLY"
DEVELOPMENT_SOURCE = "data/object_centric_stage/merge_2000/object_twinworld_2000.npz"


def _resolve(path: str) -> Path:
    path = Path(path)
    return path if path.is_absolute() else (ROOT / path)


def _to_tensor(array, dtype=torch.float32):
    return torch.as_tensor(np.asarray(array), dtype=dtype)


def _load_development(npz_path: Path, limit: int) -> dict:
    with np.load(npz_path, allow_pickle=True) as archive:
        data = {key: archive[key] for key in archive.files}
    validate_stage3_source(data, npz_path)
    # Smoke restriction: first `limit` episodes across splits (engineering only).
    for key in ("object_history", "object_factual_future", "object_counterfactual_future",
                "factual_actions", "counterfactual_actions"):
        data[key] = data[key][:limit]
    for key in ("cf_action", "cf_duration", "episode_seed", "t_star", "split"):
        data[key] = data[key][:limit]
    return data


def cmd_smoke(args: argparse.Namespace) -> int:
    npz_path = _resolve(DEVELOPMENT_SOURCE)
    if not npz_path.is_file():
        print(f"smoke: development source missing: {npz_path}", flush=True)
        return 1
    limit = int(args.episodes)
    data = _load_development(npz_path, limit)
    reporter = ProgressReporter(total=5, every=1, prefix="stage3-smoke", heartbeat_seconds=30.0)

    history = _to_tensor(data["object_history"])
    factual = _to_tensor(data["object_factual_future"])
    counterfactual = _to_tensor(data["object_counterfactual_future"])
    factual_suffix = factual[:, :10]
    factual_actions = _to_tensor(data["factual_actions"], torch.long)
    counterfactual_actions = _to_tensor(data["counterfactual_actions"], torch.long)
    cf_action = _to_tensor(data["cf_action"], torch.long)
    cf_duration = _to_tensor(data["cf_duration"], torch.long)
    t_star = _to_tensor(data["t_star"], torch.long)
    reporter.update(1, details="data-loaded")

    # 1. response predictor minimal forward (1 epoch = a few gradient steps).
    torch.manual_seed(101)
    predictor = Stage3ResponsePredictor(hidden_dim=32, response_feature_dim=16)
    optimizer = torch.optim.Adam(predictor.parameters(), lr=1e-3)
    r_true = _to_tensor(compute_response_pos(factual.numpy(), counterfactual.numpy()))
    regime_labels = _to_tensor(
        compute_regime_labels(r_true.numpy()), torch.long
    )
    eval_mask = compute_effect_eval_mask(
        data["object_history"], data["object_factual_future"], data["object_counterfactual_future"]
    )
    valid = _to_tensor(eval_mask, torch.bool)
    weights = torch.ones(4)
    from ac_twin.stage3_response import response_masked_loss
    for _ in range(max(1, int(args.epochs))):
        optimizer.zero_grad()
        out = predictor(history, factual_suffix, factual_actions, counterfactual_actions,
                        cf_action, cf_duration, t_star)
        loss = response_masked_loss(out, r_true, regime_labels, valid, weights)
        loss.backward()
        optimizer.step()
    predictor.eval()
    reporter.update(2, details=f"response-loss={float(loss.detach()):.4f}")

    # 2. Direct/Dual/Oracle minimal forwards using the predictor's frozen features.
    with torch.no_grad():
        feats = predictor(history, factual_suffix, factual_actions, counterfactual_actions,
                          cf_action, cf_duration, t_star)
        response_features = feats["response_features"].detach()
        regime_probs = torch.softmax(feats["regime_logits"].detach(), dim=-1)

    effect_target = _to_tensor(
        counterfactual.numpy()[..., 2:4] - factual.numpy()[..., 2:4]
    )
    oracle_gate = torch.rand(history.shape[0], factual_actions.shape[1], history.shape[2], 1)
    router_outputs = {}
    for condition in (Condition.DIRECT, Condition.DUAL_ROUTER, Condition.ORACLE_REFERENCE):
        torch.manual_seed(1101)
        router = Stage3Router(condition, hidden_dim=32, response_feature_dim=16)
        router.eval()
        kwargs = dict(
            history=history, factual_suffix=factual_suffix,
            factual_actions=factual_actions, counterfactual_actions=counterfactual_actions,
            response_features=response_features, regime_probs=regime_probs,
        )
        if condition == Condition.ORACLE_REFERENCE:
            kwargs["oracle_gate"] = oracle_gate
        with torch.no_grad():
            router_outputs[condition] = router(**kwargs)
    reporter.update(3, details="routers-forward-ok")

    # 3. statistics: effect-ADE and the two contrasts on the smoke batch.
    ade = {
        cond.value: s3s.effect_ade_per_episode(
            out.numpy(), effect_target.numpy(), eval_mask
        )
        for cond, out in router_outputs.items()
    }
    c_uniform = float(
        np.nanmean(ade[Condition.DIRECT.value] - ade[Condition.DUAL_ROUTER.value])
    )
    reporter.update(4, details=f"smoke-C_direct={c_uniform:+.4f}")

    # 4. smoke joint lock + verify + report signature.
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    historical = np.arange(10_000_000, 20_000_000, dtype=np.int64)
    schedules = {
        s3p.GATEC_ID_NAME: np.arange(1, 64, dtype=np.int64),
        s3p.GATEC_UNSEEN_NAME: np.arange(1000, 1064, dtype=np.int64),
    }
    smoke_config = {
        "plan_id": s3p.PLAN_ID,
        "development_source": DEVELOPMENT_SOURCE,
        "response_bins": {"zero_tolerance": 0.0001, "t1": 0.24843344965718578, "t2": 1.898992796585611},
        "formal_holdouts": {
            s3p.GATEC_ID_NAME: {"durations": [1, 3]},
            s3p.GATEC_UNSEEN_NAME: {"durations": [2]},
        },
    }
    lock_path = output_dir / "stage3_joint_lock.json"
    if lock_path.exists():
        lock_path.unlink()
    s3p.write_joint_lock(smoke_config, output_dir, historical_seeds=historical, seed_schedules=schedules)
    s3p.verify_joint_lock(output_dir)

    report = {
        "status": SMOKE_STATUS,
        "episodes": limit,
        "epochs": int(args.epochs),
        "response_loss": float(loss.detach()),
        "effect_ade": {k: [float(x) for x in v] for k, v in ade.items()},
        "smoke_c_direct_point": c_uniform,
    }
    report["output_signature"] = s3p.output_signature(report)
    report_path = output_dir / "smoke_report.json"
    tmp = report_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    tmp.replace(report_path)
    assert s3p.output_signature_from_disk(report_path) == report["output_signature"]
    reporter.update(5, details="lock+report-ok")

    print(f"[stage3-smoke] STATUS={SMOKE_STATUS} | "
          f"report={report_path} | signature={report['output_signature'][:16]}", flush=True)
    print(f"[stage3-smoke] SUCCESS: response/Direct/Dual/Oracle forwards, statistics and lock OK",
          flush=True)
    return 0


def cmd_validate_config(args: argparse.Namespace) -> int:
    config = json.loads(_resolve(args.config).read_text(encoding="utf-8"))
    s3p.validate_stage3_config(config)
    print(f"config OK: {args.config}", flush=True)
    return 0


def cmd_verify_lock(args: argparse.Namespace) -> int:
    lock = s3p.verify_joint_lock(_resolve(args.output_dir))
    print(f"joint lock OK: {lock['lock_sha256'][:16]}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AC-Twin Stage 3 state-machine entry")
    sub = parser.add_subparsers(dest="command", required=True)

    smoke = sub.add_parser("smoke", help="12-episode/1-epoch engineering smoke only")
    smoke.add_argument("--episodes", type=int, default=12)
    smoke.add_argument("--epochs", type=int, default=1)
    smoke.add_argument("--output-dir", default="experiments/stage3_routing/smoke")
    smoke.set_defaults(func=cmd_smoke)

    validate = sub.add_parser("validate-config", help="read-only config validation")
    validate.add_argument("--config", required=True)
    validate.set_defaults(func=cmd_validate_config)

    verify = sub.add_parser("verify-lock", help="read-only joint-lock verification")
    verify.add_argument("--output-dir", required=True)
    verify.set_defaults(func=cmd_verify_lock)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
