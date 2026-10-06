"""Stage 4D-D2 unified orchestration runner.

Plan ``AC-TWIN-RFER-STAGE4D-D2-METADRIVE-EXTERNAL-2026-09-21-V0.2`` section 13.
The runner only orchestrates frozen stage functions; it never re-implements
their science and never re-runs an already-burned one-shot stage.

Subcommands implemented so far (Task 1, plan section 12.2):
    preflight       frozen source-chain + platform verification (create-once receipt)
    raw-inventory   resumable SHA256 inventory + Merkle root receipt
    env-lock        freeze both isolated environments (after conversion smoke)
    verify          read-only re-check of existing Task-1 receipts

Later-task subcommands (conversion smoke, env lock, scenario index, adapter,
pilot, training, screen, power, confirm, visualization) are added with their
tasks; calling an unimplemented subcommand fails loudly instead of guessing.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "code") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "code"))

from ac_twin import stage4d_external_protocol as proto  # noqa: E402
from ac_twin import stage4d_external_split as split  # noqa: E402

CONFIG_PATH = (
    PROJECT_ROOT / "experiments/stage4_reliability/stage4d_external/config.json"
)

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_SOURCE_INVALID = 2


def _print_summary(command: str, ok: bool, detail: str, started: float) -> None:
    elapsed = time.perf_counter() - started
    status = "SUCCESS" if ok else "FAILURE"
    print(
        f"[run_stage4d_external:{command}] {status} | {detail} | "
        f"elapsed={elapsed:0.1f}s",
        flush=True,
    )


def _cmd_preflight(config: dict[str, Any]) -> str:
    receipt = proto.run_preflight(config)
    entries = receipt["source_chain"]["entries"]
    return (
        f"source-chain {len(entries)} entries MATCH/BOUND; "
        f"platform metadrive/scenarionet/waymo verified; "
        f"receipt=platform/preflight_receipt.json"
    )


def _cmd_raw_inventory(config: dict[str, Any]) -> str:
    receipt = proto.run_raw_inventory(config)
    return (
        f"shards={receipt['shard_count']} bytes={receipt['total_bytes']} "
        f"merkle_root={receipt['merkle_root_sha256'][:16]}..."
    )


def _cmd_env_lock(config: dict[str, Any]) -> str:
    lock = proto.freeze_environment_lock(config)
    return (
        f"sim_env={lock['sim_env']['name']} py{lock['sim_env']['python_report']['python']} "
        f"torch={lock['sim_env']['torch_report']['torch']}; "
        f"conversion_env={lock['conversion_env']['name']} py{lock['conversion_env']['python_report']['python']} "
        f"(WSL {lock['conversion_env']['wsl_distro']}); "
        f"receipt=platform/platform_environment_lock.json"
    )


def _cmd_verify(config: dict[str, Any]) -> str:
    """Read-only: existing Task-1 receipts must exist and re-verify."""
    platform_dir = PROJECT_ROOT / config["run_dir"] / "platform"
    required = [
        platform_dir / "preflight_receipt.json",
        platform_dir / "raw_dataset_root_receipt.json",
        platform_dir / "raw_dataset_inventory.jsonl",
    ]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise RuntimeError("missing Task-1 receipts: " + ", ".join(missing))
    proto.verify_source_chain(config)
    proto.verify_platform(config)
    return "Task-1 receipts present; source chain and platform re-verified"


def _cmd_split(config: dict[str, Any]) -> str:
    manifest = split.build_scenario_split_manifest(
        config,
        code_self_path=PROJECT_ROOT / "code/ac_twin/stage4d_external_split.py",
    )
    stats = manifest["role_stats"]
    per_role = " ".join(
        f"{role}={stats[role]['primary']}+{stats[role]['reserve']}"
        for role in split.ROLE_ORDER
    )
    return (
        f"survivors={manifest['prefilter']['survivors']} "
        f"oversubscription={manifest['oversubscription_factor']:.3f} | {per_role} | "
        f"manifest=platform/{split.MANIFEST_FILE}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=["preflight", "raw-inventory", "env-lock", "verify", "split"],
        help="Task-1/2 subcommands; later tasks extend this list",
    )
    parser.add_argument("--config", default=str(CONFIG_PATH))
    args = parser.parse_args(argv)

    config = proto.load_config(args.config)
    started = time.perf_counter()
    handlers = {
        "preflight": _cmd_preflight,
        "raw-inventory": _cmd_raw_inventory,
        "env-lock": _cmd_env_lock,
        "verify": _cmd_verify,
        "split": _cmd_split,
    }
    try:
        detail = handlers[args.command](config)
    except proto.PreflightFailure as exc:
        _print_summary(args.command, False, f"{exc.status} | {exc.detail}", started)
        return EXIT_SOURCE_INVALID
    except Exception as exc:  # noqa: BLE001 - fail loud with a summary
        _print_summary(args.command, False, f"{type(exc).__name__}: {exc}", started)
        return EXIT_ERROR
    _print_summary(args.command, True, detail, started)
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
