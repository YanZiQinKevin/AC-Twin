"""Split a collected Stage 4D-D2 screen ``groups.npz`` into sealed truth and legal inputs.

Plan ``AC-TWIN-RFER-STAGE4D-D2-METADRIVE-EXTERNAL-2026-09-21-V0.2`` section 7.1 (lines 402-406):
the screen/confirm collection is split into two physically separate files —

- ``inference_inputs.npz``: the legal model inputs and the action/scenario metadata
  (``object_history``, ``object_factual_future``, ``factual_actions``, ``inference_mask``,
  ``t_star``, the canonical action contract, group keys). It NEVER carries the counterfactual
  futures, so the scoring process is structurally unable to read the truth.
- ``sealed_truth.npz``: the ten-action true futures, the common evaluation mask and the fields
  the adjudicator needs for ``J_true`` (``object_counterfactual_futures``,
  ``object_factual_future`` for the reactive response, ``evaluation_mask``, ``history_last_xy``).

The two files are aligned by the group key (``scenario_id`` -> ``group_seed``, derived with the
frozen ``stage4d_external_training.group_seed_from_scenario_id``) and bound to the collection
receipt. The collector's ``collection_receipt.json`` records no SHA256 of ``groups.npz`` /
``groups_meta.json``, so this split computes and records them itself in ``split_receipt.json``.

The split is deterministic and verifiable: a re-run with the receipt present re-hashes the source
and both outputs and refuses any drift; it never overwrites differing content. This module is
torch-free and does NOT touch the score side beyond writing the two files.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ac_twin import stage4d_external_training as T
from ac_twin.checkpointing import atomic_save_npz
from ac_twin.stage1_data import sha256_file
from ac_twin.stage3r_protocol import write_once_json

ROOT = Path(__file__).resolve().parents[1]

#: array fields routed to the LEGAL model-inputs file (never the truth).
_INPUT_ARRAY_FIELDS = (
    "object_history",                # (G,10,5,8)
    "object_factual_future",         # (G,20,5,8)  observed/reference trajectory (legal input)
    "factual_actions",               # (G,20)
    "inference_mask",                # (G,20,5) collector's cf-independent mask
    "t_star",                        # (G,)
    "cf_actions",                    # (10,) canonical action contract
    "cf_durations",                  # (10,)
    "n_actions",                     # scalar
    "object_features_json",          # scalar
    "action_plans_json",             # scalar
)

#: array fields routed to the SEALED truth file (the score side never opens it).
_TRUTH_ARRAY_FIELDS = (
    "object_counterfactual_futures",  # (G,10,20,5,8) TRUE futures
    "object_factual_future",          # (G,20,5,8)  for the reactive-response comparison
    "inference_mask",                 # (G,20,5) -> stored as evaluation_mask
)


def split_screen_truth(
    role_dir: str | Path,
    split_dir: str | Path,
    *,
    role_salt: str,
    expect_groups: int | None = None,
) -> dict[str, Path]:
    """Split one collected screen role into ``inference_inputs.npz`` + ``sealed_truth.npz``.

    Deterministic and create-once-verifying: if ``split_receipt.json`` is present, the source
    and both outputs are re-hashed against it and any drift raises; otherwise all three files
    are written exactly once. Returns the dict of output paths. Never reads or writes any score
    artifact; never consumes a one-shot resource (the split is a pure function of the source).
    """
    role_dir = Path(role_dir)
    split_dir = Path(split_dir)
    groups_npz = role_dir / "groups.npz"
    groups_meta = role_dir / "groups_meta.json"
    collection_receipt = role_dir / "collection_receipt.json"
    inputs_path = split_dir / "inference_inputs.npz"
    truth_path = split_dir / "sealed_truth.npz"
    receipt_path = split_dir / "split_receipt.json"
    for path in (groups_npz, groups_meta, collection_receipt):
        if not path.is_file():
            raise FileNotFoundError(f"collected screen role artifact missing: {path}")

    source_groups_sha = sha256_file(groups_npz)
    source_meta_sha = sha256_file(groups_meta)
    source_receipt_sha = sha256_file(collection_receipt)

    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        bindings = receipt["bindings"]
        expected = {
            "source_groups_sha256": source_groups_sha,
            "source_meta_sha256": source_meta_sha,
            "source_collection_receipt_sha256": source_receipt_sha,
            "inference_inputs_sha256": sha256_file(inputs_path),
            "sealed_truth_sha256": sha256_file(truth_path),
        }
        for key, value in expected.items():
            if bindings.get(key) != value:
                raise ValueError(f"screen split drifted from its receipt binding {key!r}")
        return {"inference_inputs": inputs_path, "sealed_truth": truth_path,
                "split_receipt": receipt_path}

    with np.load(groups_npz, allow_pickle=True) as archive:
        data = {key: archive[key] for key in archive.files}
    meta = json.loads(groups_meta.read_text(encoding="utf-8"))
    scenario_ids = [str(m["scenario_id"]) for m in meta]
    groups = int(np.asarray(data["object_history"]).shape[0])
    n_actions = int(np.asarray(data["object_counterfactual_futures"]).shape[1])
    if n_actions != T.N_ACTIONS:
        raise ValueError(f"expected {T.N_ACTIONS} actions per group, got {n_actions}")
    if len(set(scenario_ids)) != len(scenario_ids):
        raise ValueError("the screen role has duplicate scenario_ids")
    if expect_groups is not None and groups != int(expect_groups):
        raise ValueError(f"screen role has {groups} groups, expected {expect_groups}")
    if not (np.asarray(data["object_factual_future"]).shape[0] == groups
            == np.asarray(data["object_counterfactual_futures"]).shape[0]
            == len(scenario_ids)):
        raise ValueError("screen role arrays and meta must align on the group axis")

    group_seed = np.asarray(
        [T.group_seed_from_scenario_id(sid, role_salt) for sid in scenario_ids],
        dtype=np.int64)
    if np.unique(group_seed).size != groups:
        raise ValueError("derived group_seed is not unique across the screen groups")
    history_last_xy = np.asarray(
        data["object_history"], dtype=np.float64)[:, -1, 0, 2:4]  # ego (s,d) at t_star

    shared = {
        "group_seed": group_seed,
        "scenario_ids_json": np.asarray(json.dumps(scenario_ids)),
        "source_groups_sha256": np.asarray(source_groups_sha),
    }
    inputs_payload = {key: data[key] for key in _INPUT_ARRAY_FIELDS}
    inputs_payload.update(shared)
    inputs_payload["source_meta_sha256"] = np.asarray(source_meta_sha)

    truth_payload = {
        "object_counterfactual_futures": data["object_counterfactual_futures"],
        "object_factual_future": data["object_factual_future"],
        "evaluation_mask": np.asarray(data["inference_mask"], dtype=bool),
        "history_last_xy": history_last_xy,
    }
    truth_payload.update(shared)

    split_dir.mkdir(parents=True, exist_ok=True)
    atomic_save_npz(inputs_path, **inputs_payload)
    atomic_save_npz(truth_path, **truth_payload)

    receipt = {
        "plan_id": T.PLAN_ID,
        "role": "development_screen",
        "n_groups": int(groups),
        "n_actions": int(n_actions),
        "group_seed_derivation": ("stage4d_external_training.group_seed_from_scenario_id"
                                  "(scenario_id, role_salt)"),
        "inference_inputs_fields": list(_INPUT_ARRAY_FIELDS)
        + ["group_seed", "scenario_ids_json", "source_groups_sha256", "source_meta_sha256"],
        "sealed_truth_fields": ["object_counterfactual_futures", "object_factual_future",
                                "evaluation_mask", "history_last_xy", "group_seed",
                                "scenario_ids_json", "source_groups_sha256"],
        "separation": ("the inference_inputs carry NO counterfactual future; the scoring "
                       "process is structurally unable to read the sealed truth"),
        "bindings": {
            "source_groups_sha256": source_groups_sha,
            "source_meta_sha256": source_meta_sha,
            "source_collection_receipt_sha256": source_receipt_sha,
            "inference_inputs_sha256": sha256_file(inputs_path),
            "sealed_truth_sha256": sha256_file(truth_path),
        },
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    write_once_json(receipt_path, receipt)
    return {"inference_inputs": inputs_path, "sealed_truth": truth_path,
            "split_receipt": receipt_path}


__all__ = ["split_screen_truth"]
