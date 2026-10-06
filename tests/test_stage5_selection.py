"""Blocking tests for the Stage 5 validation selection (plan v0.3 section 3.3, Task 4).

Plan ``AC-TWIN-RFER-STAGE5-SOTA-COMPARISON-2026-09-27-V0.3``. Pins the mechanical
selection contract: validity exclusions, lowest validation AURC_REGRET wins, the
frozen 1e-12 tie-break chain (REGRET@80 -> params -> inference time -> lexicographic),
the unique STRONGEST_PUBLISHED_BASELINE, and the create-once final_method_selection.json
that records EVERY candidate (winners and losers).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from ac_twin import stage5_selection as s5sel  # noqa: E402


def cand(method_id, config_key, aurc, regret80=0.02, params=10000, infer_ms=1.0,
         valid=True, reason=""):
    return s5sel.CandidateRecord(
        method_id=method_id, config_key=config_key,
        config={"config_key": config_key}, aurc_regret=aurc, regret_at_80=regret80,
        param_count=params, mean_inference_ms=infer_ms, valid=valid,
        exclusion_reason=reason)


def test_candidate_validity_checks():
    rng = np.random.RandomState(0)
    good = rng.rand(5, 100) + np.arange(5)[:, None]
    ok, reason = s5sel.check_candidate_validity(good)
    assert ok and reason == ""
    bad_finite = good.copy()
    bad_finite[2, 10] = np.nan
    ok, reason = s5sel.check_candidate_validity(bad_finite)
    assert not ok and "finite" in reason
    constant = np.ones((5, 100))
    ok, reason = s5sel.check_candidate_validity(constant)
    assert not ok and "degenerate" in reason


def test_per_method_selection_lowest_aurc_and_tiebreaks():
    cands = [
        cand("M", "b", aurc=0.030),
        cand("M", "a", aurc=0.020),  # lowest AURC wins
        cand("M", "c", aurc=0.040),
    ]
    winners = s5sel.select_config_per_method(cands)
    assert winners["M"].config_key == "a"
    # exact tie on AURC -> lower REGRET@80 wins
    cands = [cand("M", "x", aurc=0.020, regret80=0.030),
             cand("M", "y", aurc=0.020, regret80=0.025)]
    assert s5sel.select_config_per_method(cands)["M"].config_key == "y"
    # tie within 1e-12 on both -> fewer params wins
    cands = [cand("M", "x", aurc=0.020, regret80=0.030, params=20000),
             cand("M", "y", aurc=0.020 + 5e-13, regret80=0.030, params=9000)]
    assert s5sel.select_config_per_method(cands)["M"].config_key == "y"
    # then faster inference
    cands = [cand("M", "x", aurc=0.020, regret80=0.030, params=9000, infer_ms=2.0),
             cand("M", "y", aurc=0.020, regret80=0.030, params=9000, infer_ms=0.5)]
    assert s5sel.select_config_per_method(cands)["M"].config_key == "y"
    # then lexicographic config key
    cands = [cand("M", "b", aurc=0.020, regret80=0.030, params=9000, infer_ms=0.5),
             cand("M", "a", aurc=0.020, regret80=0.030, params=9000, infer_ms=0.5)]
    assert s5sel.select_config_per_method(cands)["M"].config_key == "a"


def test_invalid_candidates_never_win():
    cands = [cand("M", "bad", aurc=0.001, valid=False, reason="non-finite risk"),
             cand("M", "ok", aurc=0.050)]
    winners = s5sel.select_config_per_method(cands)
    assert winners["M"].config_key == "ok"
    with pytest.raises(ValueError):
        s5sel.select_config_per_method([cand("M", "bad", aurc=0.001, valid=False,
                                             reason="x")])


def test_strongest_published_baseline_unique_and_tiebreak():
    winners = {
        "P1": cand("P1", "a", aurc=0.021),
        "P2": cand("P2", "a", aurc=0.019),   # unique lowest
        "P3": cand("P3", "a", aurc=0.030),
    }
    assert s5sel.select_strongest_published_baseline(winners) == "P2"
    # tie within 1e-12 on AURC -> lower REGRET@80 -> ... -> lexicographic method ID
    winners = {
        "P1": cand("P1", "a", aurc=0.019, regret80=0.02),
        "P2": cand("P2", "a", aurc=0.019 + 5e-13, regret80=0.02),
    }
    assert s5sel.select_strongest_published_baseline(winners) == "P1"


def test_final_method_selection_create_once_records_all_candidates(tmp_path):
    cands = [cand("P1", "a", aurc=0.021), cand("P1", "b", aurc=0.025),
             cand("P2", "a", aurc=0.019)]
    winners = s5sel.select_config_per_method(cands)
    strongest = s5sel.select_strongest_published_baseline(winners)
    path = tmp_path / "final_method_selection.json"
    s5sel.write_final_method_selection(path, cands, winners, strongest)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["strongest_published_baseline"] == "P2"
    keys = {(c["method_id"], c["config_key"]) for c in payload["candidates"]}
    assert keys == {("P1", "a"), ("P1", "b"), ("P2", "a")}  # losers recorded too
    selected = {(m, c["config_key"]) for m, c in payload["selected"].items()}
    assert selected == {("P1", "a"), ("P2", "a")}
    with pytest.raises((FileExistsError, RuntimeError)):
        s5sel.write_final_method_selection(path, cands, winners, strongest)


def test_candidate_record_has_no_holdout_handle():
    """Selection consumes validation metrics only — no path/holdout field exists."""
    fields = set(s5sel.CandidateRecord.__dataclass_fields__)
    forbidden = [f for f in fields if "holdout" in f.lower() or "path" in f.lower()]
    assert forbidden == []
