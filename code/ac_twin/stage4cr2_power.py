"""Stage 4C-R2 conditional confirmation power diagnostic.

Plan ``AC-TWIN-RFER-STAGE4C-R2-DECISION-RISK-SCALING-2026-09-16-V0.2`` section 10. Legal
ONLY after the one-shot development screen PASSED
(``STAGE4CR2_DEV_READY_FOR_CONFIRMATION``). The ONLY input is the create-once
``development/screen_group_quantities.npz`` (+ its receipt); raw predictions, collected
bundles and truth are NEVER re-read here.

Statistics (frozen):
- the covariance comes DIRECTLY from this run's whole-group paired bootstrap draws,
  recomputed deterministically with the frozen entropy ``2026092702`` via
  :func:`stage4cr2_statistics.paired_group_bootstrap`; a 95% SE is NEVER derived from the
  80% CI width;
- candidate n = {2000, 3000, 4000, 5000}; covariance scales by ``n_reference / n``;
- effect shrink = {100%, 75%, 50%}; positive AND negative effects shrink by the same
  factor and signs are never flipped;
- the 75% scenario is the ONLY blocking selection judge; 100%/50% are sensitivity;
- the simulated confirm gate is the FULL AND-chain on each joint Monte-Carlo draw
  (10,000 draws), never a product of marginal powers: the three primary improvements
  with 95% CI lower bounds, the 4/5 positive-seed rules, the four-strata floor, the
  1%/5% trim rule and the |Spearman(ensemble, MARGIN_ONLY)| ceiling;
- target joint power = 0.80; if n=5,000 at 75% shrink still falls short, the terminal
  state is ``STAGE4CR2_CONFIRMATION_UNDERPOWERED`` and NO confirm plan/lock/collection
  may be created. This plan never creates or consumes any confirm resource.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ac_twin import stage4cr2_protocol as s4cr2
from ac_twin import stage4cr2_statistics as s4cr2st
from ac_twin.stage1_data import sha256_file
from ac_twin.stage3r_protocol import write_once_json

ROOT = Path(__file__).resolve().parents[2]

Z_95 = 1.96  # confirm-side CI level (the dev screen itself stays at 80%)

# Power quantity subset of the frozen 32-vector (stage4cr2_statistics layout), in order.
POWER_QUANTITIES = (
    s4cr2st.Q_I_AURC, s4cr2st.Q_I_REGRET80, s4cr2st.Q_PR,
    s4cr2st.Q_TRIM1, s4cr2st.Q_TRIM5,
    s4cr2st.Q_TSTAR_LOW, s4cr2st.Q_TSTAR_HIGH,
    s4cr2st.Q_INTERACT_LOW, s4cr2st.Q_INTERACT_HIGH,
    s4cr2st.Q_SPEARMAN_MARGIN,
) + s4cr2st.SEED_I_AURC_INDEX + s4cr2st.SEED_I_R80_INDEX
N_POWER_QUANTITIES = len(POWER_QUANTITIES)  # 20

# Local indices inside the power subset (lists: numpy fancy-indexing requires them).
P_I_AURC = 0
P_I_REGRET80 = 1
P_PR = 2
P_TRIM = [3, 4]
P_STRATA = [5, 6, 7, 8]
P_SPEARMAN_MARGIN = 9
P_SEED_I_AURC = [10, 11, 12, 13, 14]
P_SEED_I_R80 = [15, 16, 17, 18, 19]

POWER_QUANTITY_NAMES = tuple(
    s4cr2st._QUANTITY_NAMES[i] for i in POWER_QUANTITIES)

DIAGNOSTIC_FILENAME = "confirmation_power_diagnostic.json"


# --- result container ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PowerReport:
    """The conditional confirmation power diagnostic outcome (never a confirm resource)."""
    plan_id: str
    status: str                      # DEV_READY (viable n) or CONFIRMATION_UNDERPOWERED
    selected_n: int | None
    underpowered: bool
    target: float
    selection_scenario: float
    monte_carlo: int
    n_reference: int
    candidate_ns: tuple[int, ...]
    shrink_scenarios: tuple[float, ...]
    per_n: dict[int, Any]
    theta_hat: dict[str, float]
    se_reference: dict[str, float]
    screen_quantities_sha256: str
    screen_quantities_receipt_sha256: str
    bootstrap: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "status": self.status,
            "selected_n": self.selected_n,
            "underpowered": self.underpowered,
            "target": self.target,
            "selection_scenario": self.selection_scenario,
            "monte_carlo": self.monte_carlo,
            "n_reference": self.n_reference,
            "candidate_ns": [int(n) for n in self.candidate_ns],
            "shrink_scenarios": [float(s) for s in self.shrink_scenarios],
            "per_n": {str(n): entry for n, entry in self.per_n.items()},
            "theta_hat": dict(self.theta_hat),
            "se_reference": dict(self.se_reference),
            "screen_quantities_sha256": self.screen_quantities_sha256,
            "screen_quantities_receipt_sha256": self.screen_quantities_receipt_sha256,
            "bootstrap": dict(self.bootstrap),
        }


# --- loading: the create-once per-group quantities ONLY ------------------------------------------


def load_power_inputs(npz_path: str | Path) -> dict[str, Any]:
    """Load the create-once screen quantities NPZ + receipt; power reads ONLY these."""
    npz_path = Path(npz_path)
    receipt_path = npz_path.with_name("screen_group_quantities_receipt.json")
    if not receipt_path.is_file():
        raise FileNotFoundError(
            f"screen quantities receipt missing (create-once pair): {receipt_path}")
    quantities = s4cr2st.load_screen_group_quantities(npz_path, receipt_path)
    return {
        "quantities": quantities,
        "npz_sha256": sha256_file(npz_path),
        "receipt_sha256": sha256_file(receipt_path),
    }


# --- covariance from THIS RUN's bootstrap draws ----------------------------------------------------


def symmetrised_psd_covariance(draws: np.ndarray) -> np.ndarray:
    """Symmetrised, eigenvalue-clipped covariance of bootstrap draw columns."""
    draws = np.asarray(draws, dtype=np.float64)
    if draws.ndim != 2 or draws.shape[0] < 2:
        raise ValueError("draws must align as (n_resamples, n_quantities)")
    if not np.isfinite(draws).all():
        raise ValueError(
            "non-finite bootstrap draw in a power quantity; the covariance must come "
            "from finite whole-group paired draws (fail-closed, no nan-filling)")
    covariance = np.cov(draws, rowvar=False)
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    clipped = np.clip(eigenvalues, 1e-12, None)
    covariance = (eigenvectors * clipped) @ eigenvectors.T
    return 0.5 * (covariance + covariance.T)


def bootstrap_power_inputs(
    quantities: Mapping[str, Any], *, seed: int, n_resamples: int
) -> dict[str, Any]:
    """Reproduce THIS RUN's whole-group paired bootstrap and take theta + covariance.

    The draw stream is deterministic in (quantities, seed, n_resamples), so this yields
    the same draws the adjudication used; the identity evaluation is ``theta_hat`` and
    the covariance is taken directly from the draws — never from any CI width.
    """
    bootstrap = s4cr2st.paired_group_bootstrap(
        quantities, seed=int(seed), n_resamples=int(n_resamples))
    draws = np.asarray(bootstrap.draws, dtype=np.float64)[:, list(POWER_QUANTITIES)]
    theta_hat = np.asarray(bootstrap.point, dtype=np.float64)[list(POWER_QUANTITIES)]
    if not np.isfinite(theta_hat).all():
        raise ValueError("non-finite identity quantity in the power subset")
    return {
        "theta_hat": theta_hat,
        "covariance": symmetrised_psd_covariance(draws),
        "n_reference": int(np.asarray(quantities["regret"]).size),
        "bootstrap": bootstrap,
    }


# --- the FULL confirm-gate AND-chain on joint draws ------------------------------------------------


def evaluate_confirm_draws(
    draws: np.ndarray, standard_errors: np.ndarray
) -> dict[str, Any]:
    """Vectorised confirm AND-chain over joint Monte-Carlo draws of the power subset.

    ``draws`` is ``(replicates, 20)``; ``standard_errors`` is ``sqrt(diag(cov_n))`` for
    the same subset. CI conditions use the confirm-side 95% bound
    (``estimate - 1.96*SE > 0``) computed from the draw covariance; seed / strata / trim /
    Spearman conditions use point values, exactly mirroring the dev risk gate semantics.
    """
    draws = np.asarray(draws, dtype=np.float64)
    se = np.asarray(standard_errors, dtype=np.float64)
    if draws.ndim != 2 or draws.shape[1] != N_POWER_QUANTITIES:
        raise ValueError(f"draws must align as (replicates, {N_POWER_QUANTITIES})")
    if se.shape != (N_POWER_QUANTITIES,):
        raise ValueError("standard_errors must align with the power quantity subset")

    def ci95_lower(slot: int) -> np.ndarray:
        return draws[:, slot] - Z_95 * se[slot]

    conditions = {
        "i_aurc_regret": (draws[:, P_I_AURC] >= s4cr2.VALIDATION_I_AURC_MIN)
        & (ci95_lower(P_I_AURC) > 0.0),
        "i_regret80": (draws[:, P_I_REGRET80] >= s4cr2.VALIDATION_I_REGRET80_MIN)
        & (ci95_lower(P_I_REGRET80) > 0.0),
        "pr_auc_minus_prevalence": (draws[:, P_PR] > 0.0) & (ci95_lower(P_PR) > 0.0),
        "trims_positive": (draws[:, P_TRIM] > 0.0).all(axis=1),
        "strata_floor": (draws[:, P_STRATA] >= s4cr2.GATE_STRATA_I_AURC_FLOOR).all(axis=1),
        "margin_spearman_below_ceiling": (
            np.abs(draws[:, P_SPEARMAN_MARGIN]) < s4cr2.GATE_MARGIN_SPEARMAN_CEILING),
        "seeds_i_aurc_positive": (
            (draws[:, P_SEED_I_AURC] > 0.0).sum(axis=1)
            >= s4cr2.VALIDATION_MIN_POSITIVE_SEEDS),
        "seeds_i_regret80_positive": (
            (draws[:, P_SEED_I_R80] > 0.0).sum(axis=1)
            >= s4cr2.VALIDATION_MIN_POSITIVE_SEEDS),
    }
    passed = np.logical_and.reduce(list(conditions.values()))
    return {"passed": passed, "conditions": conditions}


# --- joint Monte-Carlo simulation + mechanical n selection -----------------------------------------


def simulate_power(
    theta_hat: np.ndarray,
    covariance: np.ndarray,
    *,
    candidate_ns: Sequence[int],
    n_reference: int,
    monte_carlo: int,
    seed: int,
    shrink_scenarios: Sequence[float],
    selection_scenario: float,
    target: float,
) -> dict[str, Any]:
    """Smallest n whose SELECTION-scenario (75%) joint power reaches the target.

    Covariance scales by ``n_reference / n``; effects shrink to ``scenario * theta`` with
    signs preserved (never flipped). Only the 75% scenario selects; 100%/50% are
    reported as sensitivity. The gate is the full AND-chain on each joint draw.
    """
    theta_hat = np.asarray(theta_hat, dtype=np.float64)
    covariance = np.asarray(covariance, dtype=np.float64)
    if theta_hat.shape != (N_POWER_QUANTITIES,) \
            or covariance.shape != (N_POWER_QUANTITIES, N_POWER_QUANTITIES):
        raise ValueError("theta/covariance must match the power quantity subset")
    if not candidate_ns:
        raise ValueError("candidate_ns must be non-empty")
    generator = np.random.default_rng(int(seed))
    max_n = int(max(candidate_ns))

    per_n: dict[int, Any] = {}
    selected_n: int | None = None
    for n in (int(v) for v in candidate_ns):
        cov_n = covariance * (float(n_reference) / n)
        se_n = np.sqrt(np.diag(cov_n))
        scenarios: dict[str, Any] = {}
        for shrink in (float(s) for s in shrink_scenarios):
            mean = float(shrink) * theta_hat  # equal scaling; signs never flip
            draws = generator.multivariate_normal(mean, cov_n, size=int(monte_carlo))
            evaluation = evaluate_confirm_draws(draws, se_n)
            scenarios[f"{shrink:.2f}"] = {
                "joint_power": float(evaluation["passed"].mean()),
                "marginal_power": {
                    k: float(v.mean()) for k, v in evaluation["conditions"].items()},
            }
        main = scenarios[f"{float(selection_scenario):.2f}"]
        binding = min(main["marginal_power"], key=main["marginal_power"].get)
        per_n[n] = {
            "scenarios": scenarios,
            "joint_power_at_selection": main["joint_power"],
            "binding_condition": binding,
            "binding_power": main["marginal_power"][binding],
        }
        if selected_n is None and main["joint_power"] >= float(target):
            selected_n = n

    return {
        "selected_n": selected_n,
        "underpowered": selected_n is None,
        "per_n": per_n,
        "n_reference": int(n_reference),
        "selection_scenario": float(selection_scenario),
        "target": float(target),
        "max_candidate_n": max_n,
        "sensitivity_at_max_n": {
            scenario: per_n[max_n]["scenarios"][scenario]["joint_power"]
            for scenario in (f"{float(s):.2f}" for s in shrink_scenarios)
        },
    }


def _monte_carlo_seed(bootstrap_seed: int, npz_sha256: str) -> int:
    """Deterministic MC seed derived from frozen entropy + data hash (no new constant)."""
    material = int(npz_sha256[:8], 16) % (2**31)
    return int(np.random.SeedSequence([int(bootstrap_seed), material])
               .generate_state(1)[0])


def estimate_confirm_power(
    screen_quantities: Path,
    candidate_ns: Sequence[int] = s4cr2.CANDIDATE_NS,
    *,
    n_resamples: int = s4cr2.BOOTSTRAP_RESAMPLES,
    monte_carlo: int = s4cr2.POWER_MONTE_CARLO,
    shrink_scenarios: Sequence[float] = s4cr2.POWER_SHRINK_SCENARIOS,
    selection_scenario: float = s4cr2.POWER_SELECTION_SCENARIO,
    target: float = s4cr2.POWER_TARGET,
    bootstrap_seed: int = s4cr2.BOOTSTRAP_ENTROPY,
    progress: Callable[[str], None] | None = None,
) -> PowerReport:
    """Conditional joint power of the confirm gate at each candidate n (plan section 10).

    Reads ONLY the create-once per-group quantities; recomputes this run's paired
    bootstrap deterministically for the covariance; simulates the FULL AND-chain; selects
    the minimum viable n under the 75% scenario. NEVER creates or consumes any confirm
    resource.
    """
    inputs = load_power_inputs(screen_quantities)
    if progress:
        progress(f"loaded create-once screen quantities "
                 f"(sha256={inputs['npz_sha256'][:12]}…); recomputing the "
                 f"{int(n_resamples)}-draw whole-group paired bootstrap "
                 f"(entropy {int(bootstrap_seed)}) for the covariance")
    power_inputs = bootstrap_power_inputs(
        inputs["quantities"], seed=bootstrap_seed, n_resamples=n_resamples)
    if progress:
        progress("bootstrap covariance ready; simulating the confirm AND-chain")
    result = simulate_power(
        power_inputs["theta_hat"], power_inputs["covariance"],
        candidate_ns=candidate_ns, n_reference=power_inputs["n_reference"],
        monte_carlo=monte_carlo,
        seed=_monte_carlo_seed(bootstrap_seed, inputs["npz_sha256"]),
        shrink_scenarios=shrink_scenarios, selection_scenario=selection_scenario,
        target=target)
    theta_named = {name: float(power_inputs["theta_hat"][i])
                   for i, name in enumerate(POWER_QUANTITY_NAMES)}
    se_named = {name: float(np.sqrt(power_inputs["covariance"][i, i]))
                for i, name in enumerate(POWER_QUANTITY_NAMES)}
    return PowerReport(
        plan_id=s4cr2.PLAN_ID,
        status=(s4cr2.STATUS_UNDERPOWERED if result["underpowered"]
                else s4cr2.STATUS_DEV_READY),
        selected_n=result["selected_n"],
        underpowered=bool(result["underpowered"]),
        target=float(target),
        selection_scenario=float(selection_scenario),
        monte_carlo=int(monte_carlo),
        n_reference=int(power_inputs["n_reference"]),
        candidate_ns=tuple(int(n) for n in candidate_ns),
        shrink_scenarios=tuple(float(s) for s in shrink_scenarios),
        per_n=result["per_n"],
        theta_hat=theta_named,
        se_reference=se_named,
        screen_quantities_sha256=inputs["npz_sha256"],
        screen_quantities_receipt_sha256=inputs["receipt_sha256"],
        bootstrap={
            "seed": int(bootstrap_seed),
            "n_resamples": int(n_resamples),
            "confidence": float(s4cr2.CONFIDENCE_SCREEN),
            "design": ("covariance taken DIRECTLY from this run's whole-group paired "
                       "bootstrap draws (deterministically recomputed with the frozen "
                       "entropy); a 95% SE is never derived from the 80% CI width"),
        },
    )


# --- driver-level orchestration: gating + create-once write ----------------------------------------


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else (ROOT / value)


def diagnostic_path(config: Mapping[str, Any]) -> Path:
    return _resolve(s4cr2.validate_config(config)["run_dir"]) / "development" / DIAGNOSTIC_FILENAME


def run_confirmation_power_diagnostic(
    config: Mapping[str, Any],
    *,
    n_resamples: int = s4cr2.BOOTSTRAP_RESAMPLES,
    monte_carlo: int = s4cr2.POWER_MONTE_CARLO,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run the plan-10 diagnostic and write the create-once JSON. No confirm resources.

    Refused unless the one-shot screen report exists AND passed
    (``STAGE4CR2_DEV_READY_FOR_CONFIRMATION``). The diagnostic JSON is written exactly
    once (a second call raises ``FileExistsError``); no confirm lock, confirm schedule or
    confirm collection is created under ANY outcome (plan Global Constraint 7).
    """
    checked = s4cr2.validate_config(config)
    paths = s4cr2st.screen_artifact_paths(checked)
    report_path = paths["report"]
    if not report_path.is_file():
        raise ValueError(
            "the confirmation power diagnostic is refused: the one-shot screen has not "
            "been adjudicated (development_screen_report.json missing)")
    screen_report = json.loads(report_path.read_text(encoding="utf-8"))
    if screen_report.get("terminal_state") != s4cr2.STATUS_DEV_READY:
        raise ValueError(
            "the confirmation power diagnostic is legal ONLY after the screen passed "
            f"({s4cr2.STATUS_DEV_READY}); terminal_state="
            f"{screen_report.get('terminal_state')!r}")
    out_path = diagnostic_path(checked)
    report = estimate_confirm_power(
        paths["quantities_npz"], tuple(int(n) for n in checked["candidate_ns"]),
        n_resamples=n_resamples, monte_carlo=monte_carlo, progress=progress)
    payload = {
        **report.to_dict(),
        "terminal_state": report.status,
        "confirm_resources_created": False,
        "screen_report_sha256": sha256_file(report_path),
        "semantics": ("conditional confirmation power diagnostic only; reports the "
                      "minimum viable n when one exists, else "
                      "STAGE4CR2_CONFIRMATION_UNDERPOWERED; this plan never creates a "
                      "confirm lock, confirm schedule or confirm collection"),
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    write_once_json(out_path, payload)  # create-once: a second write raises
    payload["diagnostic_path"] = str(out_path)
    payload["diagnostic_sha256"] = hashlib.sha256(
        out_path.read_bytes()).hexdigest()
    return payload


__all__ = [
    "DIAGNOSTIC_FILENAME",
    "N_POWER_QUANTITIES",
    "POWER_QUANTITIES",
    "POWER_QUANTITY_NAMES",
    "P_I_AURC",
    "P_I_REGRET80",
    "P_PR",
    "P_SEED_I_AURC",
    "P_SEED_I_R80",
    "P_SPEARMAN_MARGIN",
    "P_STRATA",
    "P_TRIM",
    "PowerReport",
    "Z_95",
    "bootstrap_power_inputs",
    "diagnostic_path",
    "estimate_confirm_power",
    "evaluate_confirm_draws",
    "load_power_inputs",
    "run_confirmation_power_diagnostic",
    "simulate_power",
    "symmetrised_psd_covariance",
]
