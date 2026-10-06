"""Focused tests for the non-adjudicating D2F paper figures."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from ac_twin.stage4d_paper_visualization import (  # noqa: E402
    frenet_to_world,
    peak_step,
    platform_legibility,
    select_cases,
    world_to_ego_plot,
)
from ac_twin.stage4d_external_features import PolylineFrame  # noqa: E402


def test_case_selection_uses_frozen_eligibility_and_order() -> None:
    # High risk alone is not a success: the baseline must miss it and regret > .01.
    regret = np.array([.31, .02, .60, .70, .45, .40, .35, .30, .25, .20], dtype=float)
    direct = np.array([.90, .80, .95, .01, .04, .08, .12, .16, .20, .24])
    baseline = np.array([.10, .90, .20, .30, .40, .50, .60, .70, .80, .95])
    a_hat = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    a_star = np.array([1, 2, 2, 4, 5, 6, 7, 8, 9, 0])
    selected = select_cases(regret, direct, baseline, a_hat, a_star,
                            [f"scenario-{i}" for i in range(10)])
    assert selected["success_indices"] == [0]
    assert selected["success_index"] == 0
    # The mechanically selected false negative is highest-regret with lowest risk.
    assert selected["failure_index"] == 3


def test_platform_legibility_reads_geometry_only() -> None:
    history = np.zeros((10, 5, 8), dtype=float)
    history[-1, :, 0] = 1
    history[-1, :, 2:4] = [[0, 0], [8, 3], [16, -3], [-8, 4], [-14, -4]]
    good = platform_legibility(history)
    overlapping = history.copy()
    overlapping[-1, 1, 2:4] = overlapping[-1, 0, 2:4]
    assert good["score"] > platform_legibility(overlapping)["score"]
    missing = history.copy()
    missing[-1, 4, 0] = 0
    assert good["score"] > platform_legibility(missing)["score"]


def test_frenet_world_coordinates_and_earliest_peak() -> None:
    frame = PolylineFrame(np.array([[0., 0.], [10., 0.], [20., 0.]]))
    xy = frenet_to_world(frame, np.array([2., 13.]), np.array([3., -4.]), origin_s=5.)
    np.testing.assert_allclose(xy, [[7., 3.], [18., -4.]])
    np.testing.assert_allclose(world_to_ego_plot(xy, frame, origin_s=5.),
                               [[2., 3.], [13., -4.]])
    assert peak_step(np.array([[0., 0.], [2., 0.], [2., 0.]]),
                     np.zeros((3, 2))) == 1
