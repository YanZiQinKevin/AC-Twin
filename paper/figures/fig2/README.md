# Fig. 1: Prediction and auditing workflow

The current asset is **v4**. Historical directory/filenames are retained:

- `Fig2_AC_Twin_method_overview_v4.png`: 600-dpi main-text image.
- Matching `.svg` and `.pdf`: editable/vector versions of the same figure.
- `build_fig2.py`: deterministic drawing script.
- `FIGURE_LEGEND.md`: caption.

Run `python paper/figures/fig2/build_fig2.py` from the repository root. It draws the established workflow using local `icon/` assets, without reading experiment data or computing statistics. Missing assets cause failure rather than substitution.

The final composition has no separate red training-label tag. The lower sealed-ground-truth region and red dashed supervision links remain. `Rank by audit score` uses predicted audit scores; true regret is used for training supervision and evaluation after scoring is frozen. Retention does not execute fallback control.

The canvas is 180 × 88 mm, cropped from the previous 180 × 100 mm drawing without changing physical icon or line dimensions. Font sizes were increased by 8% through `FONT_SCALE`. The figure retains five action intentions × two durations and fixed 80% retention; model dimensions, hyperparameters and equations belong in the methods/caption.

The author-specified palette is `#4D93D6`, `#A4C8EA`, `#D57873`, `#F1C2AD`, `#A8A8A9`, `#D9D9D9`: blues for method/inference, gray for the frozen predictor, warm colors for counterfactual branches and supervision/evaluation truth, light gray for neutral framing. `_PALETTE_*` constants control the palette.

Local iconfont SVGs are parsed as recolorable vectors. `audit.png` is a fixed 8 mm raster illustration; `authcode.svg` represents ranking. `Observed prefix.png` is the author-provided road/trajectory illustration. Asset filenames remain unchanged, including filenames originally supplied in Chinese.

Ten candidate plans are defined in `code/ac_twin/stage4c_objective.py::ACTION_PLANS`; coverage is `code/ac_twin/stage4c_protocol.py::COVERAGE_MAIN`. Feature/model definitions are in `stage4cp_features.py` and `stage4cp_models.py`.
