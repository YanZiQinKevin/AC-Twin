# Fig. 5 Full-Layout Rebuild Implementation Plan

> 状态：`HISTORICAL_BUILD_RECIPE / OUTPUT_AUTHOR_APPROVED_2026-10-03`。本文件仅为已完成图稿的构建来源，不是待执行任务。作者已固定现有 `Fig5_metadrive_external_validation.png`；当前正文身份、图注及配套格式见 `README.md` 与 `AUTHOR_APPROVAL.json`。下方原任务清单保留当时记录，不用于自动重建或重新选择 Fig. 5。

> **For agentic workers:** execute inline in this task. The workspace is not a Git repository, so no commit step applies.

**Goal:** Rebuild the original full Fig. 5 composition at 180 mm width while preserving the six frozen qualitative panels and using a genuine MetaDrive scene whose factual and counterfactual ego tracks are visually separable.

**Architecture:** A single Python/Matplotlib entry point reads only frozen Stage 4D-D2F paper-visualization inputs. It mechanically selects one of the existing 20 platform candidates by visible-entity count, factual/counterfactual separation, original legibility score, and hash; renders the matching MetaDrive screenshot; and draws the platform pair plus the two frozen qualitative case triptychs into one vector figure.

**Tech Stack:** Python, NumPy, Matplotlib, Pillow, frozen MetaDrive/ScenarioNet simulation environment.

## Global Constraints

- Platform illustration only; no training, inference, resampling, Gate calculation, or confirmation re-adjudication.
- Candidate pool remains the frozen K=20 pool.
- Selection order is fixed before rendering: `visible_entities` descending, `max_branch_separation_m` descending, original `score` descending, candidate `hash` ascending.
- The expected selected scene is `1cac180d139c96de`, context `13`, with five visible entities and approximately 3.134 m maximum branch separation.
- The MetaDrive FPS overlay may remain.
- The success and false-negative cases, actions, regret values, percentiles, and peak steps remain byte-for-byte sourced from the existing create-once receipt.
- Output width is 180 mm; all rendered glyphs must be at least 5 pt.

---

### Task 1: Lock the visualization-only selection rule

**Files:**
- Modify: `paper/figures/fig5/build_fig5.py`
- Test: `tests/test_fig5_paper_render.py`

- [ ] Add a failing test for lexicographic platform selection and the expected real frozen candidate.
- [ ] Run the focused test and confirm failure because the selector does not exist.
- [ ] Implement the selector and rerun the focused test.

### Task 2: Render the matching platform scene and full figure

**Files:**
- Modify: `paper/figures/fig5/build_fig5.py`

- [ ] Reuse the frozen MetaDrive replay contract to capture the mechanically selected scene at `t*`.
- [ ] Draw the red dashed counterfactual ego trajectory above the green factual trajectory and assert separation exceeds 1 m.
- [ ] Reconstruct the two frozen qualitative case rows from the create-once receipt.
- [ ] Export SVG, PDF, and 600 dpi PNG at 180 × 160 mm.

### Task 3: Verify and document

**Files:**
- Modify: `paper/figures/fig5/README.md`
- Modify: `paper/figures/fig5/FIGURE_LEGEND.md`

- [ ] Record the visualization-only selection rule and non-inferential boundary.
- [ ] Verify output existence, pixel dimensions, selected IDs, unchanged case IDs, and visible red pixels in panel (b).
- [ ] Run source, PDF text, collision, and final-size visual checks; record any intentional warnings.

