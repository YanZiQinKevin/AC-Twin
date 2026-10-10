# Section 5.2 architecture preview

- Claim: nine relative action-comparison tokens and global context feed a compact dual-output MLP trained with decision-regret supervision.
- Source: manuscript sections 5.2 and 3.5; no experimental records or plans consulted.
- Archetype: schematic-led composite, with representation, neural network, and training supervision.
- Reference adaptation: style-only inheritance (perspective modules, branching arrows, dashed grouping); the supplied graph module's operations are not reused.
- Backend: saved Python preference; matplotlib.
- Dimensions: 180 mm by 130 mm; editable SVG/PDF and 600 dpi PNG. Minimum text size 7.5 pt; panel letters 9.6 pt.
- Style: current Fig. 2 six-color semantic palette (blue #4D93D6/#A4C8EA for audit and inference, gray #A8A8A9/#D9D9D9 for frozen predictions, warm red #D57873/#F1C2AD for training truth). White background, no shadows or gradients; only flat perspective faces. Panel headings only, no overarching title or footer sentence.
- Integrity: qualitative architecture only; matrix cells indicate shape, not measured values or weights. Nine tokens each contain 20 continuous and 14 one-hot features. Five cost/rank features plus fifteen risk-channel features constitute the twenty continuous dimensions. Global context is nine-dimensional. Two hidden layers are Linear, LayerNorm, ReLU, Dropout (0.10). Final linear layer has two outputs.
- Information roles: true regret and its derived targets appear only in the training-supervision group. Solid arrows indicate forward operations; dashed orange arrows to the loss indicate training supervision.
- Initial preview scope: the unversioned figure contains Section 5.2 only.

## Integrated v2

- Current manuscript figure: `Direct_audit_head_architecture_v2.png`, numbered Fig. 3.
- Dimensions: 180 x 115 mm; min text 7.5 pt. Panel b is the architecture hero; a gives feature dimensions, c gives training-only supervision, and d gives per-seed scoring followed by five-seed averaging.
- Section 5.3 supplies the inference path: softplus transforms raw_mu, sigmoid transforms logit_high, the non-negative output is rescaled by h * expm1, multiplied by the classification score, and the five complete scores are averaged.
- Exact equations, target threshold, class weight, hidden-layer details and all training settings remain in prose and caption. No new scientific module is introduced.
- Reuse: original figure assets and the section-only preview are retained. Manuscript-facing figure numbering follows order of appearance without renaming existing asset directories.
