# Graphical Abstract Caption Draft

**Graphical abstract | AC-Twin overview: from response-conditioned factual evidence to auditable counterfactual decisions.**
(a) A shared pre-intervention history (Observed prefix) opens two worlds: the factual
continuation and an alternative ego action. (b) Three prediction variants — history-only, the
correct factual suffix and a matched shuffled suffix — and, below them, the event-specific value
`V_evt` of the factual suffix in each frozen response bin (zero / low / mid / high). The four dot
heights are the frozen Stage-2 point estimates (`0.3636`, `0.0841`, `-0.0661`, `0.2064`) and carry
no interval here; the full per-bin intervals and the three preregistered contrasts are in Fig. 1.
(c) The ten candidate action branches of one decision group enter the frozen base selector; the
selected action is scored by the directly supervised audit head, and the lowest-risk decision
groups are retained while the high-risk groups are flagged. The risk–coverage glyph at the right is
a schematic of that evaluation and **is not a result from any confirmation set**; the confirmed
risk–coverage curves are in Fig. 3. The audit ranks the selected action of each decision group; it
does not update the frozen predictor and does not execute a fallback policy. The bottom band lists
the three measured evidence scopes: in-distribution Merge and the pre-registered generator mismatch
share the frozen Merge weights, whereas MetaDrive / Waymo uses a pipeline retrained and frozen
inside that platform. Panel (b) motivates the decision-level audit in (c); the three scopes differ
in task distribution, baseline and model relation, and their effect sizes are neither ranked nor
pooled against one another.

---

## Production and interpretation notes

- No internal title or footer; the caption provides the explanation, consistently with the main figures.
- No numerical results are printed inside the image; only the **heights** of the four points in (b) use frozen quantities.
- The right-hand risk–coverage curve in (c) is **schematic**, distinct from the confirmed curves; this qualification must remain in the caption.
- The three scope blocks show names only, not effect sizes, and share one background color; scope differences do not imply different method colors.
- Avoid `external generalization`; use `platform-internal retraining` for MetaDrive.
- Use `AC-Twin audit` for the result-level method; `Direct audit head` is a component name. The image uses
  `Direct risk head`, aligned with the component terminology in the main text.
