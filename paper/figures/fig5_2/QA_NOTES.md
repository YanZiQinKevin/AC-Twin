# Figure QA notes

- Static preflight: no FAIL findings. PNG is sufficient for this preview; SVG/PDF are also exported. TIFF is not needed for a draft preview.
- Log warning: the displayed label is log(1 + r/h), with h = 0.01 and non-negative decision regret. The figure performs no logarithmic data transformation.
- Text uses Arial, with an enforced 7.5 pt minimum and editable SVG/PDF text.
- No synthetic quantitative values are drawn. Matrix grids denote token structure only.
- Source mapping: manuscript 5.2 defines representation, architecture, output transformations and supervised loss. Section 3.5 defines the legal-input/training-supervision separation.
- Normalization applies to continuous features, including global context; one-hot indicators are unchanged.
- Figure and original manuscript exports are preserved independently. The Word preview contains only section 5.2 plus the new diagram and English caption.
- Final visual review: architecture PNG inspected; both pages of the final Word preview inspected from section52_render_v2. No clipped text, overlapping labels, split figure/caption, or duplicated footer fields remain. Native manuscript equations and superscript citations are preserved.

## Integrated compact v2

- Static preflight: 13 PASS, 1 WARN, 0 FAIL. Raster-format warning is acceptable: 600 dpi PNG is used for the Word export, with SVG/PDF for editable line art.
- Figure visually inspected: labels and arrows fit; scoring multiplies the two transformed outputs per seed before averaging five complete scores. Warm-red supervision remains separate from the blue inference path.
- The complete Word export retains 7 editable tables, 14 native display equations and 55 references. Figure caption order is verified as 1–6. Section 5.2–5.3 prose was shortened by about 21% excluding the new figure/caption.
- Final integrated visual QA: all 33 pages inspected, including the compact diagram and its English caption on page 15. Internal cross-references in imported legends were remapped; pages 20 and 27 were re-inspected after the final repair, and all other 31 page PNGs were byte-identical to the reviewed render. No clipped diagram labels or split new figure/caption remain.
