# What was curated and what was not

This ZIP was assembled directly from the author's 12 uploaded archives and supplemental workbook, without altering original source `.py` code or data values. Where the supplied ZIP contained multiple historical or nested packages, they are grouped by named workflow in `workflows/`. To avoid redistributing publisher PDFs and duplicate drafts, `.pdf` files and most `.docx` files were removed; legacy SI `.docx` documents read directly by reproduction scripts were retained as necessary computational inputs. Cache/log files, Line 1 trained `.joblib` model binaries, and individual CSV outputs above 4 MB were not included in the code ZIP. The model binaries were retained separately in a companion downloadable ZIP. Original per-package hash manifests may no longer validate due to omitted files; use the new `MANIFEST_SHA256.json`. Re-extract the original user ZIP when a complete untouched historical package is required.

No hyperparameters, measured values, published R² values, or feature definitions were changed by the packaging process. Syntax and file-level integrity can be checked, but **scientific full-run numeric reproduction is not implied by packaging alone**.

## 26 November 2026: Figure 5–6 update

Added the supplied panel archives, exact manuscript composites, related Figure S16 artwork, and the final supplementary workbook alongside the historical workbook. The only adaptation to existing plotting code is the separate portable Figure 6d script, which reads exact archived input files directly instead of opening a ZIP; the original script remains intact. New workflow READMEs distinguish plotting from saved-model inference and note unavailable upstream scripts. No scientific computations were rerun.
