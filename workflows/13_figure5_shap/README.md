# Figure 5 — Model interpretation (Text S9)

All SHAP values are additive contributions to `ln(1 + C_DBP)` from the Line 2, Stage 3, publication-grouped LightGBM workflow. The canonical data comprise 995 out-of-fold records, 22 publications and 55 original features. Do not infer causality from SHAP values.

| Panel | Folder | Reproduction command (run within folder) | Scope |
|---|---|---|---|
| 5a | `figure5a_c/` | `python plot_Figure5a.py` | Canonical saved SHAP; reconstructed visual design, not exact original beeswarm positions |
| 5b–c | `figure5a_c/` | `python prepare_plot_data.py` then `python figures/plot_Figure5b_c.py` | Saved row selection and canonical SHAP; 739 DOC and 829 MP-loading records |
| 5d | `figure5d/` | `python plot_Figure5d_Aligned.py` | Replot archived publication-averaged cell summaries; minimum 10 records in treated and untreated groups |
| 5e | `figure5e/` | `python plot_Figure5e.py` | Match recorded conditions from saved training/SHAP data; equal publication weighting; 68 records, 17 conditions, 5 publications |
| S16 | `figureS16/` | See its README | Submitted artwork and reference to source data; original standalone plot script not available |

To regenerate the unified SHAP matrix from the five saved fold models (without fitting new models), use `figure5a_c/build_master.py` with the archived workbook and pinned environment in that folder; see `README_MASTER.txt`. The `legacy/` files are provenance inputs and are not the canonical SHAP values. The 5d plotting script starts from a saved cell summary, rather than recomputing it from the fitted models.

`final_manuscript/Figure5.png` is the exact composite embedded in the final manuscript of 26 November 2026. `supplied_Figure5c.png` is a separately supplied panel. Saved/replotted individual panel images need not be pixel-identical to the composite due to formatting and assembly. No scientific computation was rerun during repository packaging.
