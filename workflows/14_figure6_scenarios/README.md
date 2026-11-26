# Figure 6 — Water treatment research scenarios (Text S10)

The scenarios use the frozen Line 2 Stage 3 publication-grouped LightGBM model. They are conditional model predictions, not experimental dose-response measurements or validated optimal treatment conditions.

| Panel | Folder | Command (run within folder) | Scope |
|---|---|---|---|
| 6a | `figure6a/` | `python plot_Figure6a.py` | Replot saved disinfectant-type/dose curves |
| 6b | `figure6b/` | `python plot_Figure6b_TWP.py` | Generate pH scenarios using five saved fold models; no retraining |
| 6c | `figure6c/` | `python plot_Figure6c.py` | Replot saved 65 predictions across 13 scenarios, at 6, 12, 24, 48, 72 h |
| 6d | `figure6d/` | `python plot_Figure6d_Ref29_portable.py` | Apply saved Ref. 29 calibration selection/offset to frozen predictions; same 11 evaluation records; no retraining |

Figure 6d's original revised script is retained. Its portable version reads the five exact archived Figure 4 input/output files included in `frozen_figure4_inputs/`, rather than requiring the original ZIP beside the script. The archived input version is explicitly retained to avoid silently mixing revisions.

Figure 6b includes its original frozen model archive. Use the environment documented in `../13_figure5_shap/figure5a_c/requirements.txt`. For 6a and 6c, the supplied archives contain predictions and plotting scripts, not the complete original upstream scenario-generation scripts. The 6c CSV includes fold-level predictions; negative back-transformed PA–NDMA predictions are retained in log-space for transparency and are not physically meaningful concentrations.

`final_manuscript/Figure6.png` is the exact composite embedded in the final manuscript of 26 November 2026. The 6d revised plotting package removes the earlier footer and adjusts the layout. No scientific computation was rerun during repository packaging.
