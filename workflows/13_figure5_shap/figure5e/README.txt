Figure 5e: data-based bromide SHAP plots
Run: python plot_Figure5e.py
Dependencies: pandas, numpy, matplotlib, Pillow.
Inputs: original training records and unified Master SHAP Dataset (995 rows, 55 SHAP features), Line 2 Stage 3 publication-grouped LightGBM OOF models. No retraining or imputation performed.
Match by publication and recorded experimental fields (including identical missing-value patterns); retain positive recorded bromide and exactly one record of each TCM, BDCM, DBCM, TBM per condition. 17 candidate matched groups (68 records), five publications. Counts by ref: 3=1, 4=1, 6=6, 17=4, 23=5. Average the bromide SHAP value within each publication/species, then equally across five publications. Values are additive contributions on the ln(1+DBP concentration) prediction scale, not causal effects or measured speciation fractions. Matching is database-based and has not been verified against the full experimental details of the original publications. Positive bromide selection avoids ambiguity of encoded zero values. Lines connect species-specific study means and are not dose-response curves.
PNG: 5160 x 3540 pixels, 600 dpi. SVG: vector.

Final Option 2: only legend moved into lower-right axes, in two columns. All data and other visual settings unchanged.
