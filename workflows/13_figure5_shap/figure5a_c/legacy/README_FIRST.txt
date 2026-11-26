Figure 5b & Figure 5c — FINAL corrected versions (2026-11-07)

The files in this directory are ONLY the current final versions.

Updated from the previously approved version:
  - Fixed clipping of the left vertical SHAP-axis labels by changing ylabel labelpad from 12 pt to 6 pt.
  - No other plot settings, data, colors, fonts, legends, line positions, or axes were changed.

Figures:
  Figure5b_FINAL_600dpi.png  (739 observations)
  Figure5c_FINAL_600dpi.png  (829 observations)
  Figure5b_FINAL.pdf and Figure5c_FINAL.pdf (vector format)
  Figure5b_FINAL.svg and Figure5c_FINAL.svg (vector format)

Both PNGs are 7380 × 4800 pixels and 600 dpi.

Reproduce:
  python reproduce_FINAL_Figure5b_Figure5c.py

Requirements: numpy, pandas, matplotlib.
The scripts read the source/trend CSV files in Source_Data_and_Parameters and reproduce the figures without recomputing model predictions or SHAP values.
