Figure 6c – Integrated THM and NDMA time response
Final version: adjusted color scale (-0.06 to +0.60).

Contents
- Figure6c_integrated_source_data.csv: 65 plotted predictions, matched-measurement flags, observed values where available, and fold predictions.
- plot_Figure6c.py: portable version of the final plotting script.
- Figure6c_integrated_THM_NDMA_adjusted_colorbar_600dpi.png: final 600 dpi image.
- Figure6c_integrated_THM_NDMA_adjusted_colorbar.svg: final vector image.

Replotting: extract all files into one folder, install numpy, pandas, matplotlib, and Pillow, and run python plot_Figure6c.py. The script uses the archived predictions; it does not retrain or rerun the LightGBM models.
The NDMA 48 h values are model predictions without matched observations. Some raw NDMA predictions are negative on the concentration scale. Refer to the CSV flags.
