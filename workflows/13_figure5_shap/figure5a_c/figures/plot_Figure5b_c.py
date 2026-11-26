from pathlib import Path
# Figure 5b and Figure 5c corrected 2026-11-07.
# Only edit to the original final code: y-axis labelpad 12 -> 6,
# moving the vertical titles inward by 6 pt to prevent edge clipping.

import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, LogFormatterMathtext
from matplotlib.colors import LogNorm

# same larger font settings as last outputs
FONT_AXIS_LABEL = 22
FONT_TICKS = 18
FONT_LEGEND = 14
FONT_LEGEND_TITLE = 14
FONT_CBAR_LABEL = 18
FONT_CBAR_TICKS = 13
FONT_TITLE = 25
FONT_SUBTITLE = 13.5
DLO = 0.4
DHI = 512.863974025295

def marker_area(d):
    d = np.asarray(d, dtype=float)
    return 24 + 160 * np.log10(d / DLO) / np.log10(DHI / DLO)

plt.rcParams.update({'font.family': 'DejaVu Sans','pdf.fonttype': 42,'ps.fonttype': 42,'svg.fonttype': 'none'})

# Figure 5b
OUT = Path(__file__).resolve().parent / 'Source_Data_and_Parameters'
rows = pd.read_csv(OUT / 'Figure5b_AllDBPs_DOC_SHAP_Source_Data.csv')
trend = pd.read_csv(OUT / 'Figure5b_AllDBPs_DOC_SHAP_Trend_Summary.csv')
x = rows['DOC (mg/L C)'].to_numpy(dtype=float)
shap = rows['SHAP_DOC_on_ln1p_DBP'].to_numpy(dtype=float)
loading = rows['MP loading (g/L)'].to_numpy(dtype=float)
dose = rows['disinfectant dose (mg/L as Cl2)'].to_numpy(dtype=float)
disinfectant = rows['disinfectant'].astype(str).to_numpy()
fig = plt.figure(figsize=(12.3, 8.0), facecolor='white')
ax = fig.add_axes([.095, .125, .72, .76])
ax.set_xscale('log'); ax.set_xlim(.046,150); ax.set_ylim(-.7,.84)
ax.grid(which='major', color='#d1d9e0', linewidth=1.0, zorder=0); ax.axhline(0, color='#65778a', lw=1.5, zorder=1)
for spine in ('top','right'): ax.spines[spine].set_visible(False)
for spine in ('left','bottom'):
    ax.spines[spine].set_color('#a4b4c2'); ax.spines[spine].set_linewidth(1.0)
ax.set_xlabel('DOC (mg/L as C)', fontsize=FONT_AXIS_LABEL, labelpad=8)
ax.set_ylabel('SHAP value for DOC', fontsize=FONT_AXIS_LABEL, labelpad=6)
ax.tick_params(axis='both', labelsize=FONT_TICKS, length=5)
ax.set_yticks(np.arange(-.6,.81,.2))
ax.xaxis.set_major_locator(LogLocator(base=10, numticks=6)); ax.xaxis.set_major_formatter(LogFormatterMathtext())
ax.fill_between(trend['DOC_center'], trend['SHAP_q25'], trend['SHAP_q75'], facecolor='#9cb8d1', alpha=.36, zorder=1, edgecolor='none')
ax.plot(trend['DOC_center'], trend['SHAP_median'], lw=3.5, color='#1f5a85', zorder=3, solid_capstyle='round')
norm = LogNorm(vmin=.05, vmax=10); cmap=plt.get_cmap('viridis')
for typ, marker in [('Cl2','o'),('NH2Cl','^')]:
    idx = np.where(disinfectant==typ)[0]; idx = idx[np.argsort(dose[idx])]
    ax.scatter(x[idx], shap[idx], s=marker_area(dose[idx]), c=loading[idx], norm=norm, cmap=cmap, marker=marker, alpha=.70, edgecolors='white', linewidths=.5, zorder=4, rasterized=False)
shape_handles = [Line2D([],[], marker='o', ls='none', markersize=9, color='#557d99', markeredgecolor='white', label=f'Cl₂ (n = {sum(disinfectant == "Cl2")})'), Line2D([],[], marker='^', ls='none', markersize=9, color='#557d99', markeredgecolor='white', label=f'NH₂Cl (n = {sum(disinfectant == "NH2Cl")})')]
leg1 = ax.legend(handles=shape_handles, loc='lower right', bbox_to_anchor=(.99,.028), fontsize=FONT_LEGEND, title='Disinfectant type', title_fontsize=FONT_LEGEND_TITLE, fancybox=True, facecolor='white', edgecolor='#c7d2dd', framealpha=.95, handletextpad=.7, labelspacing=.3, borderpad=.48)
ax.add_artist(leg1)
legend_doses=[0.4,4,40,513]
size_handles=[plt.scatter([],[],s=marker_area(d),c='#557d99',edgecolors='white',linewidths=.6,label=f'{d:g}') for d in legend_doses]
# moved to upper-left region to avoid overlap; everything else unchanged
ax.legend(handles=size_handles, loc='upper left', bbox_to_anchor=(.03,.985), scatterpoints=1, fontsize=FONT_LEGEND-1, title='Disinfectant dose\n(mg/L as Cl₂; log scale)', title_fontsize=FONT_LEGEND_TITLE, fancybox=True, facecolor='white', edgecolor='#c7d2dd', framealpha=.97, labelspacing=.78, borderpad=.55, handletextpad=.9)
cax = fig.add_axes([.835,.23,.022,.61]); sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap); cb = fig.colorbar(sm, cax=cax, orientation='vertical')
cb.set_ticks([.05,.1,.5,1,2,5,10]); cb.set_ticklabels(['0.05','0.1','0.5','1','2','5','10']); cb.ax.tick_params(labelsize=FONT_CBAR_TICKS); cb.set_label('MP loading (g/L; log scale)', rotation=90, labelpad=4, fontsize=FONT_CBAR_LABEL); cb.outline.set_edgecolor('#b8c8d6')
fig.text(.095,.945,'DOC dependence of predicted DBP formation', fontweight='bold', fontsize=FONT_TITLE, color='#233e59', ha='left', va='center')
fig.text(.096,.903,f'All DBPs with measured DOC  |  n = {len(rows)} observations', fontsize=FONT_SUBTITLE, color='#566a80', ha='left', va='center')
outb = Path(__file__).resolve().parent / 'Figure5b_FINAL_600dpi.png'
fig.savefig(outb, dpi=600, facecolor='white'); fig.savefig(str(Path(__file__).resolve().parent / 'Figure5b_FINAL.svg'), facecolor='white'); fig.savefig(str(Path(__file__).resolve().parent / 'Figure5b_FINAL.pdf'), facecolor='white'); plt.close(fig)

# Figure 5c
BASE = Path(__file__).resolve().parent / 'Source_Data_and_Parameters'
rows = pd.read_csv(BASE/'Figure5c_AllDBPs_MPloading_ParticleSize_Source_Data_v2.csv')
trend = pd.read_csv(BASE/'Figure5c_AllDBPs_MPloading_Trend_Summary_v2.csv')
fig=plt.figure(figsize=(12.3,8.0),facecolor='white')
ax=fig.add_axes([.095,.125,.72,.76])
ax.set_xscale('log'); ax.set_xlim(.042,13); ax.set_ylim(-.40,.48)
ax.grid(which='major', color='#d1d9e0', linewidth=1.0, zorder=0); ax.axhline(0, color='#65778a', lw=1.5, zorder=1)
for spine in ('top','right'): ax.spines[spine].set_visible(False)
for spine in ('left','bottom'):
    ax.spines[spine].set_color('#a4b4c2'); ax.spines[spine].set_linewidth(1.0)
ax.set_xlabel('MP loading (g/L)', fontsize=FONT_AXIS_LABEL, labelpad=8)
ax.set_ylabel('SHAP value for MP loading', fontsize=FONT_AXIS_LABEL, labelpad=6)
ax.tick_params(axis='both', labelsize=FONT_TICKS, length=5); ax.set_yticks(np.arange(-.4,.41,.2)); ax.xaxis.set_major_locator(LogLocator(base=10, numticks=6)); ax.xaxis.set_major_formatter(LogFormatterMathtext())
trend_sel=trend[trend['included_in_trend']]
ax.fill_between(trend_sel['MP loading (g/L)'], trend_sel['SHAP_q25'], trend_sel['SHAP_q75'], facecolor='#9cb8d1', alpha=.36, zorder=1, edgecolor='none')
ax.plot(trend_sel['MP loading (g/L)'], trend_sel['SHAP_median'], lw=3.5, color='#1f5a85', zorder=3, solid_capstyle='round')
norm=LogNorm(vmin=5, vmax=5000); cmap=plt.get_cmap('viridis')
for typ, marker in [('Cl2','o'),('NH2Cl','^')]:
    rr = rows[rows['disinfectant']==typ].sort_values('dose_for_plotting')
    ax.scatter(rr['MP loading (g/L)'], rr['SHAP_MP_loading_on_ln1p_DBP'], s=marker_area(rr['dose_for_plotting']), c=rr['MP size (µm)'], norm=norm, cmap=cmap, marker=marker, alpha=.76, edgecolors='white', linewidths=.55, zorder=4)
shape_handles=[Line2D([],[],marker='o',ls='none',markersize=9,color='#557d99',markeredgecolor='white',label=f'Cl₂ (n = {(rows["disinfectant"]=="Cl2").sum()})'), Line2D([],[],marker='^',ls='none',markersize=9,color='#557d99',markeredgecolor='white',label=f'NH₂Cl (n = {(rows["disinfectant"]=="NH2Cl").sum()})')]
leg1=ax.legend(handles=shape_handles,loc='lower right',bbox_to_anchor=(.99,.028), fontsize=FONT_LEGEND,title='Disinfectant type',title_fontsize=FONT_LEGEND_TITLE, fancybox=True,facecolor='white',edgecolor='#c7d2dd',framealpha=.95, handletextpad=.7,labelspacing=.3,borderpad=.48)
ax.add_artist(leg1)
legend_doses=[0.4,4,40,513]
size_handles=[plt.scatter([],[],s=marker_area(d),c='#557d99',edgecolors='white',linewidth=.6,label=f'{d:g}') for d in legend_doses]
ax.legend(handles=size_handles, loc='upper left', bbox_to_anchor=(.03,.985), scatterpoints=1, fontsize=FONT_LEGEND-1, title='Disinfectant dose\n(mg/L as Cl₂; log scale)', title_fontsize=FONT_LEGEND_TITLE, fancybox=True, facecolor='white', edgecolor='#c7d2dd', framealpha=.97, labelspacing=.78, borderpad=.55, handletextpad=.9)
cax=fig.add_axes([.835,.23,.022,.61]); sm=plt.cm.ScalarMappable(norm=norm,cmap=cmap); cb=fig.colorbar(sm,cax=cax,orientation='vertical')
cb.set_ticks([5,10,50,100,500,1000,5000]); cb.set_ticklabels(['5','10','50','100','500','1,000','5,000']); cb.ax.tick_params(labelsize=FONT_CBAR_TICKS); cb.set_label('MP particle size (µm; log scale)', rotation=90, labelpad=4, fontsize=FONT_CBAR_LABEL); cb.outline.set_edgecolor('#b8c8d6')
fig.text(.095,.945,'MP-loading dependence of predicted DBP formation', fontweight='bold', fontsize=FONT_TITLE, color='#233e59', ha='left', va='center')
fig.text(.096,.903,f'All DBPs with measured MP loading and particle size  |  n = {len(rows)} observations', fontsize=FONT_SUBTITLE, color='#566a80', ha='left', va='center')
outc = Path(__file__).resolve().parent / 'Figure5c_FINAL_600dpi.png'
fig.savefig(outc, dpi=600, facecolor='white'); fig.savefig(str(Path(__file__).resolve().parent / 'Figure5c_FINAL.svg'), facecolor='white'); fig.savefig(str(Path(__file__).resolve().parent / 'Figure5c_FINAL.pdf'), facecolor='white'); plt.close(fig)
print(outb)
print(outc)
