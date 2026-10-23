"""Acta artwork from original aggregate CSVs, at 130 mm manuscript width.

Only presentation changes; no estimation or model inference is performed.
"""
from pathlib import Path
import hashlib
import json
import string
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm
from manuscript_figures import MODEL_LABELS, MODEL_COLORS

ROOT=Path(__file__).resolve().parents[1]
OUT=Path(os.environ.get('GEO_BENCH_RESULTS_DIR', str(ROOT/'manuscript')))/'acta_tex'
OUT.mkdir(parents=True,exist_ok=True)
G=ROOT/'artifacts/g6'
plt.rcParams.update({'font.family':'Arial','font.size':8,'axes.labelsize':8,
 'axes.titlesize':9,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,
 'lines.linewidth':1,'axes.linewidth':.6,'pdf.fonttype':42,'ps.fonttype':42,
 'svg.fonttype':'none','savefig.facecolor':'white','mathtext.fontset':'custom',
 'mathtext.rm':'Arial','mathtext.it':'Arial:italic','mathtext.bf':'Arial:bold',
 'mathtext.sf':'Arial','mathtext.fallback':None})
models=list(MODEL_LABELS)
markers=['o','s','^','v','D','P','X','<','>','h','p','*','d']
styles=['-','--',':','-.']
sources={}
def read(path):
 p=G/path; sources[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
 return pd.read_csv(p,encoding='utf-8-sig')
def figure(rows=1,cols=1,height=3.5):
 return plt.subplots(rows,cols,figsize=(130/25.4,height),layout='constrained',squeeze=False)
def save(fig,name):
 for ext in ['pdf','eps','svg']:
  fig.savefig(OUT/f'{name}.{ext}')
 plt.close(fig)
def label(ax,i,description=''):
 ax.set_title(f'({string.ascii_lowercase[i]}) {description}',loc='left')
def modelstyle(m):
 i=models.index(m)
 return dict(color=MODEL_COLORS[m],marker=markers[i],linestyle=styles[i%4],markersize=4)
skill=read(Path('figures/skill_horizon_u.csv'))
fig,axs=figure(1,2,4.5)
for i,split in enumerate(['test_a','test_b']):
 ax=axs[0,i]
 x=skill[skill.split==split].pivot(index='model_id',columns='horizon',values='station_equal_mase').reindex(models)[[1,7,30,90,180]]
 norm=LogNorm(.7,45);cmap=plt.get_cmap('Blues')
 im=ax.imshow(x,norm=norm,cmap=cmap,aspect='auto')
 for iy in range(13):
  for ix in range(5):
   v=x.iloc[iy,ix];rgb=cmap(norm(v))[:3]
   # WCAG relative luminance selects a contrasting black/white cell label.
   linear=[z/12.92 if z<=.04045 else ((z+.055)/1.055)**2.4 for z in rgb]
   lum=sum(a*b for a,b in zip(linear,[.2126,.7152,.0722]))
   ax.text(ix,iy,f'{v:.1f}',ha='center',va='center',fontsize=8,color='black' if lum>.179 else 'white')
 ax.set_xticks(range(5),['1','7','30','90','180'])
 ax.set_yticks(range(13),[MODEL_LABELS[m] for m in models] if i==0 else ['']*13)
 ax.tick_params(length=0);ax.set_xlabel('Horizon (days)')
 for y in [2.5,5.5,8.5]:ax.axhline(y,color='white',lw=1.5)
 label(ax,i,'2024–2025' if i==0 else '2026')
cb=fig.colorbar(im,ax=list(axs[0]),location='bottom',shrink=.85,pad=.06,aspect=35)
cb.set_ticks([1,2,5,10,20,40],labels=['1','2','5','10','20','40'])
cb.set_label('Station-equal MASE (logarithmic color scale; lower is better)')
save(fig,'Fig2')

def effects(frame,name,h3=False):
 if h3:
  groups=[frame[frame.candidate_track==t] for t in ['M','X']]
 else:
  groups=[frame[frame.reference==r] for r in ['seasonal_365','fourier_ridge']]
 fig,axs=figure(2,1,6.2)
 for j,(ax,g) in enumerate(zip(axs[:,0],groups)):
  ms=[m for m in models if m in set(g.candidate)]
  for i,m in enumerate(ms):
   for split,off,marker in [('test_a',-.13,'o'),('test_b',.13,'s')]:
    r=g[(g.candidate==m)&(g.split==split)].iloc[0]
    ax.plot([r.ci_lower,r.ci_upper],[i+off]*2,color=MODEL_COLORS[m])
    ax.plot(r.effect_delta_mase,i+off,marker,color=MODEL_COLORS[m],markersize=4,
     markerfacecolor=MODEL_COLORS[m] if r.holm_p_value<.05 else 'white')
  ax.set_yticks(range(len(ms)),[MODEL_LABELS[m] for m in ms]);ax.invert_yaxis();ax.axvline(0,color='black',ls='--',lw=.6)
  ax.set_xlabel('ΔMASE (negative favors candidate)')
  label(ax,j,(['M − U','X − U'] if h3 else ['Versus seasonal persistence','Versus Fourier ridge'])[j])
 axs[0,0].legend(handles=[Line2D([],[],color='black',marker='o',ls='',label='2024–2025'),Line2D([],[],color='black',marker='s',ls='',label='2026')],frameon=False)
 save(fig,name)
effects(read(Path('h1_primary_comparisons.csv')),'Fig3')
effects(read(Path('h3_information_gain.csv')),'Fig4',True)

p=read(Path('extended/probability_summary.csv'));p=p[(p.track=='U')&(p.horizon==90)]
fig,axs=figure(height=4.2);ax=axs[0,0]
for _,r in p.iterrows():
 ax.scatter(r.width_80,r.coverage_80,color=MODEL_COLORS[r.model_id],marker='o' if r.split=='test_a' else 's',s=30)
ax.axhline(.8,color='black',ls='--',lw=.6);ax.margins(.15,.18)
handles=[Line2D([],[],color=MODEL_COLORS[m],lw=3,label=MODEL_LABELS[m]) for m in models[-4:]]
handles += [Line2D([],[],color='black',marker=mark,ls='',label=lab) for mark,lab in [('o','2024–2025'),('s','2026')]]
ax.legend(handles=handles,ncols=3,loc='upper center',bbox_to_anchor=(.5,-.2),frameon=False)
ax.set(xlabel='Mean 80% interval width (raw units)',ylabel='Empirical 80% interval coverage');save(fig,'Fig5')

c=read(Path('extended/cost_summary.csv'));c=c[(c.track=='U')&(c.horizon==90)].merge(skill[skill.horizon==90],on=['model_id','split','horizon'])
fig,axs=figure(height=5);ax=axs[0,0]
for m in models:
 x=c[c.model_id==m]
 for _,r in x.iterrows():ax.plot(r.median_inference_seconds,r.station_equal_mase,marker=markers[models.index(m)],color=MODEL_COLORS[m],ls='',markersize=5,markerfacecolor=MODEL_COLORS[m] if r.split=='test_a' else 'white')
ax.set(xscale='log',yscale='log',xlabel='Median recorded seconds per window',ylabel='Station-equal MASE')
ax.legend(handles=[Line2D([],[],**modelstyle(m),label=MODEL_LABELS[m]) for m in models],ncols=3,loc='upper center',bbox_to_anchor=(.5,-.18),frameon=False)
save(fig,'Fig6')

ranks=read(Path('remaining/cross_year_ranks.csv'));r=ranks[ranks.horizon==90].pivot(index='model_id',columns='split',values='rank')
fig,axs=figure(height=4.8);ax=axs[0,0]
for m,row in r.iterrows():
 ax.plot([0,1],[row.test_a,row.test_b],**modelstyle(m))
 ax.text(-.04,row.test_a,MODEL_LABELS[m],ha='right',va='center');ax.text(1.04,row.test_b,MODEL_LABELS[m],va='center')
ax.set_xlim(-.55,1.55);ax.set_ylim(13.7,.3);ax.set_xticks([0,1],['2024–2025','2026']);ax.set_yticks([])
for s in ax.spines.values():s.set_visible(False)
save(fig,'FigS1')
s=read(Path('remaining/sensitivity_effects.csv'))
for name,variant in [('FigS2','Q1'),('FigS3','context365')]:
 fig,axs=figure(height=5.6);ax=axs[0,0]
 for i,m in enumerate(models):
  for split,off,mark in [('test_a',-.13,'o'),('test_b',.13,'s')]:
   row=s[(s.variant==variant)&(s.model_id==m)&(s.split==split)].iloc[0]
   ax.plot([row.ci_lower,row.ci_upper],[i+off]*2,color=MODEL_COLORS[m]);ax.plot(row.effect_delta_mase,i+off,mark,color=MODEL_COLORS[m],markersize=4,markerfacecolor=MODEL_COLORS[m] if row.ci_lower>0 or row.ci_upper<0 else 'white')
 ax.set_yticks(range(13),[MODEL_LABELS[m] for m in models]);ax.invert_yaxis();ax.axvline(0,color='black',ls='--',lw=.6);ax.set_xlabel('ΔMASE (negative favors variant)')
 save(fig,name)
h=read(Path('remaining/station_u90.csv'))
fig,axs=figure(2,1,6.4)
for i,split in enumerate(['test_a','test_b']):
 ax=axs[i,0];x=h[h.split==split].pivot(index='model_id',columns='station',values='mase').reindex(models)
 ax.imshow(x,norm=LogNorm(1,50),cmap='Blues',aspect='auto')
 for iy in range(len(x)):
  for ix in range(len(x.columns)):
   v=x.iloc[iy,ix];ax.text(ix,iy,f'{v:.1f}',ha='center',va='center',fontsize=8,color='white' if v>9 else 'black')
 ax.set_xticks(range(len(x.columns)),x.columns.astype(str));ax.tick_params(axis='x',rotation=45)
 ax.set_yticks(range(13),[MODEL_LABELS[m] for m in models]);label(ax,i,'2024–2025' if i==0 else '2026')
save(fig,'FigS4')
e=read(Path('extended/quality_event_strata.csv'))
fig,axs=figure(2,1,5.7)
for i,split in enumerate(['test_a','test_b']):
 ax=axs[i,0];x=e[(e.split==split)&(e.horizon==90)&(e.track=='U')&e.model_id.isin(models[-4:])]
 for m,g in x.groupby('model_id'):
  order=['none','spike','step','step_and_spike']
  # Preserve the stored categorical order when source uses alternate names.
  order=['no_target_event','spike_candidate','step_candidate','step_and_spike']
  g=g.set_index('event_stratum').reindex(order)
  ax.plot(range(len(order)),g.mase,label=MODEL_LABELS[m],**modelstyle(m))
 ax.set_xticks(range(len(order)),['None','Spike','Step','Step + spike']);ax.set_yscale('log');ax.set_ylabel('Station-equal MASE');label(ax,i,'2024–2025' if i==0 else '2026')
axs[1,0].legend(ncols=2,frameon=False,loc='upper center',bbox_to_anchor=(.5,-.2));save(fig,'FigS5')

# Protocol schematic; widths are schematic rather than proportional durations.
from matplotlib.patches import FancyBboxPatch
fig,axs=figure(height=5.9);ax=axs[0,0];ax.axis('off');ax.set(xlim=(0,1),ylim=(0,1))
def box(x,y,w,h,body,face='#edf3f8'):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.008,rounding_size=0.008',
                            linewidth=.7,edgecolor='#465567',facecolor=face))
 ax.text(x+w/2,y+h/2,body,ha='center',va='center',fontsize=8,linespacing=1.4)
def arrow(x1,y1,x2,y2):
 ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle='->',lw=.8,color='#465567'))
ax.text(.01,.98,'(a) Temporal protocol',fontsize=9,weight='bold',va='top')
for x,body,face in [(.01,'Training\n2016–2020','#e4eef6'),(.265,'Validation\n2021–2023','#e2f0eb'),
                    (.52,'Test A\n2024–2025','#fff0dc'),(.775,'Test B\nJan–Aug 2026','#f7e4e3')]:
 box(x,.83,.215,.09,body,face)
for x in [.225,.48,.735]:arrow(x,.875,x+.035,.875)
ax.text(.01,.79,'Configuration selected by 2023; test weights frozen',fontsize=8)
ax.text(.01,.755,'Horizons: 1, 7, 30, 90 (primary), 180 days',fontsize=8)
ax.text(.01,.72,'Shared maximum history: 730 days',fontsize=8)
ax.text(.01,.665,'(b) Controlled information tracks',fontsize=9,weight='bold')
box(.01,.53,.45,.10,'U: target history')
box(.54,.53,.45,.10,'M: same-station\nresistivity histories')
arrow(.47,.58,.53,.58)
box(.54,.385,.45,.11,'X: M + historical water level,\ntemperature and rainfall')
box(.01,.385,.45,.11,'O: X + future environment\nDefined; not evaluated','#f1f1f1')
arrow(.765,.52,.765,.505);arrow(.53,.44,.47,.44)
ax.text(.01,.335,'(c) Paired evaluation and aggregation',fontsize=9,weight='bold')
box(.01,.235,.98,.065,'Predict → pair on common successful windows')
arrow(.5,.225,.5,.195)
box(.01,.115,.98,.065,'Target days → windows → channels → stations')
ax.text(.5,.065,'Nine stations receive equal weight',ha='center',fontsize=8)
save(fig,'Fig1')
(OUT/'ARTWORK_MANIFEST.json').write_text(json.dumps({'sources':sources,'width_mm':130,'minimum_font_pt':8,'font':'Arial','generator':'analysis/acta_figures.py','matplotlib':matplotlib.__version__},indent=2))
print('Wrote 11 figures in PDF, EPS and SVG at 130 mm width.')
