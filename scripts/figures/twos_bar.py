"""Fig. 14 of the paper (TWOS): change in TWOS's PR-AUC against its own local model, with two-level 95% intervals.
The values below are the TWOS rows of family "E-11 federated" in results/review2/stats_twolevel_e11.csv
(columns mean_diff, tl_lo, tl_hi), rounded to 3 decimals. Group-level numbers only. Writes ch12_twos.pdf/.png
to the current folder."""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np
plt.rcParams.update({'font.family':'serif','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
groups=['with all four others','with SPEDIA','with LANL','with CERT r5.2','with CERT r4.2']
rules=['FedAvg, size weights','FedAvg, equal weights','First layer private']
cols=['#0072B2','#56B4E9','#009E73']
d=[[(0.109,-0.154,0.431),(0.153,-0.157,0.492),(0.281,0.069,0.472)],
   [(0.128,0.012,0.289),(0.247,0.068,0.430),(0.186,0.040,0.349)],
   [(-0.006,-0.215,0.206),(0.149,-0.074,0.363),(0.223,0.016,0.439)],
   [(0.035,-0.195,0.387),(0.060,-0.159,0.388),(0.193,-0.006,0.405)],
   [(0.017,-0.230,0.329),(0.089,-0.163,0.374),(0.176,0.006,0.363)]]
fig,ax=plt.subplots(figsize=(5.6,3.9))
h=0.26; y0=np.arange(len(groups))[::-1]
for r in range(3):
    for g in range(5):
        m,lo,hi=d[g][r]; y=y0[g]+(1-r)*h; clear=lo>0
        ax.barh(y,m,height=h*0.88,color=cols[r] if clear else 'white',edgecolor=cols[r],linewidth=1.2,hatch=None if clear else '////',zorder=2)
        ax.plot([lo,hi],[y,y],color='#555555',lw=0.9,zorder=3)
        for x in (lo,hi): ax.plot([x,x],[y-0.06,y+0.06],color='#555555',lw=0.9,zorder=3)
ax.axvline(0,color='black',lw=0.9,zorder=4)
ax.set_yticks(y0); ax.set_yticklabels(groups)
ax.set_xlabel("Change in TWOS's PR-AUC against its own model (0.423)")
ax.grid(axis='x',color='#e5e5e5',lw=0.6,zorder=0); ax.set_axisbelow(True)
from matplotlib.patches import Patch
hs=[Patch(facecolor=c,edgecolor=c,label=l) for c,l in zip(cols,rules)]
hs.append(Patch(facecolor='white',edgecolor='#555555',hatch='////',label='hatched: interval includes zero'))
ax.legend(handles=hs,loc='upper center',bbox_to_anchor=(0.45,-0.16),ncol=2,frameon=False,fontsize=8)
fig.tight_layout(); fig.savefig('ch12_twos.pdf'); fig.savefig('ch12_twos.png',dpi=150)
