"""Fig. 12 of the paper (SPEDIA under plain gradient descent): change in SPEDIA's PR-AUC against its own
plain-SGD local model. The values below are the SPEDIA rows of family "E-1 vs SGD local" in
results/review2/stats_e1.csv (column diff), rounded to 3 decimals. Writes ch13_sgd_spedia.pdf/.png to the
current folder."""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np
from matplotlib.patches import Patch
plt.rcParams.update({'font.family':'serif','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
groups=['with CERT r4.2','with CERT r5.2','with LANL','with both CERT','with all three others']
rules=['FedAvg, size weights','FedAvg, equal weights','FedNova, size weights','FedNova, equal weights']
cols=['#0072B2','#56B4E9','#CC79A7','#B5176B']
v={0:[-0.319,-0.334,-0.334,-0.353,-0.216],1:[-0.286,-0.158,-0.348,-0.332,-0.208],2:[-0.135,-0.256,-0.353,-0.240,-0.286],3:[0.043,0.056,0.020,0.043,0.036]}
fig,ax=plt.subplots(figsize=(5.6,4.1))
h=0.2; y0=np.arange(5)[::-1]*1.0
for r in range(4):
    for g in range(5):
        y=y0[g]+(1.5-r)*h
        ax.barh(y,v[r][g],height=h*0.88,color=cols[r],zorder=2)
        x=v[r][g]; ax.text(x+(0.006 if x>=0 else -0.006),y,f'{x:+.2f}',va='center',ha='left' if x>=0 else 'right',fontsize=6.5,color='#333333')
ax.axvline(0,color='black',lw=0.9,zorder=4)
ax.set_yticks(y0); ax.set_yticklabels(groups)
ax.set_xlim(-0.45,0.12)
ax.set_xlabel("Change in SPEDIA's PR-AUC against its own model (0.932)")
ax.grid(axis='x',color='#e5e5e5',lw=0.6); ax.set_axisbelow(True)
ax.legend(handles=[Patch(color=c,label=l) for c,l in zip(cols,rules)],loc='upper center',bbox_to_anchor=(0.42,-0.15),ncol=2,frameon=False,fontsize=8)
fig.tight_layout(); fig.savefig('ch13_sgd_spedia.pdf'); fig.savefig('ch13_sgd_spedia.png',dpi=150)
