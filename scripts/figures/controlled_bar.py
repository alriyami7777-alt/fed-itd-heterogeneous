# Fig. 9 of the paper: controlled test (member A, FedAvg with equal weights, 5 runs).
# Values typed from Table A.16 of the paper (mean change; filled = interval excludes 0).
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, numpy as np
plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],'font.size':8.5,'axes.edgecolor':'#666666','axes.linewidth':0.6,
 'axes.grid':True,'grid.color':'#E6E6E6','grid.linewidth':0.6,'axes.axisbelow':True,'savefig.dpi':300,'savefig.bbox':'tight','axes.spines.top':False,'axes.spines.right':False})
import os
OUT=os.environ.get('FITD_FIGURES','figures/')
# ---- ch16: controlled test (labels as in Table A.16)
V=[('Reference',.331,True),('Small (1% of users)',.365,True),('Small, normal rate (10%)',.366,True),('Higher positive rate',.032,False),
   ('Logons only (blanked)',-.071,False),('Logons only (rebuilt)',-.040,False),('Small and logons only',.126,False)]
C=['#8C8C8C','#009E73','#009E73','#E69F00','#D55E00','#D55E00','#7B4FA6']
f,ax=plt.subplots(figsize=(4.8,2.6)); y=np.arange(len(V))[::-1]
for (n,v,clear),yy,c in zip(V,y,C):
    ax.barh(yy,v,color=c if clear else 'white',edgecolor=c,height=0.6,linewidth=1.2)
    ax.text(v+(0.01 if v>=0 else -0.01),yy,f'{v:+.3f}',va='center',ha='left' if v>=0 else 'right',fontsize=7.2)
ax.set_yticks(y); ax.set_yticklabels([v[0] for v in V]); ax.axvline(0,color='#333',lw=0.8); ax.set_xlim(-0.15,0.45)
ax.set_xlabel("Change in member A's PR-AUC (FedAvg, equal weights)")
ax.text(0.99,1.02,'filled = interval excludes 0  ·  5 runs',transform=ax.transAxes,ha='right',fontsize=6.5,color='#777')
os.makedirs(OUT,exist_ok=True); f.savefig(OUT+'ch16_controlled.pdf')
print('ok')
