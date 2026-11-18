# Fig. 6 of the paper: verdict per rule over the 17 planned member-in-federation cases.
# Source: results/review2/stats_twolevel_s0-19.csv (two-level verdicts, column tl_verdict).
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, numpy as np, pandas as pd
plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],'font.size':8.5,'axes.edgecolor':'#666666','axes.linewidth':0.6,
 'axes.grid':True,'grid.color':'#E6E6E6','grid.linewidth':0.6,'axes.axisbelow':True,'savefig.dpi':300,'savefig.bbox':'tight','axes.spines.top':False,'axes.spines.right':False})
import os
F=os.environ.get('FITD_RESULTS','results/review2/')   # folder with stats_twolevel_s0-19.csv
OUT=os.environ.get('FITD_FIGURES','figures/')
O=pd.read_csv(F+'stats_twolevel_s0-19.csv'); P=O[O.planned]
names=[('FedAvg (size)','FedAvg (size)'),('FedAvg (equal)','FedAvg (equal)'),('FedProx (μ=0.001)','FedProx'),('FedPer (local output layer)','FedPer'),
 ('Private first layer','First layer private'),('FedAvg + fine-tune','FedAvg + fine-tuning'),('FedNova (size)','FedNova (size), reference'),('FedNova (equal)','FedNova (equal), reference')]
R=[]
for k,lab in names:
    g=P[P.rule==k]; assert len(g)==17,(k,len(g))
    R.append((lab,int((g.tl_verdict=='gain').sum()),int((g.tl_verdict=='no clear difference').sum()),int((g.tl_verdict=='loss').sum())))
print(R)
C=['#009E73','#C8C8C8','#D9604A']; L=['Gain','No clear difference','Negative transfer']
f,ax=plt.subplots(figsize=(4.7,2.35)); y=np.arange(len(R))[::-1]
for yy,(n,*v) in zip(y,R):
    left=0
    for k,(cnt,c) in enumerate(zip(v,C)):
        w=cnt/17; ax.barh(yy,w,left=left,color=c,height=0.7,label=L[k] if yy==y[0] else None)
        if cnt: ax.text(left+w/2,yy,str(cnt),ha='center',va='center',fontsize=7,color='white' if k!=1 else '#222')
        left+=w
ax.axhline(1.5,color='#555',lw=0.7,ls=':')
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in R]); ax.set_xlim(0,1); ax.grid(axis='y',visible=False)
ax.set_xticks([0,.25,.5,.75,1]); ax.set_xticklabels(['0%','25%','50%','75%','100%'])
ax.set_xlabel('Share of the 17 member-in-federation cases')
ax.legend(ncol=3,frameon=False,fontsize=7,loc='lower center',bbox_to_anchor=(0.4,1.0))
os.makedirs(OUT,exist_ok=True); f.savefig(OUT+'ch4_verdicts.pdf')
