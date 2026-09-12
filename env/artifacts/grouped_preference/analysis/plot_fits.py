import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).parent
result=json.loads((root/'fit_replicates.json').read_text())
rows=result['rows'];budgets=[5,10,20,40]
fig,axes=plt.subplots(1,3,figsize=(12,3.8),dpi=160)
fields=[('overall_ordering_accuracy','Overall preference ordering',True),('priority_correct','Group priority identified',True),('alpha_error','Absolute error of direction importance',False)]
rng=np.random.default_rng(91731)
ci={}
for ax,(key,label,percentage) in zip(axes,fields):
 means=[];bounds=[]
 for k in budgets:
  seedmeans=np.array([np.mean([r[key] for r in rows if r['k']==k and r['seed']==seed]) for seed in range(20)])
  bootstrap=np.mean(rng.choice(seedmeans,(5000,20),replace=True),axis=1)
  low,high=np.quantile(bootstrap,[.025,.975]);m=seedmeans.mean()
  means.append(m);bounds.append([m-low,high-m]);ci[f'{key}_{k}']=[float(low),float(high)]
 means=np.array(means);bounds=np.array(bounds).T
 factor=100 if percentage else 1
 ax.errorbar(budgets,means*factor,yerr=bounds*factor,color='#2166a9',marker='o',lw=1.8,capsize=4)
 for k,m in zip(budgets,means):ax.annotate(f'{m*factor:.1f}' if percentage else f'{m:.3f}',(k,m*factor),textcoords='offset points',xytext=(0,9),ha='center',fontsize=9)
 ax.set_title(label,fontsize=11);ax.set_xticks(budgets);ax.set_xlabel('Total scoped judgments');ax.grid(alpha=.2)
 if percentage:ax.set_ylim(45,105);ax.set_ylabel('Percent')
 else:ax.set_ylim(0,.35);ax.set_ylabel('Mean absolute error')
fig.suptitle('Two-group few-shot fitting | 20 feedback seeds × 8 synthetic profiles',fontsize=13)
fig.text(.5,.015,'Fixed rho=16, L2=0.1. Intervals: bootstrap over 20 feedback/query seeds, each averaged across 8 profiles. Synthetic labels only.',ha='center',fontsize=8)
fig.tight_layout(rect=(0,.045,1,.93))
fig.savefig(root/'few_shot_fit.png',bbox_inches='tight');fig.savefig(root/'few_shot_fit.svg',bbox_inches='tight')
(root/'fit_confidence_intervals.json').write_text(json.dumps(ci,indent=2))
