import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from env.grouped_preference import grouped_features,grouped_utility,synthetic_grouped_users
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);args=p.parse_args();root=args.directory
r=json.loads((root/'diagnostics.json').read_text());users=synthetic_grouped_users();names=list(users)
fig,axes=plt.subplots(1,2,figsize=(13,4.5),dpi=160,sharey=True);summary={}
rng=np.random.default_rng(95031)
def utility(key,preference):
 a=np.load(root/r['conditions'][key]['artifact'])
 good=(a['lengths']==65)&~a['numerical_failure']&~a['action_limit_failure']
 f=np.zeros((len(good),2,2));f[good]=grouped_features(a['positions'][good])
 return grouped_utility(f,preference)
for ax,regime in zip(axes,['centered','gaussian']):
 summary[regime]={};x=np.arange(len(names))
 for method,color,shift,label in [('learned_20','#2368ad',-.18,'Learned 20'),('oracle_soft','#e68a24',.18,'Oracle soft')]:
  means=[];errs=[]
  for name in names:
   delta=utility(f'{regime}__{name}__{method}',users[name])-utility(f'{regime}__task_only',users[name])
   mean=delta.mean();bs=delta[rng.integers(0,len(delta),(5000,len(delta)))].mean(1);lo,hi=np.quantile(bs,[.025,.975])
   means.append(mean);errs.append([mean-lo,hi-mean]);summary[regime][f'{name}__{method}']=dict(mean_gain=float(mean),paired_bootstrap_95=[float(lo),float(hi)])
  ax.bar(x+shift,means,width=.34,color=color,label=label,yerr=np.array(errs).T,capsize=2,error_kw={'linewidth':.8})
 ax.axhline(0,color='black',lw=.8);ax.set_xticks(x);ax.set_xticklabels(['UN-D','UN-W','UW-D','UW-W','LN-D','LN-W','LW-D','LW-W'],rotation=35,ha='right')
 ax.set_title(regime.title()+' initial states');ax.set_xlabel('User: preferred geometry - priority');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
axes[0].set_ylabel('True utility gain over task-only (all outcomes)');axes[1].legend(frameon=False)
fig.suptitle('Grouped inference | frozen SFPS | M=8, L=4 | 16 paired episodes per condition',fontsize=13)
fig.text(.5,.02,'D/W: direction/width priority. Incomplete outcomes have zero utility. Intervals bootstrap paired episode seeds; one feedback seed.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.06,1,.94));fig.savefig(root/'utility_gain.png',bbox_inches='tight');fig.savefig(root/'utility_gain.svg',bbox_inches='tight')
(root/'paired_utility_gains.json').write_text(json.dumps(summary,indent=2))
