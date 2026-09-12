"""Compare actual successful target modes across the paired oracle ablations."""
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent
MODES = ['upper_narrow', 'upper_wide', 'lower_narrow', 'lower_wide']
USERS = [f'{mode}_{priority}' for mode in MODES for priority in ('direction','width')]
RUNS = ['pilot-seed0', 'mode-aligned-soft-M8-std1', 'mode-aligned-best-M8-std1',
        'mode-aligned-ablation-M32-std1', 'mode-aligned-ablation-M128-std1',
        'mode-aligned-best-M128-std8']


def collect():
    rows = []
    for run in RUNS:
        directory = ROOT/run
        report = json.loads((directory/'diagnostics.json').read_text())
        config = report['config']
        for key, record in report['conditions'].items():
            pieces = key.split('__')
            if pieces[1] not in USERS:
                continue
            regime, user = pieces[:2]
            target = MODES.index(user.rsplit('_',1)[0])
            with np.load(directory/record['artifact']) as a:
                n = len(a['lengths'])
                complete = (a['lengths'] == 65) & ~a['numerical_failure'] & ~a['action_limit_failure']
                good = complete & a['success']
                y = a['positions'][good,32,1].astype(np.float64)
                mids = np.select([(y >= .12)&(y < .4), y >= .4,
                                  (y > -.4)&(y <= -.12), y <= -.4],range(4),default=4)
                count = int((mids == target).sum())
                rows.append(dict(run=run,key=key,regime=regime,user=user,
                    controller='learned20' if 'learned_' in key else 'oracle',
                    feature_kind=config.get('feature_kind','continuous'),method=record['method'],
                    M=a['current_latents'].shape[2],L=a['future_latents'].shape[3],
                    proposal_std=config.get('proposal_std',1.),episodes=n,
                    task_success_count=int(good.sum()),successful_target_count=count,
                    target_rate=count/n, fallback_count=int(a['fallback'].sum()),
                    mean_valid_current_fraction=float(a['current_valid'][a['decision_mask']].mean()),
                    batch_replan_seconds=float(a['batch_replan_seconds'].sum())))
    return rows


def main():
    rows = collect()
    out = ROOT/'analysis'/'mode-comparison';out.mkdir(exist_ok=True)
    (out/'conditions.json').write_text(json.dumps(rows,indent=2)+'\n')
    with (out/'conditions.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    columns = [(run,method) for run,method in [
        ('pilot-seed0','soft'), ('mode-aligned-soft-M8-std1','soft'),
        ('mode-aligned-best-M8-std1','best'), ('mode-aligned-ablation-M32-std1','best'),
        ('mode-aligned-ablation-M128-std1','best'), ('mode-aligned-best-M128-std8','best')]]
    lookup={(r['run'],r['regime'],r['user'],r['controller'],r['method']):r for r in rows}
    fig,axes=plt.subplots(1,2,figsize=(15,7),dpi=140,sharey=True)
    for ax,regime in zip(axes,['centered','gaussian']):
        matrix=np.array([[lookup[(run,regime,user,'oracle',method)]['target_rate']
                          for run,method in columns] for user in USERS])
        im=ax.imshow(matrix,vmin=0,vmax=1,cmap='YlGnBu',aspect='auto')
        for i,user in enumerate(USERS):
            for j,(run,method) in enumerate(columns):
                r=lookup[(run,regime,user,'oracle',method)]
                ax.text(j,i,f"{r['successful_target_count']}/{r['episodes']}",ha='center',va='center',
                        color='white' if matrix[i,j]>.55 else 'black',fontsize=10)
        ax.set_xticks(range(6),['Old feature\nsoft M8','Mode feature\nsoft M8','Mode feature\nbest M8',
                              'Mode feature\nbest M32','Mode feature\nbest M128','Mode feature\nbest M128\nlatent std8'],fontsize=9)
        ax.set_yticks(range(8),[u.replace('upper_narrow','UN').replace('upper_wide','UW')
                              .replace('lower_narrow','LN').replace('lower_wide','LW').replace('_',' / ')
                              for u in USERS],fontsize=10)
        ax.set_title(regime.title()+' starts',fontsize=14)
    fig.suptitle('Oracle ablation: actual task success AND target midpoint mode',fontsize=16)
    fig.text(.5,.02,'Same 16 episode seeds per condition, L=4, beta=16. All columns use latent std1 except the last. Repeated conditions are not independent samples.',ha='center',fontsize=9)
    fig.subplots_adjust(left=.13,right=.91,top=.88,bottom=.17,wspace=.13)
    cax=fig.add_axes([.93,.2,.014,.64]);fig.colorbar(im,cax=cax,label='Successful target-mode rate')
    for extension in ('png','svg'):fig.savefig(out/f'oracle_ablation.{extension}',bbox_inches='tight')
    plt.close(fig)
    lines=['| Profile | Centered before | Centered learned20 | Centered oracle | Gaussian before | Gaussian learned20 | Gaussian oracle |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for user in USERS:
        entries=[]
        for regime in ['centered','gaussian']:
            for run,controller,method in [('pilot-seed0','learned20','soft'),
                    ('mode-aligned-best-M128-std8','learned20','best'),
                    ('mode-aligned-best-M128-std8','oracle','best')]:
                r=lookup[(run,regime,user,controller,method)]
                entries.append(f"{r['successful_target_count']}/{r['episodes']}")
        lines.append('| '+user+' | '+' | '.join(entries)+' |')
    (out/'learned_comparison.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__ == '__main__':main()
