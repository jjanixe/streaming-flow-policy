"""Scientific GIFs from saved actual Gym inference trajectories."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path)
args=parser.parse_args();root=args.directory
report=json.loads((root/'diagnostics.json').read_text()); out=root/'animations';out.mkdir(exist_ok=True)
modes=['upper_narrow','upper_wide','lower_narrow','lower_wide']
summary={}

def mask_mode(paths,lengths,mode):
    y=paths[:,32,1]; reached=lengths>32
    if mode=='upper_narrow': mask=(y>=.12)&(y<.4)
    elif mode=='upper_wide': mask=y>=.4
    elif mode=='lower_narrow': mask=(y<=-.12)&(y>-.4)
    else: mask=y<=-.4
    return reached&mask

for regime in ['centered','gaussian']:
    xmax,ymax=1.2,.8
    for key,entry in report['conditions'].items():
        if key.startswith(regime+'__'):
            with np.load(root/entry['artifact']) as saved:
                xmax=max(xmax,1.1*float(np.nanmax(abs(saved['positions'][...,0]))))
                ymax=max(ymax,1.1*float(np.nanmax(abs(saved['positions'][...,1]))))
    fig,axes=plt.subplots(2,4,figsize=(16,9),dpi=100,sharex=True,sharey=True)
    panels=[];summary[regime]={}
    with np.load(root/report['conditions'][f'{regime}__base']['artifact']) as saved:
        base_positions=saved['positions'];base_lengths=saved['lengths']
    for r,priority in enumerate(['direction','width']):
        for c,mode in enumerate(modes):
            user=f'{mode}_{priority}'; ax=axes[r,c]
            learnt=np.load(root/report['conditions'][f'{regime}__{user}__learned_20']['artifact'])
            oracle=np.load(root/report['conditions'][f'{regime}__{user}__oracle_soft']['artifact'])
            lp,ll=learnt['positions'],learnt['lengths'];op,ol=oracle['positions'],oracle['lengths']
            hit=int(mask_mode(lp,ll,mode).sum());ohit=int(mask_mode(op,ol,mode).sum())
            n=len(ll);succ=int(learnt['success'].sum());osucc=int(oracle['success'].sum())
            summary[regime][user]=dict(count=n,learned_mode_hits=hit,oracle_mode_hits=ohit,learned_success=succ,oracle_success=osucc)
            ax.set_xlim(-xmax,xmax);ax.set_ylim(-ymax,ymax);ax.set_aspect('equal');ax.grid(alpha=.15)
            ax.axhline(0,color='gray',lw=.5)
            for y in [-.4,-.12,.12,.4]:ax.axhline(y,color='gray',lw=.5,ls=':')
            ax.scatter([-1,1],[0,0],s=[30,65],c=['black','green'],marker='o',zorder=8)
            ax.set_title(f'{mode.replace("_"," ").title()} | {priority} priority\nMode: learned {hit}/{n} · oracle {ohit}/{n}\nTask: learned {succ}/{n} · oracle {osucc}/{n}',fontsize=9)
            traces=[ax.plot([],[],color='#1f66b4',alpha=.19,lw=.8)[0] for _ in ll]
            bold=ax.plot([],[],color='#155bbb',lw=2.4,label='Learned (episode 0)')[0]
            truth=ax.plot([],[],color='#e88416',lw=1.8,ls='--',label='Oracle soft (episode 0)')[0]
            baseline=ax.plot([],[],color='#666666',lw=1.2,ls=':',label='Base (episode 0)')[0]
            point=ax.plot([],[],'o',color='#155bbb',ms=5)[0]
            fail=ax.plot([],[],'x',color='#bc2727',ms=7)[0]
            if c==0:ax.set_ylabel('y position')
            if r==1:ax.set_xlabel('x position')
            panels.append((traces,bold,truth,baseline,point,fail,lp,ll,op,ol,learnt['success']))
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,.03))
    title=fig.suptitle('',fontsize=17,y=.965)
    fig.text(.5,.075,'Thin blue: all actual episodes. Bold/dashed: fixed episode 0. Named modes are midpoint bins; preferences are continuous.',ha='center',fontsize=10)
    fig.subplots_adjust(top=.87,bottom=.15,hspace=.3,wspace=.14)
    frames=[]
    for frame in range(65):
        title.set_text(f'Grouped preference inference | {regime} starts | M=8, L=4, K=20 | step {frame}/64')
        for traces,bold,truth,baseline,point,fail,lp,ll,op,ol,success in panels:
            for i,line in enumerate(traces):
                upto=min(frame+1,int(ll[i]));line.set_data(lp[i,:upto,0],lp[i,:upto,1])
            upto=min(frame+1,int(ll[0]));bold.set_data(lp[0,:upto,0],lp[0,:upto,1])
            upto_o=min(frame+1,int(ol[0]));truth.set_data(op[0,:upto_o,0],op[0,:upto_o,1])
            upto_b=min(frame+1,int(base_lengths[0]));baseline.set_data(base_positions[0,:upto_b,0],base_positions[0,:upto_b,1])
            point.set_data([lp[0,upto-1,0]],[lp[0,upto-1,1]])
            if frame>=ll[0]-1 and not success[0]:fail.set_data([lp[0,int(ll[0])-1,0]],[lp[0,int(ll[0])-1,1]])
        fig.canvas.draw();rgba=np.asarray(fig.canvas.buffer_rgba());frames.append(Image.fromarray(rgba[:,:,:3].copy()))
        if frame in [0,32,64]:frames[-1].save(out/f'{regime}_frame_{frame:02d}.png')
    frames[0].save(out/f'{regime}_grouped_inference.gif',save_all=True,append_images=frames[1:],duration=110,loop=0,optimize=False)
    plt.close(fig);print(f'rendered {regime}',flush=True)
(out/'mode_summary.json').write_text(json.dumps(summary,indent=2))
(out/'README.md').write_text('GIFs render actual saved Gym paths from this pilot. Upper row: direction priority; lower row: width priority. Columns: upper narrow, upper wide, lower narrow, lower wide. Thin blue: all episodes. Bold blue, dashed orange and dotted gray: fixed episode 0 for learned-20, oracle-soft and raw base. Red X marks failed episode 0. Both learned and oracle use the same soft selection algorithm. A preferred continuous utility does not guarantee the named midpoint bin. See mode_summary.json for independent trajectory-derived counts.\n')
print(json.dumps(summary,indent=2))
