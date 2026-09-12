"""Scientific GIFs from saved actual Gym inference trajectories."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);parser.add_argument('--before-directory',type=Path)
args=parser.parse_args();root=args.directory
report=json.loads((root/'diagnostics.json').read_text()); out=root/'animations';out.mkdir(exist_ok=True)
config=report['config']; selector=config.get('selection_method','soft'); M=config['candidates']; scale=config.get('proposal_std',1.)
before_root=args.before_directory
before_report=json.loads((before_root/'diagnostics.json').read_text()) if before_root else None
modes=['upper_narrow','upper_wide','lower_narrow','lower_wide']
summary={}

def mask_mode(paths,lengths,mode):
    y=paths[:,32,1].astype(np.float64); reached=lengths>32
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
            oracle=np.load(root/report['conditions'][f'{regime}__{user}__oracle_{selector}']['artifact'])
            lp,ll=learnt['positions'],learnt['lengths'];op,ol=oracle['positions'],oracle['lengths']
            hit=int((mask_mode(lp,ll,mode)&learnt['success']).sum());ohit=int((mask_mode(op,ol,mode)&oracle['success']).sum())
            if before_root:
                with np.load(before_root/before_report['conditions'][f'{regime}__{user}__learned_20']['artifact']) as before:
                    bp,bl=before['positions'],before['lengths'];bhit=int((mask_mode(bp,bl,mode)&before['success']).sum())
                np.testing.assert_array_equal(bp[:,0],lp[:,0])
            else: bp,bl,bhit=base_positions,base_lengths,None
            n=len(ll);succ=int(learnt['success'].sum());osucc=int(oracle['success'].sum())
            summary[regime][user]=dict(count=n,learned_successful_mode_hits=hit,oracle_successful_mode_hits=ohit,before_successful_mode_hits=bhit,learned_success=succ,oracle_success=osucc)
            ax.set_xlim(-xmax,xmax);ax.set_ylim(-ymax,ymax);ax.set_aspect('equal');ax.grid(alpha=.15)
            ax.axhline(0,color='gray',lw=.5)
            for y in [-.4,-.12,.12,.4]:ax.axhline(y,color='gray',lw=.5,ls=':')
            ax.scatter([-1,1],[0,0],s=[30,65],c=['black','green'],marker='o',zorder=8)
            ax.set_title(f'{mode.replace("_"," ").title()} | {priority} priority\nSuccess + mode: learned {hit}/{n} · oracle {ohit}/{n}\nBefore: {bhit}/{n} · Task after: {succ}/{n}',fontsize=9)
            traces=[ax.plot([],[],color='#1f66b4',alpha=.19,lw=.8)[0] for _ in ll]
            bold=ax.plot([],[],color='#155bbb',lw=2.4,label='Learned (episode 0)')[0]
            truth=ax.plot([],[],color='#e88416',lw=1.8,ls='--',label=f'Oracle {selector} (episode 0)')[0]
            baseline=ax.plot([],[],color='#666666',lw=1.2,ls=':',label='Before (episode 0)' if before_root else 'Base (episode 0)')[0]
            point=ax.plot([],[],'o',color='#155bbb',ms=5)[0]
            fail=ax.plot([],[],'x',color='#bc2727',ms=7)[0]
            if c==0:ax.set_ylabel('y position')
            if r==1:ax.set_xlabel('x position')
            panels.append((traces,bold,truth,baseline,point,fail,lp,ll,op,ol,learnt['success'],bp,bl))
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,.03))
    title=fig.suptitle('',fontsize=17,y=.965)
    fig.text(.5,.075,'Thin blue: all actual episodes. Bold/dashed: fixed episode 0. Success + mode requires goal success and the target midpoint bin. Same episode 0 for every target.',ha='center',fontsize=10)
    fig.subplots_adjust(top=.87,bottom=.15,hspace=.3,wspace=.14)
    frames=[]
    for frame in range(65):
        title.set_text(f'Mode-aligned inference | {regime} | M={M}, L=4, K=20, latent std={scale:g} | step {frame}/64')
        for traces,bold,truth,baseline,point,fail,lp,ll,op,ol,success,bp,bl in panels:
            for i,line in enumerate(traces):
                upto=min(frame+1,int(ll[i]));line.set_data(lp[i,:upto,0],lp[i,:upto,1])
            upto=min(frame+1,int(ll[0]));bold.set_data(lp[0,:upto,0],lp[0,:upto,1])
            upto_o=min(frame+1,int(ol[0]));truth.set_data(op[0,:upto_o,0],op[0,:upto_o,1])
            upto_b=min(frame+1,int(bl[0]));baseline.set_data(bp[0,:upto_b,0],bp[0,:upto_b,1])
            point.set_data([lp[0,upto-1,0]],[lp[0,upto-1,1]])
            if frame>=ll[0]-1 and not success[0]:fail.set_data([lp[0,int(ll[0])-1,0]],[lp[0,int(ll[0])-1,1]])
        fig.canvas.draw();rgba=np.asarray(fig.canvas.buffer_rgba());frames.append(Image.fromarray(rgba[:,:,:3].copy()))
        if frame in [0,32,64]:frames[-1].save(out/f'{regime}_frame_{frame:02d}.png')
    frames[0].save(out/f'{regime}_mode_aligned_inference.gif',save_all=True,append_images=frames[1:],duration=110,loop=0,optimize=False)
    plt.close(fig);print(f'rendered {regime}',flush=True)
(out/'mode_summary.json').write_text(json.dumps(summary,indent=2))
(out/'README.md').write_text('Actual saved Gym paths. Rows: direction/width priority; columns: upper-narrow, upper-wide, lower-narrow, lower-wide. Thin blue: all after episodes; bold blue: learned20 episode0; orange: oracle episode0; gray: before learned20 episode0 if supplied, otherwise raw base. Counts require task success AND target midpoint mode. New features and synthetic labels are mode-aligned; before uses the old continuous objective. This visual comparison concerns task/mode outcomes, not a like-for-like utility value. No trajectory morphing or cherry-picked episodes. See mode_summary.json.\n')
print(json.dumps(summary,indent=2))
