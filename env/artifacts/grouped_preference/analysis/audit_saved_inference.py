"""Independently replay saved commands and recompute utility/mode outcomes."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from env.environment import PointReach2DPreferenceEnv
from env.grouped_preference import grouped_features,grouped_utility,synthetic_grouped_users

parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);args=parser.parse_args()
root=args.directory;record=json.loads((root/'diagnostics.json').read_text())
users=synthetic_grouped_users();audit={};episodes=0;decisions_checked=0
for key,metrics in record['conditions'].items():
    a=np.load(root/metrics['artifact'],allow_pickle=False)
    count=len(a['lengths']);regime=key.split('__')[0]
    for i in range(count):
        env=PointReach2DPreferenceEnv()
        try:
            obs,info=env.reset(seed=int(a['environment_seeds'][i]),options={'center_init':regime=='centered'})
            states=[obs[:2].copy()]
            for command in a['requested_actions'][i]:
                obs,_,done,truncated,info=env.step(command)
                if not info['numerical_failure'] and not info['action_limit_failure']:states.append(obs[:2].copy())
                if done or truncated:break
            states=np.array(states,dtype=np.float32)
            assert len(states)==a['lengths'][i],(key,i,'length')
            np.testing.assert_array_equal(states,a['positions'][i,:len(states)],err_msg=f'{key} episode {i}')
            for flag in ['success','numerical_failure','action_limit_failure']:assert bool(info[flag])==bool(a[flag][i]),(key,i,flag)
            assert np.isclose(info['goal_error'],a['goal_errors'][i]),(key,i,'goal')
        finally:env.close()
        episodes+=1
    for i,chunk in zip(*np.where(a['decision_mask'])):
        q=chunk*8
        f=a['continuation_features'][i,chunk]
        np.testing.assert_allclose(f.mean(axis=1),a['expected_features'][i,chunk],rtol=0,atol=1e-14)
        paths=a['continuation_positions'][i,chunk]
        np.testing.assert_array_equal(paths[:,:,:q+1],np.broadcast_to(a['positions'][i,:q+1],paths[:,:,:q+1].shape))
        np.testing.assert_array_equal(paths[:,:,:q+9],np.broadcast_to(paths[:,0:1,:q+9],paths[:,:,:q+9].shape))
        np.testing.assert_array_equal(a['observed_anchors'][i,chunk],np.broadcast_to(a['positions'][i,q],a['observed_anchors'][i,chunk].shape))
        lens=a['continuation_lengths'][i,chunk]
        midx,lidx=np.indices(lens.shape)
        last=paths[midx,lidx,lens-1].astype(float)
        cost=(np.square(last-np.array([1.,0.])).sum(-1)/.01+25*(~a['continuation_success'][i,chunk])).mean(1)
        np.testing.assert_allclose(cost,a['goal_costs'][i,chunk],rtol=1e-12,atol=1e-10)
        pref=metrics['preference']
        utility=0. if pref is None else np.sum(np.sum(f.mean(1)*np.asarray(pref['weights']),axis=-1)*np.asarray(pref['alpha']),axis=-1)
        expected_score=record['config']['beta']*utility-cost
        np.testing.assert_allclose(expected_score,a['candidate_scores'][i,chunk],rtol=1e-12,atol=1e-10)
        valid=np.flatnonzero(a['current_valid'][i,chunk]);selected=int(a['selected_indices'][i,chunk])
        if metrics['method']=='base':assert selected==0
        elif not len(valid):assert selected==-1 and a['fallback'][i,chunk]
        else:
            scores=a['candidate_scores'][i,chunk,valid];probs=np.exp(scores-scores.max());probs/=probs.sum()
            chosen=int(np.random.default_rng(a['selection_seeds'][i,chunk]).choice(valid,p=probs))
            assert chosen==selected,(key,i,chunk,'selection')
        decisions_checked+=1
    complete=(a['lengths']==65)&~a['numerical_failure']&~a['action_limit_failure']
    features=np.zeros((count,2,2));features[complete]=grouped_features(a['positions'][complete])
    y=a['positions'][:,32,1];reached=a['lengths']>32
    modes={'upper_narrow':int((reached&(y>=.12)&(y<.4)).sum()),'upper_wide':int((reached&(y>=.4)).sum()),
      'lower_narrow':int((reached&(y<=-.12)&(y>-.4)).sum()),'lower_wide':int((reached&(y<=-.4)).sum()),'other':int((reached&(abs(y)<.12)).sum())}
    audit[key]=dict(count=count,success_count=int(a['success'].sum()),midpoint_counts=modes,
        utility_by_user={u:dict(all_outcome_mean=float(grouped_utility(features,p).mean()),
           successful_mean=float(grouped_utility(features,p)[a['success']].mean()) if a['success'].any() else None) for u,p in users.items()})
print(f'Independently replayed {episodes} episodes and audited {decisions_checked} decisions; states, terminal flags, shared prefixes, mean features, task costs, scores and sampled choices agree.')
(root/'independent_audit.json').write_text(json.dumps(dict(replayed_episodes=episodes,decisions_checked=decisions_checked,conditions=audit),indent=2))
