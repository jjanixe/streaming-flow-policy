"""Exploratory two-chunk oracle lookahead on previously failed Gaussian cases.

The first bank is the saved M128/std8 bank. Eight spatially spread valid first
chunks are expanded with the original second-decision M128/std8 draws. Select
by oracle mean utility minus task cost, then execute future sample zero (never
choose a favorable realized future). This is a failure-conditioned diagnostic,
not a held-out controller evaluation or a learned-preference performance claim.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from env.config import DEFAULT_CONFIG
from env.environment import PointReach2DPreferenceEnv
from env.grouped_preference import classify_modes, synthetic_grouped_users
from env.grouped_rollout import evaluate_conditional_continuations, conditional_scores
from env.run_grouped_preference import draw_latents
from env.run_stage_b import load_trained_sfps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    source = root/'mode-aligned-best-M128-std8'
    out = root/'mode-aligned-two-chunk-probe'
    out.mkdir(exist_ok=False)
    config = json.loads((source/'resolved_config.json').read_text())
    ckpt = Path(config['checkpoint_path']); before = hashlib.sha256(ckpt.read_bytes()).hexdigest()
    torch.set_num_threads(1)
    policy, stats, _ = load_trained_sfps(ckpt, config['stats_path'],
        expected_train_data_digest='3fb4d5e8f79027bf1c113c8a85d48b82e30e43e707b8c895be96b9ab49d82e78',
        device=args.device)
    policy.requires_grad_(False)
    result = dict(source=str(source), checkpoint_digest=before, beam_size=8,
        current_M=128, second_M=128, L=4, proposal_std=8., executed_future_index=0,
        limitation='Failure-conditioned exploratory oracle diagnostic; no held-out or learned-control claim.',
        cases=[])
    modes = ['upper_narrow','upper_wide','lower_narrow','lower_wide']
    for user, pref in synthetic_grouped_users().items():
        with np.load(source/f'gaussian__{user}__oracle_best.npz') as saved:
            a = {key: saved[key] for key in ['positions','success','lengths','current_valid',
                'current_commands','current_latents','continuation_positions','environment_seeds','rollout_seeds']}
        target = modes.index(user.rsplit('_',1)[0])
        failed = np.flatnonzero(~(a['success'] & (classify_modes(a['positions'][:,32,1]) == target)))
        for episode in failed:
            started = time.perf_counter()
            valid = np.flatnonzero(a['current_valid'][episode,0])
            order = valid[np.argsort(a['current_commands'][episode,0,valid,-1,1],kind='stable')]
            indices = np.unique(np.rint(np.linspace(0,len(order)-1,min(8,len(order)))).astype(int))
            beam = order[indices]
            prefixes = a['continuation_positions'][episode,0,beam,0,:9].copy()
            streams = draw_latents([int(a['rollout_seeds'][episode])]*len(beam),
                                  chunk=1,candidates=128,continuations=4)
            current = streams['current_latents']*np.float32(8.)
            batch = evaluate_conditional_continuations(policy,stats,prefixes,current,
                streams['future_latents'],environment_config=DEFAULT_CONFIG.environment,
                integration_steps_per_action=6,feature_kind='mode')
            scores = conditional_scores(batch,pref,beta=16.)
            scores = np.where(batch.current_valid,scores,-np.inf)
            assert np.isfinite(scores).any()
            b,j = np.unravel_index(np.argmax(scores),scores.shape)
            row = (b*128+j)*4  # Always use predeclared future sample zero.
            commands = batch.continuations.requested_actions[row]
            env = PointReach2DPreferenceEnv()
            try:
                observation,info = env.reset(seed=int(a['environment_seeds'][episode]),options={'center_init':False})
                positions = [observation[:2].copy()]
                for command in commands:
                    observation,_,done,truncated,info=env.step(command)
                    if not info['action_limit_failure'] and not info['numerical_failure']:
                        positions.append(observation[:2].copy())
                    if done or truncated:break
            finally:env.close()
            positions=np.asarray(positions,np.float32)
            np.testing.assert_array_equal(positions,batch.continuations.positions[row,:len(positions)])
            hit = bool(info['success'] and len(positions)==65 and classify_modes(positions[32,1])==target)
            name=f'{user}__episode{episode}'
            np.savez_compressed(out/(name+'.npz'),positions=positions,requested_actions=commands,
                first_latent=a['current_latents'][episode,0,beam[b]],second_latent=current[b,j],
                future_latents=streams['future_latents'][b,j,0],beam_indices=beam,
                scores=scores,expected_features=batch.expected_features,current_valid=batch.current_valid)
            entry=dict(user=user,episode=int(episode),target_mode=modes[target],
                first_candidate=int(beam[b]),second_candidate=int(j),score=float(scores[b,j]),
                expected_features=batch.expected_features[b,j].tolist(),task_success=bool(info['success']),
                successful_target_mode=hit,midpoint_y=float(positions[32,1]) if len(positions)>32 else None,
                elapsed_seconds=time.perf_counter()-started,artifact=name+'.npz')
            result['cases'].append(entry)
            (out/'diagnostics.json').write_text(json.dumps(result,indent=2)+'\n')
            print(name,'task+mode',hit,'midpoint_y',entry['midpoint_y'],flush=True)
    assert before==hashlib.sha256(ckpt.read_bytes()).hexdigest()
    result['repaired_cases']=sum(case['successful_target_mode'] for case in result['cases'])
    result['total_cases']=len(result['cases'])
    (out/'diagnostics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result['repaired_cases'],'/',result['total_cases'],'previous failures repaired',flush=True)


if __name__=='__main__':main()
