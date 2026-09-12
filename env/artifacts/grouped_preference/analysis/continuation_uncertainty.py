"""Fixed-prefix Monte Carlo diagnostic against a shared L=64 sample bank."""
import argparse,json
from pathlib import Path
import numpy as np,torch
from env.config import DEFAULT_CONFIG
from env.run_preference import _seed,_initial_positions
from env.run_stage_b import load_trained_sfps
from env.grouped_preference import grouped_utility,synthetic_grouped_users
from env.grouped_rollout import evaluate_conditional_continuations

p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');a=p.parse_args()
torch.set_num_threads(1);root=Path(__file__).parent
policy,stats,_=load_trained_sfps('env/artifacts/stage_b/b2-seed0-optimized/sfps_best.pt','env/artifacts/stage_b/b2-seed0-optimized/pusht_stats.npz',expected_train_data_digest='3fb4d5e8f79027bf1c113c8a85d48b82e30e43e707b8c895be96b9ab49d82e78',device=a.device)
initial=np.concatenate([_initial_positions([_seed(0,5,i)],i==0) for i in range(3)])
current=np.random.default_rng(819300).standard_normal((3,8,2)).astype(np.float32)
future=np.random.default_rng(819301).standard_normal((3,8,64,7,2)).astype(np.float32)
b=evaluate_conditional_continuations(policy,stats,initial[:,None],current,future,environment_config=DEFAULT_CONFIG.environment)
np.savez_compressed(root/'continuation_uncertainty.npz',initial=initial,current_latents=current,future_latents=future,features=b.continuation_features,current_valid=b.current_valid,future_failure_fractions=b.future_failure_fractions)
report={}
for name,pref in synthetic_grouped_users().items():
 u=grouped_utility(b.continuation_features,pref)
 entries=[]
 for i,context in enumerate(['centered','gaussian_0','gaussian_1']):
  valid=b.current_valid[i];v=u[i,valid];ref=v.mean(1)
  if not len(v):entries.append(dict(context=context,valid_candidates=0));continue
  rows={}
  for l in [1,4,8,16]:
   estimates=v.reshape(len(v),64//l,l).mean(-1)
   rows[str(l)]=dict(rmse_to_64_sample_mean=float(np.sqrt(np.mean((estimates-ref[:,None])**2))),
       estimated_mean_standard_error=float(np.mean(np.std(v,axis=1,ddof=1)/np.sqrt(l))))
  entries.append(dict(context=context,valid_candidates=int(valid.sum()),between_candidate_Q_std=float(np.std(ref)),mean_future_failure_fraction=float(b.future_failure_fractions[i,valid].mean()),by_L=rows))
 report[name]=entries
(root/'continuation_uncertainty.json').write_text(json.dumps(dict(reference_L=64,independent_current_candidates=8,initial_states=3,warning='L64 is a finite sample reference; subgroups share that bank. Not ground-truth Q or a closed-loop performance comparison.',profiles=report),indent=2))
print(json.dumps(report['upper_narrow_direction'],indent=2))
