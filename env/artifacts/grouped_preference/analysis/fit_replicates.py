"""Independent synthetic-feedback repetitions; no policy rollouts or tuning."""
import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
from env.grouped_preference import (build_grouped_queries, grouped_features,
    grouped_probabilities, fit_grouped_preference, grouped_metrics, synthetic_grouped_users)

out=Path(__file__).parent
rows=[]
users=synthetic_grouped_users()
held=build_grouped_queries(seed=190001,count=512)
hf=grouped_features(held['trajectories']); hd=hf[:,0]-hf[:,1]
for seed in range(20):
    query=build_grouped_queries(seed=180000+seed,count=40)
    f=grouped_features(query['trajectories']); delta=f[:,0]-f[:,1]
    for ui,(name,truth) in enumerate(users.items()):
        probs=grouped_probabilities(delta,query['scopes'],truth)
        labels=np.where(np.random.default_rng(np.random.SeedSequence([170000,seed,ui])).random(40)<probs,1,-1)
        truth_p=grouped_probabilities(hd,held['scopes'],truth)
        for k in [5,10,20,40]:
            fit=fit_grouped_preference(delta[:k],labels[:k],query['scopes'][:k])
            prediction=grouped_probabilities(hd,held['scopes'],fit.preference)
            overall=held['scopes']=='overall'
            hp=np.clip(prediction,1e-15,1-1e-15)
            rows.append(dict(seed=seed,user=name,k=k,alpha_error=float(abs(fit.preference.alpha[0]-truth.alpha[0])),
                weights_mae=float(abs(fit.preference.weights-truth.weights).mean()),
                priority_correct=bool((fit.preference.alpha[0]>.5)==(truth.alpha[0]>.5)),
                within_correct=bool(np.array_equal(fit.preference.weights.argmax(1),truth.weights.argmax(1))),
                overall_ordering_accuracy=float(((prediction[overall]>.5)==(truth_p[overall]>.5)).mean()),
                expected_overall_cross_entropy=float((-(truth_p*np.log(hp)+(1-truth_p)*np.log1p(-hp)))[overall].mean()),
                converged=bool(fit.converged),rank=int(fit.jacobian_rank)))
    print(f'feedback replicate {seed+1}/20',flush=True)
summary={}
for k in [5,10,20,40]:
    entries=[r for r in rows if r['k']==k]
    summary[str(k)]={field:float(np.mean([r[field] for r in entries])) for field in
        ['alpha_error','weights_mae','priority_correct','within_correct','overall_ordering_accuracy','expected_overall_cross_entropy','converged','rank']}
    summary[str(k)]['fits']=len(entries)
(out/'fit_replicates.json').write_text(json.dumps(dict(config=dict(feedback_seeds=20,users=8,rho=16,l2=.1,heldout_pairs=512),summary=summary,rows=rows),indent=2))
print(json.dumps(summary,indent=2))
