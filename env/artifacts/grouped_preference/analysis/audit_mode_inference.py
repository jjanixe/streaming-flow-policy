"""Independent audit of saved actual executions and counterfactual decisions."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from env.environment import PointReach2DPreferenceEnv


def features_of(paths, kind):
    paths = paths.astype(np.float64)
    if kind == 'continuous':
        r = np.tanh(paths[..., 1:, 1] / .35)
        d, e = r.mean(-1), (r*r).mean(-1)
        return np.stack([np.stack([(1+d)/2, (1-d)/2], -1),
                         np.stack([e, 1-e], -1)], -2)
    assert kind == 'mode'
    y = paths[..., 32, 1]
    upper, lower = y >= .12, y <= -.12
    wide, narrow = abs(y) >= .4, (abs(y) >= .12) & (abs(y) < .4)
    return np.stack([np.stack([upper, lower], -1),
                     np.stack([wide, narrow], -1)], -2).astype(np.float64)


def audit(root):
    record = json.loads((root/'diagnostics.json').read_text())
    config = record['config']
    kind = config.get('feature_kind', 'continuous')
    beta = config['beta']
    results = {}
    episodes = decisions = 0
    for key, metrics in record['conditions'].items():
        with np.load(root/metrics['artifact'], allow_pickle=False) as saved:
            a = {name: saved[name] for name in saved.files}
        regime = key.split('__')[0]
        for i in range(len(a['lengths'])):
            env = PointReach2DPreferenceEnv()
            try:
                obs, info = env.reset(seed=int(a['environment_seeds'][i]),
                                     options={'center_init': regime == 'centered'})
                states = [obs[:2].copy()]
                for action in a['requested_actions'][i]:
                    obs, _, done, truncated, info = env.step(action)
                    if not info['numerical_failure'] and not info['action_limit_failure']:
                        states.append(obs[:2].copy())
                    if done or truncated:
                        break
                np.testing.assert_array_equal(states, a['positions'][i, :a['lengths'][i]])
                for flag in ('success', 'numerical_failure', 'action_limit_failure'):
                    assert bool(info[flag]) == bool(a[flag][i]), (key, i, flag)
                np.testing.assert_allclose(info['goal_error'], a['goal_errors'][i], rtol=1e-6)
            finally:
                env.close()
            episodes += 1
        for i, chunk in zip(*np.where(a['decision_mask'])):
            q = chunk*8
            paths, lens = a['continuation_positions'][i, chunk], a['continuation_lengths'][i, chunk]
            complete = ((lens == 65) & ~a['continuation_numerical_failure'][i, chunk]
                        & ~a['continuation_action_limit_failure'][i, chunk])
            f = np.zeros((*lens.shape, 2, 2), np.float64)
            f[complete] = features_of(paths[complete], kind)
            np.testing.assert_allclose(f, a['continuation_features'][i, chunk], rtol=0, atol=1e-14)
            np.testing.assert_allclose(f.mean(1), a['expected_features'][i, chunk], rtol=0, atol=1e-14)
            np.testing.assert_array_equal(paths[:, :, :q+1],
                np.broadcast_to(a['positions'][i, :q+1], paths[:, :, :q+1].shape))
            np.testing.assert_array_equal(paths[:, :, :q+9],
                np.broadcast_to(paths[:, :1, :q+9], paths[:, :, :q+9].shape))
            np.testing.assert_array_equal(a['observed_anchors'][i, chunk],
                np.broadcast_to(a['positions'][i, q], a['observed_anchors'][i, chunk].shape))
            mi, li = np.indices(lens.shape)
            final = paths[mi, li, lens-1].astype(np.float64)
            costs = (((final-[1, 0])**2).sum(-1)/.01
                     + 25*(~a['continuation_success'][i, chunk])).mean(1)
            np.testing.assert_allclose(costs, a['goal_costs'][i, chunk], rtol=1e-12, atol=1e-10)
            pref = metrics['preference']
            utility = 0. if pref is None else (f.mean(1)*np.asarray(pref['weights'])
                          *np.asarray(pref['alpha'])[:, None]).sum(axis=(-2, -1))
            scores = beta*utility-costs
            np.testing.assert_allclose(scores, a['candidate_scores'][i, chunk], rtol=1e-12, atol=1e-10)
            valid = np.flatnonzero(a['current_valid'][i, chunk])
            selected = int(a['selected_indices'][i, chunk])
            method = metrics['method']
            if method == 'base':
                assert selected == 0
            elif not len(valid):
                assert selected == -1 and a['fallback'][i, chunk]
            elif method == 'best':
                assert selected == valid[np.argmax(a['candidate_scores'][i, chunk, valid])], (key, i, chunk)
            else:
                assert method == 'soft'
                # Use stored scores for exact categorical RNG; numeric recomputation
                # above is checked separately with explicit tolerance.
                logits = a['candidate_scores'][i, chunk, valid]
                p = np.exp(logits-logits.max()); p /= p.sum()
                expected = np.random.default_rng(a['selection_seeds'][i, chunk]).choice(valid, p=p)
                assert selected == expected, (key, i, chunk)
            current_seed = a['current_seeds'][i, chunk]
            raw_current = np.random.default_rng(current_seed).standard_normal(
                a['current_latents'][i, chunk].shape).astype(np.float32)
            scale = 1. if method == 'base' else config.get('proposal_std', 1.)
            np.testing.assert_array_equal(raw_current*np.float32(scale), a['current_latents'][i, chunk])
            decisions += 1
        complete = (a['lengths'] == 65) & ~a['numerical_failure'] & ~a['action_limit_failure']
        good = complete & a['success']
        f = features_of(a['positions'][good], 'mode')
        counts = {name: int(((f[:, 0, d] == 1) & (f[:, 1, w] == 1)).sum())
                  for name, d, w in [('upper_narrow',0,1),('upper_wide',0,0),
                                     ('lower_narrow',1,1),('lower_wide',1,0)]}
        counts['other'] = int((~f.any(axis=(-2,-1))).sum())
        assert counts == metrics['successful_midpoint_counts'], (key, counts)
        results[key] = dict(episodes=len(a['lengths']), task_success=int(good.sum()),
                            successful_midpoint_counts=counts)
        print(key, 'audited', flush=True)
    for field, path_field in [('checkpoint_digest','checkpoint_path'), ('stats_digest','stats_path')]:
        actual = hashlib.sha256(Path(config[path_field]).read_bytes()).hexdigest()
        assert record[field] == actual, field
    result = dict(replayed_episodes=episodes, decisions_checked=decisions, feature_kind=kind,
                  all_checks_passed=True, conditions=results)
    (root/'independent_audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(f'PASS: replayed {episodes} actual episodes; audited {decisions} decisions.', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    audit(p.parse_args().directory)
