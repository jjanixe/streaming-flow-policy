"""Frozen-policy first-decision coverage; saved futures are not real executions."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from env.config import DEFAULT_CONFIG
from env.environment import PointReach2DPreferenceEnv
from env.grouped_rollout import evaluate_conditional_continuations
from env.run_grouped_preference import draw_latents
from env.run_preference import _seed
from env.run_stage_b import load_trained_sfps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--candidates', nargs='+', type=int, default=[8, 32, 128])
    parser.add_argument('--proposal-std', type=float, default=1.)
    parser.add_argument('--episodes', type=int, default=16)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a new output directory')
    args.output.mkdir(parents=True)
    torch.set_num_threads(1)
    checkpoint = Path('env/artifacts/stage_b/b2-seed0-optimized/sfps_best.pt')
    stats_path = checkpoint.with_name('pusht_stats.npz')
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    policy, stats, _ = load_trained_sfps(checkpoint, stats_path,
        expected_train_data_digest='3fb4d5e8f79027bf1c113c8a85d48b82e30e43e707b8c895be96b9ab49d82e78',
        device=args.device)
    policy.requires_grad_(False)
    environment_seeds = [_seed(0, 9, i) for i in range(args.episodes)]
    rollout_seeds = [_seed(0, 10, i) for i in range(args.episodes)]
    report = dict(checkpoint_digest=digest, proposal_std=args.proposal_std,
                  episodes=args.episodes, L=4, results={})
    for regime in ('centered', 'gaussian'):
        initial = []
        for seed in environment_seeds:
            instance = PointReach2DPreferenceEnv()
            obs, _ = instance.reset(seed=seed, options={'center_init': regime == 'centered'})
            initial.append(obs[:2].copy())
            instance.close()
        prefix = np.asarray(initial, np.float32)[:, None]
        for m in args.candidates:
            streams = draw_latents(rollout_seeds, chunk=0, candidates=m, continuations=4)
            current = streams['current_latents'] * np.float32(args.proposal_std)
            if args.device.startswith('cuda'):
                torch.cuda.synchronize(args.device)
            started = time.perf_counter()
            batch = evaluate_conditional_continuations(policy, stats, prefix, current,
                streams['future_latents'], environment_config=DEFAULT_CONFIG.environment)
            if args.device.startswith('cuda'):
                torch.cuda.synchronize(args.device)
            elapsed = time.perf_counter() - started
            paths = batch.continuations.positions.reshape(args.episodes, m, 4, 65, 2)
            ok = batch.continuations.success.reshape(args.episodes, m, 4)
            y = paths[..., 32, 1].astype(np.float64)
            mode = np.select([(y >= .12) & (y < .4), y >= .4,
                              (y <= -.12) & (y > -.4), y <= -.4], range(4), default=4)
            names = ['upper_narrow', 'upper_wide', 'lower_narrow', 'lower_wide', 'other']
            counts = {name: int(((mode == k) & ok).sum()) for k, name in enumerate(names)}
            entry = dict(seconds=elapsed, successful_futures=int(ok.sum()),
                current_valid=int(batch.current_valid.sum()), successful_mode_counts=counts,
                context_coverage={name: int(((mode == k) & ok).any(axis=(1, 2)).sum())
                                  for k, name in enumerate(names)})
            key = f'{regime}__M{m}'
            np.savez_compressed(args.output / (key+'.npz'), initial=prefix,
                current_latents=current, future_latents=streams['future_latents'],
                current_valid=batch.current_valid, positions=paths, success=ok,
                lengths=batch.continuations.lengths.reshape(args.episodes,m,4))
            report['results'][key] = entry
            (args.output/'diagnostics.json').write_text(json.dumps(report, indent=2)+'\n')
            print(key, json.dumps(entry), flush=True)
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == digest


if __name__ == '__main__':
    main()
