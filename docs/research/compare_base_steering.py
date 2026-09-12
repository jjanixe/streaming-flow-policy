"""Compare actual raw-base and learned20 paths from the published compact NPZ.

No new policy execution, fitting, or counterfactual rollout is performed.
Run from any directory; output defaults to this study's artifact directory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
RUN = 'mode-aligned-best-M128-std8'
MODES = ['upper_narrow', 'upper_wide', 'lower_narrow', 'lower_wide', 'other']
SHORT = ['UN', 'UW', 'LN', 'LW', 'Other']
COLORS = ['#159c95', '#336bcc', '#dc9423', '#b64483', '#888888']
PRIORITIES = ['direction', 'width']
REGIMES = ['centered', 'gaussian']
DEFAULT_OUT = ROOT/'docs/research/artifacts/2026-09-12-base-vs-steering'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mode_ids(y):
    """Independent implementation of the report's float64 midpoint bins."""
    y = np.asarray(y, np.float64)
    assert np.isfinite(y).all()
    return np.select([(y >= .12) & (y < .4), y >= .4,
                      (y > -.4) & (y <= -.12), y <= -.4], range(4), default=4)


def summarize(record):
    lengths = record['lengths']
    reached = lengths > 32
    ids = np.full(len(lengths), -1, dtype=int)
    ids[reached] = mode_ids(record['positions'][reached, 32, 1])
    complete = ((lengths == 65) & ~record['numerical_failure'] &
                ~record['action_limit_failure'])
    task = complete & record['success']
    return dict(count=len(lengths), task_success=int(task.sum()),
                goal_failed_episode_indices=np.flatnonzero(~task).tolist(),
                mode_counts={m: int((ids == k).sum()) for k, m in enumerate(MODES)},
                successful_mode_counts={m: int(((ids == k) & task).sum())
                                        for k, m in enumerate(MODES)},
                mode_ids=ids.tolist(), task_success_mask=task.tolist())


def load_records():
    published = ROOT/'docs/research/artifacts/2026-09-12-preference-steering'
    source = published/'actual_rollouts.npz'
    manifest = json.loads((published/'artifact_manifest.json').read_text())
    assert digest(source) == manifest['compact_actual_rollouts']['sha256']
    directory = ROOT/'env/artifacts/grouped_preference'/RUN
    diagnostics = json.loads((directory/'diagnostics.json').read_text())
    checkpoint = ROOT/'env/artifacts/stage_b/b2-seed0-optimized/sfps_best.pt'
    stats = checkpoint.with_name('pusht_stats.npz')
    assert digest(checkpoint) == diagnostics['checkpoint_digest']
    assert digest(stats) == diagnostics['stats_digest']
    records, summaries = {}, {}
    fields = ['positions', 'requested_actions', 'lengths', 'success',
              'numerical_failure', 'action_limit_failure', 'environment_seeds',
              'rollout_seeds', 'selected_current_latents', 'fallback']
    with np.load(source, allow_pickle=False) as z:
        indices = np.flatnonzero(z['experiment_names'] == RUN)
        assert len(indices) == 36
        for i in indices:
            name = str(z['condition_names'][i])
            record = {field: z[field][i] for field in fields}
            summary = summarize(record)
            saved = diagnostics['conditions'][name]
            assert summary['task_success'] == saved['success_count']
            assert summary['mode_counts'] == saved['midpoint_counts']
            assert summary['successful_mode_counts'] == saved['successful_midpoint_counts']
            assert not record['fallback'].any()
            records[name], summaries[name] = record, summary
    for regime in REGIMES:
        base = records[f'{regime}__base']
        for name, record in records.items():
            if name.startswith(regime+'__'):
                for field in ['environment_seeds', 'rollout_seeds']:
                    np.testing.assert_array_equal(base[field], record[field])
                np.testing.assert_array_equal(base['positions'][:, 0], record['positions'][:, 0])
    provenance = dict(compact_path=str(source.relative_to(ROOT)), compact_sha256=digest(source),
                      checkpoint_sha256=digest(checkpoint), stats_sha256=digest(stats),
                      source_run=RUN, source_conditions_checked=36,
                      plotted_conditions=20, plotted_episodes=320,
                      new_policy_rollouts=0, paired_initial_states_and_seeds_verified=True,
                      raw_base='No preference; one unit-normal current latent per decision.',
                      task_only='M128, current std8, L4, best; task cost only.',
                      steering='M128, current std8, L4, best, beta16; learned20 preference.',
                      note='Same base bank is reused to evaluate each requested target; rows are not independent samples.')
    return records, summaries, provenance


def comparison_rows(summaries):
    rows = []
    for regime in REGIMES:
        base, task = summaries[f'{regime}__base'], summaries[f'{regime}__task_only']
        for k, mode in enumerate(MODES[:4]):
            base_hit = np.array(base['task_success_mask']) & (np.array(base['mode_ids']) == k)
            row = dict(regime=regime, target=mode, episodes=base['count'],
                       base_task_success=base['task_success'], base_mode=base['mode_counts'][mode],
                       base_joint=base['successful_mode_counts'][mode],
                       task_only_joint=task['successful_mode_counts'][mode])
            for priority in PRIORITIES:
                steered = summaries[f'{regime}__{mode}_{priority}__learned_20']
                hit = np.array(steered['task_success_mask']) & (np.array(steered['mode_ids']) == k)
                row.update({f'{priority}_mode': steered['mode_counts'][mode],
                            f'{priority}_task': steered['task_success'],
                            f'{priority}_joint': int(hit.sum()),
                            f'{priority}_rescued': int((~base_hit & hit).sum()),
                            f'{priority}_lost': int((base_hit & ~hit).sum()),
                            f'{priority}_retained': int((base_hit & hit).sum())})
            rows.append(row)
    return rows


def plots(out, summaries, rows):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), dpi=160, sharey=True)
    labels = [('base_joint', 'Raw base', '#808080'),
              ('task_only_joint', 'Task-only: same search budget', '#e5a23a'),
              ('direction_joint', 'Steering: direction priority', '#2463b5'),
              ('width_joint', 'Steering: width priority', '#9554ae')]
    for ax, regime in zip(axes, REGIMES):
        selected = [r for r in rows if r['regime'] == regime]
        x = np.arange(4)
        for i, (key, label, color) in enumerate(labels):
            values = np.array([r[key] for r in selected])
            bars = ax.bar(x+(i-1.5)*.2, values/16, width=.18, color=color, label=label)
            for bar, n in zip(bars, values):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+.02, str(n),
                        ha='center', va='bottom', fontsize=8)
        ax.set_xticks(x, ['Upper\nnarrow', 'Upper\nwide', 'Lower\nnarrow', 'Lower\nwide'])
        ax.set_ylim(0, 1.14); ax.set_yticks([0, .25, .5, .75, 1.], ['0%', '25%', '50%', '75%', '100%'])
        ax.set_title(regime.title()+' starts'); ax.grid(axis='y', alpha=.15)
    axes[0].set_ylabel('Task success AND requested mode')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False, fontsize=9)
    fig.suptitle('Actual inference: raw base vs preference steering', fontsize=15)
    fig.text(.5, .10, 'Labels are successful counts / 16. Same seeds and initial states. Base has no requested-mode input.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .16, 1, .93))
    fig.savefig(out/'target_mode_success.png'); fig.savefig(out/'target_mode_success.svg'); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 8), dpi=150)
    row_labels = ['Raw base', 'Task-only'] + [f'{SHORT[k]} / {priority}' for k in range(4) for priority in PRIORITIES]
    for ax, regime in zip(axes, REGIMES):
        names = [f'{regime}__base', f'{regime}__task_only'] + [f'{regime}__{m}_{p}__learned_20' for m in MODES[:4] for p in PRIORITIES]
        counts = np.array([[summaries[n]['successful_mode_counts'][m] for m in MODES] +
                           [16-summaries[n]['task_success']] for n in names])
        assert np.all(counts.sum(axis=1) == 16)
        ax.imshow(counts, cmap='Blues', vmin=0, vmax=16, aspect='auto')
        for (r, c), value in np.ndenumerate(counts):
            ax.text(c, r, str(value), ha='center', va='center', color='white' if value >= 10 else 'black')
        for r in range(2, 10):
            ax.add_patch(Rectangle(((r-2)//2-.48, r-.48), .96, .96, fill=False, edgecolor='#30a35b', lw=2))
        ax.set_xticks(range(6), SHORT+['Task\nfailure'])
        ax.set_yticks(range(10), row_labels); ax.set_title(regime.title()+' starts')
    fig.suptitle('Where each policy actually went | 16 episodes per row', fontsize=15)
    fig.text(.5, .035, 'Mode columns require task success; task failures form a separate column. Green box: requested target. Each row sums to 16.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .065, 1, .94))
    fig.savefig(out/'mode_distribution.png'); fig.savefig(out/'mode_distribution.svg'); plt.close(fig)


def trajectory_gif(out, regime, records, summaries):
    fig, axes = plt.subplots(2, 5, figsize=(18, 8.8), dpi=100, sharex=True, sharey=True)
    names = [f'{regime}__base'] + [f'{regime}__{m}_{p}__learned_20' for p in PRIORITIES for m in MODES[:4]]
    xmax = max(1.15, max(float(np.nanmax(abs(records[n]['positions'][..., 0])))*1.08 for n in names))
    ymax = max(.68, max(float(np.nanmax(abs(records[n]['positions'][..., 1])))*1.1 for n in names))
    panels = []
    for r, priority in enumerate(PRIORITIES):
        for col in range(5):
            mode = MODES[col-1] if col else None
            name = f'{regime}__{mode}_{priority}__learned_20' if mode else f'{regime}__base'
            record, summary = records[name], summaries[name]
            ax = axes[r, col]; ax.set_xlim(-xmax, xmax); ax.set_ylim(-ymax, ymax)
            ax.set_aspect('equal'); ax.grid(alpha=.15)
            for y in [-.4, -.12, .12, .4]:ax.axhline(y, color='#888888', ls=':', lw=.6)
            ax.axhline(0, color='#aaaaaa', lw=.5)
            ax.add_patch(Circle((1, 0), .1, fill=False, color='#34844b', lw=1))
            ax.scatter([1], [0], marker='*', s=55, c='#34844b', zorder=8)
            if mode:
                title = (f'{mode.replace("_", " ").title()} | {priority}\n'
                         f'Mode {summary["mode_counts"][mode]}/16; task {summary["task_success"]}/16\n'
                         f'Both {summary["successful_mode_counts"][mode]}/16')
            else:
                title = (f'Raw base | repeated reference\nTask {summary["task_success"]}/16; Other {summary["mode_counts"]["other"]}/16\n'
                         + ' '.join(f'{SHORT[k]}:{summary["mode_counts"][m]}' for k,m in enumerate(MODES[:4])))
            ax.set_title(title, fontsize=9)
            ids = summary['mode_ids']
            lines = [ax.plot([], [], color=COLORS[k] if k >= 0 else '#333333', alpha=.4, lw=1)[0] for k in ids]
            bold = ax.plot([], [], color=COLORS[ids[0]], lw=2.7)[0]
            point = ax.plot([], [], 'o', color=COLORS[ids[0]], mec='black', mew=.8, ms=5)[0]
            midpoint = ax.scatter([], [], s=16, facecolors='none', edgecolors=[COLORS[k] for k in ids], linewidths=.8)
            fail = ax.plot([], [], 'x', color='#cf2028', ms=7, mew=1.5)[0]
            if col == 0:ax.set_ylabel(f'{priority.title()} priority\ny position')
            if r == 1:ax.set_xlabel('x position')
            panels.append((record, summary, lines, bold, point, midpoint, fail))
    legend = [Line2D([0],[0],color=c,lw=2,label=m) for c,m in zip(COLORS,SHORT)]
    legend += [Line2D([0],[0],color='black',lw=2.7,label='Thick: fixed episode 0'),
               Line2D([0],[0],color='#cf2028',marker='x',ls='',label='Task failure')]
    fig.legend(handles=legend, loc='lower center', ncol=7, frameon=False, fontsize=9, bbox_to_anchor=(.5,.025))
    title = fig.suptitle('', fontsize=16, y=.98)
    fig.text(.5,.09,'All 16 actual paths are visible. Colors / small open circles indicate the final midpoint mode. No favorable episode selection.',ha='center',fontsize=9)
    fig.text(.5,.067,'Same frozen SFPS, initial states and seeds. Raw base: unit-normal latent. Steering: K20, best, M128, current std8, L4 unit-normal futures.',ha='center',fontsize=9)
    fig.subplots_adjust(top=.86, bottom=.16, hspace=.33, wspace=.17)
    frames=[]
    for frame in range(65):
        title.set_text(f'Raw base vs preference steering | {regime.title()} starts | step {frame}/64')
        for record, summary, lines, bold, point, midpoint, fail in panels:
            paths, lengths = record['positions'], record['lengths']
            for i, line in enumerate(lines):
                n=min(frame+1,int(lengths[i]));line.set_data(paths[i,:n,0],paths[i,:n,1])
            n=min(frame+1,int(lengths[0]));bold.set_data(paths[0,:n,0],paths[0,:n,1]);point.set_data([paths[0,n-1,0]],[paths[0,n-1,1]])
            if frame>=32:midpoint.set_offsets(paths[:,32])
            failed=[i for i,ok in enumerate(summary['task_success_mask']) if not ok and frame>=lengths[i]-1]
            if failed:
                last=np.array([paths[i,int(lengths[i])-1] for i in failed]);fail.set_data(last[:,0],last[:,1])
        fig.canvas.draw();frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()))
        if frame in [0,32,64]:frames[-1].save(out/f'{regime}_frame_{frame:02d}.png')
    frames[0].save(out/f'{regime}_base_vs_steering.gif',save_all=True,append_images=frames[1:],duration=110,loop=0,optimize=False)
    plt.close(fig)


def write_report(out, summaries, rows, provenance):
    lines=['# Raw base와 preference steering의 mode 선택 비교','',
           '2026-09-12. [Method](2026-09-12-preference-steering-method.md), [기존 mode 실험](2026-09-12-mode-aligned-steering-results.md).', '',
           '**Goal 도달과 mode 선택을 분리하면 steering의 효과가 보인다.** Centered base는 goal에 모두 도달하지만 wide가 없고 중앙 Other가 많다. Steering은 동일한 base checkpoint를 사용해 요청한 네 mode를 모두 선택했다. Gaussian wide에는 아직 일부 mode 불일치가 남는다.', '',
           '## 비교 대상과 성공 정의','',
           '- Raw base: preference 입력 없이 매 결정 unit-normal latent 하나를 실행. 목표 mode에 따라 달라지는 policy가 아니므로, 같은 16개 base 경로의 mode 발생 빈도를 각 목표와 비교한다.',
           '- Steering: 합성 scoped 응답 20개로 학습한 preference, mode feature, best, M128, 현재 latent std8, L4/unit-normal 미래. 그룹 우선순위 두 설정 모두 표시한다.',
           '- Task-only: steering과 동일한 M128/std8/L4/best에서 preference 항만 제거. Raw base 대비 변화에는 탐색 폭·후보 수·task cost도 포함되므로, preference의 역할을 확인할 때 이 대조군을 함께 본다.',
           '- **Task success**: 65개 state 완주, numerical/action-limit failure 없음, 최종 goal tolerance .1 충족.',
           '- **Mode hit**: 실제 step32 위치가 요청한 midpoint bin에 속함. Goal 실패 여부와 별개다.',
           '- **Joint success**: task success AND mode hit. 아래 주 비교표의 성공 기준이며 분모는 항상 전체16개 episode다.',
           '- Other는 midpoint가 (-.12,.12)에 있는 중앙 경로이며 task failure와 다르다.', '',
           '동일한 초기 위치·환경 seed·rollout seed와 frozen checkpoint/stats를 확인했다. 기존 공개 compact NPZ에서 base32 + task-only32 + learned256 = 320 episode를 재집계했고, 새 학습/추론은 실행하지 않았다. 비교에 쓴 36개 source condition의 저장 지표도 독립 bin 계산과 대조했다. Profile별 반복은 독립 표본 수를 늘리지 않는다.', '',
           '이전 전후 GIF의 회색 before는 초기 continuous **steering**이었다. 이 문서의 첫 열은 preference가 없는 **raw base**로 교체한 직접 비교다.', '',
           '## Base 자체의 실제 mode 분포','',
           '각 mode 셀은 **mode 발생 수 (그중 goal까지 성공한 수)**다.', '',
           '| 시작점 | UN | UW | LN | LW | Other | Task success |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for regime in REGIMES:
        b=summaries[f'{regime}__base']
        cells=[f'{b["mode_counts"][m]} ({b["successful_mode_counts"][m]})' for m in MODES]
        lines.append('| '+regime.title()+' | '+' | '.join(cells)+f' | {b["task_success"]}/16 |')
    lines += ['', 'Gaussian base의 episode4와5는 각각 LW/UW 경로로 완주했지만 goal을 놓쳤다. 따라서 mode만 세면 LW8/UW6이며, joint success는 LW7/UW5다.', '',
              '## 요청 mode별 joint success','',
              '| 시작점 | 목표 mode | Raw base | Task-only, 동일 탐색량 | Steering 방향 우선 | Steering 폭 우선 |',
              '|---|---|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f'| {row["regime"].title()} | {row["target"]} | {row["base_joint"]}/16 | {row["task_only_joint"]}/16 | {row["direction_joint"]}/16 | {row["width_joint"]}/16 |')
    lines += ['', 'Steering과 task-only는 모든 조건에서 task 자체가16/16 성공했다. 주 표의 steering 실패는 mode 불일치다. Raw base에서 이미 요청 mode+task에 성공한 episode를 steering이 실패로 바꾼 경우는 이번 비교에서0건이었다. CSV/JSON의 rescued/retained/lost가 같은 episode별 전이를 기록한다.', '',
              '![Target mode success](artifacts/2026-09-12-base-vs-steering/target_mode_success.png)', '',
              '## Steering이 실제로 선택한 mode 전체 분포','',
              '각 행은 같은16개 seed다. Mode 열에는 task까지 성공한 episode만 넣고 task 실패를 마지막 열로 분리하여 각 행의 합이16이 되게 했다. 초록 테두리는 요청 mode다.', '',
              '![Mode distribution](artifacts/2026-09-12-base-vs-steering/mode_distribution.png)', '',
              'Gaussian upper-wide 실패는 방향 우선에서 UN3개, 폭 우선에서 LW4개였다. Lower-wide 실패는 방향 우선에서 LN1개, 폭 우선에서 UW1개였다. 우선순위가 서로 다른 속성의 포기를 유도하는 모습도 이 분포에서 확인할 수 있다.', '',
              '## 실제 추론 GIF','',
              '각 행의 첫 열은 같은 raw base reference, 나머지 네 열은 요청 mode다. 위/아래 행은 방향/폭 우선이다. 전체16개 경로를 표시하고 굵은 경로는 모든 패널에서 고정 episode0이다. 색은 최종 midpoint mode, 원형 표식은 step32 위치, 빨간 x는 task 실패다. 가상 미래나 성공 사례만 골라 만든 경로가 아니다.', '',
              '![Centered](artifacts/2026-09-12-base-vs-steering/centered_base_vs_steering.gif)', '',
              '![Gaussian](artifacts/2026-09-12-base-vs-steering/gaussian_base_vs_steering.gif)', '',
              '## 판정과 한계','',
              '이번 seed 집합에서 **Centered 네 mode와 Gaussian narrow의 preference steering은 성공**했다. Gaussian wide는 raw base보다 joint success가 늘었지만 아직 안정적인 전환을 모두 달성하지 못했다. 특히 centered에서는 base goal16/16이라는 값만으로 mode 제어가 성공했다고 판단하면 안 된다.', '',
              'Raw base 비교는 전체 inference wrapper의 효과이며 preference 하나만의 인과 효과로 해석하지 않는다. 동일 탐색량 task-only 대조군도 네 요청 mode를 따로 선택하지 못하므로, 현재 설정에서 preference 항이 목표별 분포를 바꾸는 역할을 확인할 수 있다. 다만16개 기존 pilot seed, hard-bin mode, 합성 피드백의 결과다. 16/16의 Wilson95% 하한은 약80.6%이며 새로운 seed/사용자에 대한 보장은 아니다.', '',
              '## 산출물과 재현','',
              '- [JSON: 모든 mode·task mask와 paired 전이](artifacts/2026-09-12-base-vs-steering/comparison.json)',
              '- [CSV: 목표별 비교](artifacts/2026-09-12-base-vs-steering/target_mode_comparison.csv)',
              '- [재현 스크립트](compare_base_steering.py): `MPLCONFIGDIR=/tmp/base-mode-mpl .venv/bin/python docs/research/compare_base_steering.py`.',
              '- 원본: `docs/research/artifacts/2026-09-12-preference-steering/actual_rollouts.npz`. 새 checkout에서도 이미 Git에 포함된 이 파일로 재집계한다.',
              '- Checkpoint와 source NPZ hash, 초기 상태/seed pairing 검증은 JSON `provenance`에 있다.', '']
    for row in rows:
        assert row['direction_lost'] == row['width_lost'] == 0
    (out/'comparison.json').write_text(json.dumps(dict(provenance=provenance, rows=rows, conditions=summaries),indent=2)+'\n')
    with (out/'target_mode_comparison.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    (ROOT/'docs/research/2026-09-12-base-vs-steering.md').write_text('\n'.join(lines))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--no-gif',action='store_true',help='Reuse already rendered GIFs in the study output directory.')
    args=p.parse_args();out=DEFAULT_OUT
    if args.no_gif and not all((out/f'{regime}_base_vs_steering.gif').exists() for regime in REGIMES):
        raise ValueError('--no-gif requires both previously rendered study GIFs')
    out.mkdir(parents=True,exist_ok=True)
    records,summaries,provenance=load_records();rows=comparison_rows(summaries)
    plots(out,summaries,rows)
    if not args.no_gif:
        for regime in REGIMES:
            trajectory_gif(out,regime,records,summaries)
            print('Rendered',regime,flush=True)
    write_report(out,summaries,rows,provenance)
    print('PASS: paired seeds/initial states, frozen inputs, all36 source metrics, and base-to-steering transitions.',flush=True)
    for row in rows:print(row,flush=True)


if __name__=='__main__':main()
