# Preference steering 산출물과 위치

2026-09-12 publication. [Method·수식·코드](2026-09-12-preference-steering-method.md), [최신 결과·GIF](2026-09-12-mode-aligned-steering-results.md).

추가 분석: [raw base 대비 mode 선택·성공 비교](2026-09-12-base-vs-steering.md). 기존 공개 compact NPZ를 재집계했으며 새 episode를 추가하지 않았다. 새 비교표·분포 그림·base 직접 비교 GIF는 `docs/research/artifacts/2026-09-12-base-vs-steering/`에, 재현 코드는 [compare_base_steering.py](compare_base_steering.py)에 있다. [추가 분석의 경로·해시·검증 manifest](artifacts/2026-09-12-base-vs-steering/artifact_manifest.json)를 별도로 저장했다. 아래 499-file manifest는 앞선 publication 시점의 inventory이며 이 추가 분석 파일은 포함하지 않는다.

## 저장 범위

코드·논문 조사·수식 검토·보고서와 함께, `env/artifacts/`의 JSON/CSV, 그림/GIF, 분석 스크립트, synthetic demonstration, checkpoint/stats, 소규모 이전 실험 배열을 Git에 포함한다. 전체 grouped 후보/가상 미래 NPZ는 로컬에 보존하고 아래 manifest에 경로·크기·SHA-256을 남긴다. Manifest의 `storage=local-only` 파일은 clone에 포함되지 않는다. 외부 다운로드 링크나 별도 저장소에 업로드했다고 주장하지 않는다.

- 전체 inventory: 499 files, 2.407 GiB.
- 이 중 Git 포함: 271 files, 64.32 MiB. 아래 compact NPZ·문서·검증 기록은 별도다.
- [전체 machine-readable manifest](artifacts/2026-09-12-preference-steering/artifact_manifest.json): `path`, `local_path`, `bytes`, `sha256`, `storage`, `role`와 폴더별 합계.
- 원시 산출물의 로컬 worktree: `/home/janice9902/starlab/forked/streaming-flow-policy/.worktrees/stage-b2-sfps`.

## 바로 확인할 산출물

| 산출물 | 위치 | 해석 |
|---|---|---|
| 주 결과 Centered GIF | [centered_mode_aligned_inference.gif](../../env/artifacts/grouped_preference/mode-aligned-best-M128-std8/animations/centered_mode_aligned_inference.gif) | 실제 16개 경로, fixed episode 0, before/learned20/oracle |
| 주 결과 Gaussian GIF | [gaussian_mode_aligned_inference.gif](../../env/artifacts/grouped_preference/mode-aligned-best-M128-std8/animations/gaussian_mode_aligned_inference.gif) | 실패한 wide 경로 포함 |
| Oracle 비교표/그림 | [CSV](../../env/artifacts/grouped_preference/analysis/mode-comparison/conditions.csv), [heatmap](../../env/artifacts/grouped_preference/analysis/mode-comparison/oracle_ablation.png) | M/selector/feature/std 비교; 성공은 goal AND mode |
| Few-shot 640 fit | [JSON](../../env/artifacts/grouped_preference/analysis/mode-aligned-fits/fit_replicates.json), [plot](../../env/artifacts/grouped_preference/analysis/mode-aligned-fits/few_shot_fit.png) | synthetic scoped feedback, 인간 성능 아님 |
| 실제 실행 경로 축약본 | [actual_rollouts.npz](artifacts/2026-09-12-preference-steering/actual_rollouts.npz) | 신규 172조건과 이전 continuous 36조건, 총 3,328 episode; 원본 배열과 exact equality 검사 |
| Frozen checkpoint | [sfps_best.pt](../../env/artifacts/stage_b/b2-seed0-optimized/sfps_best.pt) | 이번 inference 실험에 사용한 동일 모델 |
| Stats / demonstrations | [stats](../../env/artifacts/stage_b/b2-seed0-optimized/pusht_stats.npz), [synthetic bank](../../env/artifacts/demonstrations.npz) | 새 checkout에서 CLI 경로로 사용 |
| Fresh tests | [env-tests.log](artifacts/2026-09-12-preference-steering/env-tests.log) | publication 전 389 passed, 2 skipped, 15 warnings, 103.06초 |
| Publication 검증 | [publication_validation.json](artifacts/2026-09-12-preference-steering/publication_validation.json) | Git 파일·해시·링크 대조, 공개 입력의 CPU 추론과 compact 예제 확인 |
| 최종 독립 검토 | [review](artifacts/2026-09-12-preference-steering/reviews/2026-09-12-mode-aligned-steering-final-review.md) | 2,752 episode / 22,016 decision 검증, Gaussian wide 한계 유지 |

## 실험 폴더

`env/artifacts/` 기준 경로다. 각 폴더의 metadata는 Git에 있으며, 전체 NPZ의 포함 여부는 file manifest를 확인한다. M32/M128 oracle matrix의 `.complete.json`은 원래 full NPZ의 무결성 기록이다. Clone에 marker만 있고 full NPZ가 없으면 resume할 수 없으므로 새 output directory에서 실행한다.

| 폴더 | 전체 MiB | Git MiB | 설명 |
|---|---:|---:|---|
| `demonstrations.npz` | 0.36 | 0.36 | synthetic input |
| `grouped_preference/analysis` | 1.57 | 1.57 | 분석코드·fit반복·비교집계·anchor/coverage진단 |
| `grouped_preference/mode-aligned-ablation-M128-std1` | 1027.06 | 0.36 | oracle soft/best M128, 512 episode |
| `grouped_preference/mode-aligned-ablation-M32-std1` | 258.70 | 0.36 | oracle soft/best M32, 512 episode |
| `grouped_preference/mode-aligned-best-M128-std8` | 727.04 | 13.73 | 주 oracle/learned20 비교, 576 episode |
| `grouped_preference/mode-aligned-best-M8-std1` | 70.52 | 0.41 | mode feature / best / M8, 576 episode |
| `grouped_preference/mode-aligned-coverage-std1` | 9.85 | 0.00 | 첫 결정의 counterfactual coverage; 실제 제어 성공률 아님 |
| `grouped_preference/mode-aligned-coverage-std16` | 13.91 | 0.00 | 첫 결정의 counterfactual coverage; 실제 제어 성공률 아님 |
| `grouped_preference/mode-aligned-coverage-std4` | 7.46 | 0.00 | 첫 결정의 counterfactual coverage; 실제 제어 성공률 아님 |
| `grouped_preference/mode-aligned-coverage-std8` | 5.68 | 0.00 | 첫 결정의 counterfactual coverage; 실제 제어 성공률 아님 |
| `grouped_preference/mode-aligned-coverage-std8-M512` | 22.37 | 0.00 | 첫 결정의 counterfactual coverage; 실제 제어 성공률 아님 |
| `grouped_preference/mode-aligned-smoke-std1` | 128.25 | 0.01 | 예비 smoke; 정규 비교의 집계에서 제외 |
| `grouped_preference/mode-aligned-soft-M8-std1` | 70.54 | 0.41 | mode feature / soft / M8, 576 episode |
| `grouped_preference/mode-aligned-two-chunk-probe` | 0.08 | 0.08 | 실패 9건에 한정한 oracle 진단, 3/9 복구; 작은 NPZ 포함 |
| `grouped_preference/pilot-seed0` | 90.00 | 15.32 | 이전 continuous grouped baseline, 576 episode |
| `preference/pilot-seed0` | 21.80 | 21.80 | 이전 flat-preference baseline |
| `stage_b/b2-seed0` | 5.07 | 5.07 | 초기 base SFPS 실험 |
| `stage_b/b2-seed0-optimized` | 4.83 | 4.83 | 동결한 optimized SFPS 및 base 평가 |

## Compact NPZ schema와 검증

`experiment_names[c]`, `condition_names[c]`, `source_paths[c]`가 condition 축의 의미다. `is_formal_mode_run`으로 신규 172조건과 이전 36조건을 구분한다. 같은 16개 seed를 방법·profile별로 반복하므로 3,328 episode를 독립 성공률 표본으로 합치지 않는다.

- `positions`: `[208,16,65,2]`, `requested_actions`: `[208,16,64,2]`.
- `lengths`, `success`, `numerical_failure`, `action_limit_failure`, `goal_errors`: `[208,16]`.
- `environment_seeds`, `rollout_seeds`: `[208,16]`.
- `decision_mask`, `selected_indices`, `fallback`: `[208,16,8]`.
- `selected_current_latents`: `[208,16,8,2]`.
- `lengths`는 accepted state 수이며, 미실행 구간의 NaN과 실패 episode도 원본 그대로다. 전체 후보와 future 배열은 이 축약본에 없다.

```python
import numpy as np
z = np.load('docs/research/artifacts/2026-09-12-preference-steering/actual_rollouts.npz', allow_pickle=False)
i = np.flatnonzero((z['experiment_names'] == 'mode-aligned-best-M128-std8') & (z['condition_names'] == 'gaussian__upper_wide_width__learned_20'))[0]
y = z['positions'][i, :, 32, 1].astype(np.float64)
ok = z['success'][i] & (z['lengths'][i] == 65)
ok &= ~z['numerical_failure'][i] & ~z['action_limit_failure'][i]
assert np.count_nonzero(ok & (y >= .4)) == 12
```

## 재현 및 provenance 주의점

새 checkout용 명령은 [method의 재현 절](2026-09-12-preference-steering-method.md#8-새-checkout에서-재현)에 있다. 기존 JSON의 절대 경로는 실행 당시의 위치이며, 새 checkout의 입력 경로로 자동 해석되지 않는다. 복사한 시연 bank의 바이트와 SHA-256은 원래 공유 root 파일과 같다.

`analysis/mode-comparison/final_verification.json`과 `final-verified-sources.zip`은 publication 이전 검증 시점의 snapshot이다. 이후 문서 링크를 이관했으므로 그 report hash를 현재 문서 hash로 해석하지 않는다. 당시 SDD review의 공개 사본은 위 `reviews/`에 있다. 현재 공개 artifact의 hash는 이 publication manifest를 기준으로 한다.

Ablation resume의 source 검사는 명시된 8개 모듈만 포함한다. 간접 normalization/loader/seed helper를 포함해 어떤 code라도 바뀌면 새 폴더를 사용한다. 이번 완료된 실험에서는 관련 dependency 변경 없이 동결 검사와 독립 replay를 통과했다.
