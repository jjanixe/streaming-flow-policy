# Preference steering 연구 기록

현재 상태는 [method: 수식·핵심 구현](2026-09-12-preference-steering-method.md), [raw base 대비 steering 성공 비교](2026-09-12-base-vs-steering.md), [mode 제어 결과·GIF](2026-09-12-mode-aligned-steering-results.md), [산출물·위치·재현](2026-09-12-preference-steering-artifacts.md)에서 확인한다.

| 문서 | 범위 |
|---|---|
| [초기 diffusion/flow 조사](../../preference_guided_diffusion_flow_robotics.md) | 초기 few-shot steering 연구 제안 |
| [논문 게재·공개 코드 audit](2026-09-11-preference-publication-audit.md) | 2026-09-11에 조사한 공식 출판/구현 근거; 이번 push에서 새로 조사한 상태는 아님 |
| [Grouped formulation](../../vrhandover_grouped_few_shot_guidance_formulation.md) | 제공된 설계 문서; toy에 없는 P/M/C·handover 부분은 구현 완료로 해석하지 않음 |
| [수식 검토](2026-09-11-grouped-preference-formulation-review.md) | scope, 식별 가능성, 유한 continuation과 anchor의 조건 |
| [Flat preference pilot](2026-09-11-preference-steering-results.md) | 이전 baseline과 결과 |
| [Continuous grouped pilot](2026-09-11-grouped-preference-results.md) | 두 그룹 fitting·conditional futures 구현, mode와 utility 불일치 |
| [Mode 선택 진단](2026-09-12-mode-selection-diagnosis.md) | feature 불일치와 후보 coverage 분석 |
| [Mode-aligned 결과](2026-09-12-mode-aligned-steering-results.md) | frozen policy의 실제 제어, few-shot fit, 후보 수/탐색 폭 비교와 한계 |

과거 문서는 그 시점의 실험 기록이다. 최신 mode feature·controller 설정과 push에 포함한 파일 범위는 method 및 artifact 문서를 기준으로 본다.
