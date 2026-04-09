# Fractal Genome BTC Backtest 2026-04-09

## Scope

- Pair: `BTCUSDT`
- Date of run: `2026-04-09`
- Workflow split:
  - Implementer: fractal genome search path completion and BTC single-asset backtest path
  - Verifier: independent curriculum backtest and result contract check

## Implementation Status

- Fractal genome core uses recursive `LogicCell` composition with `ThresholdCell`, `AndCell`, `OrCell`, `NotCell`.
- Search path supports BTC single-asset mode, dynamic windows, depth curriculum, and single-asset report export.
- Automatic online LLM review path is implemented but was not exercised in this run because `OPENAI_API_KEY` was not set.

## Implementer Run

- Command family: `scripts/search_pair_subset_fractal_genome.py`
- Search result: `selection.reason = no_gate_pass`
- Reported fallback candidate metrics:

| Window | Avg Daily Return | Total Return | Max Drawdown | Sharpe | Trades |
| --- | ---: | ---: | ---: | ---: | ---: |
| Recent 2 months | `0.0063276` | `0.4491` | `-0.0882` | `6.5509` | `61` |
| Recent 6 months | `0.0078263` | `2.9997` | `-0.0907` | `6.4594` | `269` |
| Full history | `0.0052144` | `1627.0224` | `-0.1297` | `5.1848` | `2464` |

## Verifier Run

- Command family: `scripts/verify_fractal_genome_btc_curriculum.py`
- Independent selection: `deepest_stage_selected`
- Selected tree:
  - `btc_regime >= -0.05`
  - `if_true -> expert 0`
  - `if_false -> expert 1`
- Selected tree complexity:
  - `tree_depth = 1`
  - `logic_depth = 0`
  - `logic_size = 1`

### Verified windows

| Window | Start (UTC) | End (UTC) | Bars | Avg Daily Return | Total Return | Max Drawdown | Sharpe | Trades |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Recent 2 months | `2026-02-09T06:40:00+00:00` | `2026-04-09T06:40:00+00:00` | `16993` | `0.0055470` | `0.3726` | `-0.0862` | `5.5927` | `61` |
| Recent 6 months | `2025-10-09T06:40:00+00:00` | `2026-04-09T06:40:00+00:00` | `52417` | `0.0077151` | `2.8724` | `-0.0915` | `6.3481` | `286` |
| Full since first bar | `2022-04-06T00:00:00+00:00` | `2026-04-09T06:40:00+00:00` | `421713` | `0.0052100` | `1616.5641` | `-0.1297` | `5.1806` | `2460` |

## Contract Checks

- `recent_2m_end_matches_latest_data = true`
- `recent_6m_end_matches_latest_data = true`
- `full_start_matches_first_data = true`
- `full_end_matches_latest_data = true`

## Baseline Comparison

기준선은 [`models/gp_regime_mixture_btc_bnb_pairwise_repair_summary.json`](/Users/jsa/work/epic_invest/models/gp_regime_mixture_btc_bnb_pairwise_repair_summary.json) 의 `BTCUSDT` pair config를 사용했다.
다만 기준선은 `BTCUSDT + BNBUSDT breadth`를 사용하고, 현재 `BNBUSDT` 캐시는 `2026-04-08T13:05:00+00:00`까지만 존재한다.
그래서 아래 비교는 공정성을 위해 `2026-04-08T13:05:00+00:00` 공유 끝점을 기준으로 다시 재생한 결과다.

### Shared-end comparison

| Window | Baseline Win Rate | Fractal Win Rate | Delta | Baseline Return | Fractal Return | Delta | Baseline MDD | Fractal MDD | Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Recent 2 months | `0.4237` | `0.2881` | `-0.1356` | `0.4115` | `0.3998` | `-0.0117` | `-0.091081` | `-0.091014` | `+0.000067` |
| Recent 6 months | `0.4780` | `0.3571` | `-0.1209` | `2.7770` | `2.8622` | `+0.0852` | `-0.091203` | `-0.091014` | `+0.000189` |
| Full since first bar | `0.5123` | `0.2514` | `-0.2609` | `1588.1326` | `1627.0224` | `+38.8898` | `-0.131208` | `-0.129681` | `+0.001527` |

### Reading the deltas

- Win rate uses `daily_win_rate`.
- Return uses `total_return`.
- MDD delta is `fractal - baseline`; because drawdown is negative, a positive delta means the fractal result had a shallower drawdown.

### What improved and what degraded

- Recent 2 months:
  - Return slightly worse than baseline.
  - MDD almost identical, fractal marginally better.
  - Win rate materially worse.
- Recent 6 months:
  - Return better than baseline.
  - MDD marginally better.
  - Win rate still worse.
- Full history:
  - Terminal return better than baseline.
  - MDD better than baseline.
  - Win rate much worse.

### Important context

- The current fractal candidate traded far less often than the baseline:
  - Recent 2 months: `61` vs `1324`
  - Recent 6 months: `268` vs `6141`
  - Full history: `2464` vs `29365`
- So this candidate is not winning by hit rate. If it wins at all, it is doing so through lower turnover and larger average contribution per winning day.

## Interpretation

- The BTC single-asset fractal search path is operational and backtested on the requested windows.
- The independent verifier reproduced the requested window bounds and produced comparable results.
- Under the tested budget, the search still converged to a shallow tree rather than a deep fractal organ structure.
- This run should be treated as infrastructure completion plus first verified BTC backtest, not as proof that deeper recursive trees already outperform the current simpler policies.

## Cycle 2 Upgrade

이번 사이클에서는 검색기와 검증기를 한 단계 더 강화했다.

- `required_pairs` 기준 projected expert dedupe
- greedy diverse expert pool selection
- generation survivor selection에 structural diversity / depth coverage 반영
- immigrant injection 추가
- verifier에서 `latest_monotonic_stage_pass` 선택 규칙 도입
- verifier에서 breadth parity를 검색기와 일치시킴

### Cycle 2 search result

- Search command family: `scripts/search_pair_subset_fractal_genome.py`
- Config:
  - `population=24`
  - `generations=4`
  - `max_depth=4`
  - `logic_max_depth=3`
  - `survivor_diversity_weight=0.45`
  - `survivor_depth_weight=0.65`
  - `immigrant_rate=0.18`
- Expert pool diagnostics:
  - `raw_candidate_count=47`
  - `unique_projected_candidate_count=34`
  - `projection_collision_count=13`
  - `selected_count=18`
- Selection remained `no_gate_pass`, but top-k에는 `tree_depth=2`, `logic_depth=2` 후보가 실제로 포함되기 시작했다.

### Cycle 2 verifier result

- Verifier command family: `scripts/verify_fractal_genome_btc_curriculum.py`
- Config:
  - `depth_curriculum=1,2,3,4`
  - `logic_curriculum=1,1,2,3`
- Outcome:
  - `selection.reason = latest_monotonic_stage_pass`
  - `selection.stage = depth_1`
  - `curriculum_passed = false`
- Interpretation:
  - deeper stage가 생성되더라도 `selected_candidate`가 실제로 구조 증가를 보여주지 못해 `depth_2~4`는 verifier 기준에서 탈락했다.

### Cycle 2 baseline comparison

기준선과의 직접 비교는 여전히 `2026-04-08T13:05:00+00:00` 공유 끝점을 사용했다.

#### Fallback best candidate

| Window | Baseline Win Rate | Fractal Win Rate | Delta | Baseline Return | Fractal Return | Delta | Baseline MDD | Fractal MDD | Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Recent 2 months | `0.4237` | `0.2881` | `-0.1356` | `0.4115` | `0.4060` | `-0.0055` | `-0.091081` | `-0.072350` | `+0.018732` |
| Recent 6 months | `0.4780` | `0.3571` | `-0.1209` | `2.7770` | `3.0928` | `+0.3158` | `-0.091203` | `-0.074295` | `+0.016907` |
| Full since first bar | `0.5123` | `0.2794` | `-0.2329` | `1588.1326` | `1952.3231` | `+364.1905` | `-0.131208` | `-0.122096` | `+0.009112` |

#### Structural best candidate

| Window | Baseline Win Rate | Structural Win Rate | Delta | Baseline Return | Structural Return | Delta | Baseline MDD | Structural MDD | Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Recent 2 months | `0.4237` | `0.2881` | `-0.1356` | `0.4115` | `0.3432` | `-0.0683` | `-0.091081` | `-0.072350` | `+0.018732` |
| Recent 6 months | `0.4780` | `0.3571` | `-0.1209` | `2.7770` | `2.2442` | `-0.5328` | `-0.091203` | `-0.074295` | `+0.016907` |
| Full since first bar | `0.5123` | `0.2678` | `-0.2445` | `1588.1326` | `830.9851` | `-757.1475` | `-0.131208` | `-0.117802` | `+0.013406` |

### Cycle 2 conclusion

- 검색기 자체는 확실히 개선됐다. 이제 deeper / richer tree가 population과 top-k에 실제로 남는다.
- 하지만 최종 우승 후보는 여전히 shallow tree다.
- `fallback best`는 기준선 대비 `승률은 낮지만`, `6개월/전체 수익률`과 `MDD`는 더 좋았다.
- 반대로 `structural best`는 구조는 더 복잡하지만 성과는 기준선보다 명확히 낮았다.
- 따라서 현재 판정은 `엔진 구현은 진전`, `구조적 후보 생성은 성공`, `깊은 프랙탈 전략의 성과 우위는 아직 미확인`이다.

## Cycle 3 Completion

이번 사이클에서는 이전에 남아 있던 세 가지 구현 공백을 실제 코드와 검증으로 닫았다.

- `raw indicator cell space` 확장: 시간봉 기준 `RSI / ATR / MACD / Bollinger / MFI / CCI / Donchian / volume ratio / intraday return / intraday drawdown` 을 모두 `ThresholdCell` 조건으로 사용할 수 있게 했다.
- `near-frontier structural selection` 강화: 성능이 너무 멀리 밀리지 않는 범위 안에서는 더 깊은 구조를 최종 우승 후보로 선택할 수 있게 했다.
- `automatic LLM filter` 실경로 검증: 로컬 OpenAI 호환 mock server를 띄워 `POST /v1/chat/completions` 경로를 실제로 통과시켰다.

### Cycle 3 implementation status

- Main search feature catalog:
  - BTC-only run: `feature_count=41`, `condition_option_count=426`
  - Multi-pair mock-openai run: `feature_count=105`, `condition_option_count=1124`
- Main BTC search selection:
  - `selection.reason = no_gate_pass_near_frontier_structural`
  - `winner_tree_depth = 2`
  - `winner_logic_depth = 3`
  - `winner_leaf_cardinality = 3`
- 즉, 검색기 기준으로는 이제 shallow tree가 아니라 실제 recursive tree가 최종 선택 후보가 됐다.

### Cycle 3 mock-openai verification

- Command family: `scripts/verify_fractal_genome_openai_mock.py`
- Result:
  - `request_count = 1`
  - `path = /v1/chat/completions`
  - `auth_header = Bearer local-mock-key`
  - `response_format.type = json_object`
  - `auto_llm_review_events[0].attempted = 1`
  - `auto_llm_review_events[0].added = 1`
- Interpretation:
  - 자동 LLM 필터는 더 이상 “코드 경로만 있음” 상태가 아니다.
  - 외부 키 없이도 실제 HTTP review path를 독립 검증했다.

### Cycle 3 verifier result

- Command family: `scripts/verify_fractal_genome_btc_curriculum.py`
- Outcome:
  - `selection.reason = latest_monotonic_stage_pass`
  - `selection.stage = depth_1`
  - `curriculum_passed = false`
- Meaning:
  - 검색기 최종 선택은 깊어졌지만, 독립 verifier의 monotonic curriculum 기준으로는 아직 `depth_2+`가 연속 승격되지는 못했다.
  - `depth_4` 단계에서는 실제로 더 복잡한 구조가 나왔지만 `performance_pass = false` 로 탈락했다.

## Cycle 4 Leaf Gene And Baseline-Relative Search

이번 사이클에서는 `baseline을 실제로 이기도록` 목적함수와 verifier를 다시 짰다.

- 구현자 트랙:
  - `LeafNode(expert_idx)`를 `LeafNode(expert_idx + gene)`로 확장
  - leaf gene으로 `route_threshold_bias`, `mapping_shift`, `target_vol_scale`, `gross_cap_scale`, `kill_switch_scale`, `cooldown_scale`를 진화시킴
  - 검색 점수를 절대 성과가 아니라 `baseline-relative delta score`로 교체
- 검증자 트랙:
  - 독립 verifier에 `walk-forward` 폴드 검증 추가
  - `commission_multiplier=1.0, 1.5, 2.0` 스트레스 검증 추가
  - `promotion_gate_passed`를 `전 폴드 통과 + stress survival` 기준으로 강화

### Cycle 4 code verification

- `py_compile` 통과:
  - `scripts/fractal_genome_core.py`
  - `scripts/search_pair_subset_fractal_genome.py`
  - `scripts/verify_fractal_genome_btc_curriculum.py`
  - `scripts/verify_fractal_genome_cells.py`
  - `scripts/verify_fractal_genome_openai_mock.py`
- `verify_fractal_genome_cells.py --mode self-check` 통과
- `verify_fractal_genome_openai_mock.py` 재실행 통과
  - `POST /v1/chat/completions`
  - `Bearer local-mock-key`
  - `response_format.type = json_object`

### Cycle 4 search result

- Search command family: `scripts/search_pair_subset_fractal_genome.py`
- Config:
  - `population=32`
  - `generations=5`
  - `max_depth=4`
  - `logic_max_depth=3`
  - `expert_pool_size=18`
  - `survivor_diversity_weight=0.45`
  - `survivor_depth_weight=0.75`
  - `immigrant_rate=0.18`
- Runtime:
  - `evaluated_unique_candidates=141`
  - `total_seconds=2.8115`
- Feature set:
  - `feature_count=41`
  - `condition_option_count=426`
- Selection:
  - `selection.reason = no_gate_pass_near_frontier_structural`
  - `winner_tree_depth = 4`
  - `winner_logic_depth = 3`
  - `winner_leaf_cardinality = 5`

`no_gate_pass_near_frontier_structural`의 의미는 명확하다.

- `progressive_improvement`를 통과한 후보가 `0개`
- `target_060`를 통과한 후보가 `0개`
- 그래서 승격 후보는 없었고
- 그 대신 성능 frontier band 안에 있는 후보 중 구조적으로 더 풍부한 재귀 트리를 선택했다.

이번 런에서는 frontier 후보가 `1개`였고, 그 후보가 곧 `depth=4 / logic_depth=3 / leaf_cardinality=5` 구조였다.

### Cycle 4 BTC baseline comparison

아래 비교는 메인 search report에 들어 있는 `selected_candidate`와 `baseline_candidate`의 `BTCUSDT` 지표다.
주의할 점은 baseline summary가 `2026-04-06` 기준이고, 이번 candidate window는 `2026-04-09` 기준이라 끝점이 완전히 동일하지는 않다.
정확한 same-window robustness 판정은 아래 verifier walk-forward 결과를 기준으로 본다.

#### Main selected candidate vs embedded baseline

| Window | Candidate Return | Baseline Return | Delta | Candidate Win Rate | Baseline Win Rate | Delta | Candidate MDD | Baseline MDD | Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Recent 2 months | `0.4491` | `0.4600` | `-0.0109` | `0.2333` | `0.3559` | `-0.1226` | `-0.088248` | `-0.088028` | `-0.000220` |
| Recent 6 months | `2.9997` | `2.9868` | `+0.0129` | `0.3169` | `0.4780` | `-0.1611` | `-0.090735` | `-0.088152` | `-0.002584` |
| Full history | `1366.3979` | `1588.1326` | `-221.7347` | `0.2778` | `0.5127` | `-0.2348` | `-0.129681` | `-0.131208` | `+0.001527` |

읽는 법:

- win rate는 `daily_win_rate`
- return은 `total_return`
- MDD delta는 `candidate - baseline`
- drawdown이 음수이므로 delta가 양수면 candidate가 더 얕은 MDD다

이번 `depth=4` 구조는 다음 특성을 보였다.

- 최근 2개월:
  - 수익률 열세
  - 승률 열세
  - MDD도 소폭 열세
- 최근 6개월:
  - 수익률만 근소 우위
  - 승률은 크게 열세
  - MDD는 더 나쁨
- 전체 구간:
  - 승률 열세
  - 누적 수익률 열세
  - MDD만 소폭 개선

즉, `깊은 구조를 선택하는 것`에는 성공했지만, `baseline을 명확히 압도하는 것`에는 아직 실패했다.

### Cycle 4 verifier result

- Verifier command family: `scripts/verify_fractal_genome_btc_curriculum.py`
- Outcome:
  - `selection.reason = latest_monotonic_stage_pass`
  - `selection.stage = depth_2`
  - `best_stage = depth_2`
  - `curriculum_passed = false`

독립 verifier가 이제 `depth_1`이 아니라 `depth_2`를 선택한다는 점은 중요한 진전이다.
즉, 구현자 search에서만 깊은 구조가 잡히는 것이 아니라 verifier 기준으로도 한 단계 더 깊은 구조가 버티기 시작했다.

#### Walk-forward robustness

| Metric | Value |
| --- | ---: |
| Fold pass rate | `0.6667` |
| Stress survival rate mean | `0.3333` |
| Promotion gate passed | `false` |

폴드별 해석:

- `wf_5`, `wf_4`, `wf_3`, `wf_2`는 nominal 기준 통과
- `wf_6`는 수익률은 baseline보다 약간 높았지만 MDD가 아주 미세하게 더 깊어서 탈락
- `wf_1`은 최근 2개월 구간에서 수익률과 승률이 baseline보다 낮아 탈락
- 대부분 폴드에서 수수료를 `1.5x`, `2.0x`로 올리면 baseline-relative 우위가 무너져 `stress_survival_rate`가 낮게 남음

결론적으로 verifier는 이렇게 판정했다.

- `깊은 구조 자체`는 이제 독립 검증에서도 통과 가능
- 하지만 `walk-forward + commission stress`까지 넣으면 아직 baseline dominance가 아니다
- 현재 bottleneck은 구현 누락이 아니라 `robustness`다

### Cycle 3 baseline comparison

기준선 비교는 공정성을 위해 `2026-04-08` 공유 끝점 기준으로 다시 맞췄다.

#### Selected structural winner vs baseline

| Window | Baseline Win Rate | Selected Win Rate | Delta | Baseline Return | Selected Return | Delta | Baseline MDD | Selected MDD | Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Recent 2 months | `0.3559` | `0.2333` | `-0.1226` | `0.4600` | `0.4491` | `-0.0109` | `-0.088028` | `-0.088248` | `-0.000220` |
| Recent 6 months | `0.4780` | `0.3224` | `-0.1556` | `2.9868` | `3.0437` | `+0.0568` | `-0.088152` | `-0.090735` | `-0.002584` |
| Full since first bar | `0.5127` | `0.2445` | `-0.2681` | `1588.1326` | `1316.7731` | `-271.3595` | `-0.131208` | `-0.129681` | `+0.001527` |

#### Fallback best vs baseline

| Window | Baseline Win Rate | Fallback Win Rate | Delta | Baseline Return | Fallback Return | Delta | Baseline MDD | Fallback MDD | Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Recent 2 months | `0.3559` | `0.2333` | `-0.1226` | `0.4600` | `0.4491` | `-0.0109` | `-0.088028` | `-0.088248` | `-0.000220` |
| Recent 6 months | `0.4780` | `0.3224` | `-0.1556` | `2.9868` | `3.0437` | `+0.0568` | `-0.088152` | `-0.090735` | `-0.002584` |
| Full since first bar | `0.5127` | `0.2623` | `-0.2504` | `1588.1326` | `1638.4255` | `+50.2930` | `-0.131208` | `-0.129681` | `+0.001527` |

### Cycle 3 conclusion

- 구현 관점에서는 `fractal genome structure`, `raw indicator modularization`, `automatic LLM review path`, `deep-tree final selection` 까지는 실제로 닫혔다.
- 검증 관점에서는 `mock-openai path`, `self-check`, `BTC-only backtest`, `baseline comparison` 까지 끝냈다.
- 하지만 성과 관점에서는 아직 `baseline을 명확히 압도`하지 못했다.
- 현재 최선 결과는:
  - `selected structural winner`: 구조는 깊어졌지만 수익/승률 우위는 불충분
  - `fallback best`: `6개월`과 `전체` 수익률은 baseline 이상, 하지만 `승률`은 전 구간에서 낮고 `2개월` 수익률도 낮다
- 따라서 현재 판정은 `구현 완성도는 크게 올라갔고, 검증도 끝났지만, 승격 가능한 전략 우위는 아직 미달`이다.

## Cycle 5 Robustness-First Search

이번 사이클은 `baseline dominance` 자체보다 먼저 `robustness failure`를 search 단계에서 미리 걸러내도록 구조를 바꿨다.

- 구현자 트랙:
  - main search에 `walk-forward fold robustness` 점수 추가
  - `commission stress` 기반 penalty 추가
  - generation survivor selection에 `target depth persistence` 추가
- 검증자 트랙:
  - verifier fold pass를 `return + MDD + daily win rate` 기준으로 강화
  - `stress_survival_threshold = 0.67` 기준 도입
  - monotonic curriculum에서 얕은 구조의 가짜 복잡도 증가를 막도록 stage pass를 더 엄격하게 조정

### Cycle 5 code verification

- `py_compile` 통과
- `verify_fractal_genome_cells.py --mode self-check` 통과
- `verify_fractal_genome_openai_mock.py` 통과

즉, `LeafGene`, `auto-review`, `robustness fold`, `stress penalty`를 넣은 뒤에도 기본 코드 경로는 깨지지 않았다.

### Cycle 5 main search result

- Search command family: `scripts/search_pair_subset_fractal_genome.py`
- Config:
  - `population=32`
  - `generations=5`
  - `max_depth=4`
  - `logic_max_depth=3`
  - `robustness_folds=3`
  - `robustness_test_months=2`
  - `commission_stress=1.0,1.5,2.0`
  - `stress_survival_threshold=0.67`
- Runtime:
  - `evaluated_unique_candidates=143`
  - `total_seconds=3.1373`
- Selection:
  - `selection.reason = no_gate_pass_near_frontier_structural`
  - `robustness_gate_pass_count = 0`
  - `winner_tree_depth = 3`
  - `winner_logic_depth = 3`
  - `winner_leaf_cardinality = 4`

핵심은 이렇다.

- search는 이제 `깊은 구조 + 높은 수익률` 후보를 고를 수 있다
- 하지만 `robustness gate`를 통과한 후보는 여전히 `0개`다
- 즉, 성능은 커졌지만 강건성은 아직 부족하다

### Cycle 5 main selected candidate vs embedded baseline

주의:

- 아래 비교는 search report 안의 `selected_candidate`와 `baseline_candidate`를 그대로 비교한 것이다
- baseline summary는 `2026-04-06` 기준이고 candidate는 `2026-04-09` 기준이라 완전 동일 끝점 비교는 아니다
- 따라서 진짜 승격 판정은 아래 verifier walk-forward를 기준으로 본다

| Window | Candidate Return | Baseline Return | Candidate Win Rate | Baseline Win Rate | Candidate MDD | Baseline MDD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Recent 2 months | `0.4270` | `0.4600` | `0.2333` | `0.3559` | `-0.099144` | `-0.088028` |
| Recent 6 months | `3.2838` | `2.9868` | `0.3279` | `0.4780` | `-0.099144` | `-0.088152` |
| Full history | `2833.9765` | `1588.1326` | `0.2765` | `0.5127` | `-0.133927` | `-0.131208` |

읽는 법:

- 수익률은 `total_return`
- 승률은 `daily_win_rate`
- 이번 cycle의 selected candidate는 `6개월`과 `전체` 누적 수익률은 baseline보다 높다
- 하지만 `승률`은 전 구간에서 낮고, `MDD`도 전 구간에서 더 나쁘다

즉, 이번 cycle은 `더 많이 버는 후보`를 찾았지만 `더 안정적인 후보`를 찾은 것은 아니다.

### Cycle 5 search robustness summary

search 내부 robustness fold 요약은 아래와 같다.

| Metric | Value |
| --- | ---: |
| Fold pass rate | `0.0000` |
| Stress survival rate mean | `0.0000` |
| Worst fold delta total return | `-0.0221` |
| Worst fold delta MDD | `-0.0109` |
| Latest fold delta total return | `-0.0221` |
| Gate passed | `false` |

해석:

- `wf_2`와 `wf_3`에서는 수익률 우위가 나왔다
- 하지만 `wf_1` 최신 구간에서 수익률과 MDD가 모두 baseline보다 밀렸다
- 그리고 수수료를 올리면 fold 우위가 바로 무너져 `stress_survival_rate_mean=0.0`으로 남았다

### Cycle 5 verifier result

- Verifier command family: `scripts/verify_fractal_genome_btc_curriculum.py`
- Outcome:
  - `selection.reason = latest_monotonic_stage_pass`
  - `selection.stage = depth_1`
  - `best_stage = depth_1`
  - `curriculum_passed = false`

Cycle 4에서는 verifier가 `depth_2`까지 갔지만, 이번 cycle에서는 검증 기준을 더 엄격하게 만들면서 다시 `depth_1`로 내려왔다.
이건 퇴보라기보다, 이전 verifier가 허용하던 `얕지만 넓은 구조`를 이번에는 막았다는 뜻에 가깝다.

#### Walk-forward robustness

| Metric | Value |
| --- | ---: |
| Fold pass rate | `0.1667` |
| Stress survival rate mean | `0.2222` |
| Stress survival threshold | `0.67` |
| Worst fold delta total return | `-0.0797` |
| Worst fold delta MDD | `-0.0016` |
| Worst fold delta daily win rate | `-0.0508` |
| Promotion gate passed | `false` |

폴드 해석:

- 통과한 fold는 사실상 `wf_5` 하나뿐이다
- `wf_6`는 수익률은 이겼지만 MDD가 아주 미세하게 더 깊어 탈락
- `wf_4`는 수익률이 소폭 밀려 탈락
- `wf_3`, `wf_2`, `wf_1`은 수익률 또는 win rate에서 baseline보다 열세
- stress survival도 대부분 `0.0`이라 수수료 민감도가 매우 크다

### Cycle 5 conclusion

- 엔진은 더 좋아졌다:
  - robustness fold
  - stress-aware score
  - depth persistence
  - stricter verifier
- 결과도 한쪽으로는 진전이 있다:
  - main search selected candidate의 `6개월 / 전체 수익률`은 baseline보다 높다
- 하지만 승격 관점에서는 아직 더 나빠진 부분이 분명하다:
  - `robustness_gate_pass_count = 0`
  - verifier `best_stage = depth_1`
  - `promotion_gate_passed = false`
  - 승률과 MDD는 아직 baseline을 못 이긴다

현재 병목은 분명하다.

- `최신 fold(wf_1)` 방어
- `stress survival`
- `수익률 우위가 win-rate/MDD 열세로 번지는 문제`

즉, 이번 cycle은 `성능 좋은 깊은 후보를 찾는 엔진`으로는 진전했지만, 아직 `baseline을 안정적으로 압도하는 승격 후보`까지는 도달하지 못했다.
