# Fractal Genome Follow-up

이 문서는 프랙탈 게놈 구조 도입 이후, 아직 완전히 닫지 않은 장기 backlog를 다시 논의하고 구현하기 위한 메모다.

관련 코드:
- [fractal_genome_core.py](/Users/jsa/work/epic_invest/scripts/fractal_genome_core.py)
- [search_pair_subset_fractal_genome.py](/Users/jsa/work/epic_invest/scripts/search_pair_subset_fractal_genome.py)

## Deferred Backlog

현재 기준으로 장기 backlog에 남겨둔 2가지:

- [ ] 무한 재귀 생성
  현재는 bounded recursive tree + bounded logic cell depth만 허용한다. 무한 깊이 또는 사실상 무제한 성장 구조는 나중에 `depth curriculum`과 함께 다시 검토한다.
- [ ] 실 API 기반 온라인 LLM 운영 검증
  자동 LLM review 코드 경로와 OpenAI 호환 mock server 검증은 이미 붙었다. 다만 외부 API 키와 네트워크 조건을 둔 실제 운영 환경 검증은 아직 별도 단계로 남겨둔다.

## 2. 재귀적 결합

### 현재 상태
- `ConditionNode`, `LeafNode` 기반의 재귀 `If-Then-Else` 트리 구조는 이미 구현됨.
- 각 `ConditionNode`는 단일 조건 하나가 아니라, atomic threshold cell과 `AND` / `OR` / `NOT` 를 재귀적으로 결합한 `logic cell organ` 을 게이트로 사용함.
- `random_tree`, `mutate_tree`, `crossover_tree`, `evaluate_tree_codes`가 재귀 트리 생성/변이/교차/평가를 담당함.
- 현재는 `--max-depth`로 트리 깊이를 제한하는 `bounded recursive tree` 구조임.
- 조건식 내부도 `--logic-max-depth`로 제한되는 bounded recursive cell 구조임.

### 왜 무한 깊이를 허용하지 않았는가
- 탐색 공간 폭발: 후보 수가 너무 빠르게 증가함.
- 금융 과최적화 악화: 깊은 트리는 과거 구간 암기에 유리함.
- 무의미한 branch 급증: 자기모순, 동일 branch, 사실상 dead branch가 급격히 늘어남.
- 평가 비용 증가: 최근 2개월/6개월/4년/스트레스까지 함께 돌리는 현재 파이프라인에서 계산량이 커짐.

### 나중에 다시 논의할 확장 방향
1. `depth curriculum`
초기 세대는 `max_depth=2` 또는 `3`으로 시작하고, 후반 세대에서만 `4`, `5`까지 점진 확대.

2. `complexity-aware fitness`
깊이/노드 수 패널티를 넣되, 일정 성능 우위를 보이는 경우에만 더 복잡한 트리가 살아남도록 조정.

3. `pair-specific recursive trees`
공유 트리 하나가 BTC/BNB 둘 다를 라우팅하는 구조 대신, 코인별로 별도 재귀 트리를 갖게 하는 구조 검토.

4. `archive / novelty`
유사 트리끼리만 반복 생성되지 않도록 novelty 점수나 archive 기반 다양성 유지 장치 추가.

### 다시 구현할 때 바로 볼 TODO
- [ ] 세대별 `max_depth` 스케줄 추가
- [ ] `tree_size`, `tree_depth`, leaf concentration을 fitness에 더 정교하게 반영
- [ ] 코인별 개별 트리 구조 설계 여부 결정
- [ ] 공통 트리 vs pair-specific 트리의 장기 4년 성능 비교

## 3. 지능적 필터링

### 현재 상태
- `semantic_filter()` 인터페이스는 이미 구현됨.
- 현재 기본 동작은 `heuristic_fallback`.
- `llm_review_in`, `llm_review_out` 옵션을 통해 LLM 검토용 JSONL 큐를 내보내고, 사전 검토된 결과를 다시 읽어 필터에 반영할 수 있음.
- `auto` 모드에서 OpenAI 호환 endpoint로 review를 보내는 코드 경로와 mock-openai 검증 스크립트가 존재함.
- 다만 외부 실 API를 이용한 운영 검증은 아직 별도 단계로 남겨둠.

### 현재 heuristic이 걸러내는 것
- depth 초과
- 조건 없음
- 단일 leaf만 반복되는 구조
- 중복 조건만 있는 구조
- 동일 branch 구조
- leaf concentration 과도

### 왜 실 API 운영 검증을 아직 기본 경로로 두지 않았는가
- 현재 실행 환경은 네트워크/API 의존을 기본 경로로 두기 어려움.
- 탐색 중 매 후보마다 LLM을 부르면 비용과 지연이 매우 커짐.
- 재현성 문제: 같은 탐색을 다시 돌렸을 때 필터 결과가 달라질 수 있음.
- 그래서 현재는 heuristic과 mock-verified auto path를 유지하고, 실 API는 상위 후보나 별도 budget run에서만 검토하는 구조가 더 실용적임.

### 나중에 구현할 권장 순서
1. `top-N review queue`
탐색 종료 후 상위 후보만 `llm_review_out` JSONL로 추출.

2. `offline LLM review`
외부 LLM이 JSONL을 읽고 아래 형태로 판정:
- `tree_key`
- `accepted`
- `reason`
- 필요 시 `rewrite_suggestion`

3. `llm_review_in` 재주입
다음 탐색부터는 같은 `tree_key` 후보에 대해 LLM 판정을 우선 적용.

4. `llm-first` 또는 `llm-only` 모드 실험
다만 이 모드는 비용과 재현성 문제가 있으므로 기본값으로 두지 않음.

5. `semantic mutation`
추후에는 LLM이 자유 코드를 생성하는 것이 아니라,
- branch 제거
- 조건 단순화
- 의미상 중복 조건 삭제
- leaf 교체 제안
같은 제한형 수정만 수행하도록 설계.

### LLM 필터에 줄 권한 범위
- 허용:
  - 경제적 타당성 검토
  - 자기모순/중복/무의미 branch 탐지
  - 제한형 수정 제안
- 비허용:
  - 자유 코드 생성
  - 백테스트 엔진 로직 수정
  - unrestricted Python/C# 코드 생성

### 다시 구현할 때 바로 볼 TODO
- [ ] `llm_review_out` 포맷 고정
- [ ] `llm_review_in` 병합 로직에 confidence 필드 추가
- [ ] 상위 후보만 검토하도록 budget gate 추가
- [ ] heuristic reject와 llm reject를 별도 통계로 기록
- [ ] LLM 수정 제안을 실제 트리 변이 연산자로 연결할지 결정

## 구현 원칙
- 현재 활성 전략을 임의로 바꾸지 않는다.
- 새로운 구조는 먼저 별도 탐색기로 실험한다.
- 결과가 기준보다 나빠도 즉시 폐기하지 않고, 먼저 결과를 보여준 뒤 폐기 승인 후 정리한다.
- 실제 LLM 사용은 자유 코드 생성이 아니라 `제한형 DSL/AST 검토 및 수정`으로만 제한한다.
