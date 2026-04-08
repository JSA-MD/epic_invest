# Epic Invest 프로젝트 분석 보고서

**작성일:** 2026-03-09
**프로젝트:** epic_invest - GP 기반 암호화폐 자동 매매 전략 연구 플랫폼

---

## 1. 프로젝트 개요

유전 프로그래밍(Genetic Programming)을 활용하여 수학적 수식을 자동 진화시키고, BTC 선물 트레이딩 시그널을 생성하는 연구 플랫폼입니다.

### 기술 스택
- **Python 3.12.12**
- **DEAP** - 유전 프로그래밍 프레임워크
- **Streamlit** - 인터랙티브 웹 대시보드
- **NumPy/Pandas** - 데이터 처리 및 벡터화 백테스트
- **Binance Futures API** - 실시간 시장 데이터
- **Dill** - 모델 직렬화

---

## 2. 프로젝트 구조

```
/Users/jsa/work/epic_invest/
├── app.py                           # Streamlit 웹 대시보드 (메인 UI)
├── liquidation_monitor.py           # 실시간 청산 추적 유틸리티
├── scripts/
│   ├── gp_crypto_evolution.py      # 핵심 GP 진화 엔진 (497줄)
│   ├── gp_crypto_infer.py          # 모델 추론 & 분석 CLI
│   └── backtest_short_vol_breakout.py  # 비교용 변동성 돌파 전략 (686줄)
├── data/
│   ├── binance_futures/            # 1시간봉 OHLCV 데이터
│   │   ├── BTCUSDT_1h_2022-07-01_2025-03-01.csv    (1.5MB)
│   │   ├── ETHUSDT_1h_2022-07-01_2025-03-01.csv    (1.5MB)
│   │   ├── SOLUSDT_1h_2022-07-01_2025-03-01.csv    (1.4MB)
│   │   └── XRPUSDT_1h_2022-07-01_2025-03-01.csv    (1.4MB)
│   ├── binanceus/                  # Binance US 스팟 데이터
│   │   ├── daily/                  # 10개 자산 (ADA, AVAX, BNB, BTC, DOGE, ETH, LINK, SOL, TRX, XRP)
│   │   └── hourly/                 # (비어있음)
│   └── coinbase/                   # Coinbase 스팟 데이터
│       ├── daily/                  # 8개 자산 (ADA, AVAX, BTC, DOGE, ETH, LINK, SOL, XRP)
│       └── hourly/                 # (비어있음)
├── models/
│   └── best_crypto_gp.dill         # 저장된 최적 진화 모델 (5.7KB)
├── docs/
│   └── gp_parameters_guide.md      # 파라미터 튜닝 가이드
├── results/                         # 결과 출력 디렉토리 (비어있음)
└── monitor_output.log              # 청산 모니터 출력 로그
```

---

## 3. 핵심 모듈 상세

### A. app.py - Streamlit 대시보드 (메인 UI)

인터랙티브 웹 인터페이스로 GP 트레이딩 전략의 전체 워크플로우를 제공합니다.

**4개 탭 구성:**
1. **🔬 Evolution** - GP 진화 실행 (파라미터 조정 가능)
2. **📊 Backtest** - 저장된 모델 백테스트 + Walk-forward 분석
3. **📡 Live Signal** - 실시간 트레이딩 시그널 생성
4. **📖 Guide** - 파라미터 가이드 문서

**프리셋 설정:**
| 프리셋 | 인구수 | 세대수 | 예상 소요 시간 |
|--------|--------|--------|---------------|
| Quick Test | 500 | 5 | ~1분 |
| Balanced | 2,000 | 15 | ~10분 |
| Deep Search | 5,000 | 25 | ~1시간 |
| Paper-grade | 7,500 | 30 | ~2-3시간 |

**실행:** `streamlit run app.py`

---

### B. scripts/gp_crypto_evolution.py - 핵심 진화 엔진

GP 프레임워크를 이용한 트레이딩 전략 진화 구현.

**주요 구성요소:**

1. **설정** (lines 32-67)
   - 트레이딩 페어: BTC, ETH, SOL, XRP (4페어 = 16 OHLC 피처)
   - 주 거래 종목: BTCUSDT
   - 기본 파라미터: pop=2000, gen=15, depth=8, maxlen=60

2. **데이터 파이프라인** (lines 72-160)
   - `fetch_klines()` - Binance Futures API 페이지네이션 호출
   - `load_pair()` - CSV 캐싱 메커니즘
   - `load_all_pairs()` - 4개 페어 병합

3. **벡터화 백테스터** (lines 177-245)
   - NumPy 기반 고속 백테스트 (외부 라이브러리 불필요)
   - Dead-band 필터 (기본 10pp)로 과도한 거래 방지
   - 수수료 모델링 (커미션 + 왕복 2배)
   - 출력 지표: total_return, sharpe, max_drawdown, n_trades, final_equity

4. **GP Primitives** (lines 251-295)
   - 연산자: add, sub, mul, pdiv (보호 나눗셈)
   - 함수: sin, cos, tanh, gt (비교→±100), neg, abs
   - 터미널: 16개 입력 피처 + 랜덤 상수 (-1~1)

5. **적합도 평가** (lines 300-322)
   - 최소화: exp(-total_return)
   - 페널티: 낮은 거래 횟수(<20), 음수 자산, NaN 값

6. **진화 루프** (lines 331-395)
   - 토너먼트 선택 (size=3)
   - 단일점 교차 (90%)
   - 균일 돌연변이 (15%)
   - Hall of Fame: 상위 10개체 추적

7. **메인 워크플로우** (lines 439-497)
   - Phase 1: 데이터 로딩
   - Phase 2: 학습 데이터 GP 진화
   - Phase 3: 검증 세트 최적 선택
   - Phase 4: Out-of-Sample 테스트
   - Phase 5: 전체 기간 백테스트
   - Phase 6: 모델 저장 (dill + JSON 메타데이터)

---

### C. scripts/gp_crypto_infer.py - 모델 추론 & 분석 CLI

학습된 모델을 로드하여 분석 수행 (재진화 없이).

**CLI 명령어:**
```bash
python scripts/gp_crypto_infer.py backtest --start 2024-07-01 --end 2025-03-01
python scripts/gp_crypto_infer.py walkforward --window 3 --step 1
python scripts/gp_crypto_infer.py signal
```

**핵심 클래스:**
- `GPModelManager` - dill 모델 로드/시그널 생성/벡터화 시그널
- `cmd_backtest()` - 임의 기간 백테스트
- `cmd_walkforward()` - 롤링 윈도우 분석
- `cmd_signal()` - 최근 48시간 실시간 시그널

---

### D. scripts/backtest_short_vol_breakout.py - 대안 전략

GP가 아닌 규칙 기반 평균 회귀 변동성 돌파 전략 (686줄). 비교 참조용.

- 트렌드 팔로잉 + 변동성 밴드
- 트레일링 스탑
- 포지션 사이징 (자본의 10%, 3배 레버리지)
- 수수료/슬리피지 모델링

---

### E. liquidation_monitor.py - 실시간 청산 레이더

Binance Futures WebSocket을 통한 실시간 청산 이벤트 스트리밍.

- 접속: `wss://fstream.binance.com/ws/!forceOrder@arr`
- 표시: 타임스탬프, 심볼, 포지션 유형, USD 가치, 체결 가격
- 컬러 출력: 빨강(LONG 청산), 초록(SHORT 청산)

---

## 4. 데이터 설정

### 학습/검증/테스트 기간

| 구분 | 시작 | 종료 | 기간 | 용도 |
|------|------|------|------|------|
| Training | 2022-07-01 | 2024-01-01 | 18개월 | GP 진화 학습 |
| Validation | 2024-01-01 | 2024-07-01 | 6개월 | 과적합 방지 |
| Test (OOS) | 2024-07-01 | 2025-03-01 | 8개월 | 실전 검증 |

### 시장 데이터

**Binance Futures (1시간봉, 2022.07~2025.03):**
- BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT (각 ~24,000봉)
- 형식: CSV (open_time, open, high, low, close, volume)

**Binance US / Coinbase Spot:**
- 일봉 데이터 (JSON 형식)
- 시간봉 디렉토리 존재하나 비어있음

---

## 5. 진화 파라미터

| 파라미터 | 기본값 | 범위 | 영향 |
|----------|--------|------|------|
| POP_SIZE | 2,000 | 100~10,000 | 탐색 범위 (높을수록 느림) |
| N_GEN | 15 | 1~50 | 수렴 깊이 (과적합 위험 증가) |
| P_CX | 0.90 | 0.5~1.0 | 교차 확률 |
| P_MUT | 0.15 | 0.01~0.50 | 돌연변이 확률 |
| MAX_DEPTH | 8 | 3~12 | 트리 복잡도 상한 |
| MAX_LEN | 60 | 10~120 | 트리 크기 제한 |

### 트레이딩 파라미터

| 파라미터 | 기본값 | 용도 |
|----------|--------|------|
| INITIAL_CASH | $100,000 | 시작 자본 |
| COMMISSION_PCT | 0.04% | 테이커 수수료 (편도) |
| NO_TRADE_BAND | 10pp | 리밸런싱 데드밴드 |
| TIMEFRAME | 1h | 캔들 간격 |

---

## 6. 아키텍처

```
[시장 데이터] → [로드 & 전처리] → [Train/Val/Test 분할]
                                          ↓
                                [GP 진화 루프 (Training)]
                                          ↓
                                [Hall of Fame (Top 10)]
                                          ↓
                                [검증 세트 선택 (과적합 방지)]
                                          ↓
                                [Out-of-Sample 테스트]
                                          ↓
                                [모델 저장 (Dill)]
                                          ↓
                    [추론: 백테스트 / Walk-Forward / 라이브 시그널]
```

### 핵심 설계 패턴
1. **벡터화 백테스터** - NumPy 배열 기반 100배+ 속도 향상
2. **Dead-Band 리밸런싱** - 임계값 필터링으로 과도한 거래 방지
3. **3-세트 분할** - Train→Validate→Test 과적합 방지
4. **Hall of Fame** - 엘리티즘으로 세대 간 최적 개체 보존
5. **보호 연산** - pdiv로 0 나눗셈 방지, NaN 처리
6. **모듈화된 추론** - GPModelManager로 모델 로딩과 평가 분리

---

## 7. 실행 방법

### 웹 대시보드 (인터랙티브)
```bash
streamlit run app.py
```

### CLI (배치 처리)
```bash
python scripts/gp_crypto_evolution.py                              # 전체 진화 사이클
python scripts/gp_crypto_infer.py backtest --start DATE --end DATE # 백테스트
python scripts/gp_crypto_infer.py walkforward --window 3 --step 1  # Walk-forward
python scripts/gp_crypto_infer.py signal                           # 최신 시그널
python liquidation_monitor.py                                       # 실시간 청산 모니터
```

---

## 8. 의존성

### 핵심 라이브러리
| 패키지 | 버전 | 용도 |
|--------|------|------|
| deap | 1.4.3 | 유전 프로그래밍 프레임워크 |
| numpy | 2.4.2 | 벡터화 연산 |
| pandas | 2.3.3 | 데이터 처리 |
| streamlit | 1.55.0 | 웹 대시보드 |
| dill | 0.4.1 | 모델 직렬화 |
| requests | 2.32.5 | API 호출 |
| websockets | - | 청산 모니터 WebSocket |

**환경:** Python 3.12.12, venv (.venv/)

---

## 9. 강점

- ✅ 완전한 진화 파이프라인 (데이터→진화→검증→테스트)
- ✅ 인터랙티브 Streamlit 대시보드
- ✅ 벡터화 백테스터로 고속 시뮬레이션
- ✅ 멀티 에셋 시그널 생성 (4개 암호화폐)
- ✅ 실시간 시장 모니터링 (청산 레이더)
- ✅ 3-세트 검증으로 과적합 방지
- ✅ 모델 저장 및 추론 프레임워크
- ✅ 파라미터 가이드 문서화

---

## 10. 개선 제안

| 항목 | 현황 | 제안 |
|------|------|------|
| 의존성 관리 | 파일 없음 | `requirements.txt` 또는 `pyproject.toml` 추가 |
| 테스트 코드 | 없음 | unit test 추가 (핵심 백테스트/피트니스 로직) |
| 빈 디렉토리 | hourly 데이터 비어있음 | 활용 또는 정리 |
| 결과 저장 | results/ 미활용 | 체계적 실험 결과 저장 체계 구축 |
| 로깅 | print 기반 | logging 모듈 도입 |
| 설정 관리 | 코드 내 하드코딩 | config 파일 분리 (YAML/TOML) |

---

## 11. 연동된 Skills 현황

### OMC (oh-my-claudecode) - 33개

| 카테고리 | Skill | 설명 |
|----------|-------|------|
| **워크플로우** | `autopilot` | 아이디어→코드 완전 자율 실행 |
| | `ralph` | 완료까지 반복 루프 |
| | `ultrawork` | 최대 병렬 에이전트 실행 |
| | `team` | N개 에이전트 협업 |
| | `pipeline` | 에이전트 순차 체이닝 |
| | `ultraqa` | QA 반복 (테스트→수정→반복) |
| | `plan` | 전략 계획 수립 |
| | `ralplan` | 합의 기반 계획 |
| | `sciomc` | 병렬 사이언티스트 분석 |
| | `deepinit` | 코드베이스 초기화 + AGENTS.md |
| **에이전트 단축** | `analyze` | 디버깅/조사 |
| | `deepsearch` | 코드 탐색 |
| | `tdd` | TDD 가이드 |
| | `build-fix` | 빌드 오류 수정 |
| | `code-review` | 코드 리뷰 |
| | `security-review` | 보안 리뷰 |
| | `frontend-ui-ux` | UI/UX 디자인 |
| | `git-master` | Git 관리 |
| **유틸리티** | `note` | 세션 메모 |
| | `learner` | 대화에서 스킬 추출 |
| | `skill` | 스킬 관리 CLI |
| | `hud` | HUD 설정 |
| | `trace` | 에이전트 추적 |
| | `doctor` | 설치 진단 |
| | `help` | 도움말 |
| | `cancel` | 실행 모드 취소 |
| **MCP 위임** | `ask-codex` | GPT 기반 분석 |
| | `ask-gemini` | Gemini 기반 분석 |
| | `ccg` | 3모델 합성 (Claude+Codex+Gemini) |

### bkit (Vibecoding Kit) - 21개

| 카테고리 | Skill | 설명 |
|----------|-------|------|
| **PDCA 사이클** | `/pdca plan` | 계획 수립 |
| | `/pdca design` | 설계 문서 작성 |
| | `/pdca do` | 구현 실행 |
| | `/pdca analyze` | Gap 분석 |
| | `/pdca iterate` | 자동 개선 반복 |
| | `/pdca report` | 완료 보고서 |
| | `/pdca status` | 현재 상태 확인 |
| | `/pdca next` | 다음 단계 안내 |
| **프로젝트 레벨** | `/starter` | 정적 웹 (초보자) |
| | `/dynamic` | 풀스택 (중급) |
| | `/enterprise` | 마이크로서비스 (고급) |
| **개발 파이프라인** | `/development-pipeline` | 9단계 개발 가이드 |
| **Phase 가이드** | `/phase-1-schema` ~ `/phase-9-deployment` | 단계별 상세 가이드 |
| **유틸리티** | `/code-review` | 코드 리뷰 |
| | `/zero-script-qa` | 스크립트 없는 QA |
| | `/plan-plus` | 브레인스토밍 강화 계획 |
| | `/mobile-app` | 모바일 앱 가이드 |
| | `/desktop-app` | 데스크탑 앱 가이드 |

### 외부 플러그인 Skills - 17개

| 플러그인 | Skill | 설명 |
|----------|-------|------|
| **superpowers** | `brainstorming` | 창작 전 브레인스토밍 |
| | `writing-plans` | 구현 계획 작성 |
| | `executing-plans` | 계획 실행 |
| | `test-driven-development` | TDD 워크플로우 |
| | `systematic-debugging` | 체계적 디버깅 |
| | `verification-before-completion` | 완료 전 검증 |
| | `dispatching-parallel-agents` | 병렬 에이전트 디스패치 |
| | `requesting-code-review` | 코드 리뷰 요청 |
| | `receiving-code-review` | 코드 리뷰 수신 |
| | `using-git-worktrees` | Git worktree 활용 |
| **commit-commands** | `commit` | Git 커밋 |
| | `commit-push-pr` | 커밋+푸시+PR |
| | `clean_gone` | 삭제된 브랜치 정리 |
| **pr-review-toolkit** | `review-pr` | PR 종합 리뷰 |
| **coderabbit** | `code-review` | AI 코드 리뷰 |
| **claude-mem** | `make-plan`, `do`, `mem-search` | 메모리 기반 계획/실행/검색 |
| **feature-dev** | `feature-dev` | 기능 개발 가이드 |
| **vercel** | `deploy`, `setup`, `logs` | Vercel 배포/설정/로그 |

### 커스텀 스킬

| 범위 | 경로 | 수량 |
|------|------|------|
| User-level | `~/.claude/skills/omc-learned/` | 0개 (비어있음) |
| Project-level | `.omc/skills/` | 없음 |

> **Tip:** `/skill add`로 프로젝트 전용 스킬 생성, `/learner`로 대화에서 패턴 추출 가능

---

*이 보고서는 Claude Code에 의해 자동 생성되었습니다.*
