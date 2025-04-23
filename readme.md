# TQQQ/SQQQ 스위칭 전략 개발 및 최적화 프로젝트

이 프로젝트는 머신러닝 모델을 기반으로 하여 ProShares UltraPro QQQ ETF (TQQQ)와 ProShares UltraPro Short QQQ ETF (SQQQ) 간의 스위칭(전환) 트레이딩 전략을 개발하고, 해당 전략의 파라미터를 최적화하며, 성과를 백테스팅 및 분석하는 것을 목표로 합니다.

## 프로젝트 개요

시장 상황에 따라 TQQQ(상승 예상 시)와 SQQQ(하락 또는 고변동성 예상 시) 포지션 간의 전환을 목표로 하는 전략입니다. 단순히 TQQQ를 계속 보유하는 'Buy & Hold' 전략 대비 높은 위험 조정 수익률을 달성하고자 합니다.

## 주요 기능 및 워크플로우

1.  **데이터 수집 (`collect_data.py`)**:
    *   `yfinance` 라이브러리를 사용하여 TQQQ, SQQQ, VIX(변동성 지수), UUP(달러 인덱스), ^IXIC(나스닥 종합), QQQ의 과거 시세 데이터(개장가, 고가, 저가, 종가, 거래량)를 다운로드합니다.
    *   수집된 데이터는 `data/` 디렉토리에 각 티커별 CSV 파일로 저장됩니다. (`TQQQ.csv`, `SQQQ.csv` 등)
    *   기본 수집 기간은 2015년부터 2025년까지 설정되어 있습니다.

2.  **데이터 검증 (`check_data.py`)**:
    *   `data/` 디렉토리에 필요한 CSV 파일들이 제대로 다운로드되었는지 확인합니다.
    *   각 파일의 컬럼 구조, 데이터 개수, 상위 5개 행을 출력하여 기본적인 데이터 상태를 점검합니다.

3.  **데이터 분석 및 특성 공학 (`analyze_data.py`)**:
    *   `data/` 디렉토리에서 수집된 CSV 파일들을 로드하여 Pandas DataFrame으로 변환합니다. ('Date' 컬럼을 인덱스로 사용)
    *   TQQQ 데이터를 기반으로 다양한 기술적 지표를 계산하여 특성(Feature)으로 추가합니다:
        *   이동평균선 (MA5, MA10, MA20, MA50, MA200)
        *   볼린저 밴드 (BB_middle, BB_upper, BB_lower, BB_std)
        *   RSI (Relative Strength Index)
        *   MACD (Moving Average Convergence Divergence), MACD Signal, MACD Histogram
        *   이동평균선 교차 신호 (MA5 vs MA20)
        *   볼린저 밴드 스퀴즈 신호
    *   VIX 데이터 및 TQQQ 가격 데이터를 이용하여 변동성 관련 특성을 추가합니다:
        *   VIX 지수, VIX 이동평균(10일), VIX 비율 (VIX / VIX_MA10), VIX 급등 신호
        *   TQQQ 일일 가격 변동률, 절대 변동률, 가격 급등 신호
    *   자산 간 상관관계(종가, 수익률 기준)를 분석하고 히트맵으로 시각화하여 `data/` 디렉토리에 저장합니다. (`correlation_heatmap.png`, `returns_correlation_heatmap.png`)
    *   모델 학습에 사용될 최종 특성 데이터셋(`tqqq_features.csv`)을 생성하여 `data/` 디렉토리에 저장합니다. 이 데이터셋에는 위에서 생성한 기술적 지표, 변동성 지표 및 모델 예측을 위한 타겟 변수(예: 다음 날 방향, 다음 날 급등락 여부, SQQQ 전환 여부)가 포함됩니다.

4.  **머신러닝 모델 개발 (`develop_model.py`)**:
    *   `analyze_data.py`에서 생성된 특성 데이터셋 (`tqqq_features.csv`)을 로드합니다.
    *   데이터를 전처리합니다: 결측치 제거, 시계열 데이터 분할(확장 윈도우 방식: 최근 1년을 테스트셋으로 사용), 특성 스케일링(`StandardScaler` 사용, 스케일러는 `models/scaler.pkl`로 저장).
    *   세 가지 종류의 예측 모델을 학습시킵니다:
        *   **방향 예측 모델**: 다음 날 TQQQ 가격 방향(상승/하락) 예측 (`RandomForestClassifier`, `models/direction_model.pkl`로 저장)
        *   **급등락 예측 모델**: 다음 날 TQQQ 가격 급등락(큰 변동성) 여부 예측 (`GradientBoostingClassifier`, `models/big_move_model.pkl`로 저장)
        *   **SQQQ 전환 신호 예측 모델**: SQQQ로 전환해야 할지 여부 예측 (`GradientBoostingClassifier`, `models/switch_model.pkl`로 저장)
    *   각 모델의 성능(정확도, 정밀도, 재현율, F1 점수)을 평가하고, 혼동 행렬 및 특성 중요도 그래프를 시각화하여 `models/` 디렉토리에 저장합니다.
    *   학습된 모델 정보와 스케일러 등을 포함하는 `model_data` 객체를 생성하여 반환합니다.

5.  **전략 구현 및 백테스팅 함수 (`implement_strategy.py`)**:
    *   `detect_volatility`: 변동성 감지 로직 (VIX, 가격 변동 기반).
    *   `implement_switching_strategy`: 모델 예측 확률과 파라미터(임계값 등)를 기반으로 최종 매매 신호(1: TQQQ, 0: 현금, -1: SQQQ)를 생성하는 핵심 로직. 최소 보유 기간, 수수료 고려 로직 포함.
    *   `backtest_strategy`: 주어진 매매 신호에 따라 표준 백테스팅을 수행하고 성과 지표(누적 수익률, 연간 수익률, 변동성, 샤프 비율, MDD 등)를 계산하는 함수.
    *   `backtest_strategy_rolling`: 롤링 윈도우 방식으로 백테스팅을 수행하는 함수 (`optimize_parameters.py`에서 사용).
    *   `visualize_results`: 백테스팅 결과를 시각화하는 함수.

6.  **파라미터 최적화 (`optimize_parameters.py`)**:
    *   `Optuna` 라이브러리를 사용하여 `implement_strategy.py`의 전략 함수(`implement_switching_strategy`)에 사용되는 하이퍼파라미터(각 모델 예측 확률 임계값, 변동성 감지 임계값, 최소 보유 기간 등)의 최적 조합을 찾습니다.
    *   최적화 목표 함수(`objective`)는 주어진 하이퍼파라미터 조합으로 **롤링 윈도우 백테스팅**(`backtest_strategy_rolling`)을 수행하고, 그 결과(예: 샤프 비율, 총수익률, 거래 횟수 페널티 적용 값)를 반환합니다.
    *   Optuna는 이 목표 함수 값을 최대화(또는 최소화)하는 파라미터 조합을 탐색합니다.
    *   최적화 과정은 `cache/` 디렉토리에 저장될 수 있습니다.
    *   최종적으로 찾은 최적 파라미터 조합은 `models/final_optimized_params_*.json` 파일로 저장됩니다.
    *   최적 파라미터를 사용한 롤링 백테스팅의 상세 결과(일별 데이터)는 `results/optimal_rolling_trade_log*.csv` 파일로 저장됩니다.

7.  **전략 개선 (`refine_strategy.py`)**:
    *   `optimize_parameters.py`에서 찾은 최적 파라미터를 기반으로 전략 로직을 추가 개선합니다.
    *   **개선된 변동성 감지**: 단기 추세(상승/하락)를 감지하고, '고변동성+하락추세' 또는 '고변동성 종료+상승추세' 같은 강력한 신호를 정의하여 더 빠른 시장 대응을 시도합니다.
    *   **개선된 전략 구현**: 강력한 신호 발생 시 최소 보유 기간을 무시하고 즉시 전환하며, 월별 최대 거래 횟수를 제한하여 과도한 매매를 방지하는 로직을 추가합니다.
    *   개선된 전략의 백테스팅 및 결과 시각화 기능을 포함합니다. (결과는 `results_improved/` 등에 저장될 수 있음)

8.  **전략 비교 (`compare_strategy.py`)**:
    *   `optimize_parameters.py` 또는 `refine_strategy.py`를 통해 개발된 최종 스위칭 전략과 TQQQ Buy & Hold 전략의 성과를 비교합니다.
    *   표준 백테스팅(`backtest_strategy`)을 수행하여 두 전략의 주요 성과 지표(누적 수익률, 연간 수익률, 샤프 비율, MDD 등)를 계산합니다.
    *   결과를 시각화하여 두 전략의 자본 성장 곡선, Drawdown 등을 비교하고 `results/` 디렉토리에 저장합니다.

9.  **시뮬레이션 실행 (`simul.py`)**:
    *   사용자가 **특정 기간(시작일, 종료일)**을 지정하여 해당 기간 동안 최적화된 모델과 파라미터를 사용한 백테스팅 시뮬레이션을 실행합니다.
    *   표준 백테스팅(`backtest_strategy`)을 수행하고, 상세한 성과 지표를 `simulation.log` 파일과 콘솔에 출력합니다.
    *   다양한 시각화 결과(자본 성장 곡선, Drawdown, 매매 신호 등)를 생성하여 `results_simulation/` 디렉토리에 저장합니다.

10. **메인 실행 (`main.py`)**:
    *   전체 워크플로우(데이터 확인/수집 -> 데이터 분석 및 특성 공학 -> 모델 개발 -> 파라미터 최적화)를 순차적으로 실행하는 메인 스크립트입니다.
    *   각 단계의 로그는 `tqqq_model_development.log` 파일에 기록됩니다.
    *   최종 모델(`models/final_optimized_model.pkl`)과 최적 파라미터(`models/final_optimized_params_*.json`), 상세 최적화 로그(`results/optimal_rolling_trade_log*.csv`)를 저장합니다.

11. **기타 시뮬레이션 (`simul_random.py`, `simul_random_순차.py`)**:
    *   이름으로 미루어 볼 때, 랜덤 요소를 포함하거나 순차적인 방식으로 진행되는 다른 유형의 시뮬레이션 또는 전략 테스트를 위한 스크립트로 추정됩니다. (정확한 기능은 코드 상세 분석 필요)

## 디렉토리 구조

-   **`data/`**: 원본 시세 데이터(.csv), 분석 결과(상관관계 히트맵 .png), 최종 특성 데이터셋(.csv) 저장.
-   **`models/`**: 학습된 머신러닝 모델(.pkl), 특성 스케일러(.pkl), 최적화된 하이퍼파라미터(.json), 모델 평가 시각화(.png) 저장.
-   **`results/`**: 파라미터 최적화 시 생성된 상세 롤링 백테스팅 로그(.csv), 전략 비교 시각화(.png) 저장.
-   **`results_simulation/`**: `simul.py` 실행 시 생성된 특정 기간 시뮬레이션 결과 시각화(.png) 저장.
-   **`results_improved/`**: (존재한다면) `refine_strategy.py` 실행 시 생성된 개선된 전략 결과 저장.
-   **`cache/`**: (Optuna가 생성) 파라미터 최적화 과정 저장.
-   **`*.log`**: `main.py` (`tqqq_model_development.log`), `simul.py` (`simulation.log`) 등의 실행 로그 파일.

## 실행 방법

1.  **필수 라이브러리 설치**:
    ```bash
    pip install pandas numpy yfinance scikit-learn optuna joblib matplotlib seaborn
    ```
    (필요시 `requirements.txt` 파일 생성 및 `pip install -r requirements.txt` 실행)

2.  **전체 개발 워크플로우 실행 (데이터 수집부터 파라미터 최적화까지)**:
    ```bash
    python main.py
    ```
    *   실행 후 `tqqq_model_development.log` 로그 파일, `models/`, `results/` 디렉토리의 생성/업데이트된 파일을 확인합니다.

3.  **특정 기간 시뮬레이션 실행**:
    ```bash
    # 예시: 2022년 1월 1일부터 2023년 12월 31일까지 시뮬레이션
    python simul.py --start_date 2022-01-01 --end_date 2023-12-31
    ```
    *   실행 후 `simulation.log` 로그 파일과 `results_simulation/` 디렉토리의 시각화 파일을 확인합니다.

4.  **기타 스크립트 실행 (필요시)**:
    ```bash
    python check_data.py
    python compare_strategy.py
    python refine_strategy.py
    ```

## 주의사항

*   SQQQ 수익률은 백테스팅 시 TQQQ 수익률의 -0.9배로 근사하여 계산됩니다. 실제 SQQQ 데이터를 사용하면 더 정확한 백테스팅이 가능합니다.
*   백테스팅은 과거 데이터에 기반한 결과이며, 미래의 실제 투자 성과를 보장하지 않습니다.
*   머신러닝 모델과 파라미터는 사용된 데이터 기간에 과적합(Overfitting)될 수 있으므로, 실제 적용 시 주의가 필요합니다. 