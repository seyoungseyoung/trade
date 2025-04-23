import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import json
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import optuna
import warnings
import pickle
from pathlib import Path
import logging
# Import visualize_results
from implement_strategy import visualize_results 
warnings.filterwarnings('ignore')

# 로깅 설정 추가
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# 시각화 설정
plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 12
sns.set_palette('Set2')

# 모델 및 데이터 로드
def load_models_and_data():
    """학습된 모델과 테스트 데이터 로드"""
    print("모델 및 데이터 로드 중...")
    
    # 모델 로드
    models = {}
    try:
        models['direction_model'] = joblib.load('models/direction_model.pkl')
        models['big_move_model'] = joblib.load('models/big_move_model.pkl')
        models['switch_model'] = joblib.load('models/switch_model.pkl')
        models['scaler'] = joblib.load('models/scaler.pkl')
        print("모델 로드 완료")
    except Exception as e:
        print(f"모델 로드 실패: {str(e)}")
        return None, None, None
    
    # 테스트 결과 로드
    try:
        test_results = pd.read_csv('models/test_results.csv', index_col=0)
        test_results.index = pd.to_datetime(test_results.index)
        print(f"테스트 결과 로드 완료: {test_results.shape}")
    except Exception as e:
        print(f"테스트 결과 로드 실패: {str(e)}")
        test_results = None
    
    # 특성 데이터셋 로드
    try:
        features_df = pd.read_csv('data/tqqq_features.csv', index_col=0)
        features_df.index = pd.to_datetime(features_df.index)
        print(f"특성 데이터셋 로드 완료: {features_df.shape}")
    except Exception as e:
        print(f"특성 데이터셋 로드 실패: {str(e)}")
        features_df = None
    
    return models, test_results, features_df

# 변동성 감지 함수
def detect_volatility(df, vix_threshold, vix_ratio_threshold, price_change_threshold):
    """변동성 감지 함수"""
    if df is None:
        return None
    
    # 변동성 조건
    volatility_conditions = []
    
    # VIX 기반 변동성 감지
    if 'VIX' in df.columns:
        vix_high = df['VIX'] > vix_threshold
        volatility_conditions.append(vix_high)
    
    # VIX 비율 기반 변동성 감지
    if 'VIX_ratio' in df.columns:
        vix_ratio_high = df['VIX_ratio'] > vix_ratio_threshold
        volatility_conditions.append(vix_ratio_high)
    
    # 가격 변동성 기반 감지
    if 'price_change_abs' in df.columns:
        price_change_high = df['price_change_abs'] > price_change_threshold
        volatility_conditions.append(price_change_high)
    
    # 변동성 지표 통합
    if volatility_conditions:
        df['high_volatility'] = pd.concat(volatility_conditions, axis=1).any(axis=1).astype(int)
    else:
        df['high_volatility'] = 0
    
    # 변동성 기간 식별 (연속된 고변동성 날짜를 하나의 기간으로 그룹화)
    df['volatility_period_start'] = (df['high_volatility'].diff() == 1).astype(int)
    df['volatility_period_end'] = (df['high_volatility'].diff() == -1).shift(-1).fillna(0).astype(int)
    
    # 변동성 기간 시작 날짜 추출
    volatility_starts = df[df['volatility_period_start'] == 1].index
    
    return df

# SQQQ 전환 전략 구현
def implement_switching_strategy(df, models, direction_threshold, big_move_threshold, switch_threshold, fee_rate, min_holding_days):
    """TQQQ/SQQQ 전환 전략 구현"""
    if df is None or models is None:
        return None
    
    # 예측 확률 계산을 위한 특성 추출
    feature_cols = [col for col in df.columns if col not in [
        'next_day_return', 'target_direction', 'target_big_move', 'switch_to_sqqq',
        'high_volatility', 'volatility_period_start', 'volatility_period_end', 'Date'
    ]]
    
    # 스케일링에서 제외할 열
    exclude_cols = ['timestamp', 'MA_cross_up', 'MA_cross_down', 'VIX_spike', 'price_spike']
    scale_cols = [col for col in feature_cols if col not in exclude_cols and col in df.columns]
    
    # 특성 스케일링
    X = df[feature_cols].copy()
    
    # scale_cols가 비어있지 않은 경우에만 스케일링 수행
    if scale_cols:
        X[scale_cols] = models['scaler'].transform(X[scale_cols])
    
    # 모델 예측
    df = df.copy()  # 경고 방지를 위한 복사본 생성
    df['direction_prob'] = models['direction_model'].predict_proba(X)[:, 1]  # 상승 확률
    df['big_move_prob'] = models['big_move_model'].predict_proba(X)[:, 1]    # 급등락 확률
    df['switch_prob'] = models['switch_model'].predict_proba(X)[:, 1]        # SQQQ 전환 확률
    
    # 최적화된 임계값 적용
    df['pred_direction'] = (df['direction_prob'] > direction_threshold).astype(int)
    df['pred_big_move'] = (df['big_move_prob'] > big_move_threshold).astype(int)
    df['pred_switch'] = (df['switch_prob'] > switch_threshold).astype(int)
    
    # 기본 매매 신호 생성 (1: TQQQ, 0: 현금, -1: SQQQ)
    df['signal'] = 0
    
    # 기본 로직: 방향 예측이 상승이고 급등락 예측이 아니면 TQQQ 매수
    df.loc[(df['pred_direction'] == 1) & (df['pred_big_move'] == 0), 'signal'] = 1
    
    # SQQQ 전환 로직: 전환 신호가 있으면 SQQQ 매수
    df.loc[df['pred_switch'] == 1, 'signal'] = -1
    
    # 추가 규칙: 작은 하락에는 매매 수수료를 고려하여 포지션 유지
    # 이전 신호가 TQQQ 매수이고, 현재 신호가 현금 보유이며, 예상 하락폭이 수수료보다 작으면 TQQQ 유지
    for i in range(1, len(df)):
        if df['signal'].iloc[i-1] == 1 and df['signal'].iloc[i] == 0:
            expected_loss = df['direction_prob'].iloc[i] - 0.5  # 방향 확률에서 0.5를 뺀 값을 예상 손실로 사용
            if abs(expected_loss) < fee_rate:
                df.loc[df.index[i], 'signal'] = 1  # TQQQ 유지 (경고 방지를 위한 loc 사용)
    
    # 최소 보유 기간 적용 (불필요한 빈번한 매매 방지)
    holding_asset = 0  # 0: 현금, 1: TQQQ, -1: SQQQ
    holding_days = 0
    
    for i in range(len(df)):
        current_signal = df['signal'].iloc[i]
        
        # 포지션 변경 여부 확인
        if current_signal != holding_asset:
            # 최소 보유 기간 확인
            if holding_days >= min_holding_days or holding_asset == 0:
                # 포지션 변경
                holding_asset = current_signal
                holding_days = 1
            else:
                # 최소 보유 기간 미달, 기존 포지션 유지
                df.loc[df.index[i], 'signal'] = holding_asset  # 경고 방지를 위한 loc 사용
                holding_days += 1
        else:
            # 동일 포지션 유지
            holding_days += 1
    
    # 매매 신호 변경 감지
    df['signal_change'] = df['signal'].diff().fillna(0)
    
    return df

# 백테스팅 함수 (롤링 윈도우 평가 방식으로 수정 + 정보 비율 추가)
def backtest_strategy_rolling(df, initial_capital=10000, fee_rate=0.0001, window_years=1, step_months=6, risk_free_rate=0.02):
    """
    전략 백테스팅 (1년 롤링 윈도우, 6개월 스텝, 정보 비율 계산 포함)
    Args:
        df (pd.DataFrame): 백테스팅할 데이터 (signal 포함)
        initial_capital (float): 초기 자본금
        fee_rate (float): 매매 수수료율
        window_years (int): 롤링 윈도우 기간 (년)
        step_months (int): 롤링 윈도우 이동 간격 (월)
        risk_free_rate (float): 무위험 수익률 (연율화)
    Returns:
        tuple or None: (detailed_df, window_results_df, overall_performance) 튜플, 실패 시 None
                       window_results_df 에는 information_ratio 컬럼 추가됨
    """
    if df is None or df.empty or not isinstance(df.index, pd.DatetimeIndex):
        logger.warning("백테스팅 입력 데이터가 유효하지 않습니다.")
        return None

    df = df.copy()

    # 수익률 계산 (다음 날 수익률이 없는 경우 계산)
    if 'next_day_return' not in df.columns:
        if 'Close' in df.columns: # 'Close' 컬럼 사용 시도
            df['next_day_return'] = df['Close'].pct_change().shift(-1)
        elif 'TQQQ_close' in df.columns: # 'TQQQ_close' 컬럼 사용 시도
             df['next_day_return'] = df['TQQQ_close'].pct_change().shift(-1)
        else:
            logger.error("'Close' 또는 'TQQQ_close' 컬럼이 없어 next_day_return을 계산할 수 없습니다.")
            return None
        df['next_day_return'].fillna(0, inplace=True)

    # 전략 수익률 계산 (SQQQ 고려)
    df['strategy_daily_return_gross'] = 0.0
    df.loc[df['signal'] == 1, 'strategy_daily_return_gross'] = df.loc[df['signal'] == 1, 'next_day_return']
    df.loc[df['signal'] == -1, 'strategy_daily_return_gross'] = -0.9 * df.loc[df['signal'] == -1, 'next_day_return'] # SQQQ 수익률 근사

    # 수수료 적용
    df['fee'] = 0.0
    df['signal_change'] = df['signal'].diff().fillna(0) # signal_change 없으면 생성
    df.loc[df['signal_change'] != 0, 'fee'] = fee_rate
    df['strategy_daily_return_net'] = df['strategy_daily_return_gross'] - df['fee']

    # 누적 수익률 계산 (전체 기간)
    df['cum_tqqq_return'] = (1 + df['next_day_return'].fillna(0)).cumprod() - 1
    df['cum_strategy_return'] = (1 + df['strategy_daily_return_net'].fillna(0)).cumprod() - 1

    # 일별 초과 수익률 계산
    df['excess_return'] = df['strategy_daily_return_net'] - df['next_day_return']
    df['excess_return'].fillna(0, inplace=True)

    # 롤링 윈도우 설정
    start_date = df.index.min()
    end_date = df.index.max()
    window_size = pd.DateOffset(years=window_years)
    step_size = pd.DateOffset(months=step_months)

    window_results = []
    current_start = start_date

    while current_start + window_size <= end_date:
        current_end = current_start + window_size
        window_df = df.loc[current_start:current_end].copy()

        if window_df.empty or len(window_df) < 2:
            logger.debug(f"Skipping empty or too short window: {current_start.date()} to {current_end.date()}")
            current_start += step_size
            continue

        # 윈도우 내 성과 계산
        days = (window_df.index[-1] - window_df.index[0]).days
        if days <= 0:
            logger.debug(f"Skipping window with non-positive duration: {current_start.date()} to {current_end.date()}")
            current_start += step_size
            continue
        years = days / 365.0
        trading_days_in_window = len(window_df)
        annualization_factor = np.sqrt(252) # Assuming 252 trading days per year

        # 윈도우 내 기본 성과 계산
        tqqq_start_return = df.loc[:current_start, 'cum_tqqq_return'].iloc[-1] if current_start > df.index.min() else 0
        strategy_start_return = df.loc[:current_start, 'cum_strategy_return'].iloc[-1] if current_start > df.index.min() else 0
        tqqq_window_end_return = window_df['cum_tqqq_return'].iloc[-1]
        strategy_window_end_return = window_df['cum_strategy_return'].iloc[-1]
        tqqq_window_return = (1 + tqqq_window_end_return) / (1 + tqqq_start_return) - 1
        strategy_window_return = (1 + strategy_window_end_return) / (1 + strategy_start_return) - 1
        tqqq_annual_return = (1 + tqqq_window_return) ** (1 / years) - 1 if years > 0 else 0
        strategy_annual_return = (1 + strategy_window_return) ** (1 / years) - 1 if years > 0 else 0
        tqqq_volatility = window_df['next_day_return'].std() * annualization_factor if trading_days_in_window >= 2 else 0
        strategy_volatility = window_df['strategy_daily_return_net'].std() * annualization_factor if trading_days_in_window >= 2 else 0
        tqqq_sharpe = (tqqq_annual_return - risk_free_rate) / tqqq_volatility if tqqq_volatility > 0 else 0
        strategy_sharpe = (strategy_annual_return - risk_free_rate) / strategy_volatility if strategy_volatility > 0 else 0
        window_df['window_cum_tqqq'] = (1 + window_df['next_day_return'].fillna(0)).cumprod()
        window_df['window_cum_strategy'] = (1 + window_df['strategy_daily_return_net'].fillna(0)).cumprod()
        tqqq_running_max = window_df['window_cum_tqqq'].cummax()
        strategy_running_max = window_df['window_cum_strategy'].cummax()
        tqqq_drawdown = (window_df['window_cum_tqqq'] - tqqq_running_max) / tqqq_running_max
        strategy_drawdown = (window_df['window_cum_strategy'] - strategy_running_max) / strategy_running_max
        tqqq_max_drawdown = tqqq_drawdown.min() if not tqqq_drawdown.empty else 0
        strategy_max_drawdown = strategy_drawdown.min() if not strategy_drawdown.empty else 0
        trade_count = (window_df['signal_change'] != 0).sum()

        # 윈도우 내 정보 비율 계산
        avg_daily_excess_return = window_df['excess_return'].mean()
        annualized_avg_excess_return = avg_daily_excess_return * 252
        tracking_error = window_df['excess_return'].std() * annualization_factor if trading_days_in_window >= 2 else 0
        information_ratio = annualized_avg_excess_return / tracking_error if tracking_error > 1e-9 else 0
        if np.isinf(information_ratio) or pd.isna(information_ratio):
            information_ratio = 0

        # 결과 저장
        window_results.append({
            'window_start': current_start,
            'window_end': current_end,
            'days': days,
            'years': years,
            'tqqq_return': tqqq_window_return,
            'strategy_return': strategy_window_return,
            'tqqq_annual_return': tqqq_annual_return,
            'strategy_annual_return': strategy_annual_return,
            'tqqq_volatility': tqqq_volatility,
            'strategy_volatility': strategy_volatility,
            'tqqq_sharpe': tqqq_sharpe,
            'strategy_sharpe': strategy_sharpe,
            'tqqq_max_drawdown': tqqq_max_drawdown,
            'strategy_max_drawdown': strategy_max_drawdown,
            'trade_count': trade_count,
            'outperforms_tqqq': strategy_window_return > tqqq_window_return,
            'annualized_avg_excess_return': annualized_avg_excess_return,
            'tracking_error': tracking_error,
            'information_ratio': information_ratio
        })

        current_start += step_size

    if not window_results:
        logger.warning("No valid rolling windows found for backtesting.")
        return None

    results_df = pd.DataFrame(window_results)
    results_df.set_index('window_start', inplace=True)

    # 전체 기간 성과 요약 (참고용)
    overall_performance = {
        'test_period': f"{df.index.min().strftime('%Y-%m-%d')} ~ {df.index.max().strftime('%Y-%m-%d')}",
        'total_days': (df.index.max() - df.index.min()).days,
        'initial_capital': initial_capital,
        'final_tqqq_capital': initial_capital * (1 + df['cum_tqqq_return'].iloc[-1]),
        'final_strategy_capital': initial_capital * (1 + df['cum_strategy_return'].iloc[-1]),
        'tqqq_total_return': df['cum_tqqq_return'].iloc[-1],
        'strategy_total_return': df['cum_strategy_return'].iloc[-1],
        'total_trade_count': (df['signal_change'] != 0).sum(),
    }

    return df, results_df, overall_performance

# 목적 함수 (Optuna 최적화용 - 정보 비율 기반 + 학습 기간 최적화)
def objective(trial, features_df, models, initial_capital=10000, fee_rate=0.0001, trade_penalty_factor=0.01):
    """Optuna 최적화를 위한 목적 함수 (평균 정보 비율 최대화, 학습 기간[3-5년] 최적화)

    Args:
        trade_penalty_factor: 평균 거래 횟수에 대한 페널티 강도.
    """
    # --- 최적화할 파라미터 정의 (lookback_years 추가) ---
    lookback_years = trial.suggest_int('lookback_years', 3, 5)
    vix_threshold = trial.suggest_float('vix_threshold', 15.0, 35.0, step=0.05)
    vix_ratio_threshold = trial.suggest_float('vix_ratio_threshold', 1.0, 1.5, step=0.01)
    price_change_threshold = trial.suggest_float('price_change_threshold', 0.01, 0.1, step=0.001)
    direction_threshold = trial.suggest_float('direction_threshold', 0.4, 0.6, step=0.01)
    big_move_threshold = trial.suggest_float('big_move_threshold', 0.3, 0.7, step=0.01)
    switch_threshold = trial.suggest_float('switch_threshold', 0.3, 0.7, step=0.01)
    min_holding_days = trial.suggest_int('min_holding_days', 1, 5)

    # --- 데이터 준비 (lookback_years 적용) ---
    if features_df is None or features_df.empty or not isinstance(features_df.index, pd.DatetimeIndex):
        logger.warning(f"[Trial {trial.number}] Invalid features_df input.")
        return -np.inf
        
    # 전체 데이터의 마지막 날짜 기준 lookback_years 적용
    end_date = features_df.index.max()
    start_date = end_date - pd.DateOffset(years=lookback_years)
    # 데이터 시작일을 벗어나지 않도록 보정
    start_date = max(start_date, features_df.index.min())
    
    df_for_trial = features_df.loc[start_date:].copy()
    
    if df_for_trial.empty or len(df_for_trial) < 60: # 최소 데이터 기간 (예: 약 3개월) 확인
        logger.warning(f"[Trial {trial.number}] Not enough data after applying lookback_years={lookback_years}. Start: {start_date}, End: {end_date}. Filtered rows: {len(df_for_trial)}. Skipping.")
        return -np.inf # 데이터 부족 시 낮은 점수 반환
        
    logger.debug(f"[Trial {trial.number}] Using lookback_years={lookback_years}. Data period: {df_for_trial.index.min().date()} to {df_for_trial.index.max().date()}")

    # 변동성 감지 (필터링된 데이터 사용)
    df_with_volatility = detect_volatility(
        df_for_trial, vix_threshold, vix_ratio_threshold, price_change_threshold
    )
    if df_with_volatility is None: return -np.inf
    
    # 전략 구현 (필터링된 데이터 사용)
    strategy_df = implement_switching_strategy(
        df_with_volatility, models, direction_threshold, big_move_threshold, 
        switch_threshold, fee_rate, min_holding_days
    )
    if strategy_df is None: return -np.inf

    # 롤링 윈도우 백테스팅 (필터링된 데이터 사용)
    backtest_result = backtest_strategy_rolling(strategy_df, initial_capital, fee_rate)
    if backtest_result is None: 
        logger.warning(f"[Trial {trial.number}] backtest_strategy_rolling returned None for lookback={lookback_years}. Params: {trial.params}")
        return -np.inf # 백테스팅 실패 시
        
    _, window_performance_df, _ = backtest_result
    if window_performance_df is None or window_performance_df.empty: 
        logger.warning(f"[Trial {trial.number}] No valid rolling windows found for lookback={lookback_years}. Params: {trial.params}")
        return -np.inf # 윈도우 없음

    # --- 목표 함수 계산 (정보 비율 사용 - 기존 로직 유지) ---
    avg_information_ratio = window_performance_df['information_ratio'].mean()
    if pd.isna(avg_information_ratio) or np.isinf(avg_information_ratio):
        logger.warning(f"[Trial {trial.number}] Invalid Avg IR calculated: {avg_information_ratio}. Returning -1e6.")
        return -1e6

    avg_trade_count = window_performance_df['trade_count'].mean()
    trade_penalty = trade_penalty_factor * avg_trade_count
    objective_value = avg_information_ratio - trade_penalty

    # 로그 개선
    logger.debug(f"[Trial {trial.number}] Lookback={lookback_years}, Avg IR: {avg_information_ratio:.4f}, Avg Trades: {avg_trade_count:.2f} -> Obj: {objective_value:.4f}")

    return objective_value

# 최적화 실행 함수 (수정 - 정보 비율 기반 + 학습 기간 최적화)
def run_optimization(features_df, models, n_trials=200, cache_dir='cache', trade_penalty_factor=0.01):
    """파라미터 최적화 실행 (정보 비율 기반, 학습 기간[3-5년] 최적화, 캐시 기능 포함)"""
    logger.info("모델 파라미터 최적화 시작 (평균 정보 비율 최대화, 학습 기간[3-5년] 최적화)...")

    # 캐시 로직 (기존 유지, 파일명은 IR 기반 유지)
    Path(cache_dir).mkdir(exist_ok=True)
    cache_file = os.path.join(cache_dir, 'optimization_study_ir_lookback.pkl') # 캐시 파일명 변경 (lookback 추가)
    if os.path.exists(cache_file):
        logger.info(f"이전 최적화 결과를 로드합니다: {cache_file}")
        try:
            with open(cache_file, 'rb') as f:
                study = pickle.load(f)
            logger.info(f"이전 최적화에서 {len(study.trials)} 시도가 로드되었습니다.")
            if hasattr(study, 'best_value') and study.best_value is not None:
                 if pd.notna(study.best_value) and np.isfinite(study.best_value):
                     logger.info(f"현재까지의 최적 값 (Avg IR - Penalty): {study.best_value:.4f}")
                 else:
                     logger.info("로드된 연구의 최적값이 유효하지 않습니다 (NaN or Inf).")
            else:
                 logger.info("로드된 연구에 아직 최적값이 없습니다.")
        except Exception as e:
            logger.warning(f"캐시 로드 실패: {str(e)}. 새로운 연구 시작.")
            study = optuna.create_study(direction='maximize')
    else:
        logger.info("새로운 최적화 연구를 시작합니다 (평균 정보 비율 최대화, 학습 기간 최적화)...")
        study = optuna.create_study(direction='maximize')

    # 콜백 함수 (기존 유지)
    def save_study_callback(study, trial):
        with open(cache_file, 'wb') as f:
            pickle.dump(study, f)
        if trial.number > 0 and trial.number % 10 == 0:
             current_best_value = -np.inf
             if study.best_value is not None and pd.notna(study.best_value) and np.isfinite(study.best_value):
                 current_best_value = study.best_value
             logger.info(f"시도 {trial.number}/{n_trials} 완료. 현재 최적 값 (Avg IR - Penalty): {current_best_value:.4f}")

    # 최적화 실행 (objective 함수 변경 없음, 인자 전달 확인)
    try:
        study.optimize(
            lambda trial: objective(trial, features_df, models, trade_penalty_factor=trade_penalty_factor),
            n_trials=n_trials,
            callbacks=[save_study_callback],
            catch=(ValueError, RuntimeError, TypeError, ZeroDivisionError)
        )
    except KeyboardInterrupt:
        logger.warning("최적화가 사용자에 의해 중단되었습니다.")
        logger.warning("지금까지의 결과를 저장하고 최적 파라미터를 사용합니다.")
    except Exception as e:
        logger.error(f"Optuna 최적화 중 예상치 못한 오류 발생: {e}", exc_info=True)

    # --- 최적 파라미터 찾기 (기존 로직 유지) ---
    best_params = None
    best_value = -np.inf
    completed_valid_trials = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None and pd.notna(t.value) and np.isfinite(t.value)
    ]
    if completed_valid_trials:
        best_trial = max(completed_valid_trials, key=lambda t: t.value)
        best_params = best_trial.params
        best_value = best_trial.value
        logger.info("\n최적화 완료!")
        logger.info(f"최고 목적 함수 값 (Avg IR - Penalty): {best_value:.4f}")
        logger.info("최적 파라미터 (lookback_years 포함):")
        for param, value in best_params.items():
            logger.info(f"  {param}: {value}")
    else:
        logger.error("유효한 값으로 완료된 최적화 시도가 없습니다.")
        if study.trials:
            logger.warning("마지막 시도의 파라미터를 사용합니다 (성능 보장 안됨).")
            last_trial = study.trials[-1]
            best_params = last_trial.params
        else:
            logger.error("최적화 시도가 전혀 없어 파라미터를 결정할 수 없습니다.")
            return None

    if best_params is None:
        logger.error("최적 파라미터를 결정하지 못했습니다.")
        return None

    # --- 최종 평가 (최적 lookback_years 적용) ---
    logger.info("\n최적 파라미터로 최종 롤링 윈도우 백테스팅 실행 중...")
    
    # 1. 최적 lookback_years 추출 및 데이터 필터링
    best_lookback = best_params.get('lookback_years', 5) # 기본값 5년 (혹시 없을 경우 대비)
    logger.info(f"Applying optimal lookback_years={best_lookback} for final evaluation.")
    final_end_date = features_df.index.max()
    final_start_date = final_end_date - pd.DateOffset(years=best_lookback)
    final_start_date = max(final_start_date, features_df.index.min())
    features_df_final_backtest = features_df.loc[final_start_date:].copy()
    
    if features_df_final_backtest.empty or len(features_df_final_backtest) < 60:
        logger.error(f"Not enough data for final backtest with lookback_years={best_lookback}. Final period: {final_start_date} to {final_end_date}. Cannot proceed with final evaluation.")
        # 이 경우, best_params만 있는 결과를 반환할지 결정 필요
        return { 'best_params': best_params, 'performance_summary': None, 'window_results': None, 'detailed_results': None, 'optimization_history': study.trials_dataframe().to_dict() }
        
    logger.info(f"Final evaluation data period: {features_df_final_backtest.index.min().date()} to {features_df_final_backtest.index.max().date()}")
    
    # 2. 필터링된 데이터로 백테스팅 실행
    final_df_with_volatility = detect_volatility(
        features_df_final_backtest, # 필터링된 데이터 사용
        best_params.get('vix_threshold', 25.0),
        best_params.get('vix_ratio_threshold', 1.2),
        best_params.get('price_change_threshold', 0.03)
    )
    if final_df_with_volatility is None: 
        logger.error("Final evaluation: detect_volatility failed.")
        return { 'best_params': best_params, 'performance_summary': None, 'window_results': None, 'detailed_results': None, 'optimization_history': study.trials_dataframe().to_dict() }
        
    final_strategy_df = implement_switching_strategy(
        final_df_with_volatility,
        models,
        best_params.get('direction_threshold', 0.5),
        best_params.get('big_move_threshold', 0.5),
        best_params.get('switch_threshold', 0.5),
        0.0001, 
        best_params.get('min_holding_days', 1) # min_holding_days 도 best_params 에서 가져옴
    )
    if final_strategy_df is None: 
        logger.error("Final evaluation: implement_switching_strategy failed.")
        return { 'best_params': best_params, 'performance_summary': None, 'window_results': None, 'detailed_results': None, 'optimization_history': study.trials_dataframe().to_dict() }
        
    # 롤링 윈도우 백테스팅 실행 (필터링된 데이터 사용)
    final_backtest_result = backtest_strategy_rolling(final_strategy_df) 

    # 3. 최종 결과 요약 및 로깅 (기존 로직 유지)
    if final_backtest_result is None:
        logger.error("최적 파라미터로 최종 롤링 윈도우 백테스트를 수행했으나 실패했습니다.")
        final_detailed_df = None
        final_window_results_df = None
        final_overall_performance = {}
        performance_summary = None 
    else:
        final_detailed_df, final_window_results_df, final_overall_performance = final_backtest_result
        logger.info("\n최적화된 전략 최종 롤링 윈도우 백테스팅 요약 (최적 학습 기간 적용):")
        avg_ir_final = final_window_results_df['information_ratio'].mean()
        logger.info(f"  Applied Lookback Years: {best_lookback}") # 적용된 lookback 기간 로그 추가
        logger.info(f"  평균 정보 비율 (Avg IR): {avg_ir_final:.3f}" if pd.notna(avg_ir_final) else "  평균 정보 비율 (Avg IR): N/A")
        logger.info(f"  평균 연간 초과 수익률 (vs TQQQ): {final_window_results_df['annualized_avg_excess_return'].mean():.2%}")
        logger.info(f"  평균 추적 오차 (Tracking Error): {final_window_results_df['tracking_error'].mean():.2%}")
        logger.info(f"  평균 연간 수익률 (전략): {final_window_results_df['strategy_annual_return'].mean():.2%}")
        logger.info(f"  평균 연간 수익률 (TQQQ): {final_window_results_df['tqqq_annual_return'].mean():.2%}")
        logger.info(f"  평균 최대 낙폭 (전략): {final_window_results_df['strategy_max_drawdown'].mean():.2%}")
        logger.info(f"  TQQQ 대비 우위 윈도우 비율: {final_window_results_df['outperforms_tqqq'].mean():.2%}")
        logger.info(f"  평균 거래 횟수 (윈도우 당): {final_window_results_df['trade_count'].mean():.2f}")
        performance_summary = {
            'applied_lookback_years': best_lookback, # 적용된 lookback 기간 추가
            'avg_information_ratio': avg_ir_final,
            'avg_annualized_excess_return': final_window_results_df['annualized_avg_excess_return'].mean(),
            'avg_tracking_error': final_window_results_df['tracking_error'].mean(),
            'avg_strategy_annual_return': final_window_results_df['strategy_annual_return'].mean(),
            'std_strategy_annual_return': final_window_results_df['strategy_annual_return'].std(),
            'avg_tqqq_annual_return': final_window_results_df['tqqq_annual_return'].mean(),
            'avg_strategy_sharpe': final_window_results_df['strategy_sharpe'].mean(),
            'avg_strategy_mdd': final_window_results_df['strategy_max_drawdown'].mean(),
            'outperform_rate': final_window_results_df['outperforms_tqqq'].mean(),
            'avg_trades_per_window': final_window_results_df['trade_count'].mean(),
            'total_period_start': final_overall_performance.get('test_period', 'N/A').split(' ~ ')[0],
            'total_period_end': final_overall_performance.get('test_period', 'N/A').split(' ~ ')[-1],
            'final_strategy_capital': final_overall_performance.get('final_strategy_capital'),
            'final_tqqq_capital': final_overall_performance.get('final_tqqq_capital'),
            'objective_value': best_value
        }

    # 최종 시각화 호출 (기존 로직 유지)
    if final_detailed_df is not None and final_window_results_df is not None:
        logger.info("최적화된 전략의 최종 롤링 윈도우 결과 시각화 중...")
        try:
            visualize_results(final_detailed_df, final_window_results_df, performance_summary, best_params)
        except Exception as e:
            logger.error(f"최종 시각화 중 오류 발생: {e}", exc_info=True)
    else:
        logger.warning("최종 백테스트 결과가 유효하지 않아 시각화를 건너뜁니다.")
        
    # 최적화 결과 시각화 (기존 로직 유지, 파일명 수정 가능성)
    try:
        plt.figure(figsize=(14, 12))
        optimization_history = study.trials_dataframe()
        if not optimization_history.empty:
            valid_history = optimization_history.dropna(subset=['value'])
            valid_history = valid_history[np.isfinite(valid_history['value'])]
            if not valid_history.empty:
                plt.subplot(2, 1, 1)
                plt.plot(valid_history.number, valid_history.value, marker='.', linestyle='-')
                plt.title('Optimization Process (Objective Value per Trial: Avg IR - Penalty)')
                plt.xlabel('Trial Number')
                plt.ylabel('Objective Value (Avg IR - Penalty)')
                plt.grid(True, alpha=0.3)
            else:
                logger.warning("No valid objective values found for optimization history plot.")

        if completed_valid_trials:
             try:
                 param_importances = optuna.importance.get_param_importances(
                     study, target=lambda t: t.value if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None and pd.notna(t.value) and np.isfinite(t.value) else None
                 )
                 if param_importances:
                     plt.subplot(2, 1, 2)
                     params = list(param_importances.keys())
                     importances = list(param_importances.values())
                     # lookback_years 중요도도 표시됨
                     plt.barh(params, importances)
                     plt.title('Parameter Importances (based on Avg IR - Penalty)')
                     plt.xlabel('Importance')
                     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                 else:
                     logger.warning("파라미터 중요도를 계산할 수 없습니다 (importances empty).")
             except Exception as e:
                  logger.warning(f"파라미터 중요도 계산 중 오류 발생: {e}")
        else:
            logger.warning("완료된 유효한 시도가 없어 파라미터 중요도를 계산할 수 없습니다.")

        opt_hist_path = 'results/optimization_history_ir_lookback.png' # 파일명 변경
        plt.savefig(opt_hist_path)
        plt.close()
        logger.info(f"최적화 과정 및 파라미터 중요도 그래프 저장: {opt_hist_path}")
    except Exception as e:
        logger.error(f"최적화 과정 시각화 중 오류 발생: {e}")
        
    # 최적화 결과 반환 (기존 로직 유지)
    optimization_results = {
        'best_params': best_params,
        'performance_summary': performance_summary,
        'window_results': final_window_results_df,
        'detailed_results': final_detailed_df,
        'optimization_history': study.trials_dataframe().to_dict()
    }
    return optimization_results

# 메인 함수 optimize_parameters (기존 로직 유지)
def optimize_parameters(features_df, model_data):
    # ... (이 함수는 run_optimization을 호출하므로 내부 로직 변경 없음) ...
    logger.info("파라미터 최적화 시작 (평균 정보 비율 최대화)...")
    if features_df is None or model_data is None:
        logger.error("최적화를 위한 features_df 또는 model_data가 None입니다.")
        return None, None, None, None 
    required_models = ['direction_model', 'big_move_model', 'switch_model', 'scaler']
    if not all(key in model_data for key in required_models):
        logger.error(f"model_data 딕셔너리에 필요한 모델 키가 부족합니다. 필수 키: {required_models}")
        return None, None, None, None
    try:
        optimization_results = run_optimization(features_df, model_data, n_trials=100) # n_trials 수정 가능
        if optimization_results is None or not isinstance(optimization_results, dict):
            logger.error(f"run_optimization 실패 또는 유효하지 않은 결과 반환: {optimization_results}")
            return None, None, None, None
        best_params = optimization_results.get('best_params')
        performance_summary = optimization_results.get('performance_summary')
        window_results = optimization_results.get('window_results')
        detailed_results = optimization_results.get('detailed_results')
        if best_params is None or not isinstance(best_params, dict):
            logger.error(f"유효한 'best_params' (dictionary)를 최적화 결과에서 추출하지 못했습니다. Got: {best_params}")
            return None, None, window_results, detailed_results # Params 없어도 결과는 반환 시도
        if performance_summary is None:
             logger.warning("최적화는 성공했으나 최종 백테스트/요약에 실패했습니다. 최적 파라미터와 상세 결과만 반환합니다.")
             return best_params, None, window_results, detailed_results 
        logger.info("파라미터 최적화 완료!")
        return best_params, performance_summary, window_results, detailed_results
    except Exception as e:
        logger.error(f"파라미터 최적화 중 예외 발생: {str(e)}", exc_info=True)
        return None, None, None, None 

# 메인 실행 블록 (기존 로직 유지)
if __name__ == "__main__":
    # ... (기존 테스트용 코드) ...
    pass