import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import logging
import matplotlib.ticker as mticker
from pathlib import Path

# 시각화 설정
plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 12
sns.set_palette('Set2')

# 로깅 설정
logger = logging.getLogger(__name__)

# 모델 및 데이터 로드
def load_models_and_data():
    """학습된 모델과 테스트 데이터 로드"""
    logger.info("모델 및 데이터 로드 중...")
    
    # 모델 로드
    models = {}
    try:
        models['direction'] = joblib.load('models/direction_model.pkl')
        models['big_move'] = joblib.load('models/big_move_model.pkl')
        models['switch'] = joblib.load('models/switch_model.pkl')
        models['scaler'] = joblib.load('models/scaler.pkl')
        logger.info("모델 로드 완료")
    except Exception as e:
        logger.error(f"모델 로드 실패: {str(e)}")
        return None, None, None
    
    # 테스트 결과 로드
    try:
        test_results = pd.read_csv('models/test_results.csv', index_col=0)
        test_results.index = pd.to_datetime(test_results.index)
        logger.info(f"테스트 결과 로드 완료: {test_results.shape}")
    except Exception as e:
        logger.error(f"테스트 결과 로드 실패: {str(e)}")
        test_results = None
    
    # 특성 데이터셋 로드
    try:
        features_df = pd.read_csv('data/tqqq_features.csv', index_col=0)
        features_df.index = pd.to_datetime(features_df.index)
        logger.info(f"특성 데이터셋 로드 완료: {features_df.shape}")
    except Exception as e:
        logger.error(f"특성 데이터셋 로드 실패: {str(e)}")
        features_df = None
    
    return models, test_results, features_df

# 변동성 감지 함수
def detect_volatility(df, vix_threshold=25, vix_ratio_threshold=1.2, price_change_threshold=0.03):
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
    
    logger.info(f"감지된 고변동성 날짜 수: {df['high_volatility'].sum()}")
    logger.info(f"감지된 변동성 기간 수: {len(volatility_starts)}")
    
    return df

# SQQQ 전환 전략 구현 (함수 내부에서 임계값 사용하도록 수정)
def implement_switching_strategy(df, models, best_params):
    """TQQQ/SQQQ 전환 전략 구현 (best_params 에서 임계값 사용)
    Args:
        df (pd.DataFrame): 입력 데이터프레임 (변동성 감지 후)
        models (dict): 학습된 모델 딕셔너리
        best_params (dict): 최적화된 파라미터 딕셔너리
    Returns:
        pd.DataFrame or None: 전략 신호가 추가된 데이터프레임 또는 실패 시 None
    """
    if df is None or models is None or best_params is None:
        logger.error("implement_switching_strategy: Invalid input (df, models, or best_params is None)")
        return None

    logger.info("TQQQ/SQQQ 전환 전략 구현 중 (최적 임계값 사용)...")

    # 최적 파라미터에서 임계값, 수수료, 최소 보유 기간 추출 (없으면 기본값 사용)
    direction_threshold = best_params.get('direction_threshold', 0.5)
    big_move_threshold = best_params.get('big_move_threshold', 0.5)
    switch_threshold = best_params.get('switch_threshold', 0.5)
    fee_rate = best_params.get('fee_rate', 0.0001)
    min_holding_days = best_params.get('min_holding_days', 1)

    logger.info(f"Using thresholds - Dir: {direction_threshold:.4f}, BigMove: {big_move_threshold:.4f}, Switch: {switch_threshold:.4f}")
    logger.info(f"Using Fee: {fee_rate:.4f}, MinHold: {min_holding_days}")

    # 예측 확률 계산을 위한 특성 추출
    feature_cols = [col for col in df.columns if col not in [
        'next_day_return', 'target_direction', 'target_big_move', 'switch_to_sqqq',
        'high_volatility', 'volatility_period_start', 'volatility_period_end', 'Date' # Date 추가
    ]]

    # 스케일링에서 제외할 열 확인 및 적용
    exclude_cols = ['timestamp', 'MA_cross_up', 'MA_cross_down', 'VIX_spike', 'price_spike']
    scale_cols = [col for col in feature_cols if col not in exclude_cols and col in df.columns]

    # 특성 데이터 준비 및 스케일링
    X = df[feature_cols].copy()
    if not scale_cols:
        logger.warning("No columns found for scaling. Skipping scaling.")
    elif 'scaler' not in models:
        logger.error("Scaler not found in models dictionary. Cannot scale features.")
        return None
    else:
        try:
            X[scale_cols] = X[scale_cols].fillna(X[scale_cols].mean())
            X[scale_cols] = X[scale_cols].replace([np.inf, -np.inf], 0)
            X[scale_cols] = models['scaler'].transform(X[scale_cols])
        except Exception as e:
            logger.error(f"Error during feature scaling: {e}", exc_info=True)
            return None

    # 모델 예측 (모델 키 이름 확인 필요: 'model' 접미사 사용 가정)
    try:
        dir_key = 'direction_model'
        big_key = 'big_move_model'
        swi_key = 'switch_model'
        if not all(key in models for key in [dir_key, big_key, swi_key]):
             logger.error(f"One or more required models ({dir_key}, {big_key}, {swi_key}) not found in models dict.")
             return None

        df = df.copy()
        df['direction_prob'] = models[dir_key].predict_proba(X)[:, 1]
        df['big_move_prob'] = models[big_key].predict_proba(X)[:, 1]
        df['switch_prob'] = models[swi_key].predict_proba(X)[:, 1]
    except Exception as e:
        logger.error(f"Error during model prediction: {e}", exc_info=True)
        return None

    # 최적화된 임계값 적용하여 예측 결정
    df['pred_direction'] = (df['direction_prob'] > direction_threshold).astype(int)
    df['pred_big_move'] = (df['big_move_prob'] > big_move_threshold).astype(int)
    df['pred_switch'] = (df['switch_prob'] > switch_threshold).astype(int)

    # 기본 매매 신호 생성 (1: TQQQ, 0: 현금, -1: SQQQ)
    df['signal'] = 0
    df.loc[(df['pred_direction'] == 1) & (df['pred_big_move'] == 0), 'signal'] = 1
    df.loc[df['pred_switch'] == 1, 'signal'] = -1

    # 추가 규칙: 작은 하락에는 수수료 고려하여 TQQQ 유지
    for i in range(1, len(df)):
        if df['signal'].iloc[i-1] == 1 and df['signal'].iloc[i] == 0:
            expected_loss = df['direction_prob'].iloc[i] - 0.5
            if abs(expected_loss) < fee_rate:
                # loc 사용 권장
                df.loc[df.index[i], 'signal'] = 1

    # 최소 보유 기간 적용
    holding_asset = 0
    holding_days = 0
    for i in range(len(df)):
        current_signal = df['signal'].iloc[i]
        if current_signal != holding_asset:
            if holding_days >= min_holding_days or holding_asset == 0:
                holding_asset = current_signal
                holding_days = 1
            else:
                # loc 사용 권장
                df.loc[df.index[i], 'signal'] = holding_asset
                holding_days += 1
        else:
            holding_days += 1

    # 매매 신호 변경 감지
    df['signal_change'] = df['signal'].diff().fillna(0)

    logger.info(f"Strategy implemented. Signals: TQQQ({(df['signal'] == 1).sum()}), Cash({(df['signal'] == 0).sum()}), SQQQ({(df['signal'] == -1).sum()})")

    return df

# 백테스팅 함수
def backtest_strategy(df, initial_capital=10000, fee_rate=0.0001):
    """전략 백테스팅"""
    if df is None:
        return None
    
    logger.info("전략 백테스팅 중...")
    
    # 수익률 계산을 위한 데이터 준비
    df = df.copy()
    
    # 다음 날 수익률 (이미 계산되어 있음)
    if 'next_day_return' not in df.columns:
        df['next_day_return'] = df['close'].pct_change().shift(-1)
    
    # 전략 수익률 계산
    df['strategy_return'] = 0.0
    
    # TQQQ 보유 시 수익률
    df.loc[df['signal'] == 1, 'strategy_return'] = df.loc[df['signal'] == 1, 'next_day_return']
    
    # SQQQ 보유 시 수익률 (TQQQ와 반대 방향으로 움직이지만 레버리지 효과 고려)
    # SQQQ는 TQQQ의 반대 방향으로 3배 레버리지이지만, 완벽한 반대가 아니므로 -0.9를 곱함
    df.loc[df['signal'] == -1, 'strategy_return'] = -0.9 * df.loc[df['signal'] == -1, 'next_day_return']
    
    # 매매 수수료 적용
    df['fee'] = 0.0
    df.loc[df['signal_change'] != 0, 'fee'] = fee_rate
    
    # 수수료 차감
    df['strategy_return'] = df['strategy_return'] - df['fee']
    
    # 누적 수익률 계산
    df['cum_tqqq_return'] = (1 + df['next_day_return']).cumprod() - 1
    df['cum_strategy_return'] = (1 + df['strategy_return']).cumprod() - 1
    
    # 자본금 계산
    df['tqqq_capital'] = initial_capital * (1 + df['cum_tqqq_return'])
    df['strategy_capital'] = initial_capital * (1 + df['cum_strategy_return'])
    
    # 성과 지표 계산
    # 연간 수익률
    days = (df.index[-1] - df.index[0]).days
    years = days / 365
    
    tqqq_annual_return = (1 + df['cum_tqqq_return'].iloc[-1]) ** (1 / years) - 1
    strategy_annual_return = (1 + df['cum_strategy_return'].iloc[-1]) ** (1 / years) - 1
    
    # 변동성 (연율화)
    tqqq_volatility = df['next_day_return'].std() * np.sqrt(252)
    strategy_volatility = df['strategy_return'].std() * np.sqrt(252)
    
    # 샤프 비율
    risk_free_rate = 0.02  # 2% 무위험 수익률 가정
    tqqq_sharpe = (tqqq_annual_return - risk_free_rate) / tqqq_volatility
    strategy_sharpe = (strategy_annual_return - risk_free_rate) / strategy_volatility
    
    # 최대 낙폭
    tqqq_cum_returns = df['cum_tqqq_return']
    strategy_cum_returns = df['cum_strategy_return']
    
    tqqq_running_max = tqqq_cum_returns.cummax()
    strategy_running_max = strategy_cum_returns.cummax()
    
    tqqq_drawdown = (tqqq_cum_returns - tqqq_running_max) / (1 + tqqq_running_max)
    strategy_drawdown = (strategy_cum_returns - strategy_running_max) / (1 + strategy_running_max)
    
    tqqq_max_drawdown = tqqq_drawdown.min()
    strategy_max_drawdown = strategy_drawdown.min()
    
    # 결과 출력
    logger.info("\n백테스팅 결과:")
    logger.info(f"테스트 기간: {df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')} ({days}일, {years:.2f}년)")
    logger.info(f"초기 자본금: ${initial_capital:,.2f}")
    logger.info(f"최종 자본금 (TQQQ Buy & Hold): ${df['tqqq_capital'].iloc[-1]:,.2f}")
    logger.info(f"최종 자본금 (전략): ${df['strategy_capital'].iloc[-1]:,.2f}")
    logger.info(f"총 수익률 (TQQQ Buy & Hold): {df['cum_tqqq_return'].iloc[-1]:.2%}")
    logger.info(f"총 수익률 (전략): {df['cum_strategy_return'].iloc[-1]:.2%}")
    logger.info(f"연간 수익률 (TQQQ Buy & Hold): {tqqq_annual_return:.2%}")
    logger.info(f"연간 수익률 (전략): {strategy_annual_return:.2%}")
    logger.info(f"변동성 (TQQQ Buy & Hold): {tqqq_volatility:.2%}")
    logger.info(f"변동성 (전략): {strategy_volatility:.2%}")
    logger.info(f"샤프 비율 (TQQQ Buy & Hold): {tqqq_sharpe:.2f}")
    logger.info(f"샤프 비율 (전략): {strategy_sharpe:.2f}")
    logger.info(f"최대 낙폭 (TQQQ Buy & Hold): {tqqq_max_drawdown:.2%}")
    logger.info(f"최대 낙폭 (전략): {strategy_max_drawdown:.2%}")
    logger.info(f"총 거래 횟수: {(df['signal_change'] != 0).sum()}")
    logger.info(f"총 수수료: ${(df['fee'] * df['strategy_capital']).sum():,.2f}")
    
    # 성과 지표 저장
    performance = {
        'test_period': f"{df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}",
        'days': days,
        'years': years,
        'initial_capital': initial_capital,
        'final_tqqq_capital': df['tqqq_capital'].iloc[-1],
        'final_strategy_capital': df['strategy_capital'].iloc[-1],
        'tqqq_total_return': df['cum_tqqq_return'].iloc[-1],
        'strategy_total_return': df['cum_strategy_return'].iloc[-1],
        'tqqq_annual_return': tqqq_annual_return,
        'strategy_annual_return': strategy_annual_return,
        'tqqq_volatility': tqqq_volatility,
        'strategy_volatility': strategy_volatility,
        'tqqq_sharpe': tqqq_sharpe,
        'strategy_sharpe': strategy_sharpe,
        'tqqq_max_drawdown': tqqq_max_drawdown,
        'strategy_max_drawdown': strategy_max_drawdown,
        'trade_count': (df['signal_change'] != 0).sum(),
        'total_fee': (df['fee'] * df['strategy_capital']).sum()
    }
    
    return df, performance

# 결과 시각화 (롤링 윈도우 기반으로 수정 + 정보 비율 반영)
def visualize_results(detailed_df, window_results_df, performance_summary, best_params):
    """롤링 윈도우 백테스팅 결과 시각화 (정보 비율 포함)"""
    if detailed_df is None or window_results_df is None or performance_summary is None:
        logger.warning("시각화를 위한 데이터가 부족합니다 (detailed_df, window_results_df, or performance_summary is None).")
        return

    logger.info("Generating rolling window visualizations (including Information Ratio)...")
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    initial_capital = performance_summary.get('initial_capital', 10000) # Get initial capital once

    # --- 1. 롤링 윈도우 연간 수익률 차트 --- (기존 유지)
    plt.figure(figsize=(14, 7))
    plt.plot(window_results_df.index, window_results_df['tqqq_annual_return'] * 100, 'b-', label='TQQQ (Rolling 1yr Ann. Return)')
    plt.plot(window_results_df.index, window_results_df['strategy_annual_return'] * 100, 'g-', label='Strategy (Rolling 1yr Ann. Return)')
    plt.axhline(0, color='grey', linestyle='--', linewidth=0.8)
    plt.title('Rolling 1-Year Annualized Return Comparison')
    plt.xlabel('Window Start Date')
    plt.ylabel('Annualized Return (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter())
    plt.savefig(results_dir / 'rolling_annual_returns.png')
    plt.close()

    # --- 2. 롤링 윈도우 정보 비율 차트 --- (신규 추가)
    plt.figure(figsize=(14, 7))
    plt.plot(window_results_df.index, window_results_df['information_ratio'], 'purple', label='Strategy (Rolling 1yr Information Ratio)')
    plt.axhline(0, color='grey', linestyle='--', linewidth=0.8)
    plt.title('Rolling 1-Year Information Ratio')
    plt.xlabel('Window Start Date')
    plt.ylabel('Information Ratio')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(results_dir / 'rolling_information_ratio.png')
    plt.close()

    # --- 3. 롤링 윈도우 샤프 비율 차트 --- (기존 유지, 번호 변경)
    plt.figure(figsize=(14, 7))
    plt.plot(window_results_df.index, window_results_df['tqqq_sharpe'], 'b-', label='TQQQ (Rolling 1yr Sharpe)')
    plt.plot(window_results_df.index, window_results_df['strategy_sharpe'], 'g-', label='Strategy (Rolling 1yr Sharpe)')
    plt.axhline(0, color='grey', linestyle='--', linewidth=0.8)
    plt.title('Rolling 1-Year Sharpe Ratio Comparison')
    plt.xlabel('Window Start Date')
    plt.ylabel('Sharpe Ratio')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(results_dir / 'rolling_sharpe_ratio.png')
    plt.close()

    # --- 4. 롤링 윈도우 최대 낙폭 차트 --- (기존 유지, 번호 변경)
    plt.figure(figsize=(14, 7))
    plt.plot(window_results_df.index, window_results_df['tqqq_max_drawdown'] * 100, 'b-', label='TQQQ (Rolling 1yr MDD)')
    plt.plot(window_results_df.index, window_results_df['strategy_max_drawdown'] * 100, 'g-', label='Strategy (Rolling 1yr MDD)')
    plt.title('Rolling 1-Year Maximum Drawdown Comparison')
    plt.xlabel('Window Start Date')
    plt.ylabel('Maximum Drawdown (%)')
    plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter())
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(results_dir / 'rolling_max_drawdown.png')
    plt.close()

    # --- 기존 차트들 (detailed_df 사용) ---
    # Capital Growth (전체 기간, 최종 파라미터 기준) - Log Scale 추천
    plt.figure(figsize=(14, 7))
    if 'cum_tqqq_return' in detailed_df.columns: detailed_df['tqqq_capital_vis'] = initial_capital * (1 + detailed_df['cum_tqqq_return'])
    if 'cum_strategy_return' in detailed_df.columns: detailed_df['strategy_capital_vis'] = initial_capital * (1 + detailed_df['cum_strategy_return'])
    if 'tqqq_capital_vis' in detailed_df.columns: plt.plot(detailed_df.index, detailed_df['tqqq_capital_vis'], 'b-', label='TQQQ Buy & Hold (Overall)')
    if 'strategy_capital_vis' in detailed_df.columns: plt.plot(detailed_df.index, detailed_df['strategy_capital_vis'], 'g-', label='Strategy (Overall w/ Best Params)')
    plt.yscale('log')
    plt.title('Overall Capital Growth Comparison (Log Scale, Best Params)')
    plt.xlabel('Date')
    plt.ylabel('Capital ($) (Log Scale)')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.savefig(results_dir / 'overall_capital_growth_log.png')
    plt.close()

    # 매매 신호 및 가격 차트
    plt.figure(figsize=(14, 10))
    ax1 = plt.subplot(211)
    close_col = 'Close' if 'Close' in detailed_df.columns else 'TQQQ_close' if 'TQQQ_close' in detailed_df.columns else None
    if close_col and not detailed_df.empty:
        ax1.plot(detailed_df.index, detailed_df[close_col], 'k-', label=f'{close_col} Price', alpha=0.8)
        if 'signal' in detailed_df.columns:
             tqqq_buy = detailed_df[detailed_df['signal'] == 1].index
             sqqq_buy = detailed_df[detailed_df['signal'] == -1].index
             ax1.scatter(tqqq_buy, detailed_df.loc[tqqq_buy, close_col], color='lime', marker='^', s=60, label='Buy TQQQ', edgecolors='black', linewidth=0.5, zorder=3)
             ax1.scatter(sqqq_buy, detailed_df.loc[sqqq_buy, close_col], color='red', marker='v', s=60, label='Buy SQQQ', edgecolors='black', linewidth=0.5, zorder=3)
    else:
        logger.warning("Close price or signal column not found/empty in detailed_df for signal plotting.")
        ax1.plot([], [], 'k-', label='Price (Not Found)')
    ax1.set_title('Price and Trading Signals (Overall w/ Best Params)')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 변동성 및 예측 확률 차트
    ax2 = plt.subplot(212, sharex=ax1)
    if 'direction_prob' in detailed_df.columns: ax2.plot(detailed_df.index, detailed_df['direction_prob'], 'g-', label='Up Prob', alpha=0.7)
    if 'big_move_prob' in detailed_df.columns: ax2.plot(detailed_df.index, detailed_df['big_move_prob'], 'r-', label='Big Move Prob', alpha=0.7)
    if 'switch_prob' in detailed_df.columns: ax2.plot(detailed_df.index, detailed_df['switch_prob'], 'b-', label='Switch Prob', alpha=0.7)
    if 'high_volatility' in detailed_df.columns:
        high_vol_periods = detailed_df[detailed_df['high_volatility'] == 1].index
        if not high_vol_periods.empty:
            bottom, top = ax2.get_ylim()
            ax2.scatter(high_vol_periods, [bottom]*len(high_vol_periods),
                        color='purple', marker='X', s=80, label='High Volatility', zorder=3)
            ax2.set_ylim(bottom, top)
    ax2.set_title('Model Prediction Probabilities & Volatility')
    ax2.set_ylabel('Probability / Indicator')
    ax2.set_xlabel('Date')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / 'overall_signals_probs.png')
    plt.close()

    # 낙폭 차트
    plt.figure(figsize=(14, 7))
    if 'cum_tqqq_return' in detailed_df.columns and 'cum_strategy_return' in detailed_df.columns:
         tqqq_cum_returns = detailed_df['cum_tqqq_return']
         strategy_cum_returns = detailed_df['cum_strategy_return']
         tqqq_running_max = (1 + tqqq_cum_returns).cummax()
         strategy_running_max = (1 + strategy_cum_returns).cummax()
         tqqq_drawdown = ((1 + tqqq_cum_returns) / tqqq_running_max - 1) * 100
         strategy_drawdown = ((1 + strategy_cum_returns) / strategy_running_max - 1) * 100
         plt.plot(detailed_df.index, tqqq_drawdown, 'r-', label='TQQQ Buy & Hold (Overall)')
         plt.plot(detailed_df.index, strategy_drawdown, 'g-', label='Strategy (Overall w/ Best Params)')
    else:
         logger.warning("Cumulative return columns missing for Drawdown plot.")
         plt.plot([], [], 'r-', label='TQQQ Buy & Hold (Overall)')
         plt.plot([], [], 'g-', label='Strategy (Overall w/ Best Params)')
    plt.title('Drawdown Comparison (Overall w/ Best Params)')
    plt.xlabel('Date')
    plt.ylabel('Drawdown (%)')
    plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter())
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(results_dir / 'overall_drawdowns.png')
    plt.close()

    # 월별 수익률 히트맵
    try:
        if not detailed_df.empty and 'next_day_return' in detailed_df.columns and 'strategy_daily_return_net' in detailed_df.columns:
            plt.figure() # figsize 제거 시도
            detailed_df['year'] = detailed_df.index.year
            detailed_df['month'] = detailed_df.index.month
            tqqq_monthly_returns = detailed_df.groupby(['year', 'month'])['next_day_return'].apply(
                lambda x: (1 + x.fillna(0)).prod() - 1
            ).unstack() * 100
            strategy_monthly_returns = detailed_df.groupby(['year', 'month'])['strategy_daily_return_net'].apply(
                lambda x: (1 + x.fillna(0)).prod() - 1
            ).unstack() * 100
            # figsize 조정
            fig, axes = plt.subplots(2, 1, figsize=(max(10, len(tqqq_monthly_returns.columns)*0.8), 8), sharex=True)
            sns.heatmap(tqqq_monthly_returns, annot=True, fmt='.1f', cmap='RdYlGn', center=0, ax=axes[0], cbar=False)
            axes[0].set_title('TQQQ Buy & Hold Monthly Returns (%) (Overall)')
            axes[0].set_xlabel('')
            axes[0].set_ylabel('Year')
            sns.heatmap(strategy_monthly_returns, annot=True, fmt='.1f', cmap='RdYlGn', center=0, ax=axes[1], cbar=True, cbar_kws={'label': 'Monthly Return (%)'})
            axes[1].set_title('Strategy Monthly Returns (%) (Overall w/ Best Params)')
            axes[1].set_xlabel('Month')
            axes[1].set_ylabel('Year')
            plt.tight_layout()
            plt.savefig(results_dir / 'overall_monthly_returns_heatmap.png')
            plt.close(fig) # fig 명시적 닫기
            plt.close('all') # 모든 figure 닫기 (보험)
        else:
            logger.warning("Data missing for Monthly Heatmap.")
    except Exception as e:
        logger.error(f"Error generating monthly heatmap: {e}")
        plt.close('all')

    # --- 성과 지표 요약 저장 (정보 비율 관련 지표 추가) ---
    summary_file_path = results_dir / 'rolling_performance_summary_ir.txt' # 파일명 일치 확인
    with open(summary_file_path, 'w', encoding='utf-8') as f:
        f.write("# Rolling Window Strategy Performance Summary (Optimized for Information Ratio)\n\n")
        f.write("## Test Information\n")
        f.write(f"Test Period: {performance_summary.get('total_period_start', 'N/A')} ~ {performance_summary.get('total_period_end', 'N/A')}\n")
        f.write(f"Initial Capital: ${initial_capital:,.2f}\n")
        f.write(f"Final Capital (TQQQ Buy & Hold - Overall): ${performance_summary.get('final_tqqq_capital', 0):,.2f}\n")
        f.write(f"Final Capital (Strategy - Overall): ${performance_summary.get('final_strategy_capital', 0):,.2f}\n\n")

        f.write("## Rolling Window Performance (1yr window, 6mo step)\n")
        # 정보 비율 관련 지표 강조
        f.write(f"Average Information Ratio (Strategy vs TQQQ): {performance_summary.get('avg_information_ratio', 0):.3f}\n")
        f.write(f"Average Annualized Excess Return (Strategy vs TQQQ): {performance_summary.get('avg_annualized_excess_return', 0):.2%}\n")
        f.write(f"Average Tracking Error: {performance_summary.get('avg_tracking_error', 0):.2%}\n")
        f.write(f"----\n") # 구분선
        # 다른 평균 지표들
        f.write(f"Average Annual Return (Strategy): {performance_summary.get('avg_strategy_annual_return', 0):.2%}\n")
        f.write(f"Average Annual Return (TQQQ): {performance_summary.get('avg_tqqq_annual_return', 0):.2%}\n")
        f.write(f"Std Dev Annual Return (Strategy): {performance_summary.get('std_strategy_annual_return', 0):.2%}\n")
        f.write(f"Average Sharpe Ratio (Strategy, risk_free=2%): {performance_summary.get('avg_strategy_sharpe', 0):.2f}\n")
        f.write(f"Average Max Drawdown (Strategy): {performance_summary.get('avg_strategy_mdd', 0):.2%}\n")
        f.write(f"Window Outperformance Rate (Strategy vs TQQQ): {performance_summary.get('outperform_rate', 0):.2%}\n")
        f.write(f"Average Trades per Window: {performance_summary.get('avg_trades_per_window', 0):.2f}\n")
        # 최종 최적화 점수 표시
        obj_value = performance_summary.get('objective_value', 'N/A')
        f.write(f"Optimization Objective Value (Avg IR - Trade Penalty): {obj_value:.4f}\n\n" if isinstance(obj_value, (int, float)) else f"Optimization Objective Value: {obj_value}\n\n")

        f.write("## Best Parameters Found\n")
        if best_params:
            sorted_params = sorted(best_params.items())
            for key, value in sorted_params:
                 if isinstance(value, float):
                      f.write(f"{key}: {value:.4f}\n")
                 else:
                      f.write(f"{key}: {value}\n")
        else:
            f.write("No parameters available.\n")

    logger.info(f"Visualizations and summary saved to '{results_dir.name}' directory (including Information Ratio metrics).")

# implement_strategy 함수는 optimize_parameters에서 visualize_results를 호출하므로,
# 이 함수 자체의 로직은 그대로 두거나, 테스트 목적 외에는 사용하지 않도록 주석 처리 등을 고려할 수 있습니다.
# 현재는 직접적인 백테스팅/시각화는 수행하지 않는 것으로 메시지를 남깁니다.
def implement_strategy(model):
    """트레이딩 전략 구현 메인 함수 (주로 개발/테스트용)"""
    logger.info("implement_strategy 함수 실행 (개발/테스트용).")
    # ... (모델/데이터 로드 등 필요한 준비 단계) ...
    models, test_results, features_df = load_models_and_data()
    if models is None or features_df is None:
        logger.error("모델 또는 데이터 로드 실패")
        return None

    # 필요시 기본 파라미터로 전략 구현 및 테스트 백테스팅 수행 가능
    # 예: features_df = detect_volatility(features_df)
    #     strategy_df = implement_switching_strategy(features_df, models, ...)
    #     backtest_df, performance = backtest_strategy(strategy_df) # 원래 백테스팅 함수 사용
    #     # visualize_results(backtest_df, performance) # 이 호출 대신 롤링 윈도우 버전 사용

    logger.info(" implement_strategy 함수는 현재 직접적인 백테스팅/시각화를 수행하지 않고, 모델/데이터 로드만 수행합니다.")
    # 필요한 객체 반환
    return features_df, models