import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import json
from datetime import datetime
from pathlib import Path

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
        models['direction'] = joblib.load('models/direction_model.pkl')
        models['big_move'] = joblib.load('models/big_move_model.pkl')
        models['switch'] = joblib.load('models/switch_model.pkl')
        models['scaler'] = joblib.load('models/scaler.pkl')
        print("모델 로드 완료")
    except Exception as e:
        print(f"모델 로드 실패: {str(e)}")
        return None, None, None
    
    # 특성 데이터셋 로드
    try:
        features_df = pd.read_csv('data/tqqq_features.csv', index_col=0)
        features_df.index = pd.to_datetime(features_df.index)
        print(f"특성 데이터셋 로드 완료: {features_df.shape}")
    except Exception as e:
        print(f"특성 데이터셋 로드 실패: {str(e)}")
        features_df = None
    
    # 최적 파라미터 로드
    try:
        with open('results/best_params.json', 'r') as f:
            best_params = json.load(f)
        print(f"최적 파라미터 로드 완료")
    except Exception as e:
        print(f"최적 파라미터 로드 실패: {str(e)}")
        best_params = None
    
    return models, features_df, best_params

# 변동성 감지 함수 (개선된 버전)
def detect_volatility_improved(df, vix_threshold, vix_ratio_threshold, price_change_threshold, 
                              lookback_period=5, trend_threshold=0.02):
    """개선된 변동성 감지 함수 - 추세 감지 및 빠른 전환 기능 추가"""
    if df is None:
        return None
    
    df = df.copy()
    
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
    
    # 추세 감지 (개선된 기능)
    # 단기 추세: 최근 N일 동안의 가격 변화율
    df['short_term_trend'] = df['close'].pct_change(periods=lookback_period)
    
    # 하락 추세 감지: 단기 추세가 임계값보다 낮으면 하락 추세로 판단
    df['downtrend'] = (df['short_term_trend'] < -trend_threshold).astype(int)
    
    # 상승 추세 감지: 단기 추세가 임계값보다 높으면 상승 추세로 판단
    df['uptrend'] = (df['short_term_trend'] > trend_threshold).astype(int)
    
    # 변동성 기간 식별 (연속된 고변동성 날짜를 하나의 기간으로 그룹화)
    df['volatility_period_start'] = (df['high_volatility'].diff() == 1).astype(int)
    df['volatility_period_end'] = (df['high_volatility'].diff() == -1).shift(-1).fillna(0).astype(int)
    
    # 빠른 전환 신호 (개선된 기능)
    # 고변동성 + 하락 추세 = 강한 SQQQ 전환 신호
    df['strong_switch_signal'] = ((df['high_volatility'] == 1) & (df['downtrend'] == 1)).astype(int)
    
    # 고변동성 종료 + 상승 추세 = 강한 TQQQ 복귀 신호
    df['strong_return_signal'] = ((df['high_volatility'].diff() == -1) & (df['uptrend'] == 1)).astype(int)
    
    return df

# SQQQ 전환 전략 구현 (개선된 버전)
def implement_switching_strategy_improved(df, models, params, fee_rate=0.0001, 
                                         adaptive_holding=True, max_trades_per_month=10):
    """개선된 TQQQ/SQQQ 전환 전략 구현 - 수수료 최적화 및 빠른 전환 기능 추가"""
    if df is None or models is None or params is None:
        return None
    
    # 파라미터 추출
    direction_threshold = params['direction_threshold']
    big_move_threshold = params['big_move_threshold']
    switch_threshold = params['switch_threshold']
    min_holding_days = params['min_holding_days']
    
    # 예측 확률 계산을 위한 특성 추출
    feature_cols = [col for col in df.columns if col not in [
        'next_day_return', 'target_direction', 'target_big_move', 'switch_to_sqqq',
        'high_volatility', 'volatility_period_start', 'volatility_period_end',
        'short_term_trend', 'downtrend', 'uptrend', 'strong_switch_signal', 'strong_return_signal'
    ]]
    
    # 스케일링에서 제외할 열
    exclude_cols = ['timestamp', 'MA_cross_up', 'MA_cross_down', 'VIX_spike', 'price_spike']
    scale_cols = [col for col in feature_cols if col not in exclude_cols]
    
    # 특성 스케일링
    X = df[feature_cols].copy()
    X[scale_cols] = models['scaler'].transform(X[scale_cols])
    
    # 모델 예측
    df = df.copy()  # 경고 방지를 위한 복사본 생성
    df['direction_prob'] = models['direction'].predict_proba(X)[:, 1]  # 상승 확률
    df['big_move_prob'] = models['big_move'].predict_proba(X)[:, 1]    # 급등락 확률
    df['switch_prob'] = models['switch'].predict_proba(X)[:, 1]        # SQQQ 전환 확률
    
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
    
    # 개선된 로직: 강한 전환 신호가 있으면 즉시 SQQQ로 전환
    df.loc[df['strong_switch_signal'] == 1, 'signal'] = -1
    
    # 개선된 로직: 강한 복귀 신호가 있으면 즉시 TQQQ로 복귀
    df.loc[df['strong_return_signal'] == 1, 'signal'] = 1
    
    # 추가 규칙: 작은 하락에는 매매 수수료를 고려하여 포지션 유지
    # 이전 신호가 TQQQ 매수이고, 현재 신호가 현금 보유이며, 예상 하락폭이 수수료보다 작으면 TQQQ 유지
    for i in range(1, len(df)):
        if df['signal'].iloc[i-1] == 1 and df['signal'].iloc[i] == 0:
            expected_loss = df['direction_prob'].iloc[i] - 0.5  # 방향 확률에서 0.5를 뺀 값을 예상 손실로 사용
            if abs(expected_loss) < fee_rate * 2:  # 수수료의 2배보다 작은 손실이면 포지션 유지 (더 보수적)
                df.loc[df.index[i], 'signal'] = 1  # TQQQ 유지 (경고 방지를 위한 loc 사용)
    
    # 적응형 보유 기간 적용 (개선된 기능)
    holding_asset = 0  # 0: 현금, 1: TQQQ, -1: SQQQ
    holding_days = 0
    
    for i in range(len(df)):
        current_signal = df['signal'].iloc[i]
        
        # 강한 신호가 있으면 최소 보유 기간 무시하고 즉시 전환
        strong_signal = df['strong_switch_signal'].iloc[i] == 1 or df['strong_return_signal'].iloc[i] == 1
        
        # 포지션 변경 여부 확인
        if current_signal != holding_asset:
            # 최소 보유 기간 확인 (강한 신호가 있으면 무시)
            if strong_signal or holding_days >= min_holding_days or holding_asset == 0:
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
    
    # 월별 거래 횟수 제한 (개선된 기능)
    if max_trades_per_month > 0:
        df['year_month'] = df.index.to_period('M')
        
        # 각 월별로 거래 횟수 계산 및 제한
        for ym in df['year_month'].unique():
            month_mask = df['year_month'] == ym
            month_df = df[month_mask]
            
            # 해당 월의 거래 횟수 계산
            trades_in_month = (month_df['signal_change'] != 0).sum()
            
            # 거래 횟수가 제한을 초과하면 중요도가 낮은 거래 제거
            if trades_in_month > max_trades_per_month:
                # 신호 변경 크기 (절대값)로 중요도 계산
                month_df_copy = month_df.copy()
                month_df_copy['change_importance'] = month_df_copy['signal_change'].abs()
                
                # 강한 신호에 가중치 부여
                month_df_copy.loc[month_df_copy['strong_switch_signal'] == 1, 'change_importance'] += 1
                month_df_copy.loc[month_df_copy['strong_return_signal'] == 1, 'change_importance'] += 1
                
                # 중요도가 낮은 순으로 정렬
                low_importance_trades = month_df_copy[month_df_copy['signal_change'] != 0].sort_values('change_importance')
                
                # 제거할 거래 수 계산
                trades_to_remove = trades_in_month - max_trades_per_month
                
                # 중요도가 낮은 거래 제거
                for idx in low_importance_trades.index[:trades_to_remove]:
                    # 이전 신호로 되돌림
                    prev_idx = df.index.get_loc(idx) - 1
                    if prev_idx >= 0:
                        prev_signal = df['signal'].iloc[prev_idx]
                        df.loc[idx, 'signal'] = prev_signal
        
        # 임시 열 제거
        df = df.drop('year_month', axis=1)
    
    # 신호 변경 재계산
    df['signal_change'] = df['signal'].diff().fillna(0)
    
    # 매매 횟수 계산
    trade_count = (df['signal_change'] != 0).sum()
    
    print(f"총 매매 신호 변경 횟수: {trade_count}")
    print(f"TQQQ 매수 신호 (1): {(df['signal'] == 1).sum()}일")
    print(f"현금 보유 신호 (0): {(df['signal'] == 0).sum()}일")
    print(f"SQQQ 매수 신호 (-1): {(df['signal'] == -1).sum()}일")
    
    return df

# 백테스팅 함수
def backtest_strategy(df, initial_capital=10000, fee_rate=0.0001):
    """전략 백테스팅"""
    if df is None:
        return None, None
    
    print("전략 백테스팅 중...")
    
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
    tqqq_sharpe = (tqqq_annual_return - risk_free_rate) / tqqq_volatility if not np.isnan(tqqq_annual_return) else 0
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
    
    # 시장 대비 초과 수익률
    excess_return = strategy_annual_return - tqqq_annual_return if not np.isnan(tqqq_annual_return) else strategy_annual_return
    
    # 거래 횟수
    trade_count = (df['signal_change'] != 0).sum()
    
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
        'excess_return': excess_return,
        'tqqq_volatility': tqqq_volatility,
        'strategy_volatility': strategy_volatility,
        'tqqq_sharpe': tqqq_sharpe,
        'strategy_sharpe': strategy_sharpe,
        'tqqq_max_drawdown': tqqq_max_drawdown,
        'strategy_max_drawdown': strategy_max_drawdown,
        'trade_count': trade_count,
        'total_fee': (df['fee'] * df['strategy_capital']).sum(),
        'avg_fee_per_trade': (df['fee'] * df['strategy_capital']).sum() / trade_count if trade_count > 0 else 0
    }
    
    # 결과 출력
    print("\n백테스팅 결과:")
    print(f"테스트 기간: {performance['test_period']} ({performance['days']}일, {performance['years']:.2f}년)")
    print(f"초기 자본금: ${performance['initial_capital']:,.2f}")
    print(f"최종 자본금 (TQQQ Buy & Hold): ${performance['final_tqqq_capital']:,.2f}")
    print(f"최종 자본금 (전략): ${performance['final_strategy_capital']:,.2f}")
    print(f"총 수익률 (TQQQ Buy & Hold): {performance['tqqq_total_return']:.2%}")
    print(f"총 수익률 (전략): {performance['strategy_total_return']:.2%}")
    print(f"연간 수익률 (TQQQ Buy & Hold): {performance['tqqq_annual_return']:.2%}")
    print(f"연간 수익률 (전략): {performance['strategy_annual_return']:.2%}")
    print(f"시장 대비 초과 수익률: {performance['excess_return']:.2%}")
    print(f"변동성 (TQQQ Buy & Hold): {performance['tqqq_volatility']:.2%}")
    print(f"변동성 (전략): {performance['strategy_volatility']:.2%}")
    print(f"샤프 비율 (TQQQ Buy & Hold): {performance['tqqq_sharpe']:.2f}")
    print(f"샤프 비율 (전략): {performance['strategy_sharpe']:.2f}")
    print(f"최대 낙폭 (TQQQ Buy & Hold): {performance['tqqq_max_drawdown']:.2%}")
    print(f"최대 낙폭 (전략): {performance['strategy_max_drawdown']:.2%}")
    print(f"총 거래 횟수: {performance['trade_count']}")
    print(f"총 수수료: ${performance['total_fee']:,.2f}")
    print(f"거래당 평균 수수료: ${performance['avg_fee_per_trade']:,.2f}")
    
    return df, performance

# 결과 시각화
def visualize_results(df, performance, output_dir='results_improved'):
    """백테스팅 결과 시각화"""
    if df is None:
        return
    
    print("Generating visualizations...")
    
    # 결과 디렉토리 생성
    Path(output_dir).mkdir(exist_ok=True)
    
    # 수익률 차트 (로그 스케일 적용)
    plt.figure(figsize=(14, 7))
    # Plot (1 + cumulative return) for log scale compatibility (starts at 1)
    plt.plot(df.index, 1 + df['cum_tqqq_return'], 'b-', label='TQQQ Buy & Hold')
    plt.plot(df.index, 1 + df['cum_strategy_return'], 'g-', label='Improved Strategy') # Keep label
    
    # Set Y-axis to log scale
    plt.yscale('log')
    
    plt.title('Cumulative Growth Comparison (Log Scale)') 
    plt.xlabel('Date') 
    plt.ylabel('Cumulative Growth (Log Scale, Start=1)') 
    
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.3) # Show grid for log scale
    plt.savefig(os.path.join(output_dir, 'cumulative_returns_log_scale.png')) # Save with a new name
    plt.close()
    
    # 자본금 차트 (기존 유지)
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df['tqqq_capital'], 'b-', label='TQQQ Buy & Hold')
    plt.plot(df.index, df['strategy_capital'], 'g-', label='Improved Strategy') # Keep label
    plt.title('Capital Growth Comparison') 
    plt.xlabel('Date') 
    plt.ylabel('Capital ($)') 
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'capital_growth.png')) # Keep original name
    plt.close()

    # 기존 cumulative_returns.png 도 생성 (선형 스케일)
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df['cum_tqqq_return'] * 100, 'b-', label='TQQQ Buy & Hold')
    plt.plot(df.index, df['cum_strategy_return'] * 100, 'g-', label='Improved Strategy') # Keep label
    plt.title('Cumulative Returns Comparison (Linear Scale)')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'cumulative_returns_linear_scale.png')) # Save with new name
    plt.close()
    
    # 매매 신호 및 가격 차트
    plt.figure(figsize=(14, 10))
    
    # 가격 차트
    ax1 = plt.subplot(211)
    ax1.plot(df.index, df['close'], 'b-', label='TQQQ Close')
    
    # 매수/매도 신호 표시
    tqqq_buy = df[df['signal'] == 1].index
    sqqq_buy = df[df['signal'] == -1].index
    cash_hold = df[df['signal'] == 0].index
    
    ax1.scatter(tqqq_buy, df.loc[tqqq_buy, 'close'], color='g', marker='^', s=50, label='Buy TQQQ')
    ax1.scatter(sqqq_buy, df.loc[sqqq_buy, 'close'], color='r', marker='v', s=50, label='Buy SQQQ')
    ax1.scatter(cash_hold, df.loc[cash_hold, 'close'], color='y', marker='o', s=30, label='Hold Cash')
    
    ax1.set_title('TQQQ Price and Trading Signals')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 변동성 및 예측 확률 차트
    ax2 = plt.subplot(212, sharex=ax1)
    ax2.plot(df.index, df['direction_prob'], 'g-', label='Up Probability', alpha=0.7)
    ax2.plot(df.index, df['big_move_prob'], 'r-', label='Big Move Probability', alpha=0.7)
    ax2.plot(df.index, df['switch_prob'], 'b-', label='Switch to SQQQ Probability', alpha=0.7)
    
    # 고변동성 기간 표시
    if 'high_volatility' in df.columns:
        high_vol_periods = df[df['high_volatility'] == 1].index
        ax2.scatter(high_vol_periods, df.loc[high_vol_periods, 'switch_prob'], 
                   color='purple', marker='*', s=100, label='High Volatility')
    
    ax2.set_title('Model Prediction Probabilities')
    ax2.set_ylabel('Probability')
    ax2.set_xlabel('Date')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'trading_signals.png'))
    plt.close()
    
    # 낙폭 차트
    plt.figure(figsize=(14, 7))
    tqqq_cum_returns = df['cum_tqqq_return']
    strategy_cum_returns = df['cum_strategy_return']
    
    tqqq_running_max = tqqq_cum_returns.cummax()
    strategy_running_max = strategy_cum_returns.cummax()
    
    tqqq_drawdown = (tqqq_cum_returns - tqqq_running_max) / (1 + tqqq_running_max) * 100
    strategy_drawdown = (strategy_cum_returns - strategy_running_max) / (1 + strategy_running_max) * 100
    
    plt.plot(df.index, tqqq_drawdown, 'r-', label='TQQQ Buy & Hold')
    plt.plot(df.index, strategy_drawdown, 'g-', label='Improved Strategy')
    plt.title('Drawdown Comparison')
    plt.xlabel('Date')
    plt.ylabel('Drawdown (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'drawdowns.png'))
    plt.close()
    
    # 월별 수익률 히트맵
    plt.figure(figsize=(14, 8))
    
    # 월별 수익률 계산
    df['year'] = df.index.year
    df['month'] = df.index.month
    
    # TQQQ 월별 수익률
    tqqq_monthly_returns = df.groupby(['year', 'month'])['next_day_return'].apply(
        lambda x: (1 + x).prod() - 1
    ).unstack() * 100
    
    # 전략 월별 수익률
    strategy_monthly_returns = df.groupby(['year', 'month'])['strategy_return'].apply(
        lambda x: (1 + x).prod() - 1
    ).unstack() * 100
    
    # 히트맵 그리기
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))
    
    sns.heatmap(tqqq_monthly_returns, annot=True, fmt='.1f', cmap='RdYlGn', center=0, ax=axes[0])
    axes[0].set_title('TQQQ Buy & Hold Monthly Returns (%)')
    axes[0].set_xlabel('')
    
    sns.heatmap(strategy_monthly_returns, annot=True, fmt='.1f', cmap='RdYlGn', center=0, ax=axes[1])
    axes[1].set_title('Improved Strategy Monthly Returns (%)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'monthly_returns_heatmap.png'))
    plt.close()
    
    # 성과 지표 요약 저장 (영어 번역)
    with open(os.path.join(output_dir, 'performance_summary.txt'), 'w') as f:
        f.write("# Improved Strategy Performance Summary\n\n")
        f.write("## Test Information\n")
        f.write(f"Test Period: {performance['test_period']} ({performance['days']} days, {performance['years']:.2f} years)\n")
        f.write(f"Initial Capital: ${performance['initial_capital']:,.2f}\n\n")
        f.write("## Return Comparison\n")
        f.write(f"Final Capital (TQQQ Buy & Hold): ${performance['final_tqqq_capital']:,.2f}\n")
        f.write(f"Final Capital (Strategy): ${performance['final_strategy_capital']:,.2f}\n")
        f.write(f"Total Return (TQQQ Buy & Hold): {performance['tqqq_total_return']:.2%}\n")
        f.write(f"Total Return (Strategy): {performance['strategy_total_return']:.2%}\n")
        f.write(f"Annual Return (TQQQ Buy & Hold): {performance['tqqq_annual_return']:.2%}\n")
        f.write(f"Annual Return (Strategy): {performance['strategy_annual_return']:.2%}\n")
        f.write(f"Excess Return vs Market: {performance['excess_return']:.2%}\n\n")
        f.write("## Risk Metrics\n")
        f.write(f"Volatility (TQQQ Buy & Hold): {performance['tqqq_volatility']:.2%}\n")
        f.write(f"Volatility (Strategy): {performance['strategy_volatility']:.2%}\n")
        f.write(f"Sharpe Ratio (TQQQ Buy & Hold): {performance['tqqq_sharpe']:.2f}\n")
        f.write(f"Sharpe Ratio (Strategy): {performance['strategy_sharpe']:.2f}\n")
        f.write(f"Max Drawdown (TQQQ Buy & Hold): {performance['tqqq_max_drawdown']:.2%}\n")
        f.write(f"Max Drawdown (Strategy): {performance['strategy_max_drawdown']:.2%}\n")
        f.write(f"Total Trades: {performance['trade_count']}\n")
        f.write(f"Total Fees: ${performance['total_fee']:,.2f}\n")
        if 'avg_fee_per_trade' in performance:
            f.write(f"Average Fee per Trade: ${performance['avg_fee_per_trade']:,.2f}\n")