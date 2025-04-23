import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import json
from datetime import datetime
from pathlib import Path
import pickle

# 시각화 설정
plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 12
sns.set_palette('Set2')

class TQQQTradingModel:
    """TQQQ/SQQQ 트레이딩 모델 클래스"""
    
    def __init__(self, models_dir='models', params_file='results/best_params.json'):
        """모델 초기화"""
        self.models = {}
        self.params = {}
        self.models_dir = models_dir
        self.params_file = params_file
        
        # 개선된 파라미터 기본값
        self.improved_params = {
            'lookback_period': 5,
            'trend_threshold': 0.02,
            'max_trades_per_month': 8
        }
        
        # 모델 로드
        self.load_models()
        
        # 파라미터 로드
        self.load_params()
    
    def load_models(self):
        """학습된 모델 로드"""
        try:
            self.models['direction'] = joblib.load(f'{self.models_dir}/direction_model.pkl')
            self.models['big_move'] = joblib.load(f'{self.models_dir}/big_move_model.pkl')
            self.models['switch'] = joblib.load(f'{self.models_dir}/switch_model.pkl')
            self.models['scaler'] = joblib.load(f'{self.models_dir}/scaler.pkl')
            print("모델 로드 완료")
        except Exception as e:
            print(f"모델 로드 실패: {str(e)}")
    
    def load_params(self):
        """최적화된 파라미터 로드"""
        try:
            with open(self.params_file, 'r') as f:
                self.params = json.load(f)
            print("파라미터 로드 완료")
            
            # 개선된 파라미터 파일이 있으면 로드
            improved_params_file = 'results_improved/improved_params.json'
            if os.path.exists(improved_params_file):
                with open(improved_params_file, 'r') as f:
                    self.improved_params = json.load(f)
                print("개선된 파라미터 로드 완료")
        except Exception as e:
            print(f"파라미터 로드 실패: {str(e)}")
    
    def preprocess_data(self, df):
        """데이터 전처리"""
        if df is None:
            return None
        
        df = df.copy()
        
        # 기술적 지표 계산
        # 이동평균
        df['MA5'] = df['close'].rolling(window=5).mean()
        df['MA10'] = df['close'].rolling(window=10).mean()
        df['MA20'] = df['close'].rolling(window=20).mean()
        df['MA50'] = df['close'].rolling(window=50).mean()
        df['MA200'] = df['close'].rolling(window=200).mean()
        
        # 볼린저 밴드 (20일 기준)
        df['BB_middle'] = df['close'].rolling(window=20).mean()
        df['BB_std'] = df['close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + 2 * df['BB_std']
        df['BB_lower'] = df['BB_middle'] - 2 * df['BB_std']
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
        
        # RSI (14일 기준)
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        # 첫 번째 값 이후의 평균 계산
        for i in range(14, len(df)):
            avg_gain.iloc[i] = (avg_gain.iloc[i-1] * 13 + gain.iloc[i]) / 14
            avg_loss.iloc[i] = (avg_loss.iloc[i-1] * 13 + loss.iloc[i]) / 14
        
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['EMA12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_hist'] = df['MACD'] - df['MACD_signal']
        
        # 변동성 지표
        df['daily_return'] = df['close'].pct_change()
        df['volatility_20d'] = df['daily_return'].rolling(window=20).std() * np.sqrt(252)  # 연율화
        
        # 거래량 지표
        if 'volume' in df.columns:
            df['volume_MA20'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_MA20']
        
        # 가격 변화 지표
        df['price_change'] = df['close'].pct_change()
        df['price_change_abs'] = df['price_change'].abs()
        df['price_to_MA5_ratio'] = df['close'] / df['MA5']
        
        # 급등락 신호
        df['price_spike'] = (df['price_change_abs'] > 0.05).astype(int)
        
        # 추세 전환 지표
        df['MA_cross_up'] = ((df['MA5'] > df['MA50']) & (df['MA5'].shift(1) <= df['MA50'].shift(1))).astype(int)
        df['MA_cross_down'] = ((df['MA5'] < df['MA50']) & (df['MA5'].shift(1) >= df['MA50'].shift(1))).astype(int)
        
        return df
    
    def detect_volatility(self, df, vix_df=None):
        """변동성 감지"""
        if df is None:
            return None
        
        df = df.copy()
        
        # VIX 데이터 병합 (있는 경우)
        if vix_df is not None:
            # 날짜 인덱스 확인 및 조정
            vix_df = vix_df.loc[~vix_df.index.duplicated(keep='first')]
            df = df.join(vix_df['close'].rename('VIX'), how='left')
            
            # VIX 관련 지표 계산
            df['VIX_MA10'] = df['VIX'].rolling(window=10).mean()
            df['VIX_ratio'] = df['VIX'] / df['VIX_MA10']
            df['VIX_spike'] = (df['VIX_ratio'] > 1.2).astype(int)
        
        # 파라미터 추출
        vix_threshold = self.params.get('vix_threshold', 27.0)
        vix_ratio_threshold = self.params.get('vix_ratio_threshold', 1.4)
        price_change_threshold = self.params.get('price_change_threshold', 0.03)
        lookback_period = self.improved_params.get('lookback_period', 5)
        trend_threshold = self.improved_params.get('trend_threshold', 0.02)
        
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
        
        # 추세 감지
        # 단기 추세: 최근 N일 동안의 가격 변화율
        df['short_term_trend'] = df['close'].pct_change(periods=lookback_period)
        
        # 하락 추세 감지: 단기 추세가 임계값보다 낮으면 하락 추세로 판단
        df['downtrend'] = (df['short_term_trend'] < -trend_threshold).astype(int)
        
        # 상승 추세 감지: 단기 추세가 임계값보다 높으면 상승 추세로 판단
        df['uptrend'] = (df['short_term_trend'] > trend_threshold).astype(int)
        
        # 변동성 기간 식별 (연속된 고변동성 날짜를 하나의 기간으로 그룹화)
        df['volatility_period_start'] = (df['high_volatility'].diff() == 1).astype(int)
        df['volatility_period_end'] = (df['high_volatility'].diff() == -1).shift(-1).fillna(0).astype(int)
        
        # 빠른 전환 신호
        # 고변동성 + 하락 추세 = 강한 SQQQ 전환 신호
        df['strong_switch_signal'] = ((df['high_volatility'] == 1) & (df['downtrend'] == 1)).astype(int)
        
        # 고변동성 종료 + 상승 추세 = 강한 TQQQ 복귀 신호
        df['strong_return_signal'] = ((df['high_volatility'].diff() == -1) & (df['uptrend'] == 1)).astype(int)
        
        return df
    
    def generate_signals(self, df):
        """매매 신호 생성"""
        if df is None or not self.models:
            return None
        
        df = df.copy()
        
        # 파라미터 추출
        direction_threshold = self.params.get('direction_threshold', 0.54)
        big_move_threshold = self.params.get('big_move_threshold', 0.7)
        switch_threshold = self.params.get('switch_threshold', 0.7)
        min_holding_days = self.params.get('min_holding_days', 2)
        max_trades_per_month = self.improved_params.get('max_trades_per_month', 8)
        fee_rate = 0.0001  # 기본 수수료율
        
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
        X[scale_cols] = self.models['scaler'].transform(X[scale_cols])
        
        # 모델 예측
        df['direction_prob'] = self.models['direction'].predict_proba(X)[:, 1]  # 상승 확률
        df['big_move_prob'] = self.models['big_move'].predict_proba(X)[:, 1]    # 급등락 확률
        df['switch_prob'] = self.models['switch'].predict_proba(X)[:, 1]        # SQQQ 전환 확률
        
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
        
        # 적응형 보유 기간 적용
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
        
        # 월별 거래 횟수 제한
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
        
        return df
    
    def backtest(self, df, initial_capital=10000, fee_rate=0.0001):
        """전략 백테스팅"""
        if df is None:
            return None, None
        
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
        
 
(Content truncated due to size limit. Use line ranges to read in chunks)