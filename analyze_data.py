import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import logging

# 시각화 설정
plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 12
sns.set_palette('Set2')

# 로깅 설정
logger = logging.getLogger(__name__)

# 데이터 로드
def load_data():
    """데이터 로드"""
    data_dir = 'data'
    data_files = {
        'TQQQ': os.path.join(data_dir, 'TQQQ.csv'),
        'SQQQ': os.path.join(data_dir, 'SQQQ.csv'),
        'VIX': os.path.join(data_dir, 'VIX.csv'),
        'UUP': os.path.join(data_dir, 'UUP.csv'),
        'IXIC': os.path.join(data_dir, 'NASDAQ.csv'),
        'QQQ': os.path.join(data_dir, 'QQQ.csv')
    }
    
    dfs = {}
    for name, file_path in data_files.items():
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                    df.set_index('Date', inplace=True)
                    df.index.name = 'Date'
                elif 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                    df.index.name = 'Date'
                else:
                    raise ValueError(f"파일에 'Date' 또는 'date' 컬럼이 없습니다: {file_path}")
                dfs[name] = df
                logger.info(f"{name} 데이터 로드 완료: {df.shape}")
            except Exception as e:
                logger.error(f"{name} 데이터 로드 중 오류 발생: {str(e)}")
        else:
            logger.error(f"{name} 데이터 파일을 찾을 수 없습니다: {file_path}")
    
    if not dfs:
        logger.error("로드된 데이터가 없습니다.")
        return None
    
    return dfs

# 기술적 지표 생성
def add_technical_indicators(df):
    """기술적 지표 추가"""
    # 이동평균
    df['MA5'] = df['TQQQ_close'].rolling(window=5).mean()
    df['MA10'] = df['TQQQ_close'].rolling(window=10).mean()
    df['MA20'] = df['TQQQ_close'].rolling(window=20).mean()
    df['MA50'] = df['TQQQ_close'].rolling(window=50).mean()
    df['MA200'] = df['TQQQ_close'].rolling(window=200).mean()
    
    # 볼린저 밴드
    df['BB_middle'] = df['TQQQ_close'].rolling(window=20).mean()
    df['BB_std'] = df['TQQQ_close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + 2 * df['BB_std']
    df['BB_lower'] = df['BB_middle'] - 2 * df['BB_std']
    
    # RSI
    delta = df['TQQQ_close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['TQQQ_close'].ewm(span=12, adjust=False).mean()
    exp2 = df['TQQQ_close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']
    
    # 이동평균 교차
    df['MA_cross_up'] = ((df['MA5'] > df['MA20']) & (df['MA5'].shift(1) <= df['MA20'].shift(1))).astype(int)
    df['MA_cross_down'] = ((df['MA5'] < df['MA20']) & (df['MA5'].shift(1) >= df['MA20'].shift(1))).astype(int)
    
    # 볼린저 밴드 신호
    df['BB_squeeze'] = ((df['BB_upper'] - df['BB_lower']) / df['BB_middle']).rolling(window=20).mean()
    df['BB_squeeze_signal'] = (df['BB_squeeze'] < df['BB_squeeze'].rolling(window=20).mean() * 0.5).astype(int)
    
    return df

# 변동성 분석 및 급등락 감지 지표
def add_volatility_indicators(df, vix_df):
    """변동성 관련 지표 추가"""
    logger.info(f"add_volatility_indicators 시작. df 인덱스 타입: {type(df.index)}")
    
    # 입력 df 인덱스 유효성 검사
    if not isinstance(df.index, pd.DatetimeIndex):
        logger.warning("add_volatility_indicators 입력 df의 인덱스가 DatetimeIndex가 아닙니다.")
        # 필요시 인덱스 복구 시도 또는 에러 처리
        try:
            # utc=True 추가하여 타임존 인식 문제 해결 시도
            df.index = pd.to_datetime(df.index, utc=True) 
            if not isinstance(df.index, pd.DatetimeIndex):
                 raise ValueError("DatetimeIndex 복구 실패")
            logger.info("입력 df 인덱스를 utc=True로 DatetimeIndex 복구 시도 완료.")
        except Exception as e:
             logger.error(f"입력 df 인덱스 복구 중 오류: {e}")
             return df # 또는 None 반환

    # VIX 데이터 병합 (있는 경우, merge 사용)
    if vix_df is not None:
        logger.info(f"VIX 데이터 병합 시도. vix_df 인덱스 타입: {type(vix_df.index)}")
        # VIX 인덱스 유효성 검사
        if not isinstance(vix_df.index, pd.DatetimeIndex):
             logger.warning("vix_df의 인덱스가 DatetimeIndex가 아닙니다.")
             try:
                 # utc=True 추가하여 타임존 인식 문제 해결 시도
                 vix_df.index = pd.to_datetime(vix_df.index, utc=True) 
                 if not isinstance(vix_df.index, pd.DatetimeIndex):
                      raise ValueError("DatetimeIndex 복구 실패")
                 logger.info("vix_df 인덱스를 utc=True로 DatetimeIndex 복구 시도 완료.")
             except Exception as e:
                  logger.error(f"vix_df 인덱스 복구 중 오류: {e}")
                  vix_df = None # 병합 불가 처리
        
        if vix_df is not None:
             # 날짜 인덱스 중복 제거
             vix_df = vix_df.loc[~vix_df.index.duplicated(keep='first')]
             
             # 인덱스 기반 merge 수행
             # df = df.join(vix_df['^VIX_close'].rename('VIX'), how='left') # 기존 join
             df = pd.merge(df, vix_df[['^VIX_close']].rename(columns={'^VIX_close': 'VIX'}),
                           left_index=True, right_index=True, how='left')
             logger.info(f"VIX 데이터 merge 후 df 인덱스 타입: {type(df.index)}")
        else:
            logger.warning("VIX 데이터 병합 실패. 인덱스 문제 가능성.")

    # 변동성 급증 지표 (VIX 병합 성공 시에만 계산)
    if 'VIX' in df.columns and df['VIX'].notna().any(): # VIX 데이터가 실제로 병합되었는지 확인
        # 이후 계산 전 인덱스 타입 재확인
        if not isinstance(df.index, pd.DatetimeIndex):
             logger.error("VIX 관련 지표 계산 전 인덱스가 DatetimeIndex가 아닙니다!")
             # 에러 발생시키거나, 문제가 되는 컬럼만 제거
             df.drop(columns=['VIX'], inplace=True, errors='ignore') 
             # VIX 관련 컬럼 기본값 설정
             df['VIX_MA10'] = np.nan
             df['VIX_ratio'] = np.nan
             df['VIX_spike'] = 0
        else:
            df['VIX_MA10'] = df['VIX'].rolling(window=10).mean()
            df['VIX_ratio'] = df['VIX'] / df['VIX_MA10']
            df['VIX_spike'] = (df['VIX_ratio'] > 1.2).astype(int)
    else:
        logger.warning("VIX 데이터가 없거나 병합에 실패하여 VIX 관련 지표를 계산할 수 없습니다.")
        # VIX 관련 컬럼이 없는 경우 생성 및 기본값 설정
        df['VIX'] = np.nan
        df['VIX_MA10'] = np.nan
        df['VIX_ratio'] = np.nan
        df['VIX_spike'] = 0 

    # 가격 급등락 지표 등 나머지 계산 (인덱스 타입 재확인)
    if not isinstance(df.index, pd.DatetimeIndex):
        logger.error("변동성 지표 계산 후 인덱스가 DatetimeIndex가 아닙니다!")
        return df # 또는 None
        
    df['price_change'] = df['TQQQ_close'].pct_change()
    df['price_change_abs'] = df['price_change'].abs()
    # MA5 컬럼 존재 확인 추가
    if 'MA5' in df.columns:
        df['price_to_MA5_ratio'] = df['TQQQ_close'] / df['MA5']
    else:
        logger.warning("'MA5' 컬럼이 없어 'price_to_MA5_ratio'를 계산할 수 없습니다.")
        df['price_to_MA5_ratio'] = np.nan
        
    df['price_spike'] = (df['price_change_abs'] > 0.05).astype(int)
    
    # MA5, MA50 컬럼 존재 확인 추가
    if 'MA5' in df.columns and 'MA50' in df.columns:
        df['MA_cross_up'] = ((df['MA5'] > df['MA50']) & (df['MA5'].shift(1) <= df['MA50'].shift(1))).astype(int)
        df['MA_cross_down'] = ((df['MA5'] < df['MA50']) & (df['MA5'].shift(1) >= df['MA50'].shift(1))).astype(int)
    else:
        logger.warning("'MA5' 또는 'MA50' 컬럼이 없어 'MA_cross' 지표를 계산할 수 없습니다.")
        df['MA_cross_up'] = 0
        df['MA_cross_down'] = 0
    
    logger.info(f"add_volatility_indicators 완료. 최종 df 인덱스 타입: {type(df.index)}")
    return df

# 상관관계 분석
def analyze_correlations(dfs):
    """자산 간 상관관계 분석"""
    # 종가 데이터 추출
    close_data = {}
    for name, df in dfs.items():
        if name != 'merged' and 'TQQQ_close' in df.columns:
            close_data[name] = df['TQQQ_close']
    
    # 데이터프레임으로 변환
    close_df = pd.DataFrame(close_data)
    
    # 결측치 처리
    close_df = close_df.dropna()
    
    # 상관관계 계산
    corr_matrix = close_df.corr()
    
    # 상관관계 히트맵 저장
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')
    plt.title('자산 간 상관관계 (종가 기준)')
    plt.tight_layout()
    plt.savefig('data/correlation_heatmap.png')
    plt.close()
    
    # 일간 수익률 상관관계
    returns_data = {}
    for name, df in dfs.items():
        if name != 'merged' and 'TQQQ_close' in df.columns:
            returns_data[name] = df['TQQQ_close'].pct_change()
    
    returns_df = pd.DataFrame(returns_data)
    returns_df = returns_df.dropna()
    
    returns_corr = returns_df.corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(returns_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')
    plt.title('자산 간 상관관계 (일간 수익률 기준)')
    plt.tight_layout()
    plt.savefig('data/returns_correlation_heatmap.png')
    plt.close()
    
    return corr_matrix, returns_corr

# TQQQ와 SQQQ 관계 분석
def analyze_tqqq_sqqq_relationship(dfs):
    """TQQQ와 SQQQ의 관계 분석"""
    if 'TQQQ' in dfs and 'SQQQ' in dfs:
        tqqq = dfs['TQQQ']
        sqqq = dfs['SQQQ']
        
        # 공통 날짜 찾기
        common_dates = tqqq.index.intersection(sqqq.index)
        tqqq = tqqq.loc[common_dates]
        sqqq = sqqq.loc[common_dates]
        
        # 일간 수익률 계산
        tqqq_returns = tqqq['TQQQ_close'].pct_change()
        sqqq_returns = sqqq['SQQQ_close'].pct_change()
        
        # 산점도 그리기
        plt.figure(figsize=(10, 8))
        plt.scatter(tqqq_returns, sqqq_returns, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.axvline(x=0, color='r', linestyle='-', alpha=0.3)
        
        # 추세선 추가
        z = np.polyfit(tqqq_returns.dropna(), sqqq_returns.dropna(), 1)
        p = np.poly1d(z)
        plt.plot(sorted(tqqq_returns.dropna()), p(sorted(tqqq_returns.dropna())), "r--", alpha=0.8)
        
        plt.title('TQQQ vs SQQQ 일간 수익률 관계')
        plt.xlabel('TQQQ 일간 수익률')
        plt.ylabel('SQQQ 일간 수익률')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('data/tqqq_sqqq_relationship.png')
        plt.close()
        
        # 상관관계 계산
        correlation = tqqq_returns.corr(sqqq_returns)
        logger.info(f"TQQQ와 SQQQ 일간 수익률 상관관계: {correlation:.4f}")
        
        # 극단적 시장 상황에서의 관계 분석
        extreme_days = tqqq_returns.abs() > tqqq_returns.abs().quantile(0.9)  # 상위 10% 변동성 날
        extreme_tqqq = tqqq_returns[extreme_days]
        extreme_sqqq = sqqq_returns[extreme_days]
        
        extreme_correlation = extreme_tqqq.corr(extreme_sqqq)
        logger.info(f"극단적 시장 상황에서 TQQQ와 SQQQ 일간 수익률 상관관계: {extreme_correlation:.4f}")
        
        return correlation, extreme_correlation
    else:
        logger.error("TQQQ 또는 SQQQ 데이터가 없습니다.")
        return None, None

# 변동성 분석
def analyze_volatility(dfs):
    """변동성 분석 및 시각화"""
    if 'TQQQ' in dfs and 'VIX' in dfs:
        tqqq = dfs['TQQQ']
        vix = dfs['VIX']
        
        # 공통 날짜 찾기
        common_dates = tqqq.index.intersection(vix.index)
        tqqq = tqqq.loc[common_dates]
        vix = vix.loc[common_dates]
        
        # 20일 변동성 계산
        tqqq['returns'] = tqqq['TQQQ_close'].pct_change()
        tqqq['volatility_20d'] = tqqq['returns'].rolling(window=20).std() * np.sqrt(252)  # 연율화
        
        # TQQQ 변동성과 VIX 관계 시각화
        plt.figure(figsize=(14, 7))
        
        ax1 = plt.subplot(111)
        ax1.plot(tqqq.index, tqqq['volatility_20d'], 'b-', label='TQQQ 20일 변동성')
        ax1.set_ylabel('TQQQ 변동성', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        
        ax2 = ax1.twinx()
        ax2.plot(vix.index, vix['^VIX_close'], 'r-', label='VIX')
        ax2.set_ylabel('VIX', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        
        plt.title('TQQQ 변동성과 VIX 지수 비교')
        plt.grid(True, alpha=0.3)
        
        # 범례 추가
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.tight_layout()
        plt.savefig('data/tqqq_vix_relationship.png')
        plt.close()
        
        # 상관관계 계산
        correlation = tqqq['volatility_20d'].corr(vix['^VIX_close'])
        logger.info(f"TQQQ 20일 변동성과 VIX 상관관계: {correlation:.4f}")
        
        # 변동성 급증 기간 식별
        vix['vix_ma20'] = vix['^VIX_close'].rolling(window=20).mean()
        vix['vix_ratio'] = vix['^VIX_close'] / vix['vix_ma20']
        
        # VIX가 20일 평균보다 30% 이상 높은 날을 변동성 급증 기간으로 정의
        high_vol_periods = vix[vix['vix_ratio'] > 1.3]
        
        logger.info(f"변동성 급증 기간 수: {len(high_vol_periods)}")
        
        # 변동성 급증 기간의 TQQQ 성과 분석
        high_vol_dates = high_vol_periods.index
        
        if len(high_vol_dates) > 0:
            # 각 변동성 급증 기간 이후 1주일, 1개월, 3개월 TQQQ 성과
            performance_1w = []
            performance_1m = []
            performance_3m = []
            
            for date in high_vol_dates:
                try:
                    # 해당 날짜의 TQQQ 종가
                    base_price = tqqq.loc[date, 'TQQQ_close']
                    
                    # 이후 날짜들
                    future_dates = tqqq.index[tqqq.index > date]
                    
                    if len(future_dates) > 5:  # 1주일 이상의 데이터가 있는 경우
                        price_1w = tqqq.loc[future_dates[5], 'TQQQ_close']
                        perf_1w = (price_1w / base_price - 1) * 100
                        performance_1w.append(perf_1w)
                    
                    if len(future_dates) > 20:  # 1개월 이상의 데이터가 있는 경우
                        price_1m = tqqq.loc[future_dates[20], 'TQQQ_close']
                        perf_1m = (price_1m / base_price - 1) * 100
                        performance_1m.append(perf_1m)
                    
                    if len(future_dates) > 60:  # 3개월 이상의 데이터가 있는 경우
                        price_3m = tqqq.loc[future_dates[60], 'TQQQ_close']
                        perf_3m = (price_3m / base_price - 1) * 100
                        performance_3m.append(perf_3m)
                except:
                    continue
            
            # 결과 출력
            if performance_1w:
                logger.info(f"변동성 급증 이후 1주일 TQQQ 평균 수익률: {np.mean(performance_1w):.2f}%")
            if performance_1m:
                logger.info(f"변동성 급증 이후 1개월 TQQQ 평균 수익률: {np.mean(performance_1m):.2f}%")
            if performance_3m:
                logger.info(f"변동성 급증 이후 3개월 TQQQ 평균 수익률: {np.mean(performance_3m):.2f}%")
        
        return correlation, high_vol_periods
    else:
        logger.error("TQQQ 또는 VIX 데이터가 없습니다.")
        return None, None

# 특성 중요도 분석을 위한 데이터 준비
def prepare_features_dataset(dfs):
    """ML 모델을 위한 특성 데이터셋 준비"""
    if 'TQQQ' not in dfs:
        logger.error("TQQQ 데이터가 없습니다.")
        return None
    
    # TQQQ 데이터 복사
    tqqq = dfs['TQQQ'].copy()
    logger.info(f"데이터 복사 후 shape: {tqqq.shape}")
    if tqqq.empty:
        logger.error("초기 TQQQ 데이터 복사 후 비어있습니다.")
        return None
        
    # 기술적 지표 추가
    tqqq = add_technical_indicators(tqqq)
    logger.info(f"기술적 지표 추가 후 shape: {tqqq.shape}")
    if tqqq.empty:
        logger.error("기술적 지표 추가 후 데이터가 비었습니다.")
        return None
        
    # VIX 데이터가 있으면 변동성 지표 추가
    tqqq = add_volatility_indicators(tqqq, dfs.get('VIX'))
    logger.info(f"변동성 지표 추가 후 shape: {tqqq.shape}, Index Type: {type(tqqq.index)}")
    if tqqq.empty:
        logger.error("변동성 지표 추가 후 데이터가 비었습니다.")
        return None
        
    # --- Join 전 인덱스 확인 및 Merge 방식으로 변경 --- 
    logger.info(f"Join/Merge 전 tqqq Index Type: {type(tqqq.index)}")
    if not isinstance(tqqq.index, pd.DatetimeIndex):
        logger.error("tqqq의 인덱스가 DatetimeIndex가 아닙니다! Join/Merge 불가.")
        try:
            # utc=True 추가
            tqqq.index = pd.to_datetime(tqqq.index, utc=True) 
            if not isinstance(tqqq.index, pd.DatetimeIndex):
                 raise ValueError("DatetimeIndex 복구 실패")
            logger.info("tqqq 인덱스를 utc=True로 DatetimeIndex로 복구 시도 완료.")
        except Exception as e:
             logger.error(f"tqqq 인덱스 복구 중 오류: {e}")
             return None 
             
    # 달러 인덱스 데이터 추가 (Merge 사용)
    if 'UUP' in dfs:
        uup = dfs['UUP']
        # UUP 인덱스 타입 확인 및 변환 시도 (utc=True 추가)
        if not isinstance(uup.index, pd.DatetimeIndex):
            logger.warning("UUP 인덱스가 DatetimeIndex가 아닙니다. 변환 시도...")
            try:
                uup.index = pd.to_datetime(uup.index, utc=True)
                if not isinstance(uup.index, pd.DatetimeIndex):
                    raise ValueError("UUP DatetimeIndex 복구 실패")
                logger.info("UUP 인덱스를 utc=True로 DatetimeIndex 변환 완료.")
            except Exception as e:
                logger.error(f"UUP 인덱스 변환 중 오류: {e}. UUP 데이터 Merge 불가.")
                uup = None # Merge 불가 처리
                
        if uup is not None: 
            tqqq = pd.merge(tqqq, uup[['UUP_close']].rename(columns={'UUP_close': 'UUP'}), 
                            left_index=True, right_index=True, how='left')
            tqqq['UUP_change'] = tqqq['UUP'].pct_change()
            tqqq['UUP_MA20'] = tqqq['UUP'].rolling(window=20).mean()
            tqqq['UUP_ratio'] = tqqq['UUP'] / tqqq['UUP_MA20']
        # else: # uup이 None일 경우 이미 로그 출력됨
        #     logger.warning("UUP 인덱스가 DatetimeIndex가 아니거나 변환 실패하여 Merge하지 않습니다.")
    logger.info(f"UUP merge 후 shape: {tqqq.shape}, Index Type: {type(tqqq.index)}") 
    if tqqq.empty:
        logger.error("UUP merge 후 데이터가 비었습니다.")
        return None
        
    # NASDAQ 지수 데이터 추가 (Merge 사용, 인덱스 확인/변환 추가)
    if 'IXIC' in dfs:
        ixic = dfs['IXIC']
        if not isinstance(ixic.index, pd.DatetimeIndex):
            logger.warning("IXIC 인덱스가 DatetimeIndex가 아닙니다. 변환 시도...")
            try:
                ixic.index = pd.to_datetime(ixic.index, utc=True)
                if not isinstance(ixic.index, pd.DatetimeIndex):
                    raise ValueError("IXIC DatetimeIndex 복구 실패")
                logger.info("IXIC 인덱스를 utc=True로 DatetimeIndex 변환 완료.")
            except Exception as e:
                logger.error(f"IXIC 인덱스 변환 중 오류: {e}. IXIC 데이터 Merge 불가.")
                ixic = None
                
        if ixic is not None:
            tqqq = pd.merge(tqqq, ixic[['^IXIC_close']].rename(columns={'^IXIC_close': 'IXIC'}),
                            left_index=True, right_index=True, how='left')
            tqqq['IXIC_change'] = tqqq['IXIC'].pct_change()
            tqqq['IXIC_MA20'] = tqqq['IXIC'].rolling(window=20).mean()
            tqqq['IXIC_ratio'] = tqqq['IXIC'] / tqqq['IXIC_MA20']
        # else:
        #    logger.warning("IXIC 인덱스가 DatetimeIndex가 아니거나 변환 실패하여 Merge하지 않습니다.")
    logger.info(f"IXIC merge 후 shape: {tqqq.shape}, Index Type: {type(tqqq.index)}") 
    if tqqq.empty:
        logger.error("IXIC merge 후 데이터가 비었습니다.")
        return None
        
    # QQQ 데이터 추가 (Merge 사용, 인덱스 확인/변환 추가)
    if 'QQQ' in dfs:
        qqq = dfs['QQQ']
        if not isinstance(qqq.index, pd.DatetimeIndex):
            logger.warning("QQQ 인덱스가 DatetimeIndex가 아닙니다. 변환 시도...")
            try:
                qqq.index = pd.to_datetime(qqq.index, utc=True)
                if not isinstance(qqq.index, pd.DatetimeIndex):
                    raise ValueError("QQQ DatetimeIndex 복구 실패")
                logger.info("QQQ 인덱스를 utc=True로 DatetimeIndex 변환 완료.")
            except Exception as e:
                logger.error(f"QQQ 인덱스 변환 중 오류: {e}. QQQ 데이터 Merge 불가.")
                qqq = None
                
        if qqq is not None:
            tqqq = pd.merge(tqqq, qqq[['QQQ_close']].rename(columns={'QQQ_close': 'QQQ'}),
                            left_index=True, right_index=True, how='left')
            tqqq['QQQ_change'] = tqqq['QQQ'].pct_change()
            tqqq['QQQ_MA20'] = tqqq['QQQ'].rolling(window=20).mean()
            tqqq['QQQ_ratio'] = tqqq['QQQ'] / tqqq['QQQ_MA20']
        # else:
        #    logger.warning("QQQ 인덱스가 DatetimeIndex가 아니거나 변환 실패하여 Merge하지 않습니다.")
    logger.info(f"QQQ merge 후 shape: {tqqq.shape}, Index Type: {type(tqqq.index)}") 
    if tqqq.empty:
        logger.error("QQQ merge 후 데이터가 비었습니다.")
        return None

    # --- 결측치 처리 전 로그 추가 ---
    logger.info(f"결측치 처리 전 shape: {tqqq.shape}")
    na_counts_detail = tqqq.isna().sum()
    logger.info(f"결측치 처리 전 컬럼별 NA 개수 (상위 10개):\n{na_counts_detail[na_counts_detail > 0].sort_values(ascending=False).head(10)}")
    # ------------------------------
    
    # 타겟 변수 생성 (다음 날 TQQQ 수익률)
    tqqq['next_day_return'] = tqqq['TQQQ_close'].pct_change().shift(-1)
    
    # 타겟 변수 분류 (상승/하락)
    tqqq['target_direction'] = (tqqq['next_day_return'] > 0).astype(int)
    
    # 급등락 타겟 변수 (다음 날 3% 이상 변동)
    tqqq['target_big_move'] = (tqqq['next_day_return'].abs() > 0.03).astype(int)
    
    # SQQQ로 전환 신호 (VIX 급등 또는 단기 하락 추세)
    if 'VIX_spike' in tqqq.columns:
        tqqq['switch_to_sqqq'] = ((tqqq['VIX_spike'] == 1) | (tqqq['MA_cross_down'] == 1)).astype(int)
    else:
        # MA_cross_down 계산을 add_technical_indicators 또는 add_volatility_indicators에서 수행했는지 확인 필요
        if 'MA_cross_down' in tqqq.columns:
            tqqq['switch_to_sqqq'] = tqqq['MA_cross_down']
        else:
            logger.warning("'MA_cross_down' 컬럼이 없어 'switch_to_sqqq' 계산에 기본값(0) 사용")
            tqqq['switch_to_sqqq'] = 0 # 기본값 설정 또는 오류 처리
    
    # --- 수정된 결측치 처리 시작 ---
    # 타겟 변수(next_day_return) 결측치 제거
    original_rows = len(tqqq)
    tqqq = tqqq.dropna(subset=['next_day_return'])
    rows_after_target_dropna = len(tqqq)
    logger.info(f"타겟 변수(next_day_return) 결측치 제거: {original_rows - rows_after_target_dropna} 행 제거됨. 현재 shape: {tqqq.shape}")
    
    # 나머지 특성들의 결측치는 0으로 채우기
    features_before_fillna = tqqq.columns.difference(['next_day_return', 'target_direction', 'target_big_move', 'switch_to_sqqq'])
    na_counts_before = tqqq[features_before_fillna].isna().sum().sum()
    if na_counts_before > 0:
        # fillna 전에 어떤 컬럼에 NA가 많은지 확인
        na_details_before_fill = tqqq[features_before_fillna].isna().sum()
        logger.warning(f"결측치 0으로 채우기 전 NA 개수 (상위 10개):\n{na_details_before_fill[na_details_before_fill > 0].sort_values(ascending=False).head(10)}")
        tqqq.fillna(0, inplace=True)
        logger.warning(f"특성 데이터의 결측치 {na_counts_before}개를 0으로 채웠습니다. 각 특성에 적합한지 확인 필요. 현재 shape: {tqqq.shape}")
    # --- 수정된 결측치 처리 끝 ---
    
    # 최종 데이터 확인
    if tqqq.empty:
        logger.error("최종 특성 데이터셋 생성 후 데이터가 비어 있습니다!")
        return None
        
    # 특성 데이터셋 저장
    tqqq.to_csv('data/tqqq_features.csv', index=True, index_label='Date')
    
    logger.info(f"특성 데이터셋 생성 완료: {tqqq.shape}")
    logger.info(f"사용 가능한 특성: {tqqq.columns.tolist()}")
    
    return tqqq

# 특성 중요도 시각화
def visualize_feature_importance(features_df):
    """특성 중요도 시각화"""
    if features_df is None:
        return
    
    # 타겟 변수와 특성 간의 상관관계 계산
    target_corr = {}
    
    # 다음 날 수익률과의 상관관계
    for col in features_df.columns:
        if col not in ['Date', 'next_day_return', 'target_direction', 'target_big_move', 'switch_to_sqqq']:
            try:
                corr = features_df[col].corr(features_df['next_day_return'])
                target_corr[col] = corr
            except:
                continue
    
    # 상관관계 기준 상위 15개 특성
    top_features = sorted(target_corr.items(), key=lambda x: abs(x[1]), reverse=True)[:15]
    
    # 시각화
    plt.figure(figsize=(12, 8))
    feature_names = [item[0] for item in top_features]
    corr_values = [item[1] for item in top_features]
    
    colors = ['green' if x > 0 else 'red' for x in corr_values]
    
    plt.barh(feature_names, [abs(x) for x in corr_values], color=colors)
    plt.xlabel('상관관계 절대값')
    plt.title('다음 날 TQQQ 수익률과 특성 간 상관관계 (상위 15개)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('data/feature_importance.png')
    plt.close()
    
    # 급등락 예측을 위한 특성 중요도
    big_move_corr = {}
    for col in features_df.columns:
        if col not in ['Date', 'next_day_return', 'target_direction', 'target_big_move', 'switch_to_sqqq']:
            try:
                corr = features_df[col].corr(features_df['target_big_move'])
                big_move_corr[col] = corr
            except:
                continue
    
    # 상관관계 기준 상위 15개 특성
    top_big_move_features = sorted(big_move_corr.items(), key=lambda x: abs(x[1]), reverse=True)[:15]
    
    # 시각화
    plt.figure(figsize=(12, 8))
    feature_names = [item[0] for item in top_big_move_features]
    corr_values = [item[1] for item in top_big_move_features]
    
    colors = ['green' if x > 0 else 'red' for x in corr_values]
    
    plt.barh(feature_names, [abs(x) for x in corr_values], color=colors)
    plt.xlabel('상관관계 절대값')
    plt.title('급등락 예측을 위한 특성 중요도 (상위 15개)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('data/big_move_feature_importance.png')
    plt.close()
    
    return top_features, top_big_move_features

def analyze_data():
    """데이터 분석 및 특성 추출을 수행하는 메인 함수"""
    logger.info("데이터 분석 및 특성 추출 시작...")
    
    # 데이터 로드
    dfs = load_data()
    
    # 상관관계 분석
    logger.info("\n자산 간 상관관계 분석 중...")
    corr_matrix, returns_corr = analyze_correlations(dfs)
    
    # TQQQ와 SQQQ 관계 분석
    logger.info("\nTQQQ와 SQQQ 관계 분석 중...")
    tqqq_sqqq_corr, extreme_corr = analyze_tqqq_sqqq_relationship(dfs)
    
    # 변동성 분석
    logger.info("\n변동성 분석 중...")
    vol_corr, high_vol_periods = analyze_volatility(dfs)
    
    # 특성 데이터셋 준비
    logger.info("\n특성 데이터셋 준비 중...")
    features_df = prepare_features_dataset(dfs)
    
    if features_df is None:
        logger.error("특성 데이터셋 생성에 실패했습니다.")
        return None
    
    # 특성 중요도 시각화
    logger.info("\n특성 중요도 시각화 중...")
    visualize_feature_importance(features_df)
    
    logger.info("\n데이터 분석 및 특성 추출 완료!")
    return features_df

if __name__ == "__main__":
    features_df = analyze_data()
    if features_df is not None:
        logger.info(f"생성된 특성 데이터셋 크기: {features_df.shape}")