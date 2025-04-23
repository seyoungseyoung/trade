import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import logging
from pathlib import Path
import time

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 데이터 저장 경로
DATA_DIR = Path('data')
DATA_DIR.mkdir(exist_ok=True)

# 수집할 데이터 설정
SYMBOLS = {
    'TQQQ': 'TQQQ',  # 3배 레버리지 NASDAQ-100
    'SQQQ': 'SQQQ',  # 3배 인버스 NASDAQ-100
    'VIX': '^VIX',   # CBOE 변동성 지수
    'UUP': 'UUP',    # Invesco DB US Dollar Index Bullish Fund
    'NASDAQ': '^IXIC',  # NASDAQ Composite
    'QQQ': 'QQQ'     # Invesco QQQ Trust
}

def fetch_data(symbol, start_date, end_date):
    """Yahoo Finance에서 데이터를 가져오는 함수"""
    try:
        logger.info(f"{symbol} 데이터 수집 시작...")
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date)
        
        if df.empty:
            logger.warning(f"{symbol} 데이터가 비어있습니다.")
            return None
            
        # 필요한 컬럼만 선택
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        
        # 컬럼명 변경
        df.columns = [f"{symbol}_{col.lower()}" for col in df.columns]
        
        logger.info(f"{symbol} 데이터 수집 완료: {len(df)}개 행")
        return df
        
    except Exception as e:
        logger.error(f"{symbol} 데이터 수집 중 오류 발생: {str(e)}")
        return None

def collect_all_data():
    """모든 필요한 데이터를 수집하는 함수"""
    # 데이터 수집 기간 설정 (2015-2025)
    start_date = '2015-01-01'
    end_date = '2025-12-31'
    
    all_data = {}
    
    for symbol, yf_symbol in SYMBOLS.items():
        # 데이터 파일이 이미 존재하는지 확인
        file_path = DATA_DIR / f"{symbol}.csv"
        
        if file_path.exists():
            logger.info(f"{symbol} 데이터가 이미 존재합니다. 건너뜁니다.")
            continue
            
        # 데이터 수집
        df = fetch_data(yf_symbol, start_date, end_date)
        
        if df is not None:
            # 데이터 저장 시 인덱스 레이블 명시
            df.to_csv(file_path, index_label='Date')
            all_data[symbol] = df
            
        # API 요청 제한을 고려하여 잠시 대기
        time.sleep(1)
    
    # 모든 데이터를 하나의 DataFrame으로 병합
    if all_data:
        # 병합 시에도 첫 번째 데이터프레임의 인덱스 사용 확인
        # 또는 필요시 인덱스 재설정 및 Date 컬럼화 후 병합
        try:
            # 가장 긴 데이터프레임의 인덱스를 기준으로 정렬 시도
            base_df = max(all_data.values(), key=len)
            combined_df = pd.DataFrame(index=base_df.index)
            for symbol, df_item in all_data.items():
                # 인덱스가 날짜 형식이 맞는지 확인 (선택적)
                if isinstance(df_item.index, pd.DatetimeIndex):
                    combined_df = combined_df.join(df_item, how='left')
                else:
                    logger.warning(f"{symbol} 데이터프레임의 인덱스가 DatetimeIndex가 아닙니다. 병합에서 제외될 수 있습니다.")
            
            # 필요시 결측치 처리
            # combined_df = combined_df.fillna(method='ffill')
            
            # 병합된 데이터 저장 시에도 인덱스 레이블 명시
            combined_df.to_csv(DATA_DIR / 'combined_data.csv', index_label='Date')
            logger.info("모든 데이터가 성공적으로 수집 및 병합되었습니다.")
        except Exception as e:
            logger.error(f"데이터 병합 중 오류 발생: {e}")
            # 병합 실패 시 개별 파일은 저장된 상태
            logger.info("개별 데이터 파일은 저장되었으나, 병합 데이터 생성에 실패했습니다.")
            
    else:
        logger.info("수집할 새로운 데이터가 없습니다.")

if __name__ == "__main__":
    collect_all_data()
