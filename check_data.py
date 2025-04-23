import pandas as pd
import os

def check_data_files():
    data_dir = 'data'
    data_files = {
        'TQQQ': os.path.join(data_dir, 'TQQQ.csv'),
        'SQQQ': os.path.join(data_dir, 'SQQQ.csv'),
        'VIX': os.path.join(data_dir, 'VIX.csv'),
        'UUP': os.path.join(data_dir, 'UUP.csv'),
        'IXIC': os.path.join(data_dir, 'NASDAQ.csv'),
        'QQQ': os.path.join(data_dir, 'QQQ.csv')
    }
    
    for name, file_path in data_files.items():
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                print(f"\n{name} 데이터 구조:")
                print(f"컬럼: {df.columns.tolist()}")
                print(f"행 수: {len(df)}")
                print("\n첫 5행:")
                print(df.head())
            except Exception as e:
                print(f"{name} 데이터 로드 중 오류 발생: {str(e)}")
        else:
            print(f"{name} 데이터 파일을 찾을 수 없습니다: {file_path}")

if __name__ == "__main__":
    check_data_files() 