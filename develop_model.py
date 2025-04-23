import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.inspection import permutation_importance
import os
import joblib
from datetime import datetime
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 시각화 설정
plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 12
sns.set_palette('Set2')

# 확장 윈도우 테스트 기간 (년)
test_period_years = 1  # 최근 N년을 테스트 데이터로 사용

# 데이터 로드
def load_feature_data():
    """특성 데이터셋 로드"""
    feature_file = 'data/tqqq_features.csv'
    if os.path.exists(feature_file):
        df = pd.read_csv(feature_file)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        elif df.index.name != 'date':
            df.index = pd.to_datetime(df.index)
        print(f"특성 데이터셋 로드 완료: {df.shape}")
        return df
    else:
        print(f"특성 데이터셋 파일을 찾을 수 없습니다: {feature_file}")
        return None

# 데이터 전처리
def preprocess_data(df, test_size=0.2, random_state=42):
    """ML 모델을 위한 데이터 전처리"""
    if df is None:
        return None, None, None, None, None
    
    # 결측치 제거
    df = df.dropna()
    
    # 특성 및 타겟 변수 분리
    X = df.drop(['next_day_return', 'target_direction', 'target_big_move', 'switch_to_sqqq'], axis=1)
    y_direction = df['target_direction']
    y_big_move = df['target_big_move']
    y_switch = df['switch_to_sqqq']
    
    # 시계열 데이터 분할: 확장 윈도우 or 기본 80/20
    if isinstance(df.index, pd.DatetimeIndex):
        cutoff = df.index.max() - pd.DateOffset(years=test_period_years)
        logger.info(f"확장 윈도우 분할: 학습 <= {cutoff.date()} | 테스트 > {cutoff.date()}")
        X_train = X.loc[:cutoff]
        X_test = X.loc[cutoff + pd.Timedelta(days=1):]
        y_direction_train = y_direction.loc[:cutoff]
        y_direction_test = y_direction.loc[cutoff + pd.Timedelta(days=1):]
        y_big_move_train = y_big_move.loc[:cutoff]
        y_big_move_test = y_big_move.loc[cutoff + pd.Timedelta(days=1):]
        y_switch_train = y_switch.loc[:cutoff]
        y_switch_test = y_switch.loc[cutoff + pd.Timedelta(days=1):]
    else:
        # 기본 80/20 분할
        split_idx = int(len(df) * (1 - test_size))
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_direction_train = y_direction.iloc[:split_idx]
        y_direction_test = y_direction.iloc[split_idx:]
        y_big_move_train = y_big_move.iloc[:split_idx]
        y_big_move_test = y_big_move.iloc[split_idx:]
        y_switch_train = y_switch.iloc[:split_idx]
        y_switch_test = y_switch.iloc[split_idx:]
    
    # 특성 스케일링
    scaler = StandardScaler()
    exclude_cols = ['timestamp', 'MA_cross_up', 'MA_cross_down', 'VIX_spike', 'price_spike']
    scale_cols = [col for col in X_train.columns if col not in exclude_cols]
    
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[scale_cols] = scaler.fit_transform(X_train[scale_cols])
    X_test_scaled[scale_cols] = scaler.transform(X_test[scale_cols])
    
    print(f"학습 데이터 크기: {X_train_scaled.shape}")
    print(f"테스트 데이터 크기: {X_test_scaled.shape}")
    
    # 스케일러 저장
    joblib.dump(scaler, 'models/scaler.pkl')
    
    return X_train_scaled, X_test_scaled, {
        'direction': (y_direction_train, y_direction_test),
        'big_move': (y_big_move_train, y_big_move_test),
        'switch': (y_switch_train, y_switch_test)
    }, df.iloc[split_idx:] if not isinstance(df.index, pd.DatetimeIndex) else df.loc[cutoff + pd.Timedelta(days=1):], scaler

# 방향 예측 모델 (상승/하락)
def train_direction_model(X_train, y_train, X_test, y_test):
    """주가 방향 예측 모델 학습"""
    logger.info("방향 예측 모델 학습 중...")
    
    # 랜덤 포레스트 모델
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        class_weight='balanced'
    )
    
    rf_model.fit(X_train, y_train)
    
    # 예측 및 평가
    y_pred = rf_model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    logger.info(f"방향 예측 모델 성능:")
    logger.info(f"정확도: {accuracy:.4f}")
    logger.info(f"정밀도: {precision:.4f}")
    logger.info(f"재현율: {recall:.4f}")
    logger.info(f"F1 점수: {f1:.4f}")
    
    # 혼동 행렬
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('방향 예측 모델 혼동 행렬')
    plt.ylabel('실제 클래스')
    plt.xlabel('예측 클래스')
    plt.savefig('models/direction_model_cm.png')
    plt.close()
    
    # 특성 중요도
    feature_importance = rf_model.feature_importances_
    features = X_train.columns
    
    # 특성 중요도 시각화
    indices = np.argsort(feature_importance)[-15:]  # 상위 15개 특성
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(indices)), feature_importance[indices], align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.title('방향 예측 모델 특성 중요도 (상위 15개)')
    plt.tight_layout()
    plt.savefig('models/direction_model_feature_importance.png')
    plt.close()
    
    # 모델 저장
    joblib.dump(rf_model, 'models/direction_model.pkl')
    
    return rf_model, accuracy, precision, recall, f1

# 급등락 예측 모델
def train_big_move_model(X_train, y_train, X_test, y_test):
    """급등락 예측 모델 학습"""
    logger.info("급등락 예측 모델 학습 중...")
    
    # 그래디언트 부스팅 모델 (급등락 예측에 더 적합)
    gb_model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        subsample=0.8
    )
    
    gb_model.fit(X_train, y_train)
    
    # 예측 및 평가
    y_pred = gb_model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    logger.info(f"급등락 예측 모델 성능:")
    logger.info(f"정확도: {accuracy:.4f}")
    logger.info(f"정밀도: {precision:.4f}")
    logger.info(f"재현율: {recall:.4f}")
    logger.info(f"F1 점수: {f1:.4f}")
    
    # 혼동 행렬
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds')
    plt.title('급등락 예측 모델 혼동 행렬')
    plt.ylabel('실제 클래스')
    plt.xlabel('예측 클래스')
    plt.savefig('models/big_move_model_cm.png')
    plt.close()
    
    # 특성 중요도
    feature_importance = gb_model.feature_importances_
    features = X_train.columns
    
    # 특성 중요도 시각화
    indices = np.argsort(feature_importance)[-15:]  # 상위 15개 특성
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(indices)), feature_importance[indices], align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.title('급등락 예측 모델 특성 중요도 (상위 15개)')
    plt.tight_layout()
    plt.savefig('models/big_move_model_feature_importance.png')
    plt.close()
    
    # 모델 저장
    joblib.dump(gb_model, 'models/big_move_model.pkl')
    
    return gb_model, accuracy, precision, recall, f1

# SQQQ 전환 신호 예측 모델
def train_switch_model(X_train, y_train, X_test, y_test):
    """SQQQ 전환 신호 예측 모델 학습"""
    logger.info("SQQQ 전환 신호 예측 모델 학습 중...")
    
    # 그래디언트 부스팅 모델
    gb_model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        subsample=0.8
    )
    
    gb_model.fit(X_train, y_train)
    
    # 예측 및 평가
    y_pred = gb_model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    logger.info(f"SQQQ 전환 신호 예측 모델 성능:")
    logger.info(f"정확도: {accuracy:.4f}")
    logger.info(f"정밀도: {precision:.4f}")
    logger.info(f"재현율: {recall:.4f}")
    logger.info(f"F1 점수: {f1:.4f}")
    
    # 혼동 행렬
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
    plt.title('SQQQ 전환 신호 예측 모델 혼동 행렬')
    plt.ylabel('실제 클래스')
    plt.xlabel('예측 클래스')
    plt.savefig('models/switch_model_cm.png')
    plt.close()
    
    # 특성 중요도
    feature_importance = gb_model.feature_importances_
    features = X_train.columns
    
    # 특성 중요도 시각화
    indices = np.argsort(feature_importance)[-15:]  # 상위 15개 특성
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(indices)), feature_importance[indices], align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.title('SQQQ 전환 신호 예측 모델 특성 중요도 (상위 15개)')
    plt.tight_layout()
    plt.savefig('models/switch_model_feature_importance.png')
    plt.close()
    
    # 모델 저장
    joblib.dump(gb_model, 'models/switch_model.pkl')
    
    return gb_model, accuracy, precision, recall, f1

# 앙상블 모델 (최종 매매 신호 생성)
def create_ensemble_model(direction_model, big_move_model, switch_model):
    """앙상블 모델 생성 (최종 매매 신호)"""
    logger.info("앙상블 모델 생성 중...")
    
    # 앙상블 모델 정보 저장
    ensemble_info = {
        'models': {
            'direction': 'models/direction_model.pkl',
            'big_move': 'models/big_move_model.pkl',
            'switch': 'models/switch_model.pkl'
        },
        'scaler': 'models/scaler.pkl',
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # 앙상블 정보 저장
    with open('models/ensemble_info.json', 'w') as f:
        import json
        json.dump(ensemble_info, f, indent=4)
    
    logger.info("앙상블 모델 정보가 저장되었습니다.")
    
    return ensemble_info

# 모델 평가 및 시각화
def evaluate_models_on_test_data(X_test, y_dict, direction_model, big_move_model, switch_model, test_df):
    """테스트 데이터에서 모델 평가 및 시각화"""
    logger.info("테스트 데이터에서 모델 평가 중...")
    
    # 방향 예측
    y_direction_pred = direction_model.predict(X_test)
    y_direction_prob = direction_model.predict_proba(X_test)[:, 1]  # 상승 확률
    
    # 급등락 예측
    y_big_move_pred = big_move_model.predict(X_test)
    y_big_move_prob = big_move_model.predict_proba(X_test)[:, 1]  # 급등락 확률
    
    # SQQQ 전환 신호 예측
    y_switch_pred = switch_model.predict(X_test)
    y_switch_prob = switch_model.predict_proba(X_test)[:, 1]  # SQQQ 전환 확률
    
    # 결과 데이터프레임 생성
    results_df = pd.DataFrame({
        'actual_direction': y_dict['direction'][1],
        'pred_direction': y_direction_pred,
        'direction_prob': y_direction_prob,
        'actual_big_move': y_dict['big_move'][1],
        'pred_big_move': y_big_move_pred,
        'big_move_prob': y_big_move_prob,
        'actual_switch': y_dict['switch'][1],
        'pred_switch': y_switch_pred,
        'switch_prob': y_switch_prob
    }, index=X_test.index)
    
    # 실제 가격 데이터 추가
    results_df['close'] = test_df['close']
    results_df['next_day_return'] = test_df['next_day_return']
    
    # 매매 신호 생성 (앙상블 로직)
    # 1: TQQQ 매수, 0: 현금 보유, -1: SQQQ 매수
    results_df['signal'] = 0
    
    # 기본 로직: 방향 예측이 상승이고 급등락 예측이 아니면 TQQQ 매수
    results_df.loc[(results_df['pred_direction'] == 1) & (results_df['pred_big_move'] == 0), 'signal'] = 1
    
    # SQQQ 전환 로직: 전환 신호가 있으면 SQQQ 매수
    results_df.loc[results_df['pred_switch'] == 1, 'signal'] = -1
    
    # 매매 신호 시각화
    plt.figure(figsize=(14, 10))
    
    # 가격 차트
    ax1 = plt.subplot(211)
    ax1.plot(results_df.index, results_df['close'], 'b-', label='TQQQ 종가')
    
    # 매수/매도 신호 표시
    tqqq_buy = results_df[results_df['signal'] == 1].index
    sqqq_buy = results_df[results_df['signal'] == -1].index
    cash_hold = results_df[results_df['signal'] == 0].index
    
    ax1.scatter(tqqq_buy, results_df.loc[tqqq_buy, 'close'], color='g', marker='^', s=100, label='TQQQ 매수')
    ax1.scatter(sqqq_buy, results_df.loc[sqqq_buy, 'close'], color='r', marker='v', s=100, label='SQQQ 매수')
    ax1.scatter(cash_hold, results_df.loc[cash_hold, 'close'], color='y', marker='o', s=50, label='현금 보유')
    
    ax1.set_title('테스트 기간 TQQQ 가격 및 매매 신호')
    ax1.set_ylabel('가격')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 예측 확률 차트
    ax2 = plt.subplot(212, sharex=ax1)
    ax2.plot(results_df.index, results_df['direction_prob'], 'g-', label='상승 확률')
    ax2.plot(results_df.index, results_df['big_move_prob'], 'r-', label='급등락 확률')
    ax2.plot(results_df.index, results_df['switch_prob'], 'b-', label='SQQQ 전환 확률')
    
    ax2.set_title('모델 예측 확률')
    ax2.set_ylabel('확률')
    ax2.set_xlabel('날짜')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('models/test_predictions.png')
    plt.close()
    
    # 결과 저장
    results_df.to_csv('models/test_results.csv')
    
    logger.info("테스트 결과가 저장되었습니다.")
    
    return results_df

def prepare_features(features_df):
    """특성 데이터 전처리"""
    # 날짜 컬럼 제거
    if 'Date' in features_df.columns:
        features_df = features_df.drop('Date', axis=1)
    
    # 타겟 변수 분리
    X = features_df.drop(['next_day_return', 'target_direction', 'target_big_move', 'switch_to_sqqq'], axis=1)
    y_direction = features_df['target_direction']
    y_big_move = features_df['target_big_move']
    y_switch = features_df['switch_to_sqqq']
    
    # 학습/테스트 데이터 분할
    X_train, X_test, y_train_direction, y_test_direction = train_test_split(
        X, y_direction, test_size=0.2, random_state=42, shuffle=False
    )
    
    _, _, y_train_big_move, y_test_big_move = train_test_split(
        X, y_big_move, test_size=0.2, random_state=42, shuffle=False
    )
    
    _, _, y_train_switch, y_test_switch = train_test_split(
        X, y_switch, test_size=0.2, random_state=42, shuffle=False
    )
    
    return X_train, X_test, y_train_direction, y_test_direction, y_train_big_move, y_test_big_move, y_train_switch, y_test_switch

def train_models(X_train, X_test, y_train_direction, y_test_direction,
                y_train_big_move, y_test_big_move,
                y_train_switch, y_test_switch):
    """세 가지 모델 학습"""
    # 방향 예측 모델
    logger.info("방향 예측 모델 학습 중...")
    direction_model = RandomForestClassifier(n_estimators=100, random_state=42)
    direction_model.fit(X_train, y_train_direction)
    direction_pred = direction_model.predict(X_test)
    logger.info(f"방향 예측 정확도: {accuracy_score(y_test_direction, direction_pred):.4f}")
    
    # 급등락 예측 모델
    logger.info("급등락 예측 모델 학습 중...")
    big_move_model = RandomForestClassifier(n_estimators=100, random_state=42)
    big_move_model.fit(X_train, y_train_big_move)
    big_move_pred = big_move_model.predict(X_test)
    logger.info(f"급등락 예측 정확도: {accuracy_score(y_test_big_move, big_move_pred):.4f}")
    
    # SQQQ 전환 예측 모델
    logger.info("SQQQ 전환 예측 모델 학습 중...")
    switch_model = RandomForestClassifier(n_estimators=100, random_state=42)
    switch_model.fit(X_train, y_train_switch)
    switch_pred = switch_model.predict(X_test)
    logger.info(f"SQQQ 전환 예측 정확도: {accuracy_score(y_test_switch, switch_pred):.4f}")
    
    return direction_model, big_move_model, switch_model

def evaluate_models(direction_model, big_move_model, switch_model,
                   X_test, y_test_direction, y_test_big_move, y_test_switch):
    """모델 평가 및 결과 저장"""
    # 모델 저장
    joblib.dump(direction_model, 'models/direction_model.pkl')
    joblib.dump(big_move_model, 'models/big_move_model.pkl')
    joblib.dump(switch_model, 'models/switch_model.pkl')
    
    # 평가 결과 저장
    with open('models/model_evaluation.txt', 'w') as f:
        f.write("=== 방향 예측 모델 평가 ===\n")
        f.write(classification_report(y_test_direction, direction_model.predict(X_test)))
        f.write("\n=== 급등락 예측 모델 평가 ===\n")
        f.write(classification_report(y_test_big_move, big_move_model.predict(X_test)))
        f.write("\n=== SQQQ 전환 예측 모델 평가 ===\n")
        f.write(classification_report(y_test_switch, switch_model.predict(X_test)))

def develop_model(features_df):
    """ML 모델 개발 메인 함수"""
    logger.info("ML 모델 개발 시작...")
    
    # 특성 데이터 준비
    (X_train, X_test, 
     y_train_direction, y_test_direction,
     y_train_big_move, y_test_big_move,
     y_train_switch, y_test_switch) = prepare_features(features_df)
    
    # 스케일러 학습 및 저장
    scaler = StandardScaler()
    # 스케일링에서 제외할 열 (이진 특성, 날짜 등)
    exclude_cols = ['timestamp', 'MA_cross_up', 'MA_cross_down', 'VIX_spike', 'price_spike']
    scale_cols = [col for col in X_train.columns if col not in exclude_cols and col in features_df.columns]
    
    # 스케일링 적용
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[scale_cols] = scaler.fit_transform(X_train[scale_cols])
    X_test_scaled[scale_cols] = scaler.transform(X_test[scale_cols])
    
    # 스케일러 저장
    if not os.path.exists('models'):
        os.makedirs('models')
    joblib.dump(scaler, 'models/scaler.pkl')
    logger.info("Scaler가 'models/scaler.pkl'에 저장되었습니다.")
    
    # 모델 학습
    direction_model, big_move_model, switch_model = train_models(
        X_train_scaled, X_test_scaled, y_train_direction, y_test_direction,
        y_train_big_move, y_test_big_move,
        y_train_switch, y_test_switch)
    
    # 모델 평가
    evaluate_models(direction_model, big_move_model, switch_model,
                   X_test_scaled, y_test_direction, y_test_big_move, y_test_switch)
    
    logger.info("ML 모델 개발 완료!")
    return {
        'direction_model': direction_model,
        'big_move_model': big_move_model,
        'switch_model': switch_model,
        'scaler': scaler # 스케일러도 반환하여 다음 단계에서 사용할 수 있도록 함
    }

if __name__ == "__main__":
    # 테스트용 코드
    features_df = pd.read_csv('data/tqqq_features.csv', index_col=0)
    develop_model(features_df)
