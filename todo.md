# TQQQ 트레이딩 모델 개발 계획

## 데이터 수집
- [x] TQQQ 주가 데이터 수집 (2015-2025)
- [x] SQQQ 주가 데이터 수집 (2015-2025)
- [x] VIX 지수 데이터 수집 (변동성 측정)
- [x] 달러 인덱스(UUP) 데이터 수집
- [x] NASDAQ 지수 데이터 수집 (기준 지수)
- [x] QQQ 데이터 수집
- [x] 거래량 데이터 수집
- [ ] 기타 관련 지표 데이터 수집 (필요시)

## 데이터 분석 및 특성 추출
- [x] 데이터 전처리 및 정제
- [x] 상관관계 분석
- [x] 특성 중요도 평가
- [x] 기술적 지표 생성 (이동평균, RSI, MACD 등)
- [x] 변동성 측정 지표 개발

## ML 모델 개발
- [x] 데이터 학습/테스트 분할
- [x] 모델 아키텍처 설계
- [x] 모델 학습 및 검증
- [x] 하이퍼파라미터 최적화

## 전략 구현
- [x] TQQQ/SQQQ 전환 로직 개발
- [x] 변동성 감지 알고리즘 구현
- [x] 매매 신호 생성 로직 개발
- [x] 리스크 관리 전략 구현

## 백테스팅 및 평가
- [x] 백테스팅 프레임워크 구축
- [x] 다양한 시장 상황에서 전략 테스트
- [x] 성과 지표 계산 (수익률, 샤프 비율, 최대 낙폭 등)
- [x] 벤치마크 대비 성능 평가

## 최적화 및 개선
- [ ] 모델 파라미터 최적화 (시장 초과 수익률 중심)
- [ ] 수수료 효율성 개선
- [ ] 빠른 전환 로직 개선
- [ ] 강건성 테스트

## 백테스팅 및 평가
- [ ] 백테스팅 프레임워크 구축
- [ ] 다양한 시장 상황에서 전략 테스트
- [ ] 성과 지표 계산 (수익률, 샤프 비율, 최대 낙폭 등)
- [ ] 벤치마크 대비 성능 평가

## 최적화 및 개선
- [ ] 모델 파라미터 최적화
- [ ] 전략 개선 및 보완
- [ ] 강건성 테스트

## 최종 모델 및 문서화
- [ ] 최종 트레이딩 모델 구현
- [ ] 사용 설명서 작성
- [ ] 결과 보고서 작성

## 모델 개발 결과 및 구현 가이드 (요약)

### 개발 결과 요약
- **모델 특징**:
    - ML 예측: 방향, 급등락, SQQQ 전환 신호
    - 변동성 감지: VIX, 가격 변동, 추세 분석
    - 빠른 전환: 강한 신호 시 즉시 전환
    - 수수료 최적화: 월별 거래 횟수 제한 (8회), 작은 하락 시 유지
    - 적응형 보유 기간: 시장 상황 따라 최소 보유 기간 조정 (기본 2일)
- **성능 결과**:
    - 최대 낙폭: -60.76% (vs TQQQ B&H -81.75%)
    - 변동성: 51.98% (vs TQQQ B&H 67.16%)
    - 총 거래: 800회, 거래당 평균 수수료: $92.73
- **최적 파라미터**:
    - VIX 임계값: 27.0
    - VIX 비율 임계값: 1.4
    - 가격 변화 임계값: 0.03
    - 방향 예측 임계값: 0.54
    - 급등락 예측 임계값: 0.7
    - SQQQ 전환 임계값: 0.7
    - 최소 보유 기간: 2일
    - 월별 최대 거래 횟수: 8회
- **개선된 전략 특징**:
    - 단기 추세 감지 (5일) 사용
    - 강한 신호 시 최소 보유 기간 무시

### 구현 가이드
- **모델 사용 방법**:
    ```python
    from tqqq_trading_model import TQQQTradingModel
    import pandas as pd

    # 모델 로드
    model = TQQQTradingModel.load_model('final_model/tqqq_trading_model.pkl')

    # 데이터 로드 (예시)
    tqqq_data = pd.read_csv('new_tqqq_data.csv', index_col=0)
    tqqq_data.index = pd.to_datetime(tqqq_data.index)
    vix_data = pd.read_csv('new_vix_data.csv', index_col=0)
    vix_data.index = pd.to_datetime(vix_data.index)

    # NaN 처리 (필요시)
    # tqqq_data = tqqq_data.fillna(method='ffill').fillna(method='bfill')
    # vix_data = vix_data.fillna(method='ffill').fillna(method='bfill')

    # 예측
    result_df = model.predict(tqqq_data, vix_data)

    # 현재 포지션 확인
    current_position = model.get_current_position(result_df)
    if current_position == 1:
        print('현재 포지션: TQQQ 매수')
    elif current_position == -1:
        print('현재 포지션: SQQQ 매수')
    else:
        print('현재 포지션: 현금 보유')
    ```
- **주의사항 및 개선점**:
    - [ ] 데이터 전처리 시 NaN 값 처리 확인/추가 (`fillna(method='ffill').fillna(method='bfill')`)
    - [ ] 3-6개월마다 모델 정기적 재학습 계획 수립
    - [ ] 실시간 데이터 연동 기능 추가 (예: Yahoo Finance API)
- **프로젝트 파일 구조**:
    - `tqqq_trading_model.py`: 최종 모델 클래스
    - `collect_data.py`: 데이터 수집
    - `analyze_data.py`: 데이터 분석/특성 추출
    - `develop_model.py`: ML 모델 개발
    - `implement_strategy.py`: 트레이딩 전략 구현
    - `optimize_parameters.py`: 파라미터 최적화
    - `compare_strategy.py`: 전략 비교 시각화
    - `refine_strategy.py`: 빠른 전환/수수료 최적화
- **파일 위치**: `/home/ubuntu/tqqq_trading_model`

### 결론 요약
- 개발된 모델은 TQQQ B&H 대비 낮은 변동성 및 개선된 최대 낙폭 제공.
- 급등락 시 SQQQ 전환으로 하락장 방어력 강화.
- 수수료 최적화 및 빠른 전환 메커니즘 구현.
