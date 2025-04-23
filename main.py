import os
import pandas as pd
from datetime import datetime, timedelta
import logging
from pathlib import Path
import joblib
from optimize_parameters import optimize_parameters, detect_volatility, implement_switching_strategy

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tqqq_model_development.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 데이터 저장 경로 설정
DATA_DIR = Path('data')
MODEL_DIR = Path('models')
RESULTS_DIR = Path('results')

# 필요한 디렉토리 생성
for directory in [DATA_DIR, MODEL_DIR, RESULTS_DIR]:
    directory.mkdir(exist_ok=True)

def check_data_exists():
    """필요한 데이터 파일들이 존재하는지 확인"""
    required_files = [
        'TQQQ.csv', 'SQQQ.csv', 'VIX.csv', 
        'UUP.csv', 'NASDAQ.csv', 'QQQ.csv'
    ]
    missing_files = []
    
    for file in required_files:
        if not (DATA_DIR / file).exists():
            missing_files.append(file)
    
    return missing_files

def main():
    try:
        logger.info("TQQQ 트레이딩 모델 개발 프로세스 시작")
        
        # 1. 데이터 수집 확인 및 실행
        missing_files = check_data_exists()
        if missing_files:
            logger.info(f"누락된 데이터 파일들: {missing_files}")
            logger.info("데이터 수집 시작...")
            from collect_data import collect_all_data
            collect_all_data()
        else:
            logger.info("모든 필요한 데이터가 이미 존재합니다.")
        
        # 2. 데이터 분석 및 특성 추출
        logger.info("데이터 분석 및 특성 추출 시작...")
        from analyze_data import analyze_data
        features_df = analyze_data()
        
        # ===== analyze_data 결과 인덱스 확인 로그 추가 =====
        if features_df is not None:
            logger.info(f"analyze_data 완료 후 features_df 정보:")
            logger.info(f"  Shape: {features_df.shape}")
            if isinstance(features_df.index, pd.DatetimeIndex) and len(features_df.index) >= 2:
                logger.info(f"  Index Type: {type(features_df.index)}")
                logger.info(f"  Index Start: {features_df.index.min()}, Index End: {features_df.index.max()}")
                logger.info(f"  Index Is Monotonic Increasing: {features_df.index.is_monotonic_increasing}")
                logger.info(f"  Index Is Unique: {features_df.index.is_unique}")
            else:
                logger.warning("  features_df의 인덱스가 유효하지 않거나 길이가 2 미만입니다.")
        else:
            logger.warning("  analyze_data 결과 features_df가 None입니다.")
        # ===== 인덱스 확인 로그 끝 =====
        
        if features_df is None:
            raise ValueError("데이터 분석 결과가 None입니다. 데이터 분석 과정에서 오류가 발생했을 수 있습니다.")
        logger.info(f"데이터 분석 완료. 원본 특성 데이터 크기: {features_df.shape}")
        
        # 3. ML 모델 개발
        logger.info("ML 모델 개발 시작...")
        from develop_model import develop_model
        model_data = develop_model(features_df)
        
        if model_data is None:
            raise ValueError("모델 개발 결과가 None입니다. 모델 개발 과정에서 오류가 발생했을 수 있습니다.")
        logger.info("ML 모델 개발 완료.")
        
        # 4. 전략 구현 (Optional: 기본 파라미터 성능 확인용)
        # logger.info("기본 파라미터로 트레이딩 전략 구현 및 평가 시작...")
        # from implement_strategy import implement_strategy 
        # # implement_strategy가 features_df와 model_data를 받는다고 가정 (수정 필요)
        # base_strategy_df, base_performance = implement_strategy(features_df, model_data) 
        # if base_performance:
        #     logger.info(f"기본 전략 성능(샤프): {base_performance.get('strategy_sharpe', 'N/A'):.2f}")
        # else:
        #     logger.warning("기본 전략 평가 실패.")
        
        # 5. 파라미터 최적화
        logger.info("파라미터 최적화 시작 (학습 기간 포함)...")
        optimized_result = optimize_parameters(features_df, model_data)

        # --- 최적화 결과 유효성 검사 시작 ---
        if optimized_result is None or not isinstance(optimized_result, tuple) or len(optimized_result) != 4:
            logger.error("Parameter optimization failed or returned an unexpected result type.")
            raise ValueError("Parameter optimization failed or returned unexpected result.")

        best_params, performance_summary, window_results, detailed_results = optimized_result

        if best_params is None:
            # run_optimization 에서 best_params 를 못 찾은 경우 (예: 유효 trial 없음)
            logger.error("Optimization did not yield valid best parameters.")
            raise ValueError("Optimization failed to find best parameters.")
        if not isinstance(best_params, dict):
            logger.error(f"Optimization completed, but best_params is not a dictionary. Type: {type(best_params)}. Value: {best_params}")
            raise ValueError("Optimization did not return valid parameters (dictionary expected).")

        # performance_summary 유효성 검사 (None일 수 있음)
        if not isinstance(performance_summary, dict) and performance_summary is not None:
             logger.warning(f"Optimization completed, but performance_summary is not a dictionary or None. Type: {type(performance_summary)}. Value: {performance_summary}")
             # This case might indicate an issue, but allow proceeding if best_params are valid.
        elif performance_summary is None:
             logger.warning("Optimization succeeded, but the final backtest summary could not be generated.")
        # --- 최적화 결과 유효성 검사 끝 ---

        logger.info(f"파라미터 최적화 완료. 최적 파라미터: {best_params}")
        # 최적 lookback_years 로그 추가
        if 'applied_lookback_years' in performance_summary:
             logger.info(f"  최종 평가에 적용된 학습 기간(lookback): {performance_summary['applied_lookback_years']} 년")
        elif 'lookback_years' in best_params:
             logger.info(f"  최적화된 학습 기간(lookback): {best_params['lookback_years']} 년 (최종 평가 요약 누락)")
             
        if performance_summary:
            logger.info("최적화된 전략 성능 요약 (롤링 윈도우):")
            # 소수점 자리수 등 포맷팅 개선
            for key, value in performance_summary.items():
                if key == 'objective_value':
                     logger.info(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
                elif 'ratio' in key or 'sharpe' in key or 'trades' in key or 'lookback' in key:
                     logger.info(f"  {key}: {value:.3f}" if isinstance(value, float) else f"  {key}: {value}")
                elif 'return' in key or 'rate' in key or 'mdd' in key or 'error' in key:
                     logger.info(f"  {key}: {value:.2%}" if isinstance(value, float) else f"  {key}: {value}")
                elif 'capital' in key:
                     logger.info(f"  {key}: ${value:,.2f}" if isinstance(value, float) else f"  {key}: {value}")
                else:
                     logger.info(f"  {key}: {value}")

        # 6. 필요한 컬럼 선택 및 CSV 저장 (Use detailed_results from optimization)
        logger.info("최적화된 전략 상세 로그 저장 중...")
        if detailed_results is not None and not detailed_results.empty:
            log_columns = [
                'Close', 'TQQQ_close', # Add TQQQ_close if available
                'Daily_Return', 'next_day_return',# 기본 가격 정보
                'high_volatility', # 변동성 조건
                'direction_prob', 'big_move_prob', 'switch_prob', # 모델 예측 확률
                'pred_direction', 'pred_big_move', 'pred_switch', # 개별 신호
                'signal', # 최종 매매 결정 신호
                'signal_change', # 매매 발생 여부
                # 'Position', # Position is implicitly defined by signal (1, 0, -1)
                # 'Action', # Action can be inferred from signal_change
                'strategy_daily_return_gross', 'strategy_daily_return_net', # 일별 전략 수익률
                'cum_strategy_return', # 누적 전략 수익률
                'cum_tqqq_return', # 누적 Buy & Hold 수익률
                # 'strategy_capital', 'tqqq_capital', # Capital can be calculated if needed, adds complexity
                'fee' # 수수료
            ]

            # detailed_results 에 있는 컬럼 중 log_columns 에 포함된 것만 선택
            final_log_columns = [col for col in log_columns if col in detailed_results.columns]
            trade_log_df = detailed_results[final_log_columns].copy() # Use .copy()

            # 파일명에 lookback 정보 추가 (선택적)
            lookback_info = f"_lb{best_params.get('lookback_years', 'N')}" if best_params else ""
            log_file_path = RESULTS_DIR / f'optimal_rolling_trade_log_ir{lookback_info}.csv' 
            trade_log_df.to_csv(log_file_path)
            logger.info(f"Detailed trade log saved to: {log_file_path}")
        else:
            logger.warning("최적화 결과에서 상세 로그 데이터프레임(detailed_results)이 유효하지 않아 로그 파일을 저장할 수 없습니다.")

        # 7. 전략 비교 및 평가 (최적화 결과 사용)
        # logger.info("최적화된 전략 평가 시작...")

        # 8. 전략 개선 (최적화 결과 사용)
        # logger.info("전략 개선 시작...")

        # 9. 최종 모델 저장 (최적 파라미터와 함께 저장)
        logger.info("최종 모델 정보 저장 (모델 + 최적 파라미터)...")
        final_model_path = MODEL_DIR / 'final_optimized_model.pkl'
        param_file_suffix = f"_ir_lb{best_params.get('lookback_years', 'N')}.json" if best_params else "_ir.json"
        final_params_path = MODEL_DIR / f'final_optimized_params{param_file_suffix}'

        # 모델 저장 (모델 자체는 develop_model에서 저장되었을 수 있음)
        joblib.dump(model_data, final_model_path)
        import json
        try:
            # Ensure complex objects (like numpy types) are converted for JSON
            serializable_params = {k: (v.item() if hasattr(v, 'item') else v) for k, v in best_params.items()}
            with open(final_params_path, 'w') as f:
                json.dump(serializable_params, f, indent=4)
            logger.info(f"최종 모델 정보 저장 완료: {final_model_path}, {final_params_path}")
        except Exception as e:
            logger.error(f"최적 파라미터 JSON 저장 중 오류 발생: {e}")

        logger.info("모든 프로세스가 성공적으로 완료되었습니다!")
        
    except Exception as e:
        logger.error(f"프로세스 실행 중 오류 발생: {str(e)}", exc_info=True) # Traceback 포함 로깅
        # raise 제거하여 스크립트 종료 방지 (필요시 주석 해제)
        # raise 

if __name__ == "__main__":
    main() 