import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # 시각화 라이브러리 추가
import seaborn as sns          # 시각화 라이브러리 추가
import matplotlib.ticker as mticker # 시각화 라이브러리 추가
import os
import joblib
import json
from datetime import datetime
from pathlib import Path       # Path 객체 사용
import argparse
import logging
import glob # glob 모듈 추가

# 로깅 설정 (우선 정의)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('simulation.log'), # 스크립트 위치에 로그 파일 생성
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__) # 로거 객체 생성

# 시각화 기본 설정 (추가)
plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 12
sns.set_palette('Set2')

# --- 함수 임포트 (직접 임포트) ---
# try...except 제거하고 직접 import 시도
try:
    from implement_strategy import detect_volatility, implement_switching_strategy, backtest_strategy
    logger.info("Successfully imported functions from implement_strategy.")
    # 시뮬레이션에서는 optimize_parameters의 rolling backtest가 아닌, 표준 backtest 사용
    backtest_strategy_source = 'implement_strategy' 
except ImportError as e:
    logger.error(f"Fatal Error: Could not import required functions from implement_strategy.py: {e}")
    logger.error("Ensure implement_strategy.py is in the same directory and contains detect_volatility, implement_switching_strategy, and backtest_strategy.")
    # 에러 발생 시 스크립트 종료
    import sys
    sys.exit(1)
# --------------------------

# 경로 설정
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
MODEL_DIR = BASE_DIR / 'models'
RESULTS_DIR = BASE_DIR / 'results_simulation' # 시뮬레이션 결과 저장 디렉토리
RESULTS_DIR.mkdir(exist_ok=True) # 디렉토리 생성 (이미 있으면 무시)

# 모델, 파라미터, 데이터 로드 함수 (파라미터 파일 경로 수정)
def load_simulation_data():
    """시뮬레이션을 위한 모델, 파라미터, 특성 데이터 로드"""
    logger.info("Loading models, parameters, and feature data...")
    models = None
    best_params = None
    features_df = None

    # 모델 로드 (기존 로직 유지)
    model_path = MODEL_DIR / 'final_optimized_model.pkl'
    if model_path.exists():
        try:
            models = joblib.load(model_path)
            required_keys = ['direction_model', 'big_move_model', 'switch_model', 'scaler']
            if not all(key in models for key in required_keys):
                logger.warning(f"{model_path} does not contain all required model keys. Attempting to load individual files...")
                models = None
            else:
                logger.info(f"Loaded combined model data from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model data from {model_path}: {e}", exc_info=True)
            models = None

    if models is None:
        logger.info("Loading individual model files...")
        models = {}
        try:
            models['direction_model'] = joblib.load(MODEL_DIR / 'direction_model.pkl')
            models['big_move_model'] = joblib.load(MODEL_DIR / 'big_move_model.pkl')
            models['switch_model'] = joblib.load(MODEL_DIR / 'switch_model.pkl')
            models['scaler'] = joblib.load(MODEL_DIR / 'scaler.pkl')
            logger.info("Loaded individual model files successfully.")
        except Exception as e:
            logger.error(f"Error loading one or more individual model files: {e}", exc_info=True)
            return None, None, None

    # --- 최적 파라미터 로드 (파일 이름 검색 로직 수정) ---
    params_path_to_load = None
    search_pattern_ir_lb = MODEL_DIR / 'final_optimized_params_ir_lb*.json'
    params_path_rolling = MODEL_DIR / 'final_optimized_params_rolling.json'
    params_path_standard = MODEL_DIR / 'final_optimized_params.json'

    # 1. IR + Lookback 파일 검색 (가장 최신 파일 사용)
    ir_lb_files = sorted(list(MODEL_DIR.glob('final_optimized_params_ir_lb*.json')), reverse=True)
    if ir_lb_files:
        params_path_to_load = ir_lb_files[0] # 가장 최신 파일 (이름 정렬 기준)
        logger.info(f"Found IR+Lookback optimized parameters file: {params_path_to_load}")
    # 2. Rolling 파일 검색
    elif params_path_rolling.exists():
        params_path_to_load = params_path_rolling
        logger.info(f"Found rolling parameters file: {params_path_to_load}")
    # 3. Standard 파일 검색
    elif params_path_standard.exists():
        params_path_to_load = params_path_standard
        logger.info(f"Found standard parameters file: {params_path_to_load}")
    # 4. 어떤 파라미터 파일도 찾지 못한 경우
    else:
        logger.error(f"Optimized parameters file not found. Searched for patterns/files: {search_pattern_ir_lb.name}, {params_path_rolling.name}, {params_path_standard.name}")
        return None, None, None

    # 선택된 파라미터 파일 로드
    try:
        with open(params_path_to_load, 'r') as f:
            best_params = json.load(f)
        logger.info(f"Loaded optimized parameters from {params_path_to_load}")
    except Exception as e:
        logger.error(f"Error loading optimized parameters from {params_path_to_load}: {e}", exc_info=True)
        return None, None, None
    # --- 파라미터 로드 끝 ---

    # 특성 데이터 로드 (기존 로직 유지)
    features_path = DATA_DIR / 'tqqq_features.csv'
    if features_path.exists():
        try:
            features_df = pd.read_csv(features_path, index_col=0)
            features_df.index = pd.to_datetime(features_df.index)
            # 인덱스가 timezone-aware 인 경우 naive 로 변환 (비교를 위해)
            if features_df.index.tz is not None:
                 logger.info("Converting features_df index to timezone-naive.")
                 features_df.index = features_df.index.tz_localize(None)
            logger.info(f"Loaded features data from {features_path}: {features_df.shape}")
        except Exception as e:
            logger.error(f"Error loading features data from {features_path}: {e}", exc_info=True)
            return None, None, None
    else:
        logger.error(f"Features data file not found: {features_path}")
        return None, None, None

    return models, best_params, features_df

def run_simulation(start_date_str, end_date_str):
    """지정된 기간 동안 백테스팅 시뮬레이션 실행 및 결과 시각화"""
    logger.info(f"Starting simulation for period: {start_date_str} to {end_date_str}")

    # 데이터 로드
    models, best_params, features_df = load_simulation_data()
    if models is None or best_params is None or features_df is None:
        logger.error("Failed to load necessary data/models for simulation. Exiting.")
        return

    # 날짜 형식 변환 및 유효성 검사
    try:
        # 입력 문자열에서 시간 정보 제거 시도 (더 안정적인 방법)
        start_date = pd.to_datetime(start_date_str.split(' ')[0])
        end_date = pd.to_datetime(end_date_str.split(' ')[0])
        if start_date >= end_date:
            logger.error("Start date must be before end date.")
            return
    except ValueError:
        logger.error("Invalid date format. Please use YYYY-MM-DD.")
        return

    # 기간 필터링
    logger.info("Filtering data for the specified period...")
    # features_df 인덱스도 timezone-naive 여야 함 (load_simulation_data 에서 처리)
    simulation_df = features_df.loc[start_date:end_date].copy()

    if simulation_df.empty:
        logger.error(f"No data available for the specified period: {start_date_str} to {end_date_str}")
        return

    logger.info(f"Data filtered: {simulation_df.shape} rows from {simulation_df.index.min()} to {simulation_df.index.max()}")

    # 변동성 감지
    logger.info("Detecting volatility...")
    simulation_df_vol = detect_volatility(
        simulation_df.copy(), # 원본 보존을 위해 복사본 사용
        best_params.get('vix_threshold', 25.0),
        best_params.get('vix_ratio_threshold', 1.2),
        best_params.get('price_change_threshold', 0.03)
    )
    if simulation_df_vol is None:
        logger.error("Volatility detection failed.")
        return

    # 전략 구현 (수정된 implement_switching_strategy 호출)
    logger.info("Implementing switching strategy using loaded models and best parameters...")

    # implement_switching_strategy 호출 시 best_params 전달
    simulation_df_strat = implement_switching_strategy(
        simulation_df_vol, # 변동성 감지된 데이터 사용
        models,            # 로드된 모델 전달
        best_params        # 로드된 최적 파라미터 전달
    )

    if simulation_df_strat is None:
        logger.error("Strategy implementation failed.")
        return

    # 백테스팅 수행 (implement_strategy 에서 가져온 표준 백테스팅 함수 사용)
    logger.info(f"Running backtest using function from: {backtest_strategy_source}")
    backtest_result = backtest_strategy(simulation_df_strat.copy()) # 백테스팅 함수가 내부 수정할 수 있으므로 복사본 전달

    if backtest_result is None or not isinstance(backtest_result, tuple) or len(backtest_result) != 2:
        logger.error(f"Backtesting function from {backtest_strategy_source} returned unexpected result.")
        return

    backtest_df, performance = backtest_result # 결과 분리

    if backtest_df is None or performance is None:
        logger.error("Backtesting failed (returned None).")
        return

    # 결과 요약 출력 (기존 로직 유지)
    logger.info("\n--- Simulation Backtest Results ---")
    if isinstance(performance, dict):
        for key, value in performance.items():
            # 로그 출력 형식 개선
            key_str = key.replace('_', ' ').title()
            if isinstance(value, (float, np.floating)) and ('Return' in key_str or 'Drawdown' in key_str):
                 logger.info(f"{key_str}: {value:.2%}")
            elif isinstance(value, (float, np.floating)) and ('Capital' in key_str or 'Fee' in key_str):
                 logger.info(f"{key_str}: ${value:,.2f}")
            elif isinstance(value, (float, np.floating)):
                 logger.info(f"{key_str}: {value:.2f}")
            elif isinstance(value, (int, np.integer)):
                 logger.info(f"{key_str}: {value:,}")
            else:
                 logger.info(f"{key_str}: {value}")
    else:
        logger.warning(f"Performance result is not a dictionary: {performance}")
    logger.info("---------------------------------")

    # --- 시뮬레이션 결과 시각화 및 저장 (신규 추가) ---
    logger.info(f"Generating and saving visualizations to '{RESULTS_DIR.name}' directory...")
    initial_capital = performance.get('initial_capital', 10000) # Get initial capital

    # 1. 전체 기간 자본 성장 곡선 (로그 스케일)
    try:
        plt.figure(figsize=(14, 7))
        # backtest_df에 자본 컬럼이 있는지 확인 ('tqqq_capital', 'strategy_capital')
        if 'tqqq_capital' in backtest_df.columns: plt.plot(backtest_df.index, backtest_df['tqqq_capital'], 'b-', label='TQQQ Buy & Hold')
        if 'strategy_capital' in backtest_df.columns: plt.plot(backtest_df.index, backtest_df['strategy_capital'], 'g-', label='Strategy')
        plt.yscale('log')
        plt.title(f'Simulation Capital Growth ({start_date_str} to {end_date_str}) (Log Scale)')
        plt.xlabel('Date')
        plt.ylabel('Capital ($) (Log Scale)')
        plt.legend()
        plt.grid(True, which="both", ls="--", alpha=0.3)
        plt.savefig(RESULTS_DIR / f'simulation_capital_growth_log_{start_date_str}_to_{end_date_str}.png')
        plt.close()
    except Exception as e:
        logger.error(f"Error generating capital growth plot: {e}", exc_info=True)
        plt.close('all')

    # 2. 매매 신호 및 가격 차트
    try:
        plt.figure(figsize=(14, 10))
        ax1 = plt.subplot(211)
        # 원본 가격 컬럼 찾기 (Close 또는 TQQQ_close) - backtest_df 사용
        close_col = None
        if 'Close' in backtest_df.columns: close_col = 'Close'
        elif 'TQQQ_close' in backtest_df.columns: close_col = 'TQQQ_close'
        elif 'close' in backtest_df.columns: close_col = 'close' # backtest_strategy 결과 기준

        if close_col and not backtest_df.empty:
            ax1.plot(backtest_df.index, backtest_df[close_col], 'k-', label=f'{close_col} Price', alpha=0.8)
            # 신호 컬럼 ('signal') 사용
            if 'signal' in backtest_df.columns:
                 tqqq_buy = backtest_df[backtest_df['signal'] == 1].index
                 sqqq_buy = backtest_df[backtest_df['signal'] == -1].index
                 ax1.scatter(tqqq_buy, backtest_df.loc[tqqq_buy, close_col], color='lime', marker='^', s=60, label='Buy TQQQ', edgecolors='black', linewidth=0.5, zorder=3)
                 ax1.scatter(sqqq_buy, backtest_df.loc[sqqq_buy, close_col], color='red', marker='v', s=60, label='Buy SQQQ', edgecolors='black', linewidth=0.5, zorder=3)
        else:
            logger.warning("Close price or signal column not found/empty in backtest_df for signal plotting.")
            ax1.plot([], [], 'k-', label='Price (Not Found)')
        ax1.set_title(f'Simulation Price and Trading Signals ({start_date_str} to {end_date_str})')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 예측 확률 및 변동성 차트 (backtest_df 에 확률/변동성 정보 있는지 확인)
        ax2 = plt.subplot(212, sharex=ax1)
        plotted_probs = False
        if 'direction_prob' in backtest_df.columns: ax2.plot(backtest_df.index, backtest_df['direction_prob'], 'g-', label='Up Prob', alpha=0.7); plotted_probs=True
        if 'big_move_prob' in backtest_df.columns: ax2.plot(backtest_df.index, backtest_df['big_move_prob'], 'r-', label='Big Move Prob', alpha=0.7); plotted_probs=True
        if 'switch_prob' in backtest_df.columns: ax2.plot(backtest_df.index, backtest_df['switch_prob'], 'b-', label='Switch Prob', alpha=0.7); plotted_probs=True
        # high_volatility 는 simulation_df_vol 에 있음 (필요시 merge 또는 별도 처리)
        # 여기서는 backtest_df 에 있다고 가정하지 않음.

        ax2.set_title('Model Prediction Probabilities (if available in backtest_df)')
        ax2.set_ylabel('Probability')
        ax2.set_xlabel('Date')
        if plotted_probs: ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f'simulation_signals_probs_{start_date_str}_to_{end_date_str}.png')
        plt.close()
    except Exception as e:
        logger.error(f"Error generating signals/probs plot: {e}", exc_info=True)
        plt.close('all')

    # 3. 전체 기간 낙폭 차트
    try:
        plt.figure(figsize=(14, 7))
        # 누적 수익률 컬럼 ('cum_tqqq_return', 'cum_strategy_return') 사용
        if 'cum_tqqq_return' in backtest_df.columns and 'cum_strategy_return' in backtest_df.columns:
             tqqq_cum_returns = backtest_df['cum_tqqq_return']
             strategy_cum_returns = backtest_df['cum_strategy_return']
             tqqq_running_max = (1 + tqqq_cum_returns).cummax()
             strategy_running_max = (1 + strategy_cum_returns).cummax()
             tqqq_drawdown = ((1 + tqqq_cum_returns) / tqqq_running_max - 1) * 100
             strategy_drawdown = ((1 + strategy_cum_returns) / strategy_running_max - 1) * 100
             plt.plot(backtest_df.index, tqqq_drawdown, 'r-', label='TQQQ Buy & Hold Drawdown')
             plt.plot(backtest_df.index, strategy_drawdown, 'g-', label='Strategy Drawdown')
        else:
             logger.warning("Cumulative return columns missing for Drawdown plot.")
             plt.plot([], [], 'r-', label='TQQQ Buy & Hold Drawdown')
             plt.plot([], [], 'g-', label='Strategy Drawdown')
        plt.title(f'Simulation Drawdown Comparison ({start_date_str} to {end_date_str})')
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter())
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(RESULTS_DIR / f'simulation_drawdowns_{start_date_str}_to_{end_date_str}.png')
        plt.close()
    except Exception as e:
        logger.error(f"Error generating drawdown plot: {e}", exc_info=True)
        plt.close('all')

    # 4. 월별 수익률 히트맵
    try:
        if not backtest_df.empty and 'next_day_return' in backtest_df.columns and 'strategy_return' in backtest_df.columns:
            heatmap_df = backtest_df.copy()
            heatmap_df['year'] = heatmap_df.index.year
            heatmap_df['month'] = heatmap_df.index.month
            # 수익률 계산 (next_day_return: TQQQ, strategy_return: 전략)
            tqqq_monthly = heatmap_df.groupby(['year', 'month'])['next_day_return'].apply(
                lambda x: (1 + x.fillna(0)).prod() - 1
            ).unstack() * 100
            strategy_monthly = heatmap_df.groupby(['year', 'month'])['strategy_return'].apply(
                lambda x: (1 + x.fillna(0)).prod() - 1
            ).unstack() * 100

            fig, axes = plt.subplots(2, 1, figsize=(max(10, len(tqqq_monthly.columns)*0.8), 8), sharex=True)
            sns.heatmap(tqqq_monthly, annot=True, fmt='.1f', cmap='RdYlGn', center=0, ax=axes[0], cbar=False)
            axes[0].set_title('TQQQ Buy & Hold Monthly Returns (%)')
            axes[0].set_xlabel('')
            axes[0].set_ylabel('Year')
            sns.heatmap(strategy_monthly, annot=True, fmt='.1f', cmap='RdYlGn', center=0, ax=axes[1], cbar=True, cbar_kws={'label': 'Monthly Return (%)'})
            axes[1].set_title('Strategy Monthly Returns (%)')
            axes[1].set_xlabel('Month')
            axes[1].set_ylabel('Year')
            plt.suptitle(f'Simulation Monthly Returns ({start_date_str} to {end_date_str})', y=1.02)
            plt.tight_layout()
            plt.savefig(RESULTS_DIR / f'simulation_monthly_returns_heatmap_{start_date_str}_to_{end_date_str}.png')
            plt.close(fig)
        else:
            logger.warning("Data missing for Monthly Heatmap.")
    except Exception as e:
        logger.error(f"Error generating monthly heatmap: {e}", exc_info=True)
        plt.close('all')

    # 5. 성능 요약 텍스트 파일 저장
    summary_file_path = RESULTS_DIR / f'simulation_summary_{start_date_str}_to_{end_date_str}.txt'
    try:
        with open(summary_file_path, 'w', encoding='utf-8') as f:
            f.write(f"# Simulation Performance Summary\n\n")
            f.write(f"## Simulation Period: {start_date_str} to {end_date_str}\n\n")
            f.write("## Performance Metrics\n")
            if isinstance(performance, dict):
                for key, value in performance.items():
                    key_str = key.replace('_', ' ').title()
                    if isinstance(value, (float, np.floating)) and ('Return' in key_str or 'Drawdown' in key_str):
                        f.write(f"- {key_str}: {value:.2%}\n")
                    elif isinstance(value, (float, np.floating)) and ('Capital' in key_str or 'Fee' in key_str):
                        f.write(f"- {key_str}: ${value:,.2f}\n")
                    elif isinstance(value, (float, np.floating)):
                        f.write(f"- {key_str}: {value:.2f}\n")
                    elif isinstance(value, (int, np.integer)):
                        f.write(f"- {key_str}: {value:,}\n")
                    else:
                        f.write(f"- {key_str}: {value}\n")
            else:
                f.write("Performance data is not available in the expected format.\n")
            f.write("\n## Applied Parameters (from loaded file)\n")
            if best_params:
                 sorted_params = sorted(best_params.items())
                 for key, value in sorted_params:
                      if isinstance(value, float):
                          f.write(f"- {key}: {value:.4f}\n")
                      else:
                          f.write(f"- {key}: {value}\n")
            else:
                 f.write("No parameters loaded.\n")
        logger.info(f"Performance summary saved to: {summary_file_path}")
    except Exception as e:
        logger.error(f"Failed to save performance summary: {e}", exc_info=True)
    # --- 시각화 및 저장 끝 ---

    # 상세 결과 CSV 저장 (기존 유지)
    try:
        log_file_path = RESULTS_DIR / f'simulation_details_{start_date_str}_to_{end_date_str}.csv'
        # 불필요 컬럼 제거 (예시, 필요시 수정)
        cols_to_drop = ['year', 'month', 'tqqq_capital_vis', 'strategy_capital_vis'] # visualize_results 관련 임시 컬럼 제거
        backtest_df_to_save = backtest_df.drop(columns=[col for col in cols_to_drop if col in backtest_df.columns])
        backtest_df_to_save.to_csv(log_file_path)
        logger.info(f"Detailed simulation results saved to: {log_file_path}")
    except Exception as e:
        logger.error(f"Failed to save detailed simulation results: {e}")

    logger.info(f"Simulation finished for period: {start_date_str} to {end_date_str}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run trading strategy simulation for a specified period.")
    parser.add_argument("--start_date", required=True, help="Start date for simulation (YYYY-MM-DD)")
    parser.add_argument("--end_date", required=True, help="End date for simulation (YYYY-MM-DD)")

    args = parser.parse_args()

    run_simulation(args.start_date, args.end_date) 