import pandas as pd
import numpy as np
import os
import joblib
import json
from datetime import datetime
from pathlib import Path
import argparse
import logging
from dateutil.relativedelta import relativedelta # For accurate date math
import random # <--- Add random import
import sys # <--- Import sys

# --- Function Imports ---
# Attempt to import from implement_strategy
try:
    from implement_strategy import detect_volatility, implement_switching_strategy, backtest_strategy
    print("Successfully imported functions from implement_strategy.")
    backtest_strategy_source = 'implement_strategy'
except ImportError as e:
    print(f"ERROR: Could not import required functions from implement_strategy.py: {e}")
    print("Ensure implement_strategy.py is in the same directory and contains detect_volatility, implement_switching_strategy, and backtest_strategy.")
    sys.exit(1)

# --- Logging Setup ---
# Configure logging (similar to simul.py, but maybe different file name)
log_file_path = Path(__file__).resolve().parent / 'simulation_random_search.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Path Definitions ---
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
MODEL_DIR = BASE_DIR / 'models'
# RESULTS_DIR not strictly needed here unless saving detailed worst-case results

# --- Copied/Adapted Functions from simul.py ---

def load_simulation_data():
    """Loads models, parameters, and feature data needed for simulations.
       (Copied and adapted from simul.py - ensures consistency)
    """
    logger.info("Loading models, parameters, and feature data...")
    models = None
    best_params = None
    features_df = None

    # Model Load (same as simul.py)
    model_path = MODEL_DIR / 'final_optimized_model.pkl'
    if model_path.exists():
        try:
            models = joblib.load(model_path)
            required_keys = ['direction_model', 'big_move_model', 'switch_model', 'scaler']
            if not all(key in models for key in required_keys):
                logger.warning(f"{model_path} does not contain all required model keys. Attempting individual loads...")
                models = None
            else:
                logger.info(f"Loaded combined model data from {model_path}")
        except Exception as e:
            logger.error(f"Error loading combined model from {model_path}: {e}", exc_info=True)
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

    # Parameter Load (same logic as simul.py to find the correct file)
    params_path_to_load = None
    search_pattern_ir_lb = MODEL_DIR / 'final_optimized_params_ir_lb*.json'
    params_path_rolling = MODEL_DIR / 'final_optimized_params_rolling.json'
    params_path_standard = MODEL_DIR / 'final_optimized_params.json'

    ir_lb_files = sorted(list(MODEL_DIR.glob('final_optimized_params_ir_lb*.json')), reverse=True)
    if ir_lb_files:
        params_path_to_load = ir_lb_files[0]
        logger.info(f"Found IR+Lookback optimized parameters file: {params_path_to_load}")
    elif params_path_rolling.exists():
        params_path_to_load = params_path_rolling
        logger.info(f"Found rolling parameters file: {params_path_to_load}")
    elif params_path_standard.exists():
        params_path_to_load = params_path_standard
        logger.info(f"Found standard parameters file: {params_path_to_load}")
    else:
        logger.error(f"Optimized parameters file not found. Searched patterns/files: {search_pattern_ir_lb.name}, {params_path_rolling.name}, {params_path_standard.name}")
        return None, None, None

    try:
        with open(params_path_to_load, 'r') as f:
            best_params = json.load(f)
        logger.info(f"Loaded optimized parameters from {params_path_to_load}")
    except Exception as e:
        logger.error(f"Error loading optimized parameters from {params_path_to_load}: {e}", exc_info=True)
        return None, None, None

    # Feature Data Load (same as simul.py)
    features_path = DATA_DIR / 'tqqq_features.csv'
    if features_path.exists():
        try:
            features_df = pd.read_csv(features_path, index_col=0)
            features_df.index = pd.to_datetime(features_df.index)
            if features_df.index.tz is not None:
                logger.info("Converting features_df index to timezone-naive.")
                features_df.index = features_df.index.tz_localize(None)
            # Ensure data is sorted by date for time-series operations
            features_df.sort_index(inplace=True)
            logger.info(f"Loaded features data from {features_path}: {features_df.shape}")
            logger.info(f"Data range: {features_df.index.min()} to {features_df.index.max()}")
        except Exception as e:
            logger.error(f"Error loading features data from {features_path}: {e}", exc_info=True)
            return None, None, None
    else:
        logger.error(f"Features data file not found: {features_path}")
        return None, None, None

    return models, best_params, features_df

# --- Core Simulation Logic (Placeholder) ---
def run_single_period_backtest(start_date, end_date, features_df, models, best_params):
    """Runs the simulation for a specific period and returns performance."""
    # This function will contain the core logic: filter, detect, implement, backtest
    logger.debug(f"Running backtest for period: {start_date.date()} to {end_date.date()}")

    try:
        # 1. Filter data for the period
        simulation_df = features_df.loc[start_date:end_date].copy()
        if simulation_df.empty or len(simulation_df) < 2: # Need at least 2 data points for returns etc.
            logger.warning(f"Not enough data for period {start_date.date()} to {end_date.date()} (rows: {len(simulation_df)}). Skipping.")
            return None

        # 2. Detect volatility (using loaded best_params)
        simulation_df_vol = detect_volatility(
            simulation_df, # Use the filtered df
            best_params.get('vix_threshold', 25.0),
            best_params.get('vix_ratio_threshold', 1.2),
            best_params.get('price_change_threshold', 0.03)
        )
        if simulation_df_vol is None:
            logger.error(f"Volatility detection failed for period {start_date.date()} to {end_date.date()}.")
            return None

        # 3. Implement strategy (using loaded models and best_params)
        simulation_df_strat = implement_switching_strategy(
            simulation_df_vol,
            models,
            best_params
        )
        if simulation_df_strat is None:
            logger.error(f"Strategy implementation failed for period {start_date.date()} to {end_date.date()}.")
            return None

        # 4. Run backtest (using imported function)
        backtest_result = backtest_strategy(simulation_df_strat.copy()) # Pass copy

        if backtest_result is None or not isinstance(backtest_result, tuple) or len(backtest_result) != 2:
            logger.error(f"Backtesting function returned unexpected result for period {start_date.date()} to {end_date.date()}.")
            return None

        backtest_df, performance = backtest_result

        if performance is None:
            logger.error(f"Backtesting failed (returned None performance) for period {start_date.date()} to {end_date.date()}.")
            return None

        # 5. Return only the performance dictionary
        return performance

    except Exception as e:
        logger.error(f"Error during backtest for period {start_date.date()} to {end_date.date()}: {e}", exc_info=True)
        return None

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find the worst performing historical periods for the trading strategy using random sampling.")
    parser.add_argument("--min_duration", type=int, required=True, help="Minimum duration in years to test (e.g., 1).")
    parser.add_argument("--max_duration", type=int, required=True, help="Maximum duration in years to test (e.g., 25).")
    parser.add_argument("--metric", type=str, default="strategy_total_return",
                        help="Performance metric to minimize/maximize (e.g., 'strategy_total_return', 'sharpe_ratio', 'strategy_max_drawdown'). Default: 'strategy_total_return'.")
    parser.add_argument("--num_trials", type=int, default=100, help="Number of random periods to test. Default: 100.")
    # Add options for volatility parameters if needed, or rely on loaded best_params
    # parser.add_argument("--vix_threshold", type=float, help="Override VIX threshold from params file.")
    # ... other params

    args = parser.parse_args()

    if args.min_duration <= 0 or args.max_duration < args.min_duration:
        logger.error("Invalid duration specified. Min must be > 0 and Max >= Min.")
        sys.exit(1)

    # Define which metrics are minimized vs maximized
    # Based on the available keys seen in logs
    METRICS_TO_MINIMIZE = ['strategy_total_return', 'strategy_annual_return', 'strategy_sharpe', 'final_strategy_capital'] # Updated list
    METRICS_TO_MAXIMIZE = ['strategy_max_drawdown', 'tqqq_max_drawdown'] # Updated list (add others if needed)

    # Ensure the default/chosen metric is actually in the performance dict keys (re-check based on available keys)
    # This check should now pass with the updated lists and default value
    if args.metric not in METRICS_TO_MINIMIZE + METRICS_TO_MAXIMIZE:
        logger.error(f"Unsupported metric: {args.metric}. Choose from {METRICS_TO_MINIMIZE + METRICS_TO_MAXIMIZE}")
        sys.exit(1) # This should work now with sys imported

    minimize_metric = args.metric in METRICS_TO_MINIMIZE

    logger.info(f"Starting random search for worst period across {args.num_trials} trials.")
    logger.info(f"Durations between {args.min_duration} and {args.max_duration} years will be tested.")
    logger.info(f"Optimizing metric: {args.metric} ({'Minimize' if minimize_metric else 'Maximize'} to find worst)")

    # 1. Load data
    models, best_params, features_df = load_simulation_data()

    if models is None or best_params is None or features_df is None or features_df.empty:
        logger.error("Failed to load necessary data/models. Exiting.")
        sys.exit(1)

    # --- Random Sampling Loop ---
    overall_worst_period = {
        'start_date': None,
        'end_date': None,
        'duration': None,
        'metric_value': float('inf') if minimize_metric else float('-inf'),
        'metric_name': args.metric,
        'performance_details': None
    }
    num_valid_trials = 0
    num_failed_backtests = 0

    # Determine valid range for start dates
    min_data_date = features_df.index.min()
    max_data_date = features_df.index.max()
    # To ensure a full duration is possible, the latest possible start date
    # depends on the *minimum* duration we might sample.
    latest_possible_start_for_min_duration = max_data_date - relativedelta(years=args.min_duration) + pd.Timedelta(days=1)
    # Filter the index to get potential start dates
    potential_start_dates = features_df.index[(features_df.index >= min_data_date) & (features_df.index <= latest_possible_start_for_min_duration)]

    if potential_start_dates.empty:
        logger.error(f"No valid start dates available in the data range to satisfy even the minimum duration of {args.min_duration} years. Check data or duration range.")
        sys.exit(1)
    logger.info(f"Considering {len(potential_start_dates)} potential start dates between {potential_start_dates.min().date()} and {potential_start_dates.max().date()}")


    try:
        for i in range(args.num_trials):
            # 1. Randomly select duration
            duration_years = random.randint(args.min_duration, args.max_duration)

            # 2. Randomly select a valid start date
            start_date = random.choice(potential_start_dates)

            # 3. Calculate end date
            end_date = start_date + relativedelta(years=duration_years) - pd.Timedelta(days=1)

            # 4. Ensure end_date is within the actual data range (double check, though potential_start_dates should help)
            if end_date > max_data_date:
                 # This check might be redundant if potential_start_dates is calculated correctly, but safe to keep.
                 # If this happens often, review the potential_start_dates logic.
                 logger.warning(f"Trial {i+1}: Calculated end date {end_date.date()} for start {start_date.date()} ({duration_years}y) exceeds max data date {max_data_date.date()}. Skipping.")
                 continue

            logger.debug(f"Trial {i+1}/{args.num_trials}: Testing {duration_years}-year period from {start_date.date()} to {end_date.date()}")

            # 5. Run backtest
            performance = run_single_period_backtest(start_date, end_date, features_df, models, best_params)

            if performance is None:
                num_failed_backtests += 1
                continue # Skip if backtest failed

            num_valid_trials += 1
            current_metric_value = performance.get(args.metric)

            if current_metric_value is None:
                # Log a warning and the available keys if the desired metric is missing
                available_keys = list(performance.keys()) if isinstance(performance, dict) else 'N/A'
                logger.warning(f"Trial {i+1}: Metric '{args.metric}' not found in performance results for period {start_date.date()} to {end_date.date()}. Available keys: {available_keys}. Skipping comparison.")
                continue

            # 6. Check if this is the worst period found so far
            is_overall_worse = False
            current_worst_value = overall_worst_period['metric_value']
            if minimize_metric:
                if current_metric_value < current_worst_value:
                    is_overall_worse = True
            else: # Maximize metric for worst case (e.g., drawdown)
                if current_metric_value > current_worst_value:
                    is_overall_worse = True

            if is_overall_worse:
                logger.info(f"Trial {i+1}: New overall worst period found! {duration_years} years from {start_date.date()} to {end_date.date()} -> {args.metric}={current_metric_value:.4f} (Previous worst: {current_worst_value:.4f})")
                overall_worst_period['start_date'] = start_date
                actual_end_date = features_df.loc[start_date:end_date].index.max()
                overall_worst_period['end_date'] = actual_end_date
                overall_worst_period['duration'] = duration_years
                overall_worst_period['metric_value'] = current_metric_value
                overall_worst_period['performance_details'] = performance

    except KeyboardInterrupt:
        logger.warning("\n--- Process interrupted by user (Ctrl+C). Reporting best result found so far. ---")
        print("\n--- Process interrupted by user (Ctrl+C). Reporting best result found so far. ---")
    except Exception as e:
        logger.error(f"An unexpected error occurred during the trial loop: {e}", exc_info=True)
        print(f"An unexpected error occurred: {e}")

    # --- Report Final Results ---
    logger.info(f"\n--- Random Search Summary ({num_valid_trials}/{args.num_trials} valid trials completed) ---")
    if overall_worst_period['start_date'] is not None:
        data = overall_worst_period
        start_str = data['start_date'].strftime('%Y-%m-%d')
        end_str = data['end_date'].strftime('%Y-%m-%d')
        duration_val = data['duration']
        metric_val = data['metric_value']
        metric_name = data['metric_name']

        metric_value_str = f"{metric_val:.4f}"
        percentage_metrics = [
            'final_strategy_return', 'final_tqqq_return',
            'annual_strategy_return', 'annual_tqqq_return',
            'max_drawdown', 'max_strategy_drawdown', 'max_tqqq_drawdown'
        ]
        if metric_name in percentage_metrics:
            if pd.notna(metric_val):
                metric_value_str = f"{metric_val:.2%}"
            else:
                metric_value_str = "N/A"

        print(f"Overall Worst Period Found:")
        print(f"  Duration: {duration_val} years")
        print(f"  Period:   {start_str} to {end_str}")
        print(f"  Metric:   {metric_name} = {metric_value_str}")

        # --- Print Detailed Performance --- 
        print("\n  ## Detailed Performance for this Period ##")
        performance_details = data.get('performance_details')
        if performance_details and isinstance(performance_details, dict):
            # Sort items for consistent order, if desired
            sorted_items = sorted(performance_details.items())
            for key, value in sorted_items:
                key_str = key.replace('_', ' ').title()
                # Apply formatting similar to simul.py or previous reporting
                if isinstance(value, (float, np.floating)) and ('Return' in key or 'Drawdown' in key or 'Ratio' in key or 'sharpe' in key.lower() or 'volatility' in key.lower()):
                     # Check for NaN before formatting percentage
                     if pd.notna(value):
                         if 'drawdown' in key.lower() or 'Return' in key: # Explicitly format returns and drawdown as percentage
                             print(f"  - {key_str:<25}: {value:.2%}")
                         else: # Sharpe, Volatility etc. as float
                             print(f"  - {key_str:<25}: {value:.2f}")
                     else:
                         print(f"  - {key_str:<25}: N/A")
                elif isinstance(value, (float, np.floating)) and ('capital' in key.lower() or 'fee' in key.lower()):
                     print(f"  - {key_str:<25}: ${value:,.2f}")
                elif isinstance(value, (int, np.integer)):
                     print(f"  - {key_str:<25}: {value:,}")
                elif isinstance(value, str) and 'period' in key.lower(): # Don't repeat period info
                     pass
                else:
                     print(f"  - {key_str:<25}: {value}")
        else:
            print("  Detailed performance data not available.")
        # --- End Detailed Performance ---

    else:
        print("No valid periods were successfully tested, or no worse period than initial default was found.")

    if num_failed_backtests > 0:
        logger.warning(f"Note: {num_failed_backtests} backtests failed during the trials.")

    logger.info("----------------------------------")

    # Optional: Save results to a file (save the single worst period dict)
    # results_path = BASE_DIR / f'worst_period_random_summary_{args.metric}.json'
    # try:
    #     # Only save if a valid period was found
    #     if overall_worst_period['start_date'] is not None:
    #          # Convert datetime objects to strings for JSON serialization
    #          save_data = overall_worst_period.copy()
    #          save_data['start_date'] = save_data['start_date'].strftime('%Y-%m-%d')
    #          save_data['end_date'] = save_data['end_date'].strftime('%Y-%m-%d')
    #          with open(results_path, 'w') as f:
    #              json.dump(save_data, f, indent=4)
    #          logger.info(f"Worst period summary saved to {results_path}")
    #     else:
    #          logger.info("No valid worst period found to save.")
    # except Exception as e:
    #     logger.error(f"Failed to save summary results: {e}")

    logger.info("Worst period random search finished.") 