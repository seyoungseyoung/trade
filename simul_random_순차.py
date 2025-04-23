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
from joblib import Memory # <--- Import joblib Memory
import sys
import pickle # <--- Import pickle
import atexit # <--- To save cache on exit

# --- Function Imports ---
# Attempt to import from implement_strategy
try:
    from implement_strategy import detect_volatility, implement_switching_strategy, backtest_strategy
    print("Successfully imported functions from implement_strategy.")
    backtest_strategy_source = 'implement_strategy'
except ImportError as e:
    print(f"ERROR: Could not import required functions from implement_strategy.py: {e}")
    print("Ensure implement_strategy.py is in the same directory and contains detect_volatility, implement_switching_strategy, and backtest_strategy.")
    import sys
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
CACHE_FILE = BASE_DIR / 'sequential_cache.pkl' # <--- Define cache file path

# --- Cache Handling Functions ---
def load_cache(filepath):
    """Loads cache dictionary from a pickle file."""
    try:
        with open(filepath, 'rb') as f:
            cache = pickle.load(f)
        logger.info(f"Loaded cache from {filepath}, containing {len(cache)} items.")
        return cache
    except FileNotFoundError:
        logger.info(f"Cache file {filepath} not found. Starting with an empty cache.")
        return {}
    except Exception as e:
        logger.error(f"Error loading cache from {filepath}: {e}. Starting with an empty cache.", exc_info=True)
        return {}

def save_cache(cache_dict, filepath):
    """Saves the cache dictionary to a pickle file."""
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(cache_dict, f)
        logger.info(f"Saved cache to {filepath}, containing {len(cache_dict)} items.")
    except Exception as e:
        logger.error(f"Error saving cache to {filepath}: {e}.", exc_info=True)

# --- Global Cache Variable (will be loaded/saved) ---
backtest_cache = {}

# --- Register save_cache to run on exit ---
# This ensures cache is saved even on unexpected exits (if possible)
atexit.register(save_cache, backtest_cache, CACHE_FILE)

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

# --- Core Simulation Logic (remove cache decorator, restore signature) ---
# Remove @memory.cache decorator
def run_single_period_backtest(start_date, end_date, features_df, models, best_params):
    """Runs the simulation for a specific period and returns performance.
       (Restored original signature, caching handled outside)
    """
    logger.debug(f"Running backtest for period: {start_date.date()} to {end_date.date()}")
    # --- RESTORED LOGIC --- 
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
        # Ensure backtest_strategy is correctly imported and functional
        try:
            backtest_result = backtest_strategy(simulation_df_strat.copy()) # Pass copy
        except NameError:
            logger.error("backtest_strategy function not found. Check import from implement_strategy.")
            return None
        except Exception as bt_e:
            logger.error(f"Error during backtest_strategy call for {start_date.date()} to {end_date.date()}: {bt_e}", exc_info=True)
            return None
            
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
    # --- END RESTORED LOGIC --- 

# --- Main Execution ---
if __name__ == "__main__":
    # Load the cache at the beginning
    backtest_cache = load_cache(CACHE_FILE) 

    parser = argparse.ArgumentParser(description="Find the worst performing historical periods for the trading strategy.")
    parser.add_argument("--min_duration", type=int, required=True, help="Minimum duration in years to test (e.g., 1).")
    parser.add_argument("--max_duration", type=int, required=True, help="Maximum duration in years to test (e.g., 25).")
    parser.add_argument("--metric", type=str, default="strategy_total_return",
                        help="Performance metric to minimize/maximize (e.g., 'strategy_total_return', 'strategy_sharpe', 'strategy_max_drawdown'). Default: 'strategy_total_return'.")
    # Add options for volatility parameters if needed, or rely on loaded best_params
    # parser.add_argument("--vix_threshold", type=float, help="Override VIX threshold from params file.")
    # ... other params

    args = parser.parse_args()

    if args.min_duration <= 0 or args.max_duration < args.min_duration:
        logger.error("Invalid duration specified. Min must be > 0 and Max >= Min.")
        sys.exit(1)

    # Define which metrics are minimized vs maximized based on actual keys
    METRICS_TO_MINIMIZE = ['strategy_total_return', 'strategy_annual_return', 'strategy_sharpe', 'final_strategy_capital']
    METRICS_TO_MAXIMIZE = ['strategy_max_drawdown', 'tqqq_max_drawdown']

    if args.metric not in METRICS_TO_MINIMIZE + METRICS_TO_MAXIMIZE:
        logger.error(f"Unsupported metric: {args.metric}. Choose from {METRICS_TO_MINIMIZE + METRICS_TO_MAXIMIZE}")
        sys.exit(1)

    minimize_metric = args.metric in METRICS_TO_MINIMIZE

    logger.info(f"Starting sequential search for worst periods (durations {args.min_duration}-{args.max_duration} years).")
    logger.info(f"Optimizing metric: {args.metric} ({'Minimize' if minimize_metric else 'Maximize'} to find worst)")

    # 1. Load data
    models, best_params, features_df = load_simulation_data()
    if models is None or best_params is None or features_df is None or features_df.empty:
        logger.error("Failed to load necessary data/models. Exiting.")
        sys.exit(1)

    # --- Sequential Search Loop ---
    all_worst_periods = {} # Stores the final worst period found for each duration
    min_data_date = features_df.index.min()
    max_data_date = features_df.index.max()

    try:
        # Outer loop: Iterate through durations
        for duration in range(args.min_duration, args.max_duration + 1):
            logger.info(f"Starting search for worst {duration}-year period...")
            # Initialize trackers for the current duration
            worst_metric_for_duration = float('inf') if minimize_metric else float('-inf')
            worst_start_for_duration = None
            worst_end_date_for_duration = None
            worst_performance_for_duration = None
            num_tested_for_duration = 0
            num_skipped_end_date_for_duration = 0
            num_skipped_backtest_fail_for_duration = 0

            # Inner loop: Iterate through all possible start dates
            for start_date in features_df.index:
                # Calculate potential end date
                end_date = start_date + relativedelta(years=duration) - pd.Timedelta(days=1)

                # Ensure end_date does not exceed available data
                if end_date > max_data_date:
                    num_skipped_end_date_for_duration += 1
                    continue

                # Ensure start_date allows for a minimal period (redundant check here, but safe)
                if start_date < min_data_date:
                     continue

                # --- Cache Check --- 
                cache_key = (start_date, end_date)
                if cache_key in backtest_cache:
                    performance = backtest_cache[cache_key]
                    logger.debug(f"Cache hit for period {start_date.date()} - {end_date.date()}")
                    # Need to increment tested counter even on cache hit?
                    # Let's assume yes for tracking purposes.
                    num_tested_for_duration += 1 
                else:
                    # Cache miss: Run the actual backtest
                    logger.debug(f"Cache miss for period {start_date.date()} - {end_date.date()}. Running backtest.")
                    performance = run_single_period_backtest(start_date, end_date, 
                                                             features_df, # Pass actual data
                                                             models, 
                                                             best_params)
                    num_tested_for_duration += 1
                    # Store result in cache if successful
                    if performance is not None:
                        backtest_cache[cache_key] = performance
                # --- End Cache Check --- 

                if performance is None:
                    num_skipped_backtest_fail_for_duration += 1
                    continue

                current_metric_value = performance.get(args.metric)
                if current_metric_value is None:
                    # Optionally log missing metric warning
                    continue

                # Check if this period is worse than the current worst FOR THIS DURATION
                is_worse_for_duration = False
                if minimize_metric:
                    if current_metric_value < worst_metric_for_duration:
                        is_worse_for_duration = True
                else: # Maximize
                    if current_metric_value > worst_metric_for_duration:
                        is_worse_for_duration = True

                if is_worse_for_duration:
                    # Update the trackers for the current duration
                    worst_metric_for_duration = current_metric_value
                    worst_start_for_duration = start_date
                    worst_end_date_for_duration = features_df.loc[start_date:end_date].index.max() # Actual end date
                    worst_performance_for_duration = performance

                    # --- KEY CHANGE: Update the main result dictionary immediately --- 
                    # This stores the *provisional* worst result found so far for this duration
                    all_worst_periods[duration] = {
                        'start_date': worst_start_for_duration.strftime('%Y-%m-%d'),
                        'end_date': worst_end_date_for_duration.strftime('%Y-%m-%d'),
                        'metric_value': worst_metric_for_duration,
                        'metric_name': args.metric,
                        'performance_details': worst_performance_for_duration
                    }
                    # Optional: logger.debug(...) to show updates

            # Log summary after completing the inner loop for a duration
            logger.info(f"Finished search for {duration}-year period.")
            logger.info(f"  Tested start dates: {num_tested_for_duration}")
            logger.info(f"  Skipped (end date out of range): {num_skipped_end_date_for_duration}")
            logger.info(f"  Skipped (backtest/metric failed): {num_skipped_backtest_fail_for_duration}")
            if worst_start_for_duration is None:
                logger.warning(f"  No valid worst period found for {duration} years.")
            else:
                logger.info(f"  Worst {duration}-year period found: {worst_start_for_duration.date()} to {worst_end_date_for_duration.date()} -> {args.metric}={worst_metric_for_duration:.4f}")

    except KeyboardInterrupt:
        logger.warning("\n--- Process interrupted by user (Ctrl+C). Saving cache and reporting results... ---")
        print("\n--- Process interrupted by user (Ctrl+C). Saving cache and reporting results... ---")
        # Cache saving is handled by atexit or finally block

    finally:
        # Ensure cache is saved on any exit (normal, Ctrl+C, error)
        save_cache(backtest_cache, CACHE_FILE) 

    # 3. Report final results
    logger.info("\n--- Worst Period Search Summary ---")
    if all_worst_periods:
        sorted_durations = sorted(all_worst_periods.keys())
        for duration in sorted_durations:
            data = all_worst_periods[duration]
            metric_value_str = "N/A"
            if pd.notna(data['metric_value']):
                 metric_value_str = f"{data['metric_value']:.4f}"
                 # Re-apply percentage formatting if needed
                 percentage_metrics = [
                     'final_strategy_return', 'final_tqqq_return', 'strategy_total_return',
                     'annual_strategy_return', 'annual_tqqq_return', 'strategy_annual_return',
                     'max_drawdown', 'max_strategy_drawdown', 'max_tqqq_drawdown', 'strategy_max_drawdown'
                 ]
                 if data['metric_name'] in percentage_metrics:
                     metric_value_str = f"{data['metric_value']:.2%}"

            print(f"\n--- Duration: {duration:<2} years --- ")
            print(f"Worst Period: {data['start_date']} to {data['end_date']}")
            print(f"Metric Value ({data['metric_name']}): {metric_value_str}")

            # --- Print Detailed Performance --- 
            print("  Detailed Performance for this Period:")
            performance_details = data.get('performance_details')
            if performance_details and isinstance(performance_details, dict):
                sorted_items = sorted(performance_details.items())
                for key, value in sorted_items:
                    key_str = key.replace('_', ' ').title()
                    if isinstance(value, (float, np.floating)) and ('Return' in key or 'Drawdown' in key or 'Ratio' in key or 'sharpe' in key.lower() or 'volatility' in key.lower()):
                         if pd.notna(value):
                             if 'drawdown' in key.lower() or 'Return' in key: 
                                 print(f"    - {key_str:<25}: {value:.2%}")
                             else: 
                                 print(f"    - {key_str:<25}: {value:.2f}")
                         else:
                             print(f"    - {key_str:<25}: N/A")
                    elif isinstance(value, (float, np.floating)) and ('capital' in key.lower() or 'fee' in key.lower()):
                         print(f"    - {key_str:<25}: ${value:,.2f}")
                    elif isinstance(value, (int, np.integer)):
                         print(f"    - {key_str:<25}: {value:,}")
                    elif isinstance(value, str) and 'period' in key.lower():
                         pass
                    else:
                         print(f"    - {key_str:<25}: {value}")
            else:
                print("    Detailed performance data not available.")
            # --- End Detailed Performance ---

    else:
        print("No worst periods were found.")
    logger.info("----------------------------------")

    # Optional: Save results to a file
    # results_path = BASE_DIR / f'worst_periods_summary_{args.metric}.json'
    # try:
    #     with open(results_path, 'w') as f:
    #         json.dump(all_worst_periods, f, indent=4)
    #     logger.info(f"Worst periods summary saved to {results_path}")
    # except Exception as e:
    #     logger.error(f"Failed to save summary results: {e}")

    logger.info("Worst period sequential search finished.") 