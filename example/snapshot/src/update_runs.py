#!/usr/bin/env python3
"""
UPDATE RUNS - Compute runs for new data only
Loads only new trades from combined and appends new runs

Usage:
    python src/update_runs.py

Author: Quant Research Team
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

print("="*80)
print("UPDATE RUNS - COMPUTE FOR NEW DATA")
print("="*80)

script_dir = Path(__file__).parent
data_dir = script_dir.parent / 'data'
combined_file = data_dir / 'BTCUSDT_combined.csv'
runs_file = data_dir / 'BTCUSDT_runs.csv'

# ============================================================================
# COMPUTE RUNS FUNCTION
# ============================================================================
def compute_runs_from_batch(prices, timestamps, sizes, sides):
    """Compute runs from numpy arrays"""
    runs = []
    dirs = np.sign(np.diff(prices))
    time_diffs = np.diff(timestamps.astype('int64')) / 1e9
    gaps = np.where(time_diffs > 300)[0]

    start_idx, cur_dir = 0, None

    for i in range(len(dirs)):
        if i in gaps:
            start_idx, cur_dir = i + 1, None
            continue

        if dirs[i] == 0:
            continue

        if cur_dir is None:
            cur_dir, start_idx = dirs[i], i
            continue

        if dirs[i] != cur_dir:
            p_start, p_end = prices[start_idx], prices[i]
            delta_p = p_end - p_start
            n_ticks = i - start_idx + 1

            if n_ticks >= 5 and abs(delta_p) >= 1.0:
                Q = sizes[start_idx:i+1].sum()
                t_start = timestamps[start_idx]
                t_end = timestamps[i]
                duration = max(float((t_end - t_start) / np.timedelta64(1, 's')), 1.0)

                q_run = np.where(sides[start_idx:i+1] == 'B',
                                sizes[start_idx:i+1],
                                -sizes[start_idx:i+1]).sum()

                runs.append({
                    'V_eff': Q / abs(delta_p),
                    't_start': t_start,
                    't_end': t_end,
                    'duration': duration,
                    'price_start': p_start,
                    'price_end': p_end,
                    'price_mid': (p_start + p_end) / 2,
                    'dir': 'up' if cur_dir > 0 else 'down',
                    'q': q_run,
                    'delta_p': delta_p,
                    'velocity': abs(delta_p) / duration,
                    'Q': Q,
                    'n_ticks': n_ticks
                })

            start_idx, cur_dir = i, dirs[i]

    return runs

# ============================================================================
# LOAD EXISTING RUNS
# ============================================================================
print("\nLoading existing runs...")
df_runs_old = pd.read_csv(runs_file)
df_runs_old['t_end'] = pd.to_datetime(df_runs_old['t_end'])
last_run_date = df_runs_old['t_end'].max().date()
print(f"Last run date: {last_run_date}")
print(f"Total existing runs: {len(df_runs_old):,}")

# ============================================================================
# LOAD NEW DATA FROM COMBINED
# ============================================================================
print("\nLoading new trades from combined...")
df_combined = pd.read_csv(combined_file)
df_combined['timestamp'] = pd.to_datetime(df_combined['timestamp'])

# Get only new data (after last run date)
df_new = df_combined[df_combined['timestamp'].dt.date > last_run_date].copy()

if len(df_new) == 0:
    print("No new data to process!")
    sys.exit(0)

print(f"New trades: {len(df_new):,}")
print(f"Date range: {df_new['timestamp'].min()} -> {df_new['timestamp'].max()}")

# ============================================================================
# COMPUTE RUNS
# ============================================================================
print("\nComputing runs for new data...")
prices = df_new['price'].values
timestamps = df_new['timestamp'].values
sizes = df_new['size'].values
sides = df_new['side'].values

new_runs = compute_runs_from_batch(prices, timestamps, sizes, sides)
print(f"New runs: {len(new_runs):,}")

if len(new_runs) == 0:
    print("No new runs generated!")
    sys.exit(0)

# ============================================================================
# APPEND TO RUNS FILE
# ============================================================================
print("\nAppending to runs file...")
df_runs_new = pd.DataFrame(new_runs)
df_runs_new.to_csv(runs_file, mode='a', header=False, index=False)

print(f"\n[OK] Update complete!")
print(f"Total runs now: {len(df_runs_old) + len(df_runs_new):,}")
print("="*80)
