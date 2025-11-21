#!/usr/bin/env python3
"""
COMPUTE RUNS - BATCH PROCESSING WITH DETAILED LOGS
Process 1M rows at a time with progress logging

Author: Quant Research Team
"""
import pandas as pd
import numpy as np
from pathlib import Path
import time
import sys

print("="*80)
print("COMPUTE RUNS - BATCH PROCESSING (1M rows/batch)")
print("="*80)

# ============================================================================
# CONFIGURATION
# ============================================================================
script_dir = Path(__file__).parent
data_dir = script_dir.parent / 'data'
output_file = data_dir / 'BTCUSDT_runs.csv'

BATCH_SIZE = 1_000_000  # Process 1M rows at a time

print(f"\nConfiguration:")
print(f"  Input: {data_dir / 'BTCUSDT_combined_hyperliquid.csv'}")
print(f"  Output: {output_file}")
print(f"  Batch size: {BATCH_SIZE:,} rows")
sys.stdout.flush()

# ============================================================================
# COMPUTE RUNS FUNCTION
# ============================================================================

def compute_runs_from_batch(prices, timestamps, sizes, sides):
    """Compute runs from numpy arrays"""
    runs = []

    dirs = np.sign(np.diff(prices))
    time_diffs = np.diff(timestamps.astype('int64')) / 1e9  # To seconds
    gaps = np.where(time_diffs > 300)[0]  # 5-minute gaps

    start_idx, cur_dir = 0, None

    for i in range(len(dirs)):
        # Reset on time gaps
        if i in gaps:
            start_idx, cur_dir = i + 1, None
            continue

        # Skip flat movements
        if dirs[i] == 0:
            continue

        # Start new run
        if cur_dir is None:
            cur_dir, start_idx = dirs[i], i
            continue

        # Direction change = end of run
        if dirs[i] != cur_dir:
            p_start, p_end = prices[start_idx], prices[i]
            delta_p = p_end - p_start
            n_ticks = i - start_idx + 1

            # Filter: min 5 ticks, min $1 movement
            if n_ticks >= 5 and abs(delta_p) >= 1.0:
                Q = sizes[start_idx:i+1].sum()
                t_start = timestamps[start_idx]
                t_end = timestamps[i]
                duration = max(float((t_end - t_start) / np.timedelta64(1, 's')), 1.0)

                # Signed volume
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
# LOAD AND PROCESS
# ============================================================================
print("\n" + "="*80)
print("LOADING AND PROCESSING DATA")
print("="*80)

combined_file = data_dir / 'BTCUSDT_combined.csv'
if not combined_file.exists():
    print(f"ERROR: {combined_file} not found!")
    sys.exit(1)

# Count total rows
print("\nCounting rows...")
sys.stdout.flush()
total_rows = sum(1 for _ in open(combined_file)) - 1  # Subtract header
print(f"  Total rows: {total_rows:,}")
print(f"  Expected batches: {int(np.ceil(total_rows / BATCH_SIZE))}")
sys.stdout.flush()

# Process in batches
all_runs = []
batch_num = 0
total_start = time.time()

print(f"\n{'='*80}")
print("PROCESSING BATCHES")
print(f"{'='*80}\n")
sys.stdout.flush()

for chunk in pd.read_csv(combined_file, chunksize=BATCH_SIZE):
    batch_num += 1
    batch_start = time.time()

    print(f"[BATCH {batch_num}]")
    print(f"  Loading... ", end='')
    sys.stdout.flush()

    # Parse timestamps
    chunk['timestamp'] = pd.to_datetime(chunk['timestamp'])

    # Extract arrays
    prices = chunk['price'].values
    timestamps = chunk['timestamp'].values
    sizes = chunk['size'].values
    sides = chunk['side'].values

    print(f"OK ({len(chunk):,} rows)")
    print(f"  Date range: {chunk['timestamp'].min()} -> {chunk['timestamp'].max()}")
    print(f"  Price range: ${prices.min():.0f} - ${prices.max():.0f}")
    print(f"  Computing runs... ", end='')
    sys.stdout.flush()

    # Compute runs
    batch_runs = compute_runs_from_batch(prices, timestamps, sizes, sides)
    all_runs.extend(batch_runs)

    batch_time = time.time() - batch_start
    elapsed_total = time.time() - total_start

    print(f"OK ({len(batch_runs):,} runs)")
    print(f"  Batch time: {batch_time:.1f}s ({len(chunk)/batch_time:.0f} rows/sec)")
    print(f"  Total runs: {len(all_runs):,}")
    print(f"  Total elapsed: {elapsed_total:.1f}s")
    print(f"  Progress: {100 * batch_num * BATCH_SIZE / total_rows:.1f}%")
    print()
    sys.stdout.flush()

total_time = time.time() - total_start

print(f"{'='*80}")
print("PROCESSING COMPLETE")
print(f"{'='*80}")
print(f"  Total batches: {batch_num}")
print(f"  Total runs: {len(all_runs):,}")
print(f"  Total time: {total_time:.1f}s")
print(f"  Average speed: {total_rows/total_time:.0f} rows/sec")
print()
sys.stdout.flush()

# ============================================================================
# CONVERT TO DATAFRAME AND SAVE
# ============================================================================
print(f"{'='*80}")
print("SAVING RESULTS")
print(f"{'='*80}")

print("\nConverting to DataFrame... ", end='')
sys.stdout.flush()
df_runs = pd.DataFrame(all_runs)
print("OK")
sys.stdout.flush()

print(f"\nRun statistics:")
print(f"  Total runs: {len(df_runs):,}")
print(f"  UP runs: {len(df_runs[df_runs['dir']=='up']):,} ({100*len(df_runs[df_runs['dir']=='up'])/len(df_runs):.1f}%)")
print(f"  DOWN runs: {len(df_runs[df_runs['dir']=='down']):,} ({100*len(df_runs[df_runs['dir']=='down'])/len(df_runs):.1f}%)")
print(f"\nV_eff statistics:")
print(f"  Mean: {df_runs['V_eff'].mean():.4f} BTC/USD")
print(f"  Median: {df_runs['V_eff'].median():.4f} BTC/USD")
print(f"  Min: {df_runs['V_eff'].min():.4f} BTC/USD")
print(f"  Max: {df_runs['V_eff'].max():.4f} BTC/USD")
print(f"\nTime range:")
print(f"  Start: {df_runs['t_start'].min()}")
print(f"  End: {df_runs['t_end'].max()}")
sys.stdout.flush()

print(f"\nSaving to {output_file}... ", end='')
sys.stdout.flush()
save_start = time.time()
df_runs.to_csv(output_file, index=False)
save_time = time.time() - save_start
print(f"OK ({save_time:.1f}s)")
sys.stdout.flush()

file_size_mb = output_file.stat().st_size / (1024**2)
print(f"  File size: {file_size_mb:.1f} MB")
print(f"  Compression: {total_rows/len(df_runs):.1f}x")

print(f"\n{'='*80}")
print("SUCCESS - RUNS SAVED")
print(f"{'='*80}")
print(f"\nOutput: {output_file}")
print(f"Runs: {len(df_runs):,}")
print(f"Total time: {total_time + save_time:.1f}s")
print(f"\nReady for analysis!")
print(f"{'='*80}")
