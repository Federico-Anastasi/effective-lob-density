#!/usr/bin/env python3
"""
UPDATE COMBINED - Add new daily files
Simply appends new aggTrades files to existing BTCUSDT_combined.csv

Usage:
    python src/update_combined.py

Author: Quant Research Team
"""
import sys
sys.path.insert(0, str(__file__).rsplit('/', 1)[0])

import pandas as pd
from pathlib import Path
from binance_adapter import convert_binance_to_hyperliquid

print("="*80)
print("UPDATE COMBINED - ADD NEW DAILY FILES")
print("="*80)

script_dir = Path(__file__).parent
data_dir = script_dir.parent / 'data'
combined_file = data_dir / 'BTCUSDT_combined.csv'

# Load existing data (only to get last date)
print("\nLoading existing combined data...")
df_existing = pd.read_csv(combined_file, usecols=['timestamp'])
df_existing['timestamp'] = pd.to_datetime(df_existing['timestamp'])
last_date = df_existing['timestamp'].max().date()
print(f"Last date in existing data: {last_date}")

# Find new files (after last date)
all_files = sorted(data_dir.glob('BTCUSDT-aggTrades-*.csv'))
new_files = []
for file in all_files:
    parts = file.stem.split('-')
    if len(parts) == 5:
        try:
            file_date = pd.Timestamp(f"{parts[2]}-{parts[3]}-{parts[4]}").date()
            if file_date > last_date:
                new_files.append(file)
        except:
            continue

if not new_files:
    print("No new files found!")
    sys.exit(0)

print(f"\nNew files to add ({len(new_files)}):")
for f in new_files:
    print(f"  - {f.name}")

# Load new files (check if already converted)
print("\nLoading new files...")
new_dfs = []
for file in new_files:
    # Check if file is already in Hyperliquid format
    df_sample = pd.read_csv(file, nrows=1)
    if 'timestamp' in df_sample.columns and 'side' in df_sample.columns:
        # Already converted - load directly
        print(f"  Loading (already converted): {file.name}")
        df = pd.read_csv(file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    else:
        # Need conversion
        print(f"  Converting: {file.name}")
        df = convert_binance_to_hyperliquid(file)
    new_dfs.append(df)

df_new = pd.concat(new_dfs, ignore_index=True)
df_new = df_new.sort_values('timestamp').reset_index(drop=True)

print(f"\nNew data: {len(df_new):,} trades")
print(f"Date range: {df_new['timestamp'].min()} -> {df_new['timestamp'].max()}")

# Append to combined file
print("\nAppending to combined file...")
df_new.to_csv(combined_file, mode='a', header=False, index=False)

print(f"\n[OK] Update complete!")
print("Next step: python src/update_runs.py")
print("="*80)
