#!/usr/bin/env python3
"""
DOWNLOAD LATEST TRADES - Binance Futures aggTrades
Downloads missing aggTrades from last date to now in 1-hour batches

Usage:
    python src/download_latest_trades.py

Author: Quant Research Team
"""
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import time
import sys

print("="*80)
print("DOWNLOAD LATEST TRADES - BINANCE FUTURES")
print("="*80)

# ============================================================================
# CONFIGURATION
# ============================================================================
script_dir = Path(__file__).parent
data_dir = script_dir.parent / 'data'

SYMBOL = "BTCUSDT"
BASE_URL = "https://fapi.binance.com/fapi/v1/aggTrades"
LIMIT = 1000  # Max allowed by API

# ============================================================================
# SET START TIME TO TODAY AT MIDNIGHT UTC
# ============================================================================
print("\n" + "="*80)
print("SETTING TIME RANGE")
print("="*80)

# Start from today at midnight UTC
today = datetime.utcnow().date()
start_time = datetime.combine(today, datetime.min.time())
end_time = datetime.utcnow()

print(f"\nTime range to download:")
print(f"  Start: {start_time}")
print(f"  End: {end_time}")
print(f"  Duration: {end_time - start_time}")

if start_time >= end_time:
    print("\nNo new data to download!")
    sys.exit(0)

# ============================================================================
# DOWNLOAD IN BATCHES (1000 trades at a time)
# ============================================================================
print("\n" + "="*80)
print("DOWNLOADING DATA IN 1000-TRADE BATCHES")
print("="*80)

all_trades = []
current_start_ms = int(start_time.timestamp() * 1000)
end_ms = int(end_time.timestamp() * 1000)
batch_num = 0
total_trades = 0

while True:
    batch_num += 1

    params = {
        'symbol': SYMBOL,
        'startTime': current_start_ms,
        'limit': LIMIT
    }

    try:
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()

        if not data:
            print(f"\nNo more data available")
            break

        # Filter trades beyond end_time
        data = [t for t in data if t['T'] <= end_ms]
        if not data:
            print(f"\nReached end time")
            break

        all_trades.extend(data)
        total_trades += len(data)

        last_timestamp = data[-1]['T']
        last_time = datetime.fromtimestamp(last_timestamp / 1000)
        print(f"Batch {batch_num}: Downloaded {len(data)} trades (total: {total_trades:,}) | Last: {last_time}")

        # If we got less than LIMIT, we have all data
        if len(data) < LIMIT:
            print(f"\nAll data downloaded (got {len(data)} < {LIMIT})")
            break

        # Move to next batch (start from 1ms after last trade)
        current_start_ms = last_timestamp + 1

        # Rate limiting to stay under 120 req/min
        time.sleep(0.5)

    except Exception as e:
        print(f"ERROR: {e}")
        time.sleep(5)  # Wait before retry
        continue

print(f"\n[OK] Download complete!")
print(f"Total trades downloaded: {total_trades:,}")

if total_trades == 0:
    print("\nNo new trades found!")
    sys.exit(0)

# ============================================================================
# CONVERT TO HYPERLIQUID FORMAT
# ============================================================================
print("\n" + "="*80)
print("CONVERTING TO HYPERLIQUID FORMAT")
print("="*80)

df = pd.DataFrame(all_trades)
print(f"\nRaw data: {len(df):,} trades")

# Convert Binance format to Hyperliquid format
# Binance: a (id), p (price), q (quantity), T (timestamp), m (is_buyer_maker)
# Hyperliquid: timestamp, side, price, size

df_converted = pd.DataFrame({
    'timestamp': pd.to_datetime(df['T'], unit='ms'),
    'side': df['m'].apply(lambda x: 'S' if x else 'B'),  # m=True -> Sell, m=False -> Buy
    'price': df['p'].astype(float),
    'size': df['q'].astype(float)
})

# Sort by timestamp
df_converted = df_converted.sort_values('timestamp').reset_index(drop=True)

print(f"\nConverted data:")
print(f"  Trades: {len(df_converted):,}")
print(f"  Date range: {df_converted['timestamp'].min()} -> {df_converted['timestamp'].max()}")
print(f"  Sides: {df_converted['side'].value_counts().to_dict()}")

# ============================================================================
# SAVE TO DAILY FILE
# ============================================================================
print("\n" + "="*80)
print("SAVING TO DAILY FILE")
print("="*80)

# Save as single daily file for today
date_str = start_time.strftime('%Y-%m-%d')
output_file = data_dir / f'BTCUSDT-aggTrades-{date_str}.csv'

# Check if file exists and append, otherwise create new
if output_file.exists():
    print(f"\n[APPEND] {output_file.name}")
    # Load existing
    df_existing = pd.read_csv(output_file)
    df_existing['timestamp'] = pd.to_datetime(df_existing['timestamp'])

    # Combine and deduplicate
    df_combined = pd.concat([df_existing, df_converted], ignore_index=True)
    df_combined = df_combined.drop_duplicates(subset=['timestamp', 'price', 'size'], keep='last')
    df_combined = df_combined.sort_values('timestamp').reset_index(drop=True)

    df_combined.to_csv(output_file, index=False)
    print(f"  {len(df_existing):,} existing + {len(df_converted):,} new = {len(df_combined):,} total trades")
else:
    print(f"\n[NEW] {output_file.name}")
    df_converted.to_csv(output_file, index=False)
    print(f"  {len(df_converted):,} trades")

saved_files = [output_file]

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("COMPLETE")
print("="*80)
print(f"\nSaved file: {output_file.name}")
print(f"\nTotal new trades: {total_trades:,}")
print("\nNext steps:")
print("  1. python src/update_combined.py")
print("  2. python src/update_runs.py")
print("  3. python src/snapshot_analysis_heatmap.py")
print("="*80)
