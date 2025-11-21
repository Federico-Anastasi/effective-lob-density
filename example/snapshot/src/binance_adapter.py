#!/usr/bin/env python3
"""
Binance aggTrades Data Adapter
Converts Binance aggTrades CSV format to Hyperliquid-compatible format for V_eff analysis

Binance aggTrades format:
  aggTrade_id, price, quantity, first_trade_id, last_trade_id, timestamp, is_buyer_maker, is_best_match

Hyperliquid format:
  timestamp, side, price, size

Key mapping:
  - is_buyer_maker=False -> side='B' (market BUY order - taker buying)
  - is_buyer_maker=True  -> side='S' (market SELL order - taker selling)
  - timestamp: microseconds Unix -> datetime ISO format
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

def convert_binance_to_hyperliquid(input_file, output_file=None):
    """
    Convert Binance aggTrades CSV to Hyperliquid format

    Args:
        input_file: Path to Binance CSV file
        output_file: Path to output CSV (optional, auto-generated if None)

    Returns:
        DataFrame in Hyperliquid format
    """
    print(f"\n{'='*70}")
    print(f"BINANCE -> HYPERLIQUID DATA ADAPTER")
    print(f"{'='*70}")
    print(f"Input: {Path(input_file).name}")

    # Load Binance data (no header in aggTrades files)
    df = pd.read_csv(
        input_file,
        header=None,
        names=['aggTrade_id', 'price', 'quantity', 'first_trade_id',
               'last_trade_id', 'timestamp', 'is_buyer_maker', 'is_best_match']
    )

    print(f"Loaded {len(df):,} aggregate trades")

    # Convert timestamp (microseconds -> datetime)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='us')

    # Map is_buyer_maker to side
    # is_buyer_maker=False -> Market BUY (taker buying from maker's sell order) -> 'B'
    # is_buyer_maker=True  -> Market SELL (taker selling to maker's buy order) -> 'S'
    df['side'] = df['is_buyer_maker'].map({False: 'B', True: 'S'})

    # Create Hyperliquid format
    df_hl = pd.DataFrame({
        'timestamp': df['timestamp'],
        'side': df['side'],
        'price': df['price'],
        'size': df['quantity']
    })

    # Sort by timestamp
    df_hl = df_hl.sort_values('timestamp').reset_index(drop=True)

    # Validation stats
    print(f"\n{'='*70}")
    print("VALIDATION STATISTICS")
    print(f"{'='*70}")
    print(f"Total trades: {len(df_hl):,}")
    print(f"Time range: {df_hl['timestamp'].min()} -> {df_hl['timestamp'].max()}")
    print(f"Duration: {(df_hl['timestamp'].max() - df_hl['timestamp'].min()).total_seconds() / 3600:.2f} hours")
    print(f"\nPrice range: ${df_hl['price'].min():.2f} -> ${df_hl['price'].max():.2f}")
    print(f"Price movement: ${df_hl['price'].max() - df_hl['price'].min():.2f}")

    buy_trades = (df_hl['side'] == 'B').sum()
    sell_trades = (df_hl['side'] == 'S').sum()
    print(f"\nSide distribution:")
    print(f"  BUY  (B): {buy_trades:,} ({100*buy_trades/len(df_hl):.1f}%)")
    print(f"  SELL (S): {sell_trades:,} ({100*sell_trades/len(df_hl):.1f}%)")

    buy_volume = df_hl[df_hl['side'] == 'B']['size'].sum()
    sell_volume = df_hl[df_hl['side'] == 'S']['size'].sum()
    print(f"\nVolume distribution:")
    print(f"  BUY  volume: {buy_volume:.4f} BTC ({100*buy_volume/(buy_volume+sell_volume):.1f}%)")
    print(f"  SELL volume: {sell_volume:.4f} BTC ({100*sell_volume/(buy_volume+sell_volume):.1f}%)")

    # Check for time gaps
    time_diffs = df_hl['timestamp'].diff().dt.total_seconds()
    gaps = time_diffs[time_diffs > 300]  # 5-minute gaps
    if len(gaps) > 0:
        print(f"\n[!] Found {len(gaps)} time gaps > 5 minutes (max: {gaps.max():.0f}s)")
    else:
        print(f"\n[OK] No significant time gaps detected")

    # Save to file
    if output_file is None:
        input_path = Path(input_file)
        output_file = input_path.parent / f"{input_path.stem}.csv"

    df_hl.to_csv(output_file, index=False)
    print(f"\n[OK] Saved to: {output_file}")
    print(f"{'='*70}\n")

    return df_hl


def convert_all_files(data_dir):
    """Convert all Binance aggTrades files in directory"""
    data_path = Path(data_dir)
    files = sorted(data_path.glob('BTCUSDT-aggTrades-*.csv'))

    if not files:
        print(f"No files found in {data_dir}")
        return

    print(f"\nFound {len(files)} Binance aggTrades files")

    converted_dfs = []
    for file in files:
        df = convert_binance_to_hyperliquid(file)
        converted_dfs.append(df)

    # Combine all files
    if len(converted_dfs) > 1:
        print(f"\n{'='*70}")
        print("COMBINING FILES")
        print(f"{'='*70}")

        df_combined = pd.concat(converted_dfs, ignore_index=True)
        df_combined = df_combined.sort_values('timestamp').reset_index(drop=True)

        output_file = data_path / "BTCUSDT_combined.csv"
        df_combined.to_csv(output_file, index=False)

        print(f"Combined {len(files)} files -> {len(df_combined):,} total trades")
        print(f"Saved to: {output_file}")
        print(f"{'='*70}\n")

        return df_combined

    return converted_dfs[0] if converted_dfs else None


if __name__ == '__main__':
    # Convert all files in data directory
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / 'data'

    df = convert_all_files(data_dir)

    if df is not None:
        print(f"\n[OK] Conversion complete!")
        print(f"Total dataset: {len(df):,} trades")
        print(f"Ready for V_eff analysis")
