# Snapshot Analysis: BTC Crash November 2025

Temporal evolution of **directional liquidity density** (V_diff) during the Bitcoin crash from $112K to $79K (Nov 13-21, 2025).

---

## Overview

This example demonstrates how **V_diff(p) = V_down(p) - V_up(p)** evolves over time during a major market crash:

- **Snapshot 1** (Nov 13, 10:00 UTC): $102,912 - Pre-crash
- **Snapshot 2** (Nov 17, 15:00 UTC): $94,803 - Mid-crash (-8%)
- **Snapshot 3** (Nov 20, 23:59 UTC): $86,712 - Late-crash (-16%)
- **Snapshot 4** (Nov 21, 19:47 UTC): $84,256 - Bottom (-18%)

**Total drawdown**: -29% from peak ($112K on Nov 13)

---

## Key Concept: V_diff

**V_diff = V_down - V_up** reveals directional liquidity asymmetry:

- **Positive V_diff (green)**: Support zone - More liquidity from down moves → harder to push price down
- **Negative V_diff (red)**: Resistance zone - More liquidity from up moves → harder to push price up
- **Zero V_diff (white)**: Neutral zone - Symmetric liquidity

---

## Dataset

- **Source**: Binance BTC/USDT Futures (aggTrades)
- **Period**: November 1-21, 2025
- **Trades**: 34,543,267 executed trades
- **Runs**: 1,957,320 monotonic price movements
- **Temporal snapshots**: 4 key moments during crash

---

## Methodology

### 1. Run Detection

Identify continuous price movements (minimum 5 ticks, $1 displacement):

```python
V_eff = Q / |Δp|    # Volume required to move price $1
```

### 2. Directional Profiles

Separate runs by direction:

- **V_up(p)**: Weighted average V_eff from upward runs crossing price p
- **V_down(p)**: Weighted average V_eff from downward runs crossing price p

### 3. Temporal Decay

Apply exponential decay (2-hour half-life):

```python
weights = exp(-λ * time_since_run)
λ = ln(2) / (2 * 3600)  # Half-life = 2 hours
```

### 4. Consistency

- **Fixed price bins**: $150 bins (79K-112K range)
- **Global vmax**: 0.589 for all snapshots
- **Gaussian smoothing**: σ = 2 bins

---

## Usage

### Prerequisites

```bash
pip install -r requirements.txt
```

### Step 1: Download Data

**Option A: Using Binance Vision (Historical Data)**

Download historical aggTrades from Binance public data:

1. Visit: https://data.binance.vision/?prefix=data/spot/daily/aggTrades/BTCUSDT/
2. Download daily CSV files for desired period (e.g., Nov 1-21, 2025)
3. Place files in `data/` directory:
   ```
   data/
   ├── BTCUSDT-aggTrades-2025-11-01.csv
   ├── BTCUSDT-aggTrades-2025-11-02.csv
   └── ...
   ```

**Option B: Using Binance API (Live Data)**

```bash
python src/download_trades.py
```

Downloads Binance aggTrades for today using the Futures API.

### Step 2: Update Combined File

```bash
python src/update_combined.py
```

Appends new daily files to `BTCUSDT_combined.csv`.

### Step 3: Compute Runs

```bash
python src/update_runs.py
```

Calculates monotonic runs and updates `BTCUSDT_runs.csv`.

### Step 4: Generate Snapshots

```bash
python src/snapshot_analysis.py
```

**Output**: 4 heatmaps in `results/snapshot_1.png` through `snapshot_4.png`

---

## Interpretation

### Heatmap Components

**Left Panel**: Price evolution with V_diff heatmap overlay
- **Red zones**: Resistance (negative V_diff)
- **Green zones**: Support (positive V_diff)
- **White zones**: Neutral (V_diff ≈ 0)

**Right Panel**: Current V_diff profile
- **Higher values**: Stronger liquidity barrier
- **Lower values**: Weaker resistance, easier breakout

---

## Technical Details

### Configuration

```python
PRICE_BIN_SIZE = 150          # Fixed $150 bins
LAMBDA_DECAY = ln(2)/(2*3600) # 2-hour half-life
SIGMA_SMOOTH = 2              # Gaussian smoothing parameter
VDIFF_XMAX_FIXED = 0.10       # Profile X-axis limit
```

### Snapshot Timestamps

| Snapshot | Date              | Time (UTC) | Price    | Runs       |
|----------|-------------------|------------|----------|------------|
| 1        | Nov 13, 2025      | 10:00      | $102,912 | 825,502    |
| 2        | Nov 17, 2025      | 15:00      | $94,803  | 1,224,040  |
| 3        | Nov 20, 2025      | 23:59      | $86,712  | 1,618,133  |
| 4        | Nov 21, 2025      | 19:47      | $84,256  | 1,957,319  |

---

## Project Structure

```
snapshot/
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── src/
│   ├── binance_adapter.py     # Binance → Hyperliquid format converter
│   ├── compute_runs.py        # Initial batch run computation
│   ├── download_trades.py     # Binance API downloader
│   ├── update_combined.py     # Append new trades to combined file
│   ├── update_runs.py         # Incremental run updates
│   └── snapshot_analysis.py   # Main: Generate 4 snapshot heatmaps
├── data/                      # (excluded from git - too large)
│   ├── BTCUSDT_combined.csv   # All trades (34.5M rows, ~1.5GB)
│   └── BTCUSDT_runs.csv       # Computed runs (1.95M rows, ~200MB)
└── results/
    ├── snapshot_1.png         # Nov 13: $102K (pre-crash)
    ├── snapshot_2.png         # Nov 17: $94K (mid-crash)
    ├── snapshot_3.png         # Nov 20: $86K (late-crash)
    └── snapshot_4.png         # Nov 21: $84K (bottom)
```

---

## Theoretical Background

See parent repository [README](../../README.md) and [paper](../../docs/Effective_limit_order_book_density.pdf) for:

- V_eff derivation from conservation laws
- Statistical validation (528K runs)
- Independence from price/volatility
- Directional symmetry analysis

---

## Citation

```bibtex
@misc{anastasi2025lob,
  title={Effective Liquidity Density in Limit Order Books: An Empirical Validation from High-Frequency Data},
  author={Federico Anastasi},
  year={2025},
  url={https://github.com/Federico-Anastasi/effective-lob-density}
}
```

---

## Author

**Federico Anastasi**
- GitHub: [@Federico-Anastasi](https://github.com/Federico-Anastasi)
- Twitter: [@FedeAnastasi](https://twitter.com/FedeAnastasi)
- Email: federico_anastasi@outlook.com

---

**Dataset**: 34.5M trades | **Period**: Nov 1-21, 2025 | **Runs**: 1.95M | **Snapshots**: 4
