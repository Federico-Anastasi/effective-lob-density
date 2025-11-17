# Effective Liquidity Density in Limit Order Books

Empirical validation of a fundamental relationship between order flow, price dynamics, and liquidity density using high-frequency Bitcoin/USD trade data.

**Paper**: [Effective_limit_order_book_density.pdf](docs/Effective_limit_order_book_density.pdf)

---

## Overview

This repository implements and validates the **effective liquidity density** V_eff(p), a measurable market parameter that quantifies how much volume is required to move price by one dollar.

**Core concept**: During monotonic price movements, the relationship between executed volume Q and price displacement Δp reveals the effective liquidity:

```
V_eff = Q / |Δp|    [units: BTC/USD]
```

This follows from the fundamental microstructure relation:

```
q(t) dt = V(p,t) dp
```

where q(t) is order flow rate and V(p,t) is liquidity density at price p.

---

## Key Results

Using **7.9M trades** over 35 days (528,606 monotonic runs):

✅ **V_eff is stable**: Low temporal autocorrelation (60-min persistence)
✅ **Price-independent**: Correlation with price level r = 0.006
✅ **Volatility-independent**: Correlation with volatility r = 0.018
✅ **Directionally symmetric**: No significant up/down asymmetry (p = 0.46)
✅ **Reveals hidden structure**: V(p) profiles show 10× variation in liquidity across price levels

---

## Installation

```bash
pip install -r requirements.txt
```

**Requirements**: Python 3.8+, pandas, numpy, matplotlib, scipy, scikit-learn, statsmodels

---

## Usage

### 1. Visualize V(p) Profile

```bash
python src/compute_V_profile.py
```

**Output**: `results/V_profile_visualization.png`

Shows:
- **Left panel**: Price evolution over time
- **Right panel**: Liquidity density V(p) as function of price
  - Peaks = high-resistance zones (strong support/resistance)
  - Valleys = low-resistance zones (breakout candidates)

### 2. Statistical Analysis

```bash
python src/analyze_V_properties.py
```

**Output**: `results/V_analysis_N_runs.png`

Validates 6 statistical properties:
1. Distribution (log-normal shape, median 0.218 BTC/USD)
2. Temporal autocorrelation (decays to zero in 60 minutes)
3. Independence from price level
4. Independence from volatility
5. Directional symmetry (up vs down runs)
6. Intraday cycle (peak liquidity at 22:00 UTC)

---

## Methodology

### Step 1: Detect Monotonic Runs

Identify continuous price movements in one direction:
- **Filter**: Minimum 5 ticks, minimum $1 price movement
- **Compute**: Total volume Q, price displacement Δp
- **Calculate**: V_eff = Q / |Δp|

### Step 2: Aggregate to 1-Minute Candles

Duration-weighted average of V_eff within each minute:

```
V_eff(minute) = Σ(V_eff^(run) · duration_run) / Σ(duration_run)
```

### Step 3: Construct V(p) Profile

For each price level p:
- Find all runs that crossed price p
- Apply exponential time decay (2-hour half-life)
- Compute weighted average
- Apply Gaussian smoothing (σ = 2 bins)

---

## Data Format

CSV files with columns:
- `timestamp`: Trade execution time
- `price`: Execution price (USD)
- `size`: Volume (BTC)
- `side`: 'B' (buy) or 'S' (sell)

Sample data included: `data/hyperliquid_trades_BTC_1748589872.csv`

---

## Results Interpretation

**High V_eff zones (peaks)**:
- Large volume required to move price
- Strong support/resistance levels
- Potential reversal or consolidation zones

**Low V_eff zones (valleys)**:
- Small volume moves price significantly
- Weak resistance
- Breakout candidates or fast movement zones

**Asymmetry**:
- If V_above > V_below: Harder to push price up
- If V_below > V_above: Harder to push price down

---

## Applications

1. **Market impact estimation**: Predict slippage for large orders
2. **Optimal execution**: Identify low-resistance zones for trading
3. **Risk management**: Detect liquidity stress in real-time
4. **Microstructure research**: Validate theoretical LOB models

---

## Project Structure

```
effective_lob_density/
├── README.md
├── requirements.txt
├── LICENSE
├── docs/
│   └── Effective_limit_order_book_density.pdf    # Full paper (9 pages)
├── src/
│   ├── compute_V_profile.py      # Main visualization script
│   └── analyze_V_properties.py   # Statistical validation
├── data/
│   └── hyperliquid_trades_BTC_*.csv   # Sample trade data
└── results/
    ├── V_profile_visualization.png    # V(p) profile chart
    └── V_analysis_528606_runs.png     # Statistical analysis
```

---

## Theoretical Background

The model derives from the conservation law:

```
Volume executed = Liquidity consumed
```

Expressed as:

```
∫ q(t) dt = ∫ V(p,t) dp
```

For infinitesimal intervals:

```
q(t) dt = V(p,t) dp

⟹  dp/dt = q(t) / V(p,t)
```

**Interpretation**: Price velocity is proportional to order flow and inversely proportional to liquidity density.

See [paper](docs/Effective_limit_order_book_density.pdf) for complete derivation and empirical validation.

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

## License

MIT License - See [LICENSE](LICENSE) file

---

## Contact

- **GitHub**: [@Federico-Anastasi](https://github.com/Federico-Anastasi)
- **Twitter**: [@FedeAnastasi](https://twitter.com/FedeAnastasi)
- **Email**: federico_anastasi@outlook.com

---

**Dataset**: 7.9M trades | **Period**: May 27 – July 1, 2025 | **Runs**: 528,606 | **Candles**: 31,203
