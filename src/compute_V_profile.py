#!/usr/bin/env python3
"""
Visualizza profilo V(p) affiancato al grafico del prezzo
"""
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

print("="*70)
print("V_EFF PROFILE VISUALIZATION")
print("="*70)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n[1/4] Loading data...")
script_dir = Path(__file__).parent
data_dir = script_dir.parent / 'data'
files = sorted(data_dir.glob('*.csv'))

file_path = files[0]
print(f"Using file: {file_path.name}")

df = pd.read_csv(file_path)
df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
df = df.sort_values('timestamp').reset_index(drop=True)

# ============================================================================
# COMPUTE V_EFF
# ============================================================================
print("[2/4] Computing V_eff...")
prices = df['price'].values
timestamps = df['timestamp'].values
dirs = np.sign(np.diff(prices))

# Detect time gaps
time_diffs = np.diff(timestamps).astype('timedelta64[s]').astype(float)
gaps = np.where(time_diffs > 300)[0]

runs = []
start_idx, cur_dir = 0, None

for i in range(len(dirs)):
    # Gap detection
    if i in gaps:
        start_idx, cur_dir = i + 1, None
        continue

    if dirs[i] == 0:
        continue

    if cur_dir is None:
        cur_dir, start_idx = dirs[i], i
        continue

    # Direction change
    if dirs[i] != cur_dir:
        p_start, p_end = prices[start_idx], prices[i]
        delta_p = p_end - p_start

        # Filter: min 5 ticks, min $1 movement
        if (i - start_idx) >= 5 and abs(delta_p) >= 1.0:
            Q = df.iloc[start_idx:i+1]['size'].sum()
            t_start = df.iloc[start_idx]['timestamp']
            t_end = df.iloc[i]['timestamp']
            duration = max((t_end - t_start).total_seconds(), 1)

            # Signed volume
            q_run = df.iloc[start_idx:i+1].apply(
                lambda row: row['size'] if row['side'] == 'B' else -row['size'], axis=1
            ).sum()

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
                'delta_p': delta_p
            })

        start_idx, cur_dir = i, dirs[i]

print(f"Computed {len(runs)} monotonic runs")

# Convert to DataFrame
df_runs = pd.DataFrame(runs)

# ============================================================================
# AGGREGATE TO 1-MIN CANDLES
# ============================================================================
print("[3/4] Aggregating to 1-min candles...")

minute_data = []
for run in runs:
    minute_data.append({
        'minute': run['t_start'].floor('1min'),
        'V_eff': run['V_eff'],
        'weight': run['duration'],
        'price': run['price_mid'],
        'q': run['q'],
        'dir': run['dir']
    })

df_minutes = pd.DataFrame(minute_data)
df_1min = df_minutes.groupby('minute').agg({
    'V_eff': lambda x: np.average(x, weights=df_minutes.loc[x.index, 'weight']),
    'price': 'mean',
    'q': 'sum',
    'dir': lambda x: 'up' if (x == 'up').sum() > (x == 'down').sum() else 'down'
}).reset_index().set_index('minute')

print(f"Aggregated to {len(df_1min)} 1-min candles")

# Use ALL data
df_window = df_1min.copy()
print(f"Using full dataset: {len(df_window)} candles from {df_window.index[0]} to {df_window.index[-1]}")

# Get all runs in this time window
t_start_window = df_window.index[0]
t_end_window = df_window.index[-1]
df_runs_window = df_runs[
    (df_runs['t_start'] >= t_start_window) &
    (df_runs['t_end'] <= t_end_window)
].copy()
print(f"Runs in window: {len(df_runs_window)}")

# ============================================================================
# COMPUTE V_PROFILE at final time t_0
# ============================================================================
print("[4/4] Computing V(p) profile...")

# t_0 = last timestamp in window
t_0 = df_window.index[-1]
p_0 = df_window.iloc[-1]['price']

# Price range for profile
p_min = df_window['price'].min()
p_max = df_window['price'].max()
n_bins = 100
price_bins = np.linspace(p_min, p_max, n_bins)
bin_width = (p_max - p_min) / n_bins

# Decay parameter (half-life = 2 hours)
lambda_decay = np.log(2) / (2 * 3600)  # in seconds^-1

V_profile = []
V_profile_up = []
V_profile_down = []

print(f"Computing V(p) for {n_bins} price bins...")

for p in price_bins:
    # Find runs that crossed this price level
    mask = (
        ((df_runs_window['price_start'] <= p) & (df_runs_window['price_end'] >= p)) |
        ((df_runs_window['price_start'] >= p) & (df_runs_window['price_end'] <= p))
    )

    runs_at_p = df_runs_window[mask]

    if len(runs_at_p) > 0:
        # Exponential time decay weights
        time_diffs = (t_0 - runs_at_p['t_end']).dt.total_seconds()
        weights = np.exp(-lambda_decay * time_diffs)

        # Weighted average V_eff
        V_p = np.average(runs_at_p['V_eff'], weights=weights)
        V_profile.append(V_p)

        # Separate by direction
        runs_up = runs_at_p[runs_at_p['dir'] == 'up']
        runs_down = runs_at_p[runs_at_p['dir'] == 'down']

        if len(runs_up) > 0:
            weights_up = np.exp(-lambda_decay * (t_0 - runs_up['t_end']).dt.total_seconds())
            V_profile_up.append(np.average(runs_up['V_eff'], weights=weights_up))
        else:
            V_profile_up.append(np.nan)

        if len(runs_down) > 0:
            weights_down = np.exp(-lambda_decay * (t_0 - runs_down['t_end']).dt.total_seconds())
            V_profile_down.append(np.average(runs_down['V_eff'], weights=weights_down))
        else:
            V_profile_down.append(np.nan)
    else:
        V_profile.append(np.nan)
        V_profile_up.append(np.nan)
        V_profile_down.append(np.nan)

V_profile = np.array(V_profile)
V_profile_up = np.array(V_profile_up)
V_profile_down = np.array(V_profile_down)

# Smooth profile (optional - moving average)
from scipy.ndimage import gaussian_filter1d
V_profile_smooth = gaussian_filter1d(
    np.where(np.isnan(V_profile), np.nanmean(V_profile), V_profile),
    sigma=2
)

print(f"V(p) profile computed:")
print(f"  Mean: {np.nanmean(V_profile):.2f} BTC/USD")
print(f"  Min:  {np.nanmin(V_profile):.2f} BTC/USD")
print(f"  Max:  {np.nanmax(V_profile):.2f} BTC/USD")

# ============================================================================
# CREATE VISUALIZATION
# ============================================================================
print("\nCreating visualization...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True,
                               gridspec_kw={'width_ratios': [3, 1], 'wspace': 0.02})

# ============================================================================
# LEFT PANEL: Price vs Time
# ============================================================================
ax1.plot(df_window.index, df_window['price'], color='black', linewidth=1.5, label='Price')
ax1.axvline(t_0, color='orange', linestyle='--', linewidth=2, label='t_0')
ax1.set_xlabel('Time', fontsize=12)
ax1.set_ylabel('Price [USD]', fontsize=12)
ax1.set_title('Price vs Time', fontsize=14, pad=10)
ax1.legend(fontsize=10, loc='upper left')
ax1.grid(alpha=0.3)
ax1.tick_params(labelsize=10)

# ============================================================================
# RIGHT PANEL: V(p) Profile
# ============================================================================
# Total V(p) - smoothed with fill
ax2.fill_betweenx(price_bins, 0, V_profile_smooth, color='steelblue', alpha=0.3)
ax2.plot(V_profile_smooth, price_bins, color='steelblue', linewidth=2, label='V(p) total')

# V(p) up and down directions
ax2.plot(V_profile_up, price_bins, color='green', linewidth=1, linestyle=':', label='V(p) up')
ax2.plot(V_profile_down, price_bins, color='red', linewidth=1, linestyle=':', label='V(p) down')

# Mark current price p_0
V_max = np.nanmax(V_profile_smooth)
ax2.axhline(p_0, color='orange', linestyle='--', linewidth=2, label=f'p(t_0)={p_0:.0f}')

ax2.set_xlabel('V_eff [BTC/USD]', fontsize=12)
ax2.set_title('V_eff Profile', fontsize=14, pad=10)
ax2.legend(fontsize=9, loc='upper right')
ax2.grid(alpha=0.3)
ax2.tick_params(labelsize=10)

# Overall title
window_hours = len(df_window) / 60
fig.suptitle(f'V_eff PROFILE - Liquidity Density vs Price\nDecay half-life: 2h | Window: {window_hours:.1f}h | t_0: {t_0}',
             fontsize=14, y=1.02)

plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save
results_dir = script_dir.parent / 'results'
results_dir.mkdir(exist_ok=True)
output_file = results_dir / 'V_profile_visualization.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"Saved: {output_file}")

# ============================================================================
# ANALYSIS
# ============================================================================
print("\n" + "="*70)
print("PROFILE ANALYSIS")
print("="*70)

# Find peaks and valleys in V(p)
from scipy.signal import find_peaks

peaks, _ = find_peaks(V_profile_smooth, prominence=np.nanstd(V_profile_smooth)*0.5)
valleys, _ = find_peaks(-V_profile_smooth, prominence=np.nanstd(V_profile_smooth)*0.5)

print(f"\nLiquidity PEAKS (high resistance zones):")
for i in peaks[:5]:  # Top 5
    print(f"  Price: ${price_bins[i]:.0f}, V_eff: {V_profile_smooth[i]:.2f} BTC/USD")

print(f"\nLiquidity VALLEYS (low resistance zones):")
for i in valleys[:5]:  # Top 5
    print(f"  Price: ${price_bins[i]:.0f}, V_eff: {V_profile_smooth[i]:.2f} BTC/USD")

# Check where current price is
current_bin = np.argmin(np.abs(price_bins - p_0))
current_V = V_profile_smooth[current_bin]
mean_V = np.nanmean(V_profile_smooth)

print(f"\nCurrent position (p_0 = ${p_0:.0f}):")
print(f"  V_eff at p_0: {current_V:.2f} BTC/USD")
print(f"  Mean V_eff:   {mean_V:.2f} BTC/USD")

if current_V > mean_V * 1.5:
    print("  -> HIGH RESISTANCE zone (price at liquidity peak)")
    print("  -> Potential reversal or consolidation")
elif current_V < mean_V * 0.7:
    print("  -> LOW RESISTANCE zone (price at liquidity valley)")
    print("  -> Potential breakout or fast movement")
else:
    print("  -> NORMAL liquidity zone")

# Asymmetry in V profile above/below current price
V_above = V_profile_smooth[price_bins > p_0]
V_below = V_profile_smooth[price_bins < p_0]

if len(V_above) > 0 and len(V_below) > 0:
    mean_V_above = np.nanmean(V_above)
    mean_V_below = np.nanmean(V_below)

    print(f"\nLiquidity asymmetry:")
    print(f"  V_eff above p_0: {mean_V_above:.2f} BTC/USD")
    print(f"  V_eff below p_0: {mean_V_below:.2f} BTC/USD")

    if mean_V_above > mean_V_below * 1.3:
        print("  -> STRONGER resistance ABOVE (harder to push price up)")
    elif mean_V_below > mean_V_above * 1.3:
        print("  -> STRONGER resistance BELOW (harder to push price down)")
    else:
        print("  -> Balanced liquidity")

print("\n" + "="*70)
print("DONE - Open results/V_profile_visualization.png")
print("="*70)
