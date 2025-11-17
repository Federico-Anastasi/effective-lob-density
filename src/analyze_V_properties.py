#!/usr/bin/env python3
"""
Complete V_eff analysis: distribution, autocorrelation, correlations, asymmetry, daily cycle
"""
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import stats

script_dir = Path(__file__).parent
data_dir = script_dir.parent / 'data'

print("="*70)
print("V_EFF PROPERTIES ANALYSIS")
print("="*70)

# Load data
print("\n[1/5] Loading data...")
files = sorted(data_dir.glob('*.csv'))
if len(files) == 0:
    raise FileNotFoundError(f"No CSV in {data_dir}/")

df = pd.read_csv(files[0])
df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
df = df.sort_values('timestamp').reset_index(drop=True)
print(f"Loaded {len(df):,} ticks from {files[0].name}")

# Compute V_eff on monotonic runs
print("\n[2/5] Computing V_eff...")
prices = df['price'].values
timestamps = df['timestamp'].values
dirs = np.sign(np.diff(prices))

# Detect time gaps
time_diffs = np.diff(timestamps).astype('timedelta64[s]').astype(float)
gaps = np.where(time_diffs > 300)[0]

runs = []
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
        if (i - start_idx) >= 5 and abs(delta_p) >= 1.0:
            Q = df.iloc[start_idx:i+1]['size'].sum()
            t_start = df.iloc[start_idx]['timestamp']
            t_end = df.iloc[i]['timestamp']
            duration = max((t_end - t_start).total_seconds(), 1)

            runs.append({
                'V_eff': Q / abs(delta_p),
                't_start': t_start,
                'duration': duration,
                'price': (p_start + p_end) / 2,
                'delta_p': delta_p,
                'dir': 'up' if cur_dir > 0 else 'down'
            })
        start_idx, cur_dir = i, dirs[i]

print(f"Computed {len(runs)} monotonic runs")

# Aggregate to 1-min candles
print("\n[3/5] Aggregating to 1-min candles...")
minute_data = []
for run in runs:
    minute = run['t_start'].floor('1min')
    minute_data.append({
        'minute': minute,
        'V_eff': run['V_eff'],
        'weight': run['duration'],
        'price': run['price'],
        'delta_p': run['delta_p'],
        'dir': run['dir']
    })

df_minutes = pd.DataFrame(minute_data)
df_1min = df_minutes.groupby('minute').agg({
    'V_eff': lambda x: np.average(x, weights=df_minutes.loc[x.index, 'weight']),
    'price': 'mean',
    'delta_p': lambda x: x.abs().sum(),
    'dir': lambda x: 'up' if (x == 'up').sum() > (x == 'down').sum() else 'down'
}).reset_index()

df_1min['hour'] = df_1min['minute'].dt.hour
df_1min['log_V'] = np.log10(df_1min['V_eff'])
df_1min['volatility'] = df_1min['delta_p'].rolling(window=30, min_periods=10).std()

print(f"Aggregated to {len(df_1min)} 1-minute candles")

# Create analysis plots
print("\n[4/5] Creating analysis plots...")

fig = plt.figure(figsize=(16, 12))

# TEST 1: Distribution
ax1 = plt.subplot(3, 2, 1)
V_vals = df_1min['V_eff'].values
ax1.hist(np.log10(V_vals), bins=50, edgecolor='black', alpha=0.7)
ax1.axvline(np.log10(np.median(V_vals)), color='red', linestyle='--', linewidth=2,
            label=f'Median={np.median(V_vals):.4f}')
ax1.set_xlabel('log10(V_eff) [log10(BTC/USD)]')
ax1.set_ylabel('Frequency')
ax1.set_title('TEST 1: Distribution V_eff (1min)')
ax1.legend()
ax1.grid(True, alpha=0.3)

skewness = stats.skew(V_vals)
print(f"\nTEST 1 - Distribution:")
print(f"  Mean:   {np.mean(V_vals):.6f}")
print(f"  Median: {np.median(V_vals):.6f}")
print(f"  CV:     {np.std(V_vals)/np.mean(V_vals):.2f}")
print(f"  Skewness: {skewness:.2f}")

# TEST 2: Autocorrelation
ax2 = plt.subplot(3, 2, 2)
max_lag = min(60, len(df_1min) // 10)
autocorr = [df_1min['log_V'].autocorr(lag=i) for i in range(1, max_lag)]
ax2.plot(range(1, max_lag), autocorr, 'o-', markersize=3)
ax2.axhline(0, color='red', linestyle='--', linewidth=1)
ax2.axhline(0.05, color='gray', linestyle=':', linewidth=1)
ax2.axhline(-0.05, color='gray', linestyle=':', linewidth=1)
ax2.set_xlabel('Lag [minutes]')
ax2.set_ylabel('Autocorrelation')
ax2.set_title('TEST 2: Autocorrelation V_eff')
ax2.grid(True, alpha=0.3)

persistence = next((i for i, ac in enumerate(autocorr, 1) if abs(ac) < 0.05), max_lag)
print(f"\nTEST 2 - Autocorrelation:")
print(f"  AC(1min): {autocorr[0]:.3f}")
print(f"  Persistence: {persistence} min")

# TEST 3: V vs Price
ax3 = plt.subplot(3, 2, 3)
ax3.scatter(df_1min['price'], df_1min['V_eff'], alpha=0.5, s=20)
ax3.set_xlabel('Price [USD]')
ax3.set_ylabel('V_eff [BTC/USD]')
ax3.set_yscale('log')
ax3.set_title('TEST 3: V_eff vs Price')
ax3.grid(True, alpha=0.3)

corr_price = df_1min[['price', 'V_eff']].corr().iloc[0, 1]
print(f"\nTEST 3 - V vs Price:")
print(f"  Correlation: {corr_price:.3f}")

# TEST 4: V vs Volatility
ax4 = plt.subplot(3, 2, 4)
df_vol = df_1min.dropna(subset=['volatility'])
ax4.scatter(df_vol['volatility'], df_vol['V_eff'], alpha=0.5, s=20)
ax4.set_xlabel('Volatility (30min rolling std delta_p)')
ax4.set_ylabel('V_eff [BTC/USD]')
ax4.set_yscale('log')
ax4.set_title('TEST 4: V_eff vs Volatility')
ax4.grid(True, alpha=0.3)

if len(df_vol) > 10:
    corr_vol = df_vol[['volatility', 'V_eff']].corr().iloc[0, 1]
    print(f"\nTEST 4 - V vs Volatility:")
    print(f"  Correlation: {corr_vol:.3f}")

# TEST 5: Asymmetry Up/Down
ax5 = plt.subplot(3, 2, 5)
V_up = df_1min[df_1min['dir'] == 'up']['V_eff'].values
V_down = df_1min[df_1min['dir'] == 'down']['V_eff'].values

ax5.boxplot([np.log10(V_up), np.log10(V_down)], tick_labels=['UP', 'DOWN'])
ax5.set_ylabel('log10(V_eff)')
ax5.set_title('TEST 5: Directional Asymmetry')
ax5.grid(True, alpha=0.3, axis='y')

t_stat, p_value = stats.ttest_ind(V_up, V_down)
print(f"\nTEST 5 - Asymmetry:")
print(f"  V_up:   {np.median(V_up):.6f}")
print(f"  V_down: {np.median(V_down):.6f}")
print(f"  p-value: {p_value:.4f}")

# TEST 6: Daily Cycle
ax6 = plt.subplot(3, 2, 6)
hourly = df_1min.groupby('hour')['V_eff'].agg(['median', 'count'])
ax6.plot(hourly.index, hourly['median'], 'o-', linewidth=2)
ax6.set_xlabel('Hour (UTC)')
ax6.set_ylabel('V_eff median [BTC/USD]')
ax6.set_yscale('log')
ax6.set_title('TEST 6: Daily Cycle')
ax6.grid(True, alpha=0.3)

peak_hour = hourly['median'].idxmax()
trough_hour = hourly['median'].idxmin()
print(f"\nTEST 6 - Daily Cycle:")
print(f"  Peak:   {peak_hour}:00 UTC (V={hourly.loc[peak_hour, 'median']:.6f})")
print(f"  Trough: {trough_hour}:00 UTC (V={hourly.loc[trough_hour, 'median']:.6f})")

plt.tight_layout()

# Save
results_dir = script_dir.parent / 'results'
results_dir.mkdir(exist_ok=True)
output_file = results_dir / f'V_analysis_{len(runs)}_runs.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')

print(f"\n[5/5] Saved: {output_file}")
print("="*70)
print(f"ANALYSIS COMPLETE - {len(runs)} monotonic runs")
print("="*70)
