#!/usr/bin/env python3
"""
V_EFF CRASH ANALYSIS - HEATMAP VERSION
Shows price with V_diff heatmap overlay for support/resistance visualization

Author: Quant Research Team
"""
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

print("="*80)
print("V_EFF CRASH ANALYSIS - HEATMAP VERSION")
print("="*80)

# ============================================================================
# CONFIGURATION
# ============================================================================
script_dir = Path(__file__).parent
data_dir = script_dir.parent / 'data'
results_dir = script_dir.parent / 'results'
results_dir.mkdir(exist_ok=True)

# Snapshot configurations - ADD MORE SNAPSHOTS HERE
SNAPSHOTS = [
    {
        "name": "Snapshot 1",
        "zone_title": "SNAPSHOT 1",
        "heatmap_title": "GENERATING SNAPSHOT 1 HEATMAP",
        "timestamp": pd.Timestamp('2025-11-13 10:00:00'),
        "date_start": pd.Timestamp('2025-11-01 00:00:00'),
        "p_min": 79000,
        "p_max": 112000,
        "suptitle": 'Effective Limit Order Book Density Analysis | BTC/USDT Binance | Nov 2025\nSnapshot: Nov 13, 10:00 UTC | Price: $102,912',
        "output_file": 'snapshot_heatmap_1.png'
    },
    {
        "name": "Snapshot 2",
        "zone_title": "SNAPSHOT 2",
        "heatmap_title": "GENERATING SNAPSHOT 2 HEATMAP",
        "timestamp": pd.Timestamp('2025-11-17 15:00:00'),
        "date_start": pd.Timestamp('2025-11-01 00:00:00'),
        "p_min": 79000,
        "p_max": 112000,
        "suptitle": 'Effective Limit Order Book Density Analysis | BTC/USDT Binance | Nov 2025\nSnapshot: Nov 17, 15:00 UTC | Price: $94,803',
        "output_file": 'snapshot_heatmap_2.png'
    },
    {
        "name": "Snapshot 3",
        "zone_title": "SNAPSHOT 3",
        "heatmap_title": "GENERATING SNAPSHOT 3 HEATMAP",
        "timestamp": pd.Timestamp('2025-11-20 23:59:00'),
        "date_start": pd.Timestamp('2025-11-01 00:00:00'),
        "p_min": 79000,
        "p_max": 112000,
        "suptitle": 'Effective Limit Order Book Density Analysis | BTC/USDT Binance | Nov 2025\nSnapshot: Nov 20, 23:59 UTC | Price: $86,712',
        "output_file": 'snapshot_heatmap_3.png'
    },
    {
        "name": "Snapshot 4",
        "zone_title": "SNAPSHOT 4",
        "heatmap_title": "GENERATING SNAPSHOT 4 HEATMAP",
        "timestamp": pd.Timestamp('2025-11-21 19:47:44'),
        "date_start": pd.Timestamp('2025-11-01 00:00:00'),
        "p_min": 79000,  # SAME as Snapshot 3 for comparison
        "p_max": 112000,
        "suptitle": 'Effective Limit Order Book Density Analysis | BTC/USDT Binance | Nov 2025\nSnapshot: Nov 21, 19:47 UTC | Price: $84,256',
        "output_file": 'snapshot_heatmap_4.png'
    },
]

PRICE_BIN_SIZE = 150  # Fixed $150 bins for ALL snapshots (ensures alignment)
LAMBDA_DECAY = np.log(2) / (2 * 3600)
SIGMA_SMOOTH = 2

print(f"\nConfigured {len(SNAPSHOTS)} snapshots:")
for i, snap in enumerate(SNAPSHOTS, 1):
    print(f"  {i}. {snap['name']}: {snap['timestamp']}")

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n" + "="*80)
print("LOADING DATA")
print("="*80)

runs_file = data_dir / 'BTCUSDT_runs.csv'
print(f"\nLoading runs...")
df_runs = pd.read_csv(runs_file)
df_runs['t_start'] = pd.to_datetime(df_runs['t_start'])
df_runs['t_end'] = pd.to_datetime(df_runs['t_end'])
print(f"  {len(df_runs):,} runs")

trades_file = data_dir / 'BTCUSDT_combined.csv'
print(f"\nLoading price data...")
df_trades = pd.read_csv(trades_file, usecols=['timestamp', 'price'])
df_trades['timestamp'] = pd.to_datetime(df_trades['timestamp'])
print(f"  {len(df_trades):,} trades")

print("\nResampling to 5min...")
df_5min = df_trades.set_index('timestamp').resample('5min')['price'].mean().dropna()
print(f"  {len(df_5min)} points")

# ============================================================================
# COMPUTE V_PROFILES
# ============================================================================

def compute_V_profiles(df_runs, t_0, p_min, p_max, bin_size=100):
    """Compute V_up and V_down separately with FIXED bin size"""
    # Create bins with FIXED $100 steps (ensures perfect alignment across snapshots)
    price_bins = np.arange(p_min, p_max + bin_size, bin_size)

    runs_up = df_runs[df_runs['dir'] == 'up']
    runs_down = df_runs[df_runs['dir'] == 'down']

    V_up_profile = []
    V_down_profile = []

    for p in price_bins:
        # V_up
        mask_up = (
            ((runs_up['price_start'] <= p) & (runs_up['price_end'] >= p)) |
            ((runs_up['price_start'] >= p) & (runs_up['price_end'] <= p))
        )
        runs_at_p = runs_up[mask_up]

        if len(runs_at_p) > 0:
            time_diffs = (t_0 - runs_at_p['t_end']).dt.total_seconds()
            weights = np.exp(-LAMBDA_DECAY * time_diffs)
            V_up_profile.append(np.average(runs_at_p['V_eff'], weights=weights))
        else:
            V_up_profile.append(np.nan)

        # V_down
        mask_down = (
            ((runs_down['price_start'] <= p) & (runs_down['price_end'] >= p)) |
            ((runs_down['price_start'] >= p) & (runs_down['price_end'] <= p))
        )
        runs_at_p = runs_down[mask_down]

        if len(runs_at_p) > 0:
            time_diffs = (t_0 - runs_at_p['t_end']).dt.total_seconds()
            weights = np.exp(-LAMBDA_DECAY * time_diffs)
            V_down_profile.append(np.average(runs_at_p['V_eff'], weights=weights))
        else:
            V_down_profile.append(np.nan)

    V_up = np.array(V_up_profile)
    V_down = np.array(V_down_profile)

    # Replace NaN with 0 (NOT with mean) before smoothing
    # This ensures that bins without price data have V_diff = 0
    V_up_smooth = gaussian_filter1d(
        np.where(np.isnan(V_up), 0, V_up),
        sigma=SIGMA_SMOOTH
    )

    V_down_smooth = gaussian_filter1d(
        np.where(np.isnan(V_down), 0, V_down),
        sigma=SIGMA_SMOOTH
    )

    return price_bins, V_up_smooth, V_down_smooth

# ============================================================================
# COMPUTE V_PROFILES FOR ALL SNAPSHOTS
# ============================================================================
snapshot_data = []

for idx, snapshot in enumerate(SNAPSHOTS):
    print("\n" + "="*80)
    print(snapshot['zone_title'])
    print("="*80)

    runs_before = df_runs[df_runs['t_end'] < snapshot['timestamp']]
    print(f"Using {len(runs_before):,} runs")

    price_bins, V_up, V_down = compute_V_profiles(
        runs_before, snapshot['timestamp'],
        snapshot['p_min'], snapshot['p_max'], PRICE_BIN_SIZE
    )

    V_diff = V_down - V_up

    df_zone = df_5min[snapshot['date_start']:snapshot['timestamp']]
    price_now = df_zone.iloc[-1]
    print(f"Current price: ${price_now:.0f}")

    snapshot_data.append({
        'config': snapshot,
        'price_bins': price_bins,
        'V_up': V_up,
        'V_down': V_down,
        'V_diff': V_diff,
        'df_zone': df_zone,
        'price_now': price_now
    })

# ============================================================================
# CALCULATE GLOBAL vmax FOR ALL SNAPSHOTS (CONSISTENT COLOR SCALE)
# ============================================================================
print("\n" + "="*80)
print("CALCULATING GLOBAL vmax FOR CONSISTENT COLOR SCALE")
print("="*80)

vmax_global = 0.0
for data in snapshot_data:
    V_diff = data['V_diff']
    price_bins = data['price_bins']
    price_now = data['price_now']

    mask_above = price_bins >= price_now
    mask_below = price_bins < price_now

    # Check resistance zone (above current price, negative V_diff)
    if mask_above.any():
        V_diff_above = V_diff.copy()
        V_diff_above[mask_below] = np.nan
        vmax_above = np.nanmax(-V_diff_above[mask_above])
        vmax_global = max(vmax_global, vmax_above)

    # Check support zone (below current price, positive V_diff)
    if mask_below.any():
        V_diff_below = V_diff.copy()
        V_diff_below[mask_above] = np.nan
        vmax_below = np.nanmax(V_diff_below[mask_below])
        vmax_global = max(vmax_global, vmax_below)

print(f"Global vmax: {vmax_global:.6f}")
print(f"This value will be used for ALL snapshot heatmaps")

# ============================================================================
# CREATE HEATMAP COLORMAPS
# ============================================================================
# Two separate colormaps (white = weak, color = strong):
# - Above current price: White (support/neutral) -> Orange -> Red -> DarkRed (max resistance)
# - Below current price: White (resistance/neutral) -> LightGreen -> Green -> DarkGreen (max support)
colors_above = ['white', 'orange', 'red', 'darkred']  # For above current price
colors_below = ['white', 'lightgreen', 'green', 'darkgreen']  # For below current price
n_bins_cmap = 256
cmap_above = LinearSegmentedColormap.from_list('above_gradient', colors_above, N=n_bins_cmap)
cmap_below = LinearSegmentedColormap.from_list('below_gradient', colors_below, N=n_bins_cmap)

# ============================================================================
# FIXED SCALE FOR X-AXIS (PROFILE PLOT)
# ============================================================================
VDIFF_XMAX_FIXED = 0.10  # Fixed maximum for all snapshots
print(f"\nFixed X-axis scale for profile: 0 to {VDIFF_XMAX_FIXED}")

# ============================================================================
# GENERATE HEATMAPS FOR ALL SNAPSHOTS
# ============================================================================
outputs = []

for idx, data in enumerate(snapshot_data):
    snapshot = data['config']
    price_bins = data['price_bins']
    V_diff = data['V_diff']
    df_zone = data['df_zone']
    price_now = data['price_now']

    print("\n" + "="*80)
    print(snapshot['heatmap_title'])
    print("="*80)

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1], wspace=0.02)

    # LEFT: Price with V_diff heatmap overlay
    ax_left = fig.add_subplot(gs[0])

    # Create meshgrid for heatmap
    time_axis = np.arange(len(df_zone))
    price_grid, time_grid = np.meshgrid(price_bins, time_axis)
    V_diff_grid = np.tile(V_diff, (len(time_axis), 1))

    # Split into above/below current price
    mask_above = price_bins >= price_now
    mask_below = price_bins < price_now

    # USE GLOBAL vmax (calculated once for all snapshots)

    # Plot heatmap for ABOVE current price
    # 0 (white/transparent) -> Red (max resistance)
    # Map: V_diff negative (resistance) = red, starting from 0
    V_diff_above = V_diff.copy()
    V_diff_above[mask_below] = np.nan
    if mask_above.any():
        # Invert: negative V_diff (resistance) becomes positive for red mapping
        V_diff_inverted = -V_diff_above
        V_diff_grid_above = np.tile(V_diff_inverted, (len(time_axis), 1))
        im_above = ax_left.pcolormesh(time_grid, price_grid, V_diff_grid_above,
                                       cmap=cmap_above, alpha=0.4,
                                       vmin=0, vmax=vmax_global,
                                       shading='auto', zorder=1)
    else:
        im_above = None

    # Plot heatmap for BELOW current price
    # 0 (white/transparent) -> Green (max support)
    # Map: V_diff positive (support) = green, starting from 0
    V_diff_below = V_diff.copy()
    V_diff_below[mask_above] = np.nan
    if mask_below.any():
        V_diff_grid_below = np.tile(V_diff_below, (len(time_axis), 1))
        im_below = ax_left.pcolormesh(time_grid, price_grid, V_diff_grid_below,
                                       cmap=cmap_below, alpha=0.4,
                                       vmin=0, vmax=vmax_global,
                                       shading='auto', zorder=1)
    else:
        im_below = None

    # Overlay price line
    ax_left.plot(range(len(df_zone)), df_zone.values, 'k-', linewidth=1, zorder=3)
    ax_left.axhline(price_now, color='orange', linestyle='--', linewidth=3,
                    label=f'Current: ${price_now:.0f}', zorder=5)

    # X-axis: Date labels
    num_date_labels = [8, 10, 11][idx] if idx < 3 else 12
    date_indices = np.linspace(0, len(df_zone)-1, min(num_date_labels, len(df_zone))).astype(int)
    date_labels = [df_zone.index[i].strftime('%b %d') for i in date_indices]
    ax_left.set_xticks(date_indices)
    ax_left.set_xticklabels(date_labels, rotation=45, ha='right')

    ax_left.set_ylabel('Price (USD)', fontsize=14, fontweight='bold')
    ax_left.set_xlabel('Date', fontsize=14, fontweight='bold')
    ax_left.set_title('Price Evolution with Resistance/Support Heatmap', fontsize=13, fontweight='bold', pad=15)
    legend_fontsize = [12, 11, 11][idx] if idx < 3 else 11
    ax_left.legend(fontsize=legend_fontsize, loc='upper left')
    ax_left.grid(alpha=0.3, linestyle=':', zorder=2)
    ax_left.set_ylim(snapshot['p_min'], snapshot['p_max'])

    # Price ticks every 1K with format "xxx.xK"
    price_ticks = np.arange(snapshot['p_min'], snapshot['p_max'] + 1000, 1000)
    ax_left.set_yticks(price_ticks)
    ax_left.set_yticklabels([f'{p/1000:.1f}K' for p in price_ticks])

    # Calculate peak/bottom for drawdown (Zone 2 only, but don't display)
    if idx == 1:
        peak_price = df_zone.max()
        bottom_price = df_zone.min()

    # RIGHT: V_diff profile (absolute values, higher = stronger)
    ax_right = fig.add_subplot(gs[1], sharey=ax_left)

    # Add heatmap to right subplot for visual continuity
    # Create single-column meshgrid for V_diff background
    x_right = np.linspace(0, 1, 2)  # X coordinates for column
    price_right = price_bins  # Y coordinates (price levels)

    # Plot heatmap for ABOVE current price (resistance)
    if mask_above.any():
        V_diff_inverted_right = -V_diff.copy()
        V_diff_inverted_right[mask_below] = np.nan
        # Create meshgrid with proper dimensions
        X_right, Y_right = np.meshgrid(x_right, price_right)
        V_grid_right = np.tile(V_diff_inverted_right[:, np.newaxis], (1, len(x_right)))
        ax_right.pcolormesh(X_right, Y_right, V_grid_right,
                             cmap=cmap_above, alpha=0.4, vmin=0, vmax=vmax_global,
                             shading='auto', zorder=1)

    # Plot heatmap for BELOW current price (support)
    if mask_below.any():
        V_diff_right_below = V_diff.copy()
        V_diff_right_below[mask_above] = np.nan
        X_right, Y_right = np.meshgrid(x_right, price_right)
        V_grid_right_below = np.tile(V_diff_right_below[:, np.newaxis], (1, len(x_right)))
        ax_right.pcolormesh(X_right, Y_right, V_grid_right_below,
                             cmap=cmap_below, alpha=0.4, vmin=0, vmax=vmax_global,
                             shading='auto', zorder=1)

    ax_right.axhline(price_now, color='orange', linestyle='--', linewidth=3, zorder=5)

    # Process V_diff: convert to absolute magnitude (higher = stronger zone)
    V_diff_plot = V_diff.copy()
    # Above current price: negative resistance → positive magnitude
    V_diff_plot[mask_above] = -V_diff_plot[mask_above]
    # Below current price: positive support → keep positive (already correct)
    # Result: all positive values, larger = stronger zone

    ax_right.plot(V_diff_plot, price_bins, 'k-', linewidth=2.5, label='Liquidity Strength', zorder=4)
    ax_right.axvline(0, color='black', linestyle='-', linewidth=1.5, alpha=0.5, zorder=1)
    ax_right.set_title('Directional Liquidity Density Profile', fontsize=13, fontweight='bold', pad=15)
    ax_right.legend(fontsize=12, loc='upper right')
    ax_right.grid(alpha=0.3, linestyle=':')
    ax_right.tick_params(labelleft=False)
    # Fixed X-axis range: ALWAYS 0 to VDIFF_XMAX_FIXED (same for all snapshots)
    ax_right.set_xlim(0, VDIFF_XMAX_FIXED)
    ax_right.invert_xaxis()

    fig.suptitle(snapshot['suptitle'],
                 fontsize=16, fontweight='bold', y=0.98)

    # PROFESSIONAL LEGEND BOX (full width, well below plots)
    # Row 1: "Resistance" label (0-12%) + colorbar (15-95%)
    # Row 2: "Support" label (0-12%) + colorbar (15-95%)

    # Add text labels for Resistance and Support - positioned lower with more space
    fig.text(0.10, 0.04, 'Resistance', fontsize=11, fontweight='bold', va='center', ha='right')
    fig.text(0.10, 0.01, 'Support', fontsize=11, fontweight='bold', va='center', ha='right')

    # Colorbar axes: start at 15% (0.15) to leave more room for labels
    cbar_ax_resistance = fig.add_axes([0.15, 0.035, 0.75, 0.015])  # [left, bottom, width, height]
    cbar_ax_support = fig.add_axes([0.15, 0.01, 0.75, 0.015])

    if im_above is not None:
        cbar_above = fig.colorbar(im_above, cax=cbar_ax_resistance, orientation='horizontal')
        cbar_above.set_ticks([])

    if im_below is not None:
        cbar_below = fig.colorbar(im_below, cax=cbar_ax_support, orientation='horizontal')
        cbar_below.set_ticks([])

    # Add watermark inside the main plot (left subplot)
    ax_left.text(0.98, 0.02, '@FedeAnastasi', fontsize=9, color='gray',
                  alpha=0.5, ha='right', va='bottom', transform=ax_left.transAxes,
                  bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'))

    output_file = results_dir / snapshot['output_file']
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_file}")
    outputs.append(output_file)
    plt.close()


# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("COMPLETE")
print("="*80)
print(f"\nOutputs:")
for output in outputs:
    print(f"  {output}")
print("="*80)
