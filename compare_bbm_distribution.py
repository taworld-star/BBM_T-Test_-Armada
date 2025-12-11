import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def parse_datetime_safe(col):
    """
    Parse datetime with multiple format attempts
    """
    col = col.astype(str).str.strip()
    
    # Try common formats
    formats = [
        '%d/%m/%Y %H:%M:%S',
        '%Y/%m/%d %H:%M:%S',
        '%Y-%m-%d %H:%M:%S'
    ]
    
    parsed = pd.to_datetime(col, format=formats[0], errors='coerce', dayfirst=True)
    
    for fmt in formats[1:]:
        if parsed.isna().sum() > len(col) * 0.5:
            parsed = pd.to_datetime(col, format=fmt, errors='coerce')
            
    # Fallback to auto-detect
    if parsed.isna().sum() > len(col) * 0.5:
        parsed = pd.to_datetime(col, errors='coerce')
    
    return parsed

def load_data(filepath, sheet_name):
    """
    Load and clean data from Excel
    """
    print(f"Loading data from {sheet_name}...", flush=True)
    df = pd.read_excel(filepath, sheet_name=sheet_name)
    
    # Parse datetime
    df['GPSTIME'] = parse_datetime_safe(df['GPSTIME'])
    
    # Convert numeric columns
    numeric_cols = ['VALUE FUEL SENSOR', 'ODOMETER', 'SPEED']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    # Drop invalid rows
    df = df.dropna(subset=['GPSTIME'])
    df = df.sort_values('GPSTIME').reset_index(drop=True)
    
    # Filter ACC ON
    df_active = df[df['ACC'] == 'ON'].copy()
    
    print(f"  Total rows: {len(df)}")
    print(f"  ACC ON rows: {len(df_active)}")
    return df_active

def calculate_trips(df, min_distance=0.5, min_fuel=0.2):
    """
    Calculate fuel consumption per trip
    """
    df = df.copy()
    
    # Calculate differences
    df['time_diff'] = df['GPSTIME'].diff().dt.total_seconds() / 60
    df['odo_diff'] = df['ODOMETER'].diff()
    df['fuel_diff'] = df['VALUE FUEL SENSOR'].diff()
    
    # Trip detection logic: Gap > 60 mins OR Odo reset OR Refuel (>10L)
    df['new_trip'] = ((df['time_diff'] > 60) | 
                      (df['odo_diff'] < 0) | 
                      (df['fuel_diff'] > 10))
    
    df['trip_id'] = df['new_trip'].cumsum()
    
    trip_list = []
    
    for trip_id, trip_data in df.groupby('trip_id'):
        if len(trip_data) < 2:
            continue
            
        distance = trip_data['ODOMETER'].iloc[-1] - trip_data['ODOMETER'].iloc[0]
        fuel_used = trip_data['VALUE FUEL SENSOR'].iloc[0] - trip_data['VALUE FUEL SENSOR'].iloc[-1]
        
        # Valid trip filter
        if distance > min_distance and fuel_used > min_fuel:
            consumption = distance / fuel_used
            
            # Realistic consumption filter (1-20 km/L)
            if 1 <= consumption <= 20:
                trip_list.append({
                    'trip_id': trip_id,
                    'distance_km': distance,
                    'fuel_used_L': fuel_used,
                    'consumption_kmL': consumption,
                    'start_time': trip_data['GPSTIME'].iloc[0]
                })
                
    return pd.DataFrame(trip_list)

def main():
    filepath = 'Data W9371UM ULTRASONIK.xlsx'
    
    # 1. Load Data
    print("="*60)
    print("DATA LOADING")
    print("="*60)
    
    try:
        df_baru = load_data(filepath, 'W9371UM BARU')
        df_lama = load_data(filepath, 'W9371UM LAMA')
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 2. Calculate Trips
    print("\n" + "="*60)
    print("TRIP CALCULATION")
    print("="*60)
    
    trips_baru = calculate_trips(df_baru)
    trips_lama = calculate_trips(df_lama)
    
    print(f"New Data (Baru) Trips: {len(trips_baru)}")
    print(f"Old Data (Lama) Trips: {len(trips_lama)}")
    
    if len(trips_baru) < 2 or len(trips_lama) < 2:
        print("Error: Not enough data points for t-test (need at least 2 per group).")
        return

    # 3. Statistical Analysis
    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS (Independent t-test)")
    print("="*60)
    
    group1 = trips_baru['consumption_kmL']
    group2 = trips_lama['consumption_kmL']
    
    # Descriptive Stats
    print("\nDescriptive Statistics:")
    print("-" * 40)
    print(f"{'Metric':<15} {'New Data':<15} {'Old Data':<15}")
    print("-" * 40)
    print(f"{'Mean':<15} {group1.mean():.4f} {'km/L':<5} {group2.mean():.4f} {'km/L':<5}")
    print(f"{'Std Dev':<15} {group1.std():.4f} {'km/L':<5} {group2.std():.4f} {'km/L':<5}")
    print(f"{'Count':<15} {len(group1):<15} {len(group2):<15}")
    print("-" * 40)
    
    # Levene's Test (Equality of Variances)
    lev_stat, lev_p = stats.levene(group1, group2)
    print(f"\nLevene's Test for Equality of Variances:")
    print(f"  Statistic = {lev_stat:.4f}, p-value = {lev_p:.4f}")
    equal_var = lev_p > 0.05
    if equal_var:
        print("  Result: Variances are equal (p > 0.05)")
    else:
        print("  Result: Variances are NOT equal (p < 0.05) -> Using Welch's t-test")
        
    # T-test
    t_stat, t_p = stats.ttest_ind(group1, group2, equal_var=equal_var)
    
    print(f"\nIndependent Samples T-test:")
    print(f"  t-statistic = {t_stat:.4f}")
    print(f"  p-value     = {t_p:.4f}")
    
    alpha = 0.05
    print(f"\nConclusion (alpha={alpha}):")
    # Mann-Whitney U Test
    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS (Mann-Whitney U Test)")
    print("="*60)
    
    u_stat, u_p = stats.mannwhitneyu(group1, group2, alternative='two-sided')
    
    print(f"\nMann-Whitney U Test:")
    print(f"  U-statistic = {u_stat:.4f}")
    print(f"  p-value     = {u_p:.4f}")
    
    print(f"\nConclusion (alpha={alpha}):")
    if u_p < alpha:
        print("  REJECT Null Hypothesis.")
        print("  There is a SIGNIFICANT difference in fuel consumption distribution.")
    else:
        print("  FAIL TO REJECT Null Hypothesis.")
        print("  There is NO significant difference in fuel consumption distribution.")

    # 4. Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Combine data for plotting
    plot_data = pd.DataFrame({
        'Consumption (km/L)': pd.concat([group1, group2]),
        'Dataset': ['New Data'] * len(group1) + ['Old Data'] * len(group2)
    })
    
    # Plot 1: Boxplot with Swarmplot
    sns.boxplot(x='Dataset', y='Consumption (km/L)', data=plot_data, showfliers=False, ax=axes[0])
    sns.swarmplot(x='Dataset', y='Consumption (km/L)', data=plot_data, color=".25", size=8, ax=axes[0])
    axes[0].set_title('Distribution Comparison (Boxplot)', fontsize=12)
    
    # Add stats to Boxplot
    stats_text = (f"MW U-stat: {u_stat:.2f}\n"
                  f"p-value: {u_p:.4f}\n"
                  f"New Mean: {group1.mean():.2f}\n"
                  f"Old Mean: {group2.mean():.2f}")
    
    axes[0].text(0.95, 0.95, stats_text, transform=axes[0].transAxes, 
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Plot 2: ECDF (Empirical Cumulative Distribution Function)
    sns.ecdfplot(data=plot_data, x='Consumption (km/L)', hue='Dataset', ax=axes[1], linewidth=2)
    axes[1].set_title('Cumulative Distribution (ECDF)', fontsize=12)
    axes[1].set_ylabel('Proportion')
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle('Mann-Whitney U Test Visualization: New Data vs Old Data', fontsize=16)
    plt.tight_layout()
    plt.savefig('bbm_comparison_mw_ecdf.png')
    print("\nVisualization saved to 'bbm_comparison_mw_ecdf.png'", flush=True)

if __name__ == "__main__":
    main()
