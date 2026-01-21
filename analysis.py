import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

def load_data(filepath):
    """Loads the user funnel data."""
    return pd.read_csv(filepath)

def calculate_funnel_metrics(df):
    """Calculates funnel counts, conversion rates, and drop-off rates."""
    stages = ['Homepage', 'Product_Page', 'Cart', 'Checkout', 'Purchase']
    
    # Count unique users per stage
    funnel_counts = df.groupby('stage')['user_id'].nunique()
    
    # Reindex to ensure order
    funnel_counts = funnel_counts.reindex(stages).fillna(0)
    
    metrics = pd.DataFrame({
        'Users': funnel_counts,
    })
    
    # Calculate previous stage users for conversion calculation
    metrics['Prev_Users'] = metrics['Users'].shift(1)
    
    # Conversion Rate: Users / Prev_Users
    # For Homepage, it's 100% (or NaN if we strictly follow the formula, but logic says 1.0 relative to start)
    metrics['Conversion_Rate'] = metrics['Users'] / metrics['Prev_Users'] 
    metrics['Conversion_Rate'] = metrics['Conversion_Rate'].fillna(1.0) # Fill first stage
    
    # Drop-off Rate: 1 - Conversion Rate
    metrics['Drop_off_Rate'] = 1 - metrics['Conversion_Rate']

    
    # We need to fill NA for calculation, using fillna(0) for safe math, though logic handles it.
    p = metrics['Conversion_Rate']
    n = metrics['Prev_Users']
    
    metrics['Std_Err'] = np.sqrt(p * (1 - p) / n)
    metrics['Std_Err'] = metrics['Std_Err'].fillna(0) # First stage has no SE
    metrics['CI_95'] = 1.96 * metrics['Std_Err']
    
    return metrics

def plot_funnel(metrics, title="User Conversion Funnel", filename="funnel_chart.png"):
    """Plots the funnel visualization."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    stages = metrics.index
    users = metrics['Users']
    
    bars = ax.bar(stages, users, color='skyblue', edgecolor='navy', yerr=metrics['CI_95'] * users, capsize=5)

    
    N_total = users.iloc[0]
    p_cumulative = users / N_total
    se_cumulative = np.sqrt(p_cumulative * (1 - p_cumulative) / N_total)
    error_bars = 1.96 * se_cumulative * N_total
    
    # Update bar call
    bars = ax.bar(stages, users, color='skyblue', edgecolor='navy', yerr=error_bars, capsize=5)
    
    # Add labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 5, # Move text up slightly
                f'{int(height)}',
                ha='center', va='bottom')
        
    # Add conversion / drop-off text
    for i, stage in enumerate(stages):
        if i > 0:
            conv_rate = metrics.loc[stage, 'Conversion_Rate']
            drop_rate = metrics.loc[stage, 'Drop_off_Rate']
            ax.text(i, users[stage]/2, 
                    f"Conv: {conv_rate:.1%}\nDrop: {drop_rate:.1%}", 
                    ha='center', va='center', color='black', fontsize=9, fontweight='bold')

    ax.set_title(title)
    ax.set_ylabel("Number of Users")
    ax.set_xlabel("Stage")
    
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved plot to {filename}")
    plt.close()

def plot_segmented_funnel(df, segment_col, filename_prefix="segment"):
    """Plots funnel segmented by a column (e.g., Device, Source)."""
    stages = ['Homepage', 'Product_Page', 'Cart', 'Checkout', 'Purchase']
    
    segments = df[segment_col].unique()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for segment in segments:
        segment_data = df[df[segment_col] == segment]
        counts = segment_data.groupby('stage')['user_id'].nunique().reindex(stages).fillna(0)
        ax.plot(stages, counts, marker='o', label=segment)
        
    ax.set_title(f"Funnel by {segment_col}")
    ax.set_ylabel("Number of Users")
    ax.set_xlabel("Stage")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    output_file = f"{filename_prefix}_{segment_col.lower()}.png"
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Saved segmentation plot to {output_file}")
    plt.close()

import argparse
import sys


def calculate_time_to_convert(df):
    """Calculates average time between stages."""
    # Ensure timestamp is datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Sort by user and time
    df = df.sort_values(['user_id', 'timestamp'])
    
    # Calculate time difference between rows for the same user
    df['prev_stage_time'] = df.groupby('user_id')['timestamp'].shift(1)
    df['time_diff'] = (df['timestamp'] - df['prev_stage_time']).dt.total_seconds() / 60.0 # in minutes
    
    
    avg_times = df.groupby('stage')['time_diff'].mean().reindex(['Product_Page', 'Cart', 'Checkout', 'Purchase'])
    return avg_times

def plot_time_to_convert(avg_times, filename="time_to_convert.png"):
    """Plots the average time to convert between stages."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    stages = avg_times.index
    times = avg_times.values
    
    ax.bar(stages, times, color='orange', edgecolor='brown')
    
    ax.set_title("Average Time to Next Stage (Minutes)")
    ax.set_ylabel("Minutes")
    ax.set_xlabel("Target Stage")
    
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved plot to {filename}")
    plt.close()

def calculate_total_efficiency(metrics):
    """Calculates Total Funnel Efficiency."""
    initial_users = metrics['Users'].iloc[0]
    final_users = metrics['Users'].iloc[-1]
    
    if initial_users == 0:
        return 0.0
        
    efficiency = final_users / initial_users
    return efficiency

def analyze_churn_velocity(df):
    """Analyzes when users drop off."""
    # 1. Identify drop-off users and their last stage timestamp
    # Sort by user and time
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    last_actions = df.sort_values('timestamp').groupby('user_id').last()
    
    # Drop-off means they didn't reach 'Purchase'
    drop_offs = last_actions[last_actions['stage'] != 'Purchase'].copy()
    
    # Extract time features
    drop_offs['hour'] = drop_offs['timestamp'].dt.hour
    drop_offs['day_of_week'] = drop_offs['timestamp'].dt.day_name()
    
    return drop_offs

def plot_churn_velocity(drop_offs, filename="churn_velocity_hour.png"):
    """Plots drop-offs by hour of day."""
    hourly_counts = drop_offs['hour'].value_counts().sort_index()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(hourly_counts.index, hourly_counts.values, marker='o', color='red', linestyle='-')
    
    ax.set_title("Churn Velocity: Drop-offs by Hour of Day")
    ax.set_ylabel("Number of Drop-offs")
    ax.set_xlabel("Hour of Day (0-23)")
    ax.set_xticks(range(0, 24))
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved plot to {filename}")
    plt.close()

def perform_ab_test(df, segment_col='device'):
    """
    Performs Chi-squared test on conversion rates for different segments.
    We'll test if the 'Purchase' rate (Overall Conversion) differs significantly by segment.
    """
    print(f"\n--- A/B Testing (Chi-Squared) for {segment_col} ---")
    
    # Create a contingency table: Segment vs (Converted, Not Converted)
    # Converted = Reached 'Purchase'
    # Base = All Unique Users in that segment
    
    conversions = df[df['stage'] == 'Purchase'].groupby(segment_col)['user_id'].nunique()
    total_users = df.groupby(segment_col)['user_id'].nunique()
    
    # Ensure indices match
    summary = pd.DataFrame({'Total': total_users, 'Converted': conversions}).fillna(0)
    summary['Not_Converted'] = summary['Total'] - summary['Converted']
    
    print("Contingency Table:")
    print(summary[['Converted', 'Not_Converted']])
    
    # Chi-squared test
    observed = summary[['Converted', 'Not_Converted']].values
    chi2, p, dof, expected = stats.chi2_contingency(observed)
    
    print(f"\nChi-Squared Statistic: {chi2:.4f}")
    print(f"P-Value: {p:.4f}")
    
    alpha = 0.05
    if p < alpha:
        print("Result: Significant difference between segments (Reject H0)")
    else:
        print("Result: No significant difference detected (Fail to reject H0)")
    
    return summary, p

def main():
    parser = argparse.ArgumentParser(description="Analyze user funnel data.")
    parser.add_argument("file", help="Path to the CSV file containing user funnel data")
    args = parser.parse_args()

    data_file = args.file
    try:
        df = load_data(data_file)
    except FileNotFoundError:
        print(f"Error: File '{data_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)

    print(f"Analyzing data from: {data_file}")
    print("\n--- Overall Funnel Metrics ---")
    metrics = calculate_funnel_metrics(df)
    print(metrics)
    plot_funnel(metrics)
    
    efficiency = calculate_total_efficiency(metrics)
    print(f"\nTotal Funnel Efficiency: {efficiency:.2%}")
    
    print("\n--- Time to Convert ---")
    avg_times = calculate_time_to_convert(df)
    print(avg_times)
    plot_time_to_convert(avg_times)
    
    print("\n--- Churn Velocity ---")
    drop_offs = analyze_churn_velocity(df)
    print(f"Total analyzed drop-offs: {len(drop_offs)}")
    plot_churn_velocity(drop_offs)

    print("\n--- Segmenting by Device ---")
    plot_segmented_funnel(df, 'device')
    perform_ab_test(df, 'device')
    
    print("\n--- Segmenting by Source ---")
    plot_segmented_funnel(df, 'source')
    perform_ab_test(df, 'source')

if __name__ == "__main__":
    main()
