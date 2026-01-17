"""
Kendall's Tau Correlation Analysis for Inter-Rater Agreement

This script calculates Kendall's τ (tau) correlation coefficient to measure
pairwise agreement between raters across different methods and metrics.

Kendall's τ interpretation:
- τ = 1.0: Perfect positive correlation
- 0.7 < τ < 1.0: Strong positive correlation
- 0.4 < τ < 0.7: Moderate positive correlation
- 0.0 < τ < 0.4: Weak positive correlation
- τ = 0.0: No correlation
- τ < 0.0: Negative correlation
"""

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from rich import print
from itertools import combinations


def load_ratings_data(csv_path='data/human_scores.csv', num_columns=72):
    """
    Load and preprocess ratings data from CSV file.
    
    Args:
        csv_path: Path to the CSV file containing ratings
        num_columns: Number of rating columns to extract (default: 72)
    
    Returns:
        numpy.ndarray: Ratings matrix with shape (n_raters, num_columns)
    """
    data_df = pd.read_csv(csv_path, header=None)
    
    # Extract rating columns (skip first column with rater labels)
    # Skip header row and summary rows
    ratings = data_df.iloc[1:, 1:num_columns+1].to_numpy()
    
    # Convert to float, replacing invalid values with NaN
    ratings = pd.DataFrame(ratings).apply(pd.to_numeric, errors='coerce').to_numpy()
    
    return ratings


def calculate_pairwise_kendall_tau(data_slice):
    """
    Calculate pairwise Kendall's τ correlation for all rater pairs.
    
    Args:
        data_slice: Data matrix with shape (n_raters, n_items)
    
    Returns:
        tuple: (tau_values, p_values, pairs) - Lists of correlation coefficients,
               p-values, and rater pair indices
    """
    n_raters = data_slice.shape[0]
    tau_values = []
    p_values = []
    pairs = []
    
    # Calculate Kendall's τ for each pair of raters
    for r1, r2 in combinations(range(n_raters), 2):
        rater1_scores = data_slice[r1, :]
        rater2_scores = data_slice[r2, :]
        
        # Use only items where both raters provided ratings
        valid_items = ~(np.isnan(rater1_scores) | np.isnan(rater2_scores))
        
        if np.sum(valid_items) >= 2:  # Need at least 2 common ratings
            try:
                tau, p_value = kendalltau(
                    rater1_scores[valid_items],
                    rater2_scores[valid_items]
                )
                tau_values.append(tau)
                p_values.append(p_value)
                pairs.append((r1, r2))
                print(f"    Rater {r1+1} vs Rater {r2+1}: "
                      f"τ = {tau:.4f}, p = {p_value:.4f} (n={np.sum(valid_items)})")
            except Exception as e:
                print(f"    Rater {r1+1} vs Rater {r2+1}: Calculation failed - {e}")
    
    return tau_values, p_values, pairs


def calculate_statistics(tau_values):
    """
    Calculate descriptive statistics for Kendall's τ values.
    
    Args:
        tau_values: List of tau correlation coefficients
    
    Returns:
        dict: Dictionary containing mean, std, median, min, max, and count
    """
    if not tau_values:
        return {
            'mean': np.nan, 'std': np.nan, 'median': np.nan,
            'min': np.nan, 'max': np.nan, 'count': 0
        }
    
    # Remove NaN values before calculating statistics
    tau_clean = [t for t in tau_values if not np.isnan(t)]
    
    if not tau_clean:
        print(f"    ⚠️ All pairs resulted in NaN (possibly all identical values)")
        print(f"      Pair count: {len(tau_values)}")
        return {
            'mean': np.nan, 'std': np.nan, 'median': np.nan,
            'min': np.nan, 'max': np.nan, 'count': len(tau_values)
        }
    
    stats = {
        'mean': np.mean(tau_clean),
        'std': np.std(tau_clean),
        'median': np.median(tau_clean),
        'min': np.min(tau_clean),
        'max': np.max(tau_clean),
        'count': len(tau_values)
    }
    
    print(f"\n    Statistics Summary:")
    print(f"      Mean τ: {stats['mean']:.4f} ± {stats['std']:.4f}")
    print(f"      Median τ: {stats['median']:.4f}")
    print(f"      Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
    print(f"      Valid pairs: {len(tau_clean)} / {len(tau_values)}")
    
    if len(tau_clean) < len(tau_values):
        print(f"      ⚠️ {len(tau_values) - len(tau_clean)} pairs resulted in NaN")
    
    return stats


def main():
    """Main function to calculate Kendall's τ for all methods and metrics."""
    
    # Configuration
    METHODS = ['SA', 'LLMD', 'BILLY']
    METRICS = ['Originality', 'Elaboration']
    ITEM_COUNT = 12  # Number of items per metric per method
    
    # Load ratings data
    print("Loading ratings data...")
    RATINGS = load_ratings_data()
    print(f"Data shape: {RATINGS.shape}")
    print(f"Sample data:\n{RATINGS[:3, :10]}\n")

    # Calculate Kendall's τ for each method and metric
    print("\n" + "="*80)
    print("Calculating Kendall's τ (Tau) Correlation Coefficient")
    print("="*80)
    
    all_results = []
    
    for i, method in enumerate(METHODS):
        # Calculate column indices for each method
        # i=0 (SA): Originality cols 0-11, Elaboration cols 12-23
        # i=1 (LLMD): Originality cols 24-35, Elaboration cols 36-47
        # i=2 (BILLY): Originality cols 48-59, Elaboration cols 60-71
        ori_start = i * (2 * ITEM_COUNT)
        ela_start = ori_start + ITEM_COUNT
        
        print(f"\n{'='*80}")
        print(f"【{method}】")
        print(f"{'='*80}")
        
        for metric_name, start_idx in [('Originality', ori_start), ('Elaboration', ela_start)]:
            print(f"\n  【{metric_name}】 (columns {start_idx}-{start_idx + ITEM_COUNT - 1})")
            
            # Extract data for current metric
            data_slice = RATINGS[:, start_idx:start_idx + ITEM_COUNT]
            n_raters = data_slice.shape[0]
            
            # Check valid data
            valid_count = np.sum(~np.isnan(data_slice))
            print(f"    Valid ratings: {valid_count} / {data_slice.size}")
            print(f"    Number of raters: {n_raters}")
            
            # Calculate pairwise Kendall's τ
            tau_values, p_values, pairs = calculate_pairwise_kendall_tau(data_slice)
            
            # Calculate and display statistics
            stats = calculate_statistics(tau_values)
            
            # Store results
            all_results.append({
                'Method': method,
                'Metric': metric_name,
                'Mean_Tau': stats['mean'],
                'Std_Tau': stats['std'],
                'Median_Tau': stats['median'],
                'Min_Tau': stats['min'],
                'Max_Tau': stats['max'],
                'N_Pairs': stats['count']
            })
    
    # Display and save results
    print("\n" + "="*80)
    print("Kendall's τ Results Summary")
    print("="*80)
    
    results_df = pd.DataFrame(all_results)
    print(results_df.to_string(index=False))
    
    # Save detailed results
    output_path = 'data/kendall_tau_results.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\n✓ Detailed results saved to: {output_path}")
    
    # Create simplified results (mean values only)
    summary_df = results_df[['Method', 'Metric', 'Mean_Tau', 'Std_Tau']].copy()
    summary_df['Result'] = summary_df.apply(
        lambda row: f"{row['Mean_Tau']:.4f} ± {row['Std_Tau']:.4f}"
        if not np.isnan(row['Mean_Tau']) else "N/A",
        axis=1
    )
    
    print("\n" + "="*80)
    print("Simplified Results (Mean Kendall's τ ± Std Dev)")
    print("="*80)
    print(summary_df[['Method', 'Metric', 'Result']].to_string(index=False))
    
    # Calculate overall average across all methods
    print("\n" + "="*80)
    print("Total Average Correlation (Across All Methods)")
    print("="*80)
    
    ori_results = results_df[results_df['Metric'] == 'Originality']['Mean_Tau'].dropna()
    ela_results = results_df[results_df['Metric'] == 'Elaboration']['Mean_Tau'].dropna()
    
    if len(ori_results) > 0:
        ori_mean = ori_results.mean()
        ori_std = ori_results.std()
        print(f"Originality (Overall): {ori_mean:.4f} ± {ori_std:.4f}")
        print(f"  - Based on average of {len(ori_results)} methods")
    else:
        print(f"Originality (Overall): N/A")
    
    if len(ela_results) > 0:
        ela_mean = ela_results.mean()
        ela_std = ela_results.std()
        print(f"Elaboration (Overall): {ela_mean:.4f} ± {ela_std:.4f}")
        print(f"  - Based on average of {len(ela_results)} methods")
    else:
        print(f"Elaboration (Overall): N/A")
    
    # Display interpretation guide
    print("\n" + "="*80)
    print("Kendall's τ Interpretation Guide")
    print("="*80)
    print("τ = 1.0: Perfect positive correlation")
    print("0.7 < τ < 1.0: Strong positive correlation")
    print("0.4 < τ < 0.7: Moderate positive correlation")
    print("0.0 < τ < 0.4: Weak positive correlation")
    print("τ = 0.0: No correlation")
    print("τ < 0.0: Negative correlation")
    print("="*80)


if __name__ == "__main__":
    main()
print("\n註: p < 0.05 表示相關性在統計上顯著")
print("="*80)
