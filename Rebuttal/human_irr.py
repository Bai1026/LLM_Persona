"""
Inter-Rater Reliability (IRR) Analysis using Krippendorff's Alpha

This script calculates Krippendorff's Alpha for measuring inter-rater reliability
across different methods (SA, LLMD, BILLY) and metrics (Originality, Elaboration).

Krippendorff's Alpha interpretation:
- α ≥ 0.800: Excellent reliability
- 0.667 ≤ α < 0.800: Acceptable reliability
- α < 0.667: Insufficient reliability
"""

import numpy as np
import pandas as pd
from rich import print
import krippendorff


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


def calculate_krippendorff_alpha(data, level='ordinal'):
    """
    Calculate Krippendorff's Alpha for a given data slice.
    
    Args:
        data: Data matrix in (raters x items) format
        level: Level of measurement ('nominal', 'ordinal', 'interval', 'ratio')
    
    Returns:
        float: Krippendorff's Alpha value, or NaN if calculation fails
    """
    # Transpose data from (Rater x Item) to (Item x Rater) format
    data_transposed = data.T
    
    try:
        alpha = krippendorff.alpha(
            reliability_data=data_transposed,
            level_of_measurement=level
        )
        return alpha
    except Exception as e:
        print(f"  ✗ Calculation failed: {e}")
        return np.nan


def main():
    """Main function to calculate Krippendorff's Alpha for all methods and metrics."""
    
    # Configuration
    METHODS = ['SA', 'LLMD', 'BILLY']
    METRICS = ['Originality', 'Elaboration']
    ITEM_COUNT = 12  # Number of items per metric per method
    RESULTS = {}
    
    # Load ratings data
    print("Loading ratings data...")
    RATINGS = load_ratings_data()
    print(f"Data shape: {RATINGS.shape}")
    print(f"Expected shape: (n_raters, 72)")
    print(f"Sample data:\n{RATINGS[:3, :10]}\n")

    # Calculate Krippendorff's Alpha for each method and metric
    print("\n" + "="*80)
    print("Calculating Krippendorff's Alpha")
    print("="*80)
    
    for i, method in enumerate(METHODS):
        # Calculate column indices for each method
        # i=0 (SA): Originality cols 0-11, Elaboration cols 12-23
        # i=1 (LLMD): Originality cols 24-35, Elaboration cols 36-47
        # i=2 (BILLY): Originality cols 48-59, Elaboration cols 60-71
        ori_start = i * (2 * ITEM_COUNT)
        ela_start = ori_start + ITEM_COUNT
        
        print(f"\n【{method}】")
        print(f"  Originality: columns {ori_start}-{ori_start + ITEM_COUNT - 1}")
        print(f"  Elaboration: columns {ela_start}-{ela_start + ITEM_COUNT - 1}")
        
        # Process Originality metric
        ori_data = RATINGS[:, ori_start:ori_start + ITEM_COUNT]
        ori_valid_count = np.sum(~np.isnan(ori_data))
        print(f"  Originality valid ratings: {ori_valid_count} / {ori_data.size}")
        
        alpha_ori = calculate_krippendorff_alpha(ori_data)
        RESULTS[f'{method}_Originality'] = alpha_ori
        if not np.isnan(alpha_ori):
            print(f"  ✓ Originality Alpha = {alpha_ori:.4f}")
        
        # Process Elaboration metric
        ela_data = RATINGS[:, ela_start:ela_start + ITEM_COUNT]
        ela_valid_count = np.sum(~np.isnan(ela_data))
        print(f"  Elaboration valid ratings: {ela_valid_count} / {ela_data.size}")
        
        alpha_ela = calculate_krippendorff_alpha(ela_data)
        RESULTS[f'{method}_Elaboration'] = alpha_ela
        if not np.isnan(alpha_ela):
            print(f"  ✓ Elaboration Alpha = {alpha_ela:.4f}")
    
    # Display and save results
    print("\n" + "="*80)
    print("Krippendorff's Alpha Results")
    print("="*80)
    
    results_df = pd.DataFrame({
        'Method': [k.replace('_', ' ') for k in RESULTS.keys()],
        'Alpha': [f"{v:.4f}" if not np.isnan(v) else "N/A" for v in RESULTS.values()]
    })
    
    print(results_df.to_string(index=False))
    
    # Save results to CSV
    output_path = 'data/krippendorff_alpha_results.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\n✓ Results saved to: {output_path}")
    
    # Display interpretation guide
    print("\n" + "="*80)
    print("Krippendorff's Alpha Interpretation Guide")
    print("="*80)
    print("α ≥ 0.800: Excellent reliability")
    print("0.667 ≤ α < 0.800: Acceptable reliability")
    print("α < 0.667: Insufficient reliability")
    print("="*80)


if __name__ == "__main__":
    main()