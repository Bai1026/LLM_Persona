"""
LLM-Human Score Correlation Analysis

This script calculates Spearman's rank correlation and Pearson correlation
between human average scores and LLM scores to evaluate the alignment
between human judgments and automated LLM evaluation.

Correlation interpretation:
- |r| ≥ 0.7: Strong correlation
- 0.4 ≤ |r| < 0.7: Moderate correlation
- 0.2 ≤ |r| < 0.4: Weak correlation
- |r| < 0.2: Very weak or no correlation
"""

import numpy as np
from scipy.stats import spearmanr, pearsonr
from rich import print


def parse_scores(score_str):
    """
    Parse comma-separated score string into a list of floats.
    
    Args:
        score_str: Comma-separated string of numeric scores
    
    Returns:
        list: List of float values
    """
    # Remove whitespace and handle potential "..." markers
    cleaned_str = score_str.replace("...", "").replace(" ", "")
    # Convert to float and filter empty values
    return [float(x) for x in cleaned_str.split(',') if x]


def load_scores_from_csv(csv_path='data/merged_human_evaluation.csv'):
    """
    Load human and LLM scores from CSV file.
    
    Args:
        csv_path: Path to the CSV file containing score data
    
    Returns:
        tuple: (human_str, llm_str) - Raw score strings from CSV
    """
    with open(csv_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        # Extract human scores (row 2, skip first character)
        human_str = lines[1].strip()[1:]
        # Extract LLM scores (row 3, skip first 4 characters)
        llm_str = lines[2].strip()[4:]
    
    return human_str, llm_str


def calculate_correlations(human_scores, llm_scores):
    """
    Calculate both Spearman and Pearson correlation coefficients.
    
    Args:
        human_scores: List of human average scores
        llm_scores: List of LLM scores
    
    Returns:
        dict: Dictionary containing correlation results
    """
    results = {}
    
    # Spearman's Rank Correlation
    spearman_corr, spearman_p = spearmanr(human_scores, llm_scores)
    results['spearman'] = {'correlation': spearman_corr, 'p_value': spearman_p}
    
    # Pearson Correlation
    pearson_corr, pearson_p = pearsonr(human_scores, llm_scores)
    results['pearson'] = {'correlation': pearson_corr, 'p_value': pearson_p}
    
    return results


def main():
    """Main function to perform correlation analysis."""
    
    # Configuration
    N_SPLIT = 12  # Split point between Originality and Elaboration scores
    
    print("="*80)
    print("LLM-Human Score Correlation Analysis")
    print("="*80 + "\n")
    
    # Load data
    print("Loading data from CSV...")
    human_str, llm_str = load_scores_from_csv()
    
    print("\n--- Raw Data ---")
    print(f"Human Scores: {human_str}")
    print(f"LLM Scores: {llm_str}")
    
    try:
        # Parse scores
        human_scores_all = parse_scores(human_str)
        llm_scores_all = parse_scores(llm_str)
        
        # Validate data length
        if len(human_scores_all) != len(llm_scores_all):
            print("\n【Error】Number of human and LLM scores don't match. "
                  "Cannot calculate correlation.")
            return
        
        print(f"\nTotal number of paired scores: {len(human_scores_all)}")
        
        # Calculate correlation for combined data
        print("\n" + "="*80)
        print("Combined Data Correlation (Originality + Elaboration)")
        print("="*80)
        
        combined_results = calculate_correlations(human_scores_all, llm_scores_all)
        
        print(f"Spearman's Rho: {combined_results['spearman']['correlation']:.4f}")
        print(f"  P-value: {combined_results['spearman']['p_value']:.4f}")
        
        print(f"\nPearson's r: {combined_results['pearson']['correlation']:.4f}")
        print(f"  P-value: {combined_results['pearson']['p_value']:.4f}")
        
        # Calculate separate correlations for Originality and Elaboration
        print("\n" + "="*80)
        print(f"Separate Analysis (Split at index {N_SPLIT})")
        print("="*80)
        
        # Originality scores
        ori_human = human_scores_all[:N_SPLIT]
        ori_llm = llm_scores_all[:N_SPLIT]
        ori_results = calculate_correlations(ori_human, ori_llm)
        
        print(f"\n【Originality】 (n={len(ori_human)})")
        print(f"  Spearman's Rho: {ori_results['spearman']['correlation']:.4f} "
              f"(p={ori_results['spearman']['p_value']:.4f})")
        print(f"  Pearson's r: {ori_results['pearson']['correlation']:.4f} "
              f"(p={ori_results['pearson']['p_value']:.4f})")
        
        # Elaboration scores
        ela_human = human_scores_all[N_SPLIT:]
        ela_llm = llm_scores_all[N_SPLIT:]
        ela_results = calculate_correlations(ela_human, ela_llm)
        
        print(f"\n【Elaboration】 (n={len(ela_human)})")
        print(f"  Spearman's Rho: {ela_results['spearman']['correlation']:.4f} "
              f"(p={ela_results['spearman']['p_value']:.4f})")
        print(f"  Pearson's r: {ela_results['pearson']['correlation']:.4f} "
              f"(p={ela_results['pearson']['p_value']:.4f})")
        
        # Display interpretation guide
        print("\n" + "="*80)
        print("Correlation Interpretation Guide")
        print("="*80)
        print("|r| ≥ 0.7: Strong correlation")
        print("0.4 ≤ |r| < 0.7: Moderate correlation")
        print("0.2 ≤ |r| < 0.4: Weak correlation")
        print("|r| < 0.2: Very weak or no correlation")
        print("="*80)
        
        print("\nNote: Adjust N_SPLIT if the split point between "
              "Originality and Elaboration differs.")
        
    except Exception as e:
        print(f"\n【Error】Calculation failed: {e}")


if __name__ == "__main__":
    main()