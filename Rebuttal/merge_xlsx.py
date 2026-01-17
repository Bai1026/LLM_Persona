"""
Merge Multiple Excel Files for Human Evaluation Data

This script merges multiple Excel files containing human evaluation data
into a single consolidated file. The merge is performed based on email
addresses as the common key across all files.

Input: Multiple Excel files (e.g., Creativity_A.xlsx, Creativity_B.xlsx, etc.)
Output: Single merged Excel file with all data combined
"""

import pandas as pd
from pathlib import Path


def identify_email_column(df):
    """
    Automatically identify the email column in a DataFrame.
    
    Args:
        df: pandas DataFrame to search for email column
    
    Returns:
        str: Name of the email column, or None if not found
    """
    # Common email column patterns
    email_patterns = ['電子郵件', 'email', '信箱', 'e-mail', 'mail']
    
    for col in df.columns:
        col_lower = col.lower()
        if any(pattern in col_lower for pattern in email_patterns):
            return col
    
    return None


def rename_columns_with_source(df, email_col, source_index):
    """
    Rename DataFrame columns to include source identifier.
    
    Args:
        df: pandas DataFrame to rename
        email_col: Name of email column (will not be renamed)
        source_index: Index to identify the source file (1-based)
    
    Returns:
        dict: Mapping of old column names to new column names
    """
    new_cols = {}
    for col in df.columns:
        if col != email_col:
            new_cols[col] = f"{col}_problem{source_index}"
    
    return new_cols


def merge_excel_files(file_paths, output_path='data/merged_human_evaluation.xlsx'):
    """
    Merge multiple Excel files into a single file based on email addresses.
    
    Args:
        file_paths: List of paths to Excel files to merge
        output_path: Path for the output merged file
    
    Returns:
        pandas.DataFrame: Merged DataFrame
    """
    print("="*80)
    print("Merging Excel Files for Human Evaluation")
    print("="*80 + "\n")
    
    # Load all files
    dfs = []
    for i, file_path in enumerate(file_paths, 1):
        print(f"\nReading file {i}/{len(file_paths)}: {file_path}")
        df = pd.read_excel(file_path)
        print(f"  Columns: {df.columns.tolist()}")
        print(f"  Shape: {df.shape}")
        print(f"  Preview:\n{df.head()}\n")
        dfs.append(df)
    
    # Identify email column
    email_col = identify_email_column(dfs[0])
    
    if email_col is None:
        print("\n【Error】Could not identify email column.")
        print("Available columns:")
        for i, col in enumerate(dfs[0].columns):
            print(f"  {i}: {col}")
        print("\nPlease manually specify the email column name in the script.")
        return None
    
    print(f"\n✓ Using email column: '{email_col}'")
    
    # Rename columns with source identifiers
    print("\nRenaming columns to include source identifiers...")
    for i, df in enumerate(dfs, 1):
        new_cols = rename_columns_with_source(df, email_col, i)
        df.rename(columns=new_cols, inplace=True)
        print(f"  File {i}: {len(new_cols)} columns renamed")
    
    # Merge all DataFrames
    print("\nMerging DataFrames...")
    merged_df = dfs[0]
    for i in range(1, len(dfs)):
        print(f"  Merging file {i+1}...")
        merged_df = pd.merge(merged_df, dfs[i], on=email_col, how='outer')
    
    print(f"\n✓ Merge completed!")
    print(f"  Final shape: {merged_df.shape}")
    print(f"  Total columns: {len(merged_df.columns)}")
    print(f"\nPreview of merged data:")
    print(merged_df.head())
    
    # Save results
    print(f"\nSaving merged data to: {output_path}")
    merged_df.to_excel(output_path, index=False)
    print("✓ File saved successfully!")
    
    return merged_df


def main():
    """Main function to merge Excel files."""
    
    # Define input files
    # Modify this list to match your actual file names
    files = [
        'Creativity_A.xlsx',
        'Creativity_B.xlsx',
        'Creativity_C.xlsx'
    ]
    
    # Check if files exist
    print("Checking file existence...")
    for file in files:
        if not Path(file).exists():
            print(f"  ⚠️ Warning: File not found: {file}")
        else:
            print(f"  ✓ Found: {file}")
    
    # Perform merge
    merged_df = merge_excel_files(files)
    
    if merged_df is not None:
        print("\n" + "="*80)
        print("Merge Operation Completed Successfully")
        print("="*80)


if __name__ == "__main__":
    main()
