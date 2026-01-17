# read a csv and turn into dataframe
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def read_csv_to_dataframe(file_path):
    return pd.read_csv(file_path)

def create_score_histogram(df, score_columns, bins=10):
    """
    Create histogram of scores (count chart)
    x-axis: scores
    y-axis: count
    """
    plt.figure(figsize=(15, 10))
    
    # Calculate number of rows and columns for subplots
    n_cols = min(3, len(score_columns))
    n_rows = (len(score_columns) + n_cols - 1) // n_cols
    
    for i, column in enumerate(score_columns, 1):
        plt.subplot(n_rows, n_cols, i)
        
        # Remove NaN values
        data = df[column].dropna()
        
        if len(data) > 0:
            # Create histogram
            counts, bin_edges, patches = plt.hist(data, bins=bins, alpha=0.7, edgecolor='black')
            
            # Set title and labels
            plt.title(f'{column}\n(Total: {len(data)})', fontsize=10)
            plt.xlabel('Score')
            plt.ylabel('Count')
            
            # Show statistics
            mean_val = data.mean()
            std_val = data.std()
            plt.axvline(mean_val, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_val:.2f}')
            plt.legend()
            
            # Display count on bars
            for j, count in enumerate(counts):
                if count > 0:
                    plt.text((bin_edges[j] + bin_edges[j+1])/2, count + 0.1, 
                            str(int(count)), ha='center', va='bottom', fontsize=8)
        else:
            plt.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=plt.gca().transAxes)
        
        # save figure
        if 'response' in column.lower():
            save_path = f'histogram_{column}.png'
            plt.savefig(save_path)
            print(f"Saved chart: {save_path}")
    
    plt.tight_layout()
    plt.show()
    return plt

# Read CSV file
file_path = '../eval_persona_eval/Qwen2.5-7B-Instruct/creative_professional.csv'
content = read_csv_to_dataframe(file_path)

print("CSV File Information:")
print(f"Number of rows: {len(content)}")
print(f"Number of columns: {len(content.columns)}")
print("\nColumn list:")
for i, col in enumerate(content.columns):
    print(f"{i+1}. {col}")

# Identify score columns (usually numeric types)
numeric_columns = content.select_dtypes(include=[np.number]).columns.tolist()
print(f"\nNumeric columns: {numeric_columns}")


# Create count charts
if numeric_columns:
    print("\nCreating score count charts...")
    create_score_histogram(content, numeric_columns)
else:
    print("No numeric columns found")
