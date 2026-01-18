import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os  
  
def create_enhanced_comparison_chart():
    """Create enhanced comparison charts with detailed statistical analysis"""
    # Set global style
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    plt.style.use('default')
    
    # Read data
    baseline_df = pd.read_csv("./baseline.csv")
    steering_df = pd.read_csv("./Qwen2.5-7B-Instruct/creative_professional.csv")
    
    # Calculate statistics
    metrics = ['creative_professional', 'coherence']
    projection_col = "Qwen2.5-7B-Instruct_creative_professional_response_avg_diff_proj_layer20"
    
    # Create subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1])
    
    # Color configuration
    colors = {
        'baseline': '#2E86AB',
        'steering': '#A23B72', 
        'improvement': '#F18F01'
    }
    
    # 1. Score comparison bar chart
    ax1 = fig.add_subplot(gs[0, :2])
    
    conditions = ['Baseline', 'With Steering']
    creative_scores = [baseline_df['creative_professional'].mean(), 
                      steering_df['creative_professional'].mean()]
    coherence_scores = [baseline_df['coherence'].mean(), 
                       steering_df['coherence'].mean()]
    
    x = np.arange(len(conditions))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, creative_scores, width, label='Creative Professional', 
                    color=colors['baseline'], alpha=0.8, edgecolor='white', linewidth=1)
    bars2 = ax1.bar(x + width/2, coherence_scores, width, label='Coherence', 
                    color=colors['steering'], alpha=0.8, edgecolor='white', linewidth=1)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    for bar in bars2:
        height = bar.get_height()
        ax1.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax1.set_xlabel('Condition', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax1.set_title('Persona Steering Effect Comparison', fontsize=14, fontweight='bold', pad=20)
    ax1.set_xticks(x)
    ax1.set_xticklabels(conditions)
    ax1.legend(frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim(0, max(max(creative_scores), max(coherence_scores)) * 1.1)
    
    # 2. Statistics summary table
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('tight')
    ax2.axis('off')
    
    # Calculate improvement percentage
    creative_improvement = ((creative_scores[1] - creative_scores[0]) / creative_scores[0] * 100)
    coherence_improvement = ((coherence_scores[1] - coherence_scores[0]) / coherence_scores[0] * 100)
    
    table_data = [
        ['Metric', 'Baseline', 'Steering', 'Improvement'],
        ['Creative', f'{creative_scores[0]:.3f}', f'{creative_scores[1]:.3f}', f'{creative_improvement:+.1f}%'],
        ['Coherence', f'{coherence_scores[0]:.3f}', f'{coherence_scores[1]:.3f}', f'{coherence_improvement:+.1f}%']
    ]
    
    table = ax2.table(cellText=table_data[1:], colLabels=table_data[0],
                     cellLoc='center', loc='center',
                     colWidths=[0.25, 0.25, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Header colors
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax2.set_title('Statistical Summary', fontsize=12, fontweight='bold', pad=20)
    
    # 3. Projection value distribution comparison (histogram)
    ax3 = fig.add_subplot(gs[1, :])
    
    baseline_proj = baseline_df[projection_col].dropna()
    steering_proj = steering_df[projection_col].dropna()
    
    # Calculate appropriate bins
    all_data = np.concatenate([baseline_proj, steering_proj])
    bins = np.linspace(all_data.min(), all_data.max(), 30)
    
    ax3.hist(baseline_proj, bins=bins, alpha=0.6, label=f'Baseline (n={len(baseline_proj)})', 
             color=colors['baseline'], density=True, edgecolor='white')
    ax3.hist(steering_proj, bins=bins, alpha=0.6, label=f'Steering (n={len(steering_proj)})', 
             color=colors['steering'], density=True, edgecolor='white')
    
    # Add mean lines
    ax3.axvline(baseline_proj.mean(), color=colors['baseline'], linestyle='--', linewidth=2, 
                label=f'Baseline Mean: {baseline_proj.mean():.3f}')
    ax3.axvline(steering_proj.mean(), color=colors['steering'], linestyle='--', linewidth=2,
                label=f'Steering Mean: {steering_proj.mean():.3f}')
    
    ax3.set_xlabel('Projection Value', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax3.set_title('Projection Value Distribution Comparison', fontsize=14, fontweight='bold', pad=20)
    ax3.legend(frameon=True, fancybox=True, shadow=True)
    ax3.grid(True, alpha=0.3, linestyle='--')
    
    # 4. Box plot comparison
    ax4 = fig.add_subplot(gs[2, 0])
    
    box_data = [baseline_proj, steering_proj]
    box_plot = ax4.boxplot(box_data, labels=['Baseline', 'Steering'], 
                          patch_artist=True, notch=True)
    
    box_plot['boxes'][0].set_facecolor(colors['baseline'])
    box_plot['boxes'][1].set_facecolor(colors['steering'])
    box_plot['boxes'][0].set_alpha(0.7)
    box_plot['boxes'][1].set_alpha(0.7)
    
    ax4.set_ylabel('Projection Value', fontsize=12, fontweight='bold')
    ax4.set_title('Projection Distribution Statistics', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, linestyle='--')
    
    # 5. Scatter plot - Creative vs Coherence
    ax5 = fig.add_subplot(gs[2, 1])
    
    ax5.scatter(baseline_df['creative_professional'], baseline_df['coherence'], 
               alpha=0.6, color=colors['baseline'], label='Baseline', s=30)
    ax5.scatter(steering_df['creative_professional'], steering_df['coherence'], 
               alpha=0.6, color=colors['steering'], label='Steering', s=30)
    
    ax5.set_xlabel('Creative Professional Score', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Coherence Score', fontsize=11, fontweight='bold')
    ax5.set_title('Creative vs Coherence', fontsize=12, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3, linestyle='--')
    
    # 6. Score difference analysis
    ax6 = fig.add_subplot(gs[2, 2])
    
    creative_diff = steering_df['creative_professional'].mean() - baseline_df['creative_professional'].mean()
    coherence_diff = steering_df['coherence'].mean() - baseline_df['coherence'].mean()
    projection_diff = steering_proj.mean() - baseline_proj.mean()
    
    metrics_names = ['Creative\nProfessional', 'Coherence', 'Projection\nValue']
    differences = [creative_diff, coherence_diff, projection_diff]
    colors_diff = [colors['improvement'] if d > 0 else colors['baseline'] for d in differences]
    
    bars = ax6.bar(metrics_names, differences, color=colors_diff, alpha=0.8, 
                   edgecolor='white', linewidth=1)
    
    # Add value labels
    for bar, diff in zip(bars, differences):
        height = bar.get_height()
        ax6.annotate(f'{diff:+.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3 if height > 0 else -15),
                    textcoords="offset points",
                    ha='center', va='bottom' if height > 0 else 'top', 
                    fontsize=10, fontweight='bold')
    
    ax6.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax6.set_ylabel('Improvement Magnitude', fontsize=11, fontweight='bold')
    ax6.set_title('Steering Effect Analysis', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout(pad=3.0)
    
    save_path = 'enhanced_comparison_analysis.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved enhanced analysis chart: {save_path}")
    plt.close()


def create_detailed_projection_analysis():
    """Create detailed projection analysis charts"""
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    baseline_df = pd.read_csv("./baseline.csv")
    steering_df = pd.read_csv("./Qwen2.5-7B-Instruct/creative_professional.csv")
    
    projection_col = "Qwen2.5-7B-Instruct_creative_professional_response_avg_diff_proj_layer20"
    
    baseline_proj = baseline_df[projection_col].dropna()
    steering_proj = steering_df[projection_col].dropna()
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    colors = {'baseline': '#2E86AB', 'steering': '#A23B72'}
    
    # 1. Density distribution using histogram
    bins = 30
    ax1.hist(baseline_proj, bins=bins, alpha=0.5, density=True, 
            color=colors['baseline'], label='Baseline', edgecolor='white')
    ax1.hist(steering_proj, bins=bins, alpha=0.5, density=True, 
            color=colors['steering'], label='Steering', edgecolor='white')
    
    ax1.set_title('Projection Value Density Distribution', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Projection Value', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Cumulative distribution function
    ax2.hist(baseline_proj, bins=50, cumulative=True, density=True, 
            alpha=0.7, color=colors['baseline'], label='Baseline')
    ax2.hist(steering_proj, bins=50, cumulative=True, density=True, 
            alpha=0.7, color=colors['steering'], label='Steering')
    
    ax2.set_title('Cumulative Distribution Function', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Projection Value', fontsize=12)
    ax2.set_ylabel('Cumulative Probability', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Quantile comparison
    percentiles = np.arange(10, 100, 10)
    baseline_quantiles = np.percentile(baseline_proj, percentiles)
    steering_quantiles = np.percentile(steering_proj, percentiles)
    
    ax3.plot(percentiles, baseline_quantiles, 'o-', color=colors['baseline'], 
            linewidth=2, markersize=6, label='Baseline')
    ax3.plot(percentiles, steering_quantiles, 's-', color=colors['steering'], 
            linewidth=2, markersize=6, label='Steering')
    
    ax3.set_title('Quantile Comparison', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Percentile', fontsize=12)
    ax3.set_ylabel('Projection Value', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Statistical information comparison
    ax4.axis('off')
    
    stats_data = [
        ['Statistical Metric', 'Baseline', 'Steering', 'Difference'],
        ['Mean', f'{baseline_proj.mean():.4f}', f'{steering_proj.mean():.4f}', 
         f'{steering_proj.mean() - baseline_proj.mean():+.4f}'],
        ['Median', f'{baseline_proj.median():.4f}', f'{steering_proj.median():.4f}', 
         f'{steering_proj.median() - baseline_proj.median():+.4f}'],
        ['Std Dev', f'{baseline_proj.std():.4f}', f'{steering_proj.std():.4f}', 
         f'{steering_proj.std() - baseline_proj.std():+.4f}'],
        ['Min Value', f'{baseline_proj.min():.4f}', f'{steering_proj.min():.4f}', 
         f'{steering_proj.min() - baseline_proj.min():+.4f}'],
        ['Max Value', f'{baseline_proj.max():.4f}', f'{steering_proj.max():.4f}', 
         f'{steering_proj.max() - baseline_proj.max():+.4f}'],
        ['Sample Size', f'{len(baseline_proj)}', f'{len(steering_proj)}', 
         f'{len(steering_proj) - len(baseline_proj):+d}']
    ]
    
    table = ax4.table(cellText=stats_data[1:], colLabels=stats_data[0],
                     cellLoc='center', loc='center',
                     colWidths=[0.25, 0.25, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    
    # Set table style
    for i in range(len(stats_data[0])):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax4.set_title('Detailed Statistical Comparison', fontsize=14, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    
    save_path = 'detailed_projection_analysis.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved detailed projection analysis chart: {save_path}")
    plt.close()


def plot_projection_distribution():  
    baseline_df = pd.read_csv("./baseline.csv")  
    steering_df = pd.read_csv("./Qwen2.5-7B-Instruct/creative_professional.csv")
      
    # Assume projection column name is projection_column  
    projection_col = "Qwen2.5-7B-Instruct_creative_professional_response_avg_diff_proj_layer20"  
      
    plt.figure(figsize=(10, 6))  
    plt.hist(baseline_df[projection_col], alpha=0.5, label='Baseline', bins=20)  
    plt.hist(steering_df[projection_col], alpha=0.5, label='With Steering', bins=20)  
    plt.xlabel('Projection Value')  
    plt.ylabel('Frequency')  
    plt.title('Projection Value Distribution Comparison')  
    plt.legend()  
    # plt.show()
    save_path = 'projection_distribution_comparison.png'
    plt.savefig(save_path)
    print(f"Saved chart: {save_path}")

if __name__ == "__main__":
    create_enhanced_comparison_chart()
    create_detailed_projection_analysis() 
    plot_projection_distribution()