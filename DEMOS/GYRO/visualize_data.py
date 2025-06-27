#!/usr/bin/env python3
"""
HTRU_2 Dataset Class Separability Visualization
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# Configuration
SAMPLE_SIZE = 5000  # For faster visualization (None for all data)

def load_data():
    """Load the HTRU_2 dataset."""
    print("📊 Loading HTRU_2 dataset...")
    df = pd.read_csv('HTRU_2.csv', header=None)
    
    # Column names
    feature_names = [
        'Mean_Profile', 'Std_Profile', 'Kurtosis_Profile', 'Skewness_Profile',
        'Mean_DM_SNR', 'Std_DM_SNR', 'Kurtosis_DM_SNR', 'Skewness_DM_SNR'
    ]
    
    df.columns = feature_names + ['Class']
    
    # Sample data if specified
    if SAMPLE_SIZE and len(df) > SAMPLE_SIZE:
        df = df.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)
        print(f"📊 Using {len(df)} samples for visualization")
    else:
        print(f"📊 Using all {len(df)} samples")
    
    return df

def create_class_separability_plot(df):
    """Create a single comprehensive plot showing class separability."""
    feature_cols = df.columns[:-1]
    class_counts = df['Class'].value_counts()
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 12))
    axes = axes.flatten()
    
    # Calculate separability metrics for sorting
    separability_scores = []
    for feature in feature_cols:
        pulsar_data = df[df['Class'] == 1][feature]
        non_pulsar_data = df[df['Class'] == 0][feature]
        
        # Use distance between means normalized by pooled standard deviation
        mean_diff = abs(pulsar_data.mean() - non_pulsar_data.mean())
        pooled_std = np.sqrt((pulsar_data.var() + non_pulsar_data.var()) / 2)
        separability = mean_diff / pooled_std if pooled_std > 0 else 0
        separability_scores.append((feature, separability))
    
    # Sort features by separability
    separability_scores.sort(key=lambda x: x[1], reverse=True)
    
    for i, (feature, sep_score) in enumerate(separability_scores):
        pulsar_data = df[df['Class'] == 1][feature]
        non_pulsar_data = df[df['Class'] == 0][feature]
        
        # Create overlaid histograms
        axes[i].hist(non_pulsar_data, bins=40, alpha=0.7, label='Non-Pulsar', 
                    color='lightcoral', density=True)
        axes[i].hist(pulsar_data, bins=40, alpha=0.7, label='Pulsar', 
                    color='skyblue', density=True)
        
        # Add statistics text
        mean_diff = abs(pulsar_data.mean() - non_pulsar_data.mean())
        overlap_score = sep_score
        
        axes[i].set_title(f'{feature.replace("_", " ")}\nSeparability: {overlap_score:.2f}', 
                         fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Density')
        axes[i].legend(fontsize=10)
        axes[i].grid(True, alpha=0.3)
        
        # Add separability ranking
        rank_color = 'green' if i < 3 else 'orange' if i < 6 else 'red'
        axes[i].text(0.02, 0.98, f'Rank: #{i+1}', 
                    transform=axes[i].transAxes, 
                    bbox=dict(boxstyle='round', facecolor=rank_color, alpha=0.7),
                    verticalalignment='top', fontweight='bold', color='white')
    
    # Add overall title and info
    plt.suptitle(f'HTRU_2 Feature Class Separability Analysis\n' + 
                f'Dataset: {len(df):,} samples ({class_counts[0]:,} Non-Pulsar, {class_counts[1]:,} Pulsar)', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig('plots/class_separability.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Print separability ranking
    print("\n🔍 Feature Separability Ranking:")
    for i, (feature, score) in enumerate(separability_scores, 1):
        print(f"  {i}. {feature.replace('_', ' ')}: {score:.3f}")

def main():
    """Main visualization function."""
    os.makedirs("plots", exist_ok=True)
    
    print("🚀 HTRU_2 Class Separability Analysis")
    print("=" * 50)
    
    df = load_data()
    
    print("\n📊 Creating class separability visualization...")
    create_class_separability_plot(df)
    
    print(f"\n✅ Analysis complete!")
    print(f"📊 Plot saved to: plots/class_separability.png")

if __name__ == "__main__":
    main() 