#!/usr/bin/env python3
"""
Comprehensive QNN Model Comparison Study for IRIS Dataset
=========================================================

This script compares 7 different Quantum Neural Network architectures
for classifying the IRIS dataset using the model classes from models.py.

NOTE: The models are trained on two different tasks:
- Models 1, 2, 5, 6 perform 3-class classification on a single flower.
- Models 3, 4, 7 perform binary classification on a *pair* of flowers
  to determine if they are from the same class.
Direct accuracy comparison should be interpreted with this in mind.
"""

import matplotlib.pyplot as plt
import numpy as np
from qiskit_algorithms.utils import algorithm_globals
from qiskit.primitives import Estimator, Sampler
from models import get_all_models, get_model_by_number

import warnings
import argparse
import os

warnings.filterwarnings('ignore')
os.makedirs("plots", exist_ok=True)


# Configuration
NUM_TRIALS = 1
MAX_ITER = 80
algorithm_globals.random_seed = 123
np.random.seed(algorithm_globals.random_seed)

# ============================================================================
# RESULTS ANALYSIS AND VISUALIZATION
# ============================================================================
def analyze_and_visualize_results(accuracies, models):
    print("\n" + "=" * 60)
    print("📊 FINAL RESULTS SUMMARY")
    print("=" * 60)

    colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold', 'violet', 'orange', 'pink']

    # Print statistics
    for i, (model, acc) in enumerate(zip(models, accuracies)):
        mean_acc = np.mean(acc)
        std_acc = np.std(acc)
        min_acc = np.min(acc)
        max_acc = np.max(acc)
        print(f"{model.replace(chr(10), ' '):<30}: {mean_acc:.3f} ± {std_acc:.3f} (range: {min_acc:.3f} - {max_acc:.3f})")

    # Create comprehensive visualization
    fig, axes = plt.subplots(3, 3, figsize=(22, 16))
    axes = axes.flatten()

    # Individual histograms
    for i, (ax, model, acc, color) in enumerate(zip(axes, models, accuracies, colors)):
        ax.hist(acc, bins=10, alpha=0.7, color=color, edgecolor='black')
        ax.set_title(f'{model}\nMean: {np.mean(acc):.3f} ± {np.std(acc):.3f}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Test Accuracy')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)

    # Hide the unused subplots
    for i in range(len(models), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.suptitle(f'IRIS QNN Model Comparison - {NUM_TRIALS} Trials Each', fontsize=16, fontweight='bold', y=1.02)
    plt.savefig('plots/iris_comparison_histograms2.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Box plot comparison
    plt.figure(figsize=(14, 8))
    box_plot = plt.boxplot(accuracies, labels=[m.replace('\n', ' ') for m in models], patch_artist=True)

    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    plt.title(f'IRIS QNN Model Performance Comparison\n({NUM_TRIALS} Trials)', 
              fontsize=14, fontweight='bold')
    plt.ylabel('Test Accuracy', fontsize=12)
    plt.xlabel('Model Architecture & Task', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    plt.xticks(rotation=15, ha='right')


    # Add mean markers
    means = [np.mean(acc) for acc in accuracies]
    plt.scatter(range(1, len(means) + 1), means, color='red', s=100, zorder=5, label='Mean')
    plt.legend()

    plt.tight_layout()
    plt.savefig('plots/iris_comparison_boxplots2.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Statistical significance testing
    try:
        from scipy import stats
        import pandas as pd
        import seaborn as sns
        
        print("\n" + "=" * 60)
        print("📈 STATISTICAL ANALYSIS (Pairwise t-test p-values)")
        
        model_names = [m.split('\n')[0] for m in models]
        p_values = pd.DataFrame(np.ones((len(models), len(models))), index=model_names, columns=model_names)
        annotations = pd.DataFrame(np.full((len(models), len(models)), "", dtype=object), index=model_names, columns=model_names)
        
        for i in range(len(accuracies)):
            for j in range(i + 1, len(accuracies)):
                t_stat, p_value = stats.ttest_ind(accuracies[i], accuracies[j])
                p_values.iloc[i, j] = p_value
                p_values.iloc[j, i] = p_value # Symmetric matrix
                
                significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                print(f"{model_names[i]:>18s} vs {model_names[j]:<18s}: p = {p_value:.4f} {significance}")
                
                annotations.iloc[i,j] = f"{p_value:.3f}\n{significance}"
                annotations.iloc[j,i] = f"{p_value:.3f}\n{significance}"


        # Create p-value heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(p_values, annot=annotations, fmt="s", cmap="coolwarm_r", linewidths=.5, vmin=0, vmax=0.1)
        plt.title('Pairwise t-test p-value Matrix', fontsize=16, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('plots/iris_comparison_p_values2.png', dpi=150, bbox_inches='tight')
        plt.close()

    except ImportError:
        print("\n📈 Statistical analysis requires scipy, pandas, and seaborn (pip install scipy pandas seaborn)")


def main():
    parser = argparse.ArgumentParser(description="Run QNN model comparison for the IRIS dataset.")
    parser.add_argument("--model", type=int, choices=range(1, 8), help="Specify a single model number to run (1-7). If not provided, all models are run.")
    args = parser.parse_args()

    print("🚀 Starting IRIS QNN Model Comparison Study")
    sampler = Sampler()
    estimator = Estimator()

    if args.model:
        print(f"🎯 Running only Model {args.model}")
        print("=" * 60)
        
        # Run a single specified model
        model = get_model_by_number(args.model, sampler, estimator)
        accuracies, model_name = model.run_trials(NUM_TRIALS, MAX_ITER)
        
        # Simplified output for a single model
        print("\n" + "=" * 60)
        print(f"📊 FINAL RESULTS SUMMARY FOR {model_name.split(chr(10))[0]}")
        print("=" * 60)
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        min_acc = np.min(accuracies)
        max_acc = np.max(accuracies)
        print(f"{model_name.replace(chr(10), ' '):<30}: {mean_acc:.3f} ± {std_acc:.3f} (range: {min_acc:.3f} - {max_acc:.3f})")

        # Create histogram for the single model
        plt.figure(figsize=(10, 6))
        plt.hist(accuracies, bins=12, alpha=0.75, color='orange', edgecolor='black')
        plt.title(f'{model_name.replace(chr(10), " ")} Accuracy Distribution ({NUM_TRIALS} Trials)', fontsize=16, fontweight='bold')
        plt.xlabel('Test Accuracy', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.axvline(mean_acc, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_acc:.3f}')
        plt.legend()
        plt.grid(True, alpha=0.4)
        plt.xlim(0, 1)
        plot_filename = f'DEMOS/IRIS/plots/model{args.model}_histogram.png'
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        print(f"\n📊 Saved accuracy histogram to {plot_filename}")

    else:
        # Run all models and perform comparison
        models = get_all_models(sampler, estimator)
        print(f"📊 Running all {len(models)} models, with {NUM_TRIALS} trials per model")
        print("=" * 60)
        
        all_accuracies = []
        all_model_names = []
        for model in models:
            accuracies, model_name = model.run_trials(NUM_TRIALS, MAX_ITER)
            all_accuracies.append(accuracies)
            all_model_names.append(model_name)
        
        analyze_and_visualize_results(all_accuracies, all_model_names)

    print(f"\n✨ Study completed!")


if __name__ == "__main__":
    main()