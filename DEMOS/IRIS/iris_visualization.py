#!/usr/bin/env python3
"""
iris_visualization.py - Visualization Tools for IRIS QNN Results
===============================================================

This module provides visualization functions for analyzing IRIS QNN model performance
including histograms, box plots, and statistical analysis.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any, Optional
import os


def plot_individual_histograms(accuracies: List[List[float]], 
                              model_names: List[str], 
                              colors: List[str],
                              num_trials: int,
                              save_path: Optional[str] = None) -> None:
    """
    Create individual histogram plots for each IRIS model's accuracy distribution.
    
    Args:
        accuracies: List of accuracy lists for each model
        model_names: Names of the models
        colors: Colors for each model's histogram
        num_trials: Number of trials conducted
        save_path: Path to save the plot (optional)
    """
    # Determine subplot layout based on number of models
    n_models = len(model_names)
    if n_models <= 4:
        rows, cols = 2, 2
    elif n_models <= 6:
        rows, cols = 2, 3
    elif n_models <= 9:
        rows, cols = 3, 3
    else:
        rows, cols = 4, 3
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 12))
    axes = axes.flatten() if n_models > 1 else [axes]
    
    for i, (model, acc, color) in enumerate(zip(model_names, accuracies, colors)):
        if i < len(axes):
            axes[i].hist(acc, bins=20, alpha=0.7, color=color, edgecolor='black')
            axes[i].set_title(f'{model}\nMean: {np.mean(acc):.3f} ± {np.std(acc):.3f}', 
                            fontsize=10, fontweight='bold')
            axes[i].set_xlabel('Test Accuracy')
            axes[i].set_ylabel('Frequency')
            axes[i].grid(True, alpha=0.3)
            axes[i].set_xlim(0, 1)
    
    # Hide unused subplots
    for i in range(len(model_names), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.suptitle(f'IRIS QNN Model Comparison - {num_trials} Trials Each', 
                fontsize=16, fontweight='bold', y=1.02)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"IRIS histogram plot saved to {save_path}")
    
    plt.show()


def plot_box_comparison(accuracies: List[List[float]], 
                       model_names: List[str], 
                       colors: List[str],
                       num_trials: int,
                       save_path: Optional[str] = None) -> None:
    """
    Create box plot comparison of all IRIS models.
    
    Args:
        accuracies: List of accuracy lists for each model
        model_names: Names of the models
        colors: Colors for each model's box plot
        num_trials: Number of trials conducted
        save_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(15, 8))
    box_plot = plt.boxplot(accuracies, labels=model_names, patch_artist=True)
    
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    plt.title(f'IRIS QNN Model Performance Comparison\n{num_trials} Trials Each', 
              fontsize=14, fontweight='bold')
    plt.ylabel('Test Accuracy', fontsize=12)
    plt.xlabel('Model Architecture', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    
    # Add mean markers
    means = [np.mean(acc) for acc in accuracies]
    plt.scatter(range(1, len(means) + 1), means, color='red', s=100, zorder=5, label='Mean')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"IRIS box plot saved to {save_path}")
    
    plt.show()


def print_results_summary(accuracies: List[List[float]], 
                         model_names: List[str]) -> None:
    """
    Print detailed statistics summary for all IRIS models.
    
    Args:
        accuracies: List of accuracy lists for each model
        model_names: Names of the models
    """
    print("\n" + "=" * 70)
    print("🌸 IRIS QNN RESULTS SUMMARY")
    print("=" * 70)
    
    for model, acc in zip(model_names, accuracies):
        mean_acc = np.mean(acc)
        std_acc = np.std(acc)
        min_acc = np.min(acc)
        max_acc = np.max(acc)
        clean_name = model.replace('\n', ' ')
        print(f"{clean_name:<25}: {mean_acc:.3f} ± {std_acc:.3f} "
              f"(range: {min_acc:.3f} - {max_acc:.3f})")


def perform_statistical_analysis(accuracies: List[List[float]], 
                                model_names: List[str]) -> None:
    """
    Perform statistical significance testing between IRIS models.
    
    Args:
        accuracies: List of accuracy lists for each model
        model_names: Names of the models
    """
    try:
        from scipy import stats
        
        print("\n" + "=" * 70)
        print("📊 STATISTICAL ANALYSIS")
        print("=" * 70)
        
        # Perform pairwise t-tests
        print("Pairwise t-test p-values (significant if p < 0.05):")
        print("-" * 60)
        
        for i in range(len(accuracies)):
            for j in range(i + 1, len(accuracies)):
                t_stat, p_value = stats.ttest_ind(accuracies[i], accuracies[j])
                significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                clean_name_i = model_names[i].replace('\n', ' ')
                clean_name_j = model_names[j].replace('\n', ' ')
                print(f"{clean_name_i} vs {clean_name_j}: p = {p_value:.4f} {significance}")
                
    except ImportError:
        print("\n📊 Statistical analysis requires scipy (pip install scipy)")


def print_ranking(accuracies: List[List[float]], 
                 model_names: List[str]) -> None:
    """
    Print IRIS model ranking by mean accuracy.
    
    Args:
        accuracies: List of accuracy lists for each model
        model_names: Names of the models
    """
    means = [np.mean(acc) for acc in accuracies]
    
    print("\n🏆 RANKING (by mean accuracy):")
    print("-" * 40)
    ranking = sorted(zip(model_names, means, range(len(model_names))), key=lambda x: x[1], reverse=True)
    for rank, (model, mean_acc, idx) in enumerate(ranking, 1):
        clean_name = model.replace('\n', ' ')
        print(f"{rank}. {clean_name:<25}: {mean_acc:.3f}")


def create_comprehensive_analysis(results: Dict[str, List[float]], 
                                num_trials: int,
                                save_dir: str = "plots") -> None:
    """
    Create comprehensive analysis with all visualizations and statistics for IRIS models.
    
    Args:
        results: Dictionary mapping model names to accuracy lists
        num_trials: Number of trials conducted
        save_dir: Directory to save plots
    """
    model_names = list(results.keys())
    accuracies = list(results.values())
    
    # Color palette for IRIS models
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold', 'orange', 'purple', 'brown']
    
    # Ensure we have enough colors
    while len(colors) < len(model_names):
        colors.extend(['pink', 'gray', 'olive', 'cyan', 'magenta', 'yellow'])
    colors = colors[:len(model_names)]
    
    # Create plots
    plot_individual_histograms(
        accuracies, model_names, colors, num_trials,
        save_path=os.path.join(save_dir, "iris_comparison_histograms.png")
    )
    
    plot_box_comparison(
        accuracies, model_names, colors, num_trials,
        save_path=os.path.join(save_dir, "iris_comparison_boxplots.png")
    )
    
    # Print analysis
    print_results_summary(accuracies, model_names)
    perform_statistical_analysis(accuracies, model_names)
    print_ranking(accuracies, model_names)


def save_results_summary(results: Dict[str, List[float]], 
                        save_path: str,
                        num_trials: int,
                        max_iter: int) -> None:
    """
    Save IRIS results summary to a text file.
    
    Args:
        results: Dictionary mapping model names to accuracy lists
        save_path: Path to save the summary file
        num_trials: Number of trials conducted
        max_iter: Maximum iterations used for training
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        f.write(f"IRIS QNN Model Comparison Results\n")
        f.write(f"=" * 50 + "\n\n")
        f.write(f"Configuration:\n")
        f.write(f"- Trials per model: {num_trials}\n")
        f.write(f"- Max iterations: {max_iter}\n")
        f.write(f"- Dataset: IRIS (3-class classification)\n\n")
        
        f.write(f"Results Summary:\n")
        f.write(f"-" * 30 + "\n")
        
        model_names = list(results.keys())
        accuracies = list(results.values())
        
        for model, acc in zip(model_names, accuracies):
            mean_acc = np.mean(acc)
            std_acc = np.std(acc)
            min_acc = np.min(acc)
            max_acc = np.max(acc)
            clean_name = model.replace('\n', ' ')
            f.write(f"{clean_name:<25}: {mean_acc:.3f} ± {std_acc:.3f} "
                   f"(range: {min_acc:.3f} - {max_acc:.3f})\n")
        
        # Ranking
        means = [np.mean(acc) for acc in accuracies]
        ranking = sorted(zip(model_names, means), key=lambda x: x[1], reverse=True)
        
        f.write(f"\nRanking by Mean Accuracy:\n")
        f.write(f"-" * 30 + "\n")
        for rank, (model, mean_acc) in enumerate(ranking, 1):
            clean_name = model.replace('\n', ' ')
            f.write(f"{rank}. {clean_name:<25}: {mean_acc:.3f}\n")
    
    print(f"IRIS results summary saved to {save_path}")


def plot_model_comparison_by_type(results: Dict[str, List[float]], 
                                 save_path: Optional[str] = None) -> None:
    """
    Create a specialized plot comparing different types of IRIS models.
    
    Args:
        results: Dictionary mapping model names to accuracy lists
        save_path: Path to save the plot (optional)
    """
    # Categorize models
    vqc_models = {k: v for k, v in results.items() if 'VQC' in k}
    siamese_models = {k: v for k, v in results.items() if 'Siamese' in k}
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # VQC Models comparison
    if vqc_models:
        vqc_names = list(vqc_models.keys())
        vqc_accs = list(vqc_models.values())
        vqc_colors = ['skyblue', 'lightgreen', 'gold', 'orange'][:len(vqc_names)]
        
        box_plot1 = ax1.boxplot(vqc_accs, labels=vqc_names, patch_artist=True)
        for patch, color in zip(box_plot1['boxes'], vqc_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax1.set_title('VQC Models Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Test Accuracy')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
    
    # Siamese Models comparison
    if siamese_models:
        siamese_names = list(siamese_models.keys())
        siamese_accs = list(siamese_models.values())
        siamese_colors = ['lightcoral', 'purple', 'brown'][:len(siamese_names)]
        
        box_plot2 = ax2.boxplot(siamese_accs, labels=siamese_names, patch_artist=True)
        for patch, color in zip(box_plot2['boxes'], siamese_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax2.set_title('Siamese Models Comparison', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Test Accuracy')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"IRIS model type comparison saved to {save_path}")
    
    plt.show()


if __name__ == "__main__":
    # Example usage with dummy data
    print("Testing IRIS visualization functions...")
    
    # Generate dummy results for 7 models
    np.random.seed(42)
    dummy_results = {
        "1. VQC ZZ+RA 4F\n(3-Class)": np.random.normal(0.85, 0.08, 50).tolist(),
        "2. VQC ZZ+ESU2 2F\n(3-Class)": np.random.normal(0.82, 0.10, 50).tolist(),
        "3. Siamese 4F\n(Binary Pair)": np.random.normal(0.78, 0.12, 50).tolist(),
        "4. Siamese 2F\n(Binary Pair)": np.random.normal(0.75, 0.11, 50).tolist(),
        "5. VQC Custom 4F\n(3-Class)": np.random.normal(0.83, 0.09, 50).tolist(),
        "6. VQC Custom 2F\n(3-Class)": np.random.normal(0.80, 0.10, 50).tolist(),
        "7. 2Q Siamese 2F\n(Binary Pair)": np.random.normal(0.77, 0.13, 50).tolist(),
    }
    
    # Clip to valid range
    for model in dummy_results:
        dummy_results[model] = np.clip(dummy_results[model], 0, 1).tolist()
    
    create_comprehensive_analysis(dummy_results, 50) 