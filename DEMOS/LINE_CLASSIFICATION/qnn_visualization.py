#!/usr/bin/env python3
"""
qnn_visualization.py - Visualization Tools for QNN Results
=========================================================

This module provides visualization functions for analyzing QNN model performance
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
    Create individual histogram plots for each model's accuracy distribution.
    
    Args:
        accuracies: List of accuracy lists for each model
        model_names: Names of the models
        colors: Colors for each model's histogram
        num_trials: Number of trials conducted
        save_path: Path to save the plot (optional)
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, (ax, model, acc, color) in enumerate(zip(axes, model_names, accuracies, colors)):
        ax.hist(acc, bins=20, alpha=0.7, color=color, edgecolor='black')
        ax.set_title(f'{model}\nMean: {np.mean(acc):.3f} ± {np.std(acc):.3f}', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Test Accuracy')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
    
    plt.tight_layout()
    plt.suptitle(f'QNN Model Comparison - {num_trials} Trials Each', 
                fontsize=16, fontweight='bold', y=1.02)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Histogram plot saved to {save_path}")
    
    plt.show()


def plot_box_comparison(accuracies: List[List[float]], 
                       model_names: List[str], 
                       colors: List[str],
                       num_trials: int,
                       num_samples: int,
                       save_path: Optional[str] = None) -> None:
    """
    Create box plot comparison of all models.
    
    Args:
        accuracies: List of accuracy lists for each model
        model_names: Names of the models
        colors: Colors for each model's box plot
        num_trials: Number of trials conducted
        num_samples: Number of samples per trial
        save_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(12, 8))
    box_plot = plt.boxplot(accuracies, labels=model_names, patch_artist=True)
    
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    plt.title(f'QNN Model Performance Comparison\n{num_trials} Trials, {num_samples} Samples Each', 
              fontsize=14, fontweight='bold')
    plt.ylabel('Test Accuracy', fontsize=12)
    plt.xlabel('Model Architecture', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    
    # Add mean markers
    means = [np.mean(acc) for acc in accuracies]
    plt.scatter(range(1, len(means) + 1), means, color='red', s=100, zorder=5, label='Mean')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Box plot saved to {save_path}")
    
    plt.show()


def print_results_summary(accuracies: List[List[float]], 
                         model_names: List[str]) -> None:
    """
    Print detailed statistics summary for all models.
    
    Args:
        accuracies: List of accuracy lists for each model
        model_names: Names of the models
    """
    print("\n" + "=" * 60)
    print("📊 FINAL RESULTS SUMMARY")
    print("=" * 60)
    
    for model, acc in zip(model_names, accuracies):
        mean_acc = np.mean(acc)
        std_acc = np.std(acc)
        min_acc = np.min(acc)
        max_acc = np.max(acc)
        print(f"{model.replace(chr(10), ' '):<20}: {mean_acc:.3f} ± {std_acc:.3f} "
              f"(range: {min_acc:.3f} - {max_acc:.3f})")


def perform_statistical_analysis(accuracies: List[List[float]], 
                                model_names: List[str]) -> None:
    """
    Perform statistical significance testing between models.
    
    Args:
        accuracies: List of accuracy lists for each model
        model_names: Names of the models
    """
    try:
        from scipy import stats
        
        print("\n" + "=" * 60)
        print("📈 STATISTICAL ANALYSIS")
        print("=" * 60)
        
        # Perform pairwise t-tests
        print("Pairwise t-test p-values (significant if p < 0.05):")
        print("-" * 50)
        
        for i in range(len(accuracies)):
            for j in range(i + 1, len(accuracies)):
                t_stat, p_value = stats.ttest_ind(accuracies[i], accuracies[j])
                significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                print(f"{model_names[i]} vs {model_names[j]}: p = {p_value:.4f} {significance}")
                
    except ImportError:
        print("\n📈 Statistical analysis requires scipy (pip install scipy)")


def print_ranking(accuracies: List[List[float]], 
                 model_names: List[str]) -> None:
    """
    Print model ranking by mean accuracy.
    
    Args:
        accuracies: List of accuracy lists for each model
        model_names: Names of the models
    """
    means = [np.mean(acc) for acc in accuracies]
    
    print("\n🏆 RANKING (by mean accuracy):")
    print("-" * 30)
    ranking = sorted(zip(model_names, means, range(len(model_names))), key=lambda x: x[1], reverse=True)
    for rank, (model, mean_acc, idx) in enumerate(ranking, 1):
        print(f"{rank}. {model.replace(chr(10), ' '):<20}: {mean_acc:.3f}")


def create_comprehensive_analysis(results: Dict[str, List[float]], 
                                num_trials: int,
                                num_samples: int,
                                save_dir: str = "plots") -> None:
    """
    Create comprehensive analysis with all visualizations and statistics.
    
    Args:
        results: Dictionary mapping model names to accuracy lists
        num_trials: Number of trials conducted
        num_samples: Number of samples per trial
        save_dir: Directory to save plots
    """
    model_names = list(results.keys())
    accuracies = list(results.values())
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']
    
    # Ensure we have enough colors
    while len(colors) < len(model_names):
        colors.extend(['orange', 'purple', 'brown', 'pink'])
    colors = colors[:len(model_names)]
    
    # Create plots
    plot_individual_histograms(
        accuracies, model_names, colors, num_trials,
        save_path=os.path.join(save_dir, "qnn_comparison_histograms.png")
    )
    
    plot_box_comparison(
        accuracies, model_names, colors, num_trials, num_samples,
        save_path=os.path.join(save_dir, "qnn_comparison_boxplots.png")
    )
    
    # Print analysis
    print_results_summary(accuracies, model_names)
    perform_statistical_analysis(accuracies, model_names)
    print_ranking(accuracies, model_names)


def plot_training_history(training_histories: Dict[str, List[float]], 
                         save_path: Optional[str] = None) -> None:
    """
    Plot training loss/objective histories for all models.
    
    Args:
        training_histories: Dictionary mapping model names to training history lists
        save_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(12, 8))
    
    for model_name, history in training_histories.items():
        if history:  # Only plot if history exists
            plt.plot(history, label=model_name, linewidth=2)
    
    plt.xlabel('Iteration')
    plt.ylabel('Objective Value')
    plt.title('Training Progress Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    plt.show()


def save_results_summary(results: Dict[str, List[float]], 
                        save_path: str,
                        num_trials: int,
                        num_samples: int,
                        max_iter: int) -> None:
    """
    Save results summary to a text file.
    
    Args:
        results: Dictionary mapping model names to accuracy lists
        save_path: Path to save the summary file
        num_trials: Number of trials conducted
        num_samples: Number of samples per trial
        max_iter: Maximum iterations used for training
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        f.write(f"QNN Model Comparison Results\n")
        f.write(f"=" * 50 + "\n\n")
        f.write(f"Configuration:\n")
        f.write(f"- Trials per model: {num_trials}\n")
        f.write(f"- Samples per trial: {num_samples}\n")
        f.write(f"- Max iterations: {max_iter}\n\n")
        
        f.write(f"Results Summary:\n")
        f.write(f"-" * 30 + "\n")
        
        model_names = list(results.keys())
        accuracies = list(results.values())
        
        for model, acc in zip(model_names, accuracies):
            mean_acc = np.mean(acc)
            std_acc = np.std(acc)
            min_acc = np.min(acc)
            max_acc = np.max(acc)
            f.write(f"{model:<20}: {mean_acc:.3f} ± {std_acc:.3f} "
                   f"(range: {min_acc:.3f} - {max_acc:.3f})\n")
        
        # Ranking
        means = [np.mean(acc) for acc in accuracies]
        ranking = sorted(zip(model_names, means), key=lambda x: x[1], reverse=True)
        
        f.write(f"\nRanking by Mean Accuracy:\n")
        f.write(f"-" * 30 + "\n")
        for rank, (model, mean_acc) in enumerate(ranking, 1):
            f.write(f"{rank}. {model:<20}: {mean_acc:.3f}\n")
    
    print(f"Results summary saved to {save_path}")


if __name__ == "__main__":
    # Example usage with dummy data
    print("Testing visualization functions...")
    
    # Generate dummy results
    np.random.seed(42)
    dummy_results = {
        "Model 1": np.random.normal(0.75, 0.1, 100).tolist(),
        "Model 2": np.random.normal(0.70, 0.15, 100).tolist(),
        "Model 3": np.random.normal(0.80, 0.08, 100).tolist(),
        "Model 4": np.random.normal(0.78, 0.12, 100).tolist(),
    }
    
    # Clip to valid range
    for model in dummy_results:
        dummy_results[model] = np.clip(dummy_results[model], 0, 1).tolist()
    
    create_comprehensive_analysis(dummy_results, 100, 20) 