#!/usr/bin/env python3
"""
qnn_train_parallel.py - Parallel QNN Training with Multiprocessing
=================================================================

This script trains multiple QNN architectures in parallel using multiprocessing,
similar to the approach in star_train_custom.py but adapted for line classification.
"""

import argparse
import os
import sys
import time
import multiprocessing as mp
from typing import Tuple, List, Dict, Any
import warnings
from collections import defaultdict

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

# Local imports
from line_qnns import get_all_qnn_models, generate_line_dataset
from qnn_visualization import create_comprehensive_analysis, save_results_summary

warnings.filterwarnings('ignore')

# ─── CLI Arguments ───
parser = argparse.ArgumentParser("Parallel QNN Training for Line Classification")
parser.add_argument("--trials", type=int, default=100, help="Number of trials per model")
parser.add_argument("--samples", type=int, default=64, help="Number of training samples per trial")
parser.add_argument("--max-iter", type=int, default=60, help="Maximum iterations for COBYLA optimizer")
parser.add_argument("--cpus", type=int, default=None, help="Number of CPUs (default: all available)")
parser.add_argument("--seed", type=int, default=4, help="Random seed for reproducibility")
parser.add_argument("--save-results", action="store_true", help="Save detailed results to files")
parser.add_argument("--output-dir", type=str, default="results", help="Output directory for results")
args = parser.parse_args()

# Resource configuration
N_CPUS = args.cpus or mp.cpu_count()


def train_single_trial(model_class, trial_idx: int, num_samples: int, max_iter: int, 
                      base_seed: int = None) -> Tuple[str, int, float, List[float]]:
    """
    Train a single QNN model for one trial.
    
    Args:
        model_class: QNN model class to instantiate
        trial_idx: Trial index for this run
        num_samples: Number of training samples
        max_iter: Maximum iterations for optimizer
        base_seed: Base random seed
    
    Returns:
        Tuple of (model_name, trial_idx, accuracy, training_history)
    """
    # Set seeds for reproducibility
    if base_seed is not None:
        np.random.seed(base_seed + trial_idx)
    
    # Generate training and test data
    X_train, y_train = generate_line_dataset(num_samples, seed=base_seed + trial_idx if base_seed else None)
    X_test, y_test = generate_line_dataset(num_samples, seed=base_seed + trial_idx + 10000 if base_seed else None)
    
    # Create and train model
    model = model_class(max_iter=max_iter)
    
    try:
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        training_history = model.training_history
        
        return model.model_name, trial_idx, accuracy, training_history
        
    except Exception as e:
        print(f"Error in trial {trial_idx} for {model_class.__name__}: {e}")
        return model.model_name, trial_idx, 0.0, []


def train_model_trials(model_class, num_trials: int, num_samples: int, max_iter: int, 
                      base_seed: int = None) -> Tuple[str, List[float], List[List[float]]]:
    """
    Train a single model type across multiple trials in parallel.
    
    Args:
        model_class: QNN model class to train
        num_trials: Number of trials to run
        num_samples: Number of training samples per trial
        max_iter: Maximum iterations for optimizer
        base_seed: Base random seed
    
    Returns:
        Tuple of (model_name, accuracies, training_histories)
    """
    model_name = model_class(max_iter=max_iter).model_name
    print(f"🔄 Training {model_name} across {num_trials} trials...")
    
    # Parallel execution of trials
    if N_CPUS == 1:
        # Serial execution for debugging
        results = []
        for trial_idx in tqdm(range(num_trials), desc=f"{model_name}", leave=False):
            result = train_single_trial(model_class, trial_idx, num_samples, max_iter, base_seed)
            results.append(result)
    else:
        # Parallel execution
        results = Parallel(n_jobs=N_CPUS, backend="multiprocessing")(
            delayed(train_single_trial)(model_class, trial_idx, num_samples, max_iter, base_seed)
            for trial_idx in tqdm(range(num_trials), desc=f"{model_name}", leave=False)
        )
    
    # Extract results
    accuracies = []
    training_histories = []
    
    for model_name_result, trial_idx, accuracy, history in results:
        accuracies.append(accuracy)
        training_histories.append(history)
    
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    print(f"✅ {model_name} Complete - Avg Accuracy: {mean_acc:.3f} ± {std_acc:.3f}")
    
    return model_name, accuracies, training_histories


def main():
    print(f"🚀 Parallel QNN Training for Line Classification")
    print(f"📊 Running {args.trials} trials per model")
    print(f"📈 Dataset: {args.samples} samples per trial")
    print(f"⚙️  Max iterations: {args.max_iter}")
    print(f"💻 CPUs: {N_CPUS}")
    if args.seed:
        print(f"🎲 Random seed: {args.seed}")
    print("=" * 60)
    
    # Get all model classes
    model_classes = [
        lambda max_iter=args.max_iter: __import__('line_qnns').AngleEmbeddingQNN(max_iter),
        lambda max_iter=args.max_iter: __import__('line_qnns').AmplitudeEmbeddingQNN(max_iter),
        lambda max_iter=args.max_iter: __import__('line_qnns').DefaultQNN(max_iter),
        lambda max_iter=args.max_iter: __import__('line_qnns').CustomAngleQNN(max_iter)
    ]
    
    # Convert to actual classes for parallel training
    from line_qnns import AngleEmbeddingQNN, AmplitudeEmbeddingQNN, DefaultQNN, CustomAngleQNN
    model_classes = [AngleEmbeddingQNN, AmplitudeEmbeddingQNN, DefaultQNN, CustomAngleQNN]
    
    # Train all models
    start_time = time.time()
    all_results = {}
    all_training_histories = {}
    
    for model_class in model_classes:
        model_name, accuracies, training_histories = train_model_trials(
            model_class, args.trials, args.samples, args.max_iter, args.seed
        )
        all_results[model_name] = accuracies
        all_training_histories[model_name] = training_histories
    
    training_time = time.time() - start_time
    print(f"\n⏱️  Total training time: {training_time:.1f} seconds")
    print(f"📊 Total model runs: {args.trials * len(model_classes)}")
    
    # ═══════════════════════════════════════════════════════════════════
    # ANALYSIS AND VISUALIZATION
    # ═══════════════════════════════════════════════════════════════════
    
    print("\n" + "=" * 60)
    print("📈 CREATING COMPREHENSIVE ANALYSIS")
    print("=" * 60)
    
    # Create output directory
    if args.save_results:
        os.makedirs(args.output_dir, exist_ok=True)
        plots_dir = os.path.join(args.output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
    else:
        plots_dir = "plots"
    
    # Generate comprehensive analysis
    create_comprehensive_analysis(
        all_results, 
        args.trials, 
        args.samples, 
        save_dir=plots_dir
    )
    
    # Save results if requested
    if args.save_results:
        # Save detailed results
        results_file = os.path.join(args.output_dir, "qnn_comparison_results.txt")
        save_results_summary(
            all_results, 
            results_file,
            args.trials, 
            args.samples, 
            args.max_iter
        )
        
        # Save raw data
        raw_data_file = os.path.join(args.output_dir, "raw_results.npz")
        np.savez(
            raw_data_file,
            results=all_results,
            training_histories=all_training_histories,
            config={
                'trials': args.trials,
                'samples': args.samples,
                'max_iter': args.max_iter,
                'seed': args.seed,
                'training_time': training_time
            }
        )
        print(f"Raw data saved to {raw_data_file}")
        
        # Save configuration
        config_file = os.path.join(args.output_dir, "config.txt")
        with open(config_file, 'w') as f:
            f.write("QNN Parallel Training Configuration\n")
            f.write("=" * 40 + "\n")
            f.write(f"Trials per model: {args.trials}\n")
            f.write(f"Samples per trial: {args.samples}\n")
            f.write(f"Max iterations: {args.max_iter}\n")
            f.write(f"CPUs used: {N_CPUS}\n")
            f.write(f"Random seed: {args.seed}\n")
            f.write(f"Training time: {training_time:.1f} seconds\n")
            f.write(f"Total model runs: {args.trials * len(model_classes)}\n")
        print(f"Configuration saved to {config_file}")
    
    print(f"\n✨ Study completed! Analyzed {args.trials * len(model_classes)} total model runs.")


if __name__ == "__main__":
    main() 