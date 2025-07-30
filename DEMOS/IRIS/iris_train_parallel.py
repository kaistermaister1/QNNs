#!/usr/bin/env python3
"""
iris_train_parallel.py - Parallel IRIS QNN Training with Multiprocessing
=======================================================================

This script trains multiple IRIS QNN architectures in parallel using multiprocessing,
similar to the approach in star_train_custom.py but adapted for IRIS classification.
"""

import argparse
import os
import sys
import time
import multiprocessing as mp
from typing import Tuple, List, Dict, Any
import warnings

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

# Qiskit imports
from qiskit.primitives import Sampler, Estimator

# Local imports
from models import get_all_models
from iris_visualization import create_comprehensive_analysis, save_results_summary, plot_model_comparison_by_type

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION - MODIFY THESE PARAMETERS AT THE TOP
# ═══════════════════════════════════════════════════════════════════

# Training Configuration
DEFAULT_TRIALS = 50         # Number of trials per model  
DEFAULT_MAX_ITER = 100      # Maximum optimizer iterations (controls learning)
DEFAULT_TRAIN_SIZE = 0.8    # Train/test split ratio (future use)

# ═══════════════════════════════════════════════════════════════════

# ─── CLI Arguments ───
parser = argparse.ArgumentParser("Parallel IRIS QNN Training")
parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS, 
                   help=f"Number of trials per model (default: {DEFAULT_TRIALS})")
parser.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER, 
                   help=f"Maximum optimizer iterations (default: {DEFAULT_MAX_ITER})")
parser.add_argument("--cpus", type=int, default=None, 
                   help="Number of CPUs (default: all available)")
parser.add_argument("--seed", type=int, default=4, 
                   help="Random seed for reproducibility")
parser.add_argument("--save-results", action="store_true", 
                   help="Save detailed results to files")
parser.add_argument("--output-dir", type=str, default="iris_results", 
                   help="Output directory for results")
parser.add_argument("--model-subset", type=str, default=None,
                   help="Comma-separated model numbers to run (e.g., '1,3,5' for models 1, 3, 5)")
args = parser.parse_args()

# Resource configuration
N_CPUS = args.cpus or mp.cpu_count()


def train_single_model(model_class, model_args, num_trials: int, max_iter: int, 
                      model_index: int) -> Tuple[str, List[float]]:
    """
    Train a single IRIS QNN model.
    
    Args:
        model_class: Model class to instantiate
        model_args: Arguments for model constructor (sampler, estimator)
        num_trials: Number of trials to run
        max_iter: Maximum iterations for optimizer
        model_index: Index of the model (for progress tracking)
    
    Returns:
        Tuple of (model_name, accuracies)
    """
    try:
        # Create fresh sampler and estimator for this process
        sampler, estimator = model_args
        
        # Instantiate the model
        model = model_class(sampler, estimator)
        
        print(f"🔄 Starting {model.name} (Model {model_index + 1})")
        
        # Run the model's trials
        accuracies, model_display_name = model.run_trials(num_trials, max_iter)
        
        return model_display_name, accuracies
        
    except Exception as e:
        print(f"❌ Error in model {model_index + 1}: {e}")
        return f"Model {model_index + 1} (Error)", [0.0] * num_trials


def create_fresh_primitives():
    """Create fresh Qiskit primitives for each worker process."""
    return Sampler(), Estimator()


def main():
    print("🌸 Parallel IRIS QNN Training")
    print("=" * 60)
    print(f"📊 Configuration:")
    print(f"   • Trials per model: {args.trials}")
    print(f"   • Max iterations: {args.max_iter}")
    print(f"   • CPUs: {N_CPUS}")
    if args.seed:
        print(f"   • Random seed: {args.seed}")
        np.random.seed(args.seed)
    print("=" * 60)
    
    # Create primitives for model instantiation
    sampler, estimator = create_fresh_primitives()
    
    # Get all models
    all_models = get_all_models(sampler, estimator)
    
    # Filter models if subset specified
    if args.model_subset:
        try:
            model_indices = [int(x.strip()) - 1 for x in args.model_subset.split(',')]
            selected_models = [all_models[i] for i in model_indices if 0 <= i < len(all_models)]
            if not selected_models:
                print("❌ No valid models selected. Using all models.")
                selected_models = all_models
            else:
                print(f"🎯 Running subset: {len(selected_models)} models")
        except (ValueError, IndexError):
            print("❌ Invalid model subset format. Using all models.")
            selected_models = all_models
    else:
        selected_models = all_models
    
    print(f"🔬 Training {len(selected_models)} models:")
    for i, model in enumerate(selected_models, 1):
        print(f"   {i}. {model.name}: {model.description}")
    
    # Prepare model classes and arguments for parallel execution
    model_tasks = []
    for i, model in enumerate(selected_models):
        model_class = type(model)
        model_args = create_fresh_primitives()  # Fresh primitives for each model
        model_tasks.append((model_class, model_args, args.trials, args.max_iter, i))
    
    # ═══════════════════════════════════════════════════════════════════
    # PARALLEL TRAINING EXECUTION
    # ═══════════════════════════════════════════════════════════════════
    
    print(f"\n🚀 Starting parallel training...")
    start_time = time.time()
    
    if N_CPUS == 1:
        # Serial execution for debugging
        print("🔧 Running in serial mode (1 CPU)")
        results = []
        for model_class, model_args, num_trials, max_iter, model_idx in model_tasks:
            result = train_single_model(model_class, model_args, num_trials, max_iter, model_idx)
            results.append(result)
    else:
        # Parallel execution
        print(f"⚡ Running in parallel mode ({N_CPUS} CPUs)")
        results = Parallel(n_jobs=N_CPUS, backend="multiprocessing")(
            delayed(train_single_model)(model_class, model_args, num_trials, max_iter, model_idx)
            for model_class, model_args, num_trials, max_iter, model_idx in model_tasks
        )
    
    training_time = time.time() - start_time
    
    # ═══════════════════════════════════════════════════════════════════
    # RESULTS PROCESSING
    # ═══════════════════════════════════════════════════════════════════
    
    print(f"\n⏱️  Total training time: {training_time:.1f} seconds")
    print(f"📊 Total model runs: {args.trials * len(selected_models)}")
    
    # Process results
    all_results = {}
    for model_name, accuracies in results:
        all_results[model_name] = accuracies
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        print(f"✅ {model_name}: {mean_acc:.3f} ± {std_acc:.3f}")
    
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
        save_dir=plots_dir
    )
    
    # Create specialized model type comparison
    print("\n📊 Creating model type comparison...")
    plot_model_comparison_by_type(
        all_results,
        save_path=os.path.join(plots_dir, "iris_model_type_comparison.png") if args.save_results else None
    )
    
    # Save results if requested
    if args.save_results:
        # Save detailed results
        results_file = os.path.join(args.output_dir, "iris_comparison_results.txt")
        save_results_summary(
            all_results, 
            results_file,
            args.trials, 
            args.max_iter
        )
        
        # Save raw data
        raw_data_file = os.path.join(args.output_dir, "raw_iris_results.npz")
        np.savez(
            raw_data_file,
            results=all_results,
            config={
                'trials': args.trials,
                'max_iter': args.max_iter,
                'seed': args.seed,
                'training_time': training_time,
                'models_run': len(selected_models)
            }
        )
        print(f"Raw data saved to {raw_data_file}")
        
        # Save configuration
        config_file = os.path.join(args.output_dir, "iris_config.txt")
        with open(config_file, 'w') as f:
            f.write("IRIS QNN Parallel Training Configuration\n")
            f.write("=" * 45 + "\n")
            f.write(f"Dataset: IRIS (3-class classification)\n")
            f.write(f"Trials per model: {args.trials}\n")
            f.write(f"Max iterations: {args.max_iter}\n")
            f.write(f"CPUs used: {N_CPUS}\n")
            f.write(f"Random seed: {args.seed}\n")
            f.write(f"Training time: {training_time:.1f} seconds\n")
            f.write(f"Models trained: {len(selected_models)}\n")
            f.write(f"Total evaluations: {args.trials * len(selected_models)}\n")
            if args.model_subset:
                f.write(f"Model subset: {args.model_subset}\n")
        print(f"Configuration saved to {config_file}")
        
        print(f"\n💾 All results saved to {args.output_dir}/")
        print(f"   • Plots: {plots_dir}")
        print(f"   • Summary: {results_file}")
        print(f"   • Raw data: {raw_data_file}")
        print(f"   • Config: {config_file}")
    
    # Final summary
    print(f"\n✨ IRIS study completed!")
    print(f"📈 Best performing model: {max(all_results.keys(), key=lambda k: np.mean(all_results[k]))}")
    best_accuracy = max(np.mean(accs) for accs in all_results.values())
    print(f"🏆 Best mean accuracy: {best_accuracy:.3f}")


if __name__ == "__main__":
    main() 