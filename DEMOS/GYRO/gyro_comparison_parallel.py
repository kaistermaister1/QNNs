import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid tkinter issues
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp
import argparse
import time
from functools import partial
from tqdm import tqdm
import os

# Import everything from the original file
from gyro_comparison import (
    load_and_prep_data, get_optimizer, 
    plot_iteration_progress, analyze_and_visualize_results,
    NUM_FEATURES, NUM_TRAIN_SAMPLES, OPTIMIZER, MAX_ITER
)

# Import Qiskit components
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap, EfficientSU2
from qiskit_algorithms.optimizers import COBYLA, SLSQP
from qiskit.primitives import Estimator, Sampler
from qiskit_machine_learning.algorithms.classifiers import VQC

# Global variables for parallel training
iteration_count = 0
current_pbar = None

def loss_callback_parallel(weights):
    """Callback function for parallel training (simplified)."""
    global iteration_count
    iteration_count += 1
    return False

def get_optimizer_parallel(optimizer_name, max_iter):
    """Get optimizer object for parallel training."""
    if optimizer_name.upper() == 'SLSQP':
        return SLSQP(maxiter=max_iter, callback=loss_callback_parallel)
    elif optimizer_name.upper() == 'COBYLA':
        return COBYLA(maxiter=max_iter, callback=loss_callback_parallel)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}. Options: 'SLSQP', 'COBYLA'")

def train_single_trial(args):
    """Train a single trial for a specific model. This function runs in parallel."""
    model_type, trial_idx, train_features, test_features, train_labels, test_labels, num_features = args
    
    global iteration_count
    iteration_count = 0
    
    # Create fresh sampler and estimator for this process
    sampler = Sampler()
    estimator = Estimator()
    
    try:
        if model_type == 1:
            # Model 1: VQC with ZZFeatureMap + EfficientSU2
            model1_entanglement = 'circular'
            model1_loss = 'cross_entropy'
            
            num_qubits = train_features.shape[1]
            feature_map = ZZFeatureMap(feature_dimension=num_qubits, reps=2)
            ansatz = EfficientSU2(num_qubits=num_qubits, reps=2, entanglement=model1_entanglement)
            
            classifier = VQC(
                sampler=sampler,
                feature_map=feature_map,
                ansatz=ansatz,
                loss=model1_loss,
                optimizer=get_optimizer_parallel(OPTIMIZER, MAX_ITER),
            )
            
        elif model_type == 2:
            # Model 2: VQC with ZZFeatureMap + Custom 6W ansatz
            model2_loss = 'cross_entropy'
            
            num_qubits = train_features.shape[1]
            feature_map = ZZFeatureMap(feature_dimension=num_qubits, reps=2)
            
            # Create custom ansatz circuit
            ansatz = QuantumCircuit(num_qubits)
            weight_params = [Parameter(f'θ_{i}') for i in range(9)]
            
            # Apply RY-RZ-RX on each qubit
            ansatz.ry(weight_params[0], 0)
            ansatz.rz(weight_params[1], 0)
            ansatz.rx(weight_params[2], 0)
            ansatz.ry(weight_params[3], 1)
            ansatz.rz(weight_params[4], 1)
            ansatz.rx(weight_params[5], 1)
            ansatz.ry(weight_params[6], 2)
            ansatz.rz(weight_params[7], 2)
            ansatz.rx(weight_params[8], 2)
            
            classifier = VQC(
                sampler=sampler,
                feature_map=feature_map,
                ansatz=ansatz,
                loss=model2_loss,
                optimizer=get_optimizer_parallel(OPTIMIZER, MAX_ITER),
            )
            
        elif model_type == 3:
            # Model 3: VQC with ZZFeatureMap + Y gates + Custom 6W ansatz
            model3_loss = 'cross_entropy'
            
            num_qubits = train_features.shape[1]
            
            # Create enhanced feature map with ZZ encoding + Y rotations
            feature_map = QuantumCircuit(num_qubits)
            zz_feature_map = ZZFeatureMap(feature_dimension=num_qubits, reps=2)
            feature_map.compose(zz_feature_map, inplace=True)
            
            # Add parametric Y rotations using the same feature parameters
            zz_params = zz_feature_map.parameters
            for i, param in enumerate(list(zz_params)[:num_qubits]):
                feature_map.ry(param, i)
            
            # Create custom ansatz
            ansatz = QuantumCircuit(num_qubits)
            weight_params = [Parameter(f'θ_{i}') for i in range(9)]
            
            ansatz.ry(weight_params[0], 0)
            ansatz.rz(weight_params[1], 0)
            ansatz.rx(weight_params[2], 0)
            ansatz.ry(weight_params[3], 1)
            ansatz.rz(weight_params[4], 1)
            ansatz.rx(weight_params[5], 1)
            ansatz.ry(weight_params[6], 2)
            ansatz.rz(weight_params[7], 2)
            ansatz.rx(weight_params[8], 2)
            
            classifier = VQC(
                sampler=sampler,
                feature_map=feature_map,
                ansatz=ansatz,
                loss=model3_loss,
                optimizer=get_optimizer_parallel(OPTIMIZER, MAX_ITER),
            )
        
        # Train the classifier
        classifier.fit(train_features, train_labels)
        
        # Calculate accuracies
        train_accuracy = classifier.score(train_features, train_labels)
        test_accuracy = classifier.score(test_features, test_labels)
        
        return {
            'model_type': model_type,
            'trial_idx': trial_idx,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'iterations': iteration_count,
            'success': True
        }
        
    except Exception as e:
        return {
            'model_type': model_type,
            'trial_idx': trial_idx,
            'train_accuracy': 0.0,
            'test_accuracy': 0.0,
            'iterations': 0,
            'success': False,
            'error': str(e)
        }

def run_model_parallel(model_type, num_features=3, num_train_samples=None, num_trials=5, num_cpus=4):
    """Run a model with parallel training across multiple trials."""
    
    print(f"\n🚀 Starting Model {model_type} with {num_trials} trials on {num_cpus} CPUs")
    
    # Load data once for all trials
    train_features, test_features, train_labels, test_labels = load_and_prep_data(
        num_features=num_features, feature_selection_method='FS1', num_train_samples=num_train_samples
    )
    
    # Prepare arguments for parallel processing
    args_list = []
    for trial_idx in range(num_trials):
        args_list.append((
            model_type, trial_idx, train_features, test_features, 
            train_labels, test_labels, num_features
        ))
    
    # Run parallel training
    start_time = time.time()
    
    if num_cpus == 1:
        # Sequential execution for comparison
        results = []
        for args in tqdm(args_list, desc=f"Model {model_type} Sequential", ncols=80):
            results.append(train_single_trial(args))
    else:
        # Parallel execution
        with mp.Pool(processes=num_cpus) as pool:
            results = list(tqdm(
                pool.imap(train_single_trial, args_list),
                total=num_trials,
                desc=f"Model {model_type} Parallel ({num_cpus} CPUs)",
                ncols=80
            ))
    
    end_time = time.time()
    training_time = end_time - start_time
    
    # Process results
    accuracies = []
    iteration_histories = []
    successful_trials = 0
    
    for result in results:
        if result['success']:
            accuracies.append(result['test_accuracy'])
            iteration_histories.append(result['iterations'])
            successful_trials += 1
            print(f"✅ Trial {result['trial_idx']+1}: Train={result['train_accuracy']:.4f}, Test={result['test_accuracy']:.4f} ({result['iterations']} epochs)")
        else:
            print(f"❌ Trial {result['trial_idx']+1}: FAILED - {result['error']}")
    
    # Model name mapping
    model_names = {
        1: f'1. VQC ZZ+EfficientSU2 FS1-{num_features}F\n(circular entanglement, cross_entropy loss)',
        2: f'2. VQC ZZ+Custom6W FS1-{num_features}F\n(RY-RZ-RX per qubit, cross_entropy loss)',
        3: f'3. VQC ZZ+Y+Custom6W FS1-{num_features}F\n(ZZ+Y feature map, RY-RZ-RX ansatz, cross_entropy loss)'
    }
    
    print(f"⏱️  Total training time: {training_time:.2f}s ({training_time/num_trials:.2f}s per trial avg)")
    print(f"✅ Successful trials: {successful_trials}/{num_trials}")
    
    if accuracies:
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        print(f"📊 Results: {mean_acc:.3f} ± {std_acc:.3f}")
    
    return accuracies, iteration_histories, model_names[model_type]

def main():
    parser = argparse.ArgumentParser(description='HTRU_2 QNN Study - Parallel Training')
    parser.add_argument('--single-trial', action='store_true', 
                       help='Train only 1 model instead of the default number of trials')
    parser.add_argument('--model', type=int, choices=[1, 2, 3], 
                       help='Specify a single model number to run (1-3). If not provided, all models are run.')
    parser.add_argument('--num-cpus', type=int, default=mp.cpu_count()//2,
                       help=f'Number of CPU cores to use for parallel training (default: {mp.cpu_count()//2})')
    parser.add_argument('--num-trials', type=int, default=5,
                       help='Number of trials to run (default: 5)')
    args = parser.parse_args()
    
    # Determine number of trials
    num_trials = 1 if args.single_trial else args.num_trials
    
    # Ensure plots directory exists
    os.makedirs("plots", exist_ok=True)
    
    print("🚀 HTRU_2 QNN Study - PARALLEL TRAINING")
    samples_text = f"{NUM_TRAIN_SAMPLES} train samples" if NUM_TRAIN_SAMPLES else "80/20 split"
    if args.model:
        print(f"🎯 Running only Model {args.model}")
    else:
        print(f"📊 Running all models")
    print(f"🔍 FS1 ({NUM_FEATURES} features) • {samples_text} • {num_trials} trials")
    print(f"⚙️  Optimizer: {OPTIMIZER} • Max Iter: {MAX_ITER}")
    print(f"🖥️  CPUs: {args.num_cpus}/{mp.cpu_count()} available")
    if args.single_trial:
        print("🔄 Single trial mode enabled")
    print("=" * 60)
    
    start_total_time = time.time()
    
    if args.model:
        # Run a single specified model
        accuracies, iteration_histories, model_name = run_model_parallel(
            args.model, num_features=NUM_FEATURES, num_train_samples=NUM_TRAIN_SAMPLES, 
            num_trials=num_trials, num_cpus=args.num_cpus
        )
        
        if accuracies:
            # Display single model results
            mean_acc = np.mean(accuracies)
            std_acc = np.std(accuracies)
            min_acc = np.min(accuracies)
            max_acc = np.max(accuracies)
            print(f"\n✅ Final Results: {mean_acc:.3f} ± {std_acc:.3f} (range: {min_acc:.3f} - {max_acc:.3f})")

            # Create single model histogram
            plt.figure(figsize=(10, 6))
            plt.hist(accuracies, bins=12, alpha=0.75, color='orange', edgecolor='black')
            plt.title(f'{model_name.replace(chr(10), " ")} Accuracy Distribution ({num_trials} Trials)', fontsize=16, fontweight='bold')
            plt.xlabel('Test Accuracy', fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            plt.axvline(mean_acc, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_acc:.3f}')
            plt.legend()
            plt.grid(True, alpha=0.4)
            plt.xlim(0, 1)
            plot_filename = f'plots/model{args.model}_histogram_parallel.png'
            plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"📊 Saved accuracy histogram to {plot_filename}")
            
            # Create iteration progress plot for single model
            if iteration_histories:
                plot_iteration_progress([iteration_histories], [model_name])

    else:
        # Run all models and perform comparison
        all_accuracies = []
        all_iteration_histories = []
        all_model_names = []
        
        for model_type in [1, 2, 3]:
            accuracies, iteration_histories, model_name = run_model_parallel(
                model_type, num_features=NUM_FEATURES, num_train_samples=NUM_TRAIN_SAMPLES, 
                num_trials=num_trials, num_cpus=args.num_cpus
            )
            all_accuracies.append(accuracies)
            all_iteration_histories.append(iteration_histories)
            all_model_names.append(model_name)
        
        # Filter out empty results
        filtered_accuracies = [acc for acc in all_accuracies if acc]
        filtered_names = [name for i, name in enumerate(all_model_names) if all_accuracies[i]]
        filtered_iterations = [iters for i, iters in enumerate(all_iteration_histories) if all_accuracies[i]]
        
        if filtered_accuracies:
            analyze_and_visualize_results(filtered_accuracies, filtered_names, num_trials)
            
            # Create iteration progress comparison plot
            if filtered_iterations:
                plot_iteration_progress(filtered_iterations, filtered_names)
    
    end_total_time = time.time()
    total_time = end_total_time - start_total_time
    
    print(f"\n✨ Parallel study completed in {total_time:.2f}s!")
    if not args.single_trial and not args.model:
        sequential_estimate = total_time * args.num_cpus
        print(f"⚡ Estimated speedup: ~{sequential_estimate/total_time:.1f}x faster than sequential")

if __name__ == "__main__":
    # Set multiprocessing start method for compatibility
    mp.set_start_method('spawn', force=True)
    main()