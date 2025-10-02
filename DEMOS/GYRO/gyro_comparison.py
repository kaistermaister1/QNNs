import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid tkinter issues
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap, EfficientSU2
from qiskit_algorithms.optimizers import COBYLA, SLSQP
from qiskit_algorithms.utils import algorithm_globals
from qiskit.primitives import Estimator, Sampler
from qiskit.quantum_info import SparsePauliOp

from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier, VQC
from qiskit_machine_learning.neural_networks import EstimatorQNN

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
import pandas as pd
from tqdm import tqdm
import argparse

import warnings
import os

warnings.filterwarnings('ignore')
os.makedirs("plots", exist_ok=True)


# Configuration
NUM_TRIALS = 5  # Reduced for faster testing
MAX_ITER = 20  # Reduced iterations
NUM_FEATURES = 3  # Number of features to select using FS1
NUM_TRAIN_SAMPLES = 180  # Number of samples to use for training (remaining used for testing)
OPTIMIZER = 'SLSQP'  # Options: 'SLSQP', 'COBYLA'
# algorithm_globals.random_seed = 123
# np.random.seed(algorithm_globals.random_seed)


# --- Helper Functions ---

# Global variables for tracking training progress
loss_history = []
iteration_count = 0
current_pbar = None

def loss_callback(weights):
    """Callback function to track iterations during training."""
    global iteration_count, current_pbar
    iteration_count += 1
    
    # Update progress bar if it exists
    if current_pbar is not None:
        current_pbar.set_description(f"Epoch {iteration_count}")
        current_pbar.update(1)
    
    return False  # Continue optimization

def plot_iteration_progress(all_iteration_histories, model_names, trial_idx=None):
    """Plot iteration progress for all models."""
    plt.figure(figsize=(12, 8))
    
    colors = ['blue', 'red', 'green']
    for i, (iteration_histories, model_name) in enumerate(zip(all_iteration_histories, model_names)):
        if trial_idx is not None:
            # Plot single trial
            if i < len(iteration_histories) and trial_idx < len(iteration_histories[i]):
                iterations = list(range(1, iteration_histories[i][trial_idx] + 1))
                plt.plot(iterations, color=colors[i], alpha=0.8, 
                        label=f'Model {i+1}: {model_name.split(".")[1].split()[0]}')
        else:
            # Plot all trials
            if iteration_histories:
                for trial_idx_inner, trial_iterations in enumerate(iteration_histories):
                    iterations = list(range(1, trial_iterations + 1))
                    plt.plot(iterations, [1] * len(iterations), color=colors[i], alpha=0.3, linewidth=1)
                
                # Plot mean iterations
                mean_iterations = np.mean(iteration_histories)
                plt.axhline(y=1, xmin=0, xmax=mean_iterations/MAX_ITER, color=colors[i], linewidth=3, 
                           label=f'Model {i+1}: {model_name.split(".")[1].split()[0]} (avg: {mean_iterations:.1f} iters)')
    
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Training Progress', fontsize=12)
    if trial_idx is not None:
        plt.title(f'Training Iteration Progress - Trial {trial_idx + 1}', fontsize=14, fontweight='bold')
        filename = f'plots/iteration_progress_trial_{trial_idx + 1}.png'
    else:
        plt.title('Training Iteration Progress - All Models', fontsize=14, fontweight='bold')
        filename = 'plots/iteration_progress_comparison.png'
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, MAX_ITER)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📈 Iteration progress saved to {filename}")

def get_optimizer(optimizer_name, max_iter):
    """Get optimizer object based on configuration."""
    if optimizer_name.upper() == 'SLSQP':
        return SLSQP(maxiter=max_iter, callback=loss_callback)
    elif optimizer_name.upper() == 'COBYLA':
        return COBYLA(maxiter=max_iter, callback=loss_callback)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}. Options: 'SLSQP', 'COBYLA'")


# --- Data Loading and Preprocessing ---

def load_and_prep_data(num_features=None, feature_selection_method='FS1', num_train_samples=None):
    """
    Load and preprocess the HTRU_2 dataset with feature selection.
    
    Parameters:
    -----------
    num_features : int, optional
        Number of features to select. If None, uses all features.
    feature_selection_method : str
        Feature selection method to use. Currently supports 'FS1' (SelectKBest with f_classif).
    num_train_samples : int, optional
        Number of samples to use for training. Remaining samples will be used for testing.
        If None, uses 80/20 split on all samples.
    
    Returns:
    --------
    train_features : np.ndarray
        Training features (selected and scaled)
    test_features : np.ndarray  
        Testing features (selected and scaled)
    train_labels : np.ndarray
        Training labels
    test_labels : np.ndarray
        Testing labels
    """
    # Load the HTRU_2 dataset
    df = pd.read_csv('HTRU_2.csv', header=None)
    
    # Separate features and labels
    X = df.iloc[:, :-1].values  # All columns except the last one
    y = df.iloc[:, -1].values   # Last column as labels
    
    total_samples = len(X)
    
    # Split into train/test based on num_train_samples
    if num_train_samples is not None and num_train_samples < total_samples:
        # Use stratified sampling to get training set
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=num_train_samples, random_state=42, stratify=y
        )
        print(f"📊 Training: {X_train.shape[0]} samples ({np.bincount(y_train)[0]} class 0, {np.bincount(y_train)[1]} class 1)")
        print(f"📊 Testing: {X_test.shape[0]} samples ({np.bincount(y_test)[0]} class 0, {np.bincount(y_test)[1]} class 1)")
    else:
        # Use 80/20 split on all samples
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=0.8, random_state=42, stratify=y
        )
        print(f"📊 Using 80/20 split on all {total_samples} samples")
        print(f"📊 Training: {X_train.shape[0]} samples, Testing: {X_test.shape[0]} samples")
    
    # Feature Selection (FS1) - fit on training data only
    if num_features is not None and feature_selection_method == 'FS1':
        if num_features > X_train.shape[1]:
            print(f"⚠️  Warning: Requested {num_features} features, but dataset only has {X_train.shape[1]}. Using all {X_train.shape[1]} features.")
            num_features = X_train.shape[1]
        
        if num_features < X_train.shape[1]:
            # Apply SelectKBest with f_classif - fit on training data only
            selector = SelectKBest(score_func=f_classif, k=num_features)
            X_train_selected = selector.fit_transform(X_train, y_train)
            X_test_selected = selector.transform(X_test)  # Apply same transformation to test data
            
            # Get the selected feature indices for reporting
            selected_indices = selector.get_support(indices=True)
            print(f"🔍 Selected top {num_features} features: {selected_indices.tolist()}")
            
            X_train, X_test = X_train_selected, X_test_selected
    
    # Scale features to [0, 1] range - fit scaler on training data only
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)  # Apply same scaling to test data
    
    return X_train_scaled, X_test_scaled, y_train, y_test



# ============================================================================
# MODEL 1: VQC with ZZFeatureMap + EfficientSU2 (FS1 with specified features)
# ============================================================================
def run_model_1(sampler, estimator, num_features=4, num_train_samples=None, num_trials=None):
    # Model 1 specific settings
    model1_entanglement = 'circular'  # Options: 'linear', 'circular', 'full', 'pairwise', 'sca'
    model1_loss = 'cross_entropy'   # Options: 'cross_entropy', 'squared_error'
    
    # Use passed num_trials or default to global NUM_TRIALS
    trials_to_run = num_trials if num_trials is not None else NUM_TRIALS
    
    model1_accuracies = []
    model1_iteration_histories = []
    train_features, test_features, train_labels, test_labels = load_and_prep_data(
        num_features=num_features, feature_selection_method='FS1', num_train_samples=num_train_samples
    )

    # Create and save circuit diagram on first iteration
    num_qubits = train_features.shape[1]
    feature_map = ZZFeatureMap(feature_dimension=num_qubits, reps=2)
    ansatz = EfficientSU2(num_qubits=num_qubits, reps=2, entanglement=model1_entanglement)
    
    # Create the full circuit for visualization
    full_circuit = QuantumCircuit(num_qubits)
    full_circuit.compose(feature_map, inplace=True)
    full_circuit.compose(ansatz, inplace=True)
    
    # Save circuit diagram using text representation
    try:
        # Try to create a matplotlib circuit diagram
        fig = full_circuit.draw(output='mpl', style='iqp', plot_barriers=False)
        circuit_filename = f'plots/circuit_VQC_FS1_{num_features}features.png'
        fig.suptitle(f'VQC Circuit: ZZFeatureMap + EfficientSU2 ({num_features} features, {model1_entanglement} entanglement)', fontsize=16, fontweight='bold')
        fig.tight_layout()
        fig.savefig(circuit_filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"✅ Circuit diagram saved to {circuit_filename}")
    except Exception as e:
        # Fallback: save text representation
        circuit_text = str(full_circuit)
        circuit_filename = f'plots/circuit_VQC_FS1_{num_features}features.txt'
        with open(circuit_filename, 'w') as f:
            f.write(f"VQC Circuit: ZZFeatureMap + EfficientSU2 ({num_features} features, {model1_entanglement} entanglement)\n")
            f.write("=" * 60 + "\n")
            f.write(circuit_text)
        print(f"✅ Circuit text saved to {circuit_filename}")

    # Training loop with nested progress bars
    for trial in tqdm(range(trials_to_run), desc="Model 1 Trials", ncols=80, position=0):
        # Reset iteration tracking for this trial
        global iteration_count, current_pbar
        iteration_count = 0
        
        # Create nested progress bar for epochs within this trial
        with tqdm(total=MAX_ITER, desc=f"Trial {trial+1} Epochs", ncols=80, position=1, leave=False) as epoch_pbar:
            current_pbar = epoch_pbar
            
            classifier = VQC(
                sampler=sampler,
                feature_map=feature_map,
                ansatz=ansatz,
                loss=model1_loss,
                optimizer=get_optimizer(OPTIMIZER, MAX_ITER),
            )
            
            classifier.fit(train_features, train_labels)
            
            # Reset the global progress bar reference
            current_pbar = None
        
        # Store the iteration count for this trial
        model1_iteration_histories.append(iteration_count)
        
        # Calculate accuracy on both training and testing data
        train_accuracy = classifier.score(train_features, train_labels)
        test_accuracy = classifier.score(test_features, test_labels)
        
        print(f"Trial {trial+1}/{trials_to_run} - Train: {train_accuracy:.4f}, Test: {test_accuracy:.4f} ({iteration_count} epochs)")
        
        model1_accuracies.append(test_accuracy)

    return model1_accuracies, model1_iteration_histories, f'1. VQC ZZ+EfficientSU2 FS1-{num_features}F\n({model1_entanglement} entanglement, {model1_loss} loss)'

# ============================================================================
# MODEL 2: VQC with ZZFeatureMap + Custom 6W ansatz
# ============================================================================
def run_model_2(sampler, estimator, num_features=4, num_train_samples=None, num_trials=None):
    # Model 2 specific settings
    model2_loss = 'cross_entropy'   # Options: 'cross_entropy', 'squared_error'
    
    # Use passed num_trials or default to global NUM_TRIALS
    trials_to_run = num_trials if num_trials is not None else NUM_TRIALS
    
    model2_accuracies = []
    model2_iteration_histories = []
    train_features, test_features, train_labels, test_labels = load_and_prep_data(
        num_features=num_features, feature_selection_method='FS1', num_train_samples=num_train_samples
    )

    # Create custom ansatz with 6 weights (hardcoded)
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
    
    # Create the full circuit for visualization
    full_circuit = QuantumCircuit(num_qubits)
    full_circuit.compose(feature_map, inplace=True)
    full_circuit.compose(ansatz, inplace=True)
    
    # Save circuit diagram using text representation
    try:
        # Try to create a matplotlib circuit diagram
        fig = full_circuit.draw(output='mpl', style='iqp', plot_barriers=False)
        circuit_filename = f'plots/circuit_VQC_Model2_{num_features}features.png'
        fig.suptitle(f'Model 2: ZZFeatureMap + Custom 6W ({num_features} features)', fontsize=16, fontweight='bold')
        fig.tight_layout()
        fig.savefig(circuit_filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"✅ Circuit diagram saved to {circuit_filename}")
    except Exception as e:
        # Fallback: save text representation
        circuit_text = str(full_circuit)
        circuit_filename = f'plots/circuit_VQC_Model2_{num_features}features.txt'
        with open(circuit_filename, 'w') as f:
            f.write(f"Model 2: ZZFeatureMap + Custom 6W ({num_features} features)\n")
            f.write("=" * 60 + "\n")
            f.write(circuit_text)
        print(f"✅ Circuit text saved to {circuit_filename}")

    # Training loop with nested progress bars
    for trial in tqdm(range(trials_to_run), desc="Model 2 Trials", ncols=80, position=0):
        # Reset iteration tracking for this trial
        global iteration_count, current_pbar
        iteration_count = 0
        
        # Create nested progress bar for epochs within this trial
        with tqdm(total=MAX_ITER, desc=f"Trial {trial+1} Epochs", ncols=80, position=1, leave=False) as epoch_pbar:
            current_pbar = epoch_pbar
            
            classifier = VQC(
                sampler=sampler,
                feature_map=feature_map,
                ansatz=ansatz,
                loss=model2_loss,
                optimizer=get_optimizer(OPTIMIZER, MAX_ITER),
            )
            
            classifier.fit(train_features, train_labels)
            
            # Reset the global progress bar reference
            current_pbar = None
        
        # Store the iteration count for this trial
        model2_iteration_histories.append(iteration_count)
        
        # Calculate accuracy on both training and testing data
        train_accuracy = classifier.score(train_features, train_labels)
        test_accuracy = classifier.score(test_features, test_labels)
        
        print(f"Model 2 - Trial {trial+1}/{trials_to_run} - Train: {train_accuracy:.4f}, Test: {test_accuracy:.4f} ({iteration_count} epochs)")
        
        model2_accuracies.append(test_accuracy)

    return model2_accuracies, model2_iteration_histories, f'2. VQC ZZ+Custom6W FS1-{num_features}F\n(RY-RZ-RX per qubit, {model2_loss} loss)'

# ============================================================================
# MODEL 3: VQC with ZZFeatureMap + Y gates + Custom 6W ansatz
# ============================================================================
def run_model_3(sampler, estimator, num_features=4, num_train_samples=None, num_trials=None):
    # Model 3 specific settings
    model3_loss = 'cross_entropy'   # Options: 'cross_entropy', 'squared_error'
    
    # Use passed num_trials or default to global NUM_TRIALS
    trials_to_run = num_trials if num_trials is not None else NUM_TRIALS
    
    model3_accuracies = []
    model3_iteration_histories = []
    train_features, test_features, train_labels, test_labels = load_and_prep_data(
        num_features=num_features, feature_selection_method='FS1', num_train_samples=num_train_samples
    )

    # Create custom feature map with ZZFeatureMap + parametric Y rotations
    num_qubits = train_features.shape[1]
    
    # Create enhanced feature map with ZZ encoding + Y rotations driven by features
    feature_map = QuantumCircuit(num_qubits)
    
    # Add ZZFeatureMap
    zz_feature_map = ZZFeatureMap(feature_dimension=num_qubits, reps=2)
    feature_map.compose(zz_feature_map, inplace=True)
    
    # Add parametric Y rotations using the same feature parameters as ZZFeatureMap
    feature_params = list(zz_feature_map.parameters)[:num_qubits]  # Get first num_qubits parameters
    for qubit in range(num_qubits):
        feature_map.ry(feature_params[qubit], qubit)

    # Create custom ansatz circuit (extended for 3 qubits)
    ansatz = QuantumCircuit(num_qubits)
    weight_params = [Parameter(f'weight_{i}') for i in range(9)]
    
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
    
    # Create the full circuit for visualization
    full_circuit = QuantumCircuit(num_qubits)
    full_circuit.compose(feature_map, inplace=True)
    full_circuit.compose(ansatz, inplace=True)
    
    # Save circuit diagram using text representation
    try:
        # Try to create a matplotlib circuit diagram
        fig = full_circuit.draw(output='mpl', style='iqp', plot_barriers=False)
        circuit_filename = f'plots/circuit_VQC_Model3_{num_features}features.png'
        fig.suptitle(f'Model 3: ZZFeatureMap + Y gates + Custom 6W ({num_features} features)', fontsize=16, fontweight='bold')
        fig.tight_layout()
        fig.savefig(circuit_filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"✅ Circuit diagram saved to {circuit_filename}")
    except Exception as e:
        # Fallback: save text representation
        circuit_text = str(full_circuit)
        circuit_filename = f'plots/circuit_VQC_Model3_{num_features}features.txt'
        with open(circuit_filename, 'w') as f:
            f.write(f"Model 3: ZZFeatureMap + Y gates + Custom 6W ({num_features} features)\n")
            f.write("=" * 60 + "\n")
            f.write(circuit_text)
        print(f"✅ Circuit text saved to {circuit_filename}")

    # Training loop with nested progress bars
    for trial in tqdm(range(trials_to_run), desc="Model 3 Trials", ncols=80, position=0):
        # Reset iteration tracking for this trial
        global iteration_count, current_pbar
        iteration_count = 0
        
        # Create nested progress bar for epochs within this trial
        with tqdm(total=MAX_ITER, desc=f"Trial {trial+1} Epochs", ncols=80, position=1, leave=False) as epoch_pbar:
            current_pbar = epoch_pbar
            
            classifier = VQC(
                sampler=sampler,
                feature_map=feature_map,
                ansatz=ansatz,
                loss=model3_loss,
                optimizer=get_optimizer(OPTIMIZER, MAX_ITER),
            )
            
            classifier.fit(train_features, train_labels)
            
            # Reset the global progress bar reference
            current_pbar = None
        
        # Store the iteration count for this trial
        model3_iteration_histories.append(iteration_count)
        
        # Calculate accuracy on both training and testing data
        train_accuracy = classifier.score(train_features, train_labels)
        test_accuracy = classifier.score(test_features, test_labels)
        
        print(f"Model 3 - Trial {trial+1}/{trials_to_run} - Train: {train_accuracy:.4f}, Test: {test_accuracy:.4f} ({iteration_count} epochs)")
        
        model3_accuracies.append(test_accuracy)

    return model3_accuracies, model3_iteration_histories, f'3. VQC ZZ+Y+Custom6W FS1-{num_features}F\n(ZZ+Y feature map, RY-RZ-RX ansatz, {model3_loss} loss)'

# ============================================================================
# RESULTS ANALYSIS AND VISUALIZATION
# ============================================================================
def analyze_and_visualize_results(accuracies, models, num_trials):
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
    plt.suptitle(f'HTRU_2 QNN Model Comparison - {num_trials} Trials Each', fontsize=16, fontweight='bold', y=1.02)
    plt.savefig('plots/gyro_comparison_histograms.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Box plot comparison
    plt.figure(figsize=(14, 8))
    box_plot = plt.boxplot(accuracies, labels=[m.replace('\n', ' ') for m in models], patch_artist=True)

    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    plt.title(f'HTRU_2 QNN Model Performance Comparison\n({num_trials} Trials)', 
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
    plt.savefig('plots/gyro_comparison_boxplots.png', dpi=150, bbox_inches='tight')
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
        plt.savefig('plots/gyro_comparison_p_values.png', dpi=150, bbox_inches='tight')
        plt.close()

    except ImportError:
        print("\n📈 Statistical analysis requires scipy, pandas, and seaborn (pip install scipy pandas seaborn)")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='HTRU_2 QNN Study')
    parser.add_argument('--single-trial', action='store_true', 
                       help='Train only 1 model instead of the default number of trials')
    parser.add_argument('--model', type=int, choices=[1, 2, 3], 
                       help='Specify a single model number to run (1-3). If not provided, all models are run.')
    args = parser.parse_args()
    
    # Determine number of trials based on command line argument
    num_trials = 1 if args.single_trial else NUM_TRIALS
    
    # Ensure plots directory exists
    os.makedirs("plots", exist_ok=True)

    print("🚀 HTRU_2 QNN Study")
    samples_text = f"{NUM_TRAIN_SAMPLES} train samples" if NUM_TRAIN_SAMPLES else "80/20 split"
    if args.model:
        print(f"🎯 Running only Model {args.model}")
    else:
        print(f"📊 Running all models")
    print(f"🔍 FS1 ({NUM_FEATURES} features) • {samples_text} • {num_trials} trials")
    print(f"⚙️  Optimizer: {OPTIMIZER} • Max Iter: {MAX_ITER}")
    if args.single_trial:
        print("🔄 Single trial mode enabled")
    print("=" * 60)

    sampler = Sampler()
    estimator = Estimator()

    # Define model runners
    model_runners = [run_model_1, run_model_2, run_model_3]

    if args.model:
        # Run a single specified model
        accuracies, iteration_histories, model_name = model_runners[args.model - 1](sampler, estimator, num_features=NUM_FEATURES, num_train_samples=NUM_TRAIN_SAMPLES, num_trials=num_trials)
        
        # Display single model results
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        min_acc = np.min(accuracies)
        max_acc = np.max(accuracies)
        print(f"\n✅ Results: {mean_acc:.3f} ± {std_acc:.3f} (range: {min_acc:.3f} - {max_acc:.3f})")

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
        plot_filename = f'plots/model{args.model}_histogram.png'
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
        for runner in model_runners:
            accuracies, iteration_histories, model_name = runner(sampler, estimator, num_features=NUM_FEATURES, num_train_samples=NUM_TRAIN_SAMPLES, num_trials=num_trials)
            all_accuracies.append(accuracies)
            all_iteration_histories.append(iteration_histories)
            all_model_names.append(model_name)
        
        analyze_and_visualize_results(all_accuracies, all_model_names, num_trials)
        
        # Create iteration progress comparison plot
        plot_iteration_progress(all_iteration_histories, all_model_names)

    print(f"✨ Study completed!")


if __name__ == "__main__":
    main()