# Standard libraries
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to avoid threading issues
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os  # Add this import for directory operations
from IPython.display import clear_output
from tqdm import tqdm
import sympy as sp
from sympy import Matrix, simplify
from sympy.physics.quantum.tensorproduct import TensorProduct
# Sklearn
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# Qiskit Core
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap, EfficientSU2
from qiskit.primitives import Estimator, Sampler
from qiskit.quantum_info import SparsePauliOp, Statevector

# Qiskit Algorithms
from qiskit_algorithms.optimizers import COBYLA, L_BFGS_B, ADAM
from qiskit_algorithms.utils import algorithm_globals

# Qiskit Machine Learning
from qiskit_machine_learning.algorithms.classifiers import VQC, NeuralNetworkClassifier
from qiskit_machine_learning.algorithms.regressors import NeuralNetworkRegressor, VQR
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
from qiskit_machine_learning.circuit.library import QNNCircuit, RawFeatureVector

# === MODEL/PLOT CONFIGURATION ===
MODEL_NAME = "2Q, 1CNOT+4F, 6W"
PLOT_DIR = "plots"
GRID_RESOLUTION = 20  # e.g., 20x20 grid for the landscape
PARAM_RANGE = (-np.pi, np.pi)  # Range for weight values
MAX_ITER = 30  # Max iterations for the optimizer
NUM_TRAINING_RUNS = 10  # Number of times to train the model for statistics

# Define a single circuit for feature map and ansatz
qc = QuantumCircuit(2)
input_params = [Parameter(f"input{i}") for i in range(4)]
weight_params = [Parameter(f"weight{i}") for i in range(6)]

# Feature map
qc.cx(0, 1)
qc.ry(input_params[0],0)
qc.rz(input_params[1],0)
qc.ry(input_params[2],1)
qc.rz(input_params[3],1)

qc.barrier()

# Ansatz

qc.rz(weight_params[0], 0)
qc.ry(weight_params[1], 0)
qc.rx(weight_params[2], 0)
qc.ry(weight_params[3], 1)
qc.rz(weight_params[4], 1)
qc.rx(weight_params[5], 1)

# Create plots directory if it doesn't exist
os.makedirs("plots", exist_ok=True)

# Load in condensed data
iris_data = load_iris()
features = iris_data.data
features = PCA(n_components=2).fit_transform(features)
labels = iris_data.target

# Normalize data (petal length, width, etc. are on different scales)
features = MinMaxScaler().fit_transform(features)

# Create paired data for Siamese-like training
# Set seed only for consistent data generation across runs
np.random.seed(123)

def create_flower_pairs(features, labels, num_pairs=150):
    n_samples = len(features)
    paired_features = []
    paired_labels = []
    
    # Ensure balanced classes for pairs
    same_class_count = 0
    diff_class_count = 0
    target_per_class = num_pairs // 2

    while same_class_count < target_per_class or diff_class_count < target_per_class:
        idx1, idx2 = np.random.choice(n_samples, size=2, replace=False)
        same_class = 1 if labels[idx1] == labels[idx2] else -1

        if same_class == 1 and same_class_count < target_per_class:
            flower_pair = np.concatenate([features[idx1], features[idx2]])
            paired_features.append(flower_pair)
            paired_labels.append(same_class)
            same_class_count += 1
        elif same_class == -1 and diff_class_count < target_per_class:
            flower_pair = np.concatenate([features[idx1], features[idx2]])
            paired_features.append(flower_pair)
            paired_labels.append(same_class)
            diff_class_count += 1

    return np.array(paired_features), np.array(paired_labels)

# Generate flower pairs
paired_features, paired_labels = create_flower_pairs(features, labels, num_pairs=600)

# Split data
train_features, test_features, train_labels, test_labels = train_test_split(
    paired_features, paired_labels, train_size=0.8, random_state=123, stratify=paired_labels
)

# --- EstimatorQNN Training ---
estimator = Estimator()

# A single observable for a single output value.
observable = SparsePauliOp("ZZ")

# Define EstimatorQNN
estimator_qnn = EstimatorQNN(
    circuit=qc,
    estimator=estimator,
    input_params=input_params,
    weight_params=weight_params,
    observables=observable
)

objective_func_vals = []
def callback_graph(weights, obj_func_eval):
    iteration = len(objective_func_vals)
    objective_func_vals.append(obj_func_eval)

# Define classifier
classifier = NeuralNetworkClassifier(
    estimator_qnn,
    optimizer=COBYLA(),
    loss="squared_error",  # Use squared error loss for -1, 1 labels
    callback=callback_graph
)

# Update optimizer with MAX_ITER
classifier.optimizer.set_options(maxiter=MAX_ITER)



# --- Loss Calculation ---

def calculate_loss_for_qnn(qnn, X, y, weights, loss_type):
    """
    Calculates the loss for a given QNN, weights, and data.
    Supports cross-entropy for multi-class and squared error for binary.
    """
    if loss_type == 'cross_entropy':
        num_classes = len(np.unique(y))
        y_one_hot = np.eye(num_classes)[y]
        y_pred_probs = qnn.forward(X, weights)
        y_pred_probs = np.clip(y_pred_probs, 1e-9, 1 - 1e-9) # Avoid log(0)
        loss = -np.mean(np.sum(y_one_hot * np.log(y_pred_probs), axis=1))
    elif loss_type == 'squared_error':
        y_pred = qnn.forward(X, weights).flatten()
        loss = np.mean((y - y_pred)**2)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")
    return loss

# --- Landscape Calculation & Plotting ---

def calculate_loss_slice(qnn, X, y, trained_weights, loss_type, slice_params, param_range, grid_resolution):
    """Calculates a 2D slice of a high-dimensional loss landscape."""
    w1_vals = np.linspace(param_range[0], param_range[1], grid_resolution)
    w2_vals = np.linspace(param_range[0], param_range[1], grid_resolution)
    W1, W2 = np.meshgrid(w1_vals, w2_vals)

    loss_grid = np.zeros(W1.shape)

    for i in range(grid_resolution):
        for j in range(grid_resolution):
            weights = np.copy(trained_weights)
            weights[slice_params[0]] = W1[i, j]
            weights[slice_params[1]] = W2[i, j]
            loss_grid[i, j] = calculate_loss_for_qnn(qnn, X, y, weights, loss_type)

    return W1, W2, loss_grid

def plot_loss_landscape(W1, W2, loss_grid, title, initial_weights, final_weights, initial_loss, final_loss, slice_params, plot_dir):
    """Plots the 3D loss landscape slice."""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(W1, W2, loss_grid, cmap='viridis', edgecolor='none', alpha=0.8)
    fig.colorbar(surf, shrink=0.5, aspect=5, label='Loss')
    
    xlabel = f'Weight {slice_params[0]} (θ{slice_params[0]})'
    ylabel = f'Weight {slice_params[1]} (θ{slice_params[1]})'

    # Plot initial and final points projected onto the slice
    ax.scatter(initial_weights[slice_params[0]], initial_weights[slice_params[1]], initial_loss, 
               s=100, c='green', marker='o', depthshade=True, label='Initial Point (on slice)')
    ax.scatter(final_weights[slice_params[0]], final_weights[slice_params[1]], final_loss, 
               s=200, c='red', marker='*', depthshade=True, label='Optimal Point')
    ax.legend()

    ax.set_title(title, fontsize=16, pad=20)
    ax.set_xlabel(xlabel, fontsize=12, labelpad=10)
    ax.set_ylabel(ylabel, fontsize=12, labelpad=10)
    ax.set_zlabel('Loss', fontsize=12, labelpad=10)
    ax.view_init(elev=20., azim=-65)
    
    # Simple filename with weight indices first
    plot_filename = os.path.join(plot_dir, f"{slice_params[0]}_{slice_params[1]}_loss_landscape.png")
    plt.savefig(plot_filename, dpi=120, bbox_inches='tight')
    plt.close()

def save_circuit_diagram(circuit, model_name, plot_dir):
    """Saves the circuit diagram to the model-specific folder."""
    sanitized_model_name = model_name.replace(' ', '_').replace(':', '').replace('(', '').replace(')', '')
    circuit_filename = f"{sanitized_model_name}_circuit.png"
    circuit_path = os.path.join(plot_dir, circuit_filename)
    
    circuit.draw(output="mpl", style="clifford", fold=20)
    plt.suptitle(f"Circuit for {model_name}")
    plt.savefig(circuit_path, dpi=150, bbox_inches='tight')
    plt.close()

def save_training_overview(model_name, plot_dir, all_histories, train_scores, test_scores, final_losses, best_weights):
    """Saves a plot of the loss history and a text file with training stats."""
    # Plot loss histories for all runs
    plt.figure(figsize=(12, 6))
    for i, history in enumerate(all_histories):
        plt.plot(history, alpha=0.7, label=f'Run {i+1}' if len(all_histories) <= 5 else None)
    
    # Plot mean history
    if len(all_histories) > 1:
        mean_history = np.mean(all_histories, axis=0)
        plt.plot(mean_history, 'k-', linewidth=2, label='Mean')
    
    plt.title(f'Loss History for {model_name}')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    if len(all_histories) <= 5:
        plt.legend()
    plt.grid(True)
    loss_plot_path = os.path.join(plot_dir, "loss_history.png")
    plt.savefig(loss_plot_path, dpi=120)
    plt.close()

    # Calculate statistics
    train_mean, train_std = np.mean(train_scores), np.std(train_scores)
    test_mean, test_std = np.mean(test_scores), np.std(test_scores)
    loss_mean, loss_std = np.mean(final_losses), np.std(final_losses)

    # Save overview text file
    overview_path = os.path.join(plot_dir, "overview.txt")
    with open(overview_path, 'w') as f:
        f.write(f"--- Overview for {model_name} ---\n\n")
        f.write(f"Number of Training Runs: {len(train_scores)}\n\n")
        f.write(f"Training Accuracy:  {train_mean:.4f} ± {train_std:.4f}\n")
        f.write(f"Test Accuracy:      {test_mean:.4f} ± {test_std:.4f}\n")
        f.write(f"Final Loss:         {loss_mean:.4f} ± {loss_std:.4f}\n")
        f.write(f"Iterations per run: {len(all_histories[0])}\n\n")
        f.write("Individual Results:\n")
        for i in range(len(train_scores)):
            f.write(f"  Run {i+1}: Train={train_scores[i]:.4f}, Test={test_scores[i]:.4f}, Loss={final_losses[i]:.4f}\n")
        f.write(f"\nBest Weights (from best test run):\n")
        f.write(np.array2string(best_weights, formatter={'float_kind':lambda x: "%.4f" % x}))

def process_and_plot_model(model_name, classifier, qnn, X_train, y_train, X_test, y_test, loss_type, slice_pairs, num_runs=NUM_TRAINING_RUNS):
    """Train model multiple times, calculate loss slices, plot them, and save an overview."""
    
    # Sanitize model name for directory creation and create plot directory
    sanitized_model_name = model_name.replace(' ', '_').replace(':', '').replace('(', '').replace(')', '')
    model_plot_dir = os.path.join("plots", sanitized_model_name)
    os.makedirs(model_plot_dir, exist_ok=True)

    # Save circuit diagram
    save_circuit_diagram(qc, model_name, model_plot_dir)

    # Train multiple models and collect statistics
    all_histories = []
    train_scores = []
    test_scores = []
    final_losses = []
    all_weights = []

    # Progress bar for model training
    with tqdm(total=num_runs, desc=f"Training {model_name}", unit="run") as pbar:
        for run in range(num_runs):
            # Clear objective function values for this run
            objective_func_vals.clear()
            
            # Create a fresh classifier for each run to ensure random initialization
            fresh_classifier = NeuralNetworkClassifier(
                qnn,
                optimizer=COBYLA(),
                loss="squared_error",
                callback=callback_graph
            )
            fresh_classifier.optimizer.set_options(maxiter=MAX_ITER)
            
            # Train the model
            fresh_classifier.fit(X_train, y_train)
            
            # Collect results
            all_histories.append(objective_func_vals.copy())
            train_scores.append(fresh_classifier.score(X_train, y_train))
            test_scores.append(fresh_classifier.score(X_test, y_test))
            final_losses.append(calculate_loss_for_qnn(qnn, X_train, y_train, fresh_classifier.weights, loss_type))
            all_weights.append(fresh_classifier.weights.copy())
            
            # Store the last classifier for getting initial_point later
            if run == num_runs - 1:
                classifier = fresh_classifier
            
            pbar.update(1)

    # Use the best model (highest test score) for landscape plotting
    best_idx = np.argmax(test_scores)
    best_weights = all_weights[best_idx]
    best_initial_weights = classifier.initial_point

    # Save overview with all statistics BEFORE generating loss landscapes
    save_training_overview(
        model_name,
        model_plot_dir,
        all_histories=all_histories,
        train_scores=train_scores,
        test_scores=test_scores,
        final_losses=final_losses,
        best_weights=best_weights
    )

    # Progress bar for landscape calculation
    with tqdm(total=len(slice_pairs), desc="Calculating loss landscapes", unit="slice") as pbar:
        for slice_params in slice_pairs:
            # Calculate loss for the initial point projected on this slice
            initial_weights_slice = np.copy(best_weights)
            initial_weights_slice[slice_params[0]] = best_initial_weights[slice_params[0]]
            initial_weights_slice[slice_params[1]] = best_initial_weights[slice_params[1]]
            initial_loss_slice = calculate_loss_for_qnn(qnn, X_train, y_train, initial_weights_slice, loss_type)

            W1, W2, loss_slice = calculate_loss_slice(
                qnn, X_train, y_train, best_weights, loss_type,
                slice_params=slice_params, grid_resolution=GRID_RESOLUTION, param_range=PARAM_RANGE
            )

            plot_title = f"{model_name} (Slice: Weights {slice_params[0]} & {slice_params[1]})"
            plot_loss_landscape(
                W1, W2, loss_slice, plot_title,
                initial_weights=best_initial_weights, final_weights=best_weights,
                initial_loss=initial_loss_slice, final_loss=final_losses[best_idx],
                slice_params=slice_params,
                plot_dir=model_plot_dir
            )
            
            pbar.update(1)
    
    return np.mean(train_scores), np.mean(test_scores), best_weights

# --- Main Execution ---
if __name__ == "__main__":
    # Use the model name defined at the top of the file
    model_title = MODEL_NAME

    # Define the slices to plot: all possible dual permutations of the weights
    from itertools import combinations
    num_weights = len(weight_params)  # Automatically get the number of weights
    slice_pairs = list(combinations(range(num_weights), 2))

    training_score, test_score, final_weights = process_and_plot_model(
        model_name=model_title,
        classifier=classifier,
        qnn=estimator_qnn,
        X_train=train_features,
        y_train=train_labels,
        X_test=test_features,
        y_test=test_labels,
        loss_type='squared_error',
        slice_pairs=slice_pairs
    )