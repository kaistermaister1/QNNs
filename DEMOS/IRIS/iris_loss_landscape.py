#!/usr/bin/env python3
"""
3D Loss Landscape Visualization for IRIS QNN Models
======================================================

This script visualizes the 3D loss landscapes for the 7 different
QNN architectures from the `iris_comparison.py` study.

For models with more than 2 parameters (all of them), this script
visualizes 2D slices of the loss landscape, keeping other parameters
fixed at their trained optimal values.
"""

import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.utils import algorithm_globals
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier, VQC
from qiskit_machine_learning.neural_networks import EstimatorQNN, SamplerQNN
from qiskit.primitives import Estimator, Sampler
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap, EfficientSU2
from qiskit.quantum_info import SparsePauliOp

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

import warnings
import argparse
import os
from tqdm.auto import tqdm # For progress bars

warnings.filterwarnings('ignore')
# Define the output directory for plots
PLOT_DIR = "plots/loss_landscapes"
os.makedirs(PLOT_DIR, exist_ok=True)


# Configuration
RANDOM_SEED = 42
GRID_RESOLUTION = 15  # Lower for faster computation (e.g., 25x25)
PARAM_RANGE = (-np.pi, np.pi)
MAX_ITER = 60 # Optimizer iterations for finding optimal weights
algorithm_globals.random_seed = RANDOM_SEED
np.random.seed(algorithm_globals.random_seed)

# --- Data Loading and Preprocessing (from iris_comparison.py) ---

def load_and_prep_data(condensed=False):
    """Load, scale, and optionally reduce dimensionality of IRIS dataset."""
    iris_data = load_iris()
    features = iris_data.data
    labels = iris_data.target
    if condensed:
        features = PCA(n_components=2).fit_transform(features)
    features = MinMaxScaler().fit_transform(features)
    return features, labels

def create_flower_pairs(features, labels, num_pairs=120):
    """Create pairs of flowers for Siamese-like models."""
    n_samples = len(features)
    paired_features = []
    paired_labels = []
    target_per_class = num_pairs // 2
    same_class_count = 0
    diff_class_count = 0

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
    
    total_points = grid_resolution * grid_resolution
    with tqdm(total=total_points, desc=f"   Calculating Slice {slice_params}", leave=False) as pbar:
        for i in range(grid_resolution):
            for j in range(grid_resolution):
                weights = np.copy(trained_weights)
                weights[slice_params[0]] = W1[i, j]
                weights[slice_params[1]] = W2[i, j]
                loss_grid[i, j] = calculate_loss_for_qnn(qnn, X, y, weights, loss_type)
                pbar.update(1)
            
    return W1, W2, loss_grid

def plot_loss_landscape(W1, W2, loss_grid, title, initial_weights, final_weights, initial_loss, final_loss, slice_params):
    """Plots the 3D loss landscape slice."""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(W1, W2, loss_grid, cmap='viridis', edgecolor='none', alpha=0.8)
    fig.colorbar(surf, shrink=0.5, aspect=5, label='Loss')
    
    xlabel = f'Weight {slice_params[0]+1} (θ{slice_params[0]+1})'
    ylabel = f'Weight {slice_params[1]+1} (θ{slice_params[1]+1})'

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
    
    plot_filename = f"{PLOT_DIR}/iris_loss_landscape_{title.replace(' ', '_').replace(':', '').replace('(', '').replace(')', '')}.png"
    plt.savefig(plot_filename, dpi=120, bbox_inches='tight')
    print(f"   ✅ Saved plot to {plot_filename}")
    plt.close()

def process_and_plot_model(model_name, classifier, qnn, X_train, y_train, loss_type, slice_pairs):
    """Train model, calculate loss slices, and plot them."""
    print(f"\n--- Processing {model_name} ---")
    
    # Train the model to find the final weights
    print(f"   Training model ({MAX_ITER} iterations)...")
    classifier.fit(X_train, y_train)
    final_weights = classifier.weights
    initial_weights = classifier.initial_point
    
    final_loss = calculate_loss_for_qnn(qnn, X_train, y_train, final_weights, loss_type)

    for slice_params in slice_pairs:
        # Calculate loss for the initial point projected on this slice
        initial_weights_slice = np.copy(final_weights)
        initial_weights_slice[slice_params[0]] = initial_weights[slice_params[0]]
        initial_weights_slice[slice_params[1]] = initial_weights[slice_params[1]]
        initial_loss_slice = calculate_loss_for_qnn(qnn, X_train, y_train, initial_weights_slice, loss_type)

        W1, W2, loss_slice = calculate_loss_slice(
            qnn, X_train, y_train, final_weights, loss_type,
            slice_params=slice_params, grid_resolution=GRID_RESOLUTION, param_range=PARAM_RANGE
        )
        
        plot_title = f"{model_name} (Slice: Weights {slice_params[0]+1} & {slice_params[1]+1})"
        plot_loss_landscape(
            W1, W2, loss_slice, plot_title,
            initial_weights=initial_weights, final_weights=final_weights,
            initial_loss=initial_loss_slice, final_loss=final_loss,
            slice_params=slice_params
        )

# --- Main Script ---

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate 3D loss landscape visualizations for IRIS QNN models.")
    parser.add_argument(
        '--model',
        type=int,
        choices=range(1, 8),
        help="Specify a single model number to plot (1-7). If not provided, all models are run."
    )
    args = parser.parse_args()

    # --- Global Setup ---
    sampler = Sampler()
    estimator = Estimator()

    # --- Generate Datasets ---
    print("Preparing datasets...")
    features_4d, labels_cat = load_and_prep_data(condensed=False)
    features_2d, _ = load_and_prep_data(condensed=True)
    
    X_train_4d, _, y_train_cat_4d, _ = train_test_split(features_4d, labels_cat, train_size=0.8, random_state=RANDOM_SEED)
    X_train_2d, _, y_train_cat_2d, _ = train_test_split(features_2d, labels_cat, train_size=0.8, random_state=RANDOM_SEED)

    paired_f_8d, paired_l_bin_8d = create_flower_pairs(features_4d, labels_cat)
    X_train_p8d, _, y_train_p8d, _ = train_test_split(paired_f_8d, paired_l_bin_8d, train_size=0.8, random_state=RANDOM_SEED, stratify=paired_l_bin_8d)
    
    paired_f_4d, paired_l_bin_4d = create_flower_pairs(features_2d, labels_cat)
    X_train_p4d, _, y_train_p4d, _ = train_test_split(paired_f_4d, paired_l_bin_4d, train_size=0.8, random_state=RANDOM_SEED, stratify=paired_l_bin_4d)

    # Determine which models to run
    models_to_run = [args.model] if args.model else list(range(1, 8))

    # ============================================================================
    # MODEL 1: VQC with ZZFeatureMap + RealAmplitudes (4 features)
    # ============================================================================
    if 1 in models_to_run:
        feature_map = ZZFeatureMap(feature_dimension=4, reps=1)
        ansatz = RealAmplitudes(num_qubits=4, reps=3) # 16 weights
        classifier = VQC(sampler=sampler, feature_map=feature_map, ansatz=ansatz, optimizer=COBYLA(maxiter=MAX_ITER))
        process_and_plot_model("Model 1: VQC ZZ+RA 4F", classifier, classifier.neural_network, 
                               X_train_4d, y_train_cat_4d, 'cross_entropy', [(0, 1), (2, 3)])

    # ============================================================================
    # MODEL 2: VQC with ZZFeatureMap + EfficientSU2 (2 features)
    # ============================================================================
    if 2 in models_to_run:
        feature_map = ZZFeatureMap(feature_dimension=2, reps=1)
        ansatz = EfficientSU2(num_qubits=2, reps=3) # 18 weights
        classifier = VQC(sampler=sampler, feature_map=feature_map, ansatz=ansatz, optimizer=COBYLA(maxiter=MAX_ITER))
        process_and_plot_model("Model 2: VQC ZZ+ESU2 2F", classifier, classifier.neural_network,
                               X_train_2d, y_train_cat_2d, 'cross_entropy', [(0, 1), (2, 3)])

    # ============================================================================
    # MODEL 3: Siamese-like QNN (4-feature pairs)
    # ============================================================================
    if 3 in models_to_run:
        qc = QuantumCircuit(4)
        inputs = [Parameter(f"i{i}") for i in range(8)]
        weights = [Parameter(f"w{i}") for i in range(8)]
        qc.ry(inputs[0], 0); qc.rz(inputs[1], 0); qc.ry(inputs[2], 1); qc.rz(inputs[3], 1)
        qc.ry(inputs[4], 2); qc.rz(inputs[5], 2); qc.ry(inputs[6], 3); qc.rz(inputs[7], 3)
        qc.cx(0, 2); qc.cx(1, 3); qc.barrier()
        qc.ry(weights[0], 0); qc.rz(weights[1], 1); qc.ry(weights[2], 2); qc.rz(weights[3], 3)
        qc.rz(weights[4], 0); qc.ry(weights[5], 1); qc.rz(weights[6], 2); qc.ry(weights[7], 3)
        qnn = EstimatorQNN(circuit=qc, estimator=estimator, input_params=inputs, weight_params=weights, observables=SparsePauliOp("ZZZZ"))
        classifier = NeuralNetworkClassifier(qnn, optimizer=COBYLA(maxiter=MAX_ITER), loss="squared_error")
        process_and_plot_model("Model 3: Siamese 4F", classifier, qnn,
                               X_train_p8d, y_train_p8d, 'squared_error', [(0, 1), (2, 3), (4,5), (6,7)])

    # ============================================================================
    # MODEL 4: Siamese-like QNN (2-feature pairs)
    # ============================================================================
    if 4 in models_to_run:
        qc = QuantumCircuit(4)
        inputs = [Parameter(f"i{i}") for i in range(4)]
        weights = [Parameter(f"w{i}") for i in range(8)]
        qc.ry(inputs[0], 0); qc.ry(inputs[1], 1); qc.ry(inputs[2], 2); qc.ry(inputs[3], 3)
        qc.cx(0, 2); qc.cx(1, 3); qc.barrier()
        qc.ry(weights[0], 0); qc.ry(weights[1], 1); qc.ry(weights[2], 2); qc.ry(weights[3], 3)
        qc.rz(weights[4], 0); qc.rz(weights[5], 1); qc.rz(weights[6], 2); qc.rz(weights[7], 3)
        qnn = EstimatorQNN(circuit=qc, estimator=estimator, input_params=inputs, weight_params=weights, observables=SparsePauliOp("ZZZZ"))
        classifier = NeuralNetworkClassifier(qnn, optimizer=COBYLA(maxiter=MAX_ITER), loss="squared_error")
        process_and_plot_model("Model 4: Siamese 2F", classifier, qnn,
                               X_train_p4d, y_train_p4d, 'squared_error', [(0, 1), (2, 3), (4,5), (6,7)])

    # ============================================================================
    # MODEL 5: VQC with Custom Feature Map + Custom Ansatz (4 features)
    # ============================================================================
    if 5 in models_to_run:
        feature_map = QuantumCircuit(4, name="fm5")
        feature_map.ry(Parameter("i0"), 0); feature_map.ry(Parameter("i1"), 1)
        feature_map.ry(Parameter("i2"), 2); feature_map.ry(Parameter("i3"), 3)
        ansatz = QuantumCircuit(4, name="an5")
        weights = [Parameter(f"w{i}") for i in range(8)]
        ansatz.ry(weights[0], 0); ansatz.ry(weights[1], 1); ansatz.ry(weights[2], 2); ansatz.ry(weights[3], 3)
        ansatz.rz(weights[4], 0); ansatz.rz(weights[5], 1); ansatz.rz(weights[6], 2); ansatz.rz(weights[7], 3)
        classifier = VQC(sampler=sampler, feature_map=feature_map, ansatz=ansatz, optimizer=COBYLA(maxiter=MAX_ITER))
        process_and_plot_model("Model 5: VQC Custom 4F", classifier, classifier.neural_network,
                               X_train_4d, y_train_cat_4d, 'cross_entropy', [(0, 1), (2, 3), (4,5), (6,7)])

    # ============================================================================
    # MODEL 6: VQC with custom feature map and ansatz (2 features)
    # ============================================================================
    if 6 in models_to_run:
        feature_map = QuantumCircuit(2, name="fm6")
        feature_map.ry(Parameter("i0"), 0); feature_map.ry(Parameter("i1"), 1)
        ansatz = QuantumCircuit(2, name="an6")
        weights = [Parameter(f"w{i}") for i in range(4)]
        ansatz.ry(weights[0], 0); ansatz.rz(weights[1], 0)
        ansatz.ry(weights[2], 1); ansatz.rz(weights[3], 1)
        classifier = VQC(sampler=sampler, feature_map=feature_map, ansatz=ansatz, optimizer=COBYLA(maxiter=MAX_ITER))
        process_and_plot_model("Model 6: VQC Custom 2F", classifier, classifier.neural_network,
                               X_train_2d, y_train_cat_2d, 'cross_entropy', [(0, 1), (2, 3)])

    # ============================================================================
    # MODEL 7: 2-Qubit Siamese-like QNN (2-feature pairs, condensed)
    # ============================================================================
    if 7 in models_to_run:
        qc = QuantumCircuit(2)
        inputs = [Parameter(f"i{i}") for i in range(4)]
        weights = [Parameter(f"w{i}") for i in range(4)]
        qc.ry(inputs[0], 0); qc.rz(inputs[1], 0)
        qc.ry(inputs[2], 1); qc.rz(inputs[3], 1)
        qc.cx(0, 1); qc.barrier()
        qc.ry(weights[0], 0); qc.ry(weights[1], 1)
        qc.rz(weights[2], 0); qc.rz(weights[3], 1)
        qnn = EstimatorQNN(circuit=qc, estimator=estimator, input_params=inputs, weight_params=weights, observables=SparsePauliOp("ZZ"))
        classifier = NeuralNetworkClassifier(qnn, optimizer=COBYLA(maxiter=MAX_ITER), loss="squared_error")
        process_and_plot_model("Model 7: 2Q Siamese 2F", classifier, qnn,
                               X_train_p4d, y_train_p4d, 'squared_error', [(0, 1), (2, 3)])

    print("\n\n✨ Landscape visualization complete.") 