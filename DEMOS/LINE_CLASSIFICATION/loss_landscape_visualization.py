"""
3D Loss Landscape Visualization for 1-Qubit QNN Models
======================================================

This script visualizes the 3D loss landscapes for two different
1-qubit Quantum Neural Network architectures from the QNN comparison study.

1. 1-Qubit Angle Embedding (RY + RZ gates)
2. 1-Qubit Amplitude Embedding (single RY gate with preprocessed angles)
"""

import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.utils import algorithm_globals
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import Estimator
from qiskit.circuit.library import RealAmplitudes
import warnings

warnings.filterwarnings('ignore')

# Configuration
NUM_INPUTS = 2
NUM_SAMPLES = 20
RANDOM_SEED = 42
GRID_RESOLUTION = 30  # <--- Set grid size here (e.g., 30 for 30x30)
PARAM_RANGE = (-2*np.pi, 2*np.pi) # <--- Set parameter range for weights here
algorithm_globals.random_seed = RANDOM_SEED

# --- Data Generation and Preprocessing ---

def generate_dataset():
    """Generate a random binary classification dataset"""
    X = 2 * algorithm_globals.random.random([NUM_SAMPLES, NUM_INPUTS]) - 1
    y01 = 1 * (np.sum(X, axis=1) >= 0)  # Points above/below y = -x line
    y = 2 * y01 - 1  # Map to {-1, +1}
    return X, y

def preprocess_for_amplitude_embedding(X):
    """
    Convert 2D coordinates to angles for amplitude embedding.
    """
    X_processed = []
    for x_sample in X:
        norm = np.linalg.norm(x_sample)
        if norm == 0:
            angle = 0
        else:
            angle = 2 * np.arccos(x_sample[0] / norm)
        X_processed.append(angle)
    return np.array(X_processed).reshape(-1, 1)


def calculate_loss_for_weights(qnn, X, y, weights):
    """Calculates the cross-entropy loss for a specific set of weights."""
    y_01 = (y + 1) / 2
    y_pred_qnn = qnn.forward(X, weights).flatten()
    p_pred = (y_pred_qnn + 1) / 2
    loss = -np.mean(y_01 * np.log(p_pred + 1e-9) + (1 - y_01) * np.log(1 - p_pred + 1e-9))
    return loss


def calculate_loss_landscape(qnn, X, y, param_range=(-np.pi, np.pi), grid_resolution=30):
    """Calculates the loss over a grid of weight parameters using Binary Cross-Entropy."""
    w1_vals = np.linspace(param_range[0], param_range[1], grid_resolution)
    w2_vals = np.linspace(param_range[0], param_range[1], grid_resolution)
    W1, W2 = np.meshgrid(w1_vals, w2_vals)
    
    loss_grid = np.zeros(W1.shape)
    
    print(f"Calculating loss landscape ({grid_resolution}x{grid_resolution} grid)...")
    for i in range(grid_resolution):
        for j in range(grid_resolution):
            weights = np.array([W1[i, j], W2[i, j]])
            loss_grid[i, j] = calculate_loss_for_weights(qnn, X, y, weights)
            
    return W1, W2, loss_grid

def calculate_loss_slice(qnn, X, y, trained_weights, slice_params=(0, 1), param_range=(-np.pi, np.pi), grid_resolution=30):
    """Calculates a 2D slice of a high-dimensional loss landscape."""
    w1_vals = np.linspace(param_range[0], param_range[1], grid_resolution)
    w2_vals = np.linspace(param_range[0], param_range[1], grid_resolution)
    W1, W2 = np.meshgrid(w1_vals, w2_vals)
    
    loss_grid = np.zeros(W1.shape)
    
    print(f"Calculating loss slice ({grid_resolution}x{grid_resolution} grid) for params {slice_params}...")
    for i in range(grid_resolution):
        for j in range(grid_resolution):
            # Start with a copy of the trained weights
            weights = np.copy(trained_weights)
            # Overwrite the slice parameters with grid values
            weights[slice_params[0]] = W1[i, j]
            weights[slice_params[1]] = W2[i, j]
            loss_grid[i, j] = calculate_loss_for_weights(qnn, X, y, weights)
            
    return W1, W2, loss_grid

def plot_loss_landscape(W1, W2, loss_grid, title, initial_weights=None, final_weights=None, initial_loss=None, final_loss=None, slice_params=None):
    """Plots the 3D loss landscape."""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(W1, W2, loss_grid, cmap='viridis', edgecolor='none', alpha=0.8)
    fig.colorbar(surf, shrink=0.5, aspect=5, label='Loss (Cross-Entropy)')
    
    xlabel = 'Weight 1 (θ₁)'
    ylabel = 'Weight 2 (θ₂)'
    if slice_params:
        xlabel = f'Weight {slice_params[0]+1} (θ{slice_params[0]+1})'
        ylabel = f'Weight {slice_params[1]+1} (θ{slice_params[1]+1})'

    if initial_weights is not None and final_weights is not None:
        # For a slice plot, we plot the projection of initial and final weights on the plane
        if slice_params:
            # Plot initial point for the slice (Green Circle)
            ax.scatter(initial_weights[slice_params[0]], initial_weights[slice_params[1]], initial_loss, 
                       s=100, c='green', marker='o', depthshade=True, label='Initial Point (on slice)')
            # Plot final point (Red Star)
            ax.scatter(final_weights[slice_params[0]], final_weights[slice_params[1]], final_loss, 
                       s=200, c='red', marker='*', depthshade=True, label='Optimal Point')
        else:
             # Plot initial point (Green Circle)
            ax.scatter(initial_weights[0], initial_weights[1], initial_loss, 
                       s=100, c='green', marker='o', depthshade=True, label='Initial Weights')
            # Plot final point (Red Star)
            ax.scatter(final_weights[0], final_weights[1], final_loss, 
                       s=200, c='red', marker='*', depthshade=True, label='Final Weights')
        ax.legend()

    ax.set_title(title, fontsize=16, pad=20)
    ax.set_xlabel(xlabel, fontsize=12, labelpad=10)
    ax.set_ylabel(ylabel, fontsize=12, labelpad=10)
    ax.set_zlabel('Loss', fontsize=12, labelpad=10)
    ax.view_init(elev=20., azim=-65)
    plt.show()

# --- Main Script ---

if __name__ == '__main__':
    estimator = Estimator()
    # Generate a single, fixed dataset for consistent landscapes
    X_train, y_train = generate_dataset()
    
    # Define a fixed initial starting point for the optimizer
    initial_weights = np.array([0.5, -0.5])


    # ============================================================================
    # MODEL 1: 1-Qubit Angle Embedding (RY + RZ)
    # ============================================================================
    print("\n--- Processing Model 1: 1-Qubit Angle Embedding (RY + RZ) ---")

    feature_map_1 = QuantumCircuit(1, name="FeatureMap1")
    params_1 = [Parameter("input1"), Parameter("input2")]
    feature_map_1.ry(params_1[0], 0)
    feature_map_1.rz(params_1[1], 0)

    ansatz_1 = QuantumCircuit(1, name="Ansatz1")
    a_params_1 = [Parameter("theta1"), Parameter("theta2")]
    ansatz_1.rz(a_params_1[0], 0)
    ansatz_1.ry(a_params_1[1], 0)
    
    qc_1 = QuantumCircuit(1)
    qc_1.compose(feature_map_1, inplace=True)
    qc_1.compose(ansatz_1, inplace=True)
    
    estimator_qnn_1 = EstimatorQNN(
        circuit=qc_1,
        estimator=estimator,
        input_params=feature_map_1.parameters,
        weight_params=ansatz_1.parameters
    )
    
    # Train the model to find the final weights
    classifier1 = NeuralNetworkClassifier(
        estimator_qnn_1, 
        optimizer=COBYLA(maxiter=60), 
        initial_point=initial_weights
    )
    classifier1.fit(X_train, y_train)
    final_weights_1 = classifier1.weights
    
    # Calculate loss for initial and final points
    initial_loss_1 = calculate_loss_for_weights(estimator_qnn_1, X_train, y_train, initial_weights)
    final_loss_1 = calculate_loss_for_weights(estimator_qnn_1, X_train, y_train, final_weights_1)
    
    W1_1, W2_1, loss_1 = calculate_loss_landscape(estimator_qnn_1, X_train, y_train, grid_resolution=GRID_RESOLUTION, param_range=PARAM_RANGE)
    plot_loss_landscape(
        W1_1, W2_1, loss_1, 
        "Loss Landscape: 1-Qubit Angle Embedding (RY+RZ)",
        initial_weights=initial_weights, final_weights=final_weights_1,
        initial_loss=initial_loss_1, final_loss=final_loss_1
    )

    # ============================================================================
    # MODEL 2: 1-Qubit Amplitude Embedding
    # ============================================================================
    print("\n--- Processing Model 2: 1-Qubit Amplitude Embedding ---")

    X_train_amp = preprocess_for_amplitude_embedding(X_train)

    feature_map_2 = QuantumCircuit(1, name="FeatureMap2")
    theta = Parameter("input1")
    feature_map_2.ry(theta, 0)

    ansatz_2 = QuantumCircuit(1, name="Ansatz2")
    # Using the same ansatz structure as Model 1
    a_params_2 = [Parameter("theta1"), Parameter("theta2")]
    ansatz_2.rz(a_params_2[0],0)
    ansatz_2.ry(a_params_2[1],0)
    
    qc_2 = QuantumCircuit(1)
    qc_2.compose(feature_map_2, inplace=True)
    qc_2.compose(ansatz_2, inplace=True)

    estimator_qnn_2 = EstimatorQNN(
        circuit=qc_2,
        estimator=estimator,
        input_params=feature_map_2.parameters,
        weight_params=ansatz_2.parameters
    )

    # Train the model to find the final weights
    classifier2 = NeuralNetworkClassifier(
        estimator_qnn_2, 
        optimizer=COBYLA(maxiter=60), 
        initial_point=initial_weights
    )
    classifier2.fit(X_train_amp, y_train)
    final_weights_2 = classifier2.weights

    # Calculate loss for initial and final points
    initial_loss_2 = calculate_loss_for_weights(estimator_qnn_2, X_train_amp, y_train, initial_weights)
    final_loss_2 = calculate_loss_for_weights(estimator_qnn_2, X_train_amp, y_train, final_weights_2)

    W1_2, W2_2, loss_2 = calculate_loss_landscape(estimator_qnn_2, X_train_amp, y_train, grid_resolution=GRID_RESOLUTION, param_range=PARAM_RANGE)
    plot_loss_landscape(
        W1_2, W2_2, loss_2, 
        "Loss Landscape: 1-Qubit Amplitude Embedding",
        initial_weights=initial_weights, final_weights=final_weights_2,
        initial_loss=initial_loss_2, final_loss=final_loss_2
    )

    # ============================================================================
    # MODEL 4: 2-Qubit Custom Angle Embedding + RealAmplitudes
    # ============================================================================
    print("\n--- Processing Model 4: 2-Qubit Custom Angle Embedding ---")
    
    # Create 2-qubit custom angle embedding circuit
    feature_map_4 = QuantumCircuit(2)
    params_4 = [Parameter("input1"), Parameter("input2")]
    feature_map_4.ry(params_4[0], 0)
    feature_map_4.ry(params_4[1], 1)
    
    # RealAmplitudes(2, reps=1) has 4 parameters
    ansatz_4 = RealAmplitudes(2, reps=1)
    
    qc_4 = QuantumCircuit(2)
    qc_4.compose(feature_map_4, inplace=True)
    qc_4.compose(ansatz_4, inplace=True)

    estimator_qnn_4 = EstimatorQNN(
        circuit=qc_4,
        estimator=estimator,
        input_params=qc_4.parameters[:-4], # First 2 are inputs
        weight_params=qc_4.parameters[-4:]  # Last 4 are weights
    )

    # Train the model to find the final weights
    initial_weights_4 = algorithm_globals.random.random(ansatz_4.num_parameters)
    classifier4 = NeuralNetworkClassifier(
        estimator_qnn_4, 
        optimizer=COBYLA(maxiter=80), # Increased maxiter for more params
        initial_point=initial_weights_4
    )
    classifier4.fit(X_train, y_train)
    final_weights_4 = classifier4.weights

    # --- SLICE 1: Vary weights 1 and 2 ---
    slice_params_a = (0, 1)
    
    # Calculate loss for the initial and final points ON THIS SLICE
    initial_weights_slice_a = np.copy(final_weights_4)
    initial_weights_slice_a[slice_params_a[0]] = initial_weights_4[slice_params_a[0]]
    initial_weights_slice_a[slice_params_a[1]] = initial_weights_4[slice_params_a[1]]
    initial_loss_slice_a = calculate_loss_for_weights(estimator_qnn_4, X_train, y_train, initial_weights_slice_a)
    final_loss_4 = calculate_loss_for_weights(estimator_qnn_4, X_train, y_train, final_weights_4)

    # Calculate and plot the slice
    W1_4a, W2_4a, loss_4_slice_a = calculate_loss_slice(
        estimator_qnn_4, X_train, y_train, final_weights_4, 
        slice_params=slice_params_a, grid_resolution=GRID_RESOLUTION, param_range=PARAM_RANGE
    )
    plot_loss_landscape(
        W1_4a, W2_4a, loss_4_slice_a, 
        f"Loss Slice: Model 4 (Weights {slice_params_a[0]+1} and {slice_params_a[1]+1})",
        initial_weights=initial_weights_4, final_weights=final_weights_4, 
        initial_loss=initial_loss_slice_a, final_loss=final_loss_4,
        slice_params=slice_params_a
    )
    
    # --- SLICE 2: Vary weights 3 and 4 ---
    slice_params_b = (2, 3)
    
    # Calculate loss for the initial point ON THIS SLICE
    initial_weights_slice_b = np.copy(final_weights_4)
    initial_weights_slice_b[slice_params_b[0]] = initial_weights_4[slice_params_b[0]]
    initial_weights_slice_b[slice_params_b[1]] = initial_weights_4[slice_params_b[1]]
    initial_loss_slice_b = calculate_loss_for_weights(estimator_qnn_4, X_train, y_train, initial_weights_slice_b)

    # Calculate and plot the slice
    W1_4b, W2_4b, loss_4_slice_b = calculate_loss_slice(
        estimator_qnn_4, X_train, y_train, final_weights_4, 
        slice_params=slice_params_b, grid_resolution=GRID_RESOLUTION, param_range=PARAM_RANGE
    )
    plot_loss_landscape(
        W1_4b, W2_4b, loss_4_slice_b, 
        f"Loss Slice: Model 4 (Weights {slice_params_b[0]+1} and {slice_params_b[1]+1})",
        initial_weights=initial_weights_4, final_weights=final_weights_4, 
        initial_loss=initial_loss_slice_b, final_loss=final_loss_4,
        slice_params=slice_params_b
    )

    print("\n✅ Visualization complete.") 