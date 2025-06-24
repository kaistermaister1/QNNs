#!/usr/bin/env python3
"""
3D Loss Landscape Visualization for Babyshora QNN Models
=========================================================

This script visualizes the 3D loss landscapes for the 4 different
QNN architectures from the SHORA project.

For each model, this script visualizes 2D slices of the loss landscape,
keeping other parameters fixed at their trained optimal values.
"""

import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit_algorithms.optimizers import COBYLA, ADAM
from qiskit_algorithms.utils import algorithm_globals
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import Sampler
from qiskit.circuit.library import RealAmplitudes

import warnings
import argparse
import os
from tqdm.auto import tqdm

warnings.filterwarnings('ignore')
PLOT_DIR = "plots/loss_landscapes"
os.makedirs(PLOT_DIR, exist_ok=True)

# Configuration
RANDOM_SEED = 42
GRID_RESOLUTION = 15
PARAM_RANGE = (-np.pi, np.pi)
MAX_ITER = 100
algorithm_globals.random_seed = RANDOM_SEED
np.random.seed(algorithm_globals.random_seed)

# --- Data Generation (Consistent across models) ---
def get_shora_data():
    """Generates the dataset for numbers 2-15."""
    data = [[int(bit) for bit in bin(i)[2:].zfill(4)] for i in range(2, 16)]
    labels_binary = ["0010", "0011", "0010", "0101", "0010", "0111", "0010", "0011", "0010", "1011", "0010", "1101", "0010", "0011"]
    labels_decimal = [int(label, 2) for label in labels_binary]
    
    lowest_factor_map = { n: int(labels_binary[n-2], 2) for n in range(2, 16) }
    labels_one_hot = np.eye(16)[[lowest_factor_map[n] for n in range(2, 16)]]

    return data, labels_binary, labels_decimal, labels_one_hot

# --- Loss Calculation ---

def calculate_loss_for_qnn(qnn, X, y, weights, loss_func):
    """Calculates the average loss over a dataset for a given QNN and loss function."""
    total_loss = 0
    for x_i, y_i in zip(X, y):
        pred = qnn.forward([x_i], weights)
        total_loss += loss_func(pred, y_i)
    return total_loss / len(X)

# --- Landscape Calculation & Plotting ---

def calculate_loss_slice(qnn, X, y, trained_weights, loss_func, slice_params, param_range, grid_resolution):
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
                loss_grid[i, j] = calculate_loss_for_qnn(qnn, X, y, weights, loss_func)
                pbar.update(1)
    return W1, W2, loss_grid

def plot_loss_landscape(W1, W2, loss_grid, title, plot_filename, initial_weights, final_weights, initial_loss, final_loss, slice_params):
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
    
    plt.savefig(plot_filename, dpi=120, bbox_inches='tight')
    print(f"   ✅ Saved plot to {plot_filename}")
    plt.close()

def process_and_plot_model(model_name, qnn, loss_func, X_train, y_train, slice_pairs):
    """Train model, calculate loss slices, and plot them."""
    print(f"\n--- Processing {model_name} ---")
    
    initial_weights = algorithm_globals.random.random(qnn.num_weights)
    
    # Objective function for optimizer
    def objective(weights):
        return calculate_loss_for_qnn(qnn, X_train, y_train, weights, loss_func)
        
    optimizer = COBYLA(maxiter=MAX_ITER)
    print(f"   Training model ({MAX_ITER} iterations)...")
    result = optimizer.minimize(objective, initial_weights)
    final_weights = result.x
    final_loss = result.fun

    for slice_params in slice_pairs:
        # Calculate loss for the initial point projected on this slice
        initial_weights_slice = np.copy(final_weights)
        initial_weights_slice[slice_params[0]] = initial_weights[slice_params[0]]
        initial_weights_slice[slice_params[1]] = initial_weights[slice_params[1]]
        initial_loss_slice = calculate_loss_for_qnn(qnn, X_train, y_train, initial_weights_slice, loss_func)

        W1, W2, loss_slice = calculate_loss_slice(
            qnn, X_train, y_train, final_weights, loss_func,
            slice_params=slice_params, grid_resolution=GRID_RESOLUTION, param_range=PARAM_RANGE
        )
        plot_title = f"{model_name}\n(Slice: Weights {slice_params[0]+1} & {slice_params[1]+1})"
        plot_filename = f"{PLOT_DIR}/shora_loss_{model_name.replace(' ', '_')}_slice_{slice_params[0]}_{slice_params[1]}.png"
        plot_loss_landscape(
            W1, W2, loss_slice, plot_title, plot_filename,
            initial_weights=initial_weights, final_weights=final_weights,
            initial_loss=initial_loss_slice, final_loss=final_loss,
            slice_params=slice_params
        )


# --- Loss Functions ---
def hamming_loss(pred_probs, target_label_str):
    probs = np.asarray(pred_probs).flatten()
    mode_idx = np.argmax(probs)
    mode_bitstring = format(mode_idx, f'0{len(target_label_str)}b')
    return sum(c1 != c2 for c1, c2 in zip(mode_bitstring, target_label_str))

def squared_error_loss(pred_probs, target_decimal):
    probs = np.asarray(pred_probs).flatten()
    binary_vector = (probs >= 0.25).astype(int)
    decimal_value = sum(bit * (2**(len(binary_vector) - 1 - i)) for i, bit in enumerate(binary_vector))
    return (decimal_value - target_decimal) ** 2

def cross_entropy_loss(pred_probs, target_one_hot):
    probs = np.asarray(pred_probs).flatten()
    return float(-np.sum(target_one_hot * np.log(probs + 1e-10)))


# --- Main Script ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate 3D loss landscape visualizations for Babyshora QNN models.")
    parser.add_argument('--model', type=int, choices=range(1, 5), help="Specify a single model number to plot (1-4).")
    args = parser.parse_args()

    X, y_bin, y_dec, y_onehot = get_shora_data()
    sampler = Sampler()
    models_to_run = [args.model] if args.model else range(1, 5)

    if 1 in models_to_run:
        qc = QuantumCircuit(4)
        inputs = [Parameter(f"i{i}") for i in range(4)]
        weights = [Parameter(f"w{i}") for i in range(12)]
        qc.ry((np.pi/2)*inputs[0], 0); qc.ry((np.pi/2)*inputs[1], 1); qc.ry((np.pi/2)*inputs[2], 2); qc.ry((np.pi/2)*inputs[3], 3)
        qc.barrier()
        qc.rz(weights[0],0); qc.rz(weights[1],1); qc.rz(weights[2],2); qc.rz(weights[3],3)
        for i in range(2):
            qc.cx(0,1); qc.cx(1,2); qc.cx(2,3); qc.cx(3,0)
            qc.rz(weights[4+i*4],0); qc.rz(weights[5+i*4],1); qc.rz(weights[6+i*4],2); qc.rz(weights[7+i*4],3)
        qnn = SamplerQNN(circuit=qc, sampler=sampler, input_params=inputs, weight_params=weights)
        process_and_plot_model("Model 1", qnn, hamming_loss, X, y_bin, [(0, 1), (4, 5), (8, 9)])

    if 2 in models_to_run:
        qc = QuantumCircuit(4)
        inputs = [Parameter(f"i{i}") for i in range(4)]
        weights = [Parameter(f"w{i}") for i in range(16)]
        for i in range(4): qc.ry(np.pi*inputs[i], i)
        qc.rzz(np.pi*inputs[0]*inputs[1],0,1); qc.rzz(np.pi*inputs[2]*inputs[3],2,3)
        qc.cx(0,1); qc.cx(2,3); qc.cx(1,2); qc.barrier()
        qc.rz(weights[0],0); qc.rz(weights[1],1); qc.rz(weights[2],2); qc.rz(weights[3],3)
        for i in range(3):
            qc.cx(0,1); qc.cx(1,2); qc.cx(2,3); qc.cx(3,0)
            qc.rz(weights[4+i*4],0); qc.rz(weights[5+i*4],1); qc.rz(weights[6+i*4],2); qc.rz(weights[7+i*4],3)
        qnn = SamplerQNN(circuit=qc, sampler=sampler, input_params=inputs, weight_params=weights)
        process_and_plot_model("Model 2.1", qnn, hamming_loss, X, y_bin, [(0, 1), (6, 7), (12, 13)])
        
    if 3 in models_to_run:
        qc = QuantumCircuit(2)
        inputs = [Parameter(f"i{i}") for i in range(4)]
        weights = [Parameter(f"w{i}") for i in range(6)]
        qc.ry((np.pi/2)*(inputs[0]-inputs[1]), 0); qc.ry((np.pi/2)*(inputs[2]-inputs[3]), 1)
        qc.ry(weights[0],0); qc.ry(weights[1],1); qc.rz(weights[2],0); qc.rz(weights[3],1); qc.rx(weights[4],0); qc.rx(weights[5],1)
        qnn = SamplerQNN(circuit=qc, sampler=sampler, input_params=inputs, weight_params=weights)
        process_and_plot_model("Model 2.2", qnn, squared_error_loss, X, y_dec, [(0, 1), (2, 3), (4, 5)])

    if 4 in models_to_run:
        qc = QuantumCircuit(4)
        inputs = ParameterVector("x", 4)
        ansatz = RealAmplitudes(4, reps=2)
        weights = list(ansatz.parameters)
        for i in range(4): qc.ry(np.pi * inputs[i], i)
        qc.compose(ansatz, inplace=True)
        qnn = SamplerQNN(circuit=qc, sampler=sampler, input_params=list(inputs), weight_params=weights)
        process_and_plot_model("Model i", qnn, cross_entropy_loss, X, y_onehot, [(0, 1), (4, 5), (8, 9)])

    print("\n\n✨ Landscape visualization complete.") 