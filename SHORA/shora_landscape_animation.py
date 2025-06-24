#!/usr/bin/env python3
"""
Loss Landscape Animation for Babyshora QNN Models
==================================================

This script creates an animation of the COBYLA optimizer's path across
a 2D slice of the loss landscape for the Babyshora models.

It first computes the landscape for a given model, then runs the optimizer
with a callback to record the weights at each step, generates frames for
the animation, and finally compiles them into a GIF.
"""

import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.utils import algorithm_globals
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import Sampler
from qiskit.circuit.library import RealAmplitudes

import warnings
import argparse
import os
from tqdm.auto import tqdm
import imageio
import shutil

warnings.filterwarnings('ignore')
PLOT_DIR = "plots/animations"
os.makedirs(PLOT_DIR, exist_ok=True)

# --- Configuration ---
RANDOM_SEED = 42
GRID_RESOLUTION = 20
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

def calculate_average_loss(qnn, X, y, weights, loss_func):
    """Calculates the average loss over the dataset."""
    total_loss = 0
    for x_i, y_i in zip(X, y):
        pred = qnn.forward([x_i], weights)
        total_loss += loss_func(pred, y_i)
    return total_loss / len(X)

# --- Landscape and Animation Core Logic ---

def calculate_loss_landscape_slice(qnn, X, y, loss_func, trained_weights, slice_params):
    """Calculates the 2D slice of the loss landscape."""
    w1_vals = np.linspace(PARAM_RANGE[0], PARAM_RANGE[1], GRID_RESOLUTION)
    w2_vals = np.linspace(PARAM_RANGE[0], PARAM_RANGE[1], GRID_RESOLUTION)
    W1, W2 = np.meshgrid(w1_vals, w2_vals)
    loss_grid = np.zeros(W1.shape)
    
    print("Calculating loss landscape...")
    for i in tqdm(range(GRID_RESOLUTION)):
        for j in range(GRID_RESOLUTION):
            weights = np.copy(trained_weights)
            weights[slice_params[0]] = W1[i, j]
            weights[slice_params[1]] = W2[i, j]
            loss_grid[i, j] = calculate_average_loss(qnn, X, y, weights, loss_func)
    return W1, W2, loss_grid

def generate_animation_for_model(model_name, qnn, loss_func, X_data, y_data, slice_params):
    """Main execution function for a single model."""
    print(f"\n--- Starting Animation Generation for {model_name} ---")
    
    # Setup temporary directory for frames
    FRAMES_DIR = os.path.join(PLOT_DIR, f"frames_{model_name.replace(' ', '_')}")
    os.makedirs(FRAMES_DIR, exist_ok=True)

    # 1. Pre-training to find the optimal point to center the landscape
    print("Pre-training to find optimal weights...")
    initial_weights = algorithm_globals.random.random(qnn.num_weights)
    objective_fn = lambda w: calculate_average_loss(qnn, X_data, y_data, w, loss_func)
    optimizer = COBYLA(maxiter=MAX_ITER)
    result = optimizer.minimize(objective_fn, initial_weights)
    final_weights = result.x
    print(f"Optimal point found with loss: {result.fun:.4f}")

    # 2. Calculate the loss landscape around the final weights
    W1, W2, loss_grid = calculate_loss_landscape_slice(qnn, X_data, y_data, loss_func, final_weights, slice_params)

    # 3. Run the optimization again, recording history
    print("Re-running training to capture weight history...")
    weight_history = []
    def callback_recorder(params):
        weight_history.append(np.copy(params))
        
    optimizer_with_callback = COBYLA(maxiter=MAX_ITER, callback=callback_recorder)
    optimizer_with_callback.minimize(objective_fn, initial_weights)
    
    # 4. Generate animation frames
    print(f"Generating {len(weight_history)} animation frames...")
    for i, weights in enumerate(tqdm(weight_history)):
        fig, ax = plt.subplots(figsize=(10, 8))
        contour = ax.contourf(W1, W2, loss_grid, levels=20, cmap='viridis', alpha=0.8)
        fig.colorbar(contour, label='Loss')
        
        path_x = [w[slice_params[0]] for w in weight_history[:i+1]]
        path_y = [w[slice_params[1]] for w in weight_history[:i+1]]
        ax.plot(path_x, path_y, 'w-', linewidth=2, alpha=0.7, label='Optimizer Path')
        ax.scatter(path_x[-1], path_y[-1], c='red', s=100, zorder=5, marker='*', label='Current Position')
        ax.scatter(path_x[0], path_y[0], c='lime', s=100, zorder=5, marker='o', label='Start Position')
        
        ax.set_title(f'{model_name} Optimizer Path - Iteration {i+1}', fontsize=16)
        ax.set_xlabel(f'Weight {slice_params[0]+1}', fontsize=12)
        ax.set_ylabel(f'Weight {slice_params[1]+1}', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3); ax.set_xlim(PARAM_RANGE); ax.set_ylim(PARAM_RANGE)
        
        frame_path = os.path.join(FRAMES_DIR, f'frame_{i:03d}.png')
        plt.savefig(frame_path, dpi=90, bbox_inches='tight')
        plt.close(fig)

    # 5. Compile frames into a GIF
    animation_filename = f"shora_animation_{model_name.replace(' ', '_')}.gif"
    animation_path = os.path.join(PLOT_DIR, animation_filename)
    print(f"Compiling frames into {animation_filename}...")
    with imageio.get_writer(animation_path, mode='I', duration=150, loop=0) as writer:
        for i in range(len(weight_history)):
            frame_path = os.path.join(FRAMES_DIR, f'frame_{i:03d}.png')
            writer.append_data(imageio.imread(frame_path))
    
    # 6. Clean up frames
    shutil.rmtree(FRAMES_DIR)
    print(f"✨ Animation complete! Saved to {animation_path}")

# --- Main Script ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate animations of optimizer paths for Babyshora QNN models.")
    parser.add_argument('--model', type=int, choices=range(1, 5), help="Specify a single model number to animate (1-4).")
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
        generate_animation_for_model("Model 1", qnn, hamming_loss, X, y_bin, slice_params=(0, 1))

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
        generate_animation_for_model("Model 2.1", qnn, hamming_loss, X, y_bin, slice_params=(0, 1))
        
    if 3 in models_to_run:
        qc = QuantumCircuit(2)
        inputs = [Parameter(f"i{i}") for i in range(4)]
        weights = [Parameter(f"w{i}") for i in range(6)]
        qc.ry((np.pi/2)*(inputs[0]-inputs[1]), 0); qc.ry((np.pi/2)*(inputs[2]-inputs[3]), 1)
        qc.ry(weights[0],0); qc.ry(weights[1],1); qc.rz(weights[2],0); qc.rz(weights[3],1); qc.rx(weights[4],0); qc.rx(weights[5],1)
        qnn = SamplerQNN(circuit=qc, sampler=sampler, input_params=inputs, weight_params=weights)
        generate_animation_for_model("Model 2.2", qnn, squared_error_loss, X, y_dec, slice_params=(0, 1))

    if 4 in models_to_run:
        qc = QuantumCircuit(4)
        inputs = ParameterVector("x", 4)
        ansatz = RealAmplitudes(4, reps=2)
        weights = list(ansatz.parameters)
        for i in range(4): qc.ry(np.pi * inputs[i], i)
        qc.compose(ansatz, inplace=True)
        qnn = SamplerQNN(circuit=qc, sampler=sampler, input_params=list(inputs), weight_params=weights)
        generate_animation_for_model("Model i", qnn, cross_entropy_loss, X, y_onehot, slice_params=(0, 1))

    print("\n\n✨ All requested animations are complete.") 