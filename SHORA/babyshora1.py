# Standard libraries
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
from IPython.display import clear_output
import random

# Sklearn
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# Qiskit Core
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit.primitives import StatevectorSampler as Sampler, StatevectorEstimator as Estimator
from qiskit.quantum_info import SparsePauliOp

# Qiskit Algorithms
from qiskit_algorithms.optimizers import COBYLA, L_BFGS_B, ADAM
from qiskit_algorithms.utils import algorithm_globals

# Qiskit Machine Learning
from qiskit_machine_learning.algorithms.classifiers import VQC, NeuralNetworkClassifier
from qiskit_machine_learning.algorithms.regressors import NeuralNetworkRegressor, VQR
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
from qiskit_machine_learning.circuit.library import QNNCircuit, RawFeatureVector
from qiskit_machine_learning.utils import algorithm_globals as ml_algorithm_globals

# Define a single circuit for feature map and ansatz
qc = QuantumCircuit(4)
layers = 2
input_params = [Parameter(f"input{i}") for i in range(4)]
weight_params = [Parameter(f"weight{i}") for i in range(4+4*layers)]

# Feature map
qc.ry((np.pi/2)*input_params[0], 0)
qc.ry((np.pi/2)*input_params[1], 1)
qc.ry((np.pi/2)*input_params[2], 2)
qc.ry((np.pi/2)*input_params[3], 3)
qc.barrier()

# Ansatz
qc.rz(weight_params[0], 0)
qc.rz(weight_params[1], 1)
qc.rz(weight_params[2], 2)
qc.rz(weight_params[3], 3)
for i in range(layers):
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(2, 3)
    qc.cx(3, 0)
    qc.rz(weight_params[4+i*4], 0)
    qc.rz(weight_params[5+i*4], 1)
    qc.rz(weight_params[6+i*4], 2)
    qc.rz(weight_params[7+i*4], 3)

# Create plots directory if it doesn't exist
os.makedirs("plots", exist_ok=True)

# Draw and save the circuit
qc.draw(output="mpl", style="clifford", fold=20)
plt.suptitle("Combined Circuit")
plt.savefig("plots/babyshora1circuit.png", dpi=150, bbox_inches='tight')
plt.close()

# Convert digits to binary, [x1,x2,x3,x4] - starting from 2 to exclude 0 and 1
data = [[int(bit) for bit in bin(i)[2:].zfill(4)] for i in range(2, 16)]
# Labels are the binary representation of the lowest prime factor for each number 2-15
labels = ["0010", "0011", "0010", "0101", "0010", "0111", "0010", "0011", "0010", "1011", "0010", "1101", "0010", "0011"]

# Split into training and testing
random.seed(24)  # For reproducible results
idx_15 = 13  # 15 is at index 13 in our data (since we start from 2)
other_indices = [i for i in range(len(labels)) if i != idx_15]
test_indices_other = random.sample(other_indices, 3)
test_indices = test_indices_other + [idx_15]
train_indices = [i for i in range(len(labels)) if i not in test_indices]

# Create training and testing sets
train_data = [data[i] for i in train_indices]
train_labels = [labels[i] for i in train_indices]
test_data = [data[i] for i in test_indices]
test_labels = [labels[i] for i in test_indices]

print(f"Training data: {train_data}")
print(f"Training labels: {train_labels}")
print(f"Testing data: {test_data}")
print(f"Testing labels: {test_labels}")

# --- SamplerQNN Training ---
sampler = Sampler()

# Define SamplerQNN
sampler_qnn = SamplerQNN(
    circuit=qc,
    sampler=sampler,
    input_params=input_params,
    weight_params=weight_params,
)

# Set random values for weight parameters
weights = algorithm_globals.random.random(len(weight_params))

# Define loss function - HARD MODE VERSION
def hamming_loss(sampler_qnn_forward, target_label):
    """
    Hard mode Hamming loss: Only considers the most probable bitstring (mode).
    Returns the Hamming distance between the predicted mode and target.
    """
    probs = np.array(sampler_qnn_forward).reshape(-1)
    
    # Get the most probable bitstring (mode)
    mode_idx = np.argmax(probs)
    mode_bitstring = format(mode_idx, f'0{4}b')  # Convert to binary string
    
    # Calculate Hamming distance between mode and target
    hamming_dist = sum(c1 != c2 for c1, c2 in zip(mode_bitstring, target_label))
    
    return hamming_dist

# Initiate COBYLA optimizer
optimizer = COBYLA(maxiter=100)

# Build callables for COBYLA with batch training
def batch_objective_function(weights):
    """Compute average loss over all training samples"""
    total_loss = 0
    for train_input, train_label in zip(train_data, train_labels):
        output = sampler_qnn.forward(np.array(train_input), weights)
        loss = hamming_loss(output, train_label)
        total_loss += loss
    return total_loss / len(train_data)  # Return average loss

# Training with progress tracking
loss_history = []
iteration_count = [0]

def objective_with_tracking(weights):
    """Objective function with progress tracking for COBYLA"""
    loss = batch_objective_function(weights)
    loss_history.append(loss)
    iteration_count[0] += 1
    
    if iteration_count[0] % 10 == 0:
        print(f"  Iteration {iteration_count[0]}: Average Loss = {loss:.4f}")
    
    return loss

# Training loop
print("\n" + "="*50)
print("BATCH TRAINING PHASE")
print("="*50)
print(f"Training on {len(train_data)} samples simultaneously...")
print("Training data:")
for i, (inp, label) in enumerate(zip(train_data, train_labels)):
    print(f"  Sample {i+1}: {inp} → {label}")

start_time = time.time()

# Single optimization over ALL training samples
print(f"\nStarting COBYLA optimization...")
result = optimizer.minimize(objective_with_tracking, weights)
weights = result.x

training_time = time.time() - start_time
print(f"\nBatch training completed in {training_time:.2f} seconds")
print(f"Final average training loss: {loss_history[-1]:.4f}")
print(f"Total iterations: {len(loss_history)}")

# Evaluate final performance on training set
print("\n" + "="*50)
print("TRAINING SET EVALUATION")
print("="*50)

train_results = []
for i, (train_input, train_label) in enumerate(zip(train_data, train_labels)):
    output = sampler_qnn.forward(np.array(train_input), weights)
    loss = hamming_loss(output, train_label)
    
    # Get predicted bitstring (mode)
    probs = np.array(output).reshape(-1)
    pred_idx = np.argmax(probs)
    pred_bitstring = format(pred_idx, '04b')
    
    train_results.append({
        'input': train_input,
        'target': train_label,
        'predicted': pred_bitstring,
        'loss': loss,
        'confidence': probs[pred_idx]
    })
    
    print(f"Train {i+1}: Input {train_input} → Target: {train_label}, Predicted: {pred_bitstring}")
    print(f"  Loss: {loss:.4f}, Confidence: {probs[pred_idx]:.4f}")

# Testing phase
print("\n" + "="*50)
print("TESTING PHASE")
print("="*50)

test_results = []
for i, (test_input, test_label) in enumerate(zip(test_data, test_labels)):
    # Forward pass
    output = sampler_qnn.forward(np.array(test_input), weights)
    loss = hamming_loss(output, test_label)
    
    # Get predicted bitstring (mode)
    probs = np.array(output).reshape(-1)
    pred_idx = np.argmax(probs)
    pred_bitstring = format(pred_idx, '04b')
    
    test_results.append({
        'input': test_input,
        'target': test_label,
        'predicted': pred_bitstring,
        'loss': loss,
        'confidence': probs[pred_idx]
    })
    
    print(f"Test {i+1}: Input {test_input} → Target: {test_label}, Predicted: {pred_bitstring}")
    print(f"  Loss: {loss:.4f}, Confidence: {probs[pred_idx]:.4f}")

# Results summary
print("\n" + "="*50)
print("RESULTS SUMMARY")
print("="*50)

# Training accuracy
train_correct = sum(1 for r in train_results if r['predicted'] == r['target'])
train_accuracy = train_correct / len(train_results)
avg_train_loss = np.mean([r['loss'] for r in train_results])

# Test accuracy
test_correct = sum(1 for r in test_results if r['predicted'] == r['target'])
test_accuracy = test_correct / len(test_results)
avg_test_loss = np.mean([r['loss'] for r in test_results])

print(f"Training Accuracy: {train_correct}/{len(train_results)} = {train_accuracy:.2%}")
print(f"Average Training Loss: {avg_train_loss:.4f}")
print(f"Test Accuracy: {test_correct}/{len(test_results)} = {test_accuracy:.2%}")
print(f"Average Test Loss: {avg_test_loss:.4f}")
print(f"Training Time: {training_time:.2f} seconds")
print(f"Total Optimization Iterations: {len(loss_history)}")

# Plot results
plt.figure(figsize=(15, 10))

# Plot 1: Training loss curve
plt.subplot(2, 2, 1)
plt.plot(range(1, len(loss_history) + 1), loss_history, 'b-', linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('Average Batch Loss')
plt.title('Batch Training Loss History')
plt.grid(True, alpha=0.3)

# Plot 2: Training results
plt.subplot(2, 2, 2)
train_losses = [r['loss'] for r in train_results]
train_colors = ['green' if r['predicted'] == r['target'] else 'red' for r in train_results]
plt.bar(range(len(train_results)), train_losses, color=train_colors, alpha=0.7)
plt.xlabel('Training Sample')
plt.ylabel('Hamming Loss')
plt.title('Training Results (Green=Correct, Red=Wrong)')
plt.xticks(range(len(train_results)), [f"{i+1}" for i in range(len(train_results))])
plt.grid(True, alpha=0.3)

# Plot 3: Test results
plt.subplot(2, 2, 3)
test_losses = [r['loss'] for r in test_results]
test_colors = ['green' if r['predicted'] == r['target'] else 'red' for r in test_results]
plt.bar(range(len(test_results)), test_losses, color=test_colors, alpha=0.7)
plt.xlabel('Test Sample')
plt.ylabel('Hamming Loss')
plt.title('Test Results (Green=Correct, Red=Wrong)')
plt.xticks(range(len(test_results)), [f"{i+1}" for i in range(len(test_results))])
plt.grid(True, alpha=0.3)

# Plot 4: Accuracy comparison
plt.subplot(2, 2, 4)
accuracies = [train_accuracy, test_accuracy]
labels = ['Training', 'Testing']
colors = ['blue', 'orange']
bars = plt.bar(labels, accuracies, color=colors, alpha=0.7)
plt.ylabel('Accuracy')
plt.title('Training vs Test Accuracy')
plt.ylim(0, 1)
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{acc:.1%}', ha='center', va='bottom', fontweight='bold')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("plots/babyshora1.png", dpi=150, bbox_inches='tight')
plt.show()

print(f"\nPlots saved to plots/babyshora1.png")
