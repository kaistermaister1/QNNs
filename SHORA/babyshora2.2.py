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

# Simple feature map, 2 qubits

# OPTIMIZER CHOICE - Change this to switch optimizers
OPTIMIZER_CHOICE = "COBYLA"  # Options: "ADAM" or "COBYLA"

# Define a single circuit for feature map and ansatz
n_qubits = 2
qc = QuantumCircuit(n_qubits)

input_params = [Parameter(f"input{i}") for i in range(4)]
weight_params = [Parameter(f"weight{i}") for i in range(6)]

# Feature map
qc.ry((np.pi/2) * (input_params[0] - input_params[1]), 0)
qc.ry((np.pi/2) * (input_params[2] - input_params[3]), 1)

# Ansatz
qc.ry(weight_params[0], 0)
qc.ry(weight_params[1], 1)
qc.rz(weight_params[2], 0)
qc.rz(weight_params[3], 1)
qc.rx(weight_params[4], 0)
qc.rx(weight_params[5], 1)

# Draw and save the circuit
qc.draw(output="mpl", style="clifford", fold=20)
plt.suptitle("Combined Circuit")
plt.savefig("plots/babyshora2.2circuit.png", dpi=150, bbox_inches='tight')
plt.close()

# Convert digits to binary, [x1,x2,x3,x4] - starting from 2 to exclude 0 and 1
data = [[int(bit) for bit in bin(i)[2:].zfill(4)] for i in range(2, 16)]

# Labels are now the decimal value of the lowest prime factor for each number 2-15
# Converting binary strings to decimal values
binary_labels = ["0010", "0011", "0010", "0101", "0010", "0111", "0010", "0011", "0010", "1011", "0010", "1101", "0010", "0011"]
labels = [int(label, 2) for label in binary_labels]  # Convert binary strings to decimal

# Split into training and testing
# random.seed(24)  # For reproducible results
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

print(f"Training samples: {len(train_data)}, Testing samples: {len(test_data)}")

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

# Define new prediction function using probability thresholding
def predict_with_threshold(sampler_qnn_forward, threshold=0.25):
    """
    Convert SamplerQNN output to decimal using probability thresholding.
    For a 2-qubit system: 4 probabilities [p00, p01, p10, p11]
    If p_i >= threshold, set bit to 1, otherwise 0.
    This creates a 4-bit binary string, then convert to decimal.
    
    Example: [0.64, 0.42, 0.11, 0.01] with threshold=0.25
    -> [1, 1, 0, 0] -> binary "1100" -> decimal 12
    """
    probs = np.array(sampler_qnn_forward).reshape(-1)
    
    # Apply threshold to get binary vector (each probability becomes a bit)
    binary_vector = (probs >= threshold).astype(int)
    
    # Convert 4-bit binary vector to decimal (MSB first)
    # binary_vector = [bit3, bit2, bit1, bit0] representing bits from left to right
    decimal_value = 0
    for i, bit in enumerate(binary_vector):
        if bit == 1:
            decimal_value += bit * (2**(len(binary_vector) - 1 - i))
    
    return decimal_value

# Define new loss function - squared error
def squared_error_loss(sampler_qnn_forward, target_label):
    """
    Squared error loss between predicted decimal and target decimal.
    """
    predicted_decimal = predict_with_threshold(sampler_qnn_forward)
    return (predicted_decimal - target_label) ** 2

# Initiate optimizer based on choice
if OPTIMIZER_CHOICE == "ADAM":
    optimizer = ADAM(maxiter=200, lr=0.01)
    optimizer_name = "ADAM"
elif OPTIMIZER_CHOICE == "COBYLA":
    optimizer = COBYLA(maxiter=200)
    optimizer_name = "COBYLA"
else:
    raise ValueError(f"Unknown optimizer choice: {OPTIMIZER_CHOICE}. Use 'ADAM' or 'COBYLA'")

print(f"Using {optimizer_name} optimizer")

# Build callables for optimizer with batch training
def batch_objective_function(weights):
    """Compute average loss over all training samples"""
    total_loss = 0
    for train_input, train_label in zip(train_data, train_labels):
        output = sampler_qnn.forward(np.array(train_input), weights)
        loss = squared_error_loss(output, train_label)
        total_loss += loss
    return total_loss / len(train_data)  # Return average loss

# Training with progress tracking
loss_history = []
iteration_count = [0]

def objective_with_tracking(weights):
    """Objective function with progress tracking"""
    loss = batch_objective_function(weights)
    loss_history.append(loss)
    iteration_count[0] += 1
    
    if iteration_count[0] % 25 == 0:  # Reduced frequency
        print(f"  Iteration {iteration_count[0]}: Average Loss = {loss:.4f}")
    
    return loss

# Training loop
print(f"\nTraining with {optimizer_name} optimizer on {len(train_data)} samples...")

start_time = time.time()

# Single optimization over ALL training samples
result = optimizer.minimize(objective_with_tracking, weights)
weights = result.x

training_time = time.time() - start_time
print(f"Training completed in {training_time:.2f}s | Final loss: {loss_history[-1]:.4f} | Iterations: {len(loss_history)}")

# Evaluate final performance on training set
print("\nEvaluating training set...")

train_results = []
train_integers = []  # Store the actual integers for plotting
for i, (train_input, train_label) in enumerate(zip(train_data, train_labels)):
    output = sampler_qnn.forward(np.array(train_input), weights)
    loss = squared_error_loss(output, train_label)
    
    # Get predicted decimal value
    predicted_decimal = predict_with_threshold(output)
    
    # Get the actual integer (convert from train_indices back to integers 2-15)
    actual_integer = train_indices[i] + 2
    train_integers.append(actual_integer)
    
    train_results.append({
        'input': train_input,
        'target': train_label,
        'predicted': predicted_decimal,
        'loss': loss,
        'confidence': np.max(np.array(output)),
        'integer': actual_integer
    })

# Testing phase
print("Evaluating test set...")

test_results = []
test_integers = []  # Store the actual integers for plotting
for i, (test_input, test_label) in enumerate(zip(test_data, test_labels)):
    # Forward pass
    output = sampler_qnn.forward(np.array(test_input), weights)
    loss = squared_error_loss(output, test_label)
    
    # Get predicted decimal value
    predicted_decimal = predict_with_threshold(output)
    
    # Get the actual integer (convert from test_indices back to integers 2-15)
    actual_integer = test_indices[i] + 2
    test_integers.append(actual_integer)
    
    test_results.append({
        'input': test_input,
        'target': test_label,
        'predicted': predicted_decimal,
        'loss': loss,
        'confidence': np.max(np.array(output)),
        'integer': actual_integer
    })

# Results summary
print("\n--- RESULTS ---")

# Training accuracy
train_correct = sum(1 for r in train_results if r['predicted'] == r['target'])
train_accuracy = train_correct / len(train_results)
avg_train_loss = np.mean([r['loss'] for r in train_results])

# Test accuracy
test_correct = sum(1 for r in test_results if r['predicted'] == r['target'])
test_accuracy = test_correct / len(test_results)
avg_test_loss = np.mean([r['loss'] for r in test_results])

print(f"Train: {train_accuracy:.1%} ({train_correct}/{len(train_results)}) | Test: {test_accuracy:.1%} ({test_correct}/{len(test_results)})")
print(f"Train Loss: {avg_train_loss:.3f} | Test Loss: {avg_test_loss:.3f}")

# Plot results
plt.figure(figsize=(18, 12))

# Plot 1: Training loss curve
plt.subplot(2, 3, 1)
plt.plot(range(1, len(loss_history) + 1), loss_history, 'b-', linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('Average Batch Loss')
plt.title('Batch Training Loss History')
plt.grid(True, alpha=0.3)

# Plot 2: Training results
plt.subplot(2, 3, 2)
train_losses = [r['loss'] for r in train_results]
train_colors = ['green' if r['predicted'] == r['target'] else 'red' for r in train_results]
plt.bar(range(len(train_results)), train_losses, color=train_colors, alpha=0.7)
plt.xlabel('Integer')
plt.ylabel('Squared Error Loss')
plt.title('Training Results (Green=Correct, Red=Wrong)')
plt.xticks(range(len(train_results)), train_integers)
plt.grid(True, alpha=0.3)

# Plot 3: Test results
plt.subplot(2, 3, 3)
test_losses = [r['loss'] for r in test_results]
test_colors = ['green' if r['predicted'] == r['target'] else 'red' for r in test_results]
plt.bar(range(len(test_results)), test_losses, color=test_colors, alpha=0.7)
plt.xlabel('Integer')
plt.ylabel('Squared Error Loss')
plt.title('Test Results (Green=Correct, Red=Wrong)')
plt.xticks(range(len(test_results)), test_integers)
plt.grid(True, alpha=0.3)

# Plot 4: Accuracy comparison
plt.subplot(2, 3, 4)
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

# Plot 5: Predicted vs Actual Factors
plt.subplot(2, 3, 5)
# Training data points
train_predicted = [r['predicted'] for r in train_results]
train_actual = [r['target'] for r in train_results]
plt.scatter(train_actual, train_predicted, color='blue', alpha=0.7, s=60, label='Training', marker='o')

# Test data points
test_predicted = [r['predicted'] for r in test_results]
test_actual = [r['target'] for r in test_results]
plt.scatter(test_actual, test_predicted, color='red', alpha=0.7, s=60, label='Test', marker='s')

# Perfect prediction line
min_factor = min(min(train_actual), min(test_actual))
max_factor = max(max(train_actual), max(test_actual))
plt.plot([min_factor, max_factor], [min_factor, max_factor], 'k--', alpha=0.5, label='Perfect')

plt.xlabel('Actual Lowest Prime Factor')
plt.ylabel('Predicted Lowest Prime Factor')
plt.title('Predicted vs Actual Factors')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 6: Factor Distribution
plt.subplot(2, 3, 6)
all_integers = train_integers + test_integers
all_predicted = train_predicted + test_predicted
all_actual = train_actual + test_actual

# Create bar chart showing actual vs predicted for each integer
width = 0.35
x_pos = np.arange(len(all_integers))

plt.bar(x_pos - width/2, all_actual, width, label='Actual', alpha=0.7, color='lightblue')
plt.bar(x_pos + width/2, all_predicted, width, label='Predicted', alpha=0.7, color='lightcoral')

plt.xlabel('Integer')
plt.ylabel('Lowest Prime Factor')
plt.title('Actual vs Predicted Factors by Integer')
plt.xticks(x_pos, all_integers)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("plots/babyshora2.2.png", dpi=150, bbox_inches='tight')
plt.show()

print("Plots saved to plots/babyshora2.2.png")