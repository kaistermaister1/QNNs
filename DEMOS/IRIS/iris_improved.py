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

# Sklearn
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

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
qc = QuantumCircuit(4, name="VQC Circuit")
input_params = [Parameter(f"input{i}") for i in range(8)]
weight_params = [Parameter(f"weight{i}") for i in range(8)]

# Feature map part
qc.ry(input_params[0], 0)
qc.rz(input_params[1], 0)
qc.ry(input_params[2], 1)
qc.rz(input_params[3], 1)
qc.ry(input_params[4], 2)
qc.rz(input_params[5], 2)
qc.ry(input_params[6], 3)
qc.rz(input_params[7], 3)
qc.cx(0, 2)
qc.cx(1, 3)
qc.barrier()

# Ansatz part
qc.ry(weight_params[0], 0)
qc.rz(weight_params[1], 1)
qc.ry(weight_params[2], 2)
qc.rz(weight_params[3], 3)
qc.rz(weight_params[4], 0)
qc.ry(weight_params[5], 1)
qc.rz(weight_params[6], 2)
qc.ry(weight_params[7], 3)

# Create plots directory if it doesn't exist
os.makedirs("plots", exist_ok=True)

# Draw and save the circuit
qc.draw(output="mpl", style="clifford", fold=20)
plt.suptitle("Combined Circuit")
plt.savefig("DEMOS/plots/iris_improved_circuit.png", dpi=150, bbox_inches='tight')
plt.close()
print("Circuit diagram saved as 'plots/iris_improved_circuit.png'")

# Load in data
iris_data = load_iris()
features = iris_data.data
labels = iris_data.target

# Normalize data (petal length, width, etc. are on different scales)
features = MinMaxScaler().fit_transform(features)

# Create paired data for Siamese-like training
algorithm_globals.random_seed = 123
np.random.seed(algorithm_globals.random_seed)

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

print(f"\nLabels: {paired_labels}")
print(f"\nOriginal data shape: {features.shape}")
print(f"Paired data shape: {paired_features.shape}")
print(f"Same class pairs: {np.sum(paired_labels == 1)}")
print(f"Different class pairs: {np.sum(paired_labels == -1)}")

# Split data
train_features, test_features, train_labels, test_labels = train_test_split(
    paired_features, paired_labels, train_size=0.8, random_state=algorithm_globals.random_seed, stratify=paired_labels
)

print(f"Training data shape: {train_features.shape}")
print(f"Test data shape: {test_features.shape}\n")

# --- EstimatorQNN Training ---
estimator = Estimator()

# A single observable for a single output value.
observable = SparsePauliOp("ZZZZ")

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
    print(f"Iteration {iteration}: Objective value = {obj_func_eval:.6f}")
    objective_func_vals.append(obj_func_eval)
    
    # Print progress every 10 iterations
    if iteration % 10 == 0:
        print(f"  --> Progress: {iteration}/100 iterations completed")

# Define classifier
classifier = NeuralNetworkClassifier(
    estimator_qnn,
    optimizer=COBYLA(maxiter=120),
    loss="squared_error",  # Use squared error loss for -1, 1 labels
    callback=callback_graph
)

# Train the classifier
print("Training Classifier...")
start = time.time()
classifier.fit(train_features, train_labels)
elapsed = time.time() - start
print(f"Training completed in {elapsed:.2f} seconds.\n")

# --- Evaluation ---
print("Evaluating model...")
test_score = classifier.score(test_features, test_labels)
training_score = classifier.score(train_features, train_labels)
print(f"\nClassifier on the test dataset: {test_score:.2f}\n")
print(f"Classifier on the training dataset: {training_score:.2f}\n")