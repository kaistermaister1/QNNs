#!/usr/bin/env python3
"""
Comprehensive QNN Model Comparison Study
========================================

This script compares 4 different Quantum Neural Network architectures:
1. 1-Qubit Angle Embedding (RY + RZ gates)
2. 1-Qubit Amplitude Embedding (single RY gate with preprocessed angles)
3. 2-Qubit ZZFeatureMap + RealAmplitudes (default QNNCircuit)
4. 2-Qubit Custom Angle Embedding + RealAmplitudes

Each model is trained and tested on 100 different random datasets.
"""

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit_algorithms.optimizers import COBYLA, L_BFGS_B
from qiskit_algorithms.utils import algorithm_globals

from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier, VQC
from qiskit_machine_learning.algorithms.regressors import NeuralNetworkRegressor, VQR
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
from qiskit_machine_learning.circuit.library import QNNCircuit
from qiskit.primitives import Estimator

import warnings
warnings.filterwarnings('ignore')

# Configuration
NUM_TRIALS = 100
NUM_INPUTS = 2
NUM_SAMPLES = 20
MAX_ITER = 60

print("🚀 Starting QNN Model Comparison Study")
print(f"📊 Running {NUM_TRIALS} trials per model")
print(f"📈 Dataset: {NUM_SAMPLES} samples, {NUM_INPUTS} features each")
print("=" * 60)

def generate_dataset():
    """Generate a random binary classification dataset"""
    X = 2 * algorithm_globals.random.random([NUM_SAMPLES, NUM_INPUTS]) - 1
    y01 = 1 * (np.sum(X, axis=1) >= 0)  # Points above/below y = -x line
    y = 2 * y01 - 1  # Map to {-1, +1}
    return X, y

def preprocess_for_amplitude_embedding(X):
    """Convert 2D coordinates to angles for amplitude embedding"""
    X_norm = X.copy()
    X2 = []
    for i in range(len(X_norm)):
        X_norm[i] = X_norm[i] / np.sqrt(X_norm[i][0]**2 + X_norm[i][1]**2)
        angle = float(np.arccos(X_norm[i][0]))
        X2.append(angle)
    return np.array(X2, dtype=float).reshape(-1, 1)

# ============================================================================
# MODEL 1: 1-Qubit Angle Embedding (RY + RZ)
# ============================================================================
print("🔄 Model 1: 1-Qubit Angle Embedding (RY + RZ)")

model1_accuracies = []
estimator = Estimator()

for trial in range(NUM_TRIALS):
    if (trial + 1) % 20 == 0:
        print(f"   Trial {trial + 1}/{NUM_TRIALS}")
    
    # Generate data
    X_train, y_train = generate_dataset()
    X_test, y_test = generate_dataset()
    
    # Create 1-qubit angle embedding circuit
    feature_map = QuantumCircuit(1)
    params = [Parameter("input1"), Parameter("input2")]
    feature_map.ry(params[0], 0)
    feature_map.rz(params[1], 0)

    ansatz = QuantumCircuit(1)
    a_params = [Parameter("theta1"), Parameter("theta2")]
    ansatz.rz(a_params[0],0)
    ansatz.ry(a_params[1],0)
    
    qc = QNNCircuit(
        num_qubits=1,
        feature_map=feature_map,
        ansatz=ansatz
    )
    
    # Train and test
    estimator_qnn = EstimatorQNN(circuit=qc, estimator=estimator)
    classifier = NeuralNetworkClassifier(estimator_qnn, optimizer=COBYLA(maxiter=MAX_ITER))
    
    classifier.fit(X_train, y_train)
    accuracy = classifier.score(X_test, y_test)
    model1_accuracies.append(accuracy)

print(f"✅ Model 1 Complete - Avg Accuracy: {np.mean(model1_accuracies):.3f} ± {np.std(model1_accuracies):.3f}")

# ============================================================================
# MODEL 2: 1-Qubit Amplitude Embedding
# ============================================================================
print("🔄 Model 2: 1-Qubit Amplitude Embedding")

model2_accuracies = []

for trial in range(NUM_TRIALS):
    if (trial + 1) % 20 == 0:
        print(f"   Trial {trial + 1}/{NUM_TRIALS}")
    
    # Generate data
    X_train, y_train = generate_dataset()
    X_test, y_test = generate_dataset()
    
    # Preprocess for amplitude embedding
    X_train_amp = preprocess_for_amplitude_embedding(X_train)
    X_test_amp = preprocess_for_amplitude_embedding(X_test)
    
    # Create 1-qubit amplitude embedding circuit
    feature_map = QuantumCircuit(1)
    theta = Parameter("theta")
    feature_map.ry(theta, 0)

    ansatz = QuantumCircuit(1)
    a_params = [Parameter("theta1"), Parameter("theta2")]
    ansatz.rz(a_params[0],0)
    ansatz.ry(a_params[1],0)
    
    qc = QNNCircuit(
        num_qubits=1,
        feature_map=feature_map,
        ansatz=ansatz
    )
    
    # Train and test
    estimator_qnn = EstimatorQNN(circuit=qc, estimator=estimator)
    classifier = NeuralNetworkClassifier(estimator_qnn, optimizer=COBYLA(maxiter=MAX_ITER))
    
    classifier.fit(X_train_amp, y_train)
    accuracy = classifier.score(X_test_amp, y_test)
    model2_accuracies.append(accuracy)

print(f"✅ Model 2 Complete - Avg Accuracy: {np.mean(model2_accuracies):.3f} ± {np.std(model2_accuracies):.3f}")

# ============================================================================
# MODEL 3: 2-Qubit ZZFeatureMap + RealAmplitudes (Default QNNCircuit)
# ============================================================================
print("🔄 Model 3: 2-Qubit ZZFeatureMap + RealAmplitudes")

model3_accuracies = []

for trial in range(NUM_TRIALS):
    if (trial + 1) % 20 == 0:
        print(f"   Trial {trial + 1}/{NUM_TRIALS}")
    
    # Generate data
    X_train, y_train = generate_dataset()
    X_test, y_test = generate_dataset()
    
    # Create 2-qubit circuit with default ZZFeatureMap
    qc = QNNCircuit(2)  # Uses ZZFeatureMap + RealAmplitudes by default
    
    # Train and test
    estimator_qnn = EstimatorQNN(circuit=qc, estimator=estimator)
    classifier = NeuralNetworkClassifier(estimator_qnn, optimizer=COBYLA(maxiter=MAX_ITER))
    
    classifier.fit(X_train, y_train)
    accuracy = classifier.score(X_test, y_test)
    model3_accuracies.append(accuracy)

print(f"✅ Model 3 Complete - Avg Accuracy: {np.mean(model3_accuracies):.3f} ± {np.std(model3_accuracies):.3f}")

# ============================================================================
# MODEL 4: 2-Qubit Custom Angle Embedding + RealAmplitudes
# ============================================================================
print("🔄 Model 4: 2-Qubit Custom Angle Embedding + RealAmplitudes")

model4_accuracies = []

for trial in range(NUM_TRIALS):
    if (trial + 1) % 20 == 0:
        print(f"   Trial {trial + 1}/{NUM_TRIALS}")
    
    # Generate data
    X_train, y_train = generate_dataset()
    X_test, y_test = generate_dataset()
    
    # Create 2-qubit custom angle embedding circuit
    feature_map = QuantumCircuit(2)
    params = [Parameter("input1"), Parameter("input2")]
    feature_map.ry(params[0], 0)
    feature_map.ry(params[1], 1)
    
    qc = QNNCircuit(
        num_qubits=2,
        feature_map=feature_map,
        ansatz=RealAmplitudes(2, reps=1)
    )
    
    # Train and test
    estimator_qnn = EstimatorQNN(circuit=qc, estimator=estimator)
    classifier = NeuralNetworkClassifier(estimator_qnn, optimizer=COBYLA(maxiter=MAX_ITER))
    
    classifier.fit(X_train, y_train)
    accuracy = classifier.score(X_test, y_test)
    model4_accuracies.append(accuracy)

print(f"✅ Model 4 Complete - Avg Accuracy: {np.mean(model4_accuracies):.3f} ± {np.std(model4_accuracies):.3f}")

# ============================================================================
# RESULTS ANALYSIS AND VISUALIZATION
# ============================================================================
print("\n" + "=" * 60)
print("📊 FINAL RESULTS SUMMARY")
print("=" * 60)

models = ['1Q Angle\n(RY+RZ)', '1Q Amplitude\n(RY)', '2Q ZZFeature\n(Default)', '2Q Custom\n(RY+RY)']
accuracies = [model1_accuracies, model2_accuracies, model3_accuracies, model4_accuracies]
colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']

# Print statistics
for i, (model, acc) in enumerate(zip(models, accuracies)):
    mean_acc = np.mean(acc)
    std_acc = np.std(acc)
    min_acc = np.min(acc)
    max_acc = np.max(acc)
    print(f"{model.replace(chr(10), ' '):<20}: {mean_acc:.3f} ± {std_acc:.3f} (range: {min_acc:.3f} - {max_acc:.3f})")

# Create comprehensive visualization
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
axes = [ax1, ax2, ax3, ax4]

# Individual histograms
for i, (ax, model, acc, color) in enumerate(zip(axes, models, accuracies, colors)):
    ax.hist(acc, bins=20, alpha=0.7, color=color, edgecolor='black')
    ax.set_title(f'{model}\nMean: {np.mean(acc):.3f} ± {np.std(acc):.3f}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Test Accuracy')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)

plt.tight_layout()
plt.suptitle(f'QNN Model Comparison - {NUM_TRIALS} Trials Each', fontsize=16, fontweight='bold', y=1.02)
plt.show()

# Box plot comparison
plt.figure(figsize=(12, 8))
box_plot = plt.boxplot(accuracies, labels=models, patch_artist=True)

for patch, color in zip(box_plot['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

plt.title(f'QNN Model Performance Comparison\n{NUM_TRIALS} Trials, {NUM_SAMPLES} Samples Each', 
          fontsize=14, fontweight='bold')
plt.ylabel('Test Accuracy', fontsize=12)
plt.xlabel('Model Architecture', fontsize=12)
plt.grid(True, alpha=0.3)
plt.ylim(0, 1)

# Add mean markers
means = [np.mean(acc) for acc in accuracies]
plt.scatter(range(1, len(means) + 1), means, color='red', s=100, zorder=5, label='Mean')
plt.legend()

plt.tight_layout()
plt.show()

# Statistical significance testing
try:
    from scipy import stats
    
    print("\n" + "=" * 60)
    print("📈 STATISTICAL ANALYSIS")
    print("=" * 60)
    
    # Perform pairwise t-tests
    model_names = ['Model 1', 'Model 2', 'Model 3', 'Model 4']
    print("Pairwise t-test p-values (significant if p < 0.05):")
    print("-" * 50)
    
    for i in range(len(accuracies)):
        for j in range(i + 1, len(accuracies)):
            t_stat, p_value = stats.ttest_ind(accuracies[i], accuracies[j])
            significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
            print(f"{model_names[i]} vs {model_names[j]}: p = {p_value:.4f} {significance}")
            
except ImportError:
    print("\n📈 Statistical analysis requires scipy (pip install scipy)")

print("\n🏆 RANKING (by mean accuracy):")
print("-" * 30)
ranking = sorted(zip(models, means, range(len(models))), key=lambda x: x[1], reverse=True)
for rank, (model, mean_acc, idx) in enumerate(ranking, 1):
    print(f"{rank}. {model.replace(chr(10), ' '):<20}: {mean_acc:.3f}")

print(f"\n✨ Study completed! Analyzed {NUM_TRIALS * 4} total model runs.") 