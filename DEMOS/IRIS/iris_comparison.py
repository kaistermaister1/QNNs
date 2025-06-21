#!/usr/bin/env python3
"""
Comprehensive QNN Model Comparison Study for IRIS Dataset
=========================================================

This script compares 4 different Quantum Neural Network architectures
for classifying the IRIS dataset.

NOTE: The models are trained on two different tasks:
- Models 1 & 2 perform 3-class classification on a single flower.
- Models 3 & 4 perform binary classification on a *pair* of flowers
  to determine if they are from the same class.
Direct accuracy comparison should be interpreted with this in mind.

The 4 models are:
1. VQC with ZZFeatureMap and RealAmplitudes on 4 features.
2. VQC with ZZFeatureMap and EfficientSU2 on 2 (PCA) features.
3. Siamese-like QNN on pairs of 4-feature flowers.
4. Siamese-like QNN on pairs of 2-feature (PCA) flowers.
"""

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap, EfficientSU2
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.utils import algorithm_globals
from qiskit.primitives import Estimator, Sampler
from qiskit.quantum_info import SparsePauliOp

from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier, VQC
from qiskit_machine_learning.neural_networks import EstimatorQNN

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings('ignore')
import os
os.makedirs("plots", exist_ok=True)


# Configuration
NUM_TRIALS = 100
MAX_ITER = 80
algorithm_globals.random_seed = 123
np.random.seed(algorithm_globals.random_seed)

print("🚀 Starting IRIS QNN Model Comparison Study")
print(f"📊 Running {NUM_TRIALS} trials per model")
print("=" * 60)

# --- Data Loading and Preprocessing ---

def load_and_prep_data(condensed=False):
    iris_data = load_iris()
    features = iris_data.data
    labels = iris_data.target
    if condensed:
        features = PCA(n_components=2).fit_transform(features)
    features = MinMaxScaler().fit_transform(features)
    return features, labels

def create_flower_pairs(features, labels, num_pairs=150):
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

# ============================================================================
# MODEL 1: VQC with ZZFeatureMap + RealAmplitudes (4 features)
# ============================================================================
print("🔄 Model 1: VQC (4 Features, ZZFeatureMap + RealAmplitudes)")
model1_accuracies = []
features, labels = load_and_prep_data(condensed=False)
sampler = Sampler()
estimator = Estimator()

for trial in range(NUM_TRIALS):
    print(f"   Trial {trial + 1}/{NUM_TRIALS}")
    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, train_size=0.8, random_state=trial
    )

    num_features = features.shape[1]
    feature_map = ZZFeatureMap(feature_dimension=num_features, reps=1)
    ansatz = RealAmplitudes(num_qubits=num_features, reps=3)
    
    classifier = VQC(
        sampler=sampler,
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=COBYLA(maxiter=MAX_ITER),
    )
    
    classifier.fit(train_features, train_labels)
    accuracy = classifier.score(test_features, test_labels)
    model1_accuracies.append(accuracy)

print(f"✅ Model 1 Complete - Avg Accuracy: {np.mean(model1_accuracies):.3f} ± {np.std(model1_accuracies):.3f}")

# ============================================================================
# MODEL 2: VQC with ZZFeatureMap + EfficientSU2 (2 features)
# ============================================================================
print("🔄 Model 2: VQC (2 Features, ZZFeatureMap + EfficientSU2)")
model2_accuracies = []
features, labels = load_and_prep_data(condensed=True)

for trial in range(NUM_TRIALS):
    print(f"   Trial {trial + 1}/{NUM_TRIALS}")
    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, train_size=0.8, random_state=trial
    )
    
    num_features = features.shape[1]
    feature_map = ZZFeatureMap(feature_dimension=num_features, reps=1)
    ansatz = EfficientSU2(num_qubits=num_features, reps=3)

    classifier = VQC(
        sampler=sampler,
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=COBYLA(maxiter=MAX_ITER),
    )
    
    classifier.fit(train_features, train_labels)
    accuracy = classifier.score(test_features, test_labels)
    model2_accuracies.append(accuracy)
    
print(f"✅ Model 2 Complete - Avg Accuracy: {np.mean(model2_accuracies):.3f} ± {np.std(model2_accuracies):.3f}")

# ============================================================================
# MODEL 3: Siamese-like QNN (4-feature pairs)
# ============================================================================
print("🔄 Model 3: Siamese-like QNN (4-feature pairs)")
model3_accuracies = []
features, labels = load_and_prep_data(condensed=False)

# Define circuit
qc3 = QuantumCircuit(4)
input_params3 = [Parameter(f"input{i}") for i in range(8)]
weight_params3 = [Parameter(f"weight{i}") for i in range(8)]
# Feature map part
qc3.ry(input_params3[0], 0); qc3.rz(input_params3[1], 0)
qc3.ry(input_params3[2], 1); qc3.rz(input_params3[3], 1)
qc3.ry(input_params3[4], 2); qc3.rz(input_params3[5], 2)
qc3.ry(input_params3[6], 3); qc3.rz(input_params3[7], 3)
qc3.cx(0, 2); qc3.cx(1, 3)
qc3.barrier()
# Ansatz part
qc3.ry(weight_params3[0], 0); qc3.rz(weight_params3[1], 1)
qc3.ry(weight_params3[2], 2); qc3.rz(weight_params3[3], 3)
qc3.rz(weight_params3[4], 0); qc3.ry(weight_params3[5], 1)
qc3.rz(weight_params3[6], 2); qc3.ry(weight_params3[7], 3)
observable3 = SparsePauliOp("ZZZZ")
qnn3 = EstimatorQNN(circuit=qc3, estimator=estimator, input_params=input_params3, weight_params=weight_params3, observables=observable3)


for trial in range(NUM_TRIALS):
    print(f"   Trial {trial + 1}/{NUM_TRIALS}")
    paired_features, paired_labels = create_flower_pairs(features, labels, num_pairs=120)
    train_features, test_features, train_labels, test_labels = train_test_split(
        paired_features, paired_labels, train_size=0.8, random_state=trial, stratify=paired_labels
    )
    
    classifier = NeuralNetworkClassifier(qnn3, optimizer=COBYLA(maxiter=MAX_ITER), loss="squared_error")
    classifier.fit(train_features, train_labels)
    accuracy = classifier.score(test_features, test_labels)
    model3_accuracies.append(accuracy)

print(f"✅ Model 3 Complete - Avg Accuracy: {np.mean(model3_accuracies):.3f} ± {np.std(model3_accuracies):.3f}")

# ============================================================================
# MODEL 4: Siamese-like QNN (2-feature pairs)
# ============================================================================
print("🔄 Model 4: Siamese-like QNN (2-feature pairs)")
model4_accuracies = []
features, labels = load_and_prep_data(condensed=True)

# Define circuit
qc4 = QuantumCircuit(4)
input_params4 = [Parameter(f"input{i}") for i in range(4)]
weight_params4 = [Parameter(f"weight{i}") for i in range(8)]
# Feature map part
qc4.ry(input_params4[0], 0); qc4.ry(input_params4[1], 1)
qc4.ry(input_params4[2], 2); qc4.ry(input_params4[3], 3)
qc4.cx(0, 2); qc4.cx(1, 3)
qc4.barrier()
# Ansatz part
qc4.ry(weight_params4[0], 0); qc4.ry(weight_params4[1], 1)
qc4.ry(weight_params4[2], 2); qc4.ry(weight_params4[3], 3)
qc4.rz(weight_params4[4], 0); qc4.rz(weight_params4[5], 1)
qc4.rz(weight_params4[6], 2); qc4.rz(weight_params4[7], 3)
observable4 = SparsePauliOp("ZZZZ")
qnn4 = EstimatorQNN(circuit=qc4, estimator=estimator, input_params=input_params4, weight_params=weight_params4, observables=observable4)


for trial in range(NUM_TRIALS):
    print(f"   Trial {trial + 1}/{NUM_TRIALS}")
    paired_features, paired_labels = create_flower_pairs(features, labels, num_pairs=120)
    train_features, test_features, train_labels, test_labels = train_test_split(
        paired_features, paired_labels, train_size=0.8, random_state=trial, stratify=paired_labels
    )
    
    classifier = NeuralNetworkClassifier(qnn4, optimizer=COBYLA(maxiter=MAX_ITER), loss="squared_error")
    classifier.fit(train_features, train_labels)
    accuracy = classifier.score(test_features, test_labels)
    model4_accuracies.append(accuracy)

print(f"✅ Model 4 Complete - Avg Accuracy: {np.mean(model4_accuracies):.3f} ± {np.std(model4_accuracies):.3f}")


# ============================================================================
# MODEL 5: VQC with Custom Feature Map + Custom Ansatz (4 features)
# ============================================================================
print("🔄 Model 5: VQC (4 Features, Custom Feature Map + Custom Ansatz)")
model5_accuracies = []
features, labels = load_and_prep_data(condensed=False)

for trial in range(NUM_TRIALS):
    print(f"   Trial {trial + 1}/{NUM_TRIALS}")
    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, train_size=0.8, random_state=trial
    )

    num_features = features.shape[1]
    feature_map = QuantumCircuit(4)
    input_params = [Parameter(f"input{i}") for i in range(4)]
    feature_map.ry(input_params[0], 0)
    feature_map.ry(input_params[1], 1)
    feature_map.ry(input_params[2], 2)
    feature_map.ry(input_params[3], 3)
    
    ansatz = QuantumCircuit(4)
    weight_params = [Parameter(f"weight{i}") for i in range(8)]
    ansatz.ry(weight_params[0], 0)
    ansatz.ry(weight_params[1], 1)
    ansatz.ry(weight_params[2], 2)
    ansatz.ry(weight_params[3], 3)
    ansatz.rz(weight_params[4], 0)
    ansatz.rz(weight_params[5], 1)
    ansatz.rz(weight_params[6], 2)
    ansatz.rz(weight_params[7], 3)

    classifier = VQC(
        sampler=sampler,
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=COBYLA(maxiter=MAX_ITER),
    )
    
    classifier.fit(train_features, train_labels)
    accuracy = classifier.score(test_features, test_labels)
    model5_accuracies.append(accuracy)

print(f"✅ Model 5 Complete - Avg Accuracy: {np.mean(model5_accuracies):.3f} ± {np.std(model5_accuracies):.3f}")

# ============================================================================
# MODEL 6: VQC with custom feature map and ansatz (2 features)
# ============================================================================
print("🔄 Model 6: VQC (2 Features, Custom Feature Map + Custom Ansatz)")
model6_accuracies = []
features, labels = load_and_prep_data(condensed=True)

for trial in range(NUM_TRIALS):
    print(f"   Trial {trial + 1}/{NUM_TRIALS}")
    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, train_size=0.8, random_state=trial
    )
    
    num_features = features.shape[1]
    feature_map = QuantumCircuit(2)
    input_params = [Parameter(f"input{i}") for i in range(2)]
    feature_map.ry(input_params[0], 0)
    feature_map.ry(input_params[1], 1)

    ansatz = QuantumCircuit(2)
    weight_params = [Parameter(f"weight{i}") for i in range(4)]
    ansatz.ry(weight_params[0], 0)
    ansatz.rz(weight_params[1], 0)
    ansatz.ry(weight_params[2], 1)
    ansatz.rz(weight_params[3], 1)

    classifier = VQC(
        sampler=sampler,
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=COBYLA(maxiter=MAX_ITER),
    )
    
    classifier.fit(train_features, train_labels)
    accuracy = classifier.score(test_features, test_labels)
    model6_accuracies.append(accuracy)
    
print(f"✅ Model 6 Complete - Avg Accuracy: {np.mean(model6_accuracies):.3f} ± {np.std(model6_accuracies):.3f}")

# ============================================================================
# MODEL 7: 2-Qubit Siamese-like QNN (2-feature pairs, condensed)
# ============================================================================
print("🔄 Model 7: 2-Qubit Siamese-like QNN (2-feature pairs, condensed)")
model7_accuracies = []
features, labels = load_and_prep_data(condensed=True)

# Define circuit (copied from 2siamesecondensed.py)
qc7 = QuantumCircuit(2)
input_params7 = [Parameter(f"input{i}") for i in range(4)]
weight_params7 = [Parameter(f"weight{i}") for i in range(4)]

# Feature map part
qc7.ry(input_params7[0], 0)
qc7.rz(input_params7[1], 0)
qc7.ry(input_params7[2], 1)
qc7.rz(input_params7[3], 1)
qc7.cx(0, 1)
qc7.barrier()

# Ansatz part
qc7.ry(weight_params7[0], 0)
qc7.ry(weight_params7[1], 1)
qc7.rz(weight_params7[2], 0)
qc7.rz(weight_params7[3], 1)

observable7 = SparsePauliOp("ZZ")
qnn7 = EstimatorQNN(circuit=qc7, estimator=estimator, input_params=input_params7, weight_params=weight_params7, observables=observable7)

for trial in range(NUM_TRIALS):
    print(f"   Trial {trial + 1}/{NUM_TRIALS}")
    paired_features, paired_labels = create_flower_pairs(features, labels, num_pairs=120)
    train_features, test_features, train_labels, test_labels = train_test_split(
        paired_features, paired_labels, train_size=0.8, random_state=trial, stratify=paired_labels
    )
    
    classifier = NeuralNetworkClassifier(qnn7, optimizer=COBYLA(maxiter=MAX_ITER), loss="squared_error")
    classifier.fit(train_features, train_labels)
    accuracy = classifier.score(test_features, test_labels)
    model7_accuracies.append(accuracy)

print(f"✅ Model 7 Complete - Avg Accuracy: {np.mean(model7_accuracies):.3f} ± {np.std(model7_accuracies):.3f}")


# ============================================================================
# RESULTS ANALYSIS AND VISUALIZATION
# ============================================================================
print("\n" + "=" * 60)
print("📊 FINAL RESULTS SUMMARY")
print("=" * 60)

models = [
    '1. VQC ZZ+RA 4F\n(3-Class)', 
    '2. VQC ZZ+ESU2 2F\n(3-Class)', 
    '3. Siamese 4F\n(Binary Pair)', 
    '4. Siamese 2F\n(Binary Pair)',
    '5. VQC Custom 4F\n(3-Class)',
    '6. VQC Custom 2F\n(3-Class)',
    '7. 2Q Siamese 2F\n(Binary Pair)'
]
accuracies = [
    model1_accuracies, model2_accuracies, model3_accuracies, 
    model4_accuracies, model5_accuracies, model6_accuracies, model7_accuracies
]
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
plt.suptitle(f'IRIS QNN Model Comparison - {NUM_TRIALS} Trials Each', fontsize=16, fontweight='bold', y=1.02)
plt.savefig('DEMOS/IRIS/plots/iris_comparison_histograms.png', dpi=150, bbox_inches='tight')
plt.close()

# Box plot comparison
plt.figure(figsize=(14, 8))
box_plot = plt.boxplot(accuracies, labels=[m.replace('\n', ' ') for m in models], patch_artist=True)

for patch, color in zip(box_plot['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

plt.title(f'IRIS QNN Model Performance Comparison\n({NUM_TRIALS} Trials)', 
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
plt.savefig('DEMOS/IRIS/plots/iris_comparison_boxplots.png', dpi=150, bbox_inches='tight')
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
    plt.savefig('DEMOS/IRIS/plots/iris_comparison_p_values.png', dpi=150, bbox_inches='tight')
    plt.close()

except ImportError:
    print("\n📈 Statistical analysis requires scipy, pandas, and seaborn (pip install scipy pandas seaborn)")

print(f"\n✨ Study completed! All 7 models compared successfully!")
print(f"📊 Models tested:")
print(f"   1. VQC with ZZFeatureMap + RealAmplitudes (4 features)")
print(f"   2. VQC with ZZFeatureMap + EfficientSU2 (2 features)")
print(f"   3. Siamese-like QNN (4-feature pairs, 4 qubits)")
print(f"   4. Siamese-like QNN (2-feature pairs, 4 qubits)")
print(f"   5. VQC with Custom Feature Map + Ansatz (4 features)")
print(f"   6. VQC with Custom Feature Map + Ansatz (2 features)")
print(f"   7. 2-Qubit Siamese-like QNN (2-feature pairs, condensed)")
print(f"\n📁 Saved plots to DEMOS/IRIS/plots/:")
print(f"   - iris_comparison_histograms.png")
print(f"   - iris_comparison_boxplots.png") 
print(f"   - iris_comparison_p_values.png")