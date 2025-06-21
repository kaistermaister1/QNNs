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
NUM_TRIALS = 10
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

# ============================================================================
# MODEL 1: VQC with Custom Feature Map + Custom Ansatz (4 features)
# ============================================================================
print("🔄 Model 1: VQC (4 Features, Custom Feature Map + Custom Ansatz)")
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
    feature_map = QuantumCircuit(4)
    input_params = [Parameter(f"input{i}") for i in range(4)]
    weight_params = [Parameter(f"weight{i}") for i in range(8)]
    feature_map.ry(input_params[0], 0)
    feature_map.ry(input_params[1], 1)
    feature_map.ry(input_params[2], 2)
    feature_map.ry(input_params[3], 3)
    ansatz = QuantumCircuit(4)
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
    model1_accuracies.append(accuracy)

print(f"✅ Model 1 Complete - Avg Accuracy: {np.mean(model1_accuracies):.3f} ± {np.std(model1_accuracies):.3f}")

# ============================================================================
# MODEL 2: VQC with custom feature map and ansatz (2 features)
# ============================================================================
print("🔄 Model 2: VQC (2 Features, Custom Feature Map + Custom Ansatz)")
model2_accuracies = []
features, labels = load_and_prep_data(condensed=True)

for trial in range(NUM_TRIALS):
    print(f"   Trial {trial + 1}/{NUM_TRIALS}")
    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, train_size=0.8, random_state=trial
    )
    
    num_features = features.shape[1]
    feature_map = QuantumCircuit(2)
    input_params = [Parameter(f"input{i}") for i in range(2)]
    weight_params = [Parameter(f"weight{i}") for i in range(4)]
    feature_map.ry(input_params[0], 0)
    feature_map.ry(input_params[1], 1)
    ansatz = QuantumCircuit(2)
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
    model2_accuracies.append(accuracy)
    
print(f"✅ Model 2 Complete - Avg Accuracy: {np.mean(model2_accuracies):.3f} ± {np.std(model2_accuracies):.3f}")
