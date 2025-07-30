#!/usr/bin/env python3
"""
line_qnns.py - QNN Model Classes for Line Classification
======================================================

This module contains class-based implementations of 4 different QNN architectures
for binary line classification tasks:

1. AngleEmbeddingQNN - 1-Qubit with RY + RZ gates
2. AmplitudeEmbeddingQNN - 1-Qubit with preprocessed angles
3. DefaultQNN - 2-Qubit ZZFeatureMap + RealAmplitudes
4. CustomAngleQNN - 2-Qubit Custom RY + RY + RealAmplitudes
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional
import warnings

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.utils import algorithm_globals
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.circuit.library import QNNCircuit
from qiskit.primitives import Estimator

warnings.filterwarnings('ignore')


class BaseQNN(ABC):
    """Abstract base class for QNN models"""
    
    def __init__(self, model_name: str, max_iter: int = 60):
        self.model_name = model_name
        self.max_iter = max_iter
        self.estimator = Estimator()
        self.classifier = None
        self.training_history = []
        
    @abstractmethod
    def create_circuit(self) -> QNNCircuit:
        """Create the quantum circuit for this model"""
        pass
    
    @abstractmethod
    def preprocess_data(self, X: np.ndarray) -> np.ndarray:
        """Preprocess input data if needed"""
        pass
    
    def build_model(self):
        """Build the QNN classifier"""
        qc = self.create_circuit()
        estimator_qnn = EstimatorQNN(circuit=qc, estimator=self.estimator)
        self.classifier = NeuralNetworkClassifier(
            estimator_qnn, 
            optimizer=COBYLA(maxiter=self.max_iter)
        )
        return self.classifier
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> float:
        """Train the model and return final objective value"""
        if self.classifier is None:
            self.build_model()
        
        X_processed = self.preprocess_data(X)
        
        # Store training progress
        objective_history = []
        
        def callback(weights, obj_func_eval):
            objective_history.append(obj_func_eval)
        
        # Set callback if available
        if hasattr(self.classifier, '_fit_result'):
            self.classifier.fit(X_processed, y)
        else:
            self.classifier.fit(X_processed, y)
        
        self.training_history = objective_history
        return self.score(X, y)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if self.classifier is None:
            raise ValueError("Model must be fitted before making predictions")
        
        X_processed = self.preprocess_data(X)
        return self.classifier.predict(X_processed)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate accuracy score"""
        if self.classifier is None:
            raise ValueError("Model must be fitted before scoring")
        
        X_processed = self.preprocess_data(X)
        return self.classifier.score(X_processed, y)


class AngleEmbeddingQNN(BaseQNN):
    """1-Qubit Angle Embedding QNN with RY + RZ gates"""
    
    def __init__(self, max_iter: int = 60):
        super().__init__("1Q Angle (RY+RZ)", max_iter)
    
    def create_circuit(self) -> QNNCircuit:
        """Create 1-qubit angle embedding circuit"""
        # Feature map with RY + RZ gates
        feature_map = QuantumCircuit(1)
        params = [Parameter("input1"), Parameter("input2")]
        feature_map.ry(params[0], 0)
        feature_map.rz(params[1], 0)
        
        # Variational ansatz
        ansatz = QuantumCircuit(1)
        a_params = [Parameter("theta1"), Parameter("theta2")]
        ansatz.rz(a_params[0], 0)
        ansatz.ry(a_params[1], 0)
        
        return QNNCircuit(
            num_qubits=1,
            feature_map=feature_map,
            ansatz=ansatz
        )
    
    def preprocess_data(self, X: np.ndarray) -> np.ndarray:
        """No preprocessing needed for angle embedding"""
        return X


class AmplitudeEmbeddingQNN(BaseQNN):
    """1-Qubit Amplitude Embedding QNN with preprocessed angles"""
    
    def __init__(self, max_iter: int = 60):
        super().__init__("1Q Amplitude (RY)", max_iter)
    
    def create_circuit(self) -> QNNCircuit:
        """Create 1-qubit amplitude embedding circuit"""
        # Feature map with single RY gate
        feature_map = QuantumCircuit(1)
        theta = Parameter("theta")
        feature_map.ry(theta, 0)
        
        # Variational ansatz
        ansatz = QuantumCircuit(1)
        a_params = [Parameter("theta1"), Parameter("theta2")]
        ansatz.rz(a_params[0], 0)
        ansatz.ry(a_params[1], 0)
        
        return QNNCircuit(
            num_qubits=1,
            feature_map=feature_map,
            ansatz=ansatz
        )
    
    def preprocess_data(self, X: np.ndarray) -> np.ndarray:
        """Convert 2D coordinates to angles for amplitude embedding"""
        X_norm = X.copy()
        X2 = []
        for i in range(len(X_norm)):
            # Normalize the vector
            norm = np.sqrt(X_norm[i][0]**2 + X_norm[i][1]**2)
            if norm > 0:
                X_norm[i] = X_norm[i] / norm
                angle = float(np.arccos(np.clip(X_norm[i][0], -1, 1)))
            else:
                angle = 0.0
            X2.append(angle)
        return np.array(X2, dtype=float).reshape(-1, 1)


class DefaultQNN(BaseQNN):
    """2-Qubit QNN with default ZZFeatureMap + RealAmplitudes"""
    
    def __init__(self, max_iter: int = 60):
        super().__init__("2Q ZZFeature (Default)", max_iter)
    
    def create_circuit(self) -> QNNCircuit:
        """Create 2-qubit circuit with default ZZFeatureMap"""
        return QNNCircuit(2)  # Uses ZZFeatureMap + RealAmplitudes by default
    
    def preprocess_data(self, X: np.ndarray) -> np.ndarray:
        """No preprocessing needed for default circuit"""
        return X


class CustomAngleQNN(BaseQNN):
    """2-Qubit Custom Angle Embedding QNN with RY + RY + RealAmplitudes"""
    
    def __init__(self, max_iter: int = 60):
        super().__init__("2Q Custom (RY+RY)", max_iter)
    
    def create_circuit(self) -> QNNCircuit:
        """Create 2-qubit custom angle embedding circuit"""
        # Custom feature map with RY gates
        feature_map = QuantumCircuit(2)
        params = [Parameter("input1"), Parameter("input2")]
        feature_map.ry(params[0], 0)
        feature_map.ry(params[1], 1)
        
        return QNNCircuit(
            num_qubits=2,
            feature_map=feature_map,
            ansatz=RealAmplitudes(2, reps=1)
        )
    
    def preprocess_data(self, X: np.ndarray) -> np.ndarray:
        """No preprocessing needed for custom angle embedding"""
        return X


def generate_line_dataset(num_samples: int = 20, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a random binary classification dataset for line separation.
    
    Args:
        num_samples: Number of samples to generate
        seed: Random seed for reproducibility
    
    Returns:
        X: Features of shape (num_samples, 2)
        y: Labels of shape (num_samples,) with values {-1, +1}
    """
    if seed is not None:
        algorithm_globals.random_seed = seed
    
    X = 2 * algorithm_globals.random.random([num_samples, 2]) - 1
    y01 = 1 * (np.sum(X, axis=1) >= 0)  # Points above/below y = -x line
    y = 2 * y01 - 1  # Map to {-1, +1}
    return X, y


def get_all_qnn_models(max_iter: int = 60) -> list:
    """Get all QNN model instances"""
    return [
        AngleEmbeddingQNN(max_iter),
        AmplitudeEmbeddingQNN(max_iter),
        DefaultQNN(max_iter),
        CustomAngleQNN(max_iter)
    ]


if __name__ == "__main__":
    # Example usage
    print("Testing QNN models...")
    
    # Generate test data
    X_train, y_train = generate_line_dataset(20, seed=42)
    X_test, y_test = generate_line_dataset(20, seed=123)
    
    models = get_all_qnn_models(max_iter=30)
    
    for model in models:
        print(f"\nTesting {model.model_name}...")
        try:
            model.fit(X_train, y_train)
            accuracy = model.score(X_test, y_test)
            print(f"  Accuracy: {accuracy:.3f}")
        except Exception as e:
            print(f"  Error: {e}") 