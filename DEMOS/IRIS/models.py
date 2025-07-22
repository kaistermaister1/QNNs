#!/usr/bin/env python3
"""
QNN Model Classes for IRIS Dataset Classification
=================================================

This module contains class-based implementations of various Quantum Neural Network
architectures for the IRIS dataset classification task.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap, EfficientSU2
from qiskit_algorithms.optimizers import COBYLA
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier, VQC
from qiskit_machine_learning.neural_networks import EstimatorQNN
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA


# --- Data Loading and Preprocessing ---

def load_and_prep_data(condensed=False):
    """Load and preprocess IRIS dataset."""
    iris_data = load_iris()
    features = iris_data.data
    labels = iris_data.target
    if condensed:
        features = PCA(n_components=2).fit_transform(features)
    features = MinMaxScaler().fit_transform(features)
    return features, labels


def create_flower_pairs(features, labels, num_pairs=150):
    """Create pairs of flowers for Siamese-like models."""
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


# --- Base Model Class ---

class BaseQNNModel:
    """Base class for all QNN models."""
    
    def __init__(self, sampler, estimator, name, description):
        self.sampler = sampler
        self.estimator = estimator
        self.name = name
        self.description = description
    
    def prepare_data(self):
        """Prepare data for training. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement prepare_data method")
    
    def create_classifier(self, features, **kwargs):
        """Create the classifier for this model. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement create_classifier method")
    
    def run_trials(self, num_trials, max_iter):
        """Run training trials and return accuracies."""
        print(f"🔄 {self.name}: {self.description}")
        accuracies = []
        
        for trial in range(num_trials):
            print(f"   Trial {trial + 1}/{num_trials}", end='\r')
            
            # Prepare data (model-specific)
            train_features, test_features, train_labels, test_labels = self.prepare_data(trial)
            
            # Create classifier (model-specific)
            classifier = self.create_classifier(train_features, max_iter)
            
            # Train and evaluate
            classifier.fit(train_features, train_labels)
            accuracy = classifier.score(test_features, test_labels)
            accuracies.append(accuracy)
        
        avg_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        print(f"✅ {self.name} Complete - Avg Accuracy: {avg_acc:.3f} ± {std_acc:.3f}")
        
        return accuracies, f'{self.name}\n{self.description}'


# --- Individual Model Classes ---

class VQCZZRealAmplitudes4F(BaseQNNModel):
    """Model 1: VQC with ZZFeatureMap + RealAmplitudes (4 features)."""
    
    def __init__(self, sampler, estimator):
        super().__init__(sampler, estimator, "1. VQC ZZ+RA 4F", "(3-Class)")
    
    def prepare_data(self, trial):
        features, labels = load_and_prep_data(condensed=False)
        return train_test_split(features, labels, train_size=0.8, random_state=trial)
    
    def create_classifier(self, train_features, max_iter):
        num_features = train_features.shape[1]
        feature_map = ZZFeatureMap(feature_dimension=num_features, reps=1)
        ansatz = RealAmplitudes(num_qubits=num_features, reps=3)
        
        return VQC(
            sampler=self.sampler,
            feature_map=feature_map,
            ansatz=ansatz,
            optimizer=COBYLA(maxiter=max_iter),
        )


class VQCZZEfficientSU2_2F(BaseQNNModel):
    """Model 2: VQC with ZZFeatureMap + EfficientSU2 (2 features)."""
    
    def __init__(self, sampler, estimator):
        super().__init__(sampler, estimator, "2. VQC ZZ+ESU2 2F", "(3-Class)")
    
    def prepare_data(self, trial):
        features, labels = load_and_prep_data(condensed=True)
        return train_test_split(features, labels, train_size=0.8, random_state=trial)
    
    def create_classifier(self, train_features, max_iter):
        num_features = train_features.shape[1]
        feature_map = ZZFeatureMap(feature_dimension=num_features, reps=1)
        ansatz = EfficientSU2(num_qubits=num_features, reps=3)
        
        return VQC(
            sampler=self.sampler,
            feature_map=feature_map,
            ansatz=ansatz,
            optimizer=COBYLA(maxiter=max_iter),
        )


class SiameseQNN4F(BaseQNNModel):
    """Model 3: Siamese-like QNN (4-feature pairs)."""
    
    def __init__(self, sampler, estimator):
        super().__init__(sampler, estimator, "3. Siamese 4F", "(Binary Pair)")
        self._setup_circuit()
    
    def _setup_circuit(self):
        """Setup the quantum circuit for this model."""
        self.qc = QuantumCircuit(4)
        self.input_params = [Parameter(f"input{i}") for i in range(8)]
        self.weight_params = [Parameter(f"weight{i}") for i in range(8)]
        
        # Feature map part
        self.qc.ry(self.input_params[0], 0); self.qc.rz(self.input_params[1], 0)
        self.qc.ry(self.input_params[2], 1); self.qc.rz(self.input_params[3], 1)
        self.qc.ry(self.input_params[4], 2); self.qc.rz(self.input_params[5], 2)
        self.qc.ry(self.input_params[6], 3); self.qc.rz(self.input_params[7], 3)
        self.qc.cx(0, 2); self.qc.cx(1, 3)
        self.qc.barrier()
        
        # Ansatz part
        self.qc.ry(self.weight_params[0], 0); self.qc.rz(self.weight_params[1], 1)
        self.qc.ry(self.weight_params[2], 2); self.qc.rz(self.weight_params[3], 3)
        self.qc.rz(self.weight_params[4], 0); self.qc.ry(self.weight_params[5], 1)
        self.qc.rz(self.weight_params[6], 2); self.qc.ry(self.weight_params[7], 3)
        
        self.observable = SparsePauliOp("ZZZZ")
        self.qnn = EstimatorQNN(
            circuit=self.qc, 
            estimator=self.estimator, 
            input_params=self.input_params, 
            weight_params=self.weight_params, 
            observables=self.observable
        )
    
    def prepare_data(self, trial):
        features, labels = load_and_prep_data(condensed=False)
        paired_features, paired_labels = create_flower_pairs(features, labels, num_pairs=120)
        return train_test_split(
            paired_features, paired_labels, train_size=0.8, 
            random_state=trial, stratify=paired_labels
        )
    
    def create_classifier(self, train_features, max_iter):
        return NeuralNetworkClassifier(
            self.qnn, 
            optimizer=COBYLA(maxiter=max_iter), 
            loss="squared_error"
        )


class SiameseQNN2F(BaseQNNModel):
    """Model 4: Siamese-like QNN (2-feature pairs)."""
    
    def __init__(self, sampler, estimator):
        super().__init__(sampler, estimator, "4. Siamese 2F", "(Binary Pair)")
        self._setup_circuit()
    
    def _setup_circuit(self):
        """Setup the quantum circuit for this model."""
        self.qc = QuantumCircuit(4)
        self.input_params = [Parameter(f"input{i}") for i in range(4)]
        self.weight_params = [Parameter(f"weight{i}") for i in range(8)]
        
        # Feature map part
        self.qc.ry(self.input_params[0], 0); self.qc.ry(self.input_params[1], 1)
        self.qc.ry(self.input_params[2], 2); self.qc.ry(self.input_params[3], 3)
        self.qc.cx(0, 2); self.qc.cx(1, 3)
        self.qc.barrier()
        
        # Ansatz part
        self.qc.ry(self.weight_params[0], 0); self.qc.ry(self.weight_params[1], 1)
        self.qc.ry(self.weight_params[2], 2); self.qc.ry(self.weight_params[3], 3)
        self.qc.rz(self.weight_params[4], 0); self.qc.rz(self.weight_params[5], 1)
        self.qc.rz(self.weight_params[6], 2); self.qc.rz(self.weight_params[7], 3)
        
        self.observable = SparsePauliOp("ZZZZ")
        self.qnn = EstimatorQNN(
            circuit=self.qc, 
            estimator=self.estimator, 
            input_params=self.input_params, 
            weight_params=self.weight_params, 
            observables=self.observable
        )
    
    def prepare_data(self, trial):
        features, labels = load_and_prep_data(condensed=True)
        paired_features, paired_labels = create_flower_pairs(features, labels, num_pairs=120)
        return train_test_split(
            paired_features, paired_labels, train_size=0.8, 
            random_state=trial, stratify=paired_labels
        )
    
    def create_classifier(self, train_features, max_iter):
        return NeuralNetworkClassifier(
            self.qnn, 
            optimizer=COBYLA(maxiter=max_iter), 
            loss="squared_error"
        )


class VQCCustom4F(BaseQNNModel):
    """Model 5: VQC with Custom Feature Map + Custom Ansatz (4 features)."""
    
    def __init__(self, sampler, estimator):
        super().__init__(sampler, estimator, "5. VQC Custom 4F", "(3-Class)")
    
    def prepare_data(self, trial):
        features, labels = load_and_prep_data(condensed=False)
        return train_test_split(features, labels, train_size=0.8, random_state=trial)
    
    def create_classifier(self, train_features, max_iter):
        # Custom feature map
        feature_map = QuantumCircuit(4)
        input_params = [Parameter(f"input{i}") for i in range(4)]
        feature_map.ry(input_params[0], 0)
        feature_map.ry(input_params[1], 1)
        feature_map.ry(input_params[2], 2)
        feature_map.ry(input_params[3], 3)
        
        # Custom ansatz
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
        
        return VQC(
            sampler=self.sampler,
            feature_map=feature_map,
            ansatz=ansatz,
            optimizer=COBYLA(maxiter=max_iter),
        )


class VQCCustom2F(BaseQNNModel):
    """Model 6: VQC with custom feature map and ansatz (2 features)."""
    
    def __init__(self, sampler, estimator):
        super().__init__(sampler, estimator, "6. VQC Custom 2F", "(3-Class)")
    
    def prepare_data(self, trial):
        features, labels = load_and_prep_data(condensed=True)
        return train_test_split(features, labels, train_size=0.8, random_state=trial)
    
    def create_classifier(self, train_features, max_iter):
        # Custom feature map
        feature_map = QuantumCircuit(2)
        input_params = [Parameter(f"input{i}") for i in range(2)]
        feature_map.ry(input_params[0], 0)
        feature_map.ry(input_params[1], 1)
        
        # Custom ansatz
        ansatz = QuantumCircuit(2)
        weight_params = [Parameter(f"weight{i}") for i in range(4)]
        ansatz.ry(weight_params[0], 0)
        ansatz.rz(weight_params[1], 0)
        ansatz.ry(weight_params[2], 1)
        ansatz.rz(weight_params[3], 1)
        
        return VQC(
            sampler=self.sampler,
            feature_map=feature_map,
            ansatz=ansatz,
            optimizer=COBYLA(maxiter=max_iter),
        )


class Siamese2QubitCondensed(BaseQNNModel):
    """Model 7: 2-Qubit Siamese-like QNN (2-feature pairs, condensed)."""
    
    def __init__(self, sampler, estimator):
        super().__init__(sampler, estimator, "7. 2Q Siamese 2F", "(Binary Pair)")
        self._setup_circuit()
    
    def _setup_circuit(self):
        """Setup the quantum circuit for this model."""
        self.qc = QuantumCircuit(2)
        self.input_params = [Parameter(f"input{i}") for i in range(4)]
        self.weight_params = [Parameter(f"weight{i}") for i in range(4)]
        
        # Feature map
        self.qc.ry(self.input_params[0], 0)
        self.qc.rz(self.input_params[1], 0)
        self.qc.ry(self.input_params[2], 1)
        self.qc.rz(self.input_params[3], 1)
        self.qc.cx(0, 1)
        self.qc.barrier()
        
        # Ansatz
        self.qc.ry(self.weight_params[0], 0)
        self.qc.ry(self.weight_params[1], 1)
        self.qc.rz(self.weight_params[2], 0)
        self.qc.rz(self.weight_params[3], 1)
        
        self.observable = SparsePauliOp("ZZ")
        self.qnn = EstimatorQNN(
            circuit=self.qc, 
            estimator=self.estimator, 
            input_params=self.input_params, 
            weight_params=self.weight_params, 
            observables=self.observable
        )
    
    def prepare_data(self, trial):
        features, labels = load_and_prep_data(condensed=True)
        paired_features, paired_labels = create_flower_pairs(features, labels, num_pairs=120)
        return train_test_split(
            paired_features, paired_labels, train_size=0.8, 
            random_state=trial, stratify=paired_labels
        )
    
    def create_classifier(self, train_features, max_iter):
        return NeuralNetworkClassifier(
            self.qnn, 
            optimizer=COBYLA(maxiter=max_iter), 
            loss="squared_error"
        )


# --- Model Registry ---

def get_all_models(sampler, estimator):
    """Get all available model classes instantiated with given sampler and estimator."""
    return [
        VQCZZRealAmplitudes4F(sampler, estimator),
        VQCZZEfficientSU2_2F(sampler, estimator),
        SiameseQNN4F(sampler, estimator),
        SiameseQNN2F(sampler, estimator),
        VQCCustom4F(sampler, estimator),
        VQCCustom2F(sampler, estimator),
        Siamese2QubitCondensed(sampler, estimator),
    ]


def get_model_by_number(model_num, sampler, estimator):
    """Get a specific model by number (1-7)."""
    models = get_all_models(sampler, estimator)
    if 1 <= model_num <= len(models):
        return models[model_num - 1]
    else:
        raise ValueError(f"Model number must be between 1 and {len(models)}") 