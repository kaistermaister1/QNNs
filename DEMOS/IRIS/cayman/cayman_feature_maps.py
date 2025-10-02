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
NUM_TRIALS = 100 # umber of times to train and test each model
MAX_ITER = 80 # COBYLA iterations
algorithm_globals.random_seed = 123
np.random.seed(algorithm_globals.random_seed)


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

# --- Create QNN classes ---
class BaseQNNModel:
    """Base class for all QNN models."""
    
    def __init__(self, sampler, estimator, name):
        self.sampler = sampler
        self.estimator = estimator
        self.name = name
    
    def prepare_data(self):
        """Prepare data for training. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement prepare_data method")
    
    def create_classifier(self, features, **kwargs):
        """Create the classifier for this model. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement create_classifier method")

    def show_feature_map(self):
        """Show the feature map for this model."""
        raise NotImplementedError("Subclasses must implement show_feature_map method")
    
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

class FourGateFM(BaseQNNModel):
    """Model 1: 4 gate feature map with simple ansatz."""
    
    def __init__(self, sampler, estimator):
        super().__init__(sampler, estimator, "FourGateFM")
        self.description = "4 RY inputs; shallow RY+RZ ansatz"
    
    def prepare_data(self, trial):
        features, labels = load_and_prep_data(condensed=False)
        return train_test_split(features, labels, train_size=0.8, random_state=trial)
    
    def show_feature_map(self):
        """Show the feature map for this model."""
        self.feature_map.draw(output='mpl', style='clifford')
        plt.savefig('plots/FourGateFM.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def create_classifier(self, train_features, max_iter):
        num_features = train_features.shape[1]

        # Feature map
        feature_map = QuantumCircuit(4)
        input_params = [Parameter(f"input{i}") for i in range(4)]
        feature_map.ry(input_params[0], 0)
        feature_map.ry(input_params[1], 1)
        feature_map.ry(input_params[2], 2)
        feature_map.ry(input_params[3], 3)
        self.feature_map = feature_map

        # Ansatz
        weight_params = [Parameter(f"weight{i}") for i in range(8)]
        ansatz = QuantumCircuit(4)
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

class RY_H(BaseQNNModel):
    """Model 2: Feature map with R/Y and Hadamards"""
    
    def __init__(self, sampler, estimator):
        super().__init__(sampler, estimator, "RY_H")
        self.description = "Interleaved H layers with RY(x) and RY(0.5x); RY+RZ ansatz"
    
    def prepare_data(self, trial):
        features, labels = load_and_prep_data(condensed=False)
        return train_test_split(features, labels, train_size=0.8, random_state=trial)

    def show_feature_map(self):
        """Show the feature map for this model."""
        self.feature_map.draw(output='mpl', style='clifford')
        plt.savefig('plots/RY_H.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def create_classifier(self, train_features, max_iter):
        num_features = train_features.shape[1]

        # --- Feature map ---
        feature_map = QuantumCircuit(4)
        input_params = [Parameter(f"input{i}") for i in range(4)]
        # Helper function to add hadamards
        def add_h():
            for i in range(4):
                feature_map.h(i)
        add_h()
        for i in range(4): # Add rotations * 1
            feature_map.ry(input_params[i], i)
        add_h()
        for i in range(4): # Add rotations * 0.5
            feature_map.ry(0.5*input_params[i], i)
        add_h()
        self.feature_map = feature_map

        # --- Ansatz ---
        weight_params = [Parameter(f"weight{i}") for i in range(8)]
        ansatz = QuantumCircuit(4)
        for i in range(4):
            ansatz.ry(weight_params[i], i)
        for i in range(4):
            ansatz.rz(weight_params[i+4], i)
        
        return VQC(
            sampler=self.sampler,
            feature_map=feature_map,
            ansatz=ansatz,
            optimizer=COBYLA(maxiter=max_iter),
        )

class ZZ_Shifted(BaseQNNModel):
    """Model 3: Shifted ZZFeatureMap"""
    
    def __init__(self, sampler, estimator):
        super().__init__(sampler, estimator, "ZZ_Shifted")
        self.description = "Shifted ZZ-style pairwise phases; RY+RZ ansatz"
    
    def prepare_data(self, trial):
        features, labels = load_and_prep_data(condensed=False)
        return train_test_split(features, labels, train_size=0.8, random_state=trial)

    def show_feature_map(self):
        """Show the feature map for this model."""
        self.feature_map.draw(output='mpl', style='clifford')
        plt.savefig('plots/ZZ_Shifted.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def create_classifier(self, train_features, max_iter):
        num_features = train_features.shape[1]

        # --- Feature map ---
        feature_map = QuantumCircuit(4)
        x = [Parameter(f"input{i}") for i in range(4)]
        # 1) Hadamards
        for q in range(4):
            feature_map.h(q)
        # 2) Local phases: P(2*x[i])
        for i in range(4):
            feature_map.p(2.0 * x[i], i)
        # 3) Pairwise interactions: angle = 2*(pi - x[i])*(pi - x[j])
        for i in range(4):
            for j in range(i+1, 4):
                theta_ij = 2.0 * (np.pi - x[i]) * (np.pi - x[j])
                # implement ZZ-like coupling via CX–P–CX on target j
                feature_map.cx(i, j)
                feature_map.p(theta_ij, j)
                feature_map.cx(i, j)
        self.feature_map = feature_map

        # --- Ansatz ---
        weight_params = [Parameter(f"weight{i}") for i in range(8)]
        ansatz = QuantumCircuit(4)
        for i in range(4):
            ansatz.ry(weight_params[i], i)
        for i in range(4):
            ansatz.rz(weight_params[i+4], i)
        
        return VQC(
            sampler=self.sampler,
            feature_map=feature_map,
            ansatz=ansatz,
            optimizer=COBYLA(maxiter=max_iter),
        )

# --- Train models and visualize histograms in a single plot
def train_and_plot():
    sampler = Sampler()
    estimator = Estimator()

    models = [
        FourGateFM(sampler, estimator),
        RY_H(sampler, estimator),
        ZZ_Shifted(sampler, estimator),
    ]

    all_accuracies = []
    labels = []

    for model in models:
        accuracies, label = model.run_trials(NUM_TRIALS, MAX_ITER)
        all_accuracies.append(accuracies)
        labels.append(model.name)

    # Plot side-by-side histograms (1 row, 3 columns)
    bins = np.linspace(0.0, 1.0, 11)
    colors = ["tab:blue", "tab:orange", "tab:green"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for i, accs in enumerate(all_accuracies):
        axes[i].hist(
            accs,
            bins=bins,
            alpha=0.7,
            color=colors[i % len(colors)],
            edgecolor="black",
        )
        axes[i].set_title(labels[i])
        axes[i].set_xlim(0.0, 1.0)
        axes[i].grid(True, alpha=0.3)
        axes[i].set_xlabel("Accuracy")
    axes[0].set_ylabel(f"Count (out of {NUM_TRIALS})")
    fig.suptitle("Iris QNN Accuracy Distributions")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("plots/cayman_histograms.png", dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    train_and_plot()