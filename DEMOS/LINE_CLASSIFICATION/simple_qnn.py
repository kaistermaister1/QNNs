#!/usr/bin/env python3
"""
Simple QNN Implementation - Clean and Straightforward
====================================================

Exactly what you asked for:
1. Use line_qnns.py to generate 500 samples, 80/20 split
2. Simple feature map: Ry(2πx), Ry(2πy) 
3. Simple ansatz: Ry(θ) + Rx(θ) on each qubit
4. Forward pass function
5. Loss function and gradient
6. Adam loop with parameter shift, one sample per epoch
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from typing import Tuple, List
import os
from datetime import datetime
from tqdm import tqdm

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import Estimator
from qiskit.visualization import circuit_drawer

# Import data generation from existing module
from line_qnns import generate_line_dataset


class SimpleQNN:
    """
    Simple, clean QNN implementation as requested.
    """
    
    def __init__(self, epochs: int = 50, learning_rate: float = 0.001):
        """
        Initialize simple QNN.
        
        Args:
            epochs: Number of training epochs (one sample per epoch)
            learning_rate: Learning rate for Adam
        """
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.estimator = Estimator()
        self.qnn = None
        self.weights = None
        self.loss_history = []
        
    def create_circuit(self) -> QuantumCircuit:
        """
        Create the simple QNN circuit with your exact specifications.
        """
        # Feature map: Ry(2πx) on qubit 0, Ry(2πy) on qubit 1
        feature_map = QuantumCircuit(2)
        x_param = Parameter('x')
        y_param = Parameter('y')
        feature_map.ry(2 * np.pi * x_param, 0)
        feature_map.ry(2 * np.pi * y_param, 1)
        
        # Ansatz: Ry(θ) + Rx(θ) on each qubit
        ansatz = QuantumCircuit(2)
        theta_params = [Parameter(f'θ_{i}') for i in range(4)]
        ansatz.ry(theta_params[0], 0)
        ansatz.rx(theta_params[1], 0)
        ansatz.ry(theta_params[2], 1)
        ansatz.rx(theta_params[3], 1)
        
        # Combine circuits
        circuit = QuantumCircuit(2)
        circuit.compose(feature_map, inplace=True)
        circuit.compose(ansatz, inplace=True)
        
        return circuit
    
    def build_qnn(self):
        """
        Build the EstimatorQNN.
        """
        circuit = self.create_circuit()
        observable = SparsePauliOp.from_list([("ZZ", 1.0)])  # ZZ projector
        
        # Get parameter order
        feature_params = [p for p in circuit.parameters if p.name in ['x', 'y']]
        weight_params = [p for p in circuit.parameters if p.name.startswith('θ')]
        
        self.qnn = EstimatorQNN(
            circuit=circuit,
            observables=observable,
            input_params=feature_params,
            weight_params=weight_params,
            estimator=self.estimator
        )
    
    def forward_pass(self, X: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        Forward pass function.
        """
        predictions = self.qnn.forward(X, weights)
        if predictions.ndim > 1:
            predictions = predictions.flatten()
        # Normalize from [-1,1] to [0,1]
        return (predictions + 1) / 2
    
    def loss_function(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """
        Mean Squared Error (MSE) loss function.
        """
        return np.mean((predictions - targets) ** 2)
    
    def parameter_shift_gradient(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        Compute gradients using parameter shift rule.
        """
        gradients = np.zeros_like(weights)
        shift = np.pi / 2  # Parameter shift
        
        for i in range(len(weights)):
            # Forward shift
            weights_plus = weights.copy()
            weights_plus[i] += shift
            pred_plus = self.forward_pass(X, weights_plus)
            loss_plus = self.loss_function(pred_plus, y)
            
            # Backward shift
            weights_minus = weights.copy()
            weights_minus[i] -= shift
            pred_minus = self.forward_pass(X, weights_minus)
            loss_minus = self.loss_function(pred_minus, y)
            
            # Gradient
            gradients[i] = (loss_plus - loss_minus) / 2
            
        return gradients
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Adam training loop - all 500 samples per epoch, updating weights once per sample.
        """
        if self.qnn is None:
            self.build_qnn()
        
        # Preprocess data to [0,1] range and {0,1} labels
        X_processed = X_train.copy()
        if X_processed.min() < 0 or X_processed.max() > 1:
            X_processed = (X_processed - X_processed.min()) / (X_processed.max() - X_processed.min())
        
        y_processed = y_train.copy()
        if np.min(y_processed) == -1:
            y_processed = (y_processed + 1) / 2
        
        # Initialize weights
        self.weights = 2 * np.pi * np.random.random(self.qnn.num_weights)
        self.loss_history = []  # Loss after each sample
        self.epoch_losses = []  # Average loss after each epoch
        
        # Adam parameters
        m = np.zeros_like(self.weights)  # First moment
        v = np.zeros_like(self.weights)  # Second moment
        beta1, beta2 = 0.9, 0.999
        epsilon = 1e-8
        
        n_samples = len(X_processed)
        total_updates = self.epochs * n_samples
        
        print(f"Training for {self.epochs} epochs")
        print(f"Each epoch: all {n_samples} samples, {n_samples} weight updates")
        print(f"Total weight updates: {total_updates}")
        
        # Training loop with progress bar
        progress_bar = tqdm(total=total_updates, desc="Training", ncols=120)
        
        update_count = 0
        
        for epoch in range(self.epochs):
            # Shuffle samples for this epoch
            indices = np.random.permutation(n_samples)
            epoch_losses = []
            
            for i, sample_idx in enumerate(indices):
                # Get single sample
                X_sample = X_processed[sample_idx:sample_idx+1]  # Keep 2D shape
                y_sample = y_processed[sample_idx:sample_idx+1]
                
                # Forward pass
                predictions = self.forward_pass(X_sample, self.weights)
                loss = self.loss_function(predictions, y_sample)
                self.loss_history.append(loss)
                epoch_losses.append(loss)
                
                # Compute gradients using parameter shift
                gradients = self.parameter_shift_gradient(X_sample, y_sample, self.weights)
                
                # Adam update
                update_count += 1
                m = beta1 * m + (1 - beta1) * gradients
                v = beta2 * v + (1 - beta2) * gradients**2
                
                # Bias correction
                m_corrected = m / (1 - beta1**update_count)
                v_corrected = v / (1 - beta2**update_count)
                
                # Update weights
                self.weights -= self.learning_rate * m_corrected / (np.sqrt(v_corrected) + epsilon)
                
                # Update progress bar
                progress_bar.set_postfix({
                    'Epoch': f'{epoch+1}/{self.epochs}',
                    'Sample': f'{i+1}/{n_samples}',
                    'Loss': f'{loss:.6f}'
                })
                progress_bar.update(1)
            
            # Print epoch summary and store epoch loss
            epoch_avg_loss = np.mean(epoch_losses)
            self.epoch_losses.append(epoch_avg_loss)
            print(f"\nEpoch {epoch+1}/{self.epochs} completed - Avg Loss: {epoch_avg_loss:.6f}")
        
        progress_bar.close()
        print(f"\nTraining completed! Final loss: {self.loss_history[-1]:.6f}")
        print(f"Total weight updates: {len(self.loss_history)}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        """
        X_processed = X.copy()
        if X_processed.min() < 0 or X_processed.max() > 1:
            X_processed = (X_processed - X_processed.min()) / (X_processed.max() - X_processed.min())
        
        predictions = self.forward_pass(X_processed, self.weights)
        return (predictions > 0.5).astype(int)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate accuracy.
        """
        predictions = self.predict(X)
        y_processed = y.copy()
        if np.min(y_processed) == -1:
            y_processed = (y_processed + 1) / 2
        return np.mean(predictions == y_processed)
    
    def create_circuit_visualization(self) -> Tuple[QuantumCircuit, QuantumCircuit, QuantumCircuit]:
        """
        Create separate circuits for visualization: feature map, ansatz, and combined.
        
        Returns:
            Tuple of (feature_map, ansatz, combined_circuit)
        """
        # Feature map circuit
        feature_map = QuantumCircuit(2, name='Feature Map')
        x_param = Parameter('x')
        y_param = Parameter('y')
        feature_map.ry(2 * np.pi * x_param, 0)
        feature_map.ry(2 * np.pi * y_param, 1)
        
        # Ansatz circuit
        ansatz = QuantumCircuit(2, name='Ansatz')
        theta_params = [Parameter(f'θ_{i}') for i in range(4)]
        ansatz.ry(theta_params[0], 0)
        ansatz.rx(theta_params[1], 0)
        ansatz.ry(theta_params[2], 1)
        ansatz.rx(theta_params[3], 1)
        
        # Combined circuit with barrier
        combined = QuantumCircuit(2, name='QNN Circuit')
        combined.compose(feature_map, inplace=True)
        combined.barrier()  # Visual separator
        combined.compose(ansatz, inplace=True)
        
        return feature_map, ansatz, combined
    
    def save_circuit_diagram(self, timestamp: str):
        """
        Save simple circuit diagram with feature map and ansatz separated by a barrier.
        """
        try:
            feature_map, ansatz, combined = self.create_circuit_visualization()
            
            # Create figure with matplotlib backend - clean, no labels
            fig = circuit_drawer(combined, output='mpl', style='iqp', fold=-1)
            
            # Save the circuit diagram
            circuit_file = f"results/qnn_circuit_{timestamp}.png"
            fig.savefig(circuit_file, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close(fig)
            
            print(f"  🔧 Circuit diagram: {circuit_file}")
            
        except Exception as e:
            print(f"  ⚠️  Could not save circuit diagram: {e}")
            # Fallback to text representation
            try:
                _, _, combined = self.create_circuit_visualization()
                circuit_text_file = f"results/qnn_circuit_{timestamp}.txt"
                with open(circuit_text_file, 'w', encoding='utf-8') as f:
                    f.write(str(combined))
                print(f"  📝 Circuit text: {circuit_text_file}")
            except Exception as e2:
                print(f"  ❌ Could not save circuit: {e2}")
    
    def save_results(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
        """
        Save training results and plots.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("results", exist_ok=True)
        
        # Save loss plot (epoch losses only)
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.epoch_losses) + 1), self.epoch_losses, 'b-', linewidth=2, marker='o', markersize=6)
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title('Simple QNN Training: MSE Loss after Each Epoch')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        # Annotate final epoch loss
        if self.epoch_losses:
            plt.annotate(f'Final Epoch Loss: {self.epoch_losses[-1]:.6f}',
                        xy=(len(self.epoch_losses), self.epoch_losses[-1]),
                        xytext=(0.7, 0.9), textcoords='axes fraction',
                        arrowprops=dict(arrowstyle='->', color='red'),
                        fontsize=12, color='red')
        
        loss_plot_file = f"results/simple_qnn_loss_{timestamp}.png"
        plt.savefig(loss_plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save results summary
        train_acc = self.score(X_train, y_train)
        val_acc = self.score(X_val, y_val)
        
        results_text = f"""Simple QNN Training Results
{'='*40}
Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Configuration:
- Architecture: 2-qubit QNN
- Feature map: Ry(2πx), Ry(2πy)
- Ansatz: Ry(θ) + Rx(θ) on each qubit
- Observable: ZZ projector
- Loss: MSE (Mean Squared Error)
- Optimizer: Adam with parameter shift

Training:
- Epochs: {self.epochs}
- Samples per epoch: {len(X_train)} (all samples)
- Weight updates per epoch: {len(X_train)}
- Total weight updates: {len(self.loss_history)}
- Learning rate: {self.learning_rate}
- Validation samples: {len(X_val)}

Results:
- Training accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)
- Validation accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)
- Final epoch loss: {f"{self.epoch_losses[-1]:.6f}" if self.epoch_losses else "N/A"}
- Initial epoch loss: {f"{self.epoch_losses[0]:.6f}" if self.epoch_losses else "N/A"}
- Epoch loss improvement: {f"{(self.epoch_losses[0] - self.epoch_losses[-1]):.6f}" if len(self.epoch_losses) > 1 else "N/A"}
"""
        
        results_file = f"results/simple_qnn_results_{timestamp}.txt"
        with open(results_file, 'w', encoding='utf-8') as f:
            f.write(results_text)
        
        # Save both sample and epoch loss histories
        np.save(f"results/simple_qnn_sample_loss_history_{timestamp}.npy", np.array(self.loss_history))
        np.save(f"results/simple_qnn_epoch_loss_history_{timestamp}.npy", np.array(self.epoch_losses))
        
        # Save circuit diagram
        self.save_circuit_diagram(timestamp)
        
        print(f"Results saved:")
        print(f"  📊 Loss plot: {loss_plot_file}")
        print(f"  📋 Summary: {results_file}")
        print(f"  💾 Sample loss data: results/simple_qnn_sample_loss_history_{timestamp}.npy")
        print(f"  💾 Epoch loss data: results/simple_qnn_epoch_loss_history_{timestamp}.npy")


def main():
    """
    Main function - exactly as requested.
    """
    print("🎯 Simple QNN Implementation")
    print("=" * 35)
    
    # Step 1: Generate 500 samples, 80/20 split using line_qnns.py
    print("📊 Generating data using line_qnns.py...")
    X_all, y_all = generate_line_dataset(num_samples=500, seed=42)
    
    # 80/20 split
    split_idx = int(0.8 * len(X_all))
    X_train = X_all[:split_idx]
    y_train = y_all[:split_idx]
    X_val = X_all[split_idx:]
    y_val = y_all[split_idx:]
    
    print(f"✓ Generated {len(X_train)} training, {len(X_val)} validation samples")
    print(f"✓ Training class distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    
    # Step 2: Create and train simple QNN
    print(f"\n🧠 Creating Simple QNN...")
    qnn = SimpleQNN(epochs=1, learning_rate=0.001)  # 15 epochs × 400 samples = 6000 weight updates
    
    print("✓ Feature map: Ry(2πx), Ry(2πy)")
    print("✓ Ansatz: Ry(θ) + Rx(θ) on each qubit")
    print("✓ Loss: MSE with ZZ projector")
    print("✓ Optimizer: Adam with parameter shift")
    
    # Step 3: Train
    print(f"\n🏋️ Training...")
    qnn.fit(X_train, y_train)
    
    # Step 4: Evaluate
    train_acc = qnn.score(X_train, y_train)
    val_acc = qnn.score(X_val, y_val)
    
    print(f"\n📈 Results:")
    print(f"  🎯 Training Accuracy: {train_acc:.3f}")
    print(f"  ✅ Validation Accuracy: {val_acc:.3f}")
    print(f"  📉 Final Loss: {qnn.loss_history[-1]:.6f}")
    
    # Step 5: Save results
    print(f"\n💾 Saving results...")
    qnn.save_results(X_train, y_train, X_val, y_val)
    
    print(f"\n✅ Simple QNN training completed!")


if __name__ == "__main__":
    main()