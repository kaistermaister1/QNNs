"""Simple SPQC Supercomputer Training - 64x faster gradients"""

import numpy as np
import sys
import os
from joblib import Parallel, delayed
from qiskit import QuantumCircuit
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add parent directory to path to import SPQC modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from star_data import get_star_data
from star_spqc import create_spqc_circuit, create_random_weights, model
from star_eval import evaluate_model, wedge_onehot

# Configuration  
CLASSIFICATION_MODE = 'binary'
RANDOM_SEED = 42
num_data_points = 300
epochs = 5000  # Extended training for better convergence

# Resource Configuration - MODIFY THESE TO CONTROL HARDWARE USAGE
N_CPU_WORKERS = -1      # CPU cores for gradients (-1 = use all available)
USE_GPU = False         # Set True to enable GPU acceleration (may fall back to CPU)
N_GPUS = 4              # Number of GPUs to use (matches SLURM --gres setting)
MEMORY_EFFICIENT = True # Enable memory optimization for long training

# Note: GPU acceleration is experimental for SPQC circuits due to custom gates
# If you encounter GPU errors, set USE_GPU = False for reliable CPU-only training

class ParallelSPQCModel:
    def __init__(self, qc, t, m, n, r):
        self.qc, self.t, self.m, self.n, self.r = qc, t, m, n, r
        
        # Setup GPU simulator if enabled
        if USE_GPU:
            try:
                from qiskit_aer import AerSimulator
                self.gpu_simulator = AerSimulator(method='statevector', device='GPU')
            except Exception:
                self.gpu_simulator = None
        else:
            self.gpu_simulator = None

    def forward(self, input_vals, weights):
        if USE_GPU and self.gpu_simulator is not None:
            # Try GPU simulation with fallback to CPU
            try:
                return self.gpu_model(input_vals, weights)
            except Exception:
                if not hasattr(self, '_gpu_fallback_warned'):
                    self._gpu_fallback_warned = True
                    self.gpu_simulator = None  # Disable GPU for future calls
                return model(self.qc, input_vals, weights, self.t, self.m, self.n, self.r)
        else:
            # Use original CPU model
            return model(self.qc, input_vals, weights, self.t, self.m, self.n, self.r)
    
    def gpu_model(self, input_vals, weights):
        """GPU-accelerated quantum simulation with tensor slicing post-selection"""
        from star_spqc import bind_params
        from qiskit.quantum_info import Statevector
        from qiskit import transpile
        
        # Bind parameters
        spqc = bind_params(self.qc, input_vals, weights)
        
        # Decompose circuit to basic gates for GPU compatibility
        try:
            decomposed_circuit = transpile(spqc, basis_gates=['cx', 'u1', 'u2', 'u3'], optimization_level=0)
        except:
            # If transpilation fails, try with more basic gates
            decomposed_circuit = transpile(spqc, basis_gates=['cx', 'id', 'rz', 'sx', 'x'], optimization_level=0)
        
        # Run on GPU with decomposed circuit
        job = self.gpu_simulator.run(decomposed_circuit, shots=1)
        statevector = job.result().get_statevector().data
        
        # Direct post‑selection via tensor slicing (same as updated star_spqc.py)
        N = self.t + self.m + self.n * self.r + 1                   # total qubits
        tensor = statevector.reshape([2] * N)   # view, no copy

        slice_spec = [0] * self.t                   \
                   + [slice(None)] * self.m         \
                   + [0] * (self.n * self.r)        \
                   + [0]                            # ancilla

        addr = tensor[tuple(slice_spec)].reshape(2 ** self.m)

        # Renormalise
        norm = np.linalg.norm(addr)
        return addr / norm if norm != 0 else addr
    
    def loss(self, x, θ, y_true_onehot):
        amps = self.forward(x, θ)
        diff = amps - y_true_onehot
        return float(np.mean(np.abs(diff)**2))
    
    def parallel_gradient(self, x, θ, y_true_onehot, shift=np.pi/2, show_progress=False):
        """Parallel gradient computation - CPU cores controlled by N_CPU_WORKERS"""
        def compute_param_grad(i):
            θp, θm = θ.copy(), θ.copy()
            θp[i] += shift; θm[i] -= shift
            lp = self.loss(x, θp, y_true_onehot)
            lm = self.loss(x, θm, y_true_onehot)
            return 0.5 * (lp - lm) / np.sin(shift)
        
        # Use configured number of CPU workers with optional progress bar
        if show_progress:
            # Create a custom joblib progress tracker
            param_indices = list(range(len(θ)))
            with tqdm(total=len(θ), desc=f"Computing gradients ({N_CPU_WORKERS} cores)", leave=False, unit="param") as pbar:
                def update_progress(*args):
                    pbar.update(1)
                    return compute_param_grad(*args)
                
                # Note: joblib doesn't directly support progress callbacks, so we'll use a simpler approach
                grads = Parallel(n_jobs=N_CPU_WORKERS)(delayed(compute_param_grad)(i) for i in param_indices)
                pbar.update(len(θ))  # Update all at once since joblib batches internally
        else:
            grads = Parallel(n_jobs=N_CPU_WORKERS)(delayed(compute_param_grad)(i) for i in range(len(θ)))
        
        return np.array(grads)

def efficient_mesh_predictions(spqc_model, θ, mesh_points, n_jobs=-1):
    """Parallel mesh prediction computation using all available CPUs/GPUs"""
    def predict_batch(points_batch):
        return [spqc_model.forward(point, θ) for point in points_batch]
    
    # Split mesh into batches for parallel processing
    batch_size = max(1, len(mesh_points) // (4 * abs(n_jobs) if n_jobs != -1 else 4 * os.cpu_count()))
    batches = [mesh_points[i:i+batch_size] for i in range(0, len(mesh_points), batch_size)]
    
    # Process batches in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(predict_batch)(batch) for batch in batches
    )
    
    # Flatten results
    mesh_predictions = []
    for batch_result in results:
        mesh_predictions.extend(batch_result)
    
    return np.array(mesh_predictions)

def visualize_parallel_decision_boundary(spqc_model, θ, m, test_features, test_labels_onehot, mode, boundary=None, title="Decision Boundary", save_path=None):
    """Efficient parallel decision boundary visualization with saving capability"""

    # Create mesh grid
    x_min, x_max = 0, 1
    y_min, y_max = 0, 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    mesh_points = np.column_stack([xx.ravel(), yy.ravel()])

    # Get predictions for mesh grid using parallel computation
    mesh_predictions = efficient_mesh_predictions(spqc_model, θ, mesh_points, n_jobs=N_CPU_WORKERS)

    plt.figure(figsize=(12, 10))
    ax = plt.gca()
    
    true_labels = np.argmax(test_labels_onehot, axis=1)

    if mode == 'wedge':
        # Plot learned decision regions
        Z = np.argmax(mesh_predictions, axis=1).reshape(xx.shape)
        cmap = plt.get_cmap('viridis', 2**m)
        plt.contourf(xx, yy, Z, cmap=cmap, alpha=0.6, levels=np.arange(-0.5, 2**m, 1))
        plt.colorbar(ticks=range(2**m), label='Predicted Wedge Class')

        # Plot true wedge boundaries
        angles = np.linspace(0, 2 * np.pi, 2**m + 1)
        for angle in angles:
            ax.plot([0.5, 0.5 + 0.7 * np.cos(angle)], [0.5, 0.5 + 0.7 * np.sin(angle)], 'k:', linewidth=2)
        
        # Plot test data
        plt.scatter(test_features[:, 0], test_features[:, 1], c=true_labels, cmap=cmap, edgecolors='k', s=50)

    elif mode == 'binary':
        # Only consider first 2 amplitudes for binary classification
        binary_probs = mesh_predictions[:, :2]
        # Normalize to get relative probabilities between the two classes
        binary_probs_normalized = binary_probs / (binary_probs.sum(axis=1, keepdims=True) + 1e-8)
        Z = binary_probs_normalized[:, 1].reshape(xx.shape)  # Probability of class 1 (outside star)
        
        contourf = plt.contourf(xx, yy, Z, levels=20, cmap='RdYlBu_r', alpha=0.8)
        plt.colorbar(contourf, label='P(Outside Star | First 2 Classes)')
        
        # Plot learned p=0.5 decision boundary
        plt.contour(xx, yy, Z, levels=[0.5], colors='green', linewidths=2.5)
        
        # Plot true star boundary
        if boundary:
            vertices = boundary.vertices
            plt.plot(vertices[:, 0], vertices[:, 1], 'k:', linewidth=3, label='True Boundary')
        
        # Plot test data
        inside = test_features[true_labels == 0]
        outside = test_features[true_labels == 1]
        plt.scatter(inside[:, 0], inside[:, 1], c='red', s=40, alpha=0.7, marker='s', label='Inside')
        plt.scatter(outside[:, 0], outside[:, 1], c='blue', s=40, alpha=0.7, marker='s', label='Outside')
        plt.legend()

    plt.xlim(0, 1); plt.ylim(0, 1); plt.grid(True, alpha=0.2)
    plt.title(title); plt.xlabel('X coordinate'); plt.ylabel('Y coordinate')
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    
    # Save the plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

    # Calculate accuracy on the mesh grid
    if mode == 'binary' and boundary is not None:
        true_mesh_labels = np.array([0 if boundary.contains_point(p) else 1 for p in mesh_points])
        pred_mesh_labels = np.argmax(mesh_predictions[:, :2], axis=1)
    elif mode == 'wedge':
        true_mesh_labels = np.argmax(wedge_onehot(mesh_points, m), axis=1)
        pred_mesh_labels = np.argmax(mesh_predictions, axis=1)
    else:
        true_mesh_labels = None
        pred_mesh_labels = None
        
    if true_mesh_labels is not None:
        mesh_accuracy = np.mean(pred_mesh_labels == true_mesh_labels)
    else:
        mesh_accuracy = None

    return mesh_accuracy

def main():
    print(f"Starting SPQC training: {epochs} epochs, {num_data_points} data points")
    
    # Load data
    train_features, test_features, train_labels, test_labels, boundary = get_star_data(num_data_points)
    np.random.seed(RANDOM_SEED)
    
    # Setup model
    m, r, n, t = 3, 1, 2, 0
    spqc_frame = create_spqc_circuit(t=t, m=m, n=n, r=r)
    qc_qnn = QuantumCircuit(spqc_frame.num_qubits)
    for instr in spqc_frame.data:
        if instr.operation.name != 'measure':
            qc_qnn.append(instr.operation, instr.qubits, instr.clbits)
    
    spqc_model = ParallelSPQCModel(qc_qnn, t, m, n, r)
    θ = create_random_weights(spqc_frame, seed=RANDOM_SEED)
    
    # Prepare labels
    train_labels_onehot = np.eye(8)[train_labels.astype(int)]
    test_labels_onehot = np.eye(8)[test_labels.astype(int)]
    
    # Report resource usage
    import multiprocessing
    actual_cpu_workers = multiprocessing.cpu_count() if N_CPU_WORKERS == -1 else N_CPU_WORKERS
    print(f"Using {actual_cpu_workers} CPU cores, GPU: {'ON' if USE_GPU and spqc_model.gpu_simulator else 'OFF'}")
    
    # Create plots directory early
    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    # Initial evaluation
    print("\nInitial evaluation:")
    evaluate_model(spqc_model, θ, test_features, test_labels_onehot, CLASSIFICATION_MODE, "Initial")
    
    # Initial decision boundary visualization
    initial_boundary_path = os.path.join(plots_dir, f"initial_decision_boundary_{CLASSIFICATION_MODE}_{num_data_points}pts.png")
    visualize_parallel_decision_boundary(
        spqc_model, θ, m, test_features, test_labels_onehot, 
        CLASSIFICATION_MODE, boundary, 
        title=f"Initial Decision Boundary ({CLASSIFICATION_MODE})", 
        save_path=initial_boundary_path
    )
    
    # Training with parallel gradients
    m1, v1 = np.zeros_like(θ), np.zeros_like(θ)
    beta1, beta2, alpha = 0.9, 0.999, 0.01
    
    # Track loss over epochs
    epoch_losses = []
    
    # Main training loop with progress bars
    with tqdm(range(1, epochs + 1), desc="Training Progress", unit="epoch") as epoch_pbar:
        for epoch in epoch_pbar:
            # Process all samples  
            epoch_gradients = []
            epoch_loss = 0.0
            
            for sample_idx, (x, y_true) in enumerate(zip(train_features, train_labels_onehot)):
                # Compute loss for this sample
                sample_loss = spqc_model.loss(x, θ, y_true)
                epoch_loss += sample_loss
                
                # Compute gradients
                g = spqc_model.parallel_gradient(x, θ, y_true, show_progress=False)
                epoch_gradients.append(g)
            
            # Average gradients
            g = np.mean(epoch_gradients, axis=0)
            
            # Adam update
            m1 = beta1 * m1 + (1 - beta1) * g
            v1 = beta2 * v1 + (1 - beta2) * (g**2)
            m_hat, v_hat = m1 / (1 - beta1**epoch), v1 / (1 - beta2**epoch)
            θ -= alpha * m_hat / (np.sqrt(v_hat) + 1e-8)
            
            # Calculate and store average epoch loss
            avg_epoch_loss = epoch_loss / len(train_features)
            epoch_losses.append(avg_epoch_loss)
            
            # Update epoch progress bar with current info
            epoch_pbar.set_postfix(
                loss=f"{avg_epoch_loss:.4f}",
                lr=f"{alpha:.3f}",
                progress=f"{epoch/epochs*100:.1f}%"
            )
            
            # Periodic evaluation
            if epoch % 1000 == 0:
                accuracy = evaluate_model(spqc_model, θ, test_features, test_labels_onehot, CLASSIFICATION_MODE, f"Epoch {epoch}")
                print(f"Epoch {epoch} accuracy: {accuracy:.4f}")
                import time
                time.sleep(0.1)
    
    # Plot and save loss curve
    plt.figure(figsize=(12, 8))
    plt.plot(range(1, epochs + 1), epoch_losses, 'b-', linewidth=2, marker='o', markersize=2)
    plt.xlabel('Epoch')
    plt.ylabel('Average Training Loss')
    plt.title(f'Parallel Training Loss Over Time\n({CLASSIFICATION_MODE} classification, {num_data_points} data points, {epochs} epochs)')
    plt.grid(True, alpha=0.3)
    
    # Add some additional formatting for the longer training
    if epochs > 1000:
        # Use log scale for x-axis if training is very long
        plt.xscale('log')
        plt.xlabel('Epoch (log scale)')
    
    plt.tight_layout()
    
    # Save the plot
    loss_plot_path = os.path.join(plots_dir, f"parallel_training_loss_{CLASSIFICATION_MODE}_{num_data_points}pts_{epochs}epochs.png")
    plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Final evaluation
    print("\nFinal evaluation:")
    accuracy = evaluate_model(spqc_model, θ, test_features, test_labels_onehot, CLASSIFICATION_MODE, "Final")
    
    # Final decision boundary visualization
    final_boundary_path = os.path.join(plots_dir, f"final_decision_boundary_{CLASSIFICATION_MODE}_{num_data_points}pts_{epochs}epochs.png")
    final_mesh_accuracy = visualize_parallel_decision_boundary(
        spqc_model, θ, m, test_features, test_labels_onehot, 
        CLASSIFICATION_MODE, boundary, 
        title=f"Final Decision Boundary ({CLASSIFICATION_MODE}, {epochs} epochs)", 
        save_path=final_boundary_path
    )
    
    # Training summary
    print(f"\nTraining complete:")
    print(f"   Loss: {epoch_losses[0]:.6f} → {epoch_losses[-1]:.6f}")
    print(f"   Test accuracy: {accuracy:.4f}")
    if final_mesh_accuracy is not None:
        print(f"   Mesh accuracy: {final_mesh_accuracy:.4f}")
    print(f"   Files saved: loss curve + 2 decision boundaries")

if __name__ == "__main__":
    main() 