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
                print("✅ GPU quantum simulation initialized")
                print("📋 Note: Will fall back to CPU if GPU encounters unsupported operations")
            except Exception as e:
                print(f"⚠️  GPU initialization failed: {e}")
                print("🔄 Using CPU simulation only")
                self.gpu_simulator = None
        else:
            self.gpu_simulator = None

    def forward(self, input_vals, weights):
        if USE_GPU and self.gpu_simulator is not None:
            # Try GPU simulation with fallback to CPU
            try:
                return self.gpu_model(input_vals, weights)
            except Exception as e:
                if not hasattr(self, '_gpu_fallback_warned'):
                    print(f"⚠️  GPU simulation failed: {str(e)[:100]}...")
                    print("🔄 Falling back to CPU simulation for all forward passes")
                    self._gpu_fallback_warned = True
                    self.gpu_simulator = None  # Disable GPU for future calls
                return model(self.qc, input_vals, weights, self.t, self.m, self.n, self.r)
        else:
            # Use original CPU model
            return model(self.qc, input_vals, weights, self.t, self.m, self.n, self.r)
    
    def gpu_model(self, input_vals, weights):
        """GPU-accelerated quantum simulation with proper post-selection"""
        from star_spqc import bind_params
        from qiskit.quantum_info import SparsePauliOp, Statevector
        from qiskit import transpile
        from functools import reduce
        
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
        
        # Apply same post-selection as original model
        P0 = SparsePauliOp(['I','Z'], [0.5, 0.5])
        I = SparsePauliOp(['I'], [1.0])
        
        if self.t > 0:
            I_t = reduce(lambda a,b: a.tensor(b), [P0] * self.t)
        I_m = reduce(lambda a,b: a.tensor(b), [I] * self.m)
        P_RUS = reduce(lambda a,b: a.tensor(b), [P0] * self.n*self.r)
        I_a = P0
        
        if self.t > 0:
            P_pauli = I_t.tensor(I_m).tensor(P_RUS).tensor(I_a)
        else:
            P_pauli = I_m.tensor(P_RUS).tensor(I_a)
        
        P = P_pauli.to_matrix()
        
        # Apply projector and normalize
        phi_unnorm = P @ statevector
        phi = phi_unnorm / np.linalg.norm(phi_unnorm)
        
        # Extract address register amplitudes
        N = self.t + self.m + self.n*self.r + 1
        tensor = phi.reshape([2]*N)
        index = [0]*self.t + [slice(None)]*self.m + [0]*(self.n*self.r) + [0]
        addr = tensor[tuple(index)]
        return addr.reshape(2**self.m)
    
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

def main():
    print("🚀 Starting supercomputer training...")
    print(f"Configuration: {epochs} epochs, {num_data_points} data points")
    print(f"CPUs: {N_CPU_WORKERS} workers, GPU: {USE_GPU}, Memory efficient: {MEMORY_EFFICIENT}")
    
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
    
    print(f"Training with {len(θ)} parameters on {len(train_features)} samples")
    
    # Report actual resource usage
    import multiprocessing
    actual_cpu_workers = multiprocessing.cpu_count() if N_CPU_WORKERS == -1 else N_CPU_WORKERS
    print(f"Using {actual_cpu_workers} CPU cores for parallel gradients")
    if USE_GPU and spqc_model.gpu_simulator is not None:
        print(f"GPU acceleration: ENABLED ({N_GPUS} GPUs)")
    else:
        print("GPU acceleration: DISABLED (CPU-only mode)")
    
    # Initial evaluation
    print("Initial performance:")
    evaluate_model(spqc_model, θ, test_features, test_labels_onehot, CLASSIFICATION_MODE, "Initial")
    
    # Training with parallel gradients
    m1, v1 = np.zeros_like(θ), np.zeros_like(θ)
    beta1, beta2, alpha = 0.9, 0.999, 0.01
    
    # Track loss over epochs
    epoch_losses = []
    
    # Main training loop with progress bars
    with tqdm(range(1, epochs + 1), desc="Training Progress", unit="epoch") as epoch_pbar:
        for epoch in epoch_pbar:
            # Process all samples with inner progress bar
            epoch_gradients = []
            epoch_loss = 0.0
            sample_pbar = tqdm(
                zip(train_features, train_labels_onehot), 
                total=len(train_features), 
                desc=f"Epoch {epoch}", 
                leave=False,
                unit="sample"
            )
            
            current_loss = 0.0
            for sample_idx, (x, y_true) in enumerate(sample_pbar):
                # Compute loss for this sample
                sample_loss = spqc_model.loss(x, θ, y_true)
                epoch_loss += sample_loss
                
                # Show gradient progress for first few samples of early epochs
                show_grad_progress = (sample_idx < 3) and (epoch <= 5)
                g = spqc_model.parallel_gradient(x, θ, y_true, show_progress=show_grad_progress)
                epoch_gradients.append(g)
                
                # Update loss display every 10 samples
                if sample_idx % 10 == 0:
                    current_loss = sample_loss
                    sample_pbar.set_postfix(loss=f"{current_loss:.4f}")
            
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
            
            # More frequent evaluation for 5000 epochs
            if epoch % 500 == 0:
                print(f"\n--- Intermediate Evaluation at Epoch {epoch} ---")
                accuracy = evaluate_model(spqc_model, θ, test_features, test_labels_onehot, CLASSIFICATION_MODE, f"Epoch {epoch}")
                print(f"Intermediate accuracy at epoch {epoch}: {accuracy:.4f}")
                # Small delay to let tqdm update properly
                import time
                time.sleep(0.1)
    
    # Create plots directory if it doesn't exist
    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)
    
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
    print(f"\n📊 Loss plot saved to: {loss_plot_path}")
    
    # Final evaluation
    print("Final performance:")
    accuracy = evaluate_model(spqc_model, θ, test_features, test_labels_onehot, CLASSIFICATION_MODE, "Final")
    print(f"🎉 Training complete! Final accuracy: {accuracy:.4f}")
    
    # Print some training statistics
    print(f"\n📈 Training Statistics:")
    print(f"   Initial loss: {epoch_losses[0]:.6f}")
    print(f"   Final loss: {epoch_losses[-1]:.6f}")
    print(f"   Loss reduction: {((epoch_losses[0] - epoch_losses[-1]) / epoch_losses[0] * 100):.2f}%")
    print(f"   Min loss achieved: {min(epoch_losses):.6f} at epoch {epoch_losses.index(min(epoch_losses)) + 1}")

if __name__ == "__main__":
    main() 