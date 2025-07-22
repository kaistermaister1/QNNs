"""
star_train_custom.py - Custom SPQC Training Implementation
========================================================
Binary classification of star data using SPQC with binary projectors and Adam optimizer.
"""

import argparse
import os
import sys
import time
import multiprocessing as mp
from typing import Tuple, List
import warnings

import numpy as np

warnings.filterwarnings('ignore', category=UserWarning, module='qiskit_algorithms')
warnings.filterwarnings('ignore', message='.*Casting complex values to real.*', module='qiskit_algorithms')
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm import tqdm

from qiskit import QuantumCircuit, transpile
from qiskit.primitives import StatevectorEstimator
from qiskit_algorithms.gradients import ReverseEstimatorGradient
from qiskit.quantum_info import SparsePauliOp, Statevector

# ───────── project imports ─────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT))
from star_data import get_star_data
from star_spqc import create_spqc_circuit, create_random_weights

# ─── Configuration ───
RSEED = 42
EPS = 1e-10  # Small epsilon for numerical stability

# ─── CLI Arguments ───
parser = argparse.ArgumentParser("Custom SPQC Binary Classifier")
parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
parser.add_argument("--cpus", type=int, default=None, help="Number of CPUs (default: all available)")
parser.add_argument("--lr", type=float, default=0.01, help="Learning rate for Adam optimizer")
args = parser.parse_args()

# Resource configuration
N_CPUS = args.cpus or mp.cpu_count()

def create_binary_projectors(t: int, m: int, n: int, r: int) -> List[SparsePauliOp]:
    """
    Create binary projection operators according to mathematical specification.
    
    Returns:
        [P_out, P_in] where P_out projects to addresses 0-(2^m/2-1), P_in to (2^m/2)-(2^m-1)
    """
    N = t + m + n*r + 1  # Total qubits
    
    # Basic projectors: |0⟩⟨0| = (I+Z)/2, |1⟩⟨1| = (I-Z)/2
    p0 = SparsePauliOp("Z", coeffs=[0.5]) + SparsePauliOp("I", coeffs=[0.5])
    p1 = SparsePauliOp("I", coeffs=[0.5]) - SparsePauliOp("Z", coeffs=[0.5])
    
    def create_address_projector(address_states: List[int]) -> SparsePauliOp:
        """Create projector that sums over multiple address states"""
        projectors = []
        
        for addr_state in address_states:
            proj_list = [None] * N
            
            # Term qubits projected to |0⟩
            for i in range(t):
                proj_list[i] = p0
                
            # Address qubits projected based on addr_state
            for i in range(m):
                q_idx = t + i
                if ((addr_state >> i) & 1) == 0:
                    proj_list[q_idx] = p0
                else:
                    proj_list[q_idx] = p1
                    
            # Data qubits projected to |0⟩
            for i in range(n*r):
                q_idx = t + m + i
                proj_list[q_idx] = p0
                
            # Ancilla qubit projected to |0⟩
            proj_list[t + m + n*r] = p0
            
            # Tensor product (reverse order for Qiskit little-endian)
            rev_proj_list = proj_list[::-1]
            full_projector = rev_proj_list[0]
            for i in range(1, N):
                full_projector = full_projector.tensor(rev_proj_list[i])
            
            projectors.append(full_projector.simplify())
        
        # Sum all projectors for this class
        if len(projectors) == 1:
            return projectors[0]
        
        result = projectors[0]
        for proj in projectors[1:]:
            result = result + proj
        
        return result.simplify()
    
    # S_out = {0,1,2,3}, S_in = {4,5,6,7} for m=3
    num_addr_states = 2**m
    half = num_addr_states // 2
    
    out_states = list(range(0, half))        # [0,1,2,3]
    in_states = list(range(half, num_addr_states))  # [4,5,6,7]
    
    P_out = create_address_projector(out_states)
    P_in = create_address_projector(in_states)
    
    return [P_out, P_in]

def compute_loss_and_gradient(estimator, gradient_calc, circuit, projectors, 
                             x: np.ndarray, y: int, theta: np.ndarray, 
                             param_indices: List[int]) -> Tuple[float, np.ndarray]:
    """
    Compute binary cross-entropy loss and gradient for a single sample.
    
    Args:
        estimator: Qiskit estimator
        gradient_calc: ReverseEstimatorGradient calculator
        circuit: Parameterized quantum circuit
        projectors: [P_out, P_in] projection operators
        x: Input features [x_coord, y_coord]
        y: Target label (0=inside, 1=outside)
        theta: Parameter values
        param_indices: Indices of trainable parameters
        
    Returns:
        (loss, gradient) tuple
    """
    # Create parameter values: [input_params..., model_params..., address_params...]
    params = list(circuit.parameters)
    param_pos = {p: i for i, p in enumerate(params)}
    values = np.zeros(len(params))
    
    # Set input parameters
    input_params = [p for p in params if p.name.startswith('input_theta')]
    for i, param in enumerate(input_params):
        if i < len(x):
            values[param_pos[param]] = x[i]
    
    # Set trainable parameters
    for i, param_idx in enumerate(param_indices):
        if i < len(theta):
            values[param_idx] = theta[i]
    
    # Compute expectation values for both projectors
    pubs = [(circuit, projectors[0], values), (circuit, projectors[1], values)]
    
    # Get probabilities
    result = estimator.run(pubs).result()
    p_out = result[0].data.evs
    p_in = result[1].data.evs
    p_out = np.clip(p_out, EPS, 1.0 - EPS)
    p_in = np.clip(p_in, EPS, 1.0 - EPS)
    
    # Normalize probabilities
    total_prob = p_out + p_in
    if total_prob > EPS:
        p_out = p_out / total_prob
        p_in = p_in / total_prob
    
    # Binary cross-entropy loss
    if y == 0:  # Inside class
        loss = -np.log(p_in + EPS)
    else:  # Outside class (y == 1)
        loss = -np.log(p_out + EPS)
    
    # Compute gradients
    circuits = [circuit, circuit]
    observables = projectors
    parameter_values = [values, values]
    grad_result = gradient_calc.run(circuits, observables, parameter_values).result()
    grad_out = grad_result.gradients[0]
    grad_in = grad_result.gradients[1]
    
    # Chain rule for binary cross-entropy
    if y == 0:  # Inside class
        # ∂L/∂θ = -(∂p_in/∂θ) / p_in
        gradient = -grad_in[param_indices] / (p_in + EPS)
    else:  # Outside class
        # ∂L/∂θ = -(∂p_out/∂θ) / p_out
        gradient = -grad_out[param_indices] / (p_out + EPS)
    
    return float(loss), gradient

# Global variables for worker processes
g_estimator = None
g_gradient_calc = None
g_circuit = None
g_projectors = None
g_param_indices = None
g_X_train = None
g_y_train = None

def worker_init(circuit, projectors, param_indices, X_train, y_train):
    """Initialize worker process with shared objects"""
    global g_estimator, g_gradient_calc, g_circuit, g_projectors, g_param_indices, g_X_train, g_y_train
    
    g_estimator = StatevectorEstimator()
    g_gradient_calc = ReverseEstimatorGradient(g_estimator)
    g_circuit = circuit
    g_projectors = projectors
    g_param_indices = param_indices
    g_X_train = X_train
    g_y_train = y_train

def compute_sample_loss_grad(theta: np.ndarray, sample_idx: int) -> Tuple[float, np.ndarray]:
    """Compute loss and gradient for a single sample (worker function)"""
    x = g_X_train[sample_idx]
    y = int(g_y_train[sample_idx])
    
    return compute_loss_and_gradient(
        g_estimator, g_gradient_calc, g_circuit, g_projectors,
        x, y, theta, g_param_indices
    )

def evaluate_model(circuit, projectors, param_indices, theta, X_test, y_test, stage_name=""):
    """
    Evaluate model performance on test set with detailed per-class breakdown.
    
    Args:
        circuit: Quantum circuit
        projectors: [P_out, P_in] projection operators  
        param_indices: Indices of trainable parameters
        theta: Parameter values
        X_test: Test features
        y_test: Test labels
        stage_name: Description for printing (e.g., "Initial", "Final")
    """
    estimator = StatevectorEstimator()
    
    # Track predictions per class
    class_correct = {0: 0, 1: 0}  # inside=0, outside=1
    class_total = {0: 0, 1: 0}
    class_predictions = {0: [], 1: []}  # Store all predictions per true class
    
    for i in range(len(X_test)):
        x = X_test[i]
        y_true = int(y_test[i])
        
        # Get parameters
        params = list(circuit.parameters)
        param_pos = {p: i for i, p in enumerate(params)}
        values = np.zeros(len(params))
        
        # Set input parameters
        input_params = [p for p in params if p.name.startswith('input_theta')]
        for j, param in enumerate(input_params):
            if j < len(x):
                values[param_pos[param]] = x[j]
        
        # Set trained parameters
        for j, param_idx in enumerate(param_indices):
            if j < len(theta):
                values[param_idx] = theta[j]
        
        # Get predictions
        pubs = [(circuit, projectors[0], values), (circuit, projectors[1], values)]
        result = estimator.run(pubs).result()
        p_out = result[0].data.evs
        p_in = result[1].data.evs
        
        y_pred = 0 if p_in > p_out else 1
        
        # Track results
        class_total[y_true] += 1
        class_predictions[y_true].append(y_pred)
        if y_pred == y_true:
            class_correct[y_true] += 1
    
    # Calculate per-class accuracies
    class_0_acc = class_correct[0] / class_total[0] if class_total[0] > 0 else 0
    class_1_acc = class_correct[1] / class_total[1] if class_total[1] > 0 else 0
    overall_acc = (class_correct[0] + class_correct[1]) / len(X_test)
    
    # Print detailed results
    print(f"\n{'='*50}")
    print(f"{stage_name} Model Evaluation")
    print(f"{'='*50}")
    print(f"Class 0 (Inside):  {class_correct[0]:2d}/{class_total[0]:2d} correct ({class_0_acc:.3f})")
    print(f"Class 1 (Outside): {class_correct[1]:2d}/{class_total[1]:2d} correct ({class_1_acc:.3f})")
    print(f"Overall:           {class_correct[0] + class_correct[1]:2d}/{len(X_test):2d} correct ({overall_acc:.3f})")
    print(f"{'='*50}")
    
    return overall_acc

def main():
    print(f"Custom SPQC Binary Classifier")
    print(f"CPUs: {N_CPUS}")
    
    # ─── Data Loading ───
    print("\nLoading star data...")
    X_train, X_test, y_train, y_test, star_path = get_star_data(200)
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"Training labels: {np.bincount(y_train.astype(int))} (inside=0, outside=1)")
    
    # ─── Circuit Setup ───
    print("\nSetting up quantum circuit...")
    t, m, n, r = 0, 3, 2, 1  # SPQC parameters
    print(f"Circuit: t={t}, m={m}, n={n}, r={r} -> {t+m+n*r+1} qubits")
    
    # Create circuit without measurements for gradient computation
    frame = create_spqc_circuit(t=t, m=m, n=n, r=r)
    circuit = QuantumCircuit(frame.num_qubits)
    for inst in frame.data:
        if inst.operation.name != "measure":
            circuit.append(inst.operation, inst.qubits)
    
    # Transpile circuit
    print("Transpiling circuit...")
    circuit = transpile(circuit, optimization_level=1)
    
    # ─── Parameter Setup ───
    # Get trainable parameter indices (model + address parameters)
    params = list(circuit.parameters)
    param_indices = []
    for i, param in enumerate(params):
        if param.name.startswith('model') or param.name.startswith('address_theta'):
            param_indices.append(i)
    
    print(f"Total parameters: {len(params)}, Trainable: {len(param_indices)}")
    
    # Initialize weights
    np.random.seed(RSEED)
    theta = create_random_weights(frame, seed=RSEED)
    print(f"Initial weights shape: {theta.shape}")
    
    # ─── Projector Setup ───
    print("Creating binary projectors...")
    projectors = create_binary_projectors(t, m, n, r)
    print(f"Created {len(projectors)} projectors (out, in)")
    
    # ─── Initial Evaluation ───
    initial_acc = evaluate_model(circuit, projectors, param_indices, theta, X_test, y_test, "Initial (Random Weights)")
    
    # ─── Training Setup ───
    print(f"\nStarting training for {args.epochs} epochs...")
    
    # Adam optimizer parameters
    beta1, beta2 = 0.9, 0.999
    lr = args.lr
    m1 = np.zeros_like(theta)
    v1 = np.zeros_like(theta)
    
    # Training history
    losses = []
    
    # Setup parallel processing
    if N_CPUS == 1:
        # Single process - initialize directly
        worker_init(circuit, projectors, param_indices, X_train, y_train)
        pool = None
    else:
        # Multi-process
        pool = Parallel(
            n_jobs=N_CPUS,
            initializer=worker_init,
            initargs=(circuit, projectors, param_indices, X_train, y_train),
            backend="multiprocessing",
            prefer="processes"
        )
    
    # ─── Training Loop ───
    step = 0
    training_start_time = time.time()
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        # Shuffle training indices
        indices = np.random.permutation(len(X_train))
        epoch_loss_acc = 0.0
        
        num_workers = 1 if pool is None else N_CPUS
        
        with tqdm(total=len(X_train), desc=f"Epoch {epoch+1}", unit="sample") as pbar:
            # Process data in chunks of size num_workers
            for i in range(0, len(indices), num_workers):
                chunk_indices = indices[i:i+num_workers]
                
                if pool is None:
                    # Single process serial execution
                    results = [compute_sample_loss_grad(theta, idx) for idx in chunk_indices]
                else:
                    # Parallel computation of gradients for the chunk
                    results = pool(delayed(compute_sample_loss_grad)(theta, idx) for idx in chunk_indices)
                
                # Adam update for each sample in the processed chunk
                for loss, grad in results:
                    step += 1
                    m1 = beta1 * m1 + (1 - beta1) * grad
                    v1 = beta2 * v1 + (1 - beta2) * (grad**2)
                    m1_hat = m1 / (1 - beta1**step)
                    v1_hat = v1 / (1 - beta2**step)
                    theta -= lr * m1_hat / (np.sqrt(v1_hat) + 1e-8)
                    epoch_loss_acc += loss
                    pbar.update(1)

        mean_epoch_loss = epoch_loss_acc / len(X_train)
        losses.append(mean_epoch_loss)
        
        # Calculate timing
        epoch_time = time.time() - epoch_start
        total_elapsed = time.time() - training_start_time
        
        print(f"\nEpoch {epoch+1:2d} complete: Loss {mean_epoch_loss:.4f} | {epoch_time:.1f}s/epoch | Total {total_elapsed:.1f}s")
    
    # ─── Final Evaluation ───
    print(f"\nTraining complete! Final loss: {losses[-1]:.4f}")
    
    final_acc = evaluate_model(circuit, projectors, param_indices, theta, X_test, y_test, "Final (Trained Weights)")
    
    # ─── Save Results ───
    os.makedirs('plots', exist_ok=True)
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(losses)+1), losses, 'o-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Binary Cross-Entropy Loss')
    plt.title('Training Loss (Custom SPQC)')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('plots/custom_training_loss.png', dpi=300)
    print("Loss plot saved to plots/custom_training_loss.png")
    
    # Save weights
    os.makedirs('weights', exist_ok=True)
    np.savez('weights/custom_model_weights.npz', 
             final_weights=theta, 
             loss_history=losses,
             initial_accuracy=initial_acc,
             final_accuracy=final_acc)
    print("Weights saved to weights/custom_model_weights.npz")

if __name__ == "__main__":
    main() 