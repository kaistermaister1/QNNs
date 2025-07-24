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
from star_eval import evaluate_model, visualize_decision_boundary

# ─── Configuration ───
RSEED = 4
EPS = 1e-10  # Small epsilon for numerical stability

# ─── CLI Arguments ───
parser = argparse.ArgumentParser("Custom SPQC Binary Classifier")
parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
parser.add_argument("--cpus", type=int, default=None, help="Number of CPUs (default: all available)")
parser.add_argument("--lr", type=float, default=0.01, help="Learning rate for Adam optimizer")
parser.add_argument("--visualize-boundary", action="store_true",
    help="Generate and save decision boundary plots after training.")
parser.add_argument("--boundary-resolution", type=int, default=64,
    help="Grid resolution for boundary visualization (default: 64)")
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
    input_params = [p for p in params if p.name.startswith('zinput_theta')]
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
    p_out = float(result[0].data.evs)
    p_in = float(result[1].data.evs)
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
    observables = projectors  # [P_out, P_in]
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

class SPQCModel:
    """Wrapper model class to make our circuit compatible with star_eval.evaluate_model"""
    
    def __init__(self, circuit, projectors, param_indices):
        self.circuit = circuit
        self.projectors = projectors
        self.param_indices = param_indices
        
    def forward(self, x, theta):
        """
        Forward pass that returns address amplitudes compatible with star_eval.
        
        For projector-based approach, we need to compute the probabilities and
        convert them to address amplitudes format expected by star_eval.
        
        Args:
            x: Input features [x_coord, y_coord]
            theta: Parameter values
            
        Returns:
            numpy.ndarray: Address amplitudes (simulated from probabilities)
        """
        estimator = StatevectorEstimator()
        
        # Get parameters
        params = list(self.circuit.parameters)
        param_pos = {p: i for i, p in enumerate(params)}
        values = np.zeros(len(params))
        
        # Set input parameters
        input_params = [p for p in params if p.name.startswith('zinput_theta')]
        for i, param in enumerate(input_params):
            if i < len(x):
                values[param_pos[param]] = x[i]
        
        # Set trainable parameters
        for i, param_idx in enumerate(self.param_indices):
            if i < len(theta):
                values[param_idx] = theta[i]
        
        # Get probabilities from projectors
        pubs = [(self.circuit, self.projectors[0], values), (self.circuit, self.projectors[1], values)]
        
        result = estimator.run(pubs).result()
        p_out = float(result[0].data.evs)
        p_in = float(result[1].data.evs)
        
        # Convert to address amplitudes format (8 addresses for m=3)
        # p_out corresponds to addresses 0-3, p_in to addresses 4-7
        addr_amps = np.zeros(8, dtype=complex)
        
        # Distribute probabilities evenly among addresses in each class
        # Convert probabilities to amplitudes (sqrt)
        amp_out = np.sqrt(p_out / 4) if p_out > 0 else 0
        amp_in = np.sqrt(p_in / 4) if p_in > 0 else 0
        
        # Fill address amplitudes
        addr_amps[0:4] = amp_out  # outside class (addresses 0-3)
        addr_amps[4:8] = amp_in   # inside class (addresses 4-7)
        
        addr_norm = np.linalg.norm(addr_amps)
        if addr_norm > 0:
            addr_amps = addr_amps / addr_norm
        
        return addr_amps



def main():
    print(f"Custom SPQC Binary Classifier")
    print(f"CPUs: {N_CPUS}")
    
    # ─── Data Loading ───
    X_train, X_test, y_train, y_test, boundary_path = get_star_data(200)
    
    # ─── Circuit Setup ───
    t, m, n, r = 0, 3, 2, 1  # SPQC parameters
    model_name = f"t{t}_m{m}_n{n}_r{r}"
    print(f"Circuit: t={t}, m={m}, n={n}, r={r} -> {t+m+n*r+1} qubits")
    
    # Create circuit without measurements for gradient computation
    frame = create_spqc_circuit(t=t, m=m, n=n, r=r)
    circuit = QuantumCircuit(frame.num_qubits)
    for inst in frame.data:
        if inst.operation.name != "measure":
            circuit.append(inst.operation, inst.qubits)
    
    # Transpile circuit
    circuit = transpile(circuit, optimization_level=1)

    # Print parameter indices and names after transpilation
    params = list(circuit.parameters)
    print("Parameter indices and names after transpilation:")
    for idx, param in enumerate(params):
        print(f"  {idx:2d}: {param.name}")
    
    # ─── Parameter Setup ───
    # Get trainable parameter indices in the same order as create_random_weights
    # (model parameters first, then address parameters)
    params = list(circuit.parameters)
    model_indices = [i for i, param in enumerate(params) if param.name.startswith('model')]
    address_indices = [i for i, param in enumerate(params) if param.name.startswith('address_theta')]
    param_indices = model_indices + address_indices  # model first, then address (matches create_random_weights)
    
    # Initialize weights
    np.random.seed(RSEED)
    theta = create_random_weights(circuit, seed=RSEED)
    initial_theta_snapshot = theta.copy()
    
    # ─── Projector Setup ───
    projectors = create_binary_projectors(t, m, n, r)
    
    # ─── Model Wrapper Setup ───
    spqc_model = SPQCModel(circuit, projectors, param_indices)
    
    # Convert scalar labels to one-hot for binary classification (only for evaluation)
    # 0 (inside) -> [1, 0], 1 (outside) -> [0, 1]
    y_test_onehot = np.eye(2)[y_test.astype(int)]
    
    # ─── Initial Evaluation ───
    initial_acc = evaluate_model(spqc_model, theta, X_test, y_test_onehot, 'binary', "Initial (Random Weights)")
    
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
    
    for epoch in tqdm(range(args.epochs), desc='Epoch'):
        epoch_start = time.time()
        
        # Shuffle training indices
        indices = np.random.permutation(len(X_train))
        epoch_loss_acc = 0.0
        epoch_grads = []  # Collect gradients for gradient norm calculation
        
        num_workers = 1 if pool is None else N_CPUS
        
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
                epoch_grads.append(grad)
                m1 = beta1 * m1 + (1 - beta1) * grad
                v1 = beta2 * v1 + (1 - beta2) * (grad**2)
                m1_hat = m1 / (1 - beta1**step)
                v1_hat = v1 / (1 - beta2**step)
                theta -= lr * m1_hat / (np.sqrt(v1_hat) + 1e-8)
                epoch_loss_acc += loss

        mean_epoch_loss = epoch_loss_acc / len(X_train)
        losses.append(mean_epoch_loss)
        
        # Calculate gradient norm for the epoch
        if len(epoch_grads) > 0:
            mean_grad = np.mean(epoch_grads, axis=0)
            grad_norm = np.linalg.norm(mean_grad)
        else:
            grad_norm = 0.0
        
        tqdm.write(f"Epoch {epoch+1: >4}: Loss {mean_epoch_loss:.4f} | Grad norm {grad_norm:.4f}")
    
    # ─── Final Evaluation ───
    print(f"\nTraining complete! Final loss: {losses[-1]:.4f}")
    
    final_acc = evaluate_model(spqc_model, theta, X_test, y_test_onehot, 'binary', "Final (Trained Weights)")
    
    # ─── Save Results ───
    os.makedirs('plots', exist_ok=True)
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(losses)+1), losses, 'o-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Binary Cross-Entropy Loss')
    plt.title(f'Training Loss (Custom SPQC {model_name})')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    loss_plot_path = f'plots/loss_{model_name}.png'
    plt.savefig(loss_plot_path, dpi=300)
    print(f"Loss plot saved to {loss_plot_path}")
    
    # ───────── Visualize Decision Boundary ─────────
    if args.visualize_boundary:
        print("\n" + "="*50)
        print("Visualizing Decision Boundaries...")
        print("="*50)
        try:
            # --- Initial Boundary ---
            print("Visualizing initial boundary...")
            visualize_decision_boundary(
                spqc_model, initial_theta_snapshot, m, X_train, np.eye(2)[y_train.astype(int)], 'binary',
                boundary=boundary_path,
                title=f'Initial Decision Boundary (Custom {model_name})',
                resolution=args.boundary_resolution,
                save_path=f'plots/initboundary_{model_name}.png'
            )

            # --- Final Boundary ---
            print("Visualizing final boundary...")
            visualize_decision_boundary(
                spqc_model, theta, m, X_train, np.eye(2)[y_train.astype(int)], 'binary',
                boundary=boundary_path,
                title=f'Final Decision Boundary (Custom {model_name})',
                resolution=args.boundary_resolution,
                save_path=f'plots/finalboundary_{model_name}.png'
            )
            
        except ImportError:
            print("\nWarning: Could not import 'visualize_decision_boundary' from 'star_eval.py'.")
            print("         Please ensure the file exists and is in the correct path.")
        except Exception as e:
            print(f"\nAn error occurred during boundary visualization: {e}")
    
    # Save weights
    os.makedirs('weights', exist_ok=True)
    weights_path = f'weights/weights_{model_name}.npz'
    np.savez(weights_path, 
             final_weights=theta, 
             loss_history=losses,
             initial_accuracy=initial_acc,
             final_accuracy=final_acc)
    print(f"Weights saved to {weights_path}")

if __name__ == "__main__":
    main() 