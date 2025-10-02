"""
traditional_qnn_train.py - Custom QNN Training Implementation
=============================================================
Binary classification using traditional QNN (ZZFeatureMap + EfficientSU2) with cross-entropy loss and Adam optimizer.
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
from qiskit.circuit.library import ZZFeatureMap, EfficientSU2
from qiskit.primitives import StatevectorEstimator
from qiskit_algorithms.gradients import ReverseEstimatorGradient
from qiskit.quantum_info import SparsePauliOp

# ───────── project imports ─────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT))
from star_eval import evaluate_model, visualize_decision_boundary


# ─── Configuration ───
RSEED = 69
SHAPE = "circle"; CONDENSE = False  # Condense outside points near boundary
SAMPLES = 256
BATCH = 1  # Batch size for gradient/loss calculation
LOSS = "ce" # ce (cross-entropy) or mse (mean squared error)
model_name = f"{SHAPE}_traditional_qnn_{RSEED}_{SAMPLES}samples_b{BATCH}_{LOSS}{"_c"+ str(CONDENSE)[0] if (SHAPE == "star") else ""}"
EPS = 1e-10  # Small epsilon for numerical stability


# ─── CLI Arguments ───
parser = argparse.ArgumentParser("Traditional QNN Binary Classifier")
parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
parser.add_argument("--cpus", type=int, default=None, help="Number of CPUs (default: all available)")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for Adam optimizer")
parser.add_argument("--visualize-boundary", action="store_true",
    help="Generate and save decision boundary plots after training.")
parser.add_argument("--boundary-resolution", type=int, default=64,
    help="Grid resolution for boundary visualization (default: 64)")
args = parser.parse_args()

# Resource configuration
N_CPUS = args.cpus or mp.cpu_count()

def create_binary_projectors() -> List[SparsePauliOp]:
    """
    Returns properly normalized projectors for a 2-qubit circuit:
      P0 = |0><0| ⊗ I (projects first qubit to |0>, measures class 0)
      P1 = |1><1| ⊗ I (projects first qubit to |1>, measures class 1)
    """
    # Single-qubit projectors: |0><0| = (I + Z) / 2, |1><1| = (I - Z) / 2
    p0 = SparsePauliOp("Z", coeffs=[0.5]) + SparsePauliOp("I", coeffs=[0.5])
    p1 = SparsePauliOp("I", coeffs=[0.5]) - SparsePauliOp("Z", coeffs=[0.5])

    # Identity on second qubit
    I2 = SparsePauliOp("I", coeffs=[1.0])

    # Tensor products to form 2-qubit marginal projectors
    P0 = p0.tensor(I2)   # projects first qubit to |0>, ignores second
    P1 = p1.tensor(I2)   # projects first qubit to |1>, ignores second

    return [P0, P1]

def compute_loss_and_gradient(estimator, gradient_calc, circuit, projectors, 
                             x: np.ndarray, y: int, theta: np.ndarray, 
                             param_indices: List[int], mode: str = "ce") -> Tuple[float, np.ndarray]:
    """
    Compute loss and gradient for a single sample using traditional QNN.
    
    Args:
        estimator: Qiskit estimator for expectation value computation
        gradient_calc: Gradient calculator
        circuit: Quantum circuit (ZZFeatureMap + EfficientSU2)
        projectors: List of projection operators [P0, P1]
        x: Input features
        y: Target label (0 for inside, 1 for outside)
        theta: Model parameters
        param_indices: Indices of trainable parameters
        mode: Loss function mode - "ce" or "mse"
        
    Returns:
        Tuple of (loss_value, gradient_array)
    """
    # Create parameter values
    params = list(circuit.parameters)
    param_pos = {p: i for i, p in enumerate(params)}
    values = np.zeros(len(params))

    # Set input parameters (ZZFeatureMap parameters)
    input_params = [p for p in params if 'x' in p.name or 'input' in p.name]
    for i, param in enumerate(input_params):
        if i < len(x):
            values[param_pos[param]] = x[i]

    # Set trainable parameters (EfficientSU2 parameters)
    for i, param_idx in enumerate(param_indices):
        if i < len(theta):
            values[param_idx] = theta[i]

    # Compute expectation values for both projectors
    pubs = [(circuit, projectors[0], values), (circuit, projectors[1], values)]

    # Get probabilities
    result = estimator.run(pubs).result()
    p0_raw = float(result[0].data.evs)
    p1_raw = float(result[1].data.evs)

    # Normalize probabilities (for numerical stability)
    S_raw = p0_raw + p1_raw
    S = max(S_raw, EPS)
    p0_hat = p0_raw / S  # probability of class 0 (inside)
    p1_hat = p1_raw / S  # probability of class 1 (outside)

    if mode == "ce":
        # Binary cross-entropy loss
        if y == 0:  # Inside class
            loss = -np.log(p0_hat + EPS)
        else:  # Outside class (y == 1)
            loss = -np.log(p1_hat + EPS)
        
        # Compute gradients  
        circuits = [circuit, circuit]
        observables = projectors  # [P0, P1]
        parameter_values = [values, values]
        grad_result = gradient_calc.run(circuits, observables, parameter_values).result()
        grad_p0 = grad_result.gradients[0]
        grad_p1 = grad_result.gradients[1]

        # Chain rule for binary cross-entropy
        dp0_dθ = grad_p0[param_indices]
        dp1_dθ = grad_p1[param_indices]
        
        # If S is clamped, treat dS/dθ = 0
        if S_raw > EPS:
            dS_dθ = dp0_dθ + dp1_dθ
        else:
            dS_dθ = 0.0 * dp0_dθ  # same shape, all zeros
        
        # Compute derivatives of normalized probabilities using quotient rule
        dp0_hat_dθ = (dp0_dθ * S - p0_raw * dS_dθ) / (S * S)
        dp1_hat_dθ = (dp1_dθ * S - p1_raw * dS_dθ) / (S * S)
        
        if y == 0:  # Inside class: gradient = -1/(p0_hat + EPS) * d(p0_hat)/dθ
            gradient = -(dp0_hat_dθ / (p0_hat + EPS))
        else:  # Outside class: gradient = -1/(p1_hat + EPS) * d(p1_hat)/dθ
            gradient = -(dp1_hat_dθ / (p1_hat + EPS))

    elif mode == "mse":
        target = 1.0 if y == 0 else 0.0
        loss = (p0_hat - target)**2

        circuits = [circuit, circuit]
        observables = projectors  # [P0, P1]
        parameter_values = [values, values]
        grad_result = gradient_calc.run(circuits, observables, parameter_values).result()
        grad_p0 = grad_result.gradients[0]
        grad_p1 = grad_result.gradients[1]

        dp0_dθ = grad_p0[param_indices]
        dp1_dθ = grad_p1[param_indices]

        # If S is clamped, treat dS/dθ = 0
        if S_raw > EPS:
            dS_dθ = dp0_dθ + dp1_dθ
        else:
            dS_dθ = 0.0 * dp0_dθ  # same shape, all zeros

        # Quotient rule
        dp0_hat_dθ = (dp0_dθ * S - p0_raw * dS_dθ) / (S * S)

        gradient = 2.0 * (p0_hat - target) * dp0_hat_dθ
    
    else:
        raise ValueError(f"Unknown mode: {mode}. Supported modes are 'ce' and 'mse'.")

    return float(loss), gradient

# Global variables for worker processes
g_estimator = None
g_gradient_calc = None
g_circuit = None
g_projectors = None
g_param_indices = None
g_X_train = None
g_y_train = None
g_mode = None

def worker_init(circuit, projectors, param_indices, X_train, y_train, mode="ce"):
    """Initialize worker process with shared objects"""
    global g_estimator, g_gradient_calc, g_circuit, g_projectors, g_param_indices, g_X_train, g_y_train, g_mode
    
    g_estimator = StatevectorEstimator()
    g_gradient_calc = ReverseEstimatorGradient(g_estimator)
    g_circuit = circuit
    g_projectors = projectors
    g_param_indices = param_indices
    g_X_train = X_train
    g_y_train = y_train
    g_mode = mode

def compute_sample_loss_grad(theta: np.ndarray, sample_idx: int) -> Tuple[float, np.ndarray]:
    """Compute loss and gradient for a single sample (worker function)"""
    x = g_X_train[sample_idx]
    y = int(g_y_train[sample_idx])
    
    return compute_loss_and_gradient(
        g_estimator, g_gradient_calc, g_circuit, g_projectors,
        x, y, theta, g_param_indices, g_mode
    )

class TraditionalQNNModel:
    """Wrapper model class to make our QNN compatible with star_eval.evaluate_model"""
    
    def __init__(self, circuit, projectors, param_indices):
        self.circuit = circuit
        self.projectors = projectors
        self.param_indices = param_indices
        
    def forward(self, x, theta):
        """
        Forward pass that returns probabilities [p_in, p_out] compatible with star_eval.
        
        Args:
            x: Input features [x_coord, y_coord]
            theta: Parameter values
            
        Returns:
            numpy.ndarray: Probabilities [p_in, p_out] where p_in is prob of class 0, p_out is prob of class 1
        """
        estimator = StatevectorEstimator()
        
        # Get parameters
        params = list(self.circuit.parameters)
        param_pos = {p: i for i, p in enumerate(params)}
        values = np.zeros(len(params))
        
        # Set input parameters
        input_params = [p for p in params if 'x' in p.name or 'input' in p.name]
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
        p0 = float(result[0].data.evs)  # probability of class 0 (inside)
        p1 = float(result[1].data.evs)  # probability of class 1 (outside)
        
        # Normalize probabilities
        total = p0 + p1 + EPS
        p0_norm = p0 / total
        p1_norm = p1 / total
        
        # Return in format expected by star_eval: [p_in, p_out]
        return np.array([p0_norm, p1_norm])


def main():
    print(f"Traditional QNN Binary Classifier")
    print(f"CPUs: {N_CPUS}")
    print(f"Loss mode: {LOSS}")
    
    # ─── Data Loading ───
    # Dynamically import the correct data loader based on SHAPE
    _data_module = __import__(f"data.{SHAPE}_data", fromlist=[f"get_{SHAPE}_data"])
    get_data_func = getattr(_data_module, f"get_{SHAPE}_data")
    if SHAPE == "star":
        X_train, X_test, y_train, y_test, boundary_path = get_data_func(SAMPLES, condense=CONDENSE)
    else:
        X_train, X_test, y_train, y_test, boundary_path = get_data_func(SAMPLES)
    
    # Create traditional QNN circuit (ZZFeatureMap + EfficientSU2)
    print(f"Circuit: ZZFeatureMap(2) + EfficientSU2(2) -> 2 qubits")
    feature_map = ZZFeatureMap(2)
    ansatz = EfficientSU2(2)
    circuit = feature_map.compose(ansatz)
    
    # Transpile circuit
    circuit = transpile(circuit, optimization_level=1)
    params = list(circuit.parameters)
    
    # ─── Parameter Setup ───
    # Get trainable parameter indices (EfficientSU2 parameters only)
    param_indices = [i for i, param in enumerate(params) if 'θ' in param.name or 'ansatz' in param.name or param.name.startswith('θ')]
    
    # If no ansatz parameters found, use all non-input parameters
    if not param_indices:
        input_params = [p.name for p in params if 'x' in p.name or 'input' in p.name]
        param_indices = [i for i, param in enumerate(params) if param.name not in input_params]
    
    # Initialize weights
    np.random.seed(RSEED)
    theta = np.random.uniform(-np.pi, np.pi, len(param_indices))
    initial_theta_snapshot = theta.copy()
    
    # ─── Projector Setup ───
    projectors = create_binary_projectors()
    
    # ─── Model Wrapper Setup ───
    qnn_model = TraditionalQNNModel(circuit, projectors, param_indices)
    
    # Convert scalar labels to one-hot for binary classification (only for evaluation)
    # 0 (inside) -> [1, 0], 1 (outside) -> [0, 1]
    y_test_onehot = np.eye(2)[y_test.astype(int)]
    
    # ─── Initial Evaluation ───
    initial_acc = evaluate_model(qnn_model, theta, X_test, y_test_onehot, 'binary', "Initial (Random Weights)")
    
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
        worker_init(circuit, projectors, param_indices, X_train, y_train, LOSS)
        pool = None
    else:
        # Multi-process
        pool = Parallel(
            n_jobs=N_CPUS,
            initializer=worker_init,
            initargs=(circuit, projectors, param_indices, X_train, y_train, LOSS),
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
        
        # Batch accumulation variables
        grad_sum = np.zeros_like(theta)
        k = 0
        
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
            
            # Accumulate gradients and losses
            for loss, grad in results:
                grad_sum += grad
                k += 1
                epoch_grads.append(grad)
                epoch_loss_acc += loss
                
                # Trigger Adam update every BATCH samples
                if k == BATCH:
                    step += 1
                    grad_mean = grad_sum / k
                    m1 = beta1 * m1 + (1 - beta1) * grad_mean
                    v1 = beta2 * v1 + (1 - beta2) * (grad_mean**2)
                    m1_hat = m1 / (1 - beta1**step)
                    v1_hat = v1 / (1 - beta2**step)
                    theta -= lr * m1_hat / (np.sqrt(v1_hat) + 1e-8)
                    
                    # Reset accumulators
                    grad_sum = np.zeros_like(theta)
                    k = 0
        
        # Flush leftovers at epoch end if partial batch remains
        if k > 0:
            step += 1
            grad_mean = grad_sum / k
            m1 = beta1 * m1 + (1 - beta1) * grad_mean
            v1 = beta2 * v1 + (1 - beta2) * (grad_mean**2)
            m1_hat = m1 / (1 - beta1**step)
            v1_hat = v1 / (1 - beta2**step)
            theta -= lr * m1_hat / (np.sqrt(v1_hat) + 1e-8)

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
    
    final_acc = evaluate_model(qnn_model, theta, X_test, y_test_onehot, 'binary', "Final (Trained Weights)")
    
    # ─── Save Results ───
    os.makedirs(SHAPE, exist_ok=True)
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(losses)+1), losses, 'o-', linewidth=2)
    plt.xlabel('Epoch')
    loss_label = 'Binary Cross-Entropy Loss' if LOSS == 'ce' else 'MSE Loss'
    plt.ylabel(loss_label)
    plt.title(f'Training Loss - {LOSS.upper()} (Traditional QNN {model_name})')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    loss_plot_path = f'{SHAPE}/loss_{model_name}.png'
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
                qnn_model, initial_theta_snapshot, 2, X_train, np.eye(2)[y_train.astype(int)], 'binary',
                boundary=boundary_path,
                title='Initial Decision Boundary',
                resolution=args.boundary_resolution,
                save_path=f'{SHAPE}/initboundary_{model_name}.png',
                testing_accuracy=initial_acc,
                epochs=args.epochs,
                sample_size=len(X_test)
            )
            print(f"Initial boundary saved to {f'{SHAPE}/initboundary_{model_name}.png'}")

            # --- Final Boundary ---
            print("Visualizing final boundary...")
            visualize_decision_boundary(
                qnn_model, theta, 2, X_train, np.eye(2)[y_train.astype(int)], 'binary',
                boundary=boundary_path,
                title='Final Decision Boundary',
                resolution=args.boundary_resolution,
                save_path=f'{SHAPE}/finalboundary_{model_name}.png',
                testing_accuracy=final_acc,
                epochs=args.epochs,
                sample_size=len(X_test)
            )
            print(f"Final boundary saved to {f'{SHAPE}/finalboundary_{model_name}.png'}")
                        
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