"""star_train_parallel_2.py – Observable-based gradient with ReverseEstimatorGradient
================================================================================
This version uses observables and Qiskit's ReverseEstimatorGradient for more 
efficient gradient computation compared to manual parameter-shift gradients.
"""
from __future__ import annotations
import os, sys, time, argparse, multiprocessing as mp
from contextlib import suppress
from typing import List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm import tqdm
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
try:
    from qiskit.primitives import StatevectorEstimator
    USE_STATEVECTOR_ESTIMATOR = True
except ImportError:
    try:
        from qiskit_aer.primitives import Estimator as AerEstimator
        USE_STATEVECTOR_ESTIMATOR = False
    except ImportError:
        from qiskit.primitives import Estimator as BasicEstimator
        USE_STATEVECTOR_ESTIMATOR = False
from qiskit_algorithms.gradients import ReverseEstimatorGradient
from qiskit.quantum_info import SparsePauliOp, Statevector

# ───────── project imports ─────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT))
from star_data import get_star_data
from star_spqc import create_spqc_circuit, create_random_weights
from star_eval import evaluate_model

# ───────── CLI args ─────────
parser = argparse.ArgumentParser("SPQC trainer (Observable-based with ReverseEstimatorGradient)")
parser.add_argument("--cpus",  type=int, default=None)
parser.add_argument("--gpus",  type=int, default=None)
parser.add_argument("--no-gpu", action="store_true")
parser.add_argument("--epochs", type=int, default=1)
ARGS = parser.parse_args()

# resources
OS_CPUS     = mp.cpu_count()
SLURM_CPUS  = int(os.environ.get("SLURM_CPUS_PER_TASK", OS_CPUS))
N_CPUS      = ARGS.cpus or SLURM_CPUS
USE_GPU     = not ARGS.no_gpu
VISIBLE_GPU = len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(',')) if os.environ.get("CUDA_VISIBLE_DEVICES") else 0
N_GPUS      = ARGS.gpus if ARGS.gpus is not None else VISIBLE_GPU
for e in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(e,"1")

# gpu helper (per-process, no multiprocessing.Value)
def assign_visible_gpu(worker_gpu_id=None):
    if not USE_GPU or N_GPUS<1: return
    if worker_gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"]=str(worker_gpu_id)
        os.environ.setdefault("QISKIT_AER_CUDA","1")

# Build projection operators for observables
def make_projectors(t, m, k):               # k = n*r
    """Create projection operators for measuring address qubit probabilities"""
    N = t + m + k + 1
    p0 = SparsePauliOp("Z", coeffs=[0.5]) + SparsePauliOp("I", coeffs=[0.5])
    p1 = SparsePauliOp("I", coeffs=[0.5]) - SparsePauliOp("Z", coeffs=[0.5])

    memo = {}

    def proj(label):
        if label in memo:
            return memo[label]

        proj_list = [None] * N
        
        # Ancillas (if any) are projected to |0>
        for i in range(t):
            proj_list[i] = p0
            
        # Address qubits are projected based on label
        for i in range(m):
            q_idx = t + i
            if ((label >> i) & 1) == 0:
                proj_list[q_idx] = p0
            else:
                proj_list[q_idx] = p1
                
        # Data qubits are projected to |0>
        for i in range(k):
            q_idx = t + m + i
            proj_list[q_idx] = p0
            
        # Trash qubit is projected to |0>
        proj_list[t+m+k] = p0
        
        # Tensor product all projectors (Qiskit is little-endian)
        rev_proj_list = proj_list[::-1]
        
        full_projector = rev_proj_list[0]
        for i in range(1, N):
            full_projector = full_projector.tensor(rev_proj_list[i])
        
        result = full_projector.simplify()
        memo[label] = result
        return result
        
    return proj

# Helper function to create appropriate Estimator
def create_estimator():
    """Create the appropriate Estimator based on available imports and GPU settings"""
    # For V2, try to pass backend_options for GPU. If it fails, create without.
    if USE_STATEVECTOR_ESTIMATOR:
        try:
            backend_opts = {"device": "GPU"} if (USE_GPU and N_GPUS > 0) else {}
            # This is the modern way to request GPU execution for V2 primitives
            return StatevectorEstimator(backend_options=backend_opts)
        except TypeError:
            # Fallback for older versions that don't support backend_options
            return StatevectorEstimator()
    
    # For V1, use AerEstimator with an explicit AerSimulator backend
    else:
        device = "GPU" if (USE_GPU and N_GPUS > 0) else "CPU"
        backend = AerSimulator(method="statevector", device=device)
        try:
            # Try AerEstimator first (from qiskit_aer.primitives)
            return AerEstimator(backend=backend)
        except NameError:
            # Fall back to BasicEstimator if AerEstimator not available
            try:
                return BasicEstimator(backend=backend) 
            except TypeError:
                # If backend param not supported, use without it
                return BasicEstimator()

# per‑process/thread globals for estimator and gradients
g_est = None  
g_grd = None
g_proj_fn = None
g_template = None
g_params = None
g_weight_idxs = None
EPS = 1e-10

# shared objects for Windows threads
shared_est = None
shared_grd = None
shared_proj_fn = None
shared_template = None
shared_params = None
shared_weight_idxs = None

def worker_init(qc, t, m, n, r):
    """Initialize per-worker estimator, gradient calculator, and projectors"""
    print(f'init pid={os.getpid()} start')
    
    # Assign GPU to this worker (round-robin by process ID)
    worker_gpu_id = os.getpid() % N_GPUS if N_GPUS > 0 else None
    assign_visible_gpu(worker_gpu_id)
    
    # Create appropriate Estimator
    est = create_estimator()
    grd = ReverseEstimatorGradient(est)
    
    template = transpile(qc, optimization_level=0, layout_method="trivial")
    proj_fn = make_projectors(t, m, n*r)
    
    # Pre-calculate weight indices for slicing gradients
    params = list(template.parameters)
    model_params = [p for p in params if p.name.startswith('model')]
    address_params = [p for p in params if p.name.startswith('address_theta')]
    weight_idxs = [params.index(p) for p in (model_params + address_params)]

    globals().update(
        g_est=est, g_grd=grd,
        g_proj_fn=proj_fn,
        g_template=template, g_params=params,
        g_weight_idxs=weight_idxs
    )
    print(f'init pid={os.getpid()} done; params={len(g_params)} gpu={worker_gpu_id}')

def loss_and_grad(theta, label, x, t, m, n, r):
    """Compute loss and gradient using observable-based approach"""
    # Get objects from appropriate source (threads vs processes)
    if os.name == 'nt':  # Windows threads
        est = shared_est
        grd = shared_grd
        proj_fn = shared_proj_fn
        # Create a thread-local copy of the circuit to prevent race conditions
        template = shared_template.copy()
        params = shared_params
        weight_idxs = shared_weight_idxs
    else:  # Linux/HPC processes
        if g_params is None:
            raise RuntimeError(f"Worker pid={os.getpid()} started without initialisation")
        est = g_est
        grd = g_grd
        proj_fn = g_proj_fn
        template = g_template
        params = g_params
        weight_idxs = g_weight_idxs
    
    # Create parameter binding for input and weights
    param_binding = {}
    
    # Get parameter types (needed for binding)
    input_params = [p for p in params if p.name.startswith('input_theta')]
    model_params = [p for p in params if p.name.startswith('model')]
    address_params = [p for p in params if p.name.startswith('address_theta')]
    
    # Bind input parameters
    for j, param in enumerate(input_params):
        if j < len(x):
            param_binding[param] = x[j]
        else:
            param_binding[param] = 0.0
    
    # Split weights into model and address portions
    num_model_params = len(model_params)
    model_values = theta[:num_model_params]
    address_values = theta[num_model_params:]
    
    # Bind model parameters
    for j, param in enumerate(model_params):
        if j < len(model_values):
            param_binding[param] = model_values[j]
        else:
            param_binding[param] = 0.0
    
    # Bind address parameters
    for j, param in enumerate(address_params):
        if j < len(address_values):
            param_binding[param] = address_values[j]
        else:
            param_binding[param] = 0.0
    
    # Create values list aligned with template parameter order
    values = [param_binding[p] for p in params]
    
    # Get projector for this class label
    proj = proj_fn(int(label))
    
    # Get probability and gradient using mixed API approach
    # Check if we're using StatevectorEstimator (V2) or others (V1)
    is_v2_estimator = hasattr(est, '__class__') and 'StatevectorEstimator' in str(est.__class__)
    
    if is_v2_estimator:
        # V2 Estimator with V1 Gradient
        pub = (template, proj, values)
        est_result = est.run([pub]).result()
        p_val = est_result[0].data.evs
        
        # ReverseEstimatorGradient always uses V1 API
        grd_result = grd.run([template], [proj], [values]).result()
        d_p = grd_result.gradients[0]
    else:
        # V1 Estimator with V1 Gradient
        p_val = est.run([template], [proj], [values]).result().values[0]
        d_p = grd.run([template], [proj], [values]).result().gradients[0]
    
    # Compute negative log-likelihood loss and gradient
    loss = -np.log(p_val + EPS)
    grad = -d_p / (p_val + EPS)
    
    # Keep grads for model + address params only (slice to match theta shape)
    grad = grad[weight_idxs]
    
    return loss, grad

# wrapper for evaluate_model (simplified since we now work with probabilities)
def make_wrapper(qc, t, m, n, r):
    """Create a wrapper function for evaluation that mimics the old amplitude-based approach"""
    # Use the circuit directly with Statevector.from_instruction (for compatibility)
    tpl = qc.copy()
    
    # Get parameter mapping
    input_params = []
    model_params = []
    address_params = []
    
    for param in tpl.parameters:
        if param.name.startswith('input_theta'):
            input_params.append(param)
        elif param.name.startswith('model'):
            model_params.append(param)
        elif param.name.startswith('address_theta'):
            address_params.append(param)
    
    def f(x, th):
        # Bind both input values and weights
        param_binding = {}
        
        # Bind input parameters
        for i, param in enumerate(input_params):
            if i < len(x):
                param_binding[param] = x[i]
            else:
                param_binding[param] = 0.0
        
        # Split weights into model and address portions  
        num_model_params = len(model_params)
        model_values = th[:num_model_params]
        address_values = th[num_model_params:]
        
        # Bind model parameters
        for i, param in enumerate(model_params):
            if i < len(model_values):
                param_binding[param] = model_values[i]
            else:
                param_binding[param] = 0.0
        
        # Bind address parameters
        for i, param in enumerate(address_params):
            if i < len(address_values):
                param_binding[param] = address_values[i]
            else:
                param_binding[param] = 0.0
        
        bound_circuit = tpl.assign_parameters(param_binding)
        # Use Statevector.from_instruction for reliable statevector access
        sv = Statevector.from_instruction(bound_circuit).data
        return addr_amplitudes(sv, t, m, n, r)
    
    class W: 
        forward = staticmethod(f)
    return W()

# amplitude slice (kept for compatibility with evaluate_model)
def addr_amplitudes(sv: np.ndarray, t: int, m: int, n: int, r: int) -> np.ndarray:
    """Extract address amplitudes for compatibility with existing evaluation"""
    N = t + m + n * r + 1
    tensor = sv.reshape([2] * N)
    idx = [0] * t + [slice(None)] * m + [0] * (n * r) + [0]
    addr = tensor[tuple(idx)].reshape(2**m)
    # Renormalize
    norm = np.linalg.norm(addr)
    return addr / norm if norm != 0 else addr

# Note: compute_loss_sampled removed since training loop now directly 
# computes and tracks loss from observable-based approach

# ───────── main ─────────

def main():
    global USE_GPU
    print(f"CPU={N_CPUS}, GPU active={USE_GPU and N_GPUS>0}")

    # GPU sanity check
    if USE_GPU and N_GPUS > 0:
        try:
            # Test GPU backend without changing global environment
            test_backend = AerSimulator(method="statevector", device="GPU")
            test_qc = QuantumCircuit(1, 1)
            test_qc.x(0)
            test_qc.measure(0, 0)
            test_backend.run(transpile(test_qc, test_backend), shots=1).result()
            print("GPU backend verified")
        except Exception as e:
            print(f"GPU fallback: {e} → CPU only")
            USE_GPU = False

    Xtr, Xte, ytr, yte, _ = get_star_data(300)
    np.random.seed(42)

    t, m, n, r = 0, 3, 2, 1
    frame = create_spqc_circuit(t=t, m=m, n=n, r=r)
    qc = QuantumCircuit(frame.num_qubits)
    for inst in frame.data:
        if inst.operation.name != "measure": 
            qc.append(inst.operation, inst.qubits)
    theta = create_random_weights(frame, seed=42)
    print(f"Number of parameters = {len(theta)}")

    classes = 2**m
    Ytr = np.eye(classes)[ytr.astype(int)]
    Yte = np.eye(classes)[yte.astype(int)]

    evaluate_model(make_wrapper(qc, t, m, n, r), theta, Xte, Yte, 'binary', 'Initial')

    outer = min(N_CPUS, len(Xtr))
    print(f"Processes over samples: {outer}")
    if outer > 10:
        print("NOTE: If you get 'Worker not initialised' errors, try --cpus 4 to reduce worker overhead")
    if os.name == 'nt':  # Windows
        print("Windows detected: Using thread-based parallelism for stability")
    print()

    m1 = np.zeros_like(theta)
    v1 = np.zeros_like(theta)
    b1, b2, lr = 0.9, 0.999, 0.01
    losses: List[float] = []
    
    # Pre-import qiskit in main process to help worker startup
    import qiskit, qiskit_aer
    print("Pre-imported qiskit libraries")
    
    # Use threads on Windows, processes on Linux/HPC for best compatibility
    global shared_est, shared_grd, shared_proj_fn, shared_template, shared_params, shared_weight_idxs
    
    if os.name == 'nt':  # Windows
        print("Using thread-based parallelism for Windows compatibility...")
        backend_type = "threads"
        pool = Parallel(n_jobs=outer, prefer="threads")
        
        # Pre-build shared objects for thread sharing
        print("Pre-building shared estimator and gradient objects...")
        assign_visible_gpu(0)  # Set GPU for main thread
        # Create shared Estimator
        shared_est = create_estimator()
        shared_grd = ReverseEstimatorGradient(shared_est)
        shared_template = transpile(qc, optimization_level=0, layout_method="trivial")
        shared_proj_fn = make_projectors(t, m, n*r)
        shared_params = list(shared_template.parameters)
        
        # Pre-calculate and share weight indices
        model_params = [p for p in shared_params if p.name.startswith('model')]
        address_params = [p for p in shared_params if p.name.startswith('address_theta')]
        shared_weight_idxs = [shared_params.index(p) for p in (model_params + address_params)]

        print(f"Shared template ready with {len(shared_params)} parameters")
    else:  # Linux/HPC
        print("Using process-based parallelism for HPC performance...")
        backend_type = "processes"
        try:
            pool = Parallel(
                n_jobs=outer,
                initializer=worker_init,
                initargs=(qc, t, m, n, r),
                timeout=3600,
                prefer="processes",
                reuse=True
            )
        except TypeError:
            pool = Parallel(
                n_jobs=outer,
                initializer=worker_init,
                initargs=(qc, t, m, n, r),
                timeout=3600,
                prefer="processes",
            )
    print("Parallelism ready, starting training...")
    
    for ep in tqdm(range(ARGS.epochs), desc='Epoch'):
        # Compute loss and gradients for each sample
        results = pool(
            delayed(loss_and_grad)(theta, ytr[i], Xtr[i], t, m, n, r) 
            for i in range(len(Xtr))
        )
        
        # Extract losses and gradients
        sample_losses = [result[0] for result in results]
        grads = [result[1] for result in results]
        
        # Average gradient across samples
        g = np.mean(grads, axis=0)
        
        # Adam optimizer update
        m1 = b1 * m1 + (1 - b1) * g
        v1 = b2 * v1 + (1 - b2) * (g**2)
        m1h = m1 / (1 - b1**(ep + 1))
        v1h = v1 / (1 - b2**(ep + 1))
        theta -= lr * m1h / (np.sqrt(v1h) + 1e-8)

        # Track average loss
        avg_loss = np.mean(sample_losses)
        losses.append(avg_loss)
        print(f"Epoch {ep+1}: Average Loss = {avg_loss:.6f}")

    evaluate_model(make_wrapper(qc, t, m, n, r), theta, Xte, Yte, 'binary', 'Final')

    # Save loss plot
    os.makedirs('plots', exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(losses, 'o-')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Negative Log-Likelihood Loss (log scale)')
    plt.title('Training Loss Over Time (Observable-based)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/loss_observable.png', dpi=300)
    plt.close()

    print(f"Training complete. Loss plot saved to plots/loss_observable.png")

if __name__ == '__main__':
    with suppress(KeyboardInterrupt):
        main() 