"""star_train_parallel.py – Observable-based gradient with ReverseEstimatorGradient
================================================================================
This version uses observables and Qiskit's ReverseEstimatorGradient for more 
efficient gradient computation compared to manual parameter-shift gradients.
"""
from __future__ import annotations
import os, sys, time, argparse, warnings, multiprocessing as mp
from collections import defaultdict  # needed by lap class
from contextlib import suppress
from typing import List, Tuple

import numpy as np

warnings.filterwarnings('ignore', category=UserWarning, module='qiskit_algorithms')
warnings.filterwarnings('ignore', message='.*Casting complex values to real.*', module='qiskit_algorithms')

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm import tqdm
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.primitives import Estimator as BasicEstimator  # safe; may shadow later, fine
try:
    from qiskit.primitives import StatevectorEstimator
    USE_STATEVECTOR_ESTIMATOR = True
except ImportError:
    try:
        from qiskit_aer.primitives import Estimator as AerEstimator
        USE_STATEVECTOR_ESTIMATOR = False
    except ImportError:
        # BasicEstimator is already defined as a fallback
        USE_STATEVECTOR_ESTIMATOR = False
from qiskit_algorithms.gradients import ReverseEstimatorGradient
from qiskit.quantum_info import SparsePauliOp, Statevector

# ───────── Configuration ─────────
SHAPE = "star"  # Specify the shape: "star", "triangle", "square", etc.

# ───────── project imports ─────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT))
from star_spqc import create_spqc_circuit, create_random_weights
from star_eval import evaluate_model, visualize_decision_boundary

# ───────── Benchmarking Harness ─────────
bench = defaultdict(list)  # label → [times]

class lap:
    bench = defaultdict(list)
    def __init__(self, label, sync_cuda=False):
        self.label = label
        self.sync_cuda = sync_cuda

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *args):
        # Store lap time in a class-level dictionary
        lap.bench[self.label].append(time.perf_counter() - self.t0)
  
    @classmethod
    def reset(cls):
        cls.bench.clear()

    @classmethod
    def summary_str(cls):
        if not cls.bench:
            return "Fwd – | Grad – | Shots –"
        
        items = []
        
        # Sum times from all batches in the epoch
        fwd_t = np.sum(cls.bench.get("estimator.run", [])) + np.sum(cls.bench.get("estimator.adjoint", []))
        grd_t = np.sum(cls.bench.get("gradient.run", []))
        shots = len(cls.bench.get("estimator.run", [])) + len(cls.bench.get("estimator.adjoint", []))

        items.append(f"Fwd {fwd_t*1000:.1f}ms" if fwd_t > 0 else "Fwd –")
        items.append(f"Grad {grd_t*1000:.1f}ms" if grd_t > 0 else "Grad –")
        items.append(f"Shots {shots}")

        return " | ".join(items)

def choose_parallel_strategy(num_qubits, samples, cpus):
    """
    Choose a parallel strategy based on circuit size and sample count.
    
    - For small circuits, a single-process, fully-batched approach is fastest.
    - For medium to large circuits, multiprocessing is needed, but batch sizes
      are adjusted to balance IPC overhead and computation time.
    """
    # Rough crossover: statevector time grows ~O(2**(1.5*q)), Python overhead ~constant
    if num_qubits <= 12:          # small circuits
        # One big Estimator call is most efficient
        return {"style": "single", "batch_size": samples}
    if num_qubits <= 18:          # medium circuits
        # Balance IPC overhead with per-worker computation
        return {"style": "mp", "batch_size": min(32, samples), "procs": min(cpus, 8)}
    else:                         # getting heavy
        # Larger circuits need more compute time per sample
        return {"style": "mp", "batch_size": min(8, samples), "procs": cpus}

def _is_v2_estimator(est):
    """Check if the estimator is a V2 primitive."""
    return est.__class__.__name__.lower().endswith("statevectorestimator")

def _estimator_values(est, run_result):
    """
    Normalize estimator results across v1/v2.
    Returns 1D np.array of expectation values.
    """
    if _is_v2_estimator(est):
        # run_result is a PrimitiveResult; data is list of pub results
        return np.array([r.data.evs for r in run_result])
    else:
        return np.array(run_result.values)

def _gradient_arrays(grd_result):
    """
    Normalize gradient results to a list of np.arrays.
    Works for both v1 and v2 ReverseEstimatorGradient.
    """
    return list(grd_result.gradients)

# ───────── CLI args ─────────
parser = argparse.ArgumentParser("SPQC trainer (Observable-based with ReverseEstimatorGradient)")
parser.add_argument("--cpus",  type=int, default=None)
parser.add_argument("--gpus",  type=int, default=None)
parser.add_argument("--no-gpu", action="store_true")
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--lr", type=float, default=0.01, help="Learning rate for Adam optimizer")
parser.add_argument("--profile", action="store_true", help="Run in profiling mode (short epochs, extra logging)")
parser.add_argument("--small-fast", action="store_true",
    help="Force old small-qubit training path (per-sample Estimator calls, light transpile).")
parser.add_argument("--force-batched", action="store_true",
    help="Force batched training approach even for small circuits.")
parser.add_argument("--visualize-boundary", action="store_true",
    help="Generate and save decision boundary plots after training.")
parser.add_argument("--boundary-resolution", type=int, default=64,
    help="Grid resolution for boundary visualization (default: 64)")
ARGS = parser.parse_args()
BATCH = 1  # Batch size for processing samples

# resources
OS_CPUS     = mp.cpu_count()
SLURM_CPUS  = int(os.environ.get("SLURM_CPUS_PER_TASK", OS_CPUS))
N_CPUS      = ARGS.cpus or SLURM_CPUS
USE_GPU     = not ARGS.no_gpu
VISIBLE_GPU = len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(',')) if os.environ.get("CUDA_VISIBLE_DEVICES") else 0
N_GPUS      = ARGS.gpus if ARGS.gpus is not None else VISIBLE_GPU
RSEED       = 42
for e in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(e,"1")

# gpu helper (per-process, no multiprocessing.Value)
def assign_visible_gpu(worker_gpu_id=None):
    if not USE_GPU or N_GPUS<1: return
    if worker_gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"]=str(worker_gpu_id)
        os.environ.setdefault("QISKIT_AER_CUDA","1")

# Build projection operators for observables
def make_binary_projectors(t, m, k):               # k = n*r
    """Create binary projection operators for outside (0-3) vs inside (4-7) classification"""
    N = t + m + k + 1
    p0 = SparsePauliOp("Z", coeffs=[0.5]) + SparsePauliOp("I", coeffs=[0.5])
    p1 = SparsePauliOp("I", coeffs=[0.5]) - SparsePauliOp("Z", coeffs=[0.5])

    def create_address_projector(address_states):
        """Create a projector that sums over multiple address states"""
        projectors = []
        
        for addr_state in address_states:
            proj_list = [None] * N
            
            # Term qubits (if any) are projected to |0>
            for i in range(t):
                proj_list[i] = p0
                
            # Address qubits are projected based on addr_state
            for i in range(m):
                q_idx = t + i
                if ((addr_state >> i) & 1) == 0:
                    proj_list[q_idx] = p0
                else:
                    proj_list[q_idx] = p1
                    
            # Data qubits are projected to |0>
            for i in range(k):
                q_idx = t + m + i
                proj_list[q_idx] = p0
                
            # Ancilla qubit is projected to |0>
            proj_list[t+m+k] = p0
            
            # Tensor product all projectors (Qiskit is little-endian)
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

    # For m=3, we have 8 address states (0-7)
    # Outside (class 1): sum over address states 0-3  
    # Inside (class 0): sum over address states 4-7
    num_addr_states = 2**m
    half = num_addr_states // 2
    
    outside_states = list(range(0, half))      # [0, 1, 2, 3]
    inside_states = list(range(half, num_addr_states))   # [4, 5, 6, 7]
    
    outside_proj = create_address_projector(outside_states)
    inside_proj = create_address_projector(inside_states)
    
    return [outside_proj, inside_proj]  # Return list where index matches binary label

# Helper function to create appropriate Estimator
def create_estimator():
    """Create the appropriate Estimator based on available imports and GPU settings"""
    # For V2, try to pass backend_options for GPU. If it fails, create without.
    if USE_STATEVECTOR_ESTIMATOR:
        try:
            if USE_GPU and N_GPUS > 0:
                backend_opts = {"device": "GPU"}
            else:
                # Apply CPU-specific optimization to avoid Python gate overhead
                backend_opts = {"disable_pygate_callback": True}
            
            return StatevectorEstimator(backend_options=backend_opts)
        except TypeError:
            # Fallback for older versions that don't support backend_options
            return StatevectorEstimator()
    
    # For V1, use AerEstimator with an explicit AerSimulator backend
    else:
        device = "GPU" if (USE_GPU and N_GPUS > 0) else "CPU"
        
        sim_kwargs = {'method': 'statevector', 'device': device}
        if device == "CPU":
            # Apply CPU-specific optimization to avoid Python gate overhead
            sim_kwargs['disable_pygate_callback'] = True

        backend = AerSimulator(**sim_kwargs)
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
g_binary_proj_cache = None
g_template = None
g_params = None
g_weight_idxs = None
g_Xtr = None
g_ytr = None
EPS = 1e-10

# shared objects for Windows threads
shared_est = None
shared_grd = None
shared_binary_proj_cache = None
shared_template = None
shared_params = None
shared_weight_idxs = None
shared_Xtr = None
shared_ytr = None
shared_input_matrix = None

def worker_init(qc_template, t, m, n, r, Xtr_data, ytr_data, input_matrix, weight_idxs, binary_proj_cache):
    """Initialize per-worker estimator, gradient calculator, and binary projectors"""
    worker_gpu_id = os.getpid() % N_GPUS if N_GPUS > 0 else None
    assign_visible_gpu(worker_gpu_id)
    
    try:
        est = create_estimator()
    except Exception as e:
        print(f'GPU init failed in pid {os.getpid()} → CPU fallback: {e}')
        os.environ.pop("QISKIT_AER_CUDA", None)
        est = create_estimator()
    
    grd = ReverseEstimatorGradient(est)
    
    globals().update(
        g_est=est, g_grd=grd,
        g_template=qc_template,
        g_params=list(qc_template.parameters),
        g_weight_idxs=weight_idxs,
        g_Xtr=Xtr_data, g_ytr=ytr_data,
        g_binary_proj_cache=binary_proj_cache,
        g_input_matrix=input_matrix,
    )

def loss_and_grad_batch(theta, idx_group, ep):
    """Return (mean‑loss, mean‑grad) for a group of samples using binary cross-entropy."""
    if os.name == 'nt':
        est = shared_est
        grd = shared_grd
        input_matrix = shared_input_matrix
        binary_proj_cache = shared_binary_proj_cache
        template = shared_template.copy()
        weight_idxs = shared_weight_idxs
        ytr_data = shared_ytr
    else:
        est = g_est
        grd = g_grd
        input_matrix = g_input_matrix
        binary_proj_cache = g_binary_proj_cache
        template = g_template
        weight_idxs = g_weight_idxs
        ytr_data = g_ytr

    # For each sample, we need to compute both p_out and p_in
    vals_mat = input_matrix[idx_group].copy()
    vals_mat[:, weight_idxs] = theta
    vals_list = vals_mat.tolist()
    
    # Create estimator inputs for both projectors (outside and inside) for each sample
    est_circuits = []
    est_observables = []
    est_parameters = []
    
    for i, sample_idx in enumerate(idx_group):
        # Add both outside and inside projections for this sample
        est_circuits.extend([template, template])
        est_observables.extend([binary_proj_cache[0], binary_proj_cache[1]])  # [outside_proj, inside_proj]
        est_parameters.extend([vals_list[i], vals_list[i]])

    if _is_v2_estimator(est):
        triples = [(circ, obs, vals) for circ, obs, vals in zip(est_circuits, est_observables, est_parameters)]
        with lap("estimator.run"):
            est_res = est.run(triples).result()
        with lap("gradient.run"):
            grd_res = grd.run(est_circuits, est_observables, est_parameters).result()
    else: # V1 API
        with lap("estimator.adjoint"):
            est_res = est.run(est_circuits, est_observables, est_parameters).result()
        with lap("gradient.run"):
            grd_res = grd.run(est_circuits, est_observables, est_parameters).result()
    
    all_vals = _estimator_values(est, est_res)
    all_grads = _gradient_arrays(grd_res)
    
    # Reshape results: we have 2 values per sample (p_out, p_in)
    p_outs = all_vals[::2]  # Every other value starting from 0
    p_ins = all_vals[1::2]  # Every other value starting from 1
    grad_outs = all_grads[::2]  # Gradients for p_out
    grad_ins = all_grads[1::2]   # Gradients for p_in
    
    # Ensure probabilities are valid and normalized
    p_outs = np.clip(p_outs, EPS, 1.0 - EPS)
    p_ins = np.clip(p_ins, EPS, 1.0 - EPS)
    
    # Normalize so p_out + p_in = 1 (they should sum to 1 by construction, but numerical errors)
    total_probs = p_outs + p_ins
    p_outs = p_outs / total_probs
    p_ins = p_ins / total_probs
    
    # Compute binary cross-entropy loss and gradients
    losses = []
    grads = []
    
    for i, sample_idx in enumerate(idx_group):
        y = int(ytr_data[sample_idx])  # Binary label (0 or 1)
        p_out, p_in = p_outs[i], p_ins[i]
        grad_out, grad_in = grad_outs[i], grad_ins[i]
        
        # Binary cross-entropy: -[y*log(p_in) + (1-y)*log(p_out)]
        # The dataset encodes "inside" as 0 and "outside" as 1.
        if y == 0:  # Inside class
            loss = -np.log(p_in + EPS)
            # Gradient of loss w.r.t. weights: -grad_in/p_in
            grad = -grad_in[weight_idxs] / (p_in + EPS)
        else:  # Outside class (y == 1)
            loss = -np.log(p_out + EPS)
            # Gradient of loss w.r.t. weights: -grad_out/p_out
            grad = -grad_out[weight_idxs] / (p_out + EPS)
            
        losses.append(loss)
        grads.append(grad)
    
    return np.mean(losses), np.mean(grads, axis=0)

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
        if param.name.startswith('zinput_theta'):
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

def small_make_values(theta, x, params):
    # identical to old make_values()
    param_binding = {}
    input_params = [p for p in params if p.name.startswith('zinput_theta')]
    model_params = [p for p in params if p.name.startswith('model')]
    address_params = [p for p in params if p.name.startswith('address_theta')]
    for j,p in enumerate(input_params):  param_binding[p] = x[j] if j < len(x) else 0.0
    num_model = len(model_params)
    for j,p in enumerate(model_params):  param_binding[p] = theta[j] if j < num_model else 0.0
    for j,p in enumerate(address_params): param_binding[p] = theta[num_model+j] if num_model+j < len(theta) else 0.0
    return [param_binding[p] for p in params]

def small_worker_init(qc, t, m, n, r, Xtr_data, ytr_data):
    """Initialize per-worker for the 'small-fast' path. 
    Args:
        qc: Transpiled quantum circuit (template)
    """
    worker_gpu_id = os.getpid() % N_GPUS if N_GPUS > 0 else None
    assign_visible_gpu(worker_gpu_id)
    
    try:
        est = create_estimator()
    except Exception as e:
        print(f'GPU init failed in pid {os.getpid()} -> CPU fallback: {e}')
        os.environ.pop("QISKIT_AER_CUDA", None)
        est = create_estimator()
    
    grd = ReverseEstimatorGradient(est)
    
    # Create binary projectors instead of individual state projectors
    binary_proj_cache = make_binary_projectors(t, m, n*r)
    
    params = list(qc.parameters)
    model_idx    = [i for i,p in enumerate(params) if p.name.startswith("model")]
    address_idx  = [i for i,p in enumerate(params) if p.name.startswith("address_theta")]
    weight_idxs  = model_idx + address_idx

    globals().update(
        g_est=est, g_grd=grd,
        g_binary_proj_cache=binary_proj_cache,
        g_template=qc,
        g_params=params,
        g_weight_idxs=weight_idxs,
        g_Xtr=Xtr_data, g_ytr=ytr_data,
    )


def small_loss_and_grad_batch(theta, idx_group, ep):
    # Small-fast path uses g_* globals (set by worker_init or main process)
    est, grd = g_est, g_grd
    binary_proj_cache, template_base, params = g_binary_proj_cache, g_template, g_params
    weight_idxs, Xtr_data, ytr_data = g_weight_idxs, g_Xtr, g_ytr
    
    # Safety check for uninitialized globals
    if est is None or template_base is None or Xtr_data is None:
        raise RuntimeError("Worker globals not initialized. This shouldn't happen with the current setup.")
    
    is_v2 = hasattr(est, '__class__') and 'StatevectorEstimator' in str(est.__class__)
    template = template_base
    
    # For each sample, we need to compute both p_out and p_in
    est_circuits = []
    est_observables = []
    est_parameters = []
    
    for i in idx_group:
        x, label = Xtr_data[i], int(ytr_data[i])
        values = small_make_values(theta, x, params)
        
        # Add both outside and inside projections for this sample
        est_circuits.extend([template, template])
        est_observables.extend([binary_proj_cache[0], binary_proj_cache[1]])  # [outside_proj, inside_proj]
        est_parameters.extend([values, values])

    if is_v2:
        triples = [(circ, obs, vals) for circ, obs, vals in zip(est_circuits, est_observables, est_parameters)]
        with lap("estimator.run"):
            p_vals_result = est.run(triples).result()
            all_vals = [r.data.evs for r in p_vals_result]
        with lap("gradient.run"):
            grd_result = grd.run(est_circuits, est_observables, est_parameters).result()
            all_grads = grd_result.gradients
    else:
        with lap("estimator.adjoint"):
            p_vals_result = est.run(est_circuits, est_observables, est_parameters).result()
            all_vals = p_vals_result.values
        with lap("gradient.run"):
            grd_result = grd.run(est_circuits, est_observables, est_parameters).result()
            all_grads = grd_result.gradients

    # Ensure numeric arrays
    all_vals = np.asarray(all_vals, dtype=float).ravel()
    all_grads = [np.asarray(g, dtype=float) for g in all_grads]

    # Reshape results: we have 2 values per sample (p_out, p_in)
    p_outs = all_vals[::2]  # Every other value starting from 0
    p_ins = all_vals[1::2]  # Every other value starting from 1
    grad_outs = all_grads[::2]  # Gradients for p_out
    grad_ins = all_grads[1::2]   # Gradients for p_in
    
    # Ensure probabilities are valid and normalized
    p_outs = np.clip(p_outs, EPS, 1.0 - EPS)
    p_ins = np.clip(p_ins, EPS, 1.0 - EPS)
    
    # Normalize so p_out + p_in = 1
    total_probs = p_outs + p_ins
    p_outs = p_outs / total_probs
    p_ins = p_ins / total_probs

    # Compute binary cross-entropy loss and gradients
    losses = []
    grads = []
    
    for i, sample_idx in enumerate(idx_group):
        y = int(ytr_data[sample_idx])  # Binary label (0 or 1)
        p_out, p_in = p_outs[i], p_ins[i]
        grad_out, grad_in = grad_outs[i], grad_ins[i]
        
        # Binary cross-entropy: -[y*log(p_in) + (1-y)*log(p_out)]
        # The dataset encodes "inside" as 0 and "outside" as 1.
        if y == 0:  # Inside class
            loss = -np.log(p_in + EPS)
            # Gradient of loss w.r.t. weights: -grad_in/p_in
            grad = -grad_in[weight_idxs] / (p_in + EPS)
        else:  # Outside class (y == 1)
            loss = -np.log(p_out + EPS)
            # Gradient of loss w.r.t. weights: -grad_out/p_out
            grad = -grad_out[weight_idxs] / (p_out + EPS)
            
        losses.append(loss)
        grads.append(grad)
        
    return np.mean(losses), np.mean(grads, axis=0)

# Note: compute_loss_sampled removed since training loop now directly 
# computes and tracks loss from observable-based approach

# ───────── main ─────────

def main():
    global USE_GPU, BATCH
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
            msg = str(e)
            print(f"GPU fallback: {msg} → CPU only")
            if "not supported on this system" in msg:
                print("\nHint: This error usually means you need to install the GPU-enabled version of Qiskit Aer.")
                print("Try running: pip uninstall qiskit-aer -y && pip install qiskit-aer-gpu\n")
            USE_GPU = False

    # Dynamic import: from data.{SHAPE}_data import get_{SHAPE}_data
    data_module = __import__(f"data.{SHAPE}_data", fromlist=[f"get_{SHAPE}_data"])
    get_data_func = getattr(data_module, f"get_{SHAPE}_data")
    Xtr, Xte, ytr, yte, boundary_path = get_data_func(300)
    np.random.seed(RSEED)

    t, m, n, r = 0, 3, 2, 1
    model_name = f"t{t}_m{m}_n{n}_r{r}"
    num_qubits = t + m + n * r + 1
    print(f"Model configuration: {model_name} ({num_qubits} qubits)")
    FORCE_SMALL = ARGS.small_fast or (num_qubits <= 12 and not ARGS.force_batched)  # auto for <=12 unless user overrides

    frame = create_spqc_circuit(t=t, m=m, n=n, r=r)
    qc = QuantumCircuit(frame.num_qubits, name="SPQC")
    for inst in frame.data:
        if inst.operation.name != "measure": 
            qc.append(inst.operation, inst.qubits)
    
    # Transpile once at a high optimization level
    # print("Transpiling circuit...")
    # template = transpile(qc, optimization_level=3)
    # print("Transpilation complete.")

    theta = create_random_weights(frame, seed=RSEED)
    initial_theta_snapshot = theta.copy()
    
    # Temporary placeholder; real parameter mapping will happen after transpilation
    classes = 2**m
    Ytr_onehot = np.eye(classes)[ytr.astype(int)]
    Yte_onehot = np.eye(classes)[yte.astype(int)]

    initial_acc = evaluate_model(make_wrapper(qc, t, m, n, r), theta, Xte, Yte_onehot, 'binary', 'Initial (pre-transpile)')

    # ─── Choose Strategy ───
    if FORCE_SMALL:
        # mimic old behavior
        strategy = {"style": "mp", "batch_size": 1, "procs": min(N_CPUS, len(Xtr))}
        # use a *light* transpile like the old file did (faster bind/build cycles)
        print("Transpiling circuit (light)...")
        template = transpile(qc, optimization_level=0, layout_method="trivial")
        print("Transpilation complete.")
    else:
        strategy = choose_parallel_strategy(num_qubits, len(Xtr), N_CPUS)
        # keep heavy optimize for larger problems
        print("Transpiling circuit (heavy)...")
        template = transpile(qc, optimization_level=3)
        print("Transpilation complete.")

    BATCH = strategy.get("batch_size", 1)
    outer = strategy.get("procs", N_CPUS)
    print(f"Strategy: {strategy['style']} | Procs: {outer} | Batch Size: {BATCH}")

    # ─── Parameter Index Mapping (after transpilation) ───
    params = list(template.parameters)
    input_idx    = [i for i,p in enumerate(params) if p.name.startswith("zinput_theta")]
    model_idx    = [i for i,p in enumerate(params) if p.name.startswith("model")]
    address_idx  = [i for i,p in enumerate(params) if p.name.startswith("address_theta")]
    weight_idxs  = model_idx + address_idx
    num_inputs   = len(input_idx)
    num_weights  = len(weight_idxs)
    assert theta.shape[0] == num_weights, f"Theta length {theta.shape[0]} does not match number of weights {num_weights} after transpile"

    print(f"Number of parameters = {num_weights} ({num_qubits} qubits)")

    # ─── Precompute Input Matrix & Projectors (only for batched strategies) ───
    if not FORCE_SMALL:
        full_input_matrix = np.zeros((len(Xtr), len(params)), dtype=float)
        for row, x in enumerate(Xtr):
            for col_j, p_idx in enumerate(input_idx):
                if col_j < len(x):
                    full_input_matrix[row, p_idx] = x[col_j]

        # Create binary projectors for outside (0-3) vs inside (4-7) classification
        binary_proj_cache = make_binary_projectors(t, m, n*r)

    m1, v1 = np.zeros_like(theta), np.zeros_like(theta)
    b1, b2, lr = 0.9, 0.999, ARGS.lr
    losses: List[float] = []
    rng = np.random.default_rng(RSEED)
    step = 0  # Adam bias-correction step counter local to main

    # ─── SMALL-FAST STRATEGY (per-sample parallelism) ───
    if FORCE_SMALL:
        print("Using small-fast strategy (per-sample parallelism).")
        
        # For single-process case, manually initialize globals in main process
        if strategy["procs"] == 1:
            print("Single process detected, initializing in main process...")
            small_worker_init(template, t, m, n, r, Xtr, ytr)
        
        pool = Parallel(
            n_jobs=strategy["procs"],
            initializer=small_worker_init,
            initargs=(template, t, m, n, r, Xtr, ytr),
            timeout=3600,
            backend="multiprocessing",
            prefer="processes"
        )
        print("Parallelism ready, starting training...")

        for ep in tqdm(range(ARGS.epochs), desc='Epoch'):
            # BATCH=1, each job is one sample
            groups = [[i] for i in rng.permutation(len(Xtr))]
            lap.reset()

            results = pool(delayed(small_loss_and_grad_batch)(theta, grp, ep) for grp in groups)
            
            losses_epoch = [r[0] for r in results if r is not None]
            grads_epoch = [r[1] for r in results if r is not None]
            avg_loss = np.mean(losses_epoch)
            losses.append(avg_loss)
            
            # Adam update on mean gradient for the epoch
            g = np.mean(grads_epoch, axis=0)
            grad_norm = np.linalg.norm(g)
            step = ep + 1  # Simple and deterministic bias correction
            m1 = b1 * m1 + (1 - b1) * g
            v1 = b2 * v1 + (1 - b2) * (g**2)
            m1h = m1 / (1 - b1**step)
            v1h = v1 / (1 - b2**step)
            theta -= lr * m1h / (np.sqrt(v1h) + 1e-8)

            if ARGS.profile or (ep % max(1, ARGS.epochs // 20) == 0):
                summary = lap.summary_str()
                tqdm.write(f"Epoch {ep+1: >4}: Loss {avg_loss:.4f} | Grad norm {grad_norm:.4f} | {summary}")

    # ─── BATCHED STRATEGIES (single or multi-process) ───
    else:
        # ─── SINGLE-PROCESS STRATEGY ───
        if strategy['style'] == 'single':
            print("Using single-process, fully-batched strategy.")
            est = create_estimator()
            grd = ReverseEstimatorGradient(est)

            for ep in tqdm(range(ARGS.epochs), desc='Epoch'):
                lap.reset()
                perm = rng.permutation(len(Xtr))
                epoch_loss_acc = []
                
                # Simplified loop for batches, handles full batch case
                for i in range(0, len(Xtr), BATCH):
                    batch_idx = perm[i:i+BATCH]
                    if len(batch_idx) == 0: continue

                    # For each sample, we need to compute both p_out and p_in
                    vals_mat = full_input_matrix[batch_idx].copy()
                    vals_mat[:, weight_idxs] = theta
                    vals_list = vals_mat.tolist()
                    
                    # Create estimator inputs for both projectors (outside and inside) for each sample
                    est_circuits = []
                    est_observables = []
                    est_parameters = []
                    
                    for j, sample_idx in enumerate(batch_idx):
                        # Add both outside and inside projections for this sample
                        est_circuits.extend([template, template])
                        est_observables.extend([binary_proj_cache[0], binary_proj_cache[1]])  # [outside_proj, inside_proj]
                        est_parameters.extend([vals_list[j], vals_list[j]])

                    if _is_v2_estimator(est):
                        triples = [(circ, obs, vals) for circ, obs, vals in zip(est_circuits, est_observables, est_parameters)]
                        with lap("estimator.run"):
                            est_res = est.run(triples).result()
                        with lap("gradient.run"):
                            grd_res = grd.run(est_circuits, est_observables, est_parameters).result()
                    else: # V1 API
                        with lap("estimator.adjoint"):
                            est_res = est.run(est_circuits, est_observables, est_parameters).result()
                        with lap("gradient.run"):
                            grd_res = grd.run(est_circuits, est_observables, est_parameters).result()

                    all_vals = _estimator_values(est, est_res)
                    all_grads = _gradient_arrays(grd_res)
                    
                    # Reshape results: we have 2 values per sample (p_out, p_in)
                    p_outs = all_vals[::2]  # Every other value starting from 0
                    p_ins = all_vals[1::2]  # Every other value starting from 1
                    grad_outs = all_grads[::2]  # Gradients for p_out
                    grad_ins = all_grads[1::2]   # Gradients for p_in
                    
                    # Ensure probabilities are valid and normalized
                    p_outs = np.clip(p_outs, EPS, 1.0 - EPS)
                    p_ins = np.clip(p_ins, EPS, 1.0 - EPS)
                    
                    # Normalize so p_out + p_in = 1
                    total_probs = p_outs + p_ins
                    p_outs = p_outs / total_probs
                    p_ins = p_ins / total_probs
                    
                    # Compute binary cross-entropy loss and gradients
                    batch_losses = []
                    batch_grads = []
                    
                    for j, sample_idx in enumerate(batch_idx):
                        y = int(ytr[sample_idx])  # Binary label (0 or 1)
                        p_out, p_in = p_outs[j], p_ins[j]
                        grad_out, grad_in = grad_outs[j], grad_ins[j]
                        
                        # Binary cross-entropy: -[y*log(p_in) + (1-y)*log(p_out)]
                        # The dataset encodes "inside" as 0 and "outside" as 1.
                        if y == 0:  # Inside class
                            loss = -np.log(p_in + EPS)
                            # Gradient of loss w.r.t. weights: -grad_in/p_in
                            grad = -grad_in[weight_idxs] / (p_in + EPS)
                        else:  # Outside class (y == 1)
                            loss = -np.log(p_out + EPS)
                            # Gradient of loss w.r.t. weights: -grad_out/p_out
                            grad = -grad_out[weight_idxs] / (p_out + EPS)
                            
                        batch_losses.append(loss)
                        batch_grads.append(grad)
                    
                    epoch_loss_acc.extend(batch_losses)
                    g = np.mean(batch_grads, axis=0)

                    step += 1
                    # Adam update
                    m1 = b1 * m1 + (1 - b1) * g
                    v1 = b2 * v1 + (1 - b2) * (g**2)
                    m1h = m1 / (1 - b1**step)
                    v1h = v1 / (1 - b2**step)
                    theta -= lr * m1h / (np.sqrt(v1h) + 1e-8)

                avg_loss = np.mean(epoch_loss_acc)
                losses.append(avg_loss)
                
                # Calculate epoch-level gradient norm for reporting
                if len(epoch_loss_acc) > 0:
                    epoch_grad_norm = np.linalg.norm(g)
                else:
                    epoch_grad_norm = 0.0
                
                if ARGS.profile or (ep % max(1, ARGS.epochs // 20) == 0):
                    summary = lap.summary_str()
                    tqdm.write(f"Epoch {ep+1: >4}: Loss {avg_loss:.4f} | Grad norm {epoch_grad_norm:.4f} | {summary}")

        # ─── MULTI-PROCESS STRATEGY ───
        else:
            if outer > 10:
                print("NOTE: If you get 'Worker not initialised' errors, try --cpus 4 to reduce worker overhead")

            # Pre-import qiskit in main process to help worker startup
            import qiskit, qiskit_aer
            print("Pre-imported qiskit libraries")

            if os.name == 'nt':  # Windows
                print("Windows detected: Using thread-based parallelism for stability")
                global shared_est, shared_grd, shared_template, shared_params, shared_weight_idxs, shared_Xtr, shared_ytr, shared_input_matrix, shared_binary_proj_cache
                
                print("Using thread-based parallelism for Windows compatibility...")
                pool = Parallel(n_jobs=outer, prefer="threads")
                
                print("Pre-building shared estimator and gradient objects...")
                assign_visible_gpu(0)
                shared_est = create_estimator()
                shared_grd = ReverseEstimatorGradient(shared_est)
                shared_template = template # Use pre-transpiled template
                shared_params = list(shared_template.parameters)
                
                model_params = [p for p in shared_params if p.name.startswith('model')]
                address_params = [p for p in shared_params if p.name.startswith('address_theta')]
                shared_weight_idxs = [shared_params.index(p) for p in (model_params + address_params)]
                
                shared_Xtr = Xtr
                shared_ytr = ytr
                shared_input_matrix = full_input_matrix
                shared_binary_proj_cache = binary_proj_cache

                print(f"Shared template ready with {len(shared_params)} parameters")
            
            else:  # Linux/HPC
                print("Using process-based parallelism for HPC performance...")
                pool = Parallel(
                    n_jobs=outer,
                    initializer=worker_init,
                    initargs=(template, t, m, n, r, Xtr, ytr, full_input_matrix, weight_idxs, binary_proj_cache),
                    timeout=3600,
                    backend="multiprocessing",
                    prefer="processes"
                )
            print("Parallelism ready, starting training...")

            for ep in tqdm(range(ARGS.epochs), desc='Epoch'):
                perm = rng.permutation(len(Xtr))
                groups = [perm[i:i+BATCH] for i in range(0, len(Xtr), BATCH)]
                lap.reset()
                results = pool(delayed(loss_and_grad_batch)(theta, grp, ep)
                               for grp in groups)
                
                total_loss = np.mean([r[0] for r in results if r is not None])
                sample_losses = [result[0] for result in results]
                grads = [result[1] for result in results]
                
                # Apply gradient updates and track gradient norms
                grad_norms = []
                for i in range(len(grads)):
                    step += 1
                    g = grads[i]
                    grad_norms.append(np.linalg.norm(g))
                    m1 = b1 * m1 + (1 - b1) * g
                    v1 = b2 * v1 + (1 - b2) * (g**2)
                    m1h = m1 / (1 - b1**step)
                    v1h = v1 / (1 - b2**step)
                    theta -= lr * m1h / (np.sqrt(v1h) + 1e-8)
                
                sample_losses = [result[0] for result in results if result is not None]
                avg_loss = np.mean(sample_losses)
                avg_grad_norm = np.mean(grad_norms) if grad_norms else 0.0
                losses.append(avg_loss)

                if ep % max(1, ARGS.epochs // 20) == 0:
                    summary = lap.summary_str()
                    tqdm.write(f"Epoch {ep+1: >4}: Loss {avg_loss:.4f} | Grad norm {avg_grad_norm:.4f} | {summary}")

    final_acc = evaluate_model(make_wrapper(template, t, m, n, r), theta, Xte, Yte_onehot, 'binary', 'Final')

    # Save initial and final weights to a single compressed file
    os.makedirs('weights', exist_ok=True)
    weight_file = f'weights/model_weights_{model_name}.npz'
    np.savez(weight_file, initial=initial_theta_snapshot, final=theta)
    print(f"Initial and final weights saved to '{weight_file}'.")

    # ───────── Visualize Decision Boundary ─────────
    if ARGS.visualize_boundary:
        print("\n" + "="*50)
        print("Visualizing Decision Boundaries...")
        print("="*50)
        try:
            # --- Initial Boundary ---
            print("Visualizing initial boundary...")
            visualize_decision_boundary(
                make_wrapper(template, t, m, n, r), initial_theta_snapshot, m, Xtr, Ytr_onehot, 'binary',
                boundary=boundary_path,
                title='Initial Decision Boundary',
                resolution=ARGS.boundary_resolution,
                save_path=f'{SHAPE}/decision_boundary_initial_{model_name}.png',
                testing_accuracy=initial_acc,
                epochs=ARGS.epochs,
                sample_size=len(Xte)
            )

            # --- Final Boundary ---
            print("Visualizing final boundary...")
            visualize_decision_boundary(
                make_wrapper(template, t, m, n, r), theta, m, Xtr, Ytr_onehot, 'binary',
                boundary=boundary_path,
                title='Final Decision Boundary',
                resolution=ARGS.boundary_resolution,
                save_path=f'{SHAPE}/decision_boundary_final_{model_name}.png',
                testing_accuracy=final_acc,
                epochs=ARGS.epochs,
                sample_size=len(Xte)
            )
            
        except ImportError:
            print("\nWarning: Could not import 'visualize_decision_boundary' from 'star_eval.py'.")
            print("         Please ensure the file exists and is in the correct path.")
        except Exception as e:
            print(f"\nAn error occurred during boundary visualization: {e}")

    # Save loss plot
    os.makedirs(SHAPE, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(losses, 'o-')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Negative Log-Likelihood Loss (log scale)')
    plt.title(f'Training Loss Over Time ({model_name}) - Binary Cross-Entropy')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    loss_plot_file = f'{SHAPE}/loss_observable_{model_name}.png'
    plt.savefig(loss_plot_file, dpi=300)
    plt.close()

    print(f"Training complete. Loss plot saved to {loss_plot_file}")

if __name__ == '__main__':
    with suppress(KeyboardInterrupt):
        main() 