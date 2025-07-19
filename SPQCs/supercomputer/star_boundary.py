import os
import sys
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, SparsePauliOp
from joblib import Parallel, delayed
from tqdm import tqdm

try:
    from qiskit.primitives import StatevectorEstimator
    V2_ESTIMATOR_AVAILABLE = True
except ImportError:
    V2_ESTIMATOR_AVAILABLE = False

try:
    from qiskit_aer.primitives import Estimator as AerEstimator
    AER_ESTIMATOR_AVAILABLE = True
except ImportError:
    AER_ESTIMATOR_AVAILABLE = False

from qiskit.primitives import Estimator as BasicEstimator
from qiskit_aer import AerSimulator

# ───────── Add project root to sys.path ─────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT))

from star_data import get_star_data
from star_spqc import create_spqc_circuit

# ─── Globals for worker processes ───
g_estimator = None
g_template = None
g_observable = None
g_input_params = None
g_model_params = None
g_address_params = None
g_theta = None
g_param_map = None

# ──────────────────────────────────────────────────────────────────────────────
# NEW: GPU & Vectorization-Aware Worker Implementation
# ──────────────────────────────────────────────────────────────────────────────

def assign_gpu(worker_id):
    """Assigns a GPU to a worker based on its ID."""
    try:
        import os
        gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(',')
        if gpus and gpus[0]: # Check if CUDA_VISIBLE_DEVICES is set and not empty
            gpu_to_use = gpus[worker_id % len(gpus)]
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_to_use)
            os.environ.setdefault("QISKIT_AER_CUDA", "1")
            return True
    except Exception:
        pass
    return False

def create_estimator(use_gpu):
    """Creates an appropriate Estimator, configured for CPU or GPU with fallbacks."""
    
    # For V2, try to pass backend_options for GPU. If it fails, create without.
    if V2_ESTIMATOR_AVAILABLE:
        try:
            if use_gpu:
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
        device = "GPU" if use_gpu else "CPU"
        
        sim_kwargs = {'method': 'statevector', 'device': device}
        if device == "CPU":
            # Apply CPU-specific optimization to avoid Python gate overhead
            sim_kwargs['disable_pygate_callback'] = True

        backend = AerSimulator(**sim_kwargs)
        
        if AER_ESTIMATOR_AVAILABLE:
            try:
                return AerEstimator(backend=backend)
            except Exception:
                pass
        
        # Fall back to BasicEstimator
        try:
            return BasicEstimator(backend=backend) 
        except TypeError:
            # If backend param not supported, use without it
            return BasicEstimator()

def worker_init_with_id(use_gpu, template, observable, input_p, model_p, address_p, theta):
    """Initializer that assigns a unique ID to each worker."""
    worker_id = os.getpid() % 16  # Use process ID modulo 16 for worker ID
    worker_init(worker_id, use_gpu, template, observable, input_p, model_p, address_p, theta)

def worker_init(worker_id, use_gpu, template, observable, input_p, model_p, address_p, theta):
    """Initializer for each parallel worker process."""
    # On Windows, frozen executables may need to re-import the module
    # to ensure globals are correctly initialized in each worker process.
    import star_boundary
    
    global g_estimator, g_template, g_observable, g_input_params, g_model_params, g_address_params, g_theta, g_param_map
    
    gpu_assigned = False
    if use_gpu:
        gpu_assigned = assign_gpu(worker_id)
        
    g_estimator = create_estimator(use_gpu=gpu_assigned)
    g_template = template
    g_observable = observable
    g_input_params = input_p
    g_model_params = model_p
    g_address_params = address_p
    g_theta = theta
    # Cache the parameter map once per worker
    g_param_map = {p: i for i, p in enumerate(g_template.parameters)}


def _is_v2_estimator(est):
    """Check if the estimator is a V2 primitive."""
    return est.__class__.__name__.lower().endswith("statevectorestimator")

def _estimator_values(est, run_result):
    """
    Normalize estimator results across v1/v2.
    Returns 1D np.array of expectation values.
    """
    if _is_v2_estimator(est):
        # run_result is a PrimitiveResult; for a single PUB with matrix parameters,
        # we get one result with evs array containing all expectation values
        try:
            # Access the first (and only) PUB result
            pub_result = run_result[0]
            evs = pub_result.data.evs
            return np.array(evs).flatten()  # Ensure 1D array
        except (IndexError, AttributeError) as e:
            print(f"Debug: V2 result structure issue: {e}")
            print(f"Debug: run_result type: {type(run_result)}")
            if hasattr(run_result, '__len__'):
                print(f"Debug: run_result length: {len(run_result)}")
            return np.array([])
    else:
        return np.array(run_result.values)

def predict_chunk_vectorized(points_chunk):
    """
    Predicts the 'inside' probability for a chunk of data points using a
    single vectorized call to the Qiskit Estimator.
    """
    num_model_params = len(g_model_params)
    model_values = g_theta[:num_model_params]
    address_values = g_theta[num_model_params:]
    
    # 1. Pre-allocate a NumPy array for all parameter bindings in the chunk.
    # This avoids Python loop overhead.
    param_values_matrix = np.zeros((len(points_chunk), g_template.num_parameters))
    
    # Use the cached parameter map
    param_map = g_param_map
    
    # Bind input parameters (features)
    for i, param in enumerate(g_input_params):
        param_idx = param_map[param]
        param_values_matrix[:, param_idx] = points_chunk[:, i] if i < points_chunk.shape[1] else 0.0
        
    # Bind model and address parameters (weights)
    all_weights = np.concatenate([model_values, address_values])
    weight_params = g_model_params + g_address_params
    for i, param in enumerate(weight_params):
        param_idx = param_map[param]
        param_values_matrix[:, param_idx] = all_weights[i] if i < len(all_weights) else 0.0

    # 2. Run all simulations for the chunk in one vectorized call
    if _is_v2_estimator(g_estimator):
        # V2 API: use PUB format
        job = g_estimator.run([(g_template, g_observable, param_values_matrix)])
        result = job.result()
        return _estimator_values(g_estimator, result)
    else:
        # V1 API: traditional format  
        job = g_estimator.run([g_template] * len(points_chunk), [g_observable] * len(points_chunk), param_values_matrix.tolist())
        result = job.result()
        return _estimator_values(g_estimator, result)

# ──────────────────────────────────────────────────────────────────────────────
# Observable & Plotting Implementation
# ──────────────────────────────────────────────────────────────────────────────

def precompute_inside_observable(t, m, n, r):
    """
    Creates a SparsePauliOp to measure the "inside" probability.
    
    The "inside" class corresponds to states where the most significant bit
    of the `m`-qubit address register is 1. To measure this, we only need to
    measure the Pauli Z operator on that single qubit.

    The relationship between the expectation value ⟨Z⟩ and the probability
    of measuring the |1⟩ state, P(1), is:
      P(1) = (1 - ⟨Z⟩) / 2.
    
    This function returns the bare Z operator, and the calling code is
    responsible for this conversion. This is more efficient than building
    the full projector `(I - Z) / 2`.
    """
    num_qubits = t + m + n * r + 1
    
    # The address MSB is at qubit index `t + m - 1` (in Qiskit's little-endian order)
    msb_index = t + m - 1
    
    # Create the observable: Z on the MSB, I on all others.
    z_op = SparsePauliOp("I" * (num_qubits - 1 - msb_index) + "Z" + "I" * msb_index)
    return z_op

def _plot_boundary_ax(ax, Z, X, y, star_path, title):
    """
    Helper function to render the predictions onto a matplotlib axis.
    This function is responsible for all the plotting aesthetics.
    """
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    
    # Use imshow for fast rendering of the heatmap
    im = ax.imshow(Z, interpolation='bilinear', origin='lower',
                   cmap='RdYlBu_r', extent=(x_min, x_max, y_min, y_max),
                   aspect='auto', vmin=0, vmax=1)

    # Overlay contours on top of the heatmap
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, Z.shape[1]),
                         np.linspace(y_min, y_max, Z.shape[0]))

    # Learned boundary (p=0.5) in bright green
    ax.contour(xx, yy, Z, levels=[0.5], colors='lime', linewidths=3, linestyles='-')
    
    # True star boundary in solid black
    if star_path:
        vertices = star_path.vertices
        ax.plot(vertices[:, 0], vertices[:, 1], 'k-', linewidth=3, label='True Boundary')

    # Plot data points (Inside=0=Red, Outside=1=Blue)
    ax.scatter(X[y==0][:, 0], X[y==0][:, 1], c='red', edgecolors='white', linewidth=1, label='Inside', s=30, alpha=0.9)
    ax.scatter(X[y==1][:, 0], X[y==1][:, 1], c='blue', edgecolors='white', linewidth=1, label='Outside', s=30, alpha=0.9)
    
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.2)
    
    return im

def plot_decision_boundary(template, theta, X, y, star_path, t, m, n, r, save_path, title, resolution=256, use_gpu=False, num_cpus=None):
    """
    Generates and saves a high-performance plot of the decision boundary.
    This is the main public function of this script.
    """
    # 1. Pre-computation
    input_params = [p for p in template.parameters if p.name.startswith('input_theta')]
    model_params = [p for p in template.parameters if p.name.startswith('model')]
    address_params = [p for p in template.parameters if p.name.startswith('address_theta')]
    
    # Create the observable that measures the probability of being 'inside'
    # by checking the address MSB.
    z_observable = precompute_inside_observable(t, m, n, r)
    
    # Determine the number of parallel workers based on the execution mode (GPU vs CPU)
    if use_gpu:
        try:
            # For GPU execution, a common pattern is one worker per available GPU.
            gpus_str = os.environ.get("CUDA_VISIBLE_DEVICES")
            if gpus_str and gpus_str.strip():
                num_workers = len(gpus_str.split(','))
            else:
                # If the env var is not set, assume one GPU is being targeted.
                num_workers = 1
        except Exception:
            num_workers = 1 # Fallback for safety.
    else:
        # For CPU execution, use the specified number of cores, or default to all.
        num_workers = num_cpus if num_cpus is not None else mp.cpu_count()
        # Prevent BLAS/NumPy threads from competing with joblib workers by default.
        # This prevents oversubscription and is crucial for performance on many-core CPUs.
        os.environ.setdefault("OMP_NUM_THREADS", "1")
    
    # Prevent CPU oversubscription if joblib is asked to spawn more workers than there are physical cores.
    # This can happen if the user specifies a high --cpus value.
    if not use_gpu and num_workers > mp.cpu_count():
        print(f"Warning: Number of workers ({num_workers}) exceeds physical CPU cores ({mp.cpu_count()}).")
        print("Setting OMP_NUM_THREADS=1 to prevent thread contention.")
        os.environ["OMP_NUM_THREADS"] = "1"

    print(f"Generating '{title}' with {resolution}x{resolution} grid ({resolution**2:,} points) using {num_workers} workers (GPU: {use_gpu}).")

    # 2. Create the mesh grid of points to predict
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                         np.linspace(y_min, y_max, resolution))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # 3. Chunk the grid for parallel processing. Chunk size of 8192 is ideal for cache performance.
    chunk_size = 8192
    chunks = [grid_points[i:i + chunk_size] for i in range(0, len(grid_points), chunk_size)]

    # 4. Use a robust approach that works on all platforms, especially Windows
    # On Windows, we'll use threading to avoid multiprocessing issues with global variable initialization
    import platform
    is_windows = platform.system() == "Windows"
    
    if is_windows and num_workers > 1:
        print("Windows detected: Using threading backend for compatibility.")
        # Use threading backend on Windows for better compatibility
        backend = "threading"
        prefer = "threads"
        # Create shared objects that can be accessed by threads
        shared_data = {
            'estimator': create_estimator(use_gpu=use_gpu),
            'template': template,
            'observable': z_observable,
            'input_params': input_params,
            'model_params': model_params,
            'address_params': address_params,
            'theta': theta,
            'param_map': {p: i for i, p in enumerate(template.parameters)}
        }
        
        def process_chunk_shared(chunk):
            return predict_chunk_with_shared_data(chunk, shared_data)
        
        results = Parallel(n_jobs=num_workers, prefer=prefer)(
            delayed(process_chunk_shared)(chunk)
            for chunk in tqdm(chunks, desc="  > Processing chunks")
        )
    else:
        # Use process-based parallelism on Linux/HPC or single-threaded on Windows
        if num_workers == 1:
            # Single-threaded execution - initialize estimator in main process
            estimator = create_estimator(use_gpu=use_gpu)
            param_map = {p: i for i, p in enumerate(template.parameters)}
            
            def process_chunk_single(chunk):
                return predict_chunk_single_process(chunk, estimator, template, z_observable, 
                                                  input_params, model_params, address_params, 
                                                  theta, param_map)
            
            results = [process_chunk_single(chunk) for chunk in tqdm(chunks, desc="  > Processing chunks")]
        else:
            # Multi-process execution (Linux/Unix)
            init_args = (use_gpu, template, z_observable, input_params, model_params, address_params, theta)
            
            results = Parallel(n_jobs=num_workers, prefer='processes', initializer=worker_init_with_id, initargs=init_args)(
                delayed(predict_chunk_vectorized)(chunk)
                for chunk in tqdm(chunks, desc="  > Processing chunks")
            )
    
    # 5. Collect results and plot
    exp_vals = np.concatenate(results)
    
    # Convert expectation value of Z to probability of |1>
    # E[Z] = P(0) - P(1). Since P(0)+P(1)=1, E[Z] = (1-P(1)) - P(1) = 1 - 2*P(1)
    # So, P(1) = (1 - E[Z]) / 2. P(1) is our 'inside_prob'.
    inside_probs = (1 - exp_vals) / 2
    
    Z = inside_probs.reshape(xx.shape)
    
    # Convert probability of 'inside' to 'outside' to match the desired colormap
    Z_outside = 1.0 - Z
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = _plot_boundary_ax(ax, Z_outside, X, y, star_path, title)
    
    # Finalize plot layout and save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.tight_layout()
    
    # Add colorbar after tight_layout to avoid conflicts
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Probability of "Outside" Class', rotation=270, labelpad=15)
    plt.savefig(save_path, dpi=300)
    
    # Explicitly free large arrays from memory
    del Z, Z_outside, inside_probs, exp_vals, grid_points, results
    plt.close()
    
    print(f"Decision boundary plot saved to '{save_path}'\n")

def predict_chunk_with_shared_data(points_chunk, shared_data):
    """Predict using shared data objects (for threading backend)."""
    estimator = shared_data['estimator']
    template = shared_data['template']
    observable = shared_data['observable']
    input_params = shared_data['input_params']
    model_params = shared_data['model_params']
    address_params = shared_data['address_params']
    theta = shared_data['theta']
    param_map = shared_data['param_map']
    
    num_model_params = len(model_params)
    model_values = theta[:num_model_params]
    address_values = theta[num_model_params:]
    
    # 1. Pre-allocate a NumPy array for all parameter bindings in the chunk.
    param_values_matrix = np.zeros((len(points_chunk), template.num_parameters))
    
    # Bind input parameters (features)
    for i, param in enumerate(input_params):
        param_idx = param_map[param]
        param_values_matrix[:, param_idx] = points_chunk[:, i] if i < points_chunk.shape[1] else 0.0
        
    # Bind model and address parameters (weights)
    all_weights = np.concatenate([model_values, address_values])
    weight_params = model_params + address_params
    for i, param in enumerate(weight_params):
        param_idx = param_map[param]
        param_values_matrix[:, param_idx] = all_weights[i] if i < len(all_weights) else 0.0

    # 2. Run all simulations for the chunk in one vectorized call
    if _is_v2_estimator(estimator):
        # V2 API: use PUB format
        job = estimator.run([(template, observable, param_values_matrix)])
        result = job.result()
        return _estimator_values(estimator, result)
    else:
        # V1 API: traditional format
        job = estimator.run([template] * len(points_chunk), [observable] * len(points_chunk), param_values_matrix.tolist())
        result = job.result()
        return _estimator_values(estimator, result)

def predict_chunk_single_process(points_chunk, estimator, template, observable, input_params, 
                                model_params, address_params, theta, param_map):
    """Predict using single process (no multiprocessing)."""
    num_model_params = len(model_params)
    model_values = theta[:num_model_params]
    address_values = theta[num_model_params:]
    
    # 1. Pre-allocate a NumPy array for all parameter bindings in the chunk.
    param_values_matrix = np.zeros((len(points_chunk), template.num_parameters))
    
    # Bind input parameters (features)
    for i, param in enumerate(input_params):
        param_idx = param_map[param]
        param_values_matrix[:, param_idx] = points_chunk[:, i] if i < points_chunk.shape[1] else 0.0
        
    # Bind model and address parameters (weights)
    all_weights = np.concatenate([model_values, address_values])
    weight_params = model_params + address_params
    for i, param in enumerate(weight_params):
        param_idx = param_map[param]
        param_values_matrix[:, param_idx] = all_weights[i] if i < len(all_weights) else 0.0

    # 2. Run all simulations for the chunk in one vectorized call
    if _is_v2_estimator(estimator):
        # V2 API: use PUB format
        job = estimator.run([(template, observable, param_values_matrix)])
        result = job.result()
        return _estimator_values(estimator, result)
    else:
        # V1 API: traditional format
        job = estimator.run([template] * len(points_chunk), [observable] * len(points_chunk), param_values_matrix.tolist())
        result = job.result()
        return _estimator_values(estimator, result)

def main():
    """
    Main function for standalone execution. This allows running this script
    directly to generate plots from saved weights.
    """
    import argparse
    parser = argparse.ArgumentParser(description="Generate high-performance decision boundary plots.")
    parser.add_argument("--resolution", type=int, default=256, help="Grid resolution for visualization (default: 256)")
    parser.add_argument("--cpus", type=int, default=None, help="Number of CPU cores for fallback. Defaults to all available.")
    parser.add_argument("--gpu", action="store_true", help="Attempt to use GPU(s) for acceleration.")
    args = parser.parse_args()

    # --- Configuration ---
    t, m, n, r = 0, 3, 2, 1
    
    use_gpu_flag = False
    if args.gpu:
        try:
            import qiskit_aer
            # Robustly check for GPU support. This will raise an error if qiskit-aer-gpu
            # is installed but the CUDA drivers/runtime are missing or incompatible.
            sim = qiskit_aer.AerSimulator(method='statevector', device='GPU')
            if sim.properties() is None:
                raise RuntimeError("AerSimulator GPU backend failed to report properties.")
            print("GPU backend detected and available for Qiskit Aer.")
            use_gpu_flag = True
        except Exception as e:
            print(f"GPU usage requested, but an error occurred during initialization: {e}")
            print("This can happen if CUDA drivers are missing or incompatible. Falling back to CPU.")
            use_gpu_flag = False

    print("Loading data and circuit...")
    Xtr, _, ytr, _, star_path = get_star_data(300)

    # --- Create and Transpile Circuit ---
    frame = create_spqc_circuit(t=t, m=m, n=n, r=r)
    qc = QuantumCircuit(frame.num_qubits, name="SPQC")
    for inst in frame.data:
        if inst.operation.name != "measure":
            qc.append(inst.operation, inst.qubits)
            
    print("Transpiling circuit...")
    template = transpile(qc, optimization_level=1)
    print("Transpilation complete.")

    # --- Load Weights ---
    try:
        weights_data = np.load('weights/model_weights.npz')
        initial_theta = weights_data['initial']
        final_theta = weights_data['final']
    except FileNotFoundError:
        print("Error: 'weights/model_weights.npz' not found. Run training first.")
        return

    # --- Generate Plots ---
    plot_decision_boundary(
        template, initial_theta, Xtr, ytr, star_path, t, m, n, r,
        save_path='plots/boundary_initial.png',
        title='Initial Decision Boundary',
        resolution=args.resolution,
        use_gpu=use_gpu_flag,
        num_cpus=args.cpus
    )
    
    plot_decision_boundary(
        template, final_theta, Xtr, ytr, star_path, t, m, n, r,
        save_path='plots/boundary_final.png',
        title='Final Decision Boundary',
        resolution=args.resolution,
        use_gpu=use_gpu_flag,
        num_cpus=args.cpus
    )

if __name__ == '__main__':
    main() 