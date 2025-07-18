import os, sys, multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from joblib import Parallel, delayed
from tqdm import tqdm

# ───────── Add project root to sys.path ─────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT))

from star_data import get_star_data
from star_spqc import create_spqc_circuit

# ─── Globals for worker processes ───
g_tpl = None
g_input_params, g_model_params, g_address_params = None, None, None
g_theta = None
g_inside_mask = None

def worker_init(tpl, input_p, model_p, address_p, theta, inside_mask):
    """Initializer for each parallel worker."""
    global g_tpl, g_input_params, g_model_params, g_address_params, g_theta, g_inside_mask
    g_tpl = tpl
    g_input_params = input_p
    g_model_params = model_p
    g_address_params = address_p
    g_theta = theta
    g_inside_mask = inside_mask

def predict_chunk(points_chunk):
    """Predicts the 'inside' probability for a chunk of data points."""
    num_model_params = len(g_model_params)
    model_values = g_theta[:num_model_params]
    address_values = g_theta[num_model_params:]
    
    predictions = []
    for x in points_chunk:
        param_binding = {}
        for i, param in enumerate(g_input_params):
            param_binding[param] = x[i] if i < len(x) else 0.0
        for i, param in enumerate(g_model_params):
            param_binding[param] = model_values[i] if i < len(model_values) else 0.0
        for i, param in enumerate(g_address_params):
            param_binding[param] = address_values[i] if i < len(address_values) else 0.0
        
        bound_circuit = g_tpl.assign_parameters(param_binding)
        sv = Statevector.from_instruction(bound_circuit).data
        inside_prob = np.sum(np.abs(sv[g_inside_mask])**2)
        predictions.append(inside_prob)
        
    return predictions

def precompute_inside_mask(t, m, n, r):
    """Pre-computes a boolean mask for the 'inside' class."""
    num_qubits = t + m + n * r + 1
    inside_address_states = np.arange(2**(m-1), 2**m) << t
    num_other_qubits = n * r + 1
    other_states = np.arange(2**num_other_qubits) << (t + m)
    all_inside_indices = np.add.outer(inside_address_states, other_states).ravel()
    
    mask = np.zeros(2**num_qubits, dtype=bool)
    mask[all_inside_indices] = True
    return mask

def _plot_boundary_ax(ax, predict_fn, X, y, title):
    """Helper function to plot one decision boundary."""
    res = 256
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, res),
                         np.linspace(y_min, y_max, res))
    
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    print(f"  > Predicting {len(grid_points)} points for '{title}' heatmap...")
    Z = np.array(predict_fn(grid_points))
    Z = Z.reshape(xx.shape)
    
    extent = (x_min, x_max, y_min, y_max)
    im = ax.imshow(Z, interpolation='bilinear', origin='lower',
                   cmap='RdBu', extent=extent, aspect='auto',
                   alpha=0.8, vmin=0, vmax=1)
    
    ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
    ax.scatter(X[y==0][:, 0], X[y==0][:, 1], c='blue', edgecolors='k', label='Outside', s=20)
    ax.scatter(X[y==1][:, 0], X[y==1][:, 1], c='red', edgecolors='k', label='Inside', s=20)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.legend()
    ax.grid(True, alpha=0.2)
    return im

def plot_boundaries(template, initial_theta, final_theta, Xtr, ytr, t, m, n, r):
    """
    Generates and saves a side-by-side plot of the initial and final decision boundaries.
    This is the main function intended for import.
    """
    input_params = [p for p in template.parameters if p.name.startswith('input_theta')]
    model_params = [p for p in template.parameters if p.name.startswith('model')]
    address_params = [p for p in template.parameters if p.name.startswith('address_theta')]
    inside_mask = precompute_inside_mask(t, m, n, r)
    N_CPUS = mp.cpu_count()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # --- Initial Boundary ---
    print("\nPlotting initial decision boundary...")
    init_args = (template, input_params, model_params, address_params, initial_theta, inside_mask)
    with Parallel(n_jobs=N_CPUS, initializer=worker_init, initargs=init_args) as parallel:
        def predict_initial_grid(grid_points):
            num_chunks = parallel.n_jobs * 8
            chunks = np.array_split(grid_points, num_chunks)
            results = parallel(delayed(predict_chunk)(chunk) for chunk in tqdm(chunks, desc="  > Initial grid", ncols=80, smoothing=0.1))
            return np.concatenate(results)
        im1 = _plot_boundary_ax(ax1, predict_initial_grid, Xtr, ytr, 'Initial Decision Boundary')

    # --- Final Boundary ---
    print("Plotting final decision boundary...")
    init_args = (template, input_params, model_params, address_params, final_theta, inside_mask)
    with Parallel(n_jobs=N_CPUS, initializer=worker_init, initargs=init_args) as parallel:
        def predict_final_grid(grid_points):
            num_chunks = parallel.n_jobs * 8
            chunks = np.array_split(grid_points, num_chunks)
            results = parallel(delayed(predict_chunk)(chunk) for chunk in tqdm(chunks, desc="  > Final grid  ", ncols=80, smoothing=0.1))
            return np.concatenate(results)
        im2 = _plot_boundary_ax(ax2, predict_final_grid, Xtr, ytr, 'Final Decision Boundary')
    
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im2, cax=cbar_ax)
    cbar.set_label('Probability of "Inside" Class', rotation=270, labelpad=15)
    
    plots_dir = 'plots'
    os.makedirs(plots_dir, exist_ok=True)
    save_path = os.path.join(plots_dir, 'decision_boundaries.png')
    plt.suptitle('Model Decision Boundaries', fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"Decision boundary plot saved to '{save_path}'\n")

def main():
    """Main function for standalone execution."""
    Xtr, _, ytr, _, _ = get_star_data(300)
    t, m, n, r = 0, 3, 2, 1

    frame = create_spqc_circuit(t=t, m=m, n=n, r=r)
    qc = QuantumCircuit(frame.num_qubits, name="SPQC")
    for inst in frame.data:
        if inst.operation.name != "measure":
            qc.append(inst.operation, inst.qubits)
            
    print("Transpiling circuit for prediction...")
    template = transpile(qc, optimization_level=1)
    print("Transpilation complete.")

    try:
        weights_data = np.load('weights/model_weights.npz')
        initial_theta = weights_data['initial']
        final_theta = weights_data['final']
    except FileNotFoundError:
        print("Error: 'weights/model_weights.npz' not found. Run training first.")
        return

    plot_boundaries(template, initial_theta, final_theta, Xtr, ytr, t, m, n, r)

if __name__ == '__main__':
    main() 