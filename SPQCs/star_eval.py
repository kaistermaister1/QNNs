import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
from matplotlib.lines import Line2D
import multiprocessing as mp
from joblib import Parallel, delayed

def wedge_onehot(points, m=3, center=(0.5, 0.5)):
    """Assigns points to 2**m angular wedges and returns a one-hot encoding."""
    L = 2**m
    thetas = np.arctan2(points[:,1] - center[1], points[:,0] - center[0])
    thetas = (thetas + 2*np.pi) % (2*np.pi)
    bins = (thetas / (2*np.pi) * L).astype(int)
    Y = np.zeros((len(points), L))
    Y[np.arange(len(points)), bins] = 1
    return Y

def visualize_class_data(features, labels_onehot, mode, boundary=None, title="Dataset Visualization"):
    """Visualizes the dataset based on the classification mode."""
    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    plt.title(title)
    
    true_labels = np.argmax(labels_onehot, axis=1)
    
    if mode == 'wedge':
        m = int(np.log2(labels_onehot.shape[1]))
        cmap = plt.get_cmap('viridis', 2**m)
        plt.scatter(features[:, 0], features[:, 1], c=true_labels, cmap=cmap, edgecolors='k', s=50)
        
        # Plot wedge boundaries
        angles = np.linspace(0, 2 * np.pi, 2**m + 1)
        for angle in angles:
            ax.plot([0.5, 0.5 + 0.7 * np.cos(angle)], [0.5, 0.5 + 0.7 * np.sin(angle)], 'k:', linewidth=2)
        
    elif mode == 'binary':
        inside_points = features[true_labels == 0]
        outside_points = features[true_labels == 1]
        plt.scatter(inside_points[:, 0], inside_points[:, 1], c='red', s=50, alpha=0.7, label='Inside')
        plt.scatter(outside_points[:, 0], outside_points[:, 1], c='blue', s=50, alpha=0.7, label='Outside')
        
        # Plot star boundary
        if boundary:
            vertices = boundary.vertices
            plt.plot(vertices[:, 0], vertices[:, 1], 'k:', linewidth=3, label='True Boundary')
        plt.legend()

    plt.xlim(0, 1); plt.ylim(0, 1)
    plt.grid(True, alpha=0.2)
    plt.xlabel('X coordinate'); plt.ylabel('Y coordinate')
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.show()


def evaluate_model(spqc_model, θ, test_features, test_labels_onehot, mode, title="Model Evaluation"):
    """Performs a comprehensive evaluation of the model on the test set."""
    print("\n" + "="*50)
    print(title)
    print("="*50)

    test_predictions = []
    correct_predictions = 0
    true_labels = np.argmax(test_labels_onehot, axis=1)

    print("Running predictions on test set...")
    for i, x in enumerate(test_features):
        amplitudes = spqc_model.forward(x, θ) # Returns a 2^m vector of synthetic amplitudes (sum of first half are p_out, sum of second half are p_in)
        # In star_eval.py - evaluate_model function
        if mode == 'binary':
            # Sum addresses 0-3 for class 1 (outside), 4-7 for class 0 (inside)
            relevant_probs = np.abs(amplitudes)**2
            sum1 = np.sum(relevant_probs[0:4])  # addresses 0-3: class 1, outside
            sum0 = np.sum(relevant_probs[4:8])  # addresses 4-7: class 0, inside
            predicted_class = np.argmax([sum0, sum1])
        elif mode == 'wedge':
            # Use all amplitudes for wedge classification  
            predicted_class = np.argmax(np.abs(amplitudes)**2)
        if predicted_class == true_labels[i]:
            correct_predictions += 1
        test_predictions.append(predicted_class)
        
    accuracy = correct_predictions / len(test_features)
    print(f"\nTest Set Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    # --- Class-wise performance ---
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    for pred, true in zip(test_predictions, true_labels):
        class_total[true] += 1
        if pred == true:
            class_correct[true] += 1

    print(f"\nClass-wise Performance:")
    for class_id in sorted(class_total.keys()):
        class_acc = class_correct[class_id] / class_total[class_id] if class_total[class_id] > 0 else 0
        if class_id == 0:
            label_type = "inside"
        else:
            label_type = "outside"
        print(f"  Class {class_id} ({label_type}): {class_correct[class_id]}/{class_total[class_id]} = {class_acc:.4f} ({class_acc*100:.2f}%)")
    print("="*50)
    return accuracy

def visualize_decision_boundary(
    spqc_model,
    θ,
    m,
    test_features,
    test_labels_onehot,
    mode='binary',
    title="Decision boundary (p_in)",
    resolution=64,
    xlim=(0.0, 1.0),
    ylim=(0.0, 1.0),
    boundary=None,          # <— keep it, but optional
    n_jobs=None,            # kept for compatibility, but not used
    cmap="RdYlBu_r",
    save_path=None,
    chunk_size=None,        # NEW: optional chunking for memory management
):
    assert mode == 'binary', "This implementation currently supports mode='binary' only."
    
    # Import StatevectorEstimator here to avoid import issues
    from qiskit_ibm_runtime import StatevectorEstimator

    # --- Mesh ---
    xs = np.linspace(xlim[0], xlim[1], resolution)
    ys = np.linspace(ylim[0], ylim[1], resolution)
    xx, yy = np.meshgrid(xs, ys)
    mesh_points = np.column_stack([xx.ravel(), yy.ravel()])
    
    print(f"Processing {len(mesh_points)} mesh points in single batch...")

    # --- Step 1: Create one StatevectorEstimator ---
    estimator = StatevectorEstimator()

    # --- Step 2: Precompute parameter bookkeeping ---
    params = list(spqc_model.circuit.parameters)
    param_pos = {p: i for i, p in enumerate(params)}
    
    # Get input parameter indices
    input_params = [p for p in params if p.name.startswith('zinput_theta')]
    input_indices = [param_pos[p] for p in input_params]
    
    # Build base values vector with trainable parameters already filled
    base_values = np.zeros(len(params))
    for i, param_idx in enumerate(spqc_model.param_indices):
        if i < len(θ):
            base_values[param_idx] = θ[i]

    # --- Step 3 & 4: Process mesh (with optional chunking) ---
    P_out, P_in = spqc_model.projectors[0], spqc_model.projectors[1]
    
    # Determine if chunking is needed
    if chunk_size is None:
        # Auto-determine chunk size based on mesh size
        # For very large meshes (>50k points), use chunking
        if len(mesh_points) > 50000:
            chunk_size = 25000  # Process 25k points at a time
        else:
            chunk_size = len(mesh_points)  # Process all at once
    else:
        # If chunk_size is explicitly provided, ensure it's not larger than total points
        chunk_size = min(chunk_size, len(mesh_points))
    
    p_in_vals = []
    
    # Process in chunks
    total_chunks = (len(mesh_points) - 1) // chunk_size + 1
    for chunk_idx, chunk_start in enumerate(range(0, len(mesh_points), chunk_size)):
        chunk_end = min(chunk_start + chunk_size, len(mesh_points))
        chunk_points = mesh_points[chunk_start:chunk_end]
        
        if total_chunks > 1:
            print(f"Processing chunk {chunk_idx + 1}/{total_chunks} ({len(chunk_points)} points)")
        else:
            print(f"Running estimator with {len(chunk_points)*2} pubs...")
        
        # Build pubs for this chunk
        pubs = []
        for point in chunk_points:
            # Copy base values and set input parameters for this point
            values_for_point = base_values.copy()
            for i, input_idx in enumerate(input_indices):
                if i < len(point):
                    values_for_point[input_idx] = point[i]
            
            # Add two pubs for this point: (circuit, P_out, values) and (circuit, P_in, values)
            pubs.append((spqc_model.circuit, P_out, values_for_point))
            pubs.append((spqc_model.circuit, P_in, values_for_point))

        # Run estimator for this chunk
        result = estimator.run(pubs).result()

        # Post-process results for this chunk
        chunk_p_in_vals = []
        for i in range(len(chunk_points)):
            # Results come back in pairs: p_out at 2*i, p_in at 2*i+1
            p_out_raw = float(result[2*i].data.evs)
            p_in_raw = float(result[2*i+1].data.evs)
            
            # Compute normalized p_in (guard against zero sum)
            total_prob = p_in_raw + p_out_raw
            if total_prob > 1e-10:
                p_in = p_in_raw / total_prob
            else:
                p_in = 0.5  # Default to uncertain if no signal
            
            chunk_p_in_vals.append(p_in)
        
        p_in_vals.extend(chunk_p_in_vals)

    # Reshape to grid
    Z = np.array(p_in_vals).reshape(xx.shape)

    # --- Step 5: Plot as before ---
    fig, ax = plt.subplots(figsize=(8, 7))

    # test data
    true_labels = np.argmax(test_labels_onehot, axis=1)
    inside = test_features[true_labels == 0]
    outside = test_features[true_labels == 1]
    ax.scatter(inside[:, 0], inside[:, 1], c='red',  s=35, edgecolors='white', linewidth=0.8, label='Inside (true)')
    ax.scatter(outside[:, 0], outside[:, 1], c='blue', s=35, edgecolors='white', linewidth=0.8, label='Outside (true)')

    # heat map
    mappable = ax.contourf(xx, yy, Z, levels=100, cmap=cmap, alpha=0.7, vmin=0.0, vmax=1.0)
    # 0.5 line
    ax.contour(xx, yy, Z, levels=[0.5], colors='lime', linewidths=3)

    # optional true boundary
    if boundary:
        if isinstance(boundary, list):
            for i, path in enumerate(boundary):
                vertices = path.vertices
                # Only label the first blob to avoid legend spam
                label = 'Blob boundary' if i == 0 else None
                ax.plot(vertices[:, 0], vertices[:, 1], 'k-', linewidth=3, label=label)
        else:
            vertices = boundary.vertices
            ax.plot(vertices[:, 0], vertices[:, 1], 'k-', linewidth=3, label='True Boundary')

    cbar = fig.colorbar(mappable, ax=ax)
    cbar.set_label('p_in')

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    ax.grid(alpha=0.2)
    ax.legend(loc='upper right')
    fig.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

    return Z, (xx, yy)