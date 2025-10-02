import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
from matplotlib.lines import Line2D

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

import os
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

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
    n_jobs=None,
    cmap="RdYlBu_r",
    save_path=None,
    testing_accuracy=None,  # New parameter for testing accuracy
    epochs=None,           # New parameter for number of epochs
    sample_size=None,      # New parameter for sample size
):
    assert mode == 'binary', "This implementation currently supports mode='binary' only."

    # --- Mesh ---
    xs = np.linspace(xlim[0], xlim[1], resolution)
    ys = np.linspace(ylim[0], ylim[1], resolution)
    xx, yy = np.meshgrid(xs, ys)
    mesh_points = np.column_stack([xx.ravel(), yy.ravel()])

    # --- Parallel eval ---
    if n_jobs is None:
        n_jobs = mp.cpu_count()

    half = (2 ** m) // 2  # first half -> p_out, second half -> p_in

    def _p_in(point):
        amps = spqc_model.forward(point, θ)
        probs = np.abs(amps) ** 2
        return probs[half:].sum()

    p_in_vals = Parallel(n_jobs=n_jobs, backend="loky")(delayed(_p_in)(p) for p in mesh_points)
    Z = np.array(p_in_vals).reshape(xx.shape)

    # --- Plot ---
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
    
    # Add testing information below the plot if provided
    if testing_accuracy is not None or epochs is not None or sample_size is not None:
        info_text = []
        if testing_accuracy is not None:
            info_text.append(f"Testing Accuracy: {testing_accuracy:.4f} ({testing_accuracy*100:.2f}%)")
        if epochs is not None:
            info_text.append(f"Epochs: {epochs}")
        if sample_size is not None:
            info_text.append(f"Sample Size: {sample_size}")
        
        # Add text below the plot
        info_string = " | ".join(info_text)
        fig.text(0.5, 0.02, info_string, ha='center', va='bottom', fontsize=10, 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.8))
        
        # Adjust layout to make room for the text
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.12)
    else:
        fig.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

    return Z, (xx, yy)