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
        amplitudes = spqc_model.forward(x, θ)
        # In star_eval.py - evaluate_model function
        if mode == 'binary':
            # Only consider first 2 amplitudes for binary classification
            relevant_probs = np.abs(amplitudes)**2
            predicted_class = np.argmax(relevant_probs[:2])
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
        label_type = "wedge" if mode == 'wedge' else "class"
        print(f"  Class {class_id} ({label_type}): {class_correct[class_id]}/{class_total[class_id]} = {class_acc:.4f} ({class_acc*100:.2f}%)")
    print("="*50)
    return accuracy

def visualize_decision_boundary(spqc_model, θ, m, test_features, test_labels_onehot, mode, 
                                boundary=None, title="Decision Boundary", resolution=100, save_path=None):
    """
    Visualizes the learned decision boundaries for the given classification mode and saves the plot.
    """
    print("\n" + "="*50)
    print(f"Generating plot: '{title}'")
    print("="*50)

    x_min, x_max = 0, 1
    y_min, y_max = 0, 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution), np.linspace(y_min, y_max, resolution))
    mesh_points = np.column_stack([xx.ravel(), yy.ravel()])

    # Get predictions for mesh grid
    mesh_predictions = []
    for point in tqdm(mesh_points, desc=f"Mesh Grid Predictions for '{title}'"):
        amplitudes = spqc_model.forward(point, θ)
        probs = np.abs(amplitudes)**2
        mesh_predictions.append(probs)
    mesh_predictions = np.array(mesh_predictions)

    fig, ax = plt.subplots(figsize=(10, 8))
    
    true_labels = np.argmax(test_labels_onehot, axis=1)

    if mode == 'wedge':
        # This part remains unchanged
        Z = np.argmax(mesh_predictions, axis=1).reshape(xx.shape)
        cmap = plt.get_cmap('viridis', 2**m)
        mappable = ax.contourf(xx, yy, Z, cmap=cmap, alpha=0.6, levels=np.arange(-0.5, 2**m, 1))
        fig.colorbar(mappable, ax=ax, ticks=range(2**m), label='Predicted Wedge Class')
        angles = np.linspace(0, 2 * np.pi, 2**m + 1)
        for angle in angles:
            ax.plot([0.5, 0.5 + 0.7 * np.cos(angle)], [0.5, 0.5 + 0.7 * np.sin(angle)], 'k:', linewidth=2)
        ax.scatter(test_features[:, 0], test_features[:, 1], c=true_labels, cmap=cmap, edgecolors='k', s=50, label="Test Data")

    elif mode == 'binary':
        binary_probs = mesh_predictions[:, :2]
        binary_probs_normalized = binary_probs / (binary_probs.sum(axis=1, keepdims=True) + 1e-8)
        Z = binary_probs_normalized[:, 1].reshape(xx.shape)  # Probability of class 1 (outside star)
        
        # Use the Red-Yellow-Blue (reversed) colormap to match the user's image
        mappable = ax.contourf(xx, yy, Z, levels=20, cmap='RdYlBu_r', alpha=0.8, vmin=0, vmax=1)
        
        # Draw the learned boundary as a solid, bright green line for high visibility
        ax.contour(xx, yy, Z, levels=[0.5], colors='lime', linewidths=3, linestyles='-')
        
        # Draw the target boundary as a solid black line
        if boundary:
            vertices = boundary.vertices
            ax.plot(vertices[:, 0], vertices[:, 1], 'k-', linewidth=3, label='True Boundary')
        
        inside = test_features[true_labels == 0]
        outside = test_features[true_labels == 1]
        ax.scatter(inside[:, 0], inside[:, 1], c='red', s=40, alpha=0.9, marker='o', edgecolors='white', linewidth=1, label='Inside (True)')
        ax.scatter(outside[:, 0], outside[:, 1], c='blue', s=40, alpha=0.9, marker='o', edgecolors='white', linewidth=1, label='Outside (True)')

        cbar = fig.colorbar(mappable, ax=ax)
        cbar.set_label('Probability of "Outside" Class', rotation=270, labelpad=15)

    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(True, alpha=0.2)
    ax.set_title(title); ax.set_xlabel('X coordinate'); ax.set_ylabel('Y coordinate')
    ax.set_aspect('equal', adjustable='box')
    ax.legend(loc='upper right')
    fig.tight_layout()
    
    if save_path:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Plot successfully saved to '{save_path}'")
    else:
        plt.show()
    
    plt.close(fig) # Free up memory

    # Calculate accuracy on the mesh grid
    true_mesh_labels_onehot = wedge_onehot(mesh_points, 2**m) if mode == 'wedge' else None
    if mode == 'binary' and boundary is not None:
        true_mesh_labels = np.array([0 if boundary.contains_point(p) else 1 for p in mesh_points])
    elif mode == 'wedge':
        true_mesh_labels = np.argmax(wedge_onehot(mesh_points, m), axis=1)
    else:
        true_mesh_labels = None
        
    if true_mesh_labels is not None:
        if mode == 'binary':
            pred_mesh_labels = np.argmax(mesh_predictions[:, :2], axis=1)
        else:
            pred_mesh_labels = np.argmax(mesh_predictions, axis=1)
        mesh_accuracy = np.mean(pred_mesh_labels == true_mesh_labels)
        print(f"\nAccuracy on mesh grid: {mesh_accuracy:.4f} ({mesh_accuracy*100:.2f}%)")

    print("="*50) 