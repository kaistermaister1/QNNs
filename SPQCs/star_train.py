from star_data import get_star_data
from star_spqc import create_spqc_circuit, create_random_weights, model, visualize_circuit
from star_eval import evaluate_model, visualize_decision_boundary, wedge_onehot, visualize_class_data
import numpy as np
from qiskit import QuantumCircuit
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

# --- Configuration ---
CLASSIFICATION_MODE = 'binary'  # Options: 'wedge' or 'binary'
VISUALIZE_DATA = False          # Show a plot of the data before evaluation
TRAIN_MODEL = False              # Set to False to skip training and only see random performance
RANDOM_SEED = None                # Set as None to use random seed

# --- Data Loading and Preprocessing ---
num_data_points = 300
train_features, test_features, train_labels, test_labels, boundary = get_star_data(num_data_points)
np.random.seed(RANDOM_SEED)
print(f"Using {num_data_points} total data points in '{CLASSIFICATION_MODE}' mode.")

# --- Circuit and Model Setup ---
m, r, n, t = 3, 1, 2, 0  # circuit parameters
spqc_frame = create_spqc_circuit(t=t, m=m, n=n, r=r)

# --- Model and Label Configuration ---
if CLASSIFICATION_MODE == 'wedge':
    train_labels_onehot = wedge_onehot(train_features, m)
    test_labels_onehot = wedge_onehot(test_features, m)
elif CLASSIFICATION_MODE == 'binary':
    # Convert scalar labels (0, 1) to one-hot vectors
    train_labels_onehot = np.eye(8)[train_labels.astype(int)]
    test_labels_onehot = np.eye(8)[test_labels.astype(int)]
else:
    raise ValueError(f"Unknown CLASSIFICATION_MODE: {CLASSIFICATION_MODE}")

qc_qnn = QuantumCircuit(spqc_frame.num_qubits)
for instr in spqc_frame.data:
    if instr.operation.name != 'measure':
        qc_qnn.append(instr.operation, instr.qubits, instr.clbits)

visualize_circuit(qc_qnn)

# --- Create SPQC model ---
class SPQCModel:
    def __init__(self, qc, t, m, n, r):
        self.qc, self.t, self.m, self.n, self.r = qc, t, m, n, r

    def forward(self, input_vals, weights):
        return model(self.qc, input_vals, weights, self.t, self.m, self.n, self.r)
    
    def loss(self, x, θ, y_true_onehot):
        amps = self.forward(x, θ)
        diff = amps - y_true_onehot
        return float(np.mean(np.abs(diff)**2))
    
    def gradient(self, x, θ, y_true_onehot, shift=np.pi/2):
        grads = np.zeros_like(θ)
        for i in range(len(θ)):
            θp, θm = θ.copy(), θ.copy()
            θp[i] += shift; θm[i] -= shift
            lp = self.loss(x, θp, y_true_onehot)
            lm = self.loss(x, θm, y_true_onehot)
            grads[i] = 0.5 * (lp - lm) / np.sin(shift)
        return grads

# visualize_circuit(qc_qnn)
spqc_model = SPQCModel(qc_qnn, t, m, n, r)
θ = create_random_weights(spqc_frame, seed=RANDOM_SEED)

# --- Testing how to interpret model output ---
if True:
    print(f"θ: {θ[0]}")

    # Test with a single example
    print(f"\nTesting model output on first test sample:")
    print(f"Input: {test_features[0]}")
    print(f"True label (one-hot): {test_labels_onehot[0]}")

    output_amps = spqc_model.forward(test_features[0], θ)

    # Convert to probabilities
    output_probs = np.abs(output_amps)**2
    print(f"Output probabilities: {output_probs}")

    if CLASSIFICATION_MODE == 'binary':
        # Only consider first 2 amplitudes for binary classification
        relevant_probs = output_probs[:2]
        predicted_class = np.argmax(relevant_probs)
        print(f"Relevant probabilities (first 2): {relevant_probs}")
    elif CLASSIFICATION_MODE == 'wedge':
        # Use all amplitudes for wedge classification  
        predicted_class = np.argmax(output_probs)
        print(f"All probabilities: {output_probs}")

    print(f"Predicted class: {predicted_class}")
    print(f"True class: {np.argmax(test_labels_onehot[0])}")

# --- Pre-Training Visualization and Evaluation ---
if VISUALIZE_DATA:
     visualize_class_data(train_features, train_labels_onehot, CLASSIFICATION_MODE, boundary, "Training Data Distribution")

print("\n--- Evaluating model with INITIAL RANDOM weights ---")
evaluate_model(spqc_model, θ, test_features, test_labels_onehot, CLASSIFICATION_MODE, "Initial Performance")
visualize_decision_boundary(spqc_model, θ, m, test_features, test_labels_onehot, CLASSIFICATION_MODE, boundary, "Initial Decision Boundary")
     
# --- Training ---
if TRAIN_MODEL:
    print("\n--- Starting model training ---")
    m1, v1 = np.zeros_like(θ), np.zeros_like(θ)
    beta1, beta2, alpha, epochs = 0.9, 0.999, 0.01, 50
    
    # Track loss over epochs
    epoch_losses = []

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        perm = np.random.permutation(len(train_features))
        
        for i in tqdm(perm, desc=f"Training Epoch {epoch}/{epochs}"):
            x, y_true = train_features[i], train_labels_onehot[i]
            
            # Compute loss for this sample
            sample_loss = spqc_model.loss(x, θ, y_true)
            epoch_loss += sample_loss
            
            # Compute gradient and update weights
            g = spqc_model.gradient(x, θ, y_true)
            m1 = beta1 * m1 + (1 - beta1) * g
            v1 = beta2 * v1 + (1 - beta2) * (g**2)
            m_hat, v_hat = m1 / (1 - beta1**epoch), v1 / (1 - beta2**epoch)
            θ -= alpha * m_hat / (np.sqrt(v_hat) + 1e-8)
        
        # Calculate average loss for this epoch
        avg_epoch_loss = epoch_loss / len(train_features)
        epoch_losses.append(avg_epoch_loss)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs} - Average Loss: {avg_epoch_loss:.6f}")
    
    # Create plots directory if it doesn't exist
    plots_dir = "SPQCs/plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot and save loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), epoch_losses, 'b-', linewidth=2, marker='o', markersize=4)
    plt.xlabel('Epoch')
    plt.ylabel('Average Training Loss')
    plt.title(f'Training Loss Over Time\n({CLASSIFICATION_MODE} classification, {num_data_points} data points)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    loss_plot_path = os.path.join(plots_dir, f"training_loss_{CLASSIFICATION_MODE}_{num_data_points}pts.png")
    plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\nLoss plot saved to: {loss_plot_path}")
    
    print("\n--- Evaluating model with TRAINED weights ---")
    evaluate_model(spqc_model, θ, test_features, test_labels_onehot, CLASSIFICATION_MODE, "Trained Performance")
    visualize_decision_boundary(spqc_model, θ, m, test_features, test_labels_onehot, CLASSIFICATION_MODE, boundary, "Trained Decision Boundary")
else:
    print("\nTraining skipped.")