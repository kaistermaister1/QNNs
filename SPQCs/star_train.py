from star_data import get_star_data, visualize_data
from star_spqc import create_spqc_circuit, visualize_circuit, bind_params, create_random_weights, post_select, get_parameter_mapping, pre_select_convert, model, mse_loss
import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator
from tqdm import tqdm

# PyTorch imports
import torch
from torch import nn, optim
import torch.utils.data

from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit.quantum_info import SparsePauliOp, Statevector, Operator
from qiskit_aer import AerSimulator

# --- Retrieve star classification data ---
train_features, test_features, train_labels, test_labels, boundary = get_star_data()
# visualize_data()

# --- Create the SPQC circuit ---
r = 2 # Polynomial degree
m = 3 # Number of log(n) submodels  
n = 2 # Number of input features
t = 0 # Polynomial terms
spqc_frame = create_spqc_circuit(t=t, m=m, n=n, r=r)

print(f"Total qubits (not counting ancilla): {spqc_frame.num_qubits-1}")
print(f"Term register size: {t}")
print(f"Address register size: {m}")
print(f"Data register size: {n}")
print(f"Number of data registers: {r} \n")

# --- Reformat labels from 0 or 1 to one-hot vectors of size 2^m ---
def to_onehot(labels, m):
    """Convert scalar labels to one-hot vectors of size 2^m."""
    num_classes = 2**m
    onehot = np.zeros((len(labels), num_classes))
    onehot[np.arange(len(labels)), labels.astype(int)] = 1
    return onehot
train_labels = to_onehot(train_labels, m)
test_labels = to_onehot(test_labels, m)

if False:
    # --- Create and bind random weights + features ---
    random_weights = create_random_weights(spqc_frame, seed=42) # Ouputs an array of submodel + address weights
    example_input = train_features[0]
    spqc = bind_params(spqc_frame, example_input, random_weights) # Binds features and weights to circuit

    # --- Run the circuit ---
    simulator = AerSimulator() 
    circ = transpile(spqc, simulator) # Turn circuit into instructions for simulator
    sim = AerSimulator(method='automatic') # Construct simulator for circuit
    job = sim.run(circ, shots=8192)
    counts = job.result().get_counts(0)
    pre_select_probability_vector = pre_select_convert(counts, m, n, r) # Pre selected probability vector
    probability_vector = post_select(counts, m, n, r) # Post-select states with only 0s in the data registers
    print(f"Crude post-selection probability vector: {probability_vector}") # [0.25, 0.25, 0.0, 0.5]
    print(f"Amount of non-zero probabilities (crude): {np.count_nonzero(probability_vector)}")
    print(f"Sum of crude vector: {np.sum(probability_vector)}")


# --- Remove measurements from circuit ---
from qiskit import QuantumCircuit
qc_qnn = QuantumCircuit(spqc_frame.num_qubits)
for instr in spqc_frame.data:
    if instr.operation.name != 'measure':
        qc_qnn.append(instr.operation, instr.qubits, instr.clbits)

# --- Create SPQC model for ADAM ---
class SPQCModel:
    def __init__(self, qc, t, m, n, r):
        self.qc = qc
        self.t, self.m, self.n, self.r = t, m, n, r

    def forward(self, input_vals, weights):
        return model(self.qc, input_vals, weights, self.t, self.m, self.n, self.r)
    
    def loss(self, x, θ, y_true):
        y_pred = self.forward(x, θ)
        err = y_pred - y_true
        return float(np.mean(np.abs(err)**2))
    
    def gradient(self, x, θ, y_true, shift=np.pi/2):
        grads = np.zeros_like(θ)
        for i in range(len(θ)):
            θp, θm = θ.copy(), θ.copy()
            θp[i] += shift; θm[i] -= shift
            lp = self.loss(x, θp, y_true)
            lm = self.loss(x, θm, y_true)
            grads[i] = 0.5*(lp - lm)
        return grads

# visualize_circuit(qc_qnn)
     
# --- Train the model with ADAM ---
if True:
    spqc_model = SPQCModel(qc_qnn, t, m, n, r)
    θ = create_random_weights(spqc_frame, seed=42) # Initial random weights
    m1 = np.zeros_like(θ) # first moment
    v1 = np.zeros_like(θ) # second moment
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8 # Adam hyperparams
    alpha = 0.01 # learning rate
    epochs = 10 # total passes over data
    for epoch in tqdm(range(1, epochs+1), desc="Training"):
        perm = np.random.permutation(len(train_features)) # Shuffle data
        
        for i in perm:
            x = train_features[i]
            y_true = train_labels[i]

            # ---- compute gradient via parameter shift ----
            g = spqc_model.gradient(x, θ, y_true)

            # ---- update Adam moments ----
            m1 = beta1*m1 + (1 - beta1)*g
            v1 = beta2*v1 + (1 - beta2)*(g*g)
            m_hat = m1 / (1 - beta1**epoch)
            v_hat = v1 / (1 - beta2**epoch)

            # ---- update parameters ----
            θ -= alpha * m_hat / (np.sqrt(v_hat) + epsilon)

    # --- Evaluate on test set ---
    test_losses = []
    for x, y_true in zip(test_features, test_labels):
        test_losses.append(spqc_model.loss(x, θ, y_true))
    print("Test set MSE:", np.mean(test_losses))