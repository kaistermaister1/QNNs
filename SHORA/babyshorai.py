# babyshorai.py
"""
Baby Shor AI  –  A minimal quantum neural network that learns to output the
lowest (non-trivial) factor of every integer 2 … 15.

The task is treated as a supervised classification problem:
    input  : 4-bit binary encoding of the integer n (2 ≤ n ≤ 15)
    output : one-hot vector over 16 classes (index = lowest factor)
             e.g. n = 14 (1110₂) → lowest factor 2 → one-hot with 1 at index 2

Once the network predicts p, the co-factor is simply n // p, so the pair
(p, n//p) gives a full factorisation for composites and (n, 1) for primes.

The network uses a 4-qubit feature map (angle encoding) followed by a modest
entangling ansatz (RealAmplitudes reps=2).  Training is performed with
cross-entropy loss and either ADAM (gradient-based) or COBYLA (derivative-free)
optimisation – switchable via the OPTIMISER_CHOICE flag.

Tested with Qiskit > 0.45  (Sampler/Estimator primitives).
"""

from __future__ import annotations

# ─────────────────────────────  Imports  ──────────────────────────────
import numpy as np
import time
import os
from typing import List
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler
from qiskit_algorithms.optimizers import ADAM, COBYLA
from qiskit_algorithms.utils import algorithm_globals
from qiskit_machine_learning.neural_networks import SamplerQNN

# ─────────────────────  Reproducibility  ──────────────────────────────
algorithm_globals.random_seed = 42

# ─────────────────────  Data set 2 … 15  ─────────────────────────────
INPUTS: List[List[int]] = [  # 4-bit binary
    [int(b) for b in f"{n:04b}"] for n in range(2, 16)
]

LOWEST_FACTOR = {
    2: 2,
    3: 3,
    4: 2,
    5: 5,
    6: 2,
    7: 7,
    8: 2,
    9: 3,
    10: 2,
    11: 11,
    12: 2,
    13: 13,
    14: 2,
    15: 3,
}

LABELS: List[int] = [LOWEST_FACTOR[n] for n in range(2, 16)]

# One-hot targets over 16 classes (index 0/1 unused)
TARGETS = np.eye(16)[LABELS]

# ─────────────────────  Quantum circuit  ─────────────────────────────
N_QUBITS = 4
FEATURE_PARAMS = ParameterVector("x", length=N_QUBITS)

# Create ansatz and use its parameters
ansatz = RealAmplitudes(N_QUBITS, reps=2)
WEIGHT_PARAMS = list(ansatz.parameters)

qc = QuantumCircuit(N_QUBITS)

# Angle encoding: Ry(π * bit)  where bit ∈ {0,1}
for i in range(N_QUBITS):
    qc.ry(np.pi * FEATURE_PARAMS[i], i)

# Expressive ansatz
qc.compose(ansatz, inplace=True)

# Create plots directory if it doesn't exist
os.makedirs("plots", exist_ok=True)

# Draw and save the circuit
qc.draw(output="mpl", fold=20)
plt.suptitle("Baby Shor AI Circuit (4-Qubit)")
plt.savefig("plots/babyshoraicircuit.png", dpi=150, bbox_inches='tight')
plt.close()

# ─────────────────────  QNN definition  ──────────────────────────────
SAMPLER = StatevectorSampler()
QNN = SamplerQNN(
    circuit=qc,
    sampler=SAMPLER,
    input_params=list(FEATURE_PARAMS),
    weight_params=WEIGHT_PARAMS,
    # Default behavior gives us raw probabilities over 2^4 = 16 states
)

# Random initial weights
weights = algorithm_globals.random.random(QNN.num_weights)

# ─────────────────────  Optimiser choice  ────────────────────────────
OPTIMISER_CHOICE = "ADAM"  # "ADAM" or "COBYLA"
MAX_ITERS = 200

if OPTIMISER_CHOICE == "ADAM":
    optimiser = ADAM(maxiter=MAX_ITERS, lr=0.01)
elif OPTIMISER_CHOICE == "COBYLA":
    optimiser = COBYLA(maxiter=MAX_ITERS)
else:
    raise ValueError("OPTIMISER_CHOICE must be 'ADAM' or 'COBYLA'")

print(f"Using {OPTIMISER_CHOICE} optimiser with {MAX_ITERS} iterations …")

# ─────────────────────  Loss function  ───────────────────────────────
EPS = 1e-10

def cross_entropy(pred_probs: np.ndarray, target_one_hot: np.ndarray) -> float:
    """Cross-entropy between 16-element probability vector and one-hot target."""
    return float(-np.sum(target_one_hot * np.log(pred_probs + EPS)))

# ─────────────────────  Objective (batch)  ───────────────────────────
loss_history = []
iteration_count = [0]

def batch_loss(w: np.ndarray) -> float:
    loss = 0.0
    for x_bits, y_onehot in zip(INPUTS, TARGETS):
        probs = QNN.forward(np.asarray(x_bits), w)
        loss += cross_entropy(probs, y_onehot)
    
    # Track loss for plotting
    current_loss = loss / len(INPUTS)
    loss_history.append(current_loss)
    iteration_count[0] += 1
    
    if iteration_count[0] % 25 == 0:
        print(f"  Iteration {iteration_count[0]}: Average Loss = {current_loss:.4f}")
    
    return current_loss

# ─────────────────────  Training loop  ───────────────────────────────
print("Training …")
start = time.time()

# Both optimizers use the same minimize API
res = optimiser.minimize(batch_loss, weights)
weights_opt = res.x

elapsed = time.time() - start
print(f"Done in {elapsed:.2f} s  |  Final loss {batch_loss(weights_opt):.4f}\n")

# ─────────────────────  Evaluation  ──────────────────────────────────
print("Evaluating results...")
correct = 0
results = []
integers = list(range(2, 16))

for n, x_bits, y in zip(integers, INPUTS, LABELS):
    probs = QNN.forward(np.asarray(x_bits), weights_opt)
    pred_factor = int(np.argmax(probs))
    co_factor = n // pred_factor if pred_factor > 0 else None
    is_correct = pred_factor == y
    correct += is_correct
    
    results.append({
        'integer': n,
        'input': x_bits,
        'target': y,
        'predicted': pred_factor,
        'co_factor': co_factor,
        'correct': is_correct,
        'confidence': np.max(probs)
    })
    
    print(f"n={n:2d}  pred={pred_factor:2d}  target={y:2d}  » factors: ({pred_factor},{co_factor})  {'✔' if is_correct else '✘'}")

acc = correct / len(INPUTS)
print(f"\nAccuracy: {acc:.1%} ({correct}/{len(INPUTS)})")

# ─────────────────────  Plotting  ────────────────────────────────────
print("Generating plots...")

plt.figure(figsize=(18, 12))

# Plot 1: Training loss curve
plt.subplot(2, 3, 1)
plt.plot(range(1, len(loss_history) + 1), loss_history, 'b-', linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('Cross-Entropy Loss')
plt.title('Training Loss History')
plt.grid(True, alpha=0.3)

# Plot 2: Results by integer
plt.subplot(2, 3, 2)
losses = [cross_entropy(QNN.forward(np.asarray(r['input']), weights_opt), TARGETS[i]) for i, r in enumerate(results)]
colors = ['green' if r['correct'] else 'red' for r in results]
plt.bar(range(len(results)), losses, color=colors, alpha=0.7)
plt.xlabel('Integer')
plt.ylabel('Cross-Entropy Loss')
plt.title('Results by Integer (Green=Correct, Red=Wrong)')
plt.xticks(range(len(results)), integers)
plt.grid(True, alpha=0.3)

# Plot 3: Confidence scores
plt.subplot(2, 3, 3)
confidences = [r['confidence'] for r in results]
plt.bar(range(len(results)), confidences, color=colors, alpha=0.7)
plt.xlabel('Integer')
plt.ylabel('Prediction Confidence')
plt.title('Prediction Confidence by Integer')
plt.xticks(range(len(results)), integers)
plt.grid(True, alpha=0.3)

# Plot 4: Accuracy summary
plt.subplot(2, 3, 4)
plt.bar(['Overall'], [acc], color='blue', alpha=0.7)
plt.ylabel('Accuracy')
plt.title('Overall Accuracy')
plt.ylim(0, 1)
plt.text(0, acc + 0.01, f'{acc:.1%}', ha='center', va='bottom', fontweight='bold')
plt.grid(True, alpha=0.3)

# Plot 5: Predicted vs Actual Factors
plt.subplot(2, 3, 5)
predicted_factors = [r['predicted'] for r in results]
actual_factors = [r['target'] for r in results]
plt.scatter(actual_factors, predicted_factors, color='blue', alpha=0.7, s=60)

# Perfect prediction line
min_factor = min(min(actual_factors), min(predicted_factors))
max_factor = max(max(actual_factors), max(predicted_factors))
plt.plot([min_factor, max_factor], [min_factor, max_factor], 'k--', alpha=0.5, label='Perfect')

plt.xlabel('Actual Lowest Prime Factor')
plt.ylabel('Predicted Lowest Prime Factor')
plt.title('Predicted vs Actual Factors')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 6: Factor Distribution
plt.subplot(2, 3, 6)
width = 0.35
x_pos = np.arange(len(integers))

plt.bar(x_pos - width/2, actual_factors, width, label='Actual', alpha=0.7, color='lightblue')
plt.bar(x_pos + width/2, predicted_factors, width, label='Predicted', alpha=0.7, color='lightcoral')

plt.xlabel('Integer')
plt.ylabel('Lowest Prime Factor')
plt.title('Actual vs Predicted Factors by Integer')
plt.xticks(x_pos, integers)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("plots/babyshorai.png", dpi=150, bbox_inches='tight')
plt.show()

print("Circuit saved to plots/babyshoraicircuit.png")
print("Results plot saved to plots/babyshorai.png")

# The trained weights could be saved if desired 