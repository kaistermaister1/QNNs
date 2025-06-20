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
from typing import List

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

def batch_loss(w: np.ndarray) -> float:
    loss = 0.0
    for x_bits, y_onehot in zip(INPUTS, TARGETS):
        probs = QNN.forward(np.asarray(x_bits), w)
        loss += cross_entropy(probs, y_onehot)
    return loss / len(INPUTS)

# ─────────────────────  Training loop  ───────────────────────────────
print("Training …")
start = time.time()

# Both optimizers use the same minimize API
res = optimiser.minimize(batch_loss, weights)
weights_opt = res.x

elapsed = time.time() - start
print(f"Done in {elapsed:.2f} s  |  Final loss {batch_loss(weights_opt):.4f}\n")

# ─────────────────────  Evaluation  ──────────────────────────────────
correct = 0
for n, x_bits, y in zip(range(2, 16), INPUTS, LABELS):
    probs = QNN.forward(np.asarray(x_bits), weights_opt)
    pred_factor = int(np.argmax(probs))
    co_factor = n // pred_factor if pred_factor > 0 else None
    is_correct = pred_factor == y
    correct += is_correct
    print(f"n={n:2d}  pred={pred_factor:2d}  target={y:2d}  » factors: ({pred_factor},{co_factor})  {'✔' if is_correct else '✘'}")

acc = correct / len(INPUTS)
print(f"\nAccuracy: {acc:.1%} ({correct}/{len(INPUTS)})")

# The trained weights could be saved if desired 