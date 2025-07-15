"""star_train_parallel.py
===========================================================
SPQC State‑vector Training with Two‑Level Parallelism
----------------------------------------------------
* **Outer level (processes)** – parallel over **training samples**.
* **Inner level (threads)**  – parallel over **parameters** inside each
  sample’s parameter‑shift gradient.
* **OMP threads per BLAS kernel** are pinned to **1** to avoid oversub‐
  scription.
* **GPU optional** – if `--gpus` (or CUDA_VISIBLE_DEVICES) exposes at least
  one GPU *and* qiskit‑aer is built with CUDA, each outer process grabs a
  GPU in round‑robin.  Otherwise the script falls back to pure CPU.

Usage examples
--------------
```bash
# use all visible CPU cores, try GPU if installed
python star_train_parallel.py

# force CPU‑only with 32 workers
python star_train_parallel.py --cpus 32 --no-gpu

# on a 4‑GPU node: one GPU per 16 cores, 2 epochs
python star_train_parallel.py --cpus 64 --gpus 4 --epochs 2
```
Requirements
------------
Python 3.8+, numpy, joblib, tqdm, matplotlib, qiskit >=0.46, qiskit‑aer >=0.13
(with CUDA support for GPU mode).
"""
from __future__ import annotations
import os, sys, time, argparse, multiprocessing as mp
from contextlib import suppress
from typing import List

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non‑interactive backend for clusters
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm import tqdm
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# ───────── project‑local imports ─────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT))
from star_data import get_star_data
from star_spqc import create_spqc_circuit, create_random_weights, model as spqc_model
from star_eval import evaluate_model

# ───────── CLI args ─────────
parser = argparse.ArgumentParser("SPQC supercomputer trainer")
parser.add_argument("--cpus",  type=int, default=None, help="CPU workers (default: SLURM_CPUS_PER_TASK or all cores)")
parser.add_argument("--gpus",  type=int, default=None, help="GPUs to use round‑robin (default: all visible CUDA devices)")
parser.add_argument("--no-gpu", action="store_true", help="Disable GPU path entirely")
parser.add_argument("--epochs", type=int, default=1,   help="Training epochs (default 1)")
ARGS = parser.parse_args()

# ───────── resource detection ─────────
OS_CPUS   = mp.cpu_count()
SLURM_CPUS= int(os.environ.get("SLURM_CPUS_PER_TASK", OS_CPUS))
N_CPUS    = ARGS.cpus or SLURM_CPUS

USE_GPU   = not ARGS.no_gpu
ENV_CUDA  = os.environ.get("CUDA_VISIBLE_DEVICES")
VISIBLE_GPU_COUNT = len(ENV_CUDA.split(',')) if ENV_CUDA else 0
N_GPUS    = ARGS.gpus if ARGS.gpus is not None else VISIBLE_GPU_COUNT

# pin BLAS/OpenMP threads to 1 to prevent oversubscription
for env in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(env, "1")

# ───────── GPU assignment helper ─────────
_gpu_ctr = mp.Value('i', 0)

def assign_visible_gpu():
    if not USE_GPU or N_GPUS < 1:
        return
    with _gpu_ctr.get_lock():
        idx = _gpu_ctr.value % N_GPUS
        _gpu_ctr.value += 1
    os.environ["CUDA_VISIBLE_DEVICES"] = str(idx)
    os.environ.setdefault("QISKIT_AER_CUDA", "1")

# ───────── gradient per sample (inner parallel) ─────────

def sample_gradient(
    qc: QuantumCircuit,
    x: np.ndarray,
    theta: np.ndarray,
    y_true: np.ndarray,
    t:int, m:int, n:int, r:int,
    shift: float = np.pi/2,
    param_jobs: int = 1
) -> np.ndarray:
    grad = np.zeros_like(theta)
    def grad_i(i:int) -> float:
        tp, tm = theta.copy(), theta.copy()
        tp[i]+=shift; tm[i]-=shift
        amp_p = spqc_model(qc,x,tp,t,m,n,r)
        amp_m = spqc_model(qc,x,tm,t,m,n,r)
        lp = np.mean(np.abs(amp_p - y_true)**2)
        lm = np.mean(np.abs(amp_m - y_true)**2)
        return 0.5*(lp-lm)/np.sin(shift)
    if param_jobs>1:
        grad[:] = Parallel(n_jobs=param_jobs, prefer="threads")(delayed(grad_i)(i) for i in range(len(theta)))
    else:
        for i in range(len(theta)):
            grad[i]=grad_i(i)
    return grad

# ───────── wrapper for evaluate_model (expects .forward) ─────────
class ModelWrap:
    def __init__(self,qc,t,m,n,r):
        self.qc,self.t,self.m,self.n,self.r = qc,t,m,n,r
    def forward(self,x,th):
        return spqc_model(self.qc,x,th,self.t,self.m,self.n,self.r)

# ───────── main training loop ─────────

def main():
    global USE_GPU
    print(f"CPU workers: {N_CPUS} of {OS_CPUS}")
    print(f"GPU enabled: {USE_GPU}, visible GPUs: {VISIBLE_GPU_COUNT}, to use: {N_GPUS}")

    # disable GPU if none requested / available
    if USE_GPU and N_GPUS<1:
        print("[GPU fallback] no GPUs available → CPU only")
        USE_GPU=False

    # quick GPU sanity test
    if USE_GPU:
        try:
            assign_visible_gpu()
            sim = AerSimulator(method="statevector", device="GPU")
            qc_test = QuantumCircuit(1,1); qc_test.x(0); qc_test.measure(0,0)
            sim.run(transpile(qc_test, sim), shots=1).result()
            print("GPU backend initialised OK")
        except Exception as e:
            print(f"[GPU fallback] {e} → CPU only")
            USE_GPU=False

    # dataset
    X_train,X_test,y_train,y_test,_ = get_star_data(300)
    np.random.seed(42)

    # SPQC circuit + params
    t,m,n,r = 0,3,2,1
    frame = create_spqc_circuit(t=t,m=m,n=n,r=r)
    qc = QuantumCircuit(frame.num_qubits)
    for inst in frame.data:
        if inst.operation.name!="measure": qc.append(inst.operation,inst.qubits)
    theta = create_random_weights(frame,seed=42)
    print(f"|theta| = {len(theta)}\n")

    # one‑hot labels
    num_classes=2**m
    Y_train = np.eye(num_classes)[y_train.astype(int)]
    Y_test  = np.eye(num_classes)[y_test.astype(int)]

    # evaluate before training
    evaluate_model(ModelWrap(qc,t,m,n,r), theta, X_test, Y_test, 'binary', 'Initial')

    # Adam accumulators
    m1 = np.zeros_like(theta); v1 = np.zeros_like(theta)
    b1,b2,lr = 0.9,0.999,0.01

    # decide parallel layout
    outer = min(N_CPUS, len(X_train))  # processes over samples
    inner = max(1, N_CPUS//outer)      # threads over parameters
    print(f"Parallel layout → outer={outer} processes  inner={inner} threads\n")

    losses: List[float] = []
    for ep in tqdm(range(ARGS.epochs), desc='Epoch'):
        # make sure GPU env is set inside new pools
        assign_visible_gpu()

        grads = Parallel(n_jobs=outer, initializer=assign_visible_gpu)(
            delayed(sample_gradient)(qc, X_train[i], theta, Y_train[i], t,m,n,r, param_jobs=inner)
            for i in range(len(X_train)))

        g = np.mean(grads, axis=0)
        m1 = b1*m1 + (1-b1)*g; v1 = b2*v1 + (1-b2)*(g**2)
        m1h = m1/(1-b1**(ep+1)); v1h = v1/(1-b2**(ep+1))
        theta -= lr*m1h/(np.sqrt(v1h)+1e-8)

        # track loss (quick serial loop)
        losses.append(np.mean([
            np.mean(np.abs(spqc_model(qc,x,theta,t,m,n,r) - Y_train[i])**2)
            for i,x in enumerate(X_train)]))

    # evaluate after training
    evaluate_model(ModelWrap(qc,t,m,n,r), theta, X_test, Y_test, 'binary', 'Final')

    # save loss curve
    os.makedirs('plots', exist_ok=True)
    plt.figure(); plt.plot(losses,'o-'); plt.yscale('log');
    plt.xlabel('Epoch'); plt.ylabel('MSE Loss (log)'); plt.tight_layout()
    plt.savefig('plots/loss.png', dpi=300); plt.close()

if __name__ == '__main__':
    with suppress(KeyboardInterrupt):
        main()
