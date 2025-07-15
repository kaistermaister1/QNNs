"""Simple SPQC Supercomputer Training - Maximum Parallelism"""
import os
# Pin BLAS/OpenMP threads to 1 per process before NumPy loads
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import sys
import numpy as np
from joblib import Parallel, delayed
from qiskit import QuantumCircuit
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt

# Project imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from star_data import get_star_data
from star_spqc import create_spqc_circuit, create_random_weights, bind_params, model as cpu_model
from star_eval import evaluate_model, wedge_onehot

# Configuration
CLASSIFICATION_MODE = 'binary'        # or 'multiclass' if using one-hot
RANDOM_SEED = 42
NUM_DATA = 300
EPOCHS = 1
USE_GPU = False                       # enable GPU path if tested
N_GPUS = 4                           # for manual GPU assignment

# Resource detection
import multiprocessing
CPU_COUNT = multiprocessing.cpu_count()
# Use all available or override via env var
N_CPU_WORKERS = int(os.environ.get('SLURM_CPUS_PER_TASK', CPU_COUNT))

# Utility: sample-level gradient
def sample_gradient(qc, x, theta, y_true, t, m, n, r, shift=np.pi/2):
    """Compute full parameter-shift gradient for one sample."""
    grad = np.zeros_like(theta)
    for i in range(len(theta)):
        theta_p, theta_m = theta.copy(), theta.copy()
        theta_p[i] += shift
        theta_m[i] -= shift
        # Two forward evaluations
        amp_p = cpu_model(qc, x, theta_p, t, m, n, r)
        amp_m = cpu_model(qc, x, theta_m, t, m, n, r)
        loss_p = np.mean(np.abs(amp_p - y_true)**2)
        loss_m = np.mean(np.abs(amp_m - y_true)**2)
        grad[i] = 0.5 * (loss_p - loss_m) / np.sin(shift)
    return grad

# Efficient mesh prediction
def mesh_predictions(spqc_model, theta, mesh_pts, n_jobs):
    def process_batch(batch):
        batch_results = []
        for p in batch:
            amplitudes = spqc_model.forward(p, theta)
            probabilities = np.abs(amplitudes)**2  # Convert complex amplitudes to real probabilities
            batch_results.append(probabilities)
        return batch_results
    
    batch_size = max(1, len(mesh_pts) // n_jobs)
    batches = [mesh_pts[i:i+batch_size] for i in range(0, len(mesh_pts), batch_size)]
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_batch)(batch) for batch in batches
    )
    # flatten
    flattened = []
    for batch_result in results:
        flattened.extend(batch_result)
    return np.array(flattened)

# Visualization wrapper
def visualize_decision_boundary(spqc_model, theta, m, X, Y_onehot, mode, boundary, cpu_jobs, save_path=None):
    # generate grid
    xx, yy = np.meshgrid(np.linspace(0,1,200), np.linspace(0,1,200))
    pts = np.column_stack((xx.ravel(), yy.ravel()))
    preds = mesh_predictions(spqc_model, theta, pts, n_jobs=cpu_jobs)
    plt.figure(figsize=(8,8))
    if mode=='binary':
        prob = preds[:,1] / (preds.sum(axis=1)+1e-8)
        Z = prob.reshape(xx.shape)
        plt.contourf(xx, yy, Z, levels=20, cmap='RdYlBu_r')
        plt.contour(xx, yy, Z, levels=[0.5], colors='white')
        if boundary:
            V = boundary.vertices
            plt.plot(V[:,0], V[:,1], 'k--')
        labels = np.argmax(Y_onehot, axis=1)
        plt.scatter(X[:,0], X[:,1], c=labels, edgecolors='k')
    else:
        Z = preds.argmax(axis=1).reshape(xx.shape)
        cmap = plt.get_cmap('tab10', 2**m)
        plt.contourf(xx, yy, Z, levels=np.arange(-0.5,2**m,1), cmap=cmap, alpha=0.6)
        plt.colorbar(ticks=range(2**m))
        labels = np.argmax(Y_onehot, axis=1)
        plt.scatter(X[:,0], X[:,1], c=labels, cmap=cmap, edgecolors='k')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()

# Wrapper class for evaluate_model
class SPQCModelWrapper:
    def __init__(self, qc, t, m, n, r):
        self.qc, self.t, self.m, self.n, self.r = qc, t, m, n, r
    def forward(self, x, theta):
        return cpu_model(self.qc, x, theta, self.t, self.m, self.n, self.r)

# Main
if __name__=='__main__':
    print(f"Using {N_CPU_WORKERS} CPU workers (out of {CPU_COUNT})")
    # load
    X_train, X_test, y_train, y_test, boundary = get_star_data(NUM_DATA)
    np.random.seed(RANDOM_SEED)
    # build QC
    t,m,n,r = 0,3,2,1
    frame = create_spqc_circuit(t=t,m=m,n=n,r=r)
    qc = QuantumCircuit(frame.num_qubits)
    for inst in frame.data:
        if inst.operation.name!='measure': qc.append(inst.operation, inst.qubits)
    theta = create_random_weights(frame, seed=RANDOM_SEED)
    # hot
    num_classes = 2**m
    Y_train = np.eye(num_classes)[y_train.astype(int)]
    Y_test  = np.eye(num_classes)[y_test.astype(int)]
    model_wrap = SPQCModelWrapper(qc,t,m,n,r)
    # Create plots directory
    os.makedirs('plots', exist_ok=True)
    
    # initial eval
    evaluate_model(model_wrap, theta, X_test, Y_test, CLASSIFICATION_MODE, 'Initial')
    
    # Initial boundary visualization
    visualize_decision_boundary(model_wrap, theta, m, X_test, Y_test, CLASSIFICATION_MODE,
                                boundary, cpu_jobs=min(N_CPU_WORKERS,len(X_test)), save_path='plots/initial_boundary.png')
    
    # training
    m1 = np.zeros_like(theta); v1 = np.zeros_like(theta)
    b1,b2,lr = 0.9,0.999,0.01
    losses=[]
    for ep in tqdm(range(1,EPOCHS+1), desc='Epoch'):
        # compute gradients per sample
        jobs = min(N_CPU_WORKERS, len(X_train))
        grads = Parallel(n_jobs=jobs)(delayed(sample_gradient)(qc,x,theta,Y_train[i],t,m,n,r)
                                      for i,x in enumerate(X_train))
        g = np.mean(grads,axis=0)
        # adam
        m1 = b1*m1 + (1-b1)*g
        v1 = b2*v1 + (1-b2)*(g**2)
        m_hat = m1/(1-b1**ep)
        v_hat = v1/(1-b2**ep)
        theta -= lr*m_hat/(np.sqrt(v_hat)+1e-8)
        # loss
        losses.append(np.mean([np.mean(np.abs(cpu_model(qc,x,theta,t,m,n,r)-Y_train[i])**2)
                                for i,x in enumerate(X_train)]))
    # final eval
    evaluate_model(model_wrap, theta, X_test, Y_test, CLASSIFICATION_MODE, 'Final')
    # plot loss
    plt.figure(); plt.plot(losses); plt.yscale('log'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.savefig('plots/loss.png')
    # Final boundary visualization
    visualize_decision_boundary(model_wrap, theta, m, X_test, Y_test, CLASSIFICATION_MODE,
                                boundary, cpu_jobs=min(N_CPU_WORKERS,len(X_test)), save_path='plots/final_boundary.png')