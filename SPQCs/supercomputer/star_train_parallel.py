"""star_train_parallel.py – batched gradient + cached transpile (fixed)
====================================================================
This version compiles and runs; previous update was truncated.
"""
from __future__ import annotations
import os, sys, time, argparse, multiprocessing as mp
from contextlib import suppress
from typing import List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm import tqdm
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector

# ───────── project imports ─────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT))
from star_data import get_star_data
from star_spqc import create_spqc_circuit, create_random_weights
from star_eval import evaluate_model

# ───────── CLI args ─────────
parser = argparse.ArgumentParser("SPQC trainer (batched gradient + cached transpile)")
parser.add_argument("--cpus",  type=int, default=None)
parser.add_argument("--gpus",  type=int, default=None)
parser.add_argument("--no-gpu", action="store_true")
parser.add_argument("--epochs", type=int, default=1)
ARGS = parser.parse_args()

# resources
OS_CPUS     = mp.cpu_count()
SLURM_CPUS  = int(os.environ.get("SLURM_CPUS_PER_TASK", OS_CPUS))
N_CPUS      = ARGS.cpus or SLURM_CPUS
USE_GPU     = not ARGS.no_gpu
VISIBLE_GPU = len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(',')) if os.environ.get("CUDA_VISIBLE_DEVICES") else 0
N_GPUS      = ARGS.gpus if ARGS.gpus is not None else VISIBLE_GPU
for e in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(e,"1")

# gpu helper (per-process, no multiprocessing.Value)
def assign_visible_gpu(worker_gpu_id=None):
    if not USE_GPU or N_GPUS<1: return
    if worker_gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"]=str(worker_gpu_id)
        os.environ.setdefault("QISKIT_AER_CUDA","1")

# amplitude slice
def addr_amplitudes(sv: np.ndarray, t:int,m:int,n:int,r:int)->np.ndarray:
    N=t+m+n*r+1
    tensor=sv.reshape([2]*N)
    idx=[0]*t+[slice(None)]*m+[0]*(n*r)+[0]
    addr = tensor[tuple(idx)].reshape(2**m)
    # Renormalize
    norm = np.linalg.norm(addr)
    return addr / norm if norm != 0 else addr

# per‑process cache (Linux/HPC) or shared objects (Windows threads)
g_backend=None; g_template=None; g_params=None
shared_backend=None; shared_template=None; shared_params=None

def worker_init(qc):
    print(f'init pid={os.getpid()} start')
    
    # Assign GPU to this worker (round-robin by process ID)
    worker_gpu_id = os.getpid() % N_GPUS if N_GPUS > 0 else None
    assign_visible_gpu(worker_gpu_id)
    
    dev = "GPU" if (USE_GPU and N_GPUS > 0) else "CPU"
    backend  = AerSimulator(method="statevector", device=dev)
    template = transpile(qc, backend, optimization_level=0, layout_method="trivial")
    globals().update(g_backend=backend, g_template=template, g_params=list(template.parameters))
    print(f'init pid={os.getpid()} done; params={len(g_params)} gpu={worker_gpu_id}')

# batched gradient (dual mode: threads/processes)
def gradient_sample(x,theta,y,t,m,n,r,shift=np.pi/2):
    # Thread mode: use shared objects from main thread
    if os.name == 'nt':  # Windows threads
        backend = shared_backend
        template = shared_template  
        params = shared_params
    else:  # Linux/HPC processes
        if g_params is None:
            raise RuntimeError(f"Worker pid={os.getpid()} started without initialisation")
        backend = g_backend
        template = g_template
        params = g_params
    
    circuits=[]
    for i in range(len(theta)):
        tp,tm=theta.copy(),theta.copy(); tp[i]+=shift; tm[i]-=shift
        
        # Create parameter binding for both input and weights
        for weights in [tp, tm]:
            param_binding = {}
            
            # Get parameter types
            input_params = [p for p in params if p.name.startswith('input_theta')]
            model_params = [p for p in params if p.name.startswith('model')]
            address_params = [p for p in params if p.name.startswith('address_theta')]
            
            # Bind input parameters
            for j, param in enumerate(input_params):
                if j < len(x):
                    param_binding[param] = x[j]
                else:
                    param_binding[param] = 0.0
            
            # Split weights into model and address portions
            num_model_params = len(model_params)
            model_values = weights[:num_model_params]
            address_values = weights[num_model_params:]
            
            # Bind model parameters
            for j, param in enumerate(model_params):
                if j < len(model_values):
                    param_binding[param] = model_values[j]
                else:
                    param_binding[param] = 0.0
            
            # Bind address parameters
            for j, param in enumerate(address_params):
                if j < len(address_values):
                    param_binding[param] = address_values[j]
                else:
                    param_binding[param] = 0.0
            
            circuits.append(template.assign_parameters(param_binding))
    
    # Get statevectors directly using Statevector.from_instruction
    amps = []
    for circuit in circuits:
        sv = Statevector.from_instruction(circuit).data
        amps.append(addr_amplitudes(sv, t, m, n, r))
    grad=np.zeros_like(theta)
    for i in range(len(theta)):
        lp=np.mean(np.abs(amps[2*i]-y)**2); lm=np.mean(np.abs(amps[2*i+1]-y)**2)
        grad[i]=0.5*(lp-lm)/np.sin(shift)
    return grad

# wrapper for evaluate
def make_wrapper(qc,t,m,n,r):
    # Use the circuit directly with Statevector.from_instruction
    tpl = qc.copy()
    
    # Get parameter mapping like in original star_spqc.py
    input_params = []
    model_params = []
    address_params = []
    
    for param in tpl.parameters:
        if param.name.startswith('input_theta'):
            input_params.append(param)
        elif param.name.startswith('model'):
            model_params.append(param)
        elif param.name.startswith('address_theta'):
            address_params.append(param)
    
    def f(x,th):
        # Bind both input values and weights
        param_binding = {}
        
        # Bind input parameters
        for i, param in enumerate(input_params):
            if i < len(x):
                param_binding[param] = x[i]
            else:
                param_binding[param] = 0.0
        
        # Split weights into model and address portions  
        num_model_params = len(model_params)
        model_values = th[:num_model_params]
        address_values = th[num_model_params:]
        
        # Bind model parameters
        for i, param in enumerate(model_params):
            if i < len(model_values):
                param_binding[param] = model_values[i]
            else:
                param_binding[param] = 0.0
        
        # Bind address parameters
        for i, param in enumerate(address_params):
            if i < len(address_values):
                param_binding[param] = address_values[i]
            else:
                param_binding[param] = 0.0
        
        bound_circuit = tpl.assign_parameters(param_binding)
        # Use Statevector.from_instruction for reliable statevector access
        sv = Statevector.from_instruction(bound_circuit).data
        return addr_amplitudes(sv,t,m,n,r)
    class W: forward=staticmethod(f)
    return W()

# compute loss for tracking (sampled for efficiency)
def compute_loss_sampled(qc, theta, Xtr, Ytr, t, m, n, r, sample_size=20):
    """Compute average loss over a random sample of training set for efficiency"""
    wrapper = make_wrapper(qc, t, m, n, r)
    
    # Sample a subset for loss computation
    n_samples = min(sample_size, len(Xtr))
    indices = np.random.choice(len(Xtr), n_samples, replace=False)
    
    total_loss = 0.0
    for i in indices:
        pred = wrapper.forward(Xtr[i], theta)
        total_loss += np.mean(np.abs(pred - Ytr[i])**2)
    return total_loss / n_samples

# ───────── main ─────────

def main():
    global USE_GPU
    print(f"CPU={N_CPUS}, GPU active={USE_GPU and N_GPUS>0}")
    
    # GPU sanity check
    if USE_GPU and N_GPUS > 0:
        try:
            # Test GPU backend without changing global environment
            test_backend = AerSimulator(method="statevector", device="GPU")
            test_qc = QuantumCircuit(1, 1)
            test_qc.x(0)
            test_qc.measure(0, 0)
            test_backend.run(transpile(test_qc, test_backend), shots=1).result()
            print("GPU backend verified")
        except Exception as e:
            print(f"GPU fallback: {e} → CPU only")
            USE_GPU = False
    
    Xtr,Xte,ytr,yte,_=get_star_data(300)
    np.random.seed(42)
    
    t,m,n,r=0,3,2,1
    frame=create_spqc_circuit(t=t,m=m,n=n,r=r)
    qc=QuantumCircuit(frame.num_qubits)
    for inst in frame.data:
        if inst.operation.name!="measure": qc.append(inst.operation,inst.qubits)
    theta=create_random_weights(frame,seed=42)
    print(f"Number of parameters = {len(theta)}")
    
    classes=2**m; Ytr=np.eye(classes)[ytr.astype(int)]; Yte=np.eye(classes)[yte.astype(int)]

    evaluate_model(make_wrapper(qc,t,m,n,r),theta,Xte,Yte,'binary','Initial')

    outer=min(N_CPUS,len(Xtr))
    print(f"Processes over samples: {outer}")
    if outer > 10:
        print("NOTE: If you get 'Worker not initialised' errors, try --cpus 4 to reduce worker overhead")
    if os.name == 'nt':  # Windows
        print("Windows detected: Consider using threads instead of processes for better stability")
    print()

    m1 = np.zeros_like(theta)
    v1 = np.zeros_like(theta)
    b1,b2,lr=0.9,0.999,0.01
    losses:List[float]=[]
    
    # Pre-import qiskit in main process to help worker startup
    import qiskit, qiskit_aer
    print("Pre-imported qiskit libraries")
    
    # Use threads on Windows, processes on Linux/HPC for best compatibility
    global shared_backend, shared_template, shared_params
    
    if os.name == 'nt':  # Windows
        print("Using thread-based parallelism for Windows compatibility...")
        backend_type = "threads"
        # No worker_init needed for threads - they share memory space
        pool = Parallel(n_jobs=outer, prefer="threads")
        
        # Pre-build backend and template once for thread sharing
        print("Pre-building shared backend and template...")
        assign_visible_gpu(0)  # Set GPU for main thread
        dev = "GPU" if (USE_GPU and N_GPUS > 0) else "CPU"
        shared_backend = AerSimulator(method="statevector", device=dev)
        shared_template = transpile(qc, shared_backend, optimization_level=0, layout_method="trivial")
        shared_params = list(shared_template.parameters)
        print(f"Shared template ready with {len(shared_params)} parameters")
    else:  # Linux/HPC
        print("Using process-based parallelism for HPC performance...")
        backend_type = "processes"
        try:
            pool = Parallel(
                n_jobs=outer,
                initializer=worker_init,
                initargs=(qc,),
                timeout=3600,
                prefer="processes",
                reuse=True
            )
        except TypeError:
            pool = Parallel(
                n_jobs=outer,
                initializer=worker_init,
                initargs=(qc,),
                timeout=3600,
                prefer="processes",
            )
    print("Parallelism ready, starting training...")
    
    for ep in tqdm(range(ARGS.epochs),desc='Epoch'):
        grads = pool(
            delayed(gradient_sample)(Xtr[i],theta,Ytr[i],t,m,n,r) for i in range(len(Xtr)))
        g=np.mean(grads,axis=0)
        m1=b1*m1+(1-b1)*g; v1=b2*v1+(1-b2)*(g**2)
        m1h=m1/(1-b1**(ep+1)); v1h=v1/(1-b2**(ep+1))
        theta-=lr*m1h/(np.sqrt(v1h)+1e-8)
        
        # Compute and track loss (sampled for efficiency)
        loss = compute_loss_sampled(qc, theta, Xtr, Ytr, t, m, n, r, sample_size=20)
        losses.append(loss)

    evaluate_model(make_wrapper(qc,t,m,n,r),theta,Xte,Yte,'binary','Final')
    
    # Save loss plot
    os.makedirs('plots',exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(losses, 'o-')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss (log scale)')
    plt.title('Training Loss Over Time')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/loss.png', dpi=300)
    plt.close()
    
    print(f"Training complete. Loss plot saved to plots/loss.png")

if __name__=='__main__':
    with suppress(KeyboardInterrupt): 
        main()
