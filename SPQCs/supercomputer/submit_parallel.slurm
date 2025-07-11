#!/bin/bash
#SBATCH --job-name=spqc_parallel
#SBATCH --output=spqc_%j.out
#SBATCH --error=spqc_%j.err

# === RESOURCE CONFIGURATION - MODIFY HERE ===
#SBATCH --nodes=1                    # Number of compute nodes
#SBATCH --ntasks-per-node=1          # MPI tasks per node  
#SBATCH --cpus-per-task=64           # CPU cores per task (optimal for parameter parallelization)
#SBATCH --mem=150GB                  # Memory per node (150GB sufficient for CPU training)
#SBATCH --time=03:00:00              # Time limit (3 hours for reliable CPU training)

# === GPU CONFIGURATION (experimental - uncomment to try GPU acceleration) ===
# #SBATCH --partition=marylou13h     # GPU partition
# #SBATCH --gres=gpu:H200:4          # 4 H200 GPUs per node

# === CPU PARTITION (recommended for stability) ===
#SBATCH --partition=marylou12        # CPU-only partition

# Load modules
module load python/3.9
# module load cuda/11.8              # Uncomment for GPU support

# Set threading for efficiency
export OMP_NUM_THREADS=64
# export CUDA_VISIBLE_DEVICES=0,1,2,3  # Uncomment for GPU support

# Set Python path to find parent modules
export PYTHONPATH=$PYTHONPATH:$SLURM_SUBMIT_DIR/../

# Install required packages
pip install --user joblib qiskit-aer
# pip install --user qiskit-aer-gpu  # Uncomment for GPU support

# Change to supercomputer directory
cd $SLURM_SUBMIT_DIR

# Run training
echo "Starting parallel SPQC training with 5000 epochs..."
echo "Expected time: 1-3 hours (vs days on laptop)"
echo "Using $(nproc) CPU cores for maximum parallelization"
python star_train_parallel.py
echo "Training complete!" 