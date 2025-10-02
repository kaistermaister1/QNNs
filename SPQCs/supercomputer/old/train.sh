#!/bin/bash --login

#SBATCH --time=02:00:00   # walltime
#SBATCH --ntasks=5000   # number of processor cores (i.e. tasks)
#SBATCH --mem-per-cpu=512M   # memory per CPU core
#SBATCH -J "first_spqc"   # job name


# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
mamba activate spqc_parallel
cd supercomputer/
python star_train_custom.py --epochs 1000 --cpus 5000 --visualize-boundary