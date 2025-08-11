#!/bin/bash
#
# Change to your account
# Also change in the srun command below
#SBATCH -A m2113_g
#
# Job naming stuff
#SBATCH -J vasp
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#
# Requested time
#SBATCH -t 00:30:00
#
# Requested queue
#SBATCH -C gpu
#SBATCH -q debug
#
# Number of perlmutter nodes to use.
# Set the same value in the SBATCH line and NNODES
#SBATCH -N 1

#SBATCH --exclusive
NNODES=1

GPUS_PER_NODE=4
GPUS_PER_TASK=1
NGPUS=$(($GPUS_PER_NODE * $NNODES))

CPUS_PER_NODE=64
CPUS_PER_TASK=16
NCPUS=$(($CPUS_PER_NODE * $NNODES))

NUM_TASKS=$(($NGPUS / $GPUS_PER_TASK))

export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

module load vasp/6.4.3-gpu

VASP_BINARY=vasp_std

# Run command. Modify to where ever you placed the binary and input files
srun -A m2113_g --ntasks=$NUM_TASKS --cpus-per-task $CPUS_PER_TASK --cpu-bind=cores --gpu-bind=none --gpus=$NGPUS $VASP_BINARY 
