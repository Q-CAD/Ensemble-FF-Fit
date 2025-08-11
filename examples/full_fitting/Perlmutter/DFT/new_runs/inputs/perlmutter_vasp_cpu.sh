#!/bin/bash
#
# Change to your account
# Also change in the srun command below
#SBATCH -A {ALLOCATION}
#
# Job naming stuff
#SBATCH -J {JOB_NAME}
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#
# Requested time
#SBATCH -t {TIME}
#
# Requested queue
#SBATCH -q {PARTITION}
#SBATCH -C cpu
#
# Number of perlmutter nodes to use.
# Set the same value in the SBATCH line and NNODES
#SBATCH -N {NODES}

NNODES={NODES}

GPUS_PER_NODE=0
GPUS_PER_TASK=0
NGPUS=$(($GPUS_PER_NODE * $NNODES))

CPUS_PER_NODE={CPUS_PER_NODE}
CPUS_PER_TASK={CPUS_PER_TASK}
NCPUS=$(($CPUS_PER_NODE * $NNODES))

NUM_TASKS=$(($NCPUS / $CPUS_PER_TASK))

# Always provide OpenMP settings when running VASP 6
export OMP_NUM_THREADS=2
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

module load vasp/6.4.3-cpu

VASP_BINARY={VASP_EXECUTABLE}

# Run command. Modify to where ever you placed the binary and input files

srun -A {ALLOCATION} --ntasks=$NUM_TASKS --cpus-per-task=$CPUS_PER_TASK --cpu-bind=cores $VASP_BINARY 
