#!/bin/bash
#

# Number of perlmutter nodes for job runs + 1 node for management.
#SBATCH -N 4

# Change to your account
# Also change in the srun command below
#SBATCH -A m2113_g
#
# Job naming stuff
#SBATCH -J mace-matensemble
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#
#SBATCH -t 00:30:00
#
# Requested queue
#SBATCH -C gpu
#SBATCH -q debug
#
# GPU stuff
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# suppress benign shutdown logs
export FLUX_BROKER_LOG_LEVEL=warning

#---------------------- Define a cleanup function to write out job failure or successful completion --------------------------------------------

# define a cleanup function
cleanup() {
  rc=$?
  if [ $rc -eq 0 ]; then
    echo "$(date): SUCCESS" > job.${SLURM_JOB_ID}.done
  else
    echo "$(date): FAILURE (exit code $rc)" > job.${SLURM_JOB_ID}.fail
  fi
  # optionally notify via curl, mailx, etc.
}

# ensure cleanup runs on any exit
trap cleanup EXIT

#---------------------- SETUP FOR Flux + matensemble + LAMMPs IN PERLMUTTER -------------------------------------------------------------------
# Load Spack-compatible Python and required modules
module load python/3.11

# Load cudatoolkit
module load PrgEnv-gnu/8.5.0
module load cudatoolkit/12.4
module load craype-accel-nvidia80

# Activate the spack
source /global/cfs/cdirs/m5014/spack_py3.11/spack/share/spack/setup-env.sh
spack load flux-sched
unset LUA_PATH LUA_CPATH

# Set the variables
CONDA_ENV="/global/homes/r/rym/.conda/envs/lammps"
CONDA_SITE="$CONDA_ENV/lib/python3.11/site-packages"
CONDA_PYTHON="$CONDA_ENV/bin/python"

# Load the conda environment
conda activate $CONDA_ENV

# Add conda environment to PYTHONPATH before the Spack path
export PYTHONPATH="${CONDA_SITE}:${PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Execute the workflow
srun flux start lammps_matensemble --run_directory run_directory \
	                           --inputs_directory inputs_directory \
				   --ffield model.model \
				   --structure POSCAR \
				   --control None \
				   --in_lammps in.mace.json \
				   --lammps_task ase_mace.py \
				   --atoms_per_task 1000 \
				   --lammps_task_order ffield in_lammps structure \
				   --parent_levels 0 \
				   --gpus_per_task 1 \
				   --cpus_per_task 16 \
				   --atom_style atomic
