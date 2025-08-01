#!/bin/bash
#

# Number of perlmutter nodes for job runs + 1 node for management.
#SBATCH -N 3

# Change to your account
# Also change in the srun command below
#SBATCH -A m526
#
# Job naming stuff
#SBATCH -J LAMMPS_CPU
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#
# Requested time
#SBATCH -t 00:30:00
#
# Requested queue
#SBATCH -C cpu
#SBATCH -q debug

export OMP_NUM_THREADS=64
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

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
# Load cudatoolkit
module load PrgEnv-gnu/8.5.0
module load cudatoolkit/12.4
module load craype-accel-nvidia80

# Activate the spack
source /global/cfs/cdirs/m526/sbagchi/spack/share/spack/setup-env.sh
spack env activate -p spack_matensemble_env
unset LUA_PATH LUA_CPATH

CONDA_ENV="/global/homes/r/rym/.conda/envs/ensemblefffit"
CONDA_SITE="$CONDA_ENV/lib/python3.13/site-packages"
CONDA_PYTHON="$CONDA_ENV/bin/python"
SPACK_ENV="/global/cfs/cdirs/m526/sbagchi/spack/var/spack/environments/spack_matensemble_env/.spack-env/view"
SPACK_SITE="$SPACK_ENV/lib/python3.13/site-packages"

eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda deactivate
conda activate /global/homes/r/rym/.conda/envs/ensemblefffit
echo $CONDA_PREFIX

# Add conda environment to PYTHONPATH before the Spack path
export PYTHONPATH="${CONDA_SITE}:${FLUX_SITE}:${PYTHONPATH}"

srun flux start lammps_matensemble --run_directory LAMMPS_test/run_directory --inputs_directory LAMMPS_test/inputs_directory
