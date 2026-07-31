#!/bin/bash

# Number of perlmutter nodes for job runs + 1 node for management.
#SBATCH -N 8
#
# Change to your account
# Also change in the srun command below
#SBATCH -A m2113_g
#
# Job naming stuff
#SBATCH -J jaxreaxff-matensemble
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
# GPU stuff
#SBATCH -c 32 
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

#---------------------- SETUP FOR Flux + matensemble + JaxReaxFF IN PERLMUTTER -------------------------------------------------------------------
# Load cudatoolkit
module load PrgEnv-gnu/8.5.0
module load cudatoolkit/12.4
module load craype-accel-nvidia80

# Activate the spack
source /global/cfs/cdirs/m526/sbagchi/spack/share/spack/setup-env.sh
spack env activate -p spack_matensemble_env
unset LUA_PATH LUA_CPATH

CONDA_ENV="/global/homes/r/rym/.conda/envs/jaxreaxff"
CONDA_SITE="$CONDA_ENV/lib/python3.13/site-packages"
CONDA_PYTHON="$CONDA_ENV/bin/python"
SPACK_ENV="/global/cfs/cdirs/m526/sbagchi/spack/var/spack/environments/spack_matensemble_env/.spack-env/view"
SPACK_SITE="$SPACK_ENV/lib/python3.13/site-packages"

# Add conda environment with Jax-ReaxFF and Ensemble-FF-Fit to PYTHONPATH before the Spack path 
export PYTHONPATH="${CONDA_SITE}:${FLUX_SITE}:${PYTHONPATH}"

eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda deactivate
conda activate "$CONDA_ENV"

srun flux start jaxreaxff_matensemble --run_directory run_directory \
	                              --inputs_directory inputs_directory \
				      --check_files geo train_file \
				      --fits_per_runpath 3 \
				      --num_e_minim_steps 200 \
				      --e_minim_LR 1e-4 \
				      --use_valid True \
				      --out_folder outputs \
				      --save_opt all \
				      --num_trials 1 \
				      --num_steps 5 \
				      --init_FF_type fixed 
