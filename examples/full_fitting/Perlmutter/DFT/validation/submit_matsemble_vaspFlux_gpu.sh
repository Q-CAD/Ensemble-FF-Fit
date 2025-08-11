#!/bin/bash
#

# Number of perlmutter nodes for job runs + 1 node for management.
#SBATCH -N 2

# Change to your account
# Also change in the srun command below
#SBATCH -A m2113_g
#
# Job naming stuff
#SBATCH -J vasp_gpu_flux
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#
# Requested time
#SBATCH -t 00:30:00
#
# Requested queue
#SBATCH -C gpu
#SBATCH -q debug

export OMP_NUM_THREADS=1
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

#---------------------- SETUP FOR Flux + matensemble + vaspFlux IN PERLMUTTER -------------------------------------------------------------------

# Load the spack environment and Flux
source /global/cfs/cdirs/m526/sbagchi/spack/share/spack/setup-env.sh
spack env activate -p spack_matensemble_env
unset LUA_PATH LUA_CPATH
module load python/3.13
module load vasp/6.4.3-gpu

# Unload any currently loaded conda environments
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda deactivate

# Activate conda containing vaspFlux within the spack
conda activate /global/homes/r/rym/.conda/envs/ensemblefffit

# Step 1: Generate the new VASP files from any existing POSCAR or CONTCAR files; can specify arguments
echo "Generating new inputs..."
generate_vaspflux -pd Bi2Se3/ -vy inputs/vdW_single_point.yml -vs inputs/perlmutter_vasp_gpu.sh -ve vasp_std -a m2113_g

# Step 2: Run the main Flux submission workflow; can specify arguments
echo "Running the main Flux submission workflow..."
srun -c 32 --external-launcher --mpi=pmi2 --gpu-bind=closest flux start matsemble_vaspflux -pd Bi2Se3/

