#!/bin/bash

# Number of perlmutter nodes for job runs + 1 node for management.
#SBATCH -N 1
#
# Change to your account
# Also change in the srun command below
#SBATCH -A m526_g
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
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1

# Load cudatoolkit
module load PrgEnv-gnu/8.5.0
module load cudatoolkit/12.4
module load craype-accel-nvidia80

CONDA_ENV="/global/homes/r/rym/.conda/envs/jaxreaxff"
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda deactivate
conda activate "$CONDA_ENV"

srun jaxreaxff --init_FF ffield             \
               --params use_params                  \
               --geo geo                        \
               --train_file trainset.in         \
               --num_e_minim_steps 200                          \
               --e_minim_LR 1e-4                               \
               --out_folder ffields                             \
               --save_opt all                                   \
               --num_trials 1                                  \
               --num_steps 20                                  \
               --init_FF_type fixed
