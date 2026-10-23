#!/usr/bin/bash

# Start one Flux instance inside the MatEnsemble Apptainer image across the
# whole allocation (one broker per node), then run this example's pipeline
# as Flux's initial program. Works both as `sbatch launch_flux.slurm` and as
# `bash launch_flux.slurm` from inside an salloc (the #SBATCH lines are inert
# there; every SLURM_* variable read here is set by either).
#
# Run it from this example's directory -- workflow_config.yaml's paths are
# relative to the working directory:
#
#   sbatch launch_flux.slurm                                   # --stage all, workflow_config.yaml
#   CONFIG=my_config.yaml sbatch -N 2 launch_flux.slurm
#   FLUX_CMD="flux resource list" sbatch -N 2 launch_flux.slurm   # check the instance only
#
# The launch is MatEnsemble's own Frontier recipe (its `matensemble run`
# CLI: srun -N n -n n --external-launcher --mpi=pmi2 ... apptainer exec
# <sif> flux start), plus the host binds vasp_std needs (README.md step 3).
# --mpi=pmi2 is for srun to bootstrap the brokers into one instance; it
# has nothing to do with how the chores' own MPI jobs are wired.

#SBATCH -A mat201
#SBATCH -J vasp
#SBATCH -o %x-%j.out
#SBATCH -t 00:30:00
#SBATCH -q debug
#SBATCH -N 1

set -euo pipefail

# Site paths -- override as env vars, or edit these defaults.
SIF="${SIF:-/lustre/orion/mat201/scratch/rjmorelock/MatEnsemble_Update/images/matensemble-frontier-v0.5.8-ensemblefffit.sif}"
ENSEMBLE_FF_FIT="${ENSEMBLE_FF_FIT:-/lustre/orion/mat201/scratch/rjmorelock/MatEnsemble_Update/Ensemble-FF-Fit}"
# Optional `pip install --target` directory of extra Python packages, for
# trying one out before adding it to the derived image (README.md step 1c)
# -- prepended to PYTHONPATH when set.
EXTRA_SITE_PACKAGES="${EXTRA_SITE_PACKAGES:-}"
CONFIG="${CONFIG:-workflow_config.yaml}"
STAGE="${STAGE:-all}"

[[ -f "$SIF" ]] || { echo "No image at SIF=$SIF -- run build_image.slurm first" >&2; exit 1; }
[[ -f "$CONFIG" ]] || { echo "No $CONFIG in $PWD -- run this from the example directory" >&2; exit 1; }

PYTHONPATH_IN="${ENSEMBLE_FF_FIT}"
if [[ -n "$EXTRA_SITE_PACKAGES" ]]; then
    PYTHONPATH_IN="${EXTRA_SITE_PACKAGES}:${PYTHONPATH_IN}"
fi
# Scoped to the pipeline process with `env` rather than set for the whole
# container, so the Flux brokers' own startup never sees it (on Perlmutter
# a container-wide PYTHONPATH broke flux's rc1 bootstrap).
# PYTHONUNBUFFERED: otherwise the pipeline's own progress lines ("Stage:
# ...", per-run warnings) sit in Python's buffer and don't reach the Slurm
# output file until the stage, or the job, ends.
FLUX_CMD="${FLUX_CMD:-env PYTHONPATH=${PYTHONPATH_IN} PYTHONUNBUFFERED=1 python run_pipeline.py --config ${CONFIG} --stage ${STAGE}}"

# Host paths vasp/6.6.1-gpu's vasp_std needs (run_pipeline.py's
# VASP_LD_LIBRARY_PATH): Cray PE + CCE 20.0.2 runtime + libfabric/PALS
# (/opt/cray), its ROCm 6.4.2, darshan (/sw/borg), libcxi/libnl and other
# system libraries (/usr/lib64, at /opt/cray_extra/lib64). The host's
# /opt/amdgpu and /opt/xpmem go to separate mount points so the image's own
# copies at those paths stay visible. /lustre/orion and /sw/frontier (the
# VASP install and POTCARs) are in Frontier's default Apptainer bind list.
# Only vasp_std's own subprocess puts these on its library path -- never
# Flux or Python.
BINDS=(--bind /opt/cray/:/opt/cray
       --bind /usr/lib64/:/opt/cray_extra/lib64
       --bind /var/spool/slurmd
       --bind /opt/rocm-6.4.2/:/opt/rocm-6.4.2
       --bind /opt/amdgpu/:/opt/host_amdgpu
       --bind /opt/xpmem/:/opt/host_xpmem
       --bind /sw/borg/:/sw/borg)

# Flux's initial program runs only on the rank-0 broker; it goes in a file
# on Lustre so no quoting has to survive srun/apptainer/flux start.
INITIAL="${PWD}/.flux_initial_${SLURM_JOB_ID}.sh"
cat > "$INITIAL" <<EOF
echo "== flux resource list (expect ${SLURM_NNODES} node(s), $((SLURM_NNODES * 8)) GPUs)"
flux resource list
echo "== ${FLUX_CMD}"
${FLUX_CMD}
EOF

echo "Job ${SLURM_JOB_ID}: ${SLURM_NNODES} node(s) ${SLURM_JOB_NODELIST}, image ${SIF}"
status=0
# --cpu-bind=none: with Slurm's default binding the brokers' step -- and so
# the whole Flux instance -- only gets cores 10-63 of each node, and ranks
# then straddle L3/NUMA groups (examples/Frontier/RMG_MACE_ASE/README.md,
# section 9); with it, each rank gets one whole 7-core L3 group.
srun -N "$SLURM_NNODES" -n "$SLURM_NNODES" --external-launcher --gpu-bind=closest --mpi=pmi2 --cpu-bind=none \
    apptainer exec "${BINDS[@]}" "$SIF" flux start bash "$INITIAL" || status=$?
rm -f "$INITIAL"
exit $status
