#!/usr/bin/bash

# Multi-node Flux + podman-hpc launch script for the MACE_ACE_LAMMPS
# pipeline -- adapted from examples/Perlmutter/VASP_MACE_ASE/
# launch_multi_node.slurm (itself from VASP_ReaxFF_LAMMPs/, ultimately from
# Q-CAD/MatEnsemble's own reference recipe:
#   https://github.com/Q-CAD/MatEnsemble/blob/main/example_workflows/perlmutter/lammps_mace/raw_submit.slurm
# ). See that file's own long history comment for the full story of why
# every podman-hpc flag below exists (multi-node Flux rendezvous is far
# more finicky than it looks -- two earlier attempts there each produced
# TWO independent single-node Flux instances instead of one merged one).
# The short version, kept here since it's load-bearing, not decorative:
#   - --network=host -- brokers on different nodes can't reach each other
#     with the default (isolated) networking, regardless of srun's --mpi=.
#   - `-e SLURM_* -e PALS_* -e PMI_*` -- podman-hpc doesn't forward
#     arbitrary host env vars into the container by default; without this,
#     flux start never sees srun --mpi=pmi2's own rendezvous info and
#     silently falls back to a standalone single-broker instance.
#   - `-v /var/spool/slurmd -v /run/munge -v /run/nscd` -- PMI2's own
#     authentication/step-communication sockets.
#   - `-v /dev/cxi0-3 -v /dev/xpmem` -- the actual Slingshot NIC device
#     nodes and xpmem shared-memory device, needed at the kernel level for
#     cross-node broker communication.
#   - A static R.json resource description (one entry per node, parsed from
#     SLURM_JOB_NODELIST) + resource.toml (noverify=true), NOT live
#     NVML/hwloc GPU discovery -- live per-broker discovery only ever sees
#     each broker's own local node, so two independently-discovered
#     single-node views never merge into one multi-node view no matter how
#     correct each is individually.
#
# Dropped relative to VASP_MACE_ASE's own version: every VASP-specific host
# mount (`/global/common/software/nersc9/vasp`, the NVIDIA HPC SDK trees,
# `/opt/cray`, darshan) -- this pipeline never runs VASP, so those paths
# would just be dead mounts. GPU exposure still uses podman-hpc's own --gpu
# convenience flag (+ --group-add keep-groups, --privileged for the raw
# device-node access cxi*/xpmem plausibly need).
#
# IMPORTANT: run this from inside this example's directory
# (workflow_config.yaml's own keys are relative to CWD, not to this
# script's location -- `-w $PWD` below only does the right thing if $PWD
# is this example's directory when this script starts). CONFIG/STAGE pick
# the config file and stage, e.g. `CONFIG=my_config.yaml
# STAGE=teacher_preparation sbatch launch_multi_node.slurm`; both default
# to a full one-loop run. Works both as `sbatch launch_multi_node.slurm`
# and as `bash launch_multi_node.slurm` from inside an already-active
# interactive allocation (the #SBATCH lines are inert outside of sbatch;
# every SLURM_* variable this script reads is already set by any Slurm
# allocation, interactive or batch).

#SBATCH -A <your_project>_g
#SBATCH -C gpu
#SBATCH --qos interactive
#SBATCH -t 4:00:00
#SBATCH -N 1
#SBATCH -J mace_ace_lammps
#SBATCH --ntasks-per-node=1   # 1 Flux broker per node
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=closest

# Path to your Ensemble-FF-Fit checkout -- bind-mounted into the container
# and put on PYTHONPATH (the image only carries its dependencies, see
# Dockerfile.ace's own comment on why; a user is expected to iterate on
# EnsembleFFFit without rebuilding). TensorPotential (the GPU/TensorFlow
# evaluator ACE fitting delegates to via `evaluator: tensorpot` in
# workflow_config.yaml) does NOT need an env var or bind-mount here -- it's
# pip-installed fixed into the image itself (see Dockerfile.ace).
ENSEMBLE_FF_FIT="${ENSEMBLE_FF_FIT:-/path/to/your/Ensemble-FF-Fit}"
CONFIG="${CONFIG:-workflow_config.yaml}"
STAGE="${STAGE:-all}"

# The image's DEFAULT python (/opt/basic/bin/python3) only sees the flux
# python bindings vendored into /opt/basic/lib/python3.12/site-packages,
# which are INCOMPLETE -- confirmed missing _flux/_rlist.abi3.so (present:
# only _core, _hostlist, _idset), which flux.resource.Rlist (imported by
# flux.resource.list, which matensemble.manager.FluxManager._get_allocation_info
# calls directly) needs. A SEPARATE, COMPLETE flux python install exists at
# /usr/local/lib/python3.12/site-packages (flux-core built from source under
# /opt/flux-core, then `make install`'d there) -- confirmed via direct
# import test to have a working flux.resource.list AND flux.job, and to not
# shadow/break yaml, pymatgen, or matensemble itself. Listed first on
# PYTHONPATH below so it wins the import race for `flux` specifically over
# the incomplete /opt/basic/... copy.
FLUX_PYTHON_BINDINGS="/usr/local/lib/python3.12/site-packages"

# ============================INPUT IMAGE AND COMMAND=============================
# Swap FLUX_CMD to the real pipeline invocation once `flux resource list`
# confirms the multi-node merge itself works (on a single-node allocation
# this isn't a concern) -- test cheap and fast first if you're scaling this
# past 1 node.
#
# The real invocation below is wrapped in `env PYTHONPATH=... <cmd>`
# DELIBERATELY, not a plain command string -- CONFIRMED (in this project's
# own development) that setting PYTHONPATH globally at the container level
# breaks flux's OWN rc1 bootstrap ("ModuleNotFoundError: No module named
# '_cffi_backend'" importing _flux._core), even though the same PYTHONPATH
# causes no issue for a plain python invocation -- whatever's sensitive to
# it is specific to how flux's broker process sees its OWN environment
# during rc1, not python's module resolution in general. Scoping PYTHONPATH
# to just the one command that actually needs Ensemble-FF-Fit/the complete
# flux bindings avoids ever setting it on `flux start` itself.
IMAGE="matensemble:mace-ace-lammps"
# FLUX_CMD="flux resource list"
FLUX_CMD="env PYTHONPATH=${FLUX_PYTHON_BINDINGS}:${ENSEMBLE_FF_FIT} /opt/basic/bin/python run_pipeline.py --config ${CONFIG} --stage ${STAGE}"
# ================================================================================

export NNODES=${SLURM_JOB_NUM_NODES}
export NGPUS=4
export RUNSTRING="${NNODES}node-${NGPUS}gpu-J${SLURM_JOB_ID}"

export OMP_NUM_THREADS=1
export SLURM_CPU_BIND=cores
export CRAY_ACCEL_TARGET=nvidia80
export MPICH_GPU_SUPPORT_ENABLED=1

# DELIBERATELY no container-level LD_LIBRARY_PATH override here -- confirmed
# in this project's own development to break every exec'd command in the
# container (bash's forked children, mkdir, cat, python3, flux itself) via
# an ABI mismatch once the host's own libc.so.6 shadows the container's
# (Ubuntu) build. Also unnecessary: flux doesn't need to find
# libnvidia-ml.so here at all, since the R.json/noverify=true resource
# description below bypasses live NVML/hwloc GPU discovery entirely.

PODMAN_ARGS="--rm --gpu --group-add keep-groups --privileged \
  -e SLURM_* -e PALS_* -e PMI_* \
  --ipc=host --network=host --pid=host \
  -v /dev/shm:/dev/shm \
  -v /dev/cxi0:/dev/cxi0 -v /dev/cxi1:/dev/cxi1 \
  -v /dev/cxi2:/dev/cxi2 -v /dev/cxi3:/dev/cxi3 \
  -v /dev/xpmem:/dev/xpmem \
  -v /var/spool/slurmd:/var/spool/slurmd \
  -v /run/munge:/run/munge -v /run/nscd:/run/nscd \
  -e MPICH_GPU_SUPPORT_ENABLED -e CRAY_ACCEL_TARGET \
  -e OMP_NUM_THREADS -e SLURM_CPU_BIND \
  -v ${ENSEMBLE_FF_FIT}:${ENSEMBLE_FF_FIT} \
  -v $SCRATCH:$SCRATCH -w $PWD"
GPU_RANGE="0-$((NGPUS - 1))"

# Write container script to $SCRATCH (mounted inside the container) so it
# can be passed as a path argument to bash -- avoids stdin piping through
# srun/podman.
CONTAINER_SCRIPT="${SCRATCH}/flux_container_${SLURM_JOB_ID}.sh"

# Static R.json resource description (one entry per allocated node, parsed
# from SLURM_JOB_NODELIST) instead of live NVML/hwloc discovery -- see the
# top-of-file comment for why this (not a library-path fix) is what makes
# a multi-node allocation report one merged Flux instance instead of
# several separate single-node ones.
cat >"${CONTAINER_SCRIPT}" <<SCRIPT_EOF
python3 << 'PYEOF' > /tmp/R.json
import json
import socket
import os
import re

def expand_nodelist(s):
    nodes = []
    for part in re.split(r",(?![^\[]*\])", s):
        m = re.match(r"^(.*?)\[(.+)\]\$", part)
        if m:
            prefix, ranges = m.group(1), m.group(2)
            for r in ranges.split(","):
                if "-" in r:
                    a, b = r.split("-")
                    for i in range(int(a), int(b) + 1):
                        nodes.append(f"{prefix}{str(i).zfill(len(a))}")
                else:
                    nodes.append(f"{prefix}{r}")
        else:
            nodes.append(part)
    return nodes

# os.cpu_count()/os.sched_getaffinity(0) both report LOGICAL cpu count --
# hardware threads/PUs, 128 on Perlmutter's AMD EPYC 7763 (Milan) GPU nodes
# (64 physical cores x 2-way SMT) -- neither maps to what flux-shell's
# cpu-affinity plugin actually validates against (it distinguishes Core
# objects, physical, 0-63, from PU objects, logical, 0-127, as different
# hwloc topology levels; a declared "core" ID above 63 is invalid on this
# hardware regardless of cgroup/cpuset state). Hardcoded to 64 -- matches
# this hardware's own physical core count, confirmed via live hwloc
# discovery earlier in this project's development, not derived from any
# os.cpu_count()-family call (all of which report the wrong, logical count
# for this specific R.json field). Same "hardcode this Perlmutter-GPU-node
# -specific, verified constant" convention as NGPUS=4 above -- this script
# already assumes homogeneous, known-spec Perlmutter GPU nodes throughout.
ncores = 64
nodelist_str = os.environ.get("SLURM_JOB_NODELIST", socket.gethostname())
nodes = expand_nodelist(nodelist_str)
r_lite = [
    {"rank": str(i), "children": {"core": f"0-{ncores-1}", "gpu": "${GPU_RANGE}"}}
    for i in range(len(nodes))
]
r = {
    "version": 1,
    "execution": {
        "R_lite": r_lite,
        "starttime": 0.0,
        "expiration": 0.0,
        "nodelist": nodes,
    },
}
print(json.dumps(r))
PYEOF

mkdir -p /tmp/fluxcfg
cat > /tmp/fluxcfg/resource.toml << 'TOML'
[resource]
path = "/tmp/R.json"
noverify = true
TOML

FLUX_CONF_DIR=/tmp/fluxcfg flux start ${FLUX_CMD}
SCRIPT_EOF

# Run the script inside the container across the allocation -- one task per
# node (-N $SLURM_NNODES -n $SLURM_NNODES), PMI2-mediated broker rendezvous.
srun -N $SLURM_NNODES -n $SLURM_NNODES --cpu-bind=none --mpi=pmi2 podman-hpc run ${PODMAN_ARGS} ${IMAGE} bash ${CONTAINER_SCRIPT}

# Remove the script after using it
rm -f ${CONTAINER_SCRIPT}
