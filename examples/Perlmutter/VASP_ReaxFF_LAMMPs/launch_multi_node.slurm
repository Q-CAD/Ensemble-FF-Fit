#!/usr/bin/bash

# Multi-node Flux + podman-hpc launch script for the VASP_ReaxFF_LAMMPS
# pipeline, adapted from Q-CAD/MatEnsemble's own reference recipe:
#   https://github.com/Q-CAD/MatEnsemble/blob/main/example_workflows/perlmutter/lammps_mace/raw_submit.slurm
# (that file's own comment: "under the hood the matensemble CLI tool will do
# all of this... this is quite hacky but it works" -- i.e. this IS what
# `matensemble set-image`/`matensemble run` do, just spelled out manually).
#
# Two multi-node srun+flux attempts before this one (see chat history --
# `srun -n 2 podman-hpc run ... flux start flux resource list`, then a
# second attempt adding `--external-launcher --mpi=pmi2`) both produced TWO
# independent single-node Flux instances instead of one merged 2-node one.
# Reading the reference recipe explains why -- our attempts were missing
# several things that turn out to matter specifically for the multi-node
# case (none of this was needed for the single-node raw VASP/LAMMPS GPU
# tests earlier in this pipeline's development, which is why it wasn't
# already here):
#   - --network=host, not the default (isolated, pasta-based) networking --
#     without this the brokers on different nodes can't reach each other at
#     all, regardless of what srun's own --mpi=<x> claims to set up.
#   - `-e SLURM_* -e PALS_* -e PMI_*` -- explicit env-var passthrough.
#     `srun --mpi=pmi2` sets PMI rendezvous info in ITS OWN environment, but
#     podman-hpc doesn't forward arbitrary host env vars into the container
#     by default -- without this, `flux start` inside the container never
#     sees any of it and just falls back to a standalone single-broker
#     instance, silently.
#   - `-v /var/spool/slurmd -v /run/munge -v /run/nscd` -- PMI2's own
#     authentict/step-communication sockets, needed for the rendezvous
#     handshake itself to complete inside the container.
#   - `-v /dev/cxi0-3 -v /dev/xpmem` -- the actual Slingshot NIC device
#     nodes and xpmem shared-memory device. Earlier mounts only covered the
#     /opt/cray *software* tree (library resolution) -- never the *device
#     nodes* the interconnect needs at the kernel level for brokers on
#     different nodes to actually talk to each other.
#   - No live NVML/hwloc GPU discovery at all -- instead, a static R.json
#     resource description (one entry per node, parsed from
#     SLURM_JOB_NODELIST) plus a resource.toml (noverify=true) pointed at
#     via FLUX_CONF_DIR. This is *why* the reference correctly reports a
#     merged multi-node view: live per-broker discovery (what our earlier
#     libnvidia-ml.so stub-shadowing fix enabled) only ever discovers each
#     broker's own local node -- there's no mechanism for two independently
#     -discovered single-node views to merge into one, no matter how
#     correct each one is individually.
#
# NOT copied from the reference: its LD_PATH block (/opt/lammps/build/...,
# /opt/fftw/install/lib, .../site-packages/torch/lib, etc.) -- that's
# specific to ghcr.io/q-cad/matensemble:perlmutter-v0.3.101's LAMMPS+MACE
# install layout, which doesn't match matensemble:ff-fit's VASP/LAMMPS-
# ReaxFF/JAX-ReaxFF one. GPU exposure here still uses podman-hpc's own
# --gpu convenience flag (+ --group-add keep-groups) rather than the
# reference's manual --privileged/--device list, since --gpu is already
# confirmed working for this image's raw VASP/LAMMPS GPU tests -- the
# --network=host/PMI-passthrough/device-node/R.json pieces above are the
# only things that needed adding for the multi-node case specifically.
# --privileged IS still carried over, since rootless container access to
# raw device nodes like /dev/cxi* plausibly needs it -- try dropping it
# later if it turns out unnecessary.
#
# IMPORTANT: run this from inside run_pipeline/ (workflow_config.yaml's own
# keys are relative to CWD, not to this script's location -- `-w $PWD`
# below only does the right thing if $PWD is run_pipeline/ when this script
# starts). Works both as `sbatch launch_multi_node.slurm` and as
# `bash launch_multi_node.slurm` from inside an already-active interactive
# allocation (the #SBATCH lines are inert outside of sbatch; every
# SLURM_* variable this script reads is already set by any Slurm
# allocation, interactive or batch).

#SBATCH -A m2113_g
#SBATCH -C gpu
#SBATCH --qos debug
#SBATCH -t 0:30:00
#SBATCH -N 2
#SBATCH --ntasks-per-node=1   # 1 Flux broker per node
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=closest

REPO_ROOT="/pscratch/sd/r/rym/MatEnsemble"
ENSEMBLE_FF_FIT="${REPO_ROOT}/Ensemble-FF-Fit"

# The image's DEFAULT python (/opt/basic/bin/python3) only sees the flux
# python bindings vendored into /opt/basic/lib/python3.12/site-packages,
# which are INCOMPLETE -- confirmed missing _flux/_rlist.abi3.so (present:
# only _core, _hostlist, _idset), which flux.resource.Rlist (imported by
# flux.resource.list, which matensemble.manager.FluxManager._get_allocation_info
# calls directly) needs. A SEPARATE, COMPLETE flux python install exists at
# /usr/local/lib/python3.12/site-packages (flux-core built from source under
# /opt/flux-core, then `make install`'d there) -- confirmed via direct
# import test to have a working flux.resource.list AND flux.job, and to not
# shadow/break yaml, pymatgen, or matensemble itself (those still resolve
# from /opt/basic/... unaffected, since /usr/local/... has no competing
# copies of them). Listed first on PYTHONPATH below so it wins the import
# race for `flux` specifically over the incomplete /opt/basic/... copy
# (PYTHONPATH entries all precede the interpreter's own default site-packages
# search, so ordering BETWEEN PYTHONPATH entries only matters when two of
# them could both provide the same top-level package -- true here for flux,
# not for Ensemble-FF-Fit).
FLUX_PYTHON_BINDINGS="/usr/local/lib/python3.12/site-packages"

# ============================INPUT IMAGE AND COMMAND=============================
# Swap FLUX_CMD to the real pipeline invocation once `flux resource list`
# confirms the merge itself works -- test cheap and fast first, given the
# last three attempts all failed at or before this exact step; don't bury a
# fourth possible failure inside a full converge_dft_data run.
#
# The real invocation below is wrapped in `env PYTHONPATH=... <cmd>`
# DELIBERATELY, not a plain command string -- CONFIRMED (2026-09-03, via
# Q-CAD/MatEnsemble's raw_submit.slurm run unmodified except for IMAGE)
# that setting PYTHONPATH globally at the container level (this script
# previously did `-e PYTHONPATH=${ENSEMBLE_FF_FIT}` in PODMAN_ARGS) breaks
# flux's OWN rc1 bootstrap ("ModuleNotFoundError: No module named
# '_cffi_backend'" importing _flux._core) even though a plain `python3 -c
# "import _cffi_backend"` with that exact PYTHONPATH set (tested directly,
# in-container) works fine -- so whatever's sensitive to it is specific to
# how flux's broker process itself sees its OWN environment during rc1, not
# python's module resolution in general. Scoping PYTHONPATH to just the one
# command that actually needs Ensemble-FF-Fit/the complete flux bindings
# (via `env VAR=val cmd`, same principle as vasp_container_env()'s per-chore
# Resources(env=...) instead of a container-wide override) avoids ever
# setting it on `flux start` itself, so rc1 never sees it.
IMAGE="matensemble:ff-fit"
# FLUX_CMD="flux resource list"
FLUX_CMD="env PYTHONPATH=${FLUX_PYTHON_BINDINGS}:${ENSEMBLE_FF_FIT} /opt/basic/bin/python run_pipeline.py --config workflow_config.yaml --stage check_coordination_stability"
# ================================================================================

export NNODES=${SLURM_JOB_NUM_NODES}
export NGPUS=4
export RUNSTRING="${NNODES}node-${NGPUS}gpu-J${SLURM_JOB_ID}"

export OMP_NUM_THREADS=1
export SLURM_CPU_BIND=cores
export CRAY_ACCEL_TARGET=nvidia80
export MPICH_GPU_SUPPORT_ENABLED=1

# DELIBERATELY no container-level LD_LIBRARY_PATH override here (tried once,
# reverted -- see below). VASP's own library needs are scoped to the
# vasp_std chore's own Resources(env=vasp_container_env(...)) call in
# run_pipeline.py, not needed by flux/python3/mkdir/cat/bash's other forked
# children -- those are the container's OWN native (Ubuntu-glibc-linked)
# tools and need no host library paths at all to run correctly.
#
# CONFIRMED BROKEN (2026-09-03): an earlier version of this script exported
# LD_LIBRARY_PATH=...:/host_lib64:/host_lib globally via `-e` here. /host_lib
# is the remapped host /lib64, which contains the HOST's (RHEL/SLES) libc.so.6
# -- with it in LD_LIBRARY_PATH for the whole container session (not scoped
# to one subprocess the way vasp_container_env's is), EVERY dynamically
# linked binary without its own RPATH (bash's forked children, mkdir, cat,
# python3, flux itself) started resolving libc.so.6 from there instead of
# the container's own (Ubuntu) build -- same ABI-mismatch SIGSEGV class as
# the earlier "bind-mounting host /lib64 straight over the container's own
# /lib64 breaks bash" bug (see VASP_LD_LIBRARY_PATH's docstring in
# run_pipeline.py), just triggered via LD_LIBRARY_PATH search order instead
# of a direct mount collision. Symptom was every single exec'd command in
# flux_container_*.sh segfaulting (exit 139), including plain `mkdir`.
# Also unnecessary even for flux's own sake: flux doesn't need to find
# libnvidia-ml.so here at all, since the R.json/noverify=true resource
# description below bypasses live NVML/hwloc GPU discovery entirely (that
# library-path fix was only ever needed for the live-discovery path).

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
  -v /usr/lib64:/host_lib64 -v /lib64:/host_lib \
  -v /global/common/software/nersc9/vasp:/global/common/software/nersc9/vasp \
  -v /opt/nvidia/hpc_sdk/Linux_x86_64/25.9:/opt/nvidia/hpc_sdk/Linux_x86_64/25.9 \
  -v /opt/nvidia/hpc_sdk/Linux_x86_64/26.5:/opt/nvidia/hpc_sdk/Linux_x86_64/26.5 \
  -v /opt/cray:/opt/cray \
  -v /global/common/software/nersc9/darshan/3.4.6-gcc-13.2.1:/global/common/software/nersc9/darshan/3.4.6-gcc-13.2.1 \
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
# `flux resource list` report one merged NNODES=2 instance instead of two
# separate NNODES=1 ones.
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

# os.cpu_count() (used by the Q-CAD reference this was adapted from) AND
# os.sched_getaffinity(0) (this script's own first fix attempt) BOTH report
# LOGICAL cpu count -- i.e. hardware threads/PUs, 128 on Perlmutter's AMD
# EPYC 7763 (Milan) GPU nodes (64 physical cores x 2-way SMT) -- neither
# maps to what flux-shell's cpu-affinity plugin actually validates against.
# CONFIRMED WRONG (2026-09-03) TWICE: first with os.cpu_count() (128), then
# again with os.sched_getaffinity(0) (still reports the same kind of
# logical/thread count, not a physical-core count) -- both hit flux-shell's
# cpu-affinity plugin failing on specific cores >63 ("coreNNN not in
# topology") once real chores launched (a passive "flux resource list"
# never validates R.json against real hwloc topology, only flux-shell's
# actual per-task core-binding does, so neither prior attempt could have
# caught this from that cheap test alone). hwloc -- and flux's own
# cpu-affinity plugin, given the error literally says "core", not "PU" --
# distinguishes Core objects (physical, 0-63 here) from PU objects
# (logical/hardware-thread, 0-127) as different topology levels; any
# declared "core" ID above 63 is invalid on this hardware regardless of
# cgroup/cpuset state, which is why the earlier cgroup-reservation
# explanation for "124" specifically was the wrong theory -- 124-127 was
# just whichever core IDs the scheduler happened to assign out of the
# still-too-wide declared pool, not a real cutoff.
#
# Hardcoded to 64 (not derived from any os.cpu_count()-family call, all of
# which report logical/thread counts unsuitable for this specific R.json
# field) -- matches this pipeline's EARLIER, EMPIRICALLY-CONFIRMED live
# hwloc discovery result from before this script switched to a static
# R.json at all (flux resource list's own live-discovered NCORES was 64 for
# a single Perlmutter GPU node, back when NVML/hwloc live discovery worked
# for a single node -- see run_pipeline.py's VASP_LD_LIBRARY_PATH docstring
# history). Same "hardcode this Perlmutter-GPU-node-specific, verified
# constant" convention already used for NGPUS=4 below -- this script
# already assumes homogeneous, known-spec Perlmutter GPU nodes throughout,
# not something newly introduced here.
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

# DIAGNOSTIC (2026-09-03): rc1 on broker rank 0 failed with
# "ModuleNotFoundError: No module named '_cffi_backend'" importing
# _flux._core -- NOT reproducible via a bare "podman-hpc run" on the login
# node, with or without this script's exact PYTHONPATH/namespace flags
# (--privileged --pid=host --ipc=host --network=host), so something about
# the real compute-node/PMI2-launched context is different in a way that
# isn't visible from here. Printing the same checks right before flux
# actually starts, in the SAME container/rank/env the real failure happens
# in, so the next run's output pins down what's actually different instead
# of guessing further. Remove this block once the cause is confirmed.
echo "=== pre-flux-start diagnostic (\$(hostname), rank \${SLURM_PROCID:-unknown}) ==="
echo "PYTHONPATH=\${PYTHONPATH:-<unset>}"
echo "which python3: \$(which python3)"
python3 -c "import sys; print('sys.executable:', sys.executable); print('sys.path:'); [print(' ', p) for p in sys.path]"
python3 -c "import _cffi_backend; print('_cffi_backend OK:', _cffi_backend.__file__)" || echo "_cffi_backend IMPORT FAILED"
python3 -c "from _flux._core import ffi, lib; print('_flux._core OK')" || echo "_flux._core IMPORT FAILED"
echo "=== end diagnostic ==="

FLUX_CONF_DIR=/tmp/fluxcfg flux start ${FLUX_CMD}
SCRIPT_EOF

# Run the script inside the container across the allocation -- one task per
# node (-N $SLURM_NNODES -n $SLURM_NNODES), PMI2-mediated broker rendezvous.
srun -N $SLURM_NNODES -n $SLURM_NNODES --cpu-bind=none --mpi=pmi2 podman-hpc run ${PODMAN_ARGS} ${IMAGE} bash ${CONTAINER_SCRIPT}

# Remove the script after using it
rm -f ${CONTAINER_SCRIPT}
