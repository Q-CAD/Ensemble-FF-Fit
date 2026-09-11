"""
Single entry point for the VASP_ReaxFF_LAMMPS pipeline: VASP DFT convergence
-> parse2fit input generation -> finite-temperature MD structure sampling ->
JAX-ReaxFF ensemble-input construction -> JAX-ReaxFF ensemble fitting ->
LAMMPS/ReaxFF validation single points (ranked against DFT ground truth) ->
LAMMPS/ReaxFF finite-temperature MD (Kokkos GPU) -> coordination-number
stability check.

Adapted from examples/Frontier/RMG_MACE_ASE's run_pipeline.py for Perlmutter
-- see Perlmutter_Pipeline_Wiring.md/Perlmutter_Build_Order.md for the
design decisions behind the initial port, and JaxReaxFF_Integration_Plan.md
for the JAX-ReaxFF fitting/validation/coordination-check work built on top
of it. converge_dft_data targets VASP (via DFT/vasp_dft.py); fitting targets
JAX-ReaxFF; every MD stage (validation single points, finite-temperature MD)
targets LAMMPS/ReaxFF, with Kokkos GPU acceleration for the finite-
temperature MD stage specifically (see MD/finite_temperature/
coordination_check/lammps_inputs/in.npt_room_temp's own comments for the
Kokkos-specific settings ReaxFF needs).

An earlier UQ/active-learning loop (downselect -> uq_single_points ->
select_dft_candidates, iteratively picking new DFT candidates by ensemble
disagreement) has been removed -- the JAX-ReaxFF ensemble fits produced
here weren't judged reliable enough yet to drive that kind of active
selection, and parse2fit's own relative-energy handling is user-directed
rather than fully autonomous, which the UQ loop would have needed. This
example instead uses a strictly fixed training dataset, validated via
rank_reaxff_validation (static single points vs. DFT) and
check_coordination_stability (real short MD runs vs. DFT) instead.

Run one stage at a time -- each stage is its own compute allocation/
apptainer invocation (converge_dft_data, fit_and_validate,
reaxff_validation_single_points, and finite_temperature_md_batch submit
MatEnsemble/Flux chores; every other stage is pure analysis/file-ops and
can run anywhere EnsembleFFFit's analysis extras are installed)
-- or run the whole ordered sequence in one process via --stage all:

    python run_pipeline.py --config workflow_config.yaml --stage converge_dft_data
    python run_pipeline.py --config workflow_config.yaml --stage parse2fit_generation
    python run_pipeline.py --config workflow_config.yaml --stage sample_ft_md_structures
    python run_pipeline.py --config workflow_config.yaml --stage build_ff_inputs
    python run_pipeline.py --config workflow_config.yaml --stage fit_and_validate
    python run_pipeline.py --config workflow_config.yaml --stage prepare_reaxff_validation_structures
    python run_pipeline.py --config workflow_config.yaml --stage stage_reaxff_validation_force_fields
    python run_pipeline.py --config workflow_config.yaml --stage reaxff_validation_single_points
    python run_pipeline.py --config workflow_config.yaml --stage rank_reaxff_validation
    python run_pipeline.py --config workflow_config.yaml --stage stage_ft_md_force_fields
    python run_pipeline.py --config workflow_config.yaml --stage finite_temperature_md_batch
    python run_pipeline.py --config workflow_config.yaml --stage check_coordination_stability
    python run_pipeline.py --config workflow_config.yaml --stage all

matensemble is only imported inside the stage functions that actually need
it (converge_dft_data, fit_and_validate, reaxff_validation_single_points,
finite_temperature_md_batch), not at module load time, so the other stages
don't require the Flux container. best_force_field's sklearn dependency is
similarly deferred into run_rank_reaxff_validation, since it's unconfirmed
whether every environment running the other stages has sklearn installed.

converge_dft_data's VASP-specific environment/node-sizing logic (below) is
deliberately kept inline here rather than moved into EnsembleFFFit -- it's
tied to this one DFT backend/container, and will need to look different for
a different DFT driver, same reasoning as why the MD drivers stay
user-owned rather than canonicalized.
"""
import argparse
import os
import random
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import yaml

from EnsembleFFFit.analysis.dict_parsers import parse_labeled_tree, parse_reference_tree
from EnsembleFFFit.analysis.downselect_force_fields import select_and_copy
from EnsembleFFFit.utilities.general import print_and_write

# Perlmutter/Podman-HPC-specific runtime library path for the vasp_std
# subprocess specifically -- NOT exported in the surrounding shell, only ever
# passed via a chore's/Resources' own `env`, mirroring RMG_LD_LIBRARY_PATH's
# precedent on Frontier (see git history for that version if useful as a
# reference).
#
# CONFIRMED working (2026-09-03, interactive allocation on nid001925/
# nid002184, matensemble:ff-fit) against vasp/6.4.2-gpu, running noticeably
# faster than the CPU build in the same raw-executable test -- see
# Perlmutter_Build_Order.md's "VASP-in-container: confirmed recipe" section
# for the full story. Superseded the CPU build's LD_LIBRARY_PATH (kept here
# in git history if the CPU build is ever needed again) once --gpu was
# already required at container-launch time anyway (for libcuda.so.1/Flux's
# own GPU resource discovery), removing the reason to run CPU VASP at all.
#
# The GPU build (unlike the CPU build) is linked against a SPLIT NVHPC
# toolchain -- confirmed via `readelf -d` on vasp_std itself: RPATH lists
# .../25.9/compilers/{lib,extras/qd/lib}, .../25.9/math_libs/lib64, and
# .../26.5/cuda/13.2/lib64 (RPATH always wins over LD_LIBRARY_PATH, so these
# are technically redundant here, but listed anyway as a safety net for any
# of THEIR OWN transitive dependencies that don't carry their own RPATH).
# .../26.5/math_libs/13.2/lib64 (libcusolver/libcublas/libcufft's own
# NEEDED-list siblings) is NOT in vasp_std's RPATH and does need to be here.
# Cray-specific dirs come first, same reasoning as the CPU build: the real
# Cray MPICH/libfabric win over any same-named container-native library --
# /opt/cray/pe/lib64 alone (confirmed via `find`/module inspection) already
# resolves libmpi_gtl_cuda.so.0 (Cray's own GPU Transport Layer, required
# for CUDA-aware Cray MPI), libmpi_nvidia.so.12/libmpifort_nvidia.so.12, and
# libhdf5_fortran_nvidia.so.310 -- no separate cray-mpich/hdf5-parallel path
# needed. darshan (VASP's I/O profiling library, always linked in on this
# module) comes next; /host_lib64 and /host_lib (SLES/RHEL system libs --
# liblustreapi/libxpmem/libz/libc/libstdc++/etc. -- libxpmem.so.0 confirmed
# via xpmem module inspection to itself resolve from host /usr/lib64, same
# as the rest of this group) come last so they only ever fill in what the
# container genuinely lacks.
#
# /host_lib64 and /host_lib are DELIBERATE, NOT typos for /usr/lib64 and
# /lib64 -- see vdW_single_point.yml's `command` field for why: podman-hpc's
# own hook_tool prestart hook (fires on every podman-hpc run, unconditionally
# -- see podman_hpc/siteconfig.py) auto-injects the real NVIDIA driver into
# /usr/lib64 on every container start, and bind-mounting straight over that
# same path collides with it (`crun: error executing hook`). Separately,
# bind-mounting host /lib64 directly over the container's own /lib64 breaks
# the container's OWN native tools (bash itself failed with a GLIBC_PRIVATE
# symbol mismatch), because /lib64/ld-linux-x86-64.so.2 (the ELF interpreter
# path, fixed by the kernel at exec() time, unrelated to LD_LIBRARY_PATH) has
# to be paired with a matching libc build -- the container's own tools need
# Ubuntu's pairing, vasp_std needs RHEL's. The fix that lets both coexist:
# mount host /usr/lib64 and /lib64 at the non-conflicting paths /host_lib64
# and /host_lib instead, include them here in LD_LIBRARY_PATH for the many
# libraries that DO respect it, and separately invoke vasp_std through the
# real host loader explicitly (/host_lib/ld-linux-x86-64.so.2, see
# vdW_single_point.yml's `command`) for the one thing LD_LIBRARY_PATH can't
# fix -- the loader itself isn't found via LD_LIBRARY_PATH, only via the
# kernel's fixed ELF interpreter lookup or by being invoked directly.
#
# Container-launch-level requirements this depends on (NOT settable from
# this Python file -- must be present on whatever `podman-hpc run` starts
# the chore's container): --gpu (makes the real NVIDIA driver/libcuda.so.1
# visible at all -- also what Flux's OWN GPU resource discovery needs at
# `flux start` time, a separate, broker-level LD_LIBRARY_PATH concern from
# this dict, see Perlmutter_Build_Order.md), --network=none (works around a
# `pasta` rootless-networking netlink race we hit repeatedly; not needed for
# anything VASP actually does here), --group-add keep-groups (vasp_std and
# its containing directories are group-restricted to NERSC's `vasp6` group
# on the host -- without this, a rootless container can't even traverse
# into that directory, surfacing as a confusing "No such file or directory"
# rather than a permission error), and bind mounts for
# /global/common/software/nersc9/vasp,
# /opt/nvidia/hpc_sdk/Linux_x86_64/25.9,
# /opt/nvidia/hpc_sdk/Linux_x86_64/26.5, /opt/cray,
# /global/common/software/nersc9/darshan/3.4.6-gcc-13.2.1, and
# /usr/lib64:/host_lib64, /lib64:/host_lib. NOT /global/homes -- CONFIRMED
# (2026-09-04) a rootless podman-hpc bind-mount of it doesn't reliably work
# from inside the container on compute nodes (unlike /global/common,
# mounted above, which does), so pymatgen's own pseudopotentials directory
# (previously staged under a home directory for POTCAR generation) now
# lives under $SCRATCH instead (see vdW_single_point.yml's
# pseudopotentials_directory) -- no /global/homes mount needed at all.
# Re-verify every path here if the VASP module version, container image, or
# NVHPC/Cray module versions change.
#
# /usr/local/lib/flux comes FIRST, ahead even of the Cray-specific dirs
# below -- CONFIRMED (2026-09-04) this is required for vasp_std to actually
# run as a coordinated multi-rank MPI job rather than N independent solo
# ranks (symptom: "running 1 mpi-ranks... 1 GPUs detected" x N, each
# colliding on the same working_directory's HDF5 output file). Root cause:
# Cray MPICH (libmpi_nvidia.so.12, one of vasp_std's own NEEDED libraries)
# itself has a direct NEEDED dependency on libpmi.so.0/libpmi2.so.0 -- and
# /opt/cray/pe/lib64 (listed below) has its OWN libpmi.so.0/libpmi2.so.0,
# symlinked to Cray's native PMI implementation (designed for Cray's own
# Slurm-native launch mechanism, not Flux). With that resolved first, Cray
# MPICH's MPI_Init() loads Cray's own PMI client instead of Flux's -- even
# though Flux correctly sets PMI_RANK/PMI_SIZE/PMI_FD/FLUX_PMI_LIBRARY_PATH
# in this chore's own env (confirmed directly, see vasp_dft.py's PMI env
# diagnostic print), Cray's PMI client doesn't know how to talk to Flux's
# PMI_FD-based coordination at all, and MPI_Init() silently falls back to
# singleton (1-rank) mode rather than erroring loudly. Flux's own
# libpmi.so.0/libpmi2.so.0 (at /usr/local/lib/flux, confirmed to export
# exactly PMI_Init/PMI_Initialized under the matching soname) needs to
# resolve FIRST specifically for this one pair of libraries -- confirmed
# /usr/local/lib/flux contains nothing else (just libpmi*.so and one
# unrelated libreapi_cli.so) that could shadow anything else vasp_std
# needs, so putting it ahead of the Cray dirs is safe for everything else
# NEEDED still resolving from Cray's own directory as intended.
VASP_LD_LIBRARY_PATH = (
    "/usr/local/lib/flux:"
    "/opt/cray/pe/lib64:/opt/cray/libfabric/1.22.0/lib64:"
    "/opt/nvidia/hpc_sdk/Linux_x86_64/25.9/compilers/lib:"
    "/opt/nvidia/hpc_sdk/Linux_x86_64/25.9/compilers/extras/qd/lib:"
    "/opt/nvidia/hpc_sdk/Linux_x86_64/25.9/math_libs/lib64:"
    "/opt/nvidia/hpc_sdk/Linux_x86_64/26.5/cuda/13.2/lib64:"
    "/opt/nvidia/hpc_sdk/Linux_x86_64/26.5/math_libs/13.2/lib64:"
    "/global/common/software/nersc9/darshan/3.4.6-gcc-13.2.1/lib:"
    "/host_lib64:/host_lib"
)


def vasp_container_env(cores_per_task):
    """Env for the vasp_std chore/task -- see VASP_LD_LIBRARY_PATH docstring.

    OMP_NUM_THREADS derived from cores_per_task (not a second, separately
    hardcoded number) so the two can't quietly drift out of agreement --
    same reasoning as RMG_LD_LIBRARY_PATH's rmg_container_env precedent.
    PSEUDOPOTENTIAL_DIR/VDW_KERNEL_DIR/NO_STOP_MESSAGE/
    MPICH_NO_BUFFER_ALIAS_CHECK mirror vasp/6.4.2-cpu's own module-set env
    (see `module show vasp/6.4.2-cpu`) -- distinct from vasp_dft.py's own
    PMG_VASP_PSP_DIR (pymatgen's own config, used only for POTCAR
    generation, not read by vasp_std itself at runtime).

    DELIBERATELY exposed as VASP_LD_LIBRARY_PATH here, NOT the literal
    LD_LIBRARY_PATH -- CONFIRMED (2026-09-03) that setting the real
    LD_LIBRARY_PATH key applies it (via Resources(env=..., inherit_env=True))
    to this chore's WHOLE process, i.e. matensemble.runtime_worker itself (a
    plain python3 process, same as bash/mkdir/python3 in the earlier
    container-launch-level bug this exact env dict's /host_lib64:/host_lib
    entries already caused once -- see VASP_LD_LIBRARY_PATH's own docstring
    and launch_multi_node.slurm's matching history) -- not scoped to just
    the vasp_std subprocess vasp_dft.py actually needs it for. Reproduced
    directly: a bare `python3 -c "print(1)"` segfaults (rc=139) under this
    exact env with /host_lib64/host_lib actually mounted, matching the
    empty-stderr/rc=139/"no input files ever written" symptom exactly --
    matensemble.runtime_worker itself never survives startup. VASP_LD_LIBRARY_PATH
    (any other name) isn't special to the dynamic loader, so it's harmless
    at the chore-process level; vasp_dft.py reads it back out of its own
    os.environ and applies it ONLY to the vasp_std subprocess's own env=,
    the same scoping principle as launch_multi_node.slurm's `env
    PYTHONPATH=... <cmd>` fix for flux's rc1, one layer deeper.
    """
    from EnsembleFFFit.utilities.general import ensemble_fffit_pythonpath

    return {
        'PYTHONPATH': ensemble_fffit_pythonpath(),
        'OMP_NUM_THREADS': str(cores_per_task),
        'VASP_LD_LIBRARY_PATH': VASP_LD_LIBRARY_PATH,
        'PSEUDOPOTENTIAL_DIR': '/global/common/software/nersc9/vasp/dependencies/pseudopotentials',
        'VDW_KERNEL_DIR': '/global/common/software/nersc9/vasp/dependencies/vdw_kernel',
        'NO_STOP_MESSAGE': '1',
        'MPICH_NO_BUFFER_ALIAS_CHECK': '1',
    }


def lammps_container_env():
    """Env for LAMMPS/Cray-MPICH chores (MD_single_points/MD_uq_single_points/
    finite_temperature_md) -- CONFIRMED (2026-09-05) the container's own
    default LD_LIBRARY_PATH puts /usr/local/cuda/compat (a bundled
    libcuda.so.1 "forward compatibility" shim, version 575.57.08 in this
    image) ahead of anywhere /usr/lib64 would be found -- and /usr/lib64
    (where podman-hpc's own hook_tool prestart hook injects the REAL driver
    matching whatever's actually running on this node, confirmed version
    580.159.04 -- see VASP_LD_LIBRARY_PATH's own docstring for hook_tool's
    mechanism) isn't in the container's default LD_LIBRARY_PATH at all
    (unsurprising: this is an Ubuntu-base image, and /usr/lib64 isn't a
    standard Debian/Ubuntu library path). Cray MPICH's own yaksa GPU-
    datatype engine initializes CUDA at MPI_Init() time -- which LAMMPS
    always calls, even for a single-rank job -- and resolves libcuda.so.1
    to the mismatched compat shim instead of the real driver, failing with
    "CUDA Error ... unsupported display driver / cuda driver combination".
    None of MD_single_points/MD_uq_single_points/finite_temperature_md's
    chores previously set LD_LIBRARY_PATH at all (only PYTHONPATH), so they
    all inherited this same broken default via inherit_env=True.

    Prepending /usr/lib64 (not a host bind-mount -- the container's own
    native path hook_tool injects the real driver into) fixes this the same
    way VASP_LD_LIBRARY_PATH's own /host_lib64 priority ordering fixed the
    analogous libnvidia-ml.so/libpmi.so bugs elsewhere in this pipeline.
    Reads back the container's own current LD_LIBRARY_PATH (rather than
    hardcoding a snapshot of it) so everything else LAMMPS needs from that
    default (KIM/FFTW/HDF5/PLUMED/etc. paths) is preserved, not lost.
    """
    from EnsembleFFFit.utilities.general import ensemble_fffit_pythonpath

    return {
        'PYTHONPATH': ensemble_fffit_pythonpath(),
        'LD_LIBRARY_PATH': '/usr/lib64:' + os.environ.get('LD_LIBRARY_PATH', ''),
    }


def jax_reaxff_container_env():
    """Env for fit_reaxff chores.

    jaxreaxff and parse2fit are real pip dependencies now (the `jaxreaxff`
    extra in Ensemble-FF-Fit/pyproject.toml, pulled from
    github.com/Q-CAD/JAX-ReaxFF's own `develop` branch and
    github.com/Q-CAD/parse2fit respectively) -- CHANGED (2026-09) from an
    earlier PYTHONPATH-injection-from-a-local-clone approach (see git
    history if that's ever needed again), once neither package needed
    further active local editing as part of this pipeline's own
    development. So there's no separate jaxreaxff-repo-root/deps-dir
    PYTHONPATH construction needed here anymore -- same shape as
    lammps_container_env() above.

    LD_LIBRARY_PATH mirrors lammps_container_env()'s own fix (prepending
    /usr/lib64, where podman-hpc's hook_tool injects the real NVIDIA
    driver). PREDICTED, not yet confirmed: jax-cuda12-plugin resolves
    libcuda.so.1 at import time the same way Cray MPICH's yaksa GPU-
    datatype engine does for LAMMPS, so the same /usr/local/cuda/compat-
    vs-/usr/lib64 version-mismatch risk likely applies here too -- verify
    against a real `jax.devices()` check, don't trust this blindly.
    """
    from EnsembleFFFit.utilities.general import ensemble_fffit_pythonpath

    return {
        'PYTHONPATH': ensemble_fffit_pythonpath(),
        'LD_LIBRARY_PATH': '/usr/lib64:' + os.environ.get('LD_LIBRARY_PATH', ''),
    }


def _require(cfg, stage_name, keys):
    # cfg is None whenever the whole section is commented out/absent (e.g.
    # copy_force_fields/MD_single_points while fine_tuning is enabled but
    # they aren't) -- report that plainly rather than crashing on cfg.get()
    # with an unhelpful AttributeError.
    if cfg is None:
        raise ValueError(f"Missing required section '{stage_name}' in the config file.")
    missing = [k for k in keys if cfg.get(k) is None]
    if missing:
        raise ValueError(f"Missing required key(s) {missing} under '{stage_name}' in the config file.")


# ---------------------------------------------------------------------------
# Stage: converge_dft_data (Pipeline 0) -- VASP
# ---------------------------------------------------------------------------

def vasp_resource_estimator(structure, input_args, options):
    """
    `resource_estimator` for DFTMatEnsemble.build_dft_dcts, VASP-shaped:
    a plain atoms-per-node node count (see
    logic_locations/VASP/generate_vasp_flux_cli.py's get_num_nodes), not
    RMG's processor-grid search -- this is the concrete case
    build_dft_dcts's resource_estimator parameter exists for. `input_args`
    is the parsed vasp_yaml; 'nodes' there (default 0) triggers this
    estimate, a positive value pins the node count directly, mirroring
    generate_vasp_flux_cli.py's --nodes/--atoms_per_node CLI flags as
    ordinary yaml keys instead of argparse options.
    """
    nodes = input_args.get('nodes', 0)
    if nodes and nodes > 0:
        return nodes
    atoms_per_node = input_args.get('atoms_per_node', options.get('atoms_per_node', 100))
    return max(1, -(-len(structure) // atoms_per_node))  # ceiling division -- same "don't floor to 0" reasoning as RMG's own clamp below


def run_converge_dft_data(config):
    """
    Submit one VASP execution chore per (structure, vasp_yaml) pair found
    under directory, each sized from that task's own computed
    'allocated_nodes' (via vasp_resource_estimator) -- DFT jobs are never
    batched together the way MD/MACE tasks are, one pipe.call per job,
    sized per-job, with no Pipeline.strategy dependency chain (this stage
    doesn't spawn anything reactively -- fit_and_validate's fitting stage
    just expects directory/DFT/training and DFT/validation to already be
    converged by the time it runs).

    `directory` serves as both run_directory and inputs_directory for
    DFTMatEnsemble. build_full_runs anchors its task_dir computation on
    whatever check_files points at (default "dft_recipe") -- safe whether
    directory is a single structure leaf or a parent tree spanning multiple
    structures against one shared recipe, since either way exactly one
    recipe file is expected in the tree. Don't override check_files to
    "structure_filename" for the multiple-structures case: since
    structure_filename would then match once per structure on *both* sides
    of build_full_runs' nested loop, it produces an N^2 cross-product
    instead of N.

    gpus_per_task (config key, default 1 -- matching every other GPU-facing
    stage's own default in this file, e.g. fine_tuning/MD_single_points)
    flows straight into both the chore registration and each per-job
    Resources() below, the same way cores_per_task already does. num_tasks
    itself is still sized from allocated_nodes*cores_per_task (one MPI rank
    per unit of cores_per_task, NOT one rank per GPU) -- this is UNVERIFIED
    against a real Flux-dispatched allocation (only confirmed as a raw,
    single-rank executable so far, see vasp_dft.py's docstring); rely on
    max_tasks_per_job (workflow_config.yaml, currently 4 to match one GPU
    node's worth of A100s) to cap this at something sane regardless of
    what cores_per_task computes, and re-check the whole split the first
    time a real Flux-submitted VASP chore actually runs.
    """
    from matensemble.pipeline import Pipeline
    from matensemble.model import Resources
    from EnsembleFFFit.base import DFTMatEnsemble

    dft_cfg = config['converge_dft_data']
    _require(dft_cfg, 'converge_dft_data', ['directory', 'dft_task'])

    directory = dft_cfg['directory']
    dft_task = str(Path(dft_cfg['dft_task']).resolve())
    structure_filename = dft_cfg.get('structure_filename', 'POSCAR')
    vasp_yaml_name = dft_cfg.get('vasp_yaml_name', 'vdW_single_point.yml')
    entry_point = dft_cfg.get('entry_point', 'run_vasp_calculation')
    check_files = dft_cfg.get('check_files', ['dft_recipe'])
    finished_file = dft_cfg.get('finished_file')
    # Perlmutter CPU node core count (2x AMD EPYC 7763, 64 cores each) -- a
    # default, not a verified-correct value for how VASP should actually be
    # decomposed across a node; override via config once confirmed.
    cores_per_task = dft_cfg.get('cores_per_task', 16)
    # Default 1 (not 0) to match every other GPU-facing stage's own
    # gpus_per_task default in this file -- vasp/6.4.2-gpu is now the only
    # build vdW_single_point.yml points at (see VASP_LD_LIBRARY_PATH), so
    # requesting 0 GPUs here would be the unusual case, not the norm.
    gpus_per_task = dft_cfg.get('gpus_per_task', 1)
    atoms_per_node = dft_cfg.get('atoms_per_node', 100)
    max_tasks_per_job = dft_cfg.get('max_tasks_per_job')

    options = {
        'structure_filename': structure_filename,
        'dft_recipe': vasp_yaml_name,
        'atoms_per_node': atoms_per_node,
    }

    dft = DFTMatEnsemble(directory, directory, **options)
    dft_dct_list = dft.build_dft_dcts(dft_task, check_files, entry_point, finished_file=finished_file,
                                       resource_estimator=vasp_resource_estimator)

    if not dft_dct_list:
        # build_full_runs itself is cheap (just directory walks/proximity matching) --
        # the expensive part (Structure.from_file + resource_estimator) only
        # runs in build_dft_dcts' own loop over whatever survives finished_file
        # filtering. Re-running the cheap half unfiltered here, purely to give a
        # precise error, doesn't meaningfully duplicate the expensive work above.
        recipe_keys, structure_keys = ['dft_recipe'], ['structure_filename']
        labels = check_files + structure_keys + recipe_keys
        _, raw_task_dirs = dft.build_full_runs(
            root0=dft.run_directory, files0=[dft.options[c] for c in check_files],
            root1=dft.inputs_directory, files1=[dft.options[k] for k in structure_keys],
            recipe_files=[dft.options[k] for k in recipe_keys],
            labels=labels, ordered_labels=labels, finished_file=None,
            run_directory=dft.run_directory, inputs_directory=dft.inputs_directory,
        )
        if raw_task_dirs and finished_file is not None:
            sample = raw_task_dirs[:3]
            raise ValueError(
                f"Found {len(raw_task_dirs)} (structure, vasp_yaml) pair(s) under {directory}, but every "
                f"one of them already has a '{finished_file}' match in its working directory -- there's "
                f"nothing new to run. Sample working director{'y' if len(sample) == 1 else 'ies'}: "
                f"{sample}{', ...' if len(raw_task_dirs) > 3 else ''}. Set finished_file to null (or a "
                f"pattern that doesn't already exist everywhere) to rerun them anyway."
            )
        raise ValueError(f"No (structure, vasp_yaml) pairs found under {directory}")

    pipe = Pipeline()
    env = vasp_container_env(cores_per_task)

    # mpi=True -- CONFIRMED (2026-09-04) required: matensemble.fluxlet.Fluxlet.submit
    # only sets the jobspec's own "mpi"="pmi2" shell option when
    # chore.resources.mpi is truthy. Without it (mpi=False, this chore's
    # original value, copied forward from the "bare executable, Flux owns
    # launch semantics" RMG-era convention), Flux launches num_tasks=4
    # separate, UNCOORDINATED copies of this chore function -- each one's
    # own bare `vasp_std` subprocess calls MPI_Init() independently and
    # elects itself a solo rank-0-of-1 job, never joining one shared 4-rank
    # communicator. Confirmed directly from stderr: three separate "running
    # 1 mpi-ranks... 1 GPUs detected" blocks (different PIDs) writing to the
    # same working_directory at once, colliding on an HDF5 file lock
    # ("unable to lock the file... errno = 11") -- not a GPU-affinity/
    # CUDA_VISIBLE_DEVICES problem, a missing MPI-rank-coordination one.
    # VASP's own NPAR/KPAR band/k-point parallelization (in
    # vdW_single_point.yml's user_incar_settings) assumes exactly this
    # coordinated-multi-rank model -- "bare executable" was never actually
    # correct for VASP specifically, unlike RMG's own precedent this was
    # copied from (RMG apparently never exercised num_tasks>1 in a way that
    # surfaced this).
    # cores_per_task=cores_per_task (the config value, e.g. 16), not
    # hardcoded 1 -- CONFIRMED (2026-09-04) the hardcoded 1 was a real
    # resource-accounting bug once mpi=True made this a genuinely
    # coordinated multi-rank job: OMP_NUM_THREADS (vasp_container_env,
    # above) is already set to this same config value, so VASP will try to
    # use cores_per_task OMP threads per rank regardless of what Flux
    # itself thinks it allocated -- Resources(cores_per_task=1, ...) told
    # Flux each rank only needed 1 core while VASP actually tried to use
    # 16, a 64x undercount for this job's real CPU footprint (4 ranks x 16
    # threads = 64, the whole node, not coincidentally, since NPAR=16 in
    # vdW_single_point.yml already assumes exactly this split). FRAGILE:
    # this only stays correct because cores_per_task (16) x
    # max_tasks_per_job (4, workflow_config.yaml) == 64 (this node type's
    # physical core count) -- nothing derives one from the other
    # automatically, so changing max_tasks_per_job without also adjusting
    # cores_per_task (or vice versa) would silently reintroduce a mismatch.
    @pipe.chore(name="run_vasp", num_tasks=1, cores_per_task=cores_per_task, gpus_per_task=gpus_per_task,
                env=env, inherit_env=True, mpi=True)
    def run_vasp_chore(task_dict):
        """Thin chore wrapper -- the real VASP-execution logic lives in DFTMatEnsemble.run_individual."""
        return DFTMatEnsemble.run_individual(task_dict)

    for dft_dct in dft_dct_list:
        # dft_dct already carries 'dft_task'/'entry_point' -- embedded by
        # build_dft_dcts itself now, not injected here (see FFMatEnsemble's
        # build_ff_dcts for the same convention). Copied (not aliased) since
        # 'allocated_nodes' may get overridden below, per-iteration.
        task_dict = dict(dft_dct)

        # Sized per-job from this task's own computed allocated_nodes, not a
        # single global value -- different structures/recipes can legitimately
        # need different node counts, and each pipe.call carries its own
        # Resources override for exactly this reason. One MPI rank per core
        # (the common CPU-VASP decomposition) -- UNVERIFIED, see this
        # function's docstring.
        num_tasks = dft_dct['allocated_nodes'] * cores_per_task
        if max_tasks_per_job is not None and num_tasks > max_tasks_per_job:
            # Same ceiling-division clamp as the RMG precedent (floor gave 0
            # whenever max_tasks_per_job < cores_per_task, an actual bug:
            # Resources(num_tasks=0, ...) is invalid).
            clamped_num_tasks = max_tasks_per_job
            clamped_allocated_nodes = max(1, -(-clamped_num_tasks // cores_per_task))  # ceil division
            print(f"WARNING: {task_dict['structure_filename']}: auto-computed "
                  f"allocated_nodes*cores_per_task = {num_tasks} tasks exceeds "
                  f"max_tasks_per_job={max_tasks_per_job}; falling back to {clamped_num_tasks} "
                  f"task(s) ({clamped_allocated_nodes} node(s) worth) instead of the auto-computed value.",
                  file=sys.stderr)
            num_tasks = clamped_num_tasks
            task_dict['allocated_nodes'] = clamped_allocated_nodes
        resources = Resources(num_tasks=num_tasks, cores_per_task=cores_per_task, gpus_per_task=gpus_per_task,
                               env=env, inherit_env=True, mpi=True)
        pipe.call("run_vasp", task_dict, resources=resources)

    # set_cpu_affinity=False (matensemble's own default is True, unset
    # anywhere else in this file until now): CONFIRMED (2026-09-03) this
    # default is what caused "flux-shell[0]: ERROR: cpu-affinity: affinity:
    # core124 not in topology" -- matensemble.fluxlet sets shell option
    # cpu-affinity="per-task" whenever True, and THAT flux-shell plugin does
    # its own live topology probe at task-launch time, entirely separate
    # from launch_multi_node.slurm's R.json/resource.toml (noverify=true
    # only bypasses the BROKER's own resource-module verification, not this
    # per-task shell plugin) -- so no R.json fix could ever have addressed
    # this. Since R.json/noverify=true already means we don't trust this
    # container's live-probed topology for scheduling in the first place,
    # disabling the live-probe-dependent pinning optimization is more
    # robust than chasing whatever core count would make it agree -- this
    # only affects CPU pinning as a performance optimization, not which
    # cores get allocated (the scheduler's own R.json-based decision is
    # unaffected). set_gpu_affinity stays True: unlike CPU cores, GPU
    # assignment via CUDA_VISIBLE_DEVICES is load-bearing for correctness
    # here (multiple GPU-bound tasks per node need distinct GPUs, not just
    # a performance pin), and it hasn't shown this failure mode.
    future = pipe.submit(log_delay=25, set_gpu_affinity=True, set_cpu_affinity=False)
    future.result()


# ---------------------------------------------------------------------------
# Stage: parse2fit_generation
# ---------------------------------------------------------------------------

def run_parse2fit_generation(config):
    """
    Runs parse2fit's own input-generation step in-process -- see
    JaxReaxFF_Integration_Plan.md Stage 3c. Converting DFT-converged
    structures into ReaxFF's geo/trainset.in format was explicitly out of
    scope for run_build_ff_inputs (see that function's own docstring:
    "parse2fit's job... explicitly out of scope for this pass") -- this is
    that job, now wired in.

    yaml_directive is a separate, user-provided parse2fit YAML directive
    file -- its own format, specific to parse2fit itself, not this
    pipeline's workflow_config.yaml convention (see parse2fit's own
    README/examples for its shape). It is NOT generated or templated by
    this pipeline -- how a given DFT dataset's directories/energy
    references/weights get expressed is inherently dataset-specific,
    same reasoning as why fine_tuning's own reaxff_inputs aren't
    templated here either.

    Every path parse2fit's own readwrite.py resolves (directories/
    subtract/add) is required to already be ABSOLUTE in that directive
    file -- confirmed directly (2026-09-09) that
    parse2fit.io.readwrite.RW.__init__ unconditionally calls
    self._get_absolute_paths() immediately at
    ReadWriteFactory(path).get_writer() time (before write_input_files()
    even runs), which does os.path.abspath() on every one of those paths
    -- resolved against whatever this process's cwd happens to be at that
    exact moment. A relative path in the directive file would silently
    resolve wrong depending on where run_pipeline.py was launched from;
    baking absolute paths into the directive file itself sidesteps that
    entirely, by design (deliberately not "fixed" here via os.chdir() --
    see JaxReaxFF_Integration_Plan.md's own sign-off on this point).

    Plain CPU-only stage, not a Flux chore -- parse2fit's own work here
    (parsing already-completed vasprun.xml-derived POSCAR/properties.json
    and writing geo/trainset.in variants) is pure Python file parsing/
    generation, no GPU or multi-node coordination needed, matching
    run_build_ff_inputs/run_rank_reaxff_validation's own convention rather
    than fit_reaxff/run_vasp's.

    Writes runs_to_generate geo/trainset.in variant folders (named
    f"{output_format}_run_{i}" by parse2fit's own readwrite.py) under
    whatever output_directory the directive file itself specifies.
    Combining these with build_reaxff_ensemble_inputs.py's own
    params-blocking-scheme variants is deliberately NOT done here --
    separate follow-up work, not yet designed.
    """
    cfg = config['parse2fit_generation']
    _require(cfg, 'parse2fit_generation', ['yaml_directive'])

    # parse2fit is a real pip dependency now (Ensemble-FF-Fit's own
    # `jaxreaxff` extra, pulled from github.com/Q-CAD/parse2fit) -- CHANGED
    # (2026-09) from an earlier sys.path-injection-from-a-local-clone
    # approach (see git history if that's ever needed again), once it no
    # longer needed further active local editing as part of this
    # pipeline's own development.
    from parse2fit.io.readwrite import ReadWriteFactory

    yaml_directive = str(Path(cfg['yaml_directive']).resolve())
    rw = ReadWriteFactory(yaml_directive).get_writer()
    rw.write_input_files()


# ---------------------------------------------------------------------------
# Stage: sample_ft_md_structures
# ---------------------------------------------------------------------------

def _find_structures(root, structure_filename="POSCAR"):
    """Every leaf directory under root containing structure_filename."""
    paths = []
    for dirpath, _, files in os.walk(root):
        if structure_filename in files:
            paths.append(dirpath)
    return paths


def _group_by_mpid(paths):
    """{mpid: [paths]} for every path containing an 'mp-<digits>' component
    (matched anywhere in the path, so it groups across compositions/defect
    types that happen to share the same mp-id -- each mp-id belongs to
    exactly one composition here, so this also spreads across compositions)."""
    groups = defaultdict(list)
    for p in paths:
        match = re.search(r"mp-\d+", p)
        groups[match.group(0) if match else "no_mpid"].append(p)
    return groups


def _sample_evenly_by_mpid(root, total, structure_filename="POSCAR", seed=None):
    """
    Sample `total` structure directories from `root`, spread as evenly as
    possible across the distinct mp-ids found. Each mp-id gets
    total // n_mpids structures, with the remainder (total % n_mpids)
    distributed one-at-a-time to the first few mp-ids in sorted order (for
    reproducibility) -- so a total that doesn't divide evenly still adds up
    to exactly `total`, not silently more or fewer.
    """
    rng = random.Random(seed)
    groups = _group_by_mpid(_find_structures(root, structure_filename))
    mpids = sorted(groups.keys())
    n_mpids = len(mpids)
    if n_mpids == 0:
        return []

    base, remainder = divmod(total, n_mpids)

    selected = []
    for i, mpid in enumerate(mpids):
        n_here = min(base + (1 if i < remainder else 0), len(groups[mpid]))
        selected.extend(rng.sample(groups[mpid], n_here))

    return selected


def _copy_starting_structures(paths, source_root, dest_root, structure_filename="POSCAR",
                               convert_to_lmp=True, supercell_kwargs=None):
    """Copy structure_filename from each of `paths` into dest_root, mirroring
    each path's position relative to source_root.

    convert_to_lmp (default True): writes a LAMMPS data file (structure.lmp,
    via EnsembleFFFit.molecular_dynamics.lammps.poscar_to_structure_lmp)
    instead of a plain copy of structure_filename -- REQUIRED, not a plain
    copy, because finite_temperature_md.structure_filename is
    "structure.lmp" (LAMMPS's own read_data needs its own data format, not
    a raw POSCAR/CONTCAR), same reasoning as MD_single_points/
    MD_uq_single_points' own structure.lmp conversion. This was a real,
    previously un-exercised gap: this stage has never actually been run
    for finite_temperature_md before now -- every FT-MD test so far used a
    structure.lmp staged by hand, bypassing this stage entirely, so a
    plain POSCAR copy here was never actually caught. Set False only if a
    future finite_temperature_md setup genuinely wants a non-LAMMPS
    structure_filename copied as-is.

    supercell_kwargs (optional dict, e.g. {"min_atoms": 200, "min_length":
    15.0}): if given, applies pymatgen's CubicSupercellTransformation to
    each structure BEFORE writing structure.lmp -- required for the
    finite-temperature coordination-stability check (see
    run_sample_ft_md_structures' own docstring): the small DFT unit cells
    under DFT/training artificially freeze coordination-relevant motion
    during NPT thermalization, so FT-MD needs to run on a reasonably-sized
    supercell of each DFT structure, not the bare cell itself. Ignored
    when convert_to_lmp is False (a supercell only makes sense on the way
    to a structure.lmp; a plain POSCAR copy is left untouched).
    """
    from EnsembleFFFit.molecular_dynamics.lammps.poscar_to_structure_lmp import poscar_to_structure_lmp, structure_to_lmp

    copied = []
    for path in paths:
        rel = os.path.relpath(path, source_root)
        dest_dir = os.path.join(dest_root, rel)
        os.makedirs(dest_dir, exist_ok=True)
        source_file = os.path.join(path, structure_filename)
        if convert_to_lmp and supercell_kwargs:
            from pymatgen.core.structure import Structure
            from pymatgen.transformations.advanced_transformations import CubicSupercellTransformation
            structure = Structure.from_file(source_file)
            supercell = CubicSupercellTransformation(**supercell_kwargs).apply_transformation(structure)
            print(f"  {path}: {len(structure)} atoms -> supercell {len(supercell)} atoms")
            structure_to_lmp(supercell, os.path.join(dest_dir, 'structure.lmp'))
        elif convert_to_lmp:
            poscar_to_structure_lmp(source_file, os.path.join(dest_dir, 'structure.lmp'))
        else:
            shutil.copy2(source_file, os.path.join(dest_dir, structure_filename))
        copied.append(dest_dir)
    return copied


def run_sample_ft_md_structures(config):
    """
    Selects starting structures for finite-temperature MD with the
    foundation model, either (a) a fixed total number sampled from
    converge_dft_data's training subtree, spread as evenly as possible
    across distinct mp-ids (the active-learning default -- representative
    spread matters more than which exact structures), or (b) an explicit
    named_paths list (the ReaxFF coordination-environment-stability setup
    -- comparing FT-MD's starting vs. final structure only means anything
    for specific, chosen structures, not a random spread). Ported from the
    standalone sample_training_structures.py script into a proper stage,
    matching the same single-entry-point consolidation as every other
    stage here -- still deliberately its own stage/function (not folded
    into fit_and_validate or moved into EnsembleFFFit), since which
    structures feed finite-temperature MD is exactly the kind of
    site-specific selection logic a user should be able to read and adjust
    directly.

    source_root defaults to converge_dft_data.directory/training when null.
    dest_root defaults to finite_temperature_md.lammps_inputs_directory/structures
    when null -- structures live inside lammps_inputs_directory (the same
    convention MD_single_points/MD_uq_single_points use: the driver script
    and its structures share one inputs_directory), not in a separately
    tracked directory, so there's one place (not two) to point at each.

    Selected structures are written as structure.lmp (LAMMPS data format),
    not a plain copy of structure_filename -- see _copy_starting_structures'
    own docstring for why this is required, not optional, and why it was a
    real, previously un-exercised gap (this stage has never actually been
    run for finite_temperature_md before now).

    supercell_kwargs (optional, e.g. {min_atoms: 200, min_length: 15.0}):
    passed straight through to _copy_starting_structures, which applies
    pymatgen's CubicSupercellTransformation before writing structure.lmp.
    Needed specifically for the coordination-stability named_paths case
    above: running NPT MD on the bare DFT unit cell can artificially
    freeze coordination-relevant motion during thermalization (too few
    atoms/too small a periodic image), so a reasonably-sized, regular
    supercell of each named structure is used instead. Not relevant to
    the active-learning random-sampling case (there, MD_single_points
    already runs directly against DFT-sized single points, no FT-MD
    thermalization involved).
    """
    cfg = config.get('sample_ft_md_structures', {})

    source_root = cfg.get('source_root')
    if not source_root:
        _require(config['converge_dft_data'], 'converge_dft_data', ['directory'])
        source_root = os.path.join(config['converge_dft_data']['directory'], cfg.get('training_subpath', 'training'))

    dest_root = cfg.get('dest_root')
    if not dest_root:
        ft_md_cfg = config['finite_temperature_md']
        dest_root = os.path.join(ft_md_cfg['lammps_inputs_directory'], ft_md_cfg.get('structures_subpath', 'structures'))
    structure_filename = cfg.get('structure_filename', 'POSCAR')
    convert_to_lmp = cfg.get('convert_to_lmp', True)
    supercell_kwargs = cfg.get('supercell_kwargs')

    named_paths = cfg.get('named_paths')
    if named_paths:
        selected = [str(Path(p).resolve()) for p in named_paths]
        missing = [p for p in selected if not os.path.isfile(os.path.join(p, structure_filename))]
        if missing:
            raise FileNotFoundError(
                f"named_paths entries missing a {structure_filename}: {missing}"
            )
        print(f"Using {len(selected)} explicitly named structure(s) (named_paths), skipping random sampling")
    else:
        total = cfg.get('total', 20)
        seed = cfg.get('seed', 0)
        selected = _sample_evenly_by_mpid(source_root, total, structure_filename=structure_filename, seed=seed)
        print(f"Sampled {len(selected)} structure(s) across mp-ids under {source_root}")

    copied = _copy_starting_structures(selected, source_root, dest_root,
                                        structure_filename=structure_filename,
                                        convert_to_lmp=convert_to_lmp,
                                        supercell_kwargs=supercell_kwargs)
    for d in copied:
        print(f"  -> {d}")


# ---------------------------------------------------------------------------
# Stage: build_ff_inputs -- JAX-ReaxFF
# ---------------------------------------------------------------------------

def run_build_ff_inputs(config):
    """
    Builds the ensemble of JAX-ReaxFF fitting-input folders -- one per
    (parse2fit-generated geo/trainset.in variant, blocking-scheme label)
    combination, a full cross product (see
    JaxReaxFF_Integration_Plan.md's sign-off on this over e.g. a 1:1
    pairing) -- that fit_and_validate reads from, via
    FF/build_reaxff_ensemble_inputs.build_reaxff_ensemble_inputs.

    parse2fit_root points at run_parse2fit_generation's own output (that
    earlier stage's job -- converting DFT-converged structures into
    ReaxFF's geo/trainset.in format -- is not duplicated here). Only
    `params`' per-member variation, plus this cross-product/combination
    step, is generated here -- see build_reaxff_ensemble_inputs.py's own
    docstring for the full design rationale (recovered from git history:
    no automated version of this existed before this pipeline) and for
    why `ffield` (the shared seed force field itself, as opposed to
    `params`) is handled separately, via fine_tuning.run_directory, not
    here -- it's used below only to READ current parameter values for
    bound-widening, never copied anywhere by this stage.

    ffield (optional) enables that bound-widening against the seed force
    field's own actual current values -- needs jax/jax_md/jaxreaxff
    importable, same as fit_reaxff chores. jaxreaxff is a real pip
    dependency now (Ensemble-FF-Fit's own `jaxreaxff` extra), so no
    sys.path injection is needed here anymore -- CHANGED (2026-09) from an
    earlier approach that sys.path-injected a local JAX-ReaxFF clone (see
    git history if that's ever needed again).

    build_reaxff_ensemble_inputs.py is pipeline-local (FF/), not an
    installed EnsembleFFFit module -- loaded via import_module_from_path,
    same mechanism the driver scripts use, rather than a plain import.
    """
    from EnsembleFFFit.utilities.general import import_module_from_path

    cfg = config['build_ff_inputs']
    _require(cfg, 'build_ff_inputs', ['parse2fit_root', 'catalog_params', 'blocking_scheme'])

    reaxff_inputs_dir = Path(cfg.get('reaxff_inputs_dir') or config['fine_tuning']['inputs_directory'])
    ffield_path = cfg.get('ffield')

    ensemble_module = import_module_from_path(
        'build_reaxff_ensemble_inputs',
        str(Path(__file__).parent / 'FF' / 'build_reaxff_ensemble_inputs.py'),
    )

    written_folders = ensemble_module.build_reaxff_ensemble_inputs(
        parse2fit_root=cfg['parse2fit_root'],
        catalog_params_path=cfg['catalog_params'],
        blocking_scheme=cfg['blocking_scheme'],
        output_dir=reaxff_inputs_dir,
        ffield_path=ffield_path,
        unbounded_half_width=cfg.get('unbounded_half_width', 1e4),
    )
    print(f"Wrote {len(written_folders)} reaxff_inputs folder(s) under {reaxff_inputs_dir}")


# ---------------------------------------------------------------------------
# Stage: fit_and_validate (Pipeline 1)
# ---------------------------------------------------------------------------

def run_fit_and_validate(config):
    """
    Builds and submits a Pipeline with up to two parts that are NOT
    dependencies of one another -- they're only ever registered on the same
    Pipeline object because a Pipeline supports at most one
    Pipeline.strategy() registration at a time:

    - fine_tuning (if configured): JAX-ReaxFF ensemble fitting (fit_reaxff),
      one chore per (parse2fit variant, blocking-scheme label) combination
      built by build_ff_inputs. Validation against DFT ground truth happens
      afterward, as its own separate stage (reaxff_validation_single_points/
      rank_reaxff_validation), not reactively per completed fit here -- an
      earlier reactive per-fit validation-spawn design (copy_force_fields/
      MD_single_points/validation_mirroring) was removed (2026-09) once the
      flat reaxff_validation_* stage family fully superseded it.
    - finite_temperature_md (if configured): a flat, upfront batch of
      finite-temperature MD chores (run_ft_md), independent of any fit
      result -- submitted unconditionally, not gated on fit_reaxff at all.
      (This is a separate, disabled-by-default one-model smoke test, not
      the active finite_temperature_md_batch stage that runs the full
      fitted ensemble -- see that stage's own docstring.)

    fine_tuning and finite_temperature_md are independently optional
    (either, both, or neither... though at least one is required) so
    finite_temperature_md can be run/tested on its own: `--stage
    fit_and_validate` with only finite_temperature_md configured submits
    ONLY run_ft_md chores, no fit_reaxff chores at all. This is more than a
    config nicety -- FFMatEnsemble.build_ff_dcts (via
    _make_proximity_combinations) raises FileNotFoundError the moment
    fine_tuning.run_directory/inputs_directory don't already contain real,
    matching content, so without this `if fine_tuning_cfg:` gate, leaving
    fine_tuning unstaged would crash before finite_temperature_md's own
    (already fully independent) submission code ever ran.
    """
    from matensemble.pipeline import Pipeline
    from matensemble.model import Resources
    from EnsembleFFFit.base import FFMatEnsemble, MDMatEnsemble

    fine_tuning_cfg = config.get('fine_tuning')
    ft_md_cfg = config.get('finite_temperature_md')

    if not fine_tuning_cfg and not ft_md_cfg:
        raise ValueError(
            "run_fit_and_validate: neither 'fine_tuning' nor 'finite_temperature_md' is "
            "configured -- nothing for this stage to do."
        )

    if fine_tuning_cfg:
        _require(fine_tuning_cfg, 'fine_tuning', ['run_directory', 'inputs_directory', 'ff_task'])
    if ft_md_cfg:
        _require(ft_md_cfg, 'finite_temperature_md',
                  ['output_directory', 'foundation_model', 'in_file', 'lammps_inputs_directory'])

    pipe = Pipeline()

    if fine_tuning_cfg:
        ff_run_directory = fine_tuning_cfg['run_directory']

        @pipe.chore(name="fit_reaxff",
                    num_tasks=fine_tuning_cfg.get('num_tasks', 1),
                    cores_per_task=fine_tuning_cfg.get('cores_per_task', 1),
                    gpus_per_task=fine_tuning_cfg.get('gpus_per_task', 1),
                    env=jax_reaxff_container_env(), inherit_env=True)
        def fit_reaxff_chore(task_dict):
            """Thin chore wrapper -- the real fitting logic lives in whatever driver script
            fine_tuning.ff_task points at (see FFMatEnsemble.run_individual)."""
            return FFMatEnsemble.run_individual(task_dict)

        # Keys deliberately match driver.py's own argparse attribute names
        # (init_FF/params/geo/train_file), not a friendlier renamed set -- see
        # jax_reaxff_fit.py's docstring for why: FFMatEnsemble.run_individual
        # hands this whole dict to jax_reaxff_fit.py's run_jax_reaxff_fit as
        # `overrides`, which setattr()s each key straight onto the parsed args
        # Namespace, so the keys have to already be real attribute names.
        fine_tuning_options = {'init_FF': fine_tuning_cfg.get('init_FF', 'ffield'),
                               'params': fine_tuning_cfg.get('params', 'params'),
                               'geo': fine_tuning_cfg.get('geo', 'geo'),
                               'train_file': fine_tuning_cfg.get('train_file', 'trainset.in')}
        fine_tuning_options = {k: v for k, v in fine_tuning_options.items() if v is not None}

        ff_task = str(Path(fine_tuning_cfg['ff_task']).resolve())
        ff_entry_point = fine_tuning_cfg.get('entry_point', 'run_jax_reaxff_fit')

        ff_matensemble = FFMatEnsemble(ff_run_directory, fine_tuning_cfg['inputs_directory'], **fine_tuning_options)
        check_files = fine_tuning_cfg.get('check_files', ['init_FF'])
        # Deliberately finished_file=None here: every variant must get a
        # fit_reaxff chore submitted, even already-finished ones -- the
        # skip-if-already-fit check happens at execution time inside the
        # driver script (jax_reaxff_fit.py's run_jax_reaxff_fit) via
        # FFMatEnsemble.run_individual instead.
        reaxff_arg_dict_list = ff_matensemble.build_ff_dcts(ff_task, check_files, ff_entry_point, finished_file=None)
        ff_finished_file = fine_tuning_cfg.get('finished_file')
        if ff_finished_file:
            for overrides in reaxff_arg_dict_list:
                overrides['finished_file'] = ff_finished_file

        fitting_resources = Resources(num_tasks=fine_tuning_cfg.get('num_tasks', 1),
                                       cores_per_task=fine_tuning_cfg.get('cores_per_task', 1),
                                       gpus_per_task=fine_tuning_cfg.get('gpus_per_task', 1),
                                       env=jax_reaxff_container_env(),
                                       inherit_env=True)

        for reaxff_inputs in reaxff_arg_dict_list:
            pipe.call("fit_reaxff", reaxff_inputs, resources=fitting_resources)

    if ft_md_cfg:
        ft_md_resources_kwargs = dict(num_tasks=ft_md_cfg.get('num_tasks', 1),
                                       cores_per_task=ft_md_cfg.get('cores_per_task', 1),
                                       gpus_per_task=ft_md_cfg.get('gpus_per_task', 1))

        @pipe.chore(name="run_ft_md", **ft_md_resources_kwargs,
                    env=lammps_container_env(), inherit_env=True)
        def run_ft_md_chore(task_dict):
            """Thin chore wrapper -- the real FT-MD execution logic lives in MDMatEnsemble.run_individual."""
            return MDMatEnsemble.run_individual(task_dict)

        ft_md_task_command = os.path.abspath(
            os.path.join(ft_md_cfg['lammps_inputs_directory'], ft_md_cfg.get('lammps_task', 'lammps_reaxff_md.py')))
        # Structures live inside lammps_inputs_directory (structures_subpath,
        # default "structures") -- the same inputs_directory convention
        # MD_single_points/MD_uq_single_points use -- not a separately
        # tracked structures_directory.
        ft_md_structures_root = os.path.join(
            ft_md_cfg['lammps_inputs_directory'], ft_md_cfg.get('structures_subpath', 'structures'))
        ft_md_task_dicts = MDMatEnsemble.build_flat_task_dicts(
            structures_root=ft_md_structures_root,
            foundation_model=ft_md_cfg['foundation_model'],
            in_file=ft_md_cfg['in_file'],
            output_root=ft_md_cfg['output_directory'],
            task_command=ft_md_task_command,
            entry_point=ft_md_cfg.get('entry_point', 'run_finite_temperature_md'),
            structure_filename=ft_md_cfg.get('structure_filename', 'POSCAR'),
            finished_file=ft_md_cfg.get('finished_file'),
        )
        if not ft_md_task_dicts:
            raise ValueError(
                f"finite_temperature_md is configured but no structures were found under "
                f"{ft_md_structures_root} -- did you run the sample_ft_md_structures stage first? "
                f"(Silently submitting zero run_ft_md chores here previously masked exactly this: an "
                f"empty structures directory produced no error, just no finite-temperature MD.)"
            )
        ft_md_resources = Resources(**ft_md_resources_kwargs,
                                     env=lammps_container_env(), inherit_env=True)
        for task_dict in ft_md_task_dicts:
            pipe.call("run_ft_md", task_dict, resources=ft_md_resources)

    # set_cpu_affinity=False -- see run_converge_dft_data's pipe.submit for why
    # (matensemble's own live per-task topology probe, unrelated to R.json).
    future = pipe.submit(log_delay=25, set_gpu_affinity=True, set_cpu_affinity=False)
    future.result()


# ---------------------------------------------------------------------------
# Stage: prepare_reaxff_validation_structures
# ---------------------------------------------------------------------------

def run_prepare_reaxff_validation_structures(config):
    """
    Converts every POSCAR under DFT/validation and DFT/training into a
    mirrored structure.lmp (LAMMPS data format) tree -- needed because
    in.single_point's own `read_data ${structure}` requires LAMMPS's own
    data format, not pymatgen's POSCAR format, same reasoning as every
    other structure.lmp conversion in this pipeline (MD_single_points/
    MD_uq_single_points/sample_ft_md_structures). Uses
    EnsembleFFFit.utilities.general.copy_and_transform_files with
    poscar_to_structure_lmp as the transform, same mechanism used to build
    MD_uq_single_points' own structures/ tree earlier in this pipeline.

    ALSO copies the LAMMPS recipe files (in.single_point/control/
    lammps_reaxff_single_point.py, from lammps_recipe_source_dir -- the
    canonical copy lives at MD/single_points/reaxff_validation/
    lammps_inputs) into each *_dest_root's own PARENT directory (not the
    structures/ subtree itself) -- CONFIRMED necessary, not optional: MDMatEnsemble.
    build_task_dicts resolves the driver script via a bare
    `os.path.abspath(os.path.join(self.inputs_directory, lammps_task))`
    (base.py:323), raising ValueError if that exact file doesn't exist --
    no searching/proximity-matching involved -- and
    lammps_reaxff_single_point.py itself discovers `control` as a fixed
    sibling of `in_file` in ITS OWN directory (see that driver's own
    docstring), so copying the script alone without its two sibling files
    into the same new location would still fail once a chore actually
    tries to run it.

    Kept as its own separate stage (not folded into
    reaxff_validation_single_points) since it only needs to run once --
    DFT/validation and DFT/training don't change between separate fitting
    rounds -- and is pure file conversion, not a Flux chore.
    """
    from EnsembleFFFit.utilities.general import copy_and_transform_files, make_mirrored_rename_dest_path_fn
    from EnsembleFFFit.molecular_dynamics.lammps.poscar_to_structure_lmp import poscar_to_structure_lmp

    cfg = config['prepare_reaxff_validation_structures']
    _require(cfg, 'prepare_reaxff_validation_structures',
              ['validation_source_root', 'validation_dest_root',
               'training_source_root', 'training_dest_root',
               'lammps_recipe_source_dir'])

    recipe_source_dir = Path(cfg['lammps_recipe_source_dir'])
    recipe_files = cfg.get('lammps_recipe_files',
                            ['in.single_point', 'control', 'lammps_reaxff_single_point.py'])

    for label, source_root, dest_root in (
        ('validation', cfg['validation_source_root'], cfg['validation_dest_root']),
        ('training', cfg['training_source_root'], cfg['training_dest_root']),
    ):
        dest_path_fn = make_mirrored_rename_dest_path_fn(source_root, dest_root, 'structure.lmp')
        print(f"Converting POSCAR -> structure.lmp ({label}): {source_root} -> {dest_root}")
        copy_and_transform_files(source_root, dest_root, pattern='POSCAR',
                                  dest_path_fn=dest_path_fn, transform_fn=poscar_to_structure_lmp)

        # dest_root is .../lammps_inputs/structures -- recipe files belong
        # one level up, alongside (not inside) the structures/ subtree,
        # matching MD_uq_single_points' own inputs_directory convention.
        recipe_dest_dir = Path(dest_root).parent
        recipe_dest_dir.mkdir(parents=True, exist_ok=True)
        for name in recipe_files:
            src = recipe_source_dir / name
            if not src.is_file():
                raise FileNotFoundError(f"lammps_recipe_source_dir is missing {name}: {src}")
            shutil.copy2(src, recipe_dest_dir / name)
        print(f"Copied {len(recipe_files)} LAMMPS recipe file(s) from {recipe_source_dir} into {recipe_dest_dir}")


# ---------------------------------------------------------------------------
# Stage: stage_reaxff_validation_force_fields
# ---------------------------------------------------------------------------

def _stage_force_fields(source_root, dest_roots, target_name='ffield'):
    """
    Copies+renames every already-fitted ReaxFF force field
    (new_FF_<unique_id>_<loss_str>, excluding *.txt reports) from
    source_root into every listed dest_roots (plural -- each caller's
    downstream stage needs its OWN run_directory tree, even when the
    staged ffield content is identical across all of them: MDMatEnsemble
    writes each run's properties.json back into the same run_directory
    the ffield came from, so sharing one run_directory across multiple
    downstream stages would silently collide/overwrite), each mirrored as
    <same subtree>/target_name -- needed so MDMatEnsemble's own
    check_files-anchored proximity matching can find them. Shared by
    run_stage_reaxff_validation_force_fields and
    run_stage_ft_md_force_fields (see each for its own source/dest
    config) -- a flat, one-time pass over every already-completed fit,
    not reactive per fit_reaxff completion (see
    JaxReaxFF_Integration_Plan.md's own sign-off on flat over reactive for
    this kind of validation step).
    """
    source_root = Path(source_root)
    dest_roots = [Path(d) for d in dest_roots]

    copied = []
    skipped = []
    for results_dir in sorted(p for p in source_root.rglob('*') if p.is_dir()):
        matches = sorted(p for p in results_dir.glob("new_FF_*") if not p.name.endswith('.txt'))
        if len(matches) != 1:
            if matches:
                skipped.append((results_dir, len(matches)))
            continue
        rel = results_dir.relative_to(source_root)
        for dest_root in dest_roots:
            dest_path = dest_root / rel / target_name
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(matches[0], dest_path)
            copied.append(dest_path)

    print(f"Staged {len(copied)} fitted-force-field copy(ies) from {source_root} into {dest_roots}")
    if skipped:
        print(f"Skipped {len(skipped)} directory(ies) with != 1 new_FF_* match (ambiguous, num_trials>1?):")
        for d, n in skipped:
            print(f"  {d}: {n} matches")


def run_stage_reaxff_validation_force_fields(config):
    """See _stage_force_fields. Stages fine_tuning.run_directory's fitted
    force fields into reaxff_validation_single_points' aimd/training
    force_fields trees."""
    cfg = config['reaxff_validation_force_fields']
    _require(cfg, 'reaxff_validation_force_fields', ['source_root', 'dest_roots'])
    _stage_force_fields(cfg['source_root'], cfg['dest_roots'], cfg.get('target_name', 'ffield'))


def run_stage_ft_md_force_fields(config):
    """
    See _stage_force_fields. Stages fine_tuning.run_directory's fitted
    force fields (all of them, by explicit choice -- confirmed 2026-09:
    running the finite-temperature coordination-stability check across
    every fitted candidate, not just rank_reaxff_validation's top picks,
    catches a motif breakdown that the static single-point ranking might
    miss) into finite_temperature_md_batch's own force_fields tree.
    """
    cfg = config['ft_md_force_fields']
    _require(cfg, 'ft_md_force_fields', ['source_root', 'dest_roots'])
    _stage_force_fields(cfg['source_root'], cfg['dest_roots'], cfg.get('target_name', 'ffield'))


# ---------------------------------------------------------------------------
# Stage: reaxff_validation_single_points
# ---------------------------------------------------------------------------

def _submit_reaxff_single_points_batch(sub_cfg, chore_name):
    """
    Shared submission logic for one (validation or training) LAMMPS
    single-points batch -- flat, non-reactive: every staged force field x
    every structure under that batch's own inputs_directory, batched via
    an intentionally oversized parent_levels so MDMatEnsemble.
    batch_by_parent's own merge-child-paths step collapses everything
    under one force field into a single chore regardless of the
    structures tree's own nesting depth (very different between
    validation's AIMD trajectory tree and training's much more
    heterogeneous one -- a single fixed depth, like MD_uq_single_points'
    own parent_levels=8, can't describe both at once).
    """
    from matensemble.pipeline import Pipeline
    from matensemble.model import Resources
    from EnsembleFFFit.base import MDMatEnsemble

    _require(sub_cfg, chore_name, ['run_directory', 'inputs_directory'])

    options = {'ffield': sub_cfg.get('ffield', 'ffield'),
               'in_file': sub_cfg.get('in_file'),
               'structure': sub_cfg.get('structure', 'structure.lmp'),
               'lammps_task': sub_cfg.get('lammps_task', 'lammps_reaxff_single_point.py')}
    options = {k: v for k, v in options.items() if v is not None}

    resources_kwargs = dict(num_tasks=sub_cfg.get('num_tasks', 1),
                             cores_per_task=sub_cfg.get('cores_per_task', 1),
                             gpus_per_task=sub_cfg.get('gpus_per_task', 1))

    md = MDMatEnsemble(sub_cfg['run_directory'], sub_cfg['inputs_directory'], **options)
    task_dicts = md.build_task_dicts(
        sub_cfg.get('lammps_task', 'lammps_reaxff_single_point.py'),
        sub_cfg.get('parent_levels', 20),
        sub_cfg.get('check_files', ['ffield']),
        sub_cfg.get('entry_point', 'run_lammps_single_points'),
        finished_file=sub_cfg.get('finished_file'),
    )
    if not task_dicts:
        raise ValueError(f"No (force field, structure) batches found for {chore_name} -- check "
                          f"run_directory/inputs_directory.")

    pipe = Pipeline()

    @pipe.chore(name=chore_name, **resources_kwargs, env=lammps_container_env(), inherit_env=True)
    def chore_fn(task_dict):
        """Thin chore wrapper -- the real MD-execution logic lives in MDMatEnsemble.run_individual."""
        return MDMatEnsemble.run_individual(task_dict)

    resources = Resources(**resources_kwargs, env=lammps_container_env(), inherit_env=True)
    for task_dict in task_dicts:
        pipe.call(chore_name, task_dict, resources=resources)

    future = pipe.submit(log_delay=25, set_gpu_affinity=True, set_cpu_affinity=False)
    future.result()
    print(f"{chore_name}: submitted {len(task_dicts)} chore(s)")


def run_reaxff_validation_single_points(config):
    """
    Runs LAMMPS single points for every staged force field (see
    stage_reaxff_validation_force_fields) x every structure.lmp under both
    the validation (DFT/validation's AIMD trajectories) and training
    (DFT/training) structure trees -- see
    prepare_reaxff_validation_structures for how those get built. Two
    separate, sequential Flux submissions within this one allocation (not
    one combined submission), each with its own chore name, so validation
    and training batches can't collide and a failure in one doesn't block
    the other from at least being attempted.
    """
    cfg = config['reaxff_validation_single_points']
    _require(cfg, 'reaxff_validation_single_points', ['validation', 'training'])

    _submit_reaxff_single_points_batch(cfg['validation'], 'run_reaxff_val_lammps')
    _submit_reaxff_single_points_batch(cfg['training'], 'run_reaxff_train_lammps')


# ---------------------------------------------------------------------------
# Stage: finite_temperature_md_batch
# ---------------------------------------------------------------------------

def run_finite_temperature_md_batch(config):
    """
    Runs finite-temperature LAMMPS/ReaxFF MD (minimize -> room-temperature
    NPT, via lammps_reaxff_md.py/in.npt_room_temp -- NOT the full
    minimize/NPT/melt/anneal cycle in.matensemble runs, which is for a
    different, structure-diversity-generation purpose) for every staged
    force field (see stage_ft_md_force_fields) x every supercell structure
    under sample_ft_md_structures' own dest_root -- replaces the old
    single-foundation-model finite_temperature_md block inside
    fit_and_validate (still present there, disabled, for that original
    one-model smoke-test purpose) with a proper ff x structure cross
    product, needed now that this checks EVERY fitted candidate's own
    coordination-environment stability under real dynamics, not just one
    reference model.

    Reuses _submit_reaxff_single_points_batch as-is -- it's already
    generic over (run_directory, inputs_directory, ffield/structure/
    in_file/lammps_task/entry_point, parent_levels), the exact same
    ff x structure batching this needs, just pointed at a different
    driver/recipe and a much shallower, 3-structure tree.

    parent_levels below batches one chore per FORCE FIELD (each chore
    then runs all named structures for that ffield sequentially on one
    GPU, same persistent-lmp-instance-with-clear() pattern
    lammps_reaxff_md.py already uses) -- CONFIRMED via a direct sweep
    against the real staged force_fields/lammps_inputs trees (see
    workflow_config.yaml's own comment on finite_temperature_md_batch),
    not guessed. If per-force-field batching turns out to cause problems
    (e.g. one slow/stuck structure blocking its ffield's remaining
    structures), the swept values also confirmed the user's own fallback
    works: drop parent_levels to 0-1 so batch_by_parent lands one chore
    per (structure, force field) pair instead -- a config change only, no
    code change needed.
    """
    cfg = config['finite_temperature_md_batch']
    _require(cfg, 'finite_temperature_md_batch', ['run_directory', 'inputs_directory'])

    _submit_reaxff_single_points_batch(cfg, 'run_ft_md_coordination')


# ---------------------------------------------------------------------------
# Stage: check_coordination_stability
# ---------------------------------------------------------------------------

def run_check_coordination_stability(config):
    """
    Compares each finite-temperature MD run's final structure
    (run_directory/<ff_combo>/<structure_rel>/check_file, a LAMMPS data
    file written by write_data, e.g. data.npt_relax) against that same
    structure's own starting point (inputs_directory/<structure_rel>/
    structure.lmp, the pre-MD supercell sample_ft_md_structures wrote) via
    a per-site, per-element coordination-number deviation -- see
    coordination_task's own docstring for the actual metric.

    coordination_task/entry_point (dynamically imported via
    EnsembleFFFit.utilities.general.import_module_from_path, same
    mechanism as fine_tuning.ff_task/converge_dft_data.dft_task/
    MDMatEnsemble's own lammps_task dispatch) name a swappable driver
    script + its entry-point function -- deliberately NOT a hardcoded
    import of MD/finite_temperature/coordination_check/cn_checker.py,
    so a future non-pymatgen-CrystalNN metric (per JaxReaxFF_Integration_
    Plan.md's own note that this first pass may not be the most
    descriptive one) is a config change (point coordination_task at a
    new script with a matching check_coordination_stability(...) entry
    point), not a run_pipeline.py change.
    """
    from EnsembleFFFit.utilities.general import import_module_from_path

    cfg = config['check_coordination_stability']
    _require(cfg, 'check_coordination_stability',
              ['run_directory', 'inputs_directory', 'coordination_task'])

    module = import_module_from_path('coordination_task_module', cfg['coordination_task'])
    entry_point = getattr(module, cfg.get('entry_point', 'check_coordination_stability'))

    json_file = cfg.get('json_file', os.path.join(cfg['run_directory'], 'coordination_comparison.json'))
    deviation_dct, failures = entry_point(
        run_directory=cfg['run_directory'],
        inputs_directory=cfg['inputs_directory'],
        check_file=cfg.get('check_file', 'data.npt_relax'),
        structure=cfg.get('structure', 'structure.lmp'),
        atom_style=cfg.get('atom_style', 'charge'),
        oxi_dct=cfg.get('oxi_dct'),
        use_weights=cfg.get('use_weights', True),
        json_file=json_file,
    )
    n_failed_structures = sum(len(d) for d in failures.values())
    print(f"Wrote coordination comparison for {len(deviation_dct)} force field(s) to {json_file} "
          f"({n_failed_structures} failed run(s) across {len(failures)} force field(s) -- see coordination_task's "
          f"own failures reporting, e.g. an unstable MD simulation)")

    # Full table: one column per (structure, element) pair, grouped by
    # structure then split out by element within each structure (per the
    # user's own request, 2026-09 -- not folded into one combined mean per
    # row), plus a trailing mean_cn_deviation column used to rank rows.
    # The swappable part of this stage is the per-site deviation metric
    # itself (coordination_task), not this table-shaping/reporting step,
    # so it's kept here rather than pushed into the swappable script too.
    def _short_structure_label(md_name):
        # e.g. "mp_bulk/bulk_sp/Bi-Se/Bi2Se3/mp-23164/volume_1" ->
        # "Bi2Se3_mp-23164" -- md_name alone isn't unique enough to skip
        # (sample_ft_md_structures' named_paths all share the same leaf
        # dirname, "volume_1"), and the full path is unwieldy as a column
        # header, so pull out the compound+mp-id segment via the same
        # mp-\d+ pattern _group_by_mpid already uses elsewhere in this
        # file. Falls back to the full md_name if no mp-id is found (e.g.
        # a future named_paths entry with no Materials Project id).
        parts = md_name.split('/')
        for i, p in enumerate(parts):
            if re.fullmatch(r"mp-\d+", p):
                return f"{parts[i - 1]}_{p}" if i > 0 else p
        return md_name

    md_names = sorted({md_name for structure_dct in deviation_dct.values() for md_name in structure_dct})
    columns = [
        (md_name, el)
        for md_name in md_names
        for el in sorted({el for structure_dct in deviation_dct.values() for el in structure_dct.get(md_name, {})})
    ]
    col_labels = [f"{_short_structure_label(md_name)}_{el}" for md_name, el in columns]

    means = {}
    for ff_label, structure_dct in deviation_dct.items():
        values = [v for element_dct in structure_dct.values() for v in element_dct.values()]
        if values:
            means[ff_label] = sum(values) / len(values)
    sorted_labels = sorted(means, key=means.get)

    ranking_path = cfg.get('ranking_path', os.path.join(cfg['run_directory'], 'coordination_ranking.txt'))
    ff_label_width = max([len('ff_label')] + [len(l) for l in sorted_labels])
    col_width = max([17] + [len(c) for c in col_labels])

    header = (f"{'rank':>4}  {'ff_label':<{ff_label_width}}  "
              + "  ".join(f"{c:>{col_width}}" for c in col_labels)
              + f"  {'mean_cn_deviation':>17}")
    lines = [header]
    for rank, ff_label in enumerate(sorted_labels, start=1):
        structure_dct = deviation_dct[ff_label]
        row_values = []
        for md_name, el in columns:
            v = structure_dct.get(md_name, {}).get(el)
            row_values.append(f"{v:>{col_width}.6f}" if v is not None else f"{'--':>{col_width}}")
        lines.append(f"{rank:>4}  {ff_label:<{ff_label_width}}  " + "  ".join(row_values)
                     + f"  {means[ff_label]:>17.6f}")

    # Force fields with a failed MD run (an unstable simulation, e.g.
    # "Non-numeric pressure" -- see lammps_reaxff_md.py's own try/except)
    # -- listed explicitly rather than left as a silent "--" in the table
    # above (partial failures) or a silent absence (a force field that
    # failed on every structure never appears in deviation_dct/means at
    # all, so it wouldn't show up above in any form otherwise). This is a
    # genuine, useful outcome per the user's own framing (2026-09): a
    # force field that can't survive a short, real MD run on a relevant
    # phase is itself evidence it's not a good fit for that phase.
    if failures:
        lines.append("")
        lines.append("Failed MD run(s) (simulation unstable -- see error below; excluded from ranking above):")
        all_failed_only = sorted(set(failures) - set(means))
        if all_failed_only:
            lines.append(f"  Force field(s) with EVERY structure failed (absent from the table above entirely): "
                          f"{', '.join(all_failed_only)}")
        for ff_label in sorted(failures):
            for md_name, error in sorted(failures[ff_label].items()):
                lines.append(f"  {ff_label} / {_short_structure_label(md_name)}: {error}")

    print_and_write(lines, ranking_path)
    print(f"Wrote coordination ranking to {ranking_path}")


# ---------------------------------------------------------------------------
# Stage: rank_reaxff_validation
# ---------------------------------------------------------------------------

def _drop_first_run_segment(run_image_dct):
    """
    Strips the first '/'-separated segment off every "run" key of a
    {run: {image: props}} dict. CONFIRMED NECESSARY (2026-09), not
    defensive: MDMatEnsemble.build_full_runs' own task_dir derivation
    mirrors inputs_directory's own structures_subpath segment (here,
    literally "structures") into every fitted force field's own output
    tree (e.g. force_fields/<combo>/structures/aimd/2000K/<mp-id>/
    single_image/<frame>/properties.json), so dict_parsers.
    parse_labeled_tree's own "run" key for the FF side comes out as
    "structures/aimd/2000K/<mp-id>/single_image" -- but
    parse_reference_tree's "run" key for the DFT side (parsed directly
    from DFT/validation or DFT/training, which never has that extra
    segment) comes out as "aimd/2000K/<mp-id>/single_image", one segment
    shorter. Without stripping it here, best_force_field.
    assert_comparable_dicts would raise KeyError the first time it looked
    up an FF's own "run" key in the reference dict -- checked directly
    against both trees' actual on-disk paths, not assumed.
    """
    out = {}
    for run, images in run_image_dct.items():
        new_run = run.split('/', 1)[1] if '/' in run else ''
        out[new_run] = images
    return out

def run_rank_reaxff_validation(config):
    """
    Parses the LAMMPS single-point output from reaxff_validation_single_points
    plus the DFT ground truth (properties.json under DFT/validation and
    DFT/training, from DFT/vasprun_to_properties.py) into
    {label: {run: {image: props}}} dicts, unit-converts the LAMMPS side
    (real units -> eV -- lammps_energy_to_ev below MUST be updated if the
    LAMMPS recipe's own `units` directive ever changes away from "real",
    see EnsembleFFFit.analysis.best_force_field.convert_units's own
    docstring), and ranks every fitted force field via
    EnsembleFFFit.analysis.best_force_field.rank_force_fields_combined --
    see that function's own docstring for the full validation/training
    combination scheme (relative energy + forces on DFT/validation's AIMD
    trajectories, forces only on DFT/training). Prints + persists the
    ranking table, then optionally downselects/copies the top performers
    forward (e.g. for the coordination-number/FT-MD comparison this feeds
    into next) via the same select_and_copy mechanism downselect_force_fields
    used to use.
    """
    from EnsembleFFFit.analysis.best_force_field import convert_units, rank_force_fields_combined
    from EnsembleFFFit.analysis.dict_parsers import PropertiesOnlyParser

    cfg = config['rank_reaxff_validation']
    _require(cfg, 'rank_reaxff_validation',
              ['validation_ff_dir', 'validation_reference_root',
               'training_ff_dir', 'training_reference_root'])

    lammps_energy_to_ev = cfg.get('lammps_energy_to_ev', 23.060548867)

    # PropertiesOnlyParser, not the default ASEParser -- CONFIRMED (2026-09)
    # ASEParser's existence_check requires a co-located POSCAR alongside
    # properties.json, which MDMatEnsemble-style single-point output never
    # has (structure.lmp/POSCAR lives entirely under inputs_directory, not
    # alongside the chore's own output under run_directory) -- silently
    # produced a fully empty parsed dict, no error, until it later crashed
    # select_and_copy with "IndexError: list index out of range" on an
    # empty label list. Used for BOTH ff_dct and reference_dct here (not
    # just the LAMMPS side) for consistency -- "structure" is never read
    # by rank_force_fields_combined/get_ff_deviations anyway, only
    # energy/fx/fy/fz.
    validation_ff_dct_raw = parse_labeled_tree(cfg['validation_ff_dir'], parser_cls=PropertiesOnlyParser)
    validation_reference_dct = {"DFT": parse_reference_tree(cfg['validation_reference_root'], parser_cls=PropertiesOnlyParser)}
    validation_ff_dct = {label: convert_units(_drop_first_run_segment(run_dct), lammps_energy_to_ev)
                         for label, run_dct in validation_ff_dct_raw.items()}

    training_ff_dct_raw = parse_labeled_tree(cfg['training_ff_dir'], parser_cls=PropertiesOnlyParser)
    training_reference_dct = {"DFT": parse_reference_tree(cfg['training_reference_root'], parser_cls=PropertiesOnlyParser)}
    training_ff_dct = {label: convert_units(_drop_first_run_segment(run_dct), lammps_energy_to_ev)
                       for label, run_dct in training_ff_dct_raw.items()}

    table_lines, labels, scores = rank_force_fields_combined(
        validation_ff_dct, validation_reference_dct,
        training_ff_dct, training_reference_dct,
        validation_energy_weight=cfg.get('validation_energy_weight', 1.0),
        validation_force_weight=cfg.get('validation_force_weight', 1.0),
        training_force_weight=cfg.get('training_force_weight', 1.0),
    )
    print("\n".join(table_lines))

    dest_dir = cfg.get('dest_dir')
    selected_lines = []
    if dest_dir:
        strategy = cfg.get('strategy', 'top')
        number_to_copy = cfg.get('number_to_copy', 25)
        seed = cfg.get('seed')
        selected = select_and_copy(cfg['validation_ff_dir'], dest_dir, labels,
                                    number_to_copy=number_to_copy, strategy=strategy,
                                    patterns=(cfg.get('target_name', 'ffield'),), seed=seed)
        selected_lines = [f"\nSelected {len(selected)} force field(s) via strategy='{strategy}':"]
        for label, old_path, new_path in selected:
            selected_lines.append(f"  {label}: {old_path} -> {new_path}")
        print("\n".join(selected_lines))

    ranking_path = os.path.join(dest_dir or cfg['validation_ff_dir'], "ranking.txt")
    header = (f"validation_energy_weight={cfg.get('validation_energy_weight', 1.0)} "
              f"validation_force_weight={cfg.get('validation_force_weight', 1.0)} "
              f"training_force_weight={cfg.get('training_force_weight', 1.0)} "
              f"lammps_energy_to_ev={lammps_energy_to_ev}")
    print_and_write(table_lines + selected_lines, ranking_path, header=header)
    print(f"Wrote ranking to {ranking_path}")


# Order matters for --stage all -- this is the actual pipeline sequence,
# each stage depending on the previous one's output.
STAGES = {
    'converge_dft_data': run_converge_dft_data,
    'parse2fit_generation': run_parse2fit_generation,
    'sample_ft_md_structures': run_sample_ft_md_structures,
    'build_ff_inputs': run_build_ff_inputs,
    'fit_and_validate': run_fit_and_validate,
    'prepare_reaxff_validation_structures': run_prepare_reaxff_validation_structures,
    'stage_reaxff_validation_force_fields': run_stage_reaxff_validation_force_fields,
    'reaxff_validation_single_points': run_reaxff_validation_single_points,
    'rank_reaxff_validation': run_rank_reaxff_validation,
    'stage_ft_md_force_fields': run_stage_ft_md_force_fields,
    'finite_temperature_md_batch': run_finite_temperature_md_batch,
    'check_coordination_stability': run_check_coordination_stability,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", "-c", required=True, help="Path to the YAML workflow config file")
    parser.add_argument("--stage", "-s", required=True, choices=list(STAGES.keys()) + ['all'])
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.stage == 'all':
        # Runs the whole sequence in one process/allocation -- each stage
        # still waits for the previous one's MatEnsemble/Flux submission (if
        # any) to fully complete via .result() before the next stage starts,
        # same as running them by hand in order would. Only use this when
        # one allocation genuinely covers the whole pipeline's walltime.
        for name, stage_fn in STAGES.items():
            print(f"\n{'=' * 20} Stage: {name} {'=' * 20}")
            stage_fn(config)
    else:
        STAGES[args.stage](config)


if __name__ == "__main__":
    main()
