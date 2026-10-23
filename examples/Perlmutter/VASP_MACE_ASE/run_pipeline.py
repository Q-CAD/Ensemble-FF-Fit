"""
Single entry point for the Perlmutter VASP -> MACE -> ASE active-learning
pipeline (Pd-Se): VASP DFT convergence (isolated-atom E0s, then each
round's selected candidates) -> finite-temperature MD structure sampling
(from the training .xyz frames) -> MACE ensemble-input construction ->
MACE ensemble fitting -> validation single points against the AIMD testing
set -> finite-temperature MD -> FF ranking/downselection -> trajectory-frame
unpacking -> UQ single points -> DFT candidate selection.

Adapted from examples/Frontier/RMG_MACE_ASE's run_pipeline.py: every stage
from fit_and_validate onward is that example's code, with converge_dft_data
swapped for examples/Perlmutter/VASP_ReaxFF_LAMMPs's VASP version, plus the
Perlmutter Flux settings that example established (reserve_broker_node=False
on every Pipeline, set_cpu_affinity=False on every submit). The training
and testing data here start as pre-computed .xyz files (DFT/training,
DFT/testing -- provenance in README.md) rather than POSCARs to
converge, so sample_ft_md_structures/build_ff_inputs/validation_unpacking
read .xyz frames directly instead of converged DFT run directories.

Run one stage at a time -- each stage is its own compute allocation/
container invocation (converge_dft_data, fit_and_validate, and
uq_single_points submit MatEnsemble/Flux chores; sample_ft_md_structures,
build_ff_inputs, downselect, and select_dft_candidates are pure analysis/
file-ops and can run anywhere EnsembleFFFit's analysis extras are installed)
-- or run one full selection loop in one process via --stage all:

    python run_pipeline.py --config workflow_config.yaml --stage converge_dft_data
    python run_pipeline.py --config workflow_config.yaml --stage sample_ft_md_structures
    python run_pipeline.py --config workflow_config.yaml --stage build_ff_inputs
    python run_pipeline.py --config workflow_config.yaml --stage fit_and_validate
    python run_pipeline.py --config workflow_config.yaml --stage downselect
    python run_pipeline.py --config workflow_config.yaml --stage uq_single_points
    python run_pipeline.py --config workflow_config.yaml --stage select_dft_candidates
    python run_pipeline.py --config workflow_config.yaml --stage all

The next loop is another --stage all after bumping workflow_config.yaml's
"# GEN"-tagged keys: its converge_dft_data stage runs VASP on the POSCARs
the previous loop's select_dft_candidates wrote under DFT/uq/<n> (every
already-converged directory is skipped via finished_file), and
build_ff_inputs.extra_training_subpaths folds those results into training.

matensemble is only imported inside the three stage functions that actually
need it (converge_dft_data, fit_and_validate, uq_single_points), not at
module load time, so the other four stages don't require the Flux container.
best_force_field's sklearn dependency is similarly deferred into
run_downselect.
"""
import argparse
import glob
import os
import random
import shutil
import sys
from pathlib import Path

import yaml

from EnsembleFFFit.analysis.dict_parsers import parse_labeled_tree, parse_reference_tree
from EnsembleFFFit.analysis.downselect_force_fields import select_and_copy
from EnsembleFFFit.analysis.variance import format_candidate_table, get_structures_scores, select_structures
from EnsembleFFFit.utilities.general import print_and_write, unpack_trajectory_frames

# Perlmutter/Podman-HPC-specific runtime library path for the vasp_std
# subprocess only -- copied from examples/Perlmutter/VASP_ReaxFF_LAMMPs's
# run_pipeline.py, see its VASP_LD_LIBRARY_PATH comment for how each entry
# was confirmed (vasp/6.4.2-gpu's split NVHPC 25.9/26.5 toolchain, Flux's
# own libpmi ahead of Cray's so MPI_Init joins one multi-rank job, and the
# /host_lib64:/host_lib remapped host system libraries). Passed to the
# chore as VASP_LD_LIBRARY_PATH, NOT LD_LIBRARY_PATH: the host libc in
# /host_lib segfaults the chore's own Python process if applied to it --
# DFT/vasp_dft.py applies it to vasp_std's subprocess env only. Re-verify
# every path if the VASP module, container image, or NVHPC/Cray module
# versions change.
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
    """Env for the vasp_std chore -- see VASP_LD_LIBRARY_PATH above.

    OMP_NUM_THREADS is derived from cores_per_task so the two can't drift
    apart. PSEUDOPOTENTIAL_DIR/VDW_KERNEL_DIR/NO_STOP_MESSAGE/
    MPICH_NO_BUFFER_ALIAS_CHECK mirror the NERSC vasp module's own env
    (`module show vasp/6.4.2-gpu`) -- distinct from vdW_single_point.yml's
    pseudopotentials_directory, which pymatgen uses for POTCAR generation.
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


def mace_container_env():
    """Env for every MACE/ASE chore (fitting, validation/UQ single points,
    finite-temperature MD). Prepends /usr/lib64 -- where podman-hpc's
    hook_tool injects the node's real NVIDIA driver -- ahead of the image's
    default LD_LIBRARY_PATH, whose /usr/local/cuda/compat libcuda.so.1 shim
    doesn't match Perlmutter's driver (the same fix as
    VASP_ReaxFF_LAMMPs's lammps_container_env/jax_reaxff_container_env).
    The rest of the image's own LD_LIBRARY_PATH is preserved.
    """
    from EnsembleFFFit.utilities.general import ensemble_fffit_pythonpath

    return {
        'PYTHONPATH': ensemble_fffit_pythonpath(),
        'LD_LIBRARY_PATH': '/usr/lib64:' + os.environ.get('LD_LIBRARY_PATH', ''),
    }


def _with_defaults(cfg, defaults):
    """
    Copy of `cfg` with every key in `defaults` that is missing or null in
    `cfg` filled from `defaults` -- lets downstream path keys (e.g.
    downselect_force_fields.force_fields_dir) default to the upstream key
    that produces them (copy_force_fields.target_directory), so moving to
    the next generation only means editing the upstream "# GEN" keys in
    workflow_config.yaml, not every copy of each path.
    """
    cfg = dict(cfg or {})
    for key, value in defaults.items():
        if cfg.get(key) is None:
            cfg[key] = value
    return cfg


def _require(cfg, stage_name, keys):
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

    Sizing (GPU VASP): one MPI rank per GPU, so num_tasks = nodes *
    ranks_per_node (default 4, one per Perlmutter A100), each rank with
    gpus_per_task=1 and cores_per_task OpenMP threads. `nodes` is
    vasp_resource_estimator's per-structure estimate (the recipe's `nodes`
    pin, else ceil(n_atoms / atoms_per_node)), capped at max_nodes_per_job
    (default 4, NERSC's recommended maximum for VASP). Flux places at most 4
    of these ranks per node (4 GPUs), so a multi-node job spans exactly
    `nodes` nodes. CHANGED from the VASP_ReaxFF_LAMMPs example, which sized
    num_tasks = allocated_nodes * cores_per_task (one rank per CPU core, 1
    GPU each -- e.g. 32 GPUs for a 2-node job, which could never fit) and
    hid that behind max_tasks_per_job=4.

    Core budget: MatEnsemble/Flux offers 64 * N - 1 cores on an N-node
    allocation (one core stays with the broker). ranks_per_node *
    cores_per_task = 64 fills a whole node, so a job that needs every node
    of the allocation won't fit at 16 cores per rank -- use 15 there (e.g.
    on any 1-node allocation), or leave one node of headroom.
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
    ranks_per_node = dft_cfg.get('ranks_per_node', 4)
    max_nodes_per_job = dft_cfg.get('max_nodes_per_job', 4)

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
            # Every (structure, vasp_yaml) pair already converged -- a clean,
            # expected no-op (DFT convergence is a one-time prerequisite,
            # shared across every fitting generation, not something later
            # generations' own runs should ever need to redo), NOT an error.
            # CHANGED (2026-10) from raising ValueError here: that made
            # --stage all hard-crash at the very first stage on every
            # generation after the first, since by then the entire DFT
            # dataset is always already converged. Still raised below for
            # the genuinely-different case (nothing found at all, e.g. a
            # misconfigured directory) -- only THIS "found plenty, all
            # already finished" case is a no-op now.
            print(f"converge_dft_data: all {len(raw_task_dirs)} (structure, vasp_yaml) pair(s) under "
                  f"{directory} already have a '{finished_file}' match -- nothing new to run, skipping. "
                  f"Set finished_file to null (or a pattern that doesn't already exist everywhere) to "
                  f"force a rerun.")
            return
        raise ValueError(f"No (structure, vasp_yaml) pairs found under {directory}")

    # reserve_broker_node=False on every Pipeline in this file: no reason to
    # dedicate a whole node to just the Flux broker for chores this size;
    # frees that node for actual work.
    pipe = Pipeline(reserve_broker_node=False)
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
    # cores_per_task=cores_per_task (the config value), not hardcoded 1:
    # OMP_NUM_THREADS (vasp_container_env) is set to the same value, so
    # Flux must reserve that many cores per rank -- see this function's
    # docstring for the per-node core budget.
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

        # Sized per-job from this structure's own node estimate -- different
        # structures can legitimately need different node counts, and each
        # pipe.call carries its own Resources for exactly this reason.
        nodes = dft_dct['allocated_nodes']
        if max_nodes_per_job is not None and nodes > max_nodes_per_job:
            print(f"WARNING: {task_dict['structure_filename']}: estimated {nodes} node(s) exceeds "
                  f"max_nodes_per_job={max_nodes_per_job}; running on {max_nodes_per_job}.", file=sys.stderr)
            nodes = max_nodes_per_job
            task_dict['allocated_nodes'] = nodes
        num_tasks = nodes * ranks_per_node
        print(f"converge_dft_data: {task_dict['working_directory']}: {nodes} node(s), {num_tasks} MPI rank(s) "
              f"x {cores_per_task} core(s) x {gpus_per_task} GPU(s)")
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

    # A failed/rejected chore doesn't raise out of future.result() -- CONFIRMED
    # in this example's first smoke test, where both isolated-atom chores were
    # rejected up front ("requires more resources than the allocation can
    # provide") and the stage still returned normally, surfacing only later as
    # build_ff_inputs' "no isolated-atom runs" error. Check finished_file
    # directly: warn per failure (one bad DFT candidate shouldn't stop a
    # loop), but fail loudly if nothing converged at all.
    if finished_file is not None:
        missing = [d['working_directory'] for d in dft_dct_list
                   if not glob.glob(os.path.join(d['working_directory'], finished_file))]
        for working_directory in missing:
            print(f"WARNING: converge_dft_data: no '{finished_file}' in {working_directory} -- that VASP "
                  f"run failed; see the matensemble_workflow-*/ logs.", file=sys.stderr)
        if len(missing) == len(dft_dct_list):
            raise RuntimeError(f"converge_dft_data: all {len(missing)} VASP run(s) failed (no '{finished_file}' "
                               f"written) -- see the matensemble_workflow-*/ logs.")


# ---------------------------------------------------------------------------
# Stage: sample_ft_md_structures
# ---------------------------------------------------------------------------

def _discover_xyz_labels(root, labels=None):
    """root/<label>/*.xyz -> {label: [xyz paths]} for every immediate
    subdirectory of root holding at least one .xyz (restricted to `labels`,
    in that order, when given)."""
    root = Path(root)
    found = {}
    for label_dir in sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []:
        paths = sorted(str(p) for p in label_dir.glob('*.xyz'))
        if paths:
            found[label_dir.name] = paths
    if labels is None:
        return found
    missing = [label for label in labels if label not in found]
    if missing:
        raise ValueError(f"Label(s) {missing} have no .xyz files under {root} (found: {sorted(found)})")
    return {label: found[label] for label in labels}


def _read_label_frames(xyz_paths):
    """Every frame of every file in xyz_paths, concatenated in order."""
    from ase.io import read

    frames = []
    for path in xyz_paths:
        frames.extend(read(path, index=':', format='extxyz'))
    return frames


def _sample_evenly_by_label(frames_by_label, total, seed=None):
    """
    Sample `total` (label, frame_index) pairs, spread as evenly as possible
    across labels -- same scheme as RMG_MACE_ASE's mp-id sampler: each
    label gets total // n_labels frames, with the remainder handed out one
    at a time to the first few labels in sorted order, so the result adds
    up to exactly `total` (unless a label has fewer frames than its share).
    """
    rng = random.Random(seed)
    labels = sorted(frames_by_label)
    if not labels:
        return []

    base, remainder = divmod(total, len(labels))

    selected = []
    for i, label in enumerate(labels):
        n_here = min(base + (1 if i < remainder else 0), len(frames_by_label[label]))
        indices = sorted(rng.sample(range(len(frames_by_label[label])), n_here))
        selected.extend((label, idx) for idx in indices)
    return selected


def run_sample_ft_md_structures(config):
    """
    Sample a fixed total number of frames out of the training .xyz files
    (DFT/training/<label>/*.xyz), spread evenly across the configured
    labels, and write each as dest_root/<label>/<frame_index>/POSCAR -- the
    base cell as stored, no supercell or perturbation here, since
    ase_mace_md.py's own balanced_supercell/defect-mutation steps do that at
    MD time. Replaces RMG_MACE_ASE's POSCAR/mp-id sampler: this example's
    training data has no per-structure POSCARs or mp-ids to group by.

    `labels` restricts which training subsets seed MD -- e.g. leaving out
    thermal_cluster, whose isolated clusters in a vacuum box would have
    their box collapsed by the driver's FrechetCellFilter cell relaxation.
    source_root defaults to converge_dft_data.directory/training_subpath
    when null; dest_root defaults to finite_temperature_md.
    ase_inputs_directory/structures_subpath when null.
    """
    from ase.io import write

    cfg = config.get('sample_ft_md_structures', {})

    source_root = cfg.get('source_root')
    if not source_root:
        _require(config['converge_dft_data'], 'converge_dft_data', ['directory'])
        source_root = os.path.join(config['converge_dft_data']['directory'], cfg.get('training_subpath', 'training'))

    dest_root = cfg.get('dest_root')
    if not dest_root:
        ft_md_cfg = config['finite_temperature_md']
        dest_root = os.path.join(ft_md_cfg['ase_inputs_directory'], ft_md_cfg.get('structures_subpath', 'structures'))
    structure_filename = cfg.get('structure_filename', 'POSCAR')
    total = cfg.get('total', 20)
    seed = cfg.get('seed', 0)

    frames_by_label = {label: _read_label_frames(paths)
                       for label, paths in _discover_xyz_labels(source_root, cfg.get('labels')).items()}
    selected = _sample_evenly_by_label(frames_by_label, total, seed=seed)
    print(f"Sampled {len(selected)} frame(s) across labels {sorted(frames_by_label)} under {source_root}")

    for label, idx in selected:
        dest_dir = os.path.join(dest_root, label, str(idx))
        os.makedirs(dest_dir, exist_ok=True)
        atoms = frames_by_label[label][idx]
        write(os.path.join(dest_dir, structure_filename), atoms, format='vasp', direct=True, sort=True)
        print(f"  -> {dest_dir} ({atoms.get_chemical_formula()}, {len(atoms)} atoms)")


# ---------------------------------------------------------------------------
# Stage: build_ff_inputs
# ---------------------------------------------------------------------------

def _concatenate_label(xyz_paths, out_path):
    from EnsembleFFFit.potential.mace.build_ensemble_inputs import concatenate_xyz_files

    out_path.parent.mkdir(parents=True, exist_ok=True)
    concatenate_xyz_files(xyz_paths, out_path)
    return str(out_path)


def run_build_ff_inputs(config):
    """
    Builds the randomly-sampled ensemble of MACE training-input folders
    (one per label-combo x weight-combo sample) that fit_and_validate reads
    from:
      1. one combinable training label per DFT/training/<label>/ subset
         (already extxyz -- provenance in README.md), plus one per
         extra_training_subpaths entry (converged VASP runs, e.g. a previous
         loop's DFT/uq/1, converted via write_training_xyz -- empty in the
         first loop, see the "# GEN" keys in workflow_config.yaml);
      2. a fixed test.xyz -- every DFT/testing/**/*.xyz AIMD trajectory
         together, never split/sampled (the same set fit_and_validate's
         validation single points rank against);
      3. isolated-atom E0s from DFT/isolated_elements (converge_dft_data);
    then build_ensemble_inputs.build_mace_ensemble_inputs materializes each
    sampled (data_combo, weight_combo) pair as its own numbered folder.
    mace_inputs_dir defaults to fine_tuning.inputs_directory when null.
    """
    from EnsembleFFFit.potential.mace.write_training_xyz import write_training_xyz
    from EnsembleFFFit.potential.mace.build_ensemble_inputs import get_isolated_atom_e0s, build_mace_ensemble_inputs

    cfg = config['build_ff_inputs']
    _require(cfg, 'build_ff_inputs', ['dft_root'])

    dft_root = Path(cfg['dft_root'])
    mace_inputs_dir = Path(cfg.get('mace_inputs_dir') or config['fine_tuning']['inputs_directory'])
    structure_filename = cfg.get('structure_filename', 'POSCAR')

    training_root = dft_root / cfg.get('training_subpath', 'training')
    testing_root = dft_root / cfg.get('testing_subpath', 'testing')
    isolated_elements_root = dft_root / cfg.get('isolated_elements_subpath', 'isolated_elements')

    xyz_root = mace_inputs_dir.parent / "xyz_data"

    # 1a. Pre-computed training subsets -- each becomes one combinable label.
    training_xyz_by_label = {}
    for label, paths in _discover_xyz_labels(training_root, cfg.get('training_labels')).items():
        training_xyz_by_label[label] = _concatenate_label(paths, xyz_root / "training" / f"{label}.xyz")
    print(f"Training labels from {training_root}: {sorted(training_xyz_by_label)}")

    # 1b. Converged VASP runs from earlier loops (e.g. uq/1) -- one label each.
    for subpath in cfg.get('extra_training_subpaths') or []:
        label = subpath.strip('/').replace('/', '_')
        out_path = xyz_root / "training" / f"{label}.xyz"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists():
            out_path.unlink()  # write_training_xyz appends -- start clean each run
        written = write_training_xyz(dft_root / subpath, out_path, structure_filename=structure_filename)
        print(f"  {label}: {len(written)} completed run(s) under {dft_root / subpath} -> {out_path}")
        if not written:
            raise RuntimeError(f"extra_training_subpaths entry {subpath!r} has no completed runs under "
                                f"{dft_root / subpath} -- has converge_dft_data finished them?")
        training_xyz_by_label[label] = str(out_path)

    if not training_xyz_by_label:
        raise RuntimeError(f"No training .xyz files found under {training_root}")

    # 2. Fixed test set -- every AIMD trajectory, shared by every ensemble member.
    testing_xyz_paths = sorted(str(p) for p in testing_root.rglob('*.xyz'))
    print(f"Testing: {len(testing_xyz_paths)} trajectory file(s) under {testing_root}")
    if not testing_xyz_paths:
        raise RuntimeError(f"No testing .xyz files found under {testing_root}")

    # 3. Isolated-atom E0s.
    e0s = get_isolated_atom_e0s(isolated_elements_root, properties_filename="properties.json")
    print(f"Isolated-atom E0s (atomic number -> energy): {e0s}")
    if not e0s:
        raise RuntimeError(f"No completed isolated-atom runs found under {isolated_elements_root} -- "
                            f"has converge_dft_data finished?")

    # 4. Randomly-sampled ensemble of mace_inputs folders. base_config's
    # overrides get baked into each folder's own config.yml once, here.
    written_folders = build_mace_ensemble_inputs(
        training_xyz_by_label=training_xyz_by_label,
        validation_xyz_paths=testing_xyz_paths,
        output_dir=mace_inputs_dir,
        e0s=e0s,
        total_cap=cfg.get('total_cap', 100),
        seed=cfg.get('seed', 0),
        weight_grid_kwargs={
            "energy_num": cfg.get('energy_num', 3),
            "force_num": cfg.get('force_num', 3),
            "stress_num": cfg.get('stress_num', 1),
        },
        always_include_labels=tuple(cfg.get('always_include_labels') or ()),
        base_config=cfg.get('mace_overrides'),
    )
    print(f"Wrote {len(written_folders)} mace_inputs folder(s) under {mace_inputs_dir}")


# ---------------------------------------------------------------------------
# Stage: fit_and_validate (Pipeline 1)
# ---------------------------------------------------------------------------

def _unpack_xyz_reference_frames(source_root, dest_root, structure_filename="POSCAR"):
    """
    Expand every extxyz trajectory under source_root (e.g. DFT/testing/
    Pd-Se/PdSe2/mp-2418.xyz) into one leaf per frame,
    dest_root/<relpath-without-.xyz>/<frame_index>/{POSCAR, properties.json}
    -- the DFT-ground-truth leaf shape RMG_MACE_ASE's validation_mirroring
    copied from converged DFT runs, which MD_single_points'
    structures/ tree, ase_mace.py, and downselect's parse_reference_tree
    all expect. properties.json carries the reference energy/fx/fy/fz in
    the same shape DFT/vasp_dft.py writes. Returns (num_files, num_frames).
    """
    import json
    from ase.io import read, write

    num_files = num_frames = 0
    for xyz_path in sorted(Path(source_root).rglob('*.xyz')):
        rel = xyz_path.relative_to(source_root).with_suffix('')
        for i, atoms in enumerate(read(str(xyz_path), index=':', format='extxyz')):
            out_dir = Path(dest_root) / rel / str(i)
            out_dir.mkdir(parents=True, exist_ok=True)
            write(str(out_dir / structure_filename), atoms, format='vasp', direct=True)
            forces = atoms.get_forces()
            with open(out_dir / 'properties.json', 'w') as f:
                json.dump({'energy': float(atoms.get_potential_energy()),
                           'fx': forces[:, 0].tolist(), 'fy': forces[:, 1].tolist(),
                           'fz': forces[:, 2].tolist()}, f, indent=4)
            num_frames += 1
        num_files += 1
    return num_files, num_frames


def run_fit_and_validate(config):
    """
    Unpacks the AIMD testing trajectories (DFT/testing) into
    MD_single_points' inputs_directory as per-frame DFT ground truth, then
    builds and submits a Pipeline with: MACE
    ensemble fitting (fit_mace); a reactive strategy (copy_and_spawn_md)
    that, per completed fit, copies the model and spawns one run_ase chore
    covering all of that variant's validation structures; and (if
    finite_temperature_md is configured) a flat, upfront batch of
    finite-temperature MD chores, independent of any fit result. A
    Pipeline supports only one Pipeline.strategy() at a time, so both the
    reactive spawn and the flat FT-MD submission live in this one Pipeline.
    """
    from matensemble.pipeline import Pipeline
    from matensemble.model import Resources
    from matensemble.chore import ChoreSpec
    from EnsembleFFFit.base import FFMatEnsemble, MDMatEnsemble
    from EnsembleFFFit.utilities.general import ensemble_fffit_pythonpath, make_mirrored_rename_dest_path_fn

    env = mace_container_env()

    fine_tuning_cfg = config['fine_tuning']
    copy_cfg = config['copy_force_fields']
    md_cfg = config['MD_single_points']
    ft_md_cfg = config.get('finite_temperature_md')
    unpack_cfg = config.get('validation_unpacking')

    _require(fine_tuning_cfg, 'fine_tuning', ['run_directory', 'inputs_directory', 'ff_task'])
    _require(copy_cfg, 'copy_force_fields', ['target_directory'])
    _require(md_cfg, 'MD_single_points', ['inputs_directory'])
    if ft_md_cfg:
        _require(ft_md_cfg, 'finite_temperature_md',
                  ['output_directory', 'foundation_model', 'in_file', 'ase_inputs_directory'])

    if unpack_cfg:
        _require(unpack_cfg, 'validation_unpacking', ['source_root', 'dest_root'])
        num_files, num_frames = _unpack_xyz_reference_frames(unpack_cfg['source_root'], unpack_cfg['dest_root'])
        print(f"Unpacked {num_frames} reference frame(s) from {num_files} trajectory file(s) under "
              f"{unpack_cfg['source_root']} into {unpack_cfg['dest_root']}")

    mace_run_directory = fine_tuning_cfg['run_directory']
    # Absolute, not config-relative: copy_and_spawn_md below executes inside
    # its own chore working directory (matensemble_workflow-*/out/chore-*/),
    # not this process's cwd -- CONFIRMED in this example's smoke test, where
    # relative paths copied each model into that chore directory and then
    # failed to find ase_mace.py there.
    md_run_directory = os.path.abspath(copy_cfg['target_directory'])
    md_inputs_directory = os.path.abspath(md_cfg['inputs_directory'])
    target_name = copy_cfg.get('target_name', 'model.model')

    md_options = {'ffield': md_cfg.get('ffield', 'ffield'),
                  'in_file': md_cfg.get('in_file'),
                  'control': md_cfg.get('control'),
                  'structure': md_cfg.get('structure', 'structure.lmp'),
                  'lammps_task': md_cfg.get('lammps_task', 'lammps_task.py')}
    md_options = {k: v for k, v in md_options.items() if v is not None}

    md_check_files = md_cfg.get('check_files', ['ffield'])
    md_lammps_task = md_cfg.get('lammps_task', 'lammps_task.py')
    md_entry_point = md_cfg.get('entry_point', 'run_ase_single_points')
    md_finished_file = md_cfg.get('finished_file')

    md_resources_kwargs = dict(num_tasks=md_cfg.get('num_tasks', 1),
                                cores_per_task=md_cfg.get('cores_per_task', 1),
                                gpus_per_task=md_cfg.get('gpus_per_task', 0))

    pipe = Pipeline(reserve_broker_node=False)

    # Registration order matters: Pipeline.strategy()'s bolo_list is checked
    # against the registry immediately at decoration time, so "fit_mace"
    # must already be registered before copy_and_spawn_md is decorated.

    @pipe.chore(name="fit_mace",
                num_tasks=fine_tuning_cfg.get('num_tasks', 1),
                cores_per_task=fine_tuning_cfg.get('cores_per_task', 1),
                gpus_per_task=fine_tuning_cfg.get('gpus_per_task', 1))
    def fit_mace_chore(task_dict):
        """Thin chore wrapper -- the real fitting logic lives in whatever driver script
        fine_tuning.ff_task points at (see FFMatEnsemble.run_individual)."""
        return FFMatEnsemble.run_individual(task_dict)

    @pipe.chore(name="run_ase", **md_resources_kwargs)
    def run_ase_chore(task_dict):
        """Thin chore wrapper -- the real MD-execution logic lives in MDMatEnsemble.run_individual."""
        return MDMatEnsemble.run_individual(task_dict)

    if ft_md_cfg:
        ft_md_resources_kwargs = dict(num_tasks=ft_md_cfg.get('num_tasks', 1),
                                       cores_per_task=ft_md_cfg.get('cores_per_task', 1),
                                       gpus_per_task=ft_md_cfg.get('gpus_per_task', 1))

        @pipe.chore(name="run_ft_md", **ft_md_resources_kwargs)
        def run_ft_md_chore(task_dict):
            """Thin chore wrapper -- the real FT-MD execution logic lives in MDMatEnsemble.run_individual."""
            return MDMatEnsemble.run_individual(task_dict)

    @pipe.strategy(bolo_list=["fit_mace"], name="copy_and_spawn_md",
                   num_tasks=1, cores_per_task=1, gpus_per_task=0,
                   env={'PYTHONPATH': ensemble_fffit_pythonpath()}, inherit_env=True)
    def copy_and_spawn_md(fit_result):
        """
        Runs once per completed fit_mace chore: copies that one fitted
        model to its mirrored location under md_run_directory, builds this
        one variant's validation task_dicts (parent_levels=0, one batch per
        structure), then combines them all into a single run_ase call --
        combine_task_dicts works regardless of the validation structure
        tree's depth/shape, unlike a fixed parent_levels tuned to one
        specific tree. A failed fit_mace chore never reaches here.
        """
        source_path = Path(fit_result['results_dir']) / f"{fit_result['name']}.model"
        # Mirror relative to this fit's own results_dir parent (== the
        # foundation model's own containing directory, constant across every
        # variant), NOT the configured run_directory -- the two only
        # coincide when the foundation model sits directly at run_directory's
        # root. When it's nested (e.g. run_directory=FF/generation_1 but the
        # seed model lives at FF/generation_1/96/model.model), mirroring
        # against run_directory would carry that extra "96" segment into
        # every copy, collapsing all variants under one shared prefix
        # instead of each getting its own -- confirmed as a real bug against
        # a generation_1/96-nested foundation model.
        source_directory = Path(fit_result['results_dir']).parent
        dest_path_fn = make_mirrored_rename_dest_path_fn(source_directory, md_run_directory, target_name)
        dest_path = dest_path_fn(source_path, None)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, dest_path)

        md = MDMatEnsemble(str(dest_path.parent), md_inputs_directory, **md_options)
        task_dicts = md.build_task_dicts(md_lammps_task, 0, md_check_files, md_entry_point,
                                          finished_file=md_finished_file)
        combined = MDMatEnsemble.combine_task_dicts(task_dicts)
        if combined is None:
            return None  # no structures matched this variant -- nothing to spawn

        return ChoreSpec(args=(combined,), kwargs={}, qualname="run_ase",
                          resources=Resources(**md_resources_kwargs, env=env, inherit_env=True))

    fine_tuning_options = {'foundation_model': fine_tuning_cfg.get('foundation_model', 'model.model'),
                           'config': fine_tuning_cfg.get('config'),
                           'train_file': fine_tuning_cfg.get('train_file', 'train.xyz'),
                           'test_file': fine_tuning_cfg.get('test_file', 'test.xyz')}
    fine_tuning_options = {k: v for k, v in fine_tuning_options.items() if v is not None}

    ff_task = str(Path(fine_tuning_cfg['ff_task']).resolve())
    ff_entry_point = fine_tuning_cfg.get('entry_point', 'run_mace_fit')

    ff_matensemble = FFMatEnsemble(mace_run_directory, fine_tuning_cfg['inputs_directory'], **fine_tuning_options)
    check_files = fine_tuning_cfg.get('check_files', ['foundation_model'])
    # Deliberately finished_file=None here: every variant must get a fit_mace
    # chore submitted, even already-finished ones, so its completion still
    # triggers copy_and_spawn_md -- the skip-if-already-fit check happens at
    # execution time inside the driver script (mace_fit.py's run_mace_fit)
    # via FFMatEnsemble.run_individual instead.
    mace_arg_dict_list = ff_matensemble.build_ff_dcts(ff_task, check_files, ff_entry_point, finished_file=None)
    mace_finished_file = fine_tuning_cfg.get('finished_file')
    if mace_finished_file:
        for overrides in mace_arg_dict_list:
            overrides['finished_file'] = mace_finished_file

    fitting_resources = Resources(num_tasks=fine_tuning_cfg.get('num_tasks', 1),
                                   cores_per_task=fine_tuning_cfg.get('cores_per_task', 1),
                                   gpus_per_task=fine_tuning_cfg.get('gpus_per_task', 1),
                                   env=env, inherit_env=True)

    for mace_inputs in mace_arg_dict_list:
        pipe.call("fit_mace", mace_inputs, resources=fitting_resources)

    if ft_md_cfg:
        ft_md_task_command = os.path.abspath(
            os.path.join(ft_md_cfg['ase_inputs_directory'], ft_md_cfg.get('lammps_task', 'ase_mace_md.py')))
        # Structures live inside ase_inputs_directory (structures_subpath,
        # default "structures") -- the same inputs_directory convention
        # MD_single_points/MD_uq_single_points use -- not a separately
        # tracked structures_directory.
        ft_md_structures_root = os.path.join(
            ft_md_cfg['ase_inputs_directory'], ft_md_cfg.get('structures_subpath', 'structures'))
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
        structure_filename = ft_md_cfg.get('structure_filename', 'POSCAR')
        if not any(structure_filename in files for _, _, files in os.walk(ft_md_structures_root)):
            raise ValueError(
                f"finite_temperature_md is configured but no structures were found under "
                f"{ft_md_structures_root} -- did you run the sample_ft_md_structures stage first? "
                f"(Silently submitting zero run_ft_md chores here previously masked exactly this: an "
                f"empty structures directory produced no error, just no finite-temperature MD.)"
            )
        if not ft_md_task_dicts:
            # Structures exist but every one already has finished_file -- a
            # resumed run, not a misconfiguration (CONFIRMED: RMG_MACE_ASE's
            # version raised here, breaking --stage all reruns).
            print(f"finite_temperature_md: every structure under {ft_md_structures_root} already has "
                  f"'{ft_md_cfg.get('finished_file')}' -- skipping FT-MD.")
        ft_md_resources = Resources(**ft_md_resources_kwargs, env=env, inherit_env=True)
        for task_dict in ft_md_task_dicts:
            pipe.call("run_ft_md", task_dict, resources=ft_md_resources)

    future = pipe.submit(log_delay=25, set_gpu_affinity=True, set_cpu_affinity=False)
    future.result()


# ---------------------------------------------------------------------------
# Stage: downselect
# ---------------------------------------------------------------------------

def run_downselect(config):
    """
    Rank fitted force fields against DFT ground truth, print + persist the
    ranking, downselect and copy the top performers into
    MD_uq_single_points.run_directory, then unpack every saved FT-MD
    trajectory frame into MD_uq_single_points.inputs_directory's
    structures/ subtree.
    """
    from EnsembleFFFit.analysis.best_force_field import format_ranking_table

    ff_cfg = _with_defaults(config.get('downselect_force_fields'), {
        'force_fields_dir': config['copy_force_fields'].get('target_directory'),
        'ase_inputs_dir': config['MD_single_points'].get('inputs_directory'),
        'dest_dir': config['MD_uq_single_points'].get('run_directory'),
    })
    _require(ff_cfg, 'downselect_force_fields', ['force_fields_dir', 'ase_inputs_dir', 'dest_dir'])

    ff_dct = parse_labeled_tree(ff_cfg['force_fields_dir'])
    if not ff_dct:
        raise RuntimeError(f"No validation single points found under {ff_cfg['force_fields_dir']} -- did "
                           f"fit_and_validate's fit_mace/copy_and_spawn_md/run_ase chores succeed? (see the "
                           f"matensemble_workflow-*/ logs)")
    reference_dct = {"DFT": parse_reference_tree(ff_cfg['ase_inputs_dir'])}

    energy_weight = ff_cfg.get('energy_weight', 1.0)
    force_weight = ff_cfg.get('force_weight', 1.0)
    table_lines, labels, scores = format_ranking_table(ff_dct, reference_dct, energy_weight, force_weight)
    print("\n".join(table_lines))

    strategy = ff_cfg.get('strategy', 'top')
    number_to_copy = ff_cfg.get('number_to_copy', 25)
    seed = ff_cfg.get('seed')
    selected = select_and_copy(ff_cfg['force_fields_dir'], ff_cfg['dest_dir'], labels,
                                number_to_copy=number_to_copy, strategy=strategy, seed=seed)

    selected_lines = [f"\nSelected {len(selected)} force field(s) via strategy='{strategy}':"]
    for label, old_path, new_path in selected:
        selected_lines.append(f"  {label}: {old_path} -> {new_path}")
    print("\n".join(selected_lines))

    ranking_path = os.path.join(ff_cfg['dest_dir'], "ranking.txt")
    header = (f"energy_weight={energy_weight} force_weight={force_weight} "
              f"number_to_copy={number_to_copy} strategy='{strategy}' seed={seed}")
    print_and_write(table_lines + selected_lines, ranking_path, header=header)
    print(f"Wrote ranking to {ranking_path}")

    if 'trajectory_unpacking' in config:
        uq_inputs = config['MD_uq_single_points'].get('inputs_directory')
        traj_cfg = _with_defaults(config['trajectory_unpacking'], {
            'source_root': (config.get('finite_temperature_md') or {}).get('output_directory'),
            'dest_root': os.path.join(uq_inputs, 'structures') if uq_inputs else None,
        })
        _require(traj_cfg, 'trajectory_unpacking', ['source_root', 'dest_root'])
        num_runs, num_frames = unpack_trajectory_frames(traj_cfg['source_root'], traj_cfg['dest_root'])
        print(f"\nUnpacked {num_frames} frame(s) from {num_runs} run(s) under {traj_cfg['source_root']} "
              f"into {traj_cfg['dest_root']}")


# ---------------------------------------------------------------------------
# Stage: uq_single_points (Pipeline 2)
# ---------------------------------------------------------------------------

def run_uq_single_points(config):
    """
    Flat, non-reactive submission: every downselected force field x every
    unpacked FT-MD frame, batched via parent_levels so each force field
    gets one chore covering all of its structures rather than one chore
    per (ffield, structure) pair.

    lammps_task is normally a bare filename resolved against
    inputs_directory (MDMatEnsemble.build_lists); a value containing a path
    separator is instead resolved against the working directory, so each
    generation's inputs_directory (its own unpacked frames only) can share
    one driver script rather than each needing its own copy.
    """
    from matensemble.pipeline import Pipeline
    from matensemble.model import Resources
    from EnsembleFFFit.base import MDMatEnsemble

    uq_cfg = config['MD_uq_single_points']
    _require(uq_cfg, 'MD_uq_single_points', ['run_directory', 'inputs_directory'])

    lammps_task = uq_cfg.get('lammps_task', 'ase_mace.py')
    if os.sep in lammps_task:
        lammps_task = os.path.abspath(lammps_task)  # os.path.join(inputs_directory, <abs>) == <abs>

    options = {'ffield': uq_cfg.get('ffield', 'model.model'),
               'in_file': uq_cfg.get('in_file'),
               'control': uq_cfg.get('control'),
               'structure': uq_cfg.get('structure', 'POSCAR'),
               'lammps_task': lammps_task}
    options = {k: v for k, v in options.items() if v is not None}

    resources_kwargs = dict(num_tasks=uq_cfg.get('num_tasks', 1),
                             cores_per_task=uq_cfg.get('cores_per_task', 1),
                             gpus_per_task=uq_cfg.get('gpus_per_task', 1))

    md = MDMatEnsemble(uq_cfg['run_directory'], uq_cfg['inputs_directory'], **options)
    task_dicts = md.build_task_dicts(
        lammps_task,
        uq_cfg.get('parent_levels', 4),
        uq_cfg.get('check_files', ['ffield']),
        uq_cfg.get('entry_point', 'run_ase_single_points'),
        finished_file=uq_cfg.get('finished_file'),
    )
    if not task_dicts:
        if uq_cfg.get('finished_file') and md.build_task_dicts(
                lammps_task, uq_cfg.get('parent_levels', 4), uq_cfg.get('check_files', ['ffield']),
                uq_cfg.get('entry_point', 'run_ase_single_points'), finished_file=None):
            # Every (model, frame) pair already finished -- a resumed run.
            print("uq_single_points: every (force field, structure) pair already has "
                  f"'{uq_cfg['finished_file']}' -- nothing new to run, skipping.")
            return
        raise ValueError("No (force field, structure) batches found -- check run_directory/inputs_directory "
                          "in the 'MD_uq_single_points' config section.")

    pipe = Pipeline(reserve_broker_node=False)

    @pipe.chore(name="run_uq_ase", **resources_kwargs)
    def run_uq_ase_chore(task_dict):
        """Thin chore wrapper -- the real MD-execution logic lives in MDMatEnsemble.run_individual."""
        return MDMatEnsemble.run_individual(task_dict)

    resources = Resources(**resources_kwargs, env=mace_container_env(), inherit_env=True)
    for task_dict in task_dicts:
        pipe.call("run_uq_ase", task_dict, resources=resources)

    future = pipe.submit(log_delay=25, set_gpu_affinity=True, set_cpu_affinity=False)
    future.result()


# ---------------------------------------------------------------------------
# Stage: select_dft_candidates
# ---------------------------------------------------------------------------

def run_select_dft_candidates(config):
    """
    Score UQ single points by cross-FF ensemble disagreement (no DFT
    reference needed -- this measures ensemble disagreement, not agreement
    with ground truth), downselect the most-uncertain, well-spread frames,
    and write them as POSCAR files for the next round of DFT calculations.
    """
    cfg = _with_defaults(config.get('dft_candidate_selection'), {
        'force_fields_dir': config['MD_uq_single_points'].get('run_directory'),
    })
    _require(cfg, 'dft_candidate_selection', ['force_fields_dir', 'dest_root'])

    single_point_dct = parse_labeled_tree(cfg['force_fields_dir'])

    energy_weight = cfg.get('energy_weight', 1.0)
    force_weight = cfg.get('force_weight', 1.0)
    labels, images, structures, scores = get_structures_scores(
        single_point_dct, energy_weight, force_weight, reverse=True)
    print(f"Scored {len(labels)} candidate image(s) across {len(single_point_dct)} force field(s).\n")

    total = cfg.get('total')
    total = total if total is not None and total > 0 else None
    score_cap = cfg.get('score_cap')
    max_per_run = cfg.get('max_per_run', 6)
    image_distance = cfg.get('image_distance', 2)

    s_labels, s_images, s_structures, s_scores = select_structures(
        labels, images, structures, scores,
        total=total, score_cap=score_cap, max_per_label=max_per_run, image_distance=image_distance)

    output_dirs = [str(i) for i in range(len(s_labels))]
    table_lines = format_candidate_table(s_labels, s_images, s_scores, output_dirs)
    print("\n".join(table_lines))

    dest_root = cfg['dest_root']
    written = []
    for structure, out_dir in zip(s_structures, output_dirs):
        full_dir = os.path.join(dest_root, out_dir)
        os.makedirs(full_dir, exist_ok=True)
        structure.get_sorted_structure().to(fmt="poscar", filename=os.path.join(full_dir, "POSCAR"))
        written.append(full_dir)
    print(f"\nWrote {len(written)} structure(s) to {dest_root}")

    manifest_path = os.path.join(dest_root, "selection_manifest.txt")
    header = (f"Scored {len(labels)} candidate image(s) across {len(single_point_dct)} force field(s). "
              f"energy_weight={energy_weight} force_weight={force_weight} max_per_run={max_per_run} "
              f"image_distance={image_distance} total={total} score_cap={score_cap}")
    print_and_write(table_lines, manifest_path, header=header)
    print(f"Wrote selection manifest to {manifest_path}")


# Order matters for --stage all -- this is the actual pipeline sequence,
# each stage depending on the previous one's output.
STAGES = {
    'converge_dft_data': run_converge_dft_data,
    'sample_ft_md_structures': run_sample_ft_md_structures,
    'build_ff_inputs': run_build_ff_inputs,
    'fit_and_validate': run_fit_and_validate,
    'downselect': run_downselect,
    'uq_single_points': run_uq_single_points,
    'select_dft_candidates': run_select_dft_candidates,
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
