"""
Single entry point for the Consolidated_Pipeline: RMG DFT convergence ->
finite-temperature MD structure sampling -> MACE ensemble-input construction
-> MACE ensemble fitting -> validation single points -> finite-temperature MD
-> FF downselection -> trajectory-frame unpacking -> UQ single points -> DFT
candidate selection.

Run one stage at a time -- each stage is its own compute allocation/
apptainer invocation (converge_dft_data, fit_and_validate, and
uq_single_points submit MatEnsemble/Flux chores; sample_ft_md_structures,
build_ff_inputs, downselect, and select_dft_candidates are pure analysis/
file-ops and can run anywhere EnsembleFFFit's analysis extras are installed)
-- or run the whole ordered sequence in one process via --stage all:

    python run_pipeline.py --config workflow_config.yaml --stage converge_dft_data
    python run_pipeline.py --config workflow_config.yaml --stage sample_ft_md_structures
    python run_pipeline.py --config workflow_config.yaml --stage build_ff_inputs
    python run_pipeline.py --config workflow_config.yaml --stage fit_and_validate
    python run_pipeline.py --config workflow_config.yaml --stage downselect
    python run_pipeline.py --config workflow_config.yaml --stage uq_single_points
    python run_pipeline.py --config workflow_config.yaml --stage select_dft_candidates
    python run_pipeline.py --config workflow_config.yaml --stage all

matensemble is only imported inside the three stage functions that actually
need it (converge_dft_data, fit_and_validate, uq_single_points), not at
module load time, so the other four stages don't require the Flux container.
best_force_field's sklearn dependency is similarly deferred into
run_downselect, since it's unconfirmed whether every environment running
the other stages has sklearn installed.

converge_dft_data's RMG-specific environment/node-sizing logic (below) is
deliberately kept inline here rather than moved into EnsembleFFFit -- it's
tied to this one DFT backend/container, and will need to look different for
a different DFT driver, same reasoning as why the MACE/ASE drivers stay
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
from EnsembleFFFit.analysis.variance import format_candidate_table, get_structures_scores, select_structures
from EnsembleFFFit.utilities.general import mirror_completed_leaves, print_and_write, unpack_trajectory_frames

# Frontier/Apptainer-specific runtime library path for the rmg-gpu subprocess
# specifically -- NOT exported in the surrounding shell, only ever passed via
# a chore's/Resources' own `env`, so it never reaches MatEnsemble/Flux's own
# Python process (doing that broke `flux start` and `fi_info` in testing --
# see build_apptainer_RMG_Frontier.md). Cray-specific dirs come first so the
# real Cray libfabric/MPICH/FFTW/HDF5/LibSci win over any same-named
# container-native library; the container's own multiarch dirs come next to
# protect glibc/libstdc++/libssl from the host's SLES15 versions; the host
# /usr/lib64 bind (for libcxi.so.1/libnl-3.so.200) comes last so it only ever
# fills in what the container genuinely lacks. Re-verify every path here
# against build_apptainer_RMG_Frontier.md if the container image, ROCm
# version, or Cray module versions change.
RMG_LD_LIBRARY_PATH = (
    "/opt/cray/pe/lib64:/opt/cray/libfabric/2.3.1/lib64:/opt/cray/pals/1.8/lib:"
    "/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:"
    "/opt/rocm-6.3.3/lib:/opt/xpmem/2.7.4/lib:"
    "/sw/frontier/spack-envs/cpe24.03-cpu/opt/gcc-13.2/boost-1.85.0-3gvl6ws5xm7hrfkeyevgh45bv434ceol/lib:"
    "/sw/frontier/spack-envs/base/opt/linux-sles15-x86_64/gcc-7.5.0/bzip2-1.0.8-st7di5r4yikef76nw4xenvocycgp3god/lib:"
    "/opt/cray_extra/lib64"
)


def rmg_container_env(cores_per_task, rmg_num_threads):
    """Env for the rmg-gpu chore/task specifically -- see RMG_LD_LIBRARY_PATH docstring.

    OMP_NUM_THREADS is derived from cores_per_task (not a second, separately
    hardcoded number) so the two can't quietly drift out of agreement.
    RMG_NUM_THREADS is intentionally independent -- Frontier's own reference
    job script sets it lower than OMP_NUM_THREADS/cores_per_task for
    cache-effect reasons, not by mistake. OMP_PROC_BIND=false works around a
    confirmed GOMP affinity bug under Apptainer -- OMP_PLACES must stay
    genuinely unset (not an empty string, which crashes libgomp).
    """
    from EnsembleFFFit.utilities.general import ensemble_fffit_pythonpath

    return {
        'PYTHONPATH': ensemble_fffit_pythonpath(),
        'OMP_PROC_BIND': 'false',
        'OMP_NUM_THREADS': str(cores_per_task),
        'RMG_NUM_THREADS': str(rmg_num_threads),
        'MPICH_OFI_NIC_POLICY': 'NUMA',
        'MPICH_GPU_SUPPORT_ENABLED': '0',
        'LD_LIBRARY_PATH': RMG_LD_LIBRARY_PATH,
    }


def _require(cfg, stage_name, keys):
    missing = [k for k in keys if cfg.get(k) is None]
    if missing:
        raise ValueError(f"Missing required key(s) {missing} under '{stage_name}' in the config file.")


# ---------------------------------------------------------------------------
# Stage: converge_dft_data (Pipeline 0)
# ---------------------------------------------------------------------------

def run_converge_dft_data(config):
    """
    Submit one RMG execution chore per (structure, rmg_yaml) pair found
    under directory, each sized from that task's own computed
    'allocated_nodes' -- DFT jobs are never batched together the way MD/MACE
    tasks are, one pipe.call per job, sized per-job, with no
    Pipeline.strategy dependency chain (this stage doesn't spawn anything
    reactively -- fit_and_validate's fitting stage just expects
    directory/DFT/training and DFT/validation to already be converged by
    the time it runs).

    `directory` serves as both run_directory and inputs_directory for
    DFTMatEnsemble. build_full_runs anchors its task_dir computation on
    whatever check_files points at (default "rmg_yaml") -- safe whether
    directory is a single structure leaf or a parent tree spanning multiple
    structures against one shared recipe, since either way exactly one
    recipe file is expected in the tree. Don't override check_files to
    "structure_filename" for the multiple-structures case: since
    structure_filename would then match once per structure on *both* sides
    of build_full_runs' nested loop, it produces an N^2 cross-product
    instead of N.
    """
    from matensemble.pipeline import Pipeline
    from matensemble.model import Resources
    from EnsembleFFFit.base import DFTMatEnsemble

    dft_cfg = config['converge_dft_data']
    _require(dft_cfg, 'converge_dft_data', ['directory', 'dft_task'])

    directory = dft_cfg['directory']
    dft_task = str(Path(dft_cfg['dft_task']).resolve())
    structure_filename = dft_cfg.get('structure_filename', 'CONTCAR')
    rmg_yaml_name = dft_cfg.get('rmg_yaml_name', 'vdW_quench.yml')
    entry_point = dft_cfg.get('entry_point', 'run_rmg_calculation')
    check_files = dft_cfg.get('check_files', ['rmg_yaml'])
    finished_file = dft_cfg.get('finished_file')
    gpus_per_node = dft_cfg.get('gpus_per_node', 8)
    cores_per_task = dft_cfg.get('cores_per_task', 7)
    rmg_num_threads = dft_cfg.get('rmg_num_threads', 5)
    max_tasks_per_job = dft_cfg.get('max_tasks_per_job')

    options = {
        'structure_filename': structure_filename,
        'rmg_yaml': rmg_yaml_name,
        'gpus_per_node': gpus_per_node,
    }

    dft = DFTMatEnsemble(directory, directory, **options)
    dft_dct_list = dft.build_dft_dcts(check_files, finished_file=finished_file)

    if not dft_dct_list:
        # build_full_runs itself is cheap (just directory walks/proximity matching) --
        # the expensive part (Structure.from_file + compute_grid_and_resources) only
        # runs in build_dft_dcts' own loop over whatever survives finished_file
        # filtering. Re-running the cheap half unfiltered here, purely to give a
        # precise error, doesn't meaningfully duplicate the expensive work above.
        recipe_keys, structure_keys = ['rmg_yaml'], ['structure_filename']
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
                f"Found {len(raw_task_dirs)} (structure, rmg_yaml) pair(s) under {directory}, but every "
                f"one of them already has a '{finished_file}' match in its working directory -- there's "
                f"nothing new to run. Sample working director{'y' if len(sample) == 1 else 'ies'}: "
                f"{sample}{', ...' if len(raw_task_dirs) > 3 else ''}. Set finished_file to null (or a "
                f"pattern that doesn't already exist everywhere) to rerun them anyway."
            )
        raise ValueError(f"No (structure, rmg_yaml) pairs found under {directory}")

    pipe = Pipeline()
    env = rmg_container_env(cores_per_task, rmg_num_threads)

    @pipe.chore(name="run_rmg", num_tasks=1, cores_per_task=cores_per_task, gpus_per_task=1,
                env=env, inherit_env=True, mpi=False)
    def run_rmg_chore(task_dict):
        """Thin chore wrapper -- the real RMG-execution logic lives in DFTMatEnsemble.run_individual."""
        return DFTMatEnsemble.run_individual(task_dict)

    for dft_dct in dft_dct_list:
        task_dict = {**dft_dct, 'dft_task': dft_task, 'entry_point': entry_point}

        # Sized per-job from this task's own computed allocated_nodes, not a
        # single global value -- different structures/recipes can legitimately
        # need different node counts, and each pipe.call carries its own
        # Resources override for exactly this reason.
        num_tasks = dft_dct['allocated_nodes'] * gpus_per_node
        if max_tasks_per_job is not None and num_tasks > max_tasks_per_job:
            # Clamp num_tasks directly to max_tasks_per_job -- RMG's own AutoSet.cpp
            # already re-derives processor_grid at runtime whenever the joined
            # communicator size doesn't match the grid's product, so it can genuinely
            # run with fewer resources than this build-time estimate wanted, including
            # fewer than one full node's worth of GPUs (e.g. max_tasks_per_job=1 for a
            # single-MPI-task diagnostic run -- Resources() itself has no requirement
            # that num_tasks be a multiple of gpus_per_node, only RMG's own
            # allocated_nodes bookkeeping cares, handled below). allocated_nodes is set
            # to the smallest node count that could plausibly host num_tasks (ceiling
            # division, not floor -- floor gave 0 whenever max_tasks_per_job <
            # gpus_per_node, an actual bug: Resources(num_tasks=0, ...) is invalid),
            # since RMG.write_input's own consistency check compares against this value.
            clamped_num_tasks = max_tasks_per_job
            clamped_allocated_nodes = max(1, -(-clamped_num_tasks // gpus_per_node))  # ceil division
            print(f"WARNING: {task_dict['structure_filename']}: auto-computed "
                  f"allocated_nodes*gpus_per_node = {num_tasks} tasks exceeds "
                  f"max_tasks_per_job={max_tasks_per_job}; falling back to {clamped_num_tasks} "
                  f"task(s) ({clamped_allocated_nodes} node(s) worth) instead of the auto-computed value.",
                  file=sys.stderr)
            num_tasks = clamped_num_tasks
            task_dict['allocated_nodes'] = clamped_allocated_nodes
        # mpi=False (Flux's default -- no "mpi"="pmi2" shell option) deliberately
        # matches the confirmed-working manual invocation: `flux run --ntasks=8
        # --cores-per-task=7 --gpus-per-task=1 ... rmg-gpu` run directly inside a
        # `flux start` instance, with NO -o mpi=... option set. (The --mpi=pmi2
        # seen in that workflow belongs to the OUTER `srun --external-launcher
        # --mpi=pmi2 ... flux start ...` that bootstraps the Flux broker itself --
        # a completely different thing from this inner job's own shell options.)
        # mpi=True was tried here first (forcing pmi2 on the inner job) on the
        # theory that Flux was launching num_tasks uncoordinated
        # runtime_worker.py/rmg-gpu processes with zero PMI wiring between them --
        # it produced a measurable partial change (single RMG startup banner
        # instead of several) but did NOT fix the underlying
        # HSA/hipErrorInvalidDeviceFunction failures, and the manual command
        # proves pmi2 was never necessary in the first place. Left as mpi=False
        # to exactly match the known-good baseline while the real cause is still
        # being isolated.
        resources = Resources(num_tasks=num_tasks, cores_per_task=cores_per_task, gpus_per_task=1,
                               env=env, inherit_env=True, mpi=False)
        pipe.call("run_rmg", task_dict, resources=resources)

    # set_gpu_affinity intentionally omitted (defaults to Flux's own choice) --
    # the manual command sets no -o gpu-affinity=... option either, and forcing
    # it here was confirmed to inject a per-task CUDA_VISIBLE_DEVICES (which
    # this ROCm/HIP binary doesn't consult) with no equivalent restriction on
    # ROCR_VISIBLE_DEVICES -- an unaccounted-for divergence from the working
    # baseline, dropped here while isolating the real failure cause.
    future = pipe.submit(log_delay=25)
    future.result()


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


def _copy_starting_structures(paths, source_root, dest_root, structure_filename="POSCAR"):
    """Copy structure_filename from each of `paths` into dest_root, mirroring
    each path's position relative to source_root."""
    copied = []
    for path in paths:
        rel = os.path.relpath(path, source_root)
        dest_dir = os.path.join(dest_root, rel)
        os.makedirs(dest_dir, exist_ok=True)
        shutil.copy2(os.path.join(path, structure_filename), os.path.join(dest_dir, structure_filename))
        copied.append(dest_dir)
    return copied


def run_sample_ft_md_structures(config):
    """
    Sample a fixed total number of structures from converge_dft_data's
    training subtree, spread as evenly as possible across distinct mp-ids,
    to use as starting structures for finite-temperature MD with the
    foundation model. Ported from the standalone
    sample_training_structures.py script into a proper stage, matching the
    same single-entry-point consolidation as every other stage here --
    still deliberately its own stage/function (not folded into
    fit_and_validate or moved into EnsembleFFFit), since which structures
    feed finite-temperature MD is exactly the kind of site-specific
    selection logic a user should be able to read and adjust directly.

    source_root defaults to converge_dft_data.directory/training when null.
    dest_root defaults to finite_temperature_md.ase_inputs_directory/structures
    when null -- structures live inside ase_inputs_directory (the same
    convention MD_single_points/MD_uq_single_points use: the driver script
    and its structures share one inputs_directory), not in a separately
    tracked directory, so there's one place (not two) to point at each.
    """
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

    selected = _sample_evenly_by_mpid(source_root, total, structure_filename=structure_filename, seed=seed)
    print(f"Sampled {len(selected)} structure(s) across mp-ids under {source_root}")

    copied = _copy_starting_structures(selected, source_root, dest_root, structure_filename=structure_filename)
    for d in copied:
        print(f"  -> {d}")


# ---------------------------------------------------------------------------
# Stage: build_ff_inputs
# ---------------------------------------------------------------------------

def _discover_composition_dirs(root):
    """root/<system>/<composition>/ -> {composition: Path} for every
    system/composition pair found exactly two levels under root."""
    dirs = {}
    root = Path(root)
    if not root.is_dir():
        return dirs
    for system_dir in sorted(root.iterdir()):
        if not system_dir.is_dir():
            continue
        for comp_dir in sorted(system_dir.iterdir()):
            if comp_dir.is_dir():
                dirs[comp_dir.name] = comp_dir
    return dirs


def run_build_ff_inputs(config):
    """
    Builds the randomly-sampled ensemble of MACE training-input folders
    (one per composition-combo x weight-combo sample) that fit_and_validate
    reads from -- per-composition training .xyz (one per
    DFT/training/Defects/<system>/<composition>/ subtree, each a combinable
    label), a fixed combined validation .xyz (DFT/validation/EoS, never
    itself split/sampled), isolated-atom E0s (DFT/isolated_elements), then
    build_ensemble_inputs.build_mace_ensemble_inputs materializes each
    sampled (data_combo, weight_combo) pair as its own numbered folder.

    Reuses EnsembleFFFit.potential.mace.write_training_xyz (DFT run ->
    extxyz) and .build_ensemble_inputs (the combinatorial sampling itself)
    rather than re-deriving either -- this stage's own job is discovering
    this project's composition directories and wiring the pieces together.
    Ported from the site-specific test/Full_Pipeline/run/FF/build_ff_inputs.py
    driver.
    """
    from EnsembleFFFit.potential.mace.write_training_xyz import write_training_xyz
    from EnsembleFFFit.potential.mace.build_ensemble_inputs import get_isolated_atom_e0s, build_mace_ensemble_inputs

    cfg = config['build_ff_inputs']
    _require(cfg, 'build_ff_inputs', ['dft_root'])

    dft_root = Path(cfg['dft_root'])
    mace_inputs_dir = Path(cfg.get('mace_inputs_dir') or config['fine_tuning']['inputs_directory'])
    structure_filename = cfg.get('structure_filename', 'POSCAR')

    training_defects_root = dft_root / cfg.get('training_subpath', 'training/Defects')
    validation_eos_root = dft_root / cfg.get('validation_subpath', 'validation/EoS')
    isolated_elements_root = dft_root / cfg.get('isolated_elements_subpath', 'isolated_elements')

    xyz_root = mace_inputs_dir.parent / "xyz_data"

    # 1. Per-composition training .xyz -- each becomes one combinable label.
    composition_dirs = _discover_composition_dirs(training_defects_root)
    print(f"Discovered {len(composition_dirs)} composition(s): {sorted(composition_dirs)}")

    training_xyz_by_label = {}
    for comp, comp_dir in composition_dirs.items():
        out_path = xyz_root / "training" / "Defects" / f"{comp}.xyz"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists():
            out_path.unlink()  # write_training_xyz appends -- start clean each run
        written = write_training_xyz(comp_dir, out_path, structure_filename=structure_filename)
        print(f"  {comp}: {len(written)} completed run(s) under {comp_dir} -> {out_path}")
        if written:
            training_xyz_by_label[comp] = str(out_path)

    if not training_xyz_by_label:
        raise RuntimeError(f"No completed training runs found under {training_defects_root} -- "
                            f"has the DFT stage actually finished?")

    # 2. Fixed, combined validation .xyz -- every composition together, not
    # split into separate labels (every ensemble member gets the same full
    # validation set, never sampled/combined per-composition).
    val_out_path = xyz_root / "validation" / "EoS" / "validation.xyz"
    val_out_path.parent.mkdir(parents=True, exist_ok=True)
    if val_out_path.exists():
        val_out_path.unlink()
    val_written = write_training_xyz(validation_eos_root, val_out_path, structure_filename=structure_filename)
    print(f"Validation: {len(val_written)} completed run(s) under {validation_eos_root} -> {val_out_path}")
    if not val_written:
        raise RuntimeError(f"No completed validation runs found under {validation_eos_root}")

    # 3. Isolated-atom E0s.
    e0s = get_isolated_atom_e0s(isolated_elements_root, properties_filename="properties.json")
    print(f"Isolated-atom E0s (atomic number -> energy): {e0s}")
    if not e0s:
        raise RuntimeError(f"No completed isolated-atom runs found under {isolated_elements_root}")

    # 4. Randomly-sampled ensemble of mace_inputs folders. base_config's
    # overrides get baked into each folder's own config.yml once, here --
    # fit_and_validate has no need to also carry/reapply them, since by the
    # time fitting runs, everything it needs is already in that config.yml.
    base_config = cfg.get('mace_overrides')

    written_folders = build_mace_ensemble_inputs(
        training_xyz_by_label=training_xyz_by_label,
        validation_xyz_paths=[str(val_out_path)],
        output_dir=mace_inputs_dir,
        e0s=e0s,
        total_cap=cfg.get('total_cap', 100),
        seed=cfg.get('seed', 0),
        weight_grid_kwargs={
            "energy_num": cfg.get('energy_num', 3),
            "force_num": cfg.get('force_num', 3),
            "stress_num": cfg.get('stress_num', 1),
        },
        base_config=base_config,
    )
    print(f"Wrote {len(written_folders)} mace_inputs folder(s) under {mace_inputs_dir}")


# ---------------------------------------------------------------------------
# Stage: fit_and_validate (Pipeline 1)
# ---------------------------------------------------------------------------

def run_fit_and_validate(config):
    """
    Mirrors completed DFT validation results into MD_single_points'
    inputs_directory, then builds and submits a Pipeline with: MACE
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
    from EnsembleFFFit.base import MACEMatEnsemble, MDMatEnsemble
    from EnsembleFFFit.utilities.general import ensemble_fffit_pythonpath, make_mirrored_rename_dest_path_fn

    fine_tuning_cfg = config['fine_tuning']
    copy_cfg = config['copy_force_fields']
    md_cfg = config['MD_single_points']
    ft_md_cfg = config.get('finite_temperature_md')
    mirror_cfg = config.get('validation_mirroring')

    _require(fine_tuning_cfg, 'fine_tuning', ['run_directory', 'inputs_directory'])
    _require(copy_cfg, 'copy_force_fields', ['target_directory'])
    _require(md_cfg, 'MD_single_points', ['inputs_directory'])
    if ft_md_cfg:
        _require(ft_md_cfg, 'finite_temperature_md',
                  ['output_directory', 'foundation_model', 'in_file', 'ase_inputs_directory'])

    if mirror_cfg:
        _require(mirror_cfg, 'validation_mirroring', ['source_root', 'dest_root'])
        copied = mirror_completed_leaves(mirror_cfg['source_root'], mirror_cfg['dest_root'],
                                          tuple(mirror_cfg.get('filenames', ['POSCAR', 'properties.json'])))
        print(f"Mirrored {len(copied)} completed leaf(ves) from {mirror_cfg['source_root']} to {mirror_cfg['dest_root']}")

    mace_run_directory = fine_tuning_cfg['run_directory']
    md_run_directory = copy_cfg['target_directory']
    md_inputs_directory = md_cfg['inputs_directory']
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

    pipe = Pipeline()

    # Registration order matters: Pipeline.strategy()'s bolo_list is checked
    # against the registry immediately at decoration time, so "fit_mace"
    # must already be registered before copy_and_spawn_md is decorated.

    @pipe.chore(name="fit_mace",
                num_tasks=fine_tuning_cfg.get('num_tasks', 1),
                cores_per_task=fine_tuning_cfg.get('cores_per_task', 1),
                gpus_per_task=fine_tuning_cfg.get('gpus_per_task', 1))
    def fit_mace_chore(overrides):
        """Thin chore wrapper -- the real MACE-fitting logic lives in MACEMatEnsemble.run_individual."""
        return MACEMatEnsemble.run_individual(overrides)

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
                          resources=Resources(**md_resources_kwargs))

    fine_tuning_options = {'foundation_model': fine_tuning_cfg.get('foundation_model', 'model.model'),
                           'config': fine_tuning_cfg.get('config'),
                           'train_file': fine_tuning_cfg.get('train_file', 'train.xyz'),
                           'test_file': fine_tuning_cfg.get('test_file', 'test.xyz')}
    fine_tuning_options = {k: v for k, v in fine_tuning_options.items() if v is not None}

    mace_matensemble = MACEMatEnsemble(mace_run_directory, fine_tuning_cfg['inputs_directory'], **fine_tuning_options)
    check_files = fine_tuning_cfg.get('check_files', ['foundation_model'])
    # Deliberately finished_file=None here: every variant must get a fit_mace
    # chore submitted, even already-finished ones, so its completion still
    # triggers copy_and_spawn_md -- the skip-if-already-fit check happens at
    # execution time inside MACEMatEnsemble.run_individual instead.
    mace_arg_dict_list = mace_matensemble.build_mace_dcts(check_files, finished_file=None)
    mace_finished_file = fine_tuning_cfg.get('finished_file')
    if mace_finished_file:
        for overrides in mace_arg_dict_list:
            overrides['finished_file'] = mace_finished_file

    fitting_resources = Resources(num_tasks=fine_tuning_cfg.get('num_tasks', 1),
                                   cores_per_task=fine_tuning_cfg.get('cores_per_task', 1),
                                   gpus_per_task=fine_tuning_cfg.get('gpus_per_task', 1),
                                   env={'PYTHONPATH': ensemble_fffit_pythonpath()},
                                   inherit_env=True)

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
        if not ft_md_task_dicts:
            raise ValueError(
                f"finite_temperature_md is configured but no structures were found under "
                f"{ft_md_structures_root} -- did you run the sample_ft_md_structures stage first? "
                f"(Silently submitting zero run_ft_md chores here previously masked exactly this: an "
                f"empty structures directory produced no error, just no finite-temperature MD.)"
            )
        ft_md_resources = Resources(**ft_md_resources_kwargs,
                                     env={'PYTHONPATH': ensemble_fffit_pythonpath()}, inherit_env=True)
        for task_dict in ft_md_task_dicts:
            pipe.call("run_ft_md", task_dict, resources=ft_md_resources)

    future = pipe.submit(log_delay=25, set_gpu_affinity=True)
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

    ff_cfg = config['downselect_force_fields']
    _require(ff_cfg, 'downselect_force_fields', ['force_fields_dir', 'ase_inputs_dir', 'dest_dir'])

    ff_dct = parse_labeled_tree(ff_cfg['force_fields_dir'])
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

    traj_cfg = config.get('trajectory_unpacking')
    if traj_cfg:
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
    """
    from matensemble.pipeline import Pipeline
    from matensemble.model import Resources
    from EnsembleFFFit.base import MDMatEnsemble
    from EnsembleFFFit.utilities.general import ensemble_fffit_pythonpath

    uq_cfg = config['MD_uq_single_points']
    _require(uq_cfg, 'MD_uq_single_points', ['run_directory', 'inputs_directory'])

    options = {'ffield': uq_cfg.get('ffield', 'model.model'),
               'in_file': uq_cfg.get('in_file'),
               'control': uq_cfg.get('control'),
               'structure': uq_cfg.get('structure', 'POSCAR'),
               'lammps_task': uq_cfg.get('lammps_task', 'ase_mace.py')}
    options = {k: v for k, v in options.items() if v is not None}

    resources_kwargs = dict(num_tasks=uq_cfg.get('num_tasks', 1),
                             cores_per_task=uq_cfg.get('cores_per_task', 1),
                             gpus_per_task=uq_cfg.get('gpus_per_task', 1))

    md = MDMatEnsemble(uq_cfg['run_directory'], uq_cfg['inputs_directory'], **options)
    task_dicts = md.build_task_dicts(
        uq_cfg.get('lammps_task', 'ase_mace.py'),
        uq_cfg.get('parent_levels', 8),
        uq_cfg.get('check_files', ['ffield']),
        uq_cfg.get('entry_point', 'run_ase_single_points'),
        finished_file=uq_cfg.get('finished_file'),
    )
    if not task_dicts:
        raise ValueError("No (force field, structure) batches found -- check run_directory/inputs_directory "
                          "in the 'MD_uq_single_points' config section.")

    pipe = Pipeline()

    @pipe.chore(name="run_uq_ase", **resources_kwargs)
    def run_uq_ase_chore(task_dict):
        """Thin chore wrapper -- the real MD-execution logic lives in MDMatEnsemble.run_individual."""
        return MDMatEnsemble.run_individual(task_dict)

    resources = Resources(**resources_kwargs, env={'PYTHONPATH': ensemble_fffit_pythonpath()}, inherit_env=True)
    for task_dict in task_dicts:
        pipe.call("run_uq_ase", task_dict, resources=resources)

    future = pipe.submit(log_delay=25, set_gpu_affinity=True)
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
    cfg = config['dft_candidate_selection']
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
