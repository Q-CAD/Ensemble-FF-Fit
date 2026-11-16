"""
Single entry point for the MACE -> ACE -> LAMMPS distillation example:
a fine-tuned MACE potential (the teacher, provided by path -- fine-tuning
itself is a separate, prior step not covered here) labels training data,
an ensemble of ACE potentials is fit against it, and the best performer is
selected by combined energy+force validation.

This file is scaffolding for an incremental port -- see this example's
README for the full stage list and which stages exist yet. Only
teacher_preparation is implemented so far.

Run one stage at a time -- each stage is its own compute allocation/
container invocation if it submits MatEnsemble/Flux chores, or can run
anywhere EnsembleFFFit's dependencies are installed if it's pure
analysis/file-ops (teacher_preparation is the latter: a single
subprocess call, no Flux submission) -- or run the whole implemented
sequence in one process via --stage all:

    python run_pipeline.py --config workflow_config.yaml --stage teacher_preparation
    python run_pipeline.py --config workflow_config.yaml --stage all
"""
import argparse
import csv
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


def _require(cfg, stage_name, keys):
    # cfg is None whenever the whole section is commented out/absent --
    # report that plainly rather than crashing on cfg.get() with an
    # unhelpful AttributeError.
    if cfg is None:
        raise ValueError(f"Missing required section '{stage_name}' in the config file.")
    missing = [k for k in keys if cfg.get(k) is None]
    if missing:
        raise ValueError(f"Missing required key(s) {missing} under '{stage_name}' in the config file.")


def lammps_container_env():
    """
    Env for GPU-backed MACE/MD chores. CONFIRMED (against the companion
    Pd-Se pipeline) the container's own default LD_LIBRARY_PATH can put a
    bundled "forward compatibility" libcuda.so.1 shim ahead of where
    podman-hpc's own prestart hook injects the REAL driver -- Cray MPICH's
    own yaksa GPU-datatype engine initializes CUDA at MPI_Init() time
    (which every MD/relaxation/single-point chore here triggers, even a
    single-rank one), and resolves libcuda.so.1 to the mismatched shim
    instead, failing with "CUDA Error ... unsupported display driver /
    cuda driver combination". Prepending /usr/lib64 (the container's own
    native path the driver gets injected into, not a host bind-mount)
    fixes this. Reads back the container's own current LD_LIBRARY_PATH
    (rather than hardcoding a snapshot) so everything else it needs
    (FFTW/HDF5/etc. paths) is preserved, not lost.
    """
    from EnsembleFFFit.utilities.general import ensemble_fffit_pythonpath

    return {
        'PYTHONPATH': ensemble_fffit_pythonpath(),
        'LD_LIBRARY_PATH': '/usr/lib64:' + os.environ.get('LD_LIBRARY_PATH', ''),
    }


# ---------------------------------------------------------------------------
# Shared dispatch body for run_relaxation/run_md/run_single_points
# ---------------------------------------------------------------------------

def _run_md_batch(cfg, stage_name, chore_name, default_entry_point, default_chunk_size, require_in_file):
    """
    One task_dict per structure (MDMatEnsemble.build_flat_task_dicts -- no
    proximity-matching needed, every caller has exactly one fixed
    foundation_model), chunked into chunk_size-sized groups and merged
    (MDMatEnsemble.combine_task_dicts) so each chunk runs as one chore
    instead of one-chore-per-structure. No sampling: every structure under
    structures_root gets processed.
    """
    from matensemble.pipeline import Pipeline
    from matensemble.model import Resources
    from EnsembleFFFit.base import MDMatEnsemble

    required = ['structures_root', 'foundation_model', 'task_command', 'output_root']
    if require_in_file:
        required.append('in_file')
    _require(cfg, stage_name, required)

    task_dicts = MDMatEnsemble.build_flat_task_dicts(
        structures_root=cfg['structures_root'],
        foundation_model=cfg['foundation_model'],
        in_file=cfg.get('in_file'),
        output_root=cfg['output_root'],
        task_command=cfg['task_command'],
        entry_point=cfg.get('entry_point', default_entry_point),
        structure_filename=cfg.get('structure_filename', 'POSCAR'),
        finished_file=cfg.get('finished_file', 'properties.json'),
    )
    if not task_dicts:
        raise ValueError(f"No structures found under {cfg['structures_root']} -- "
                          f"did you run the generation stage for this tier first?")

    chunk_size = cfg.get('chunk_size', default_chunk_size)
    chunks = [task_dicts[i:i + chunk_size] for i in range(0, len(task_dicts), chunk_size)]
    print(f"{stage_name}: {len(task_dicts)} structure(s) in {len(chunks)} chunk(s) of up to {chunk_size}")

    resources_kwargs = dict(num_tasks=cfg.get('num_tasks', 1),
                             cores_per_task=cfg.get('cores_per_task', 1),
                             gpus_per_task=cfg.get('gpus_per_task', 1))

    # reserve_broker_node=False: matensemble's own default (None) reserves
    # an entire broker rank for its own controller once the Flux instance
    # is multi-rank (true for any multi-node launch_multi_node.slurm
    # allocation), which would otherwise silently shrink how many GPUs are
    # actually schedulable.
    pipe = Pipeline(reserve_broker_node=False)

    @pipe.chore(name=chore_name, **resources_kwargs)
    def _chore(task_dict):
        """Thin chore wrapper -- the real execution logic lives in MDMatEnsemble.run_individual."""
        return MDMatEnsemble.run_individual(task_dict)

    resources = Resources(**resources_kwargs, env=lammps_container_env(), inherit_env=True)
    for chunk in chunks:
        combined = MDMatEnsemble.combine_task_dicts(chunk)
        pipe.call(chore_name, combined, resources=resources)

    future = pipe.submit(log_delay=25, set_gpu_affinity=True)
    future.result()


# ---------------------------------------------------------------------------
# Stage: generate_structures -- any dynamically-imported structure generator
# ---------------------------------------------------------------------------

def generate_structures(config, stage_key):
    """
    Fully generic: dynamically imports whichever generator script
    `generation_task` names (same import_module_from_path/swappable-driver
    convention used throughout this example), looks up
    `generation_entry_point` on it, and calls it with exactly whatever
    keyword arguments `generator_args` supplies in the config -- this
    function has NO knowledge of any one generator's own parameter shape
    (pyxtal's compositions/space_groups/seeds vs. the defect generator's
    concentrations/n_placements vs. the motif generator's phase_labels/
    n_seeds all pass through unchanged), so a new generator with a
    completely different signature is a config change, not a code change.

    `scale: smoke_test | full` (read from the TOP-LEVEL config, not this
    stage's own block) selects which of `generator_args_smoke_test` /
    `generator_args_full` is used, if both are present; a plain
    `generator_args` (no smoke_test/full split) is used as-is regardless
    of `scale`, for a stage variant that doesn't need two sizes (e.g. one
    that's already cheap).
    """
    from EnsembleFFFit.utilities.general import import_module_from_path

    cfg = config[stage_key]
    _require(cfg, stage_key, ['generation_task', 'generation_entry_point'])

    scale = config.get('scale', 'smoke_test')
    scaled_key = f'generator_args_{scale}'
    generator_args = cfg.get(scaled_key, cfg.get('generator_args'))
    if generator_args is None:
        raise ValueError(f"{stage_key}: no '{scaled_key}' or 'generator_args' found in config "
                          f"(top-level scale={scale!r}).")

    module = import_module_from_path(f'{stage_key}_generation_task', cfg['generation_task'])
    entry_point = getattr(module, cfg['generation_entry_point'])

    result = entry_point(**generator_args)
    print(f"{stage_key}: {result}")


def generate_structures_tier1(config):
    """Thin wrapper -- see generate_structures_tier1's own comment in workflow_config.yaml."""
    return generate_structures(config, stage_key='generate_structures_tier1')


def generate_structures_tier3_defect(config):
    """Thin wrapper -- see generate_structures_tier3_defect's own comment in workflow_config.yaml."""
    return generate_structures(config, stage_key='generate_structures_tier3_defect')


def generate_structures_tier4_defect(config):
    """Thin wrapper -- see generate_structures_tier4_defect's own comment in workflow_config.yaml."""
    return generate_structures(config, stage_key='generate_structures_tier4_defect')


def generate_structures_tier4_motif(config):
    """Thin wrapper -- see generate_structures_tier4_motif's own comment in workflow_config.yaml."""
    return generate_structures(config, stage_key='generate_structures_tier4_motif')


def generate_isolated_elements(config):
    """Thin wrapper -- see generate_isolated_elements's own comment in workflow_config.yaml."""
    return generate_structures(config, stage_key='generate_isolated_elements')


# ---------------------------------------------------------------------------
# Stage: generate_validation_structures -- Materials Project query
# ---------------------------------------------------------------------------

def generate_validation_structures(config, stage_key='generate_validation_structures'):
    """
    Pulls real compounds from the Materials Project via
    Ensemble-FF-Fit's own mp_query_cli.py, invoked as a standalone
    subprocess (NOT dynamically imported -- unlike the structure
    generators above, there's only one "pull real compounds from MP"
    implementation worth having here, so swappability isn't needed).

    Requires the MP_API_KEY environment variable (get a free key at
    https://materialsproject.org/api) -- deliberately NOT read from any
    committed file or config value, so this example never needs (and must
    never be given) a real key of its own. Passed explicitly via
    mp_query_cli.py's own --MP_api_key flag, which overrides whatever that
    script's own default lookup would otherwise do.
    """
    cfg = config[stage_key]
    _require(cfg, stage_key, ['validation_output_root', 'ensemble_fffit_root'])

    api_key = os.environ.get('MP_API_KEY')
    if not api_key:
        raise RuntimeError(
            "generate_validation_structures: the MP_API_KEY environment variable is not set. "
            "Get a free Materials Project API key at https://materialsproject.org/api and "
            "export MP_API_KEY=<your key> before running this stage.")

    mp_cfg = cfg.get('mp_query', {})
    chemsys = mp_cfg.get('chemsys', ['Pd-Se'])
    query_yaml_path = mp_cfg.get('query_yaml_path',
                                  os.path.join(cfg['validation_output_root'], 'mp_query.yaml'))
    os.makedirs(os.path.dirname(query_yaml_path), exist_ok=True)
    with open(query_yaml_path, 'w') as f:
        yaml.safe_dump({'chemsys': chemsys}, f)

    mp_query_cli_path = os.path.join(
        cfg['ensemble_fffit_root'], 'EnsembleFFFit', 'structures', 'materials_project', 'mp_query_cli.py')

    cmd = [sys.executable, mp_query_cli_path,
           '--poscars_directory', cfg['validation_output_root'],
           '--MP_yaml', query_yaml_path,
           '--MP_api_key', api_key]
    if mp_cfg.get('max_atoms'):
        cmd += ['--max_atoms', str(mp_cfg['max_atoms'])]
    if mp_cfg.get('field_order'):
        cmd += ['--field_order'] + list(mp_cfg['field_order'])

    print(f"generate_validation_structures: querying Materials Project (chemsys={chemsys})")
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# Stage: downselect_structures -- general screening/downselection
# ---------------------------------------------------------------------------

def downselect_structures(config, stage_key='downselect_structures'):
    """
    See MACE/structures/downselect_structures.py's own module docstring
    for the full design (generalized from the companion pipeline's own
    tier-2-specific downselect_low_energy_structures: arbitrary
    training_root, metadata-field group_by instead of path parsing,
    optional rescale_output post-step).
    """
    from EnsembleFFFit.utilities.general import import_module_from_path

    cfg = config[stage_key]
    _require(cfg, stage_key,
              ['training_root', 'isolated_elements_root', 'validation_root',
               'output_root', 'downselection_task', 'group_by', 'n_per_group'])

    module = import_module_from_path(f'{stage_key}_downselection_task', cfg['downselection_task'])
    entry_point = getattr(module, cfg.get('downselection_entry_point', 'downselect_structures'))

    result = entry_point(
        training_root=cfg['training_root'],
        isolated_elements_directory=cfg['isolated_elements_root'],
        validation_root=cfg['validation_root'],
        output_root=cfg['output_root'],
        group_by=list(cfg['group_by']),
        n_per_group=cfg['n_per_group'],
        rank_by=cfg.get('rank_by', 'energy_per_atom'),
        rescale_output=cfg.get('rescale_output'),
        structure_filename=cfg.get('structure_filename', 'POSCAR'),
        max_abs_energy_per_atom=cfg.get('max_abs_energy_per_atom', 50.0),
        max_abs_force_component=cfg.get('max_abs_force_component', 50.0),
        stability_floor_margin=cfg.get('stability_floor_margin', 0.5),
    )
    print(f"{stage_key}: wrote {len(result['written'])} structure(s) under {cfg['output_root']}")


def downselect_structures_tier2(config):
    """Thin wrapper -- see downselect_structures_tier2's own comment in workflow_config.yaml."""
    return downselect_structures(config, stage_key='downselect_structures_tier2')


# ---------------------------------------------------------------------------
# Stage: run_relaxation -- ionic relaxation (+ pre/post single points)
# ---------------------------------------------------------------------------

def run_relaxation(config, stage_key='relax_structures'):
    """
    Ionically relaxes every structure under structures_root via
    MACE/ase_inputs/ase_mace.py's own run_ase_relaxation (FIRE/
    FrechetCellFilter; also writes the pre/post-relaxation single points --
    see that function's own docstring for why). Writes IN PLACE
    (output_root == structures_root, by config convention): the relaxed
    POSCAR overwrites the raw one, properties.json/rattled_properties.json/
    metadata.json land alongside it.
    """
    cfg = config[stage_key]
    _run_md_batch(cfg, stage_key, stage_key,
                  default_entry_point='run_ase_relaxation', default_chunk_size=50,
                  require_in_file=False)


# ---------------------------------------------------------------------------
# Stage: run_md -- finite-temperature MD
# ---------------------------------------------------------------------------

def run_md(config, stage_key='finite_temperature_md'):
    """
    Runs MACE/ase_inputs/ase_mace_md.py's own run_finite_temperature_md on
    every structure under structures_root. Writes IN PLACE (output_root ==
    structures_root): md_run.traj and the aggregated per-step
    properties.json land alongside each structure's own (relaxed) POSCAR.
    `in_file` (required) is this tier's own MD-protocol json (timestep/
    temperature schedule/constant_volume/etc. -- see ase_mace_md.py's own
    docstring).
    """
    cfg = config[stage_key]
    _run_md_batch(cfg, stage_key, stage_key,
                  default_entry_point='run_finite_temperature_md', default_chunk_size=20,
                  require_in_file=True)


def run_md_tier1(config):
    """Thin wrapper -- see run_md_tier1's own comment in workflow_config.yaml."""
    return run_md(config, stage_key='run_md_tier1')


def run_md_tier2(config):
    """Thin wrapper -- see run_md_tier2's own comment in workflow_config.yaml."""
    return run_md(config, stage_key='run_md_tier2')


def run_md_tier3(config):
    """Thin wrapper -- see run_md_tier3's own comment in workflow_config.yaml."""
    return run_md(config, stage_key='run_md_tier3')


def run_md_tier4_motif_300k(config):
    """Thin wrapper -- see run_md_tier4_motif_300k's own comment in workflow_config.yaml."""
    return run_md(config, stage_key='run_md_tier4_motif_300k')


def run_md_tier4_motif_500k(config):
    """Thin wrapper -- see run_md_tier4_motif_500k's own comment in workflow_config.yaml."""
    return run_md(config, stage_key='run_md_tier4_motif_500k')


def run_md_tier4_motif_700k(config):
    """Thin wrapper -- see run_md_tier4_motif_700k's own comment in workflow_config.yaml."""
    return run_md(config, stage_key='run_md_tier4_motif_700k')


def run_md_tier4_motif_900k(config):
    """Thin wrapper -- see run_md_tier4_motif_900k's own comment in workflow_config.yaml."""
    return run_md(config, stage_key='run_md_tier4_motif_900k')


def run_md_tier4_defect(config):
    """Thin wrapper -- see run_md_tier4_defect's own comment in workflow_config.yaml."""
    return run_md(config, stage_key='run_md_tier4_defect')


# ---------------------------------------------------------------------------
# Stage: run_single_points -- MD-trajectory single points + metadata
# ---------------------------------------------------------------------------

def run_single_points(config, stage_key='single_point_structures'):
    """
    Two steps:

    (a) unpack_trajectory_frames (EnsembleFFFit.utilities.general) walks
    finite_temperature_structures_root for every md_run.traj and writes
    each frame as its own POSCAR under
    structures_root/<mirrored-relpath>/<frame-index>/.

    (b) Single-points every extracted frame with the teacher MACE model
    (MACE/ase_inputs/ase_mace.py's own run_ase_single_points), writing IN
    PLACE under structures_root (== output_root, by config convention).
    `in_file` (required) names a small json file with a `metadata` dict --
    whichever of {tier, phase, strain, defect_concentration, md_type,
    temperature} are known/constant for this whole tier (see this
    example's README for the full field list) -- written by
    run_ase_single_points as metadata.json alongside each properties.json,
    plus two values it derives itself per structure ('stoichiometry',
    'trajectory_id'). This is what lets downselect_structures and the
    later ACE-fitting stage's training_subsets selection group/filter by
    these fields without parsing directory paths.
    """
    from EnsembleFFFit.utilities.general import unpack_trajectory_frames

    cfg = config[stage_key]
    _require(cfg, stage_key, ['finite_temperature_structures_root', 'structures_root'])

    num_runs, num_frames = unpack_trajectory_frames(
        cfg['finite_temperature_structures_root'], cfg['structures_root'],
        traj_filename=cfg.get('traj_filename', 'md_run.traj'))
    print(f"{stage_key}: unpacked {num_frames} frame(s) from {num_runs} MD run(s)")

    _run_md_batch(cfg, stage_key, stage_key,
                  default_entry_point='run_ase_single_points', default_chunk_size=50,
                  require_in_file=True)


def run_single_points_tier1(config):
    """Thin wrapper -- see run_single_points_tier1's own comment in workflow_config.yaml."""
    return run_single_points(config, stage_key='run_single_points_tier1')


def run_single_points_tier2(config):
    """Thin wrapper -- see run_single_points_tier2's own comment in workflow_config.yaml."""
    return run_single_points(config, stage_key='run_single_points_tier2')


def run_single_points_tier3(config):
    """Thin wrapper -- see run_single_points_tier3's own comment in workflow_config.yaml."""
    return run_single_points(config, stage_key='run_single_points_tier3')


def run_single_points_tier4(config):
    """Thin wrapper -- see run_single_points_tier4's own comment in workflow_config.yaml."""
    return run_single_points(config, stage_key='run_single_points_tier4')


# ---------------------------------------------------------------------------
# Stage: teacher_preparation
# ---------------------------------------------------------------------------

def run_teacher_preparation(config):
    """
    Converts the fine-tuned MACE teacher model to LAMMPS's MLIAP format via
    `mace_create_lammps_model`'s own CLI (--format=mliap -- produces
    <source_model>-mliap_lammps.pt, not something this function invents),
    then stages the result under dest_root/target_name.

    `source_model` has no default -- it must point at your own already
    fine-tuned MACE model (see this example's README for how that model
    itself is produced; this stage does not repeat fine-tuning).

    Ported from the companion Pd-Se pipeline's own
    `run_stage_mace_reference_force_field`, simplified: that version staged
    into multiple dest_roots at once (a validation tree and a training
    tree, each needing its own copy for a downstream LAMMPS run that writes
    its own output back into whichever tree it was staged in) -- this
    example only needs one destination, so `dest_roots` collapses to a
    single `dest_root`.
    """
    cfg = config['teacher_preparation']
    _require(cfg, 'teacher_preparation', ['source_model', 'dest_root'])

    source_model = Path(cfg['source_model'])
    if not source_model.is_file():
        raise FileNotFoundError(
            f"teacher_preparation.source_model does not exist: {source_model} -- "
            f"point this at your own fine-tuned MACE model.")
    target_name = cfg.get('target_name', 'model-mliap_lammps.pt')
    dest_root = Path(cfg['dest_root'])

    converted_path = source_model.parent / (source_model.name + '-mliap_lammps.pt')
    if not converted_path.is_file():
        subprocess.run(['mace_create_lammps_model', str(source_model), '--format=mliap'], check=True)
    if not converted_path.is_file():
        raise RuntimeError(f"mace_create_lammps_model did not produce the expected {converted_path}")

    dest_path = dest_root / target_name
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(converted_path, dest_path)

    print(f"Converted {source_model} -> {converted_path}, staged as {dest_path}")


# ---------------------------------------------------------------------------
# Stage: build_ff_inputs / fit_and_validate -- ACE ensemble fitting over
# training-subset combinations
# ---------------------------------------------------------------------------

# Cold-start basis -- used whenever a combo's own warm_start_potential is
# unset. Small (2-body + 3-body only) relative to build_ace_ensemble_inputs'
# own DEFAULT_POTENTIAL_CONFIG, mainly in rcut (5.5, not that module's own
# 4) -- sized for real Pd-Se bond lengths (~2.3-2.8 Angstrom). Ported from
# the companion Pd-Se pipeline's own build_ff_inputs config -- confirmed
# there that a much larger body-order-5 basis failed to complete even one
# BFGS iteration across 10 members in ~3.8 hours, so this smaller basis is
# a deliberate, confirmed-to-actually-fit choice, not an arbitrary
# smoke-test shortcut.
DEFAULT_COLD_START_POTENTIAL_CONFIG = {
    "functions": {
        "ALL": {
            "nradmax_by_orders": [4, 2],
            "lmax_by_orders": [0, 1],
        },
    },
    "bonds": {
        "ALL": {
            "radbase": "ChebExpCos",
            "radparameters": [5.25],
            "rcut": 5.5,
            "dcut": 0.01,
            "NameOfCutoffFunction": "cos",
            "core-repulsion": [0.0, 5.0],
        },
    },
}


def ace_fit_env():
    """
    Env for the ACE-fitting chore. Needs EnsembleFFFit
    (ensemble_fffit_pythonpath()) -- TensorPotential does NOT need a
    PYTHONPATH entry here, unlike EnsembleFFFit: it's pip-installed into the
    image itself (see Dockerfile.ace), not PYTHONPATH-injected from a
    separate checkout. Still needs the same CUDA LD_LIBRARY_PATH fix
    lammps_container_env() above already documents (TensorFlow hits the
    identical mismatched-compat-shim driver issue at MPI_Init() time).
    Explicit, not relying on inherit_env=True alone: a chore's own env
    dict's PYTHONPATH key OVERWRITES (not merges with) whatever ambient
    PYTHONPATH the launch script's own inherited environment carries, even
    with inherit_env=True -- confirmed as a real, previously-hit bug in the
    companion Pd-Se pipeline, not a hypothetical concern -- so
    EnsembleFFFit's path must be read here and included explicitly, not
    assumed to survive inheritance.
    """
    from EnsembleFFFit.utilities.general import ensemble_fffit_pythonpath

    return {
        'PYTHONPATH': ensemble_fffit_pythonpath(),
        'LD_LIBRARY_PATH': '/usr/lib64:' + os.environ.get('LD_LIBRARY_PATH', ''),
    }


def build_ff_inputs(config, stage_key='build_ff_inputs'):
    """
    ACE ensemble fitting, generalized over MULTIPLE training-subset
    combinations in one stage call (`training_subsets_by_combo`, a list)
    rather than needing one hand-written stage/config block per combo.

    This is smart reuse, not an EFF-core change: EnsembleFFFit.potential.
    ace.build_ace_dataset.write_ace_dataset and .build_ace_ensemble_inputs.
    build_ace_ensemble_inputs already support being called in a loop, once
    per combo, with zero modification -- the only genuinely new logic is
    FF/subset_combination.py, which resolves each combo's own
    {tier, phases?} specs into the directory list write_ace_dataset's own
    training_root argument already accepts (a single path, or a list,
    pooled unweighted -- see that function's own docstring). (This
    example's own first pass deliberately has NO per-subset weighting --
    see FF/subset_combination.py's own docstring and this project's
    design notes for why.)

    For each combo: resolves its own training_root, builds its own dataset
    (write_ace_dataset, scoped to just that combo's structures), then
    samples its own seed x kappa ensemble (build_ace_ensemble_inputs,
    UNCHANGED from how a single-combo fit would already call it) into
    ace_inputs_dir/<combo_label>/<member>/ -- one level deeper than a
    single-combo fit's own flat ace_inputs_dir/<member>/, so multiple
    combos' members never collide.

    Cold start (a combo's own warm_start_potential unset/null): falls back
    to DEFAULT_COLD_START_POTENTIAL_CONFIG above. Warm start:
    potential.filename (NOT potential.initial_potential, which would
    trigger pyace's "ladder scheme" -- progressive basis-size growth --
    rather than genuine continued optimization of an already-same-size
    basis).
    """
    from EnsembleFFFit.potential.ace.build_ace_dataset import write_ace_dataset
    from EnsembleFFFit.potential.ace.build_ace_ensemble_inputs import build_ace_ensemble_inputs
    from FF.subset_combination import resolve_subsets

    cfg = config[stage_key]
    _require(cfg, stage_key,
              ['training_subsets_by_combo', 'isolated_elements_root', 'validation_root', 'ace_inputs_dir'])

    isolated_elements_root = cfg['isolated_elements_root']
    validation_root = cfg['validation_root']
    ace_inputs_dir = Path(cfg['ace_inputs_dir'])
    elements = cfg.get('elements', ['Pd', 'Se'])
    structure_filename = cfg.get('structure_filename', 'POSCAR')
    max_abs_energy_per_atom = cfg.get('max_abs_energy_per_atom', 50.0)
    max_abs_force_component = cfg.get('max_abs_force_component', 50.0)
    stability_floor_margin = cfg.get('stability_floor_margin', 0.5)
    evaluator = cfg.get('evaluator', 'pyace')
    cores_per_task = cfg.get('cores_per_task', 1)

    combos = cfg['training_subsets_by_combo']
    if not combos:
        raise ValueError(f"{stage_key}.training_subsets_by_combo must list at least one combo.")

    for combo in combos:
        label = combo['label']
        combo_dir = ace_inputs_dir / label
        training_root = resolve_subsets(combo['subsets'])
        print(f"{stage_key}: combo {label!r} resolves to training_root={training_root}")

        dataset_path = combo_dir / 'ace_dataset' / 'full_dataset.pckl.gzip'
        validation_dataset_path = combo_dir / 'ace_dataset' / 'validation_dataset.pckl.gzip'

        df, dataset_path = write_ace_dataset(
            run_directory=training_root,
            isolated_elements_directory=isolated_elements_root,
            output_path=str(dataset_path),
            structure_filename=structure_filename,
            max_abs_energy_per_atom=max_abs_energy_per_atom,
            max_abs_force_component=max_abs_force_component,
            validation_directory=validation_root,
            stability_floor_margin=stability_floor_margin,
            validation_output_path=str(validation_dataset_path),
        )
        counts = df.attrs['filter_counts']
        print(f"{stage_key}: combo {label!r} -- wrote {len(df)} structure(s) to {dataset_path} "
              f"({counts['kept']}/{counts['considered']} kept)")

        # Absolute -- ace_fit.py chdirs into each member's own work_dir
        # before GeneralACEFit reads data.filename, so a relative path here
        # would resolve against the wrong directory at fit time.
        dataset_path = Path(dataset_path).resolve()
        resolved_validation_dataset_path = df.attrs.get('validation_dataset_path')
        if resolved_validation_dataset_path:
            resolved_validation_dataset_path = str(Path(resolved_validation_dataset_path).resolve())

        if evaluator == 'tensorpot':
            backend_config = {
                'evaluator': 'tensorpot',
                'gpu_config': {'gpu_ind': combo.get('gpu_index', 0), 'mem_limit': combo.get('gpu_mem_limit', 0)},
                'batch_size': combo.get('batch_size', 10),
            }
        else:
            backend_config = {'evaluator': 'pyace', 'parallel_mode': 'process', 'n_workers': cores_per_task}

        warm_start_potential = combo.get('warm_start_potential')
        if warm_start_potential:
            potential_config = {
                'filename': str(Path(warm_start_potential).resolve()),
                'reset': combo.get('warm_start_reset', False),
            }
            print(f"{stage_key}: combo {label!r} warm-starting from {potential_config['filename']} "
                  f"(reset={potential_config['reset']})")
        else:
            potential_config = combo.get('potential_config', DEFAULT_COLD_START_POTENTIAL_CONFIG)
            print(f"{stage_key}: combo {label!r} cold start (no warm_start_potential configured)")

        written = build_ace_ensemble_inputs(
            dataset_path=dataset_path,
            output_dir=combo_dir,
            elements=elements,
            seeds=tuple(combo.get('seeds', (0, 1, 2))),
            kappa_grid_kwargs=combo.get('kappa_grid_kwargs'),
            total_cap=combo.get('total_cap', 10),
            sample_seed=combo.get('seed', 0),
            potential_config=potential_config,
            fit_config=combo.get('fit_config'),
            backend_config=backend_config,
            test_filename=resolved_validation_dataset_path,
        )
        print(f"{stage_key}: combo {label!r} -- wrote {len(written)} ensemble member folder(s) under {combo_dir}")


def fit_and_validate(config, stage_key='fit_and_validate'):
    """
    Submits one ACE-fitting chore per (combo_label, member) folder under
    ace_inputs_dir/<combo_label>/<member>/ -- generalized from a
    single-combo fit's own flat ace_inputs_dir/<member>/ listing to walk
    one level deeper, since build_ff_inputs (above) now nests members
    under a combo label. FF/ace_fit.py itself needs NO changes -- it only
    ever sees one work_dir/results_dir per chore, nesting depth above that
    is invisible to it.
    """
    from matensemble.pipeline import Pipeline
    from matensemble.model import Resources
    from EnsembleFFFit.base import FFMatEnsemble

    cfg = config[stage_key]
    _require(cfg, stage_key, ['ace_inputs_dir', 'ff_task'])

    ace_inputs_dir = Path(cfg['ace_inputs_dir']).resolve()
    config_filename = cfg.get('config', 'input.yaml')
    ff_task = str(Path(cfg['ff_task']).resolve())
    entry_point = cfg.get('entry_point', 'run_ace_fit')
    finished_file = cfg.get('finished_file')

    combo_dirs = sorted(d for d in ace_inputs_dir.iterdir() if d.is_dir())
    if not combo_dirs:
        raise ValueError(f"No combo subdirectory found under {ace_inputs_dir} -- has build_ff_inputs been run yet?")

    chore_specs = []
    for combo_dir in combo_dirs:
        member_dirs = sorted(
            (d for d in combo_dir.iterdir() if d.is_dir() and (d / config_filename).is_file()),
            key=lambda d: d.name,
        )
        for member_dir in member_dirs:
            chore_specs.append((combo_dir.name, {
                'config': str((member_dir / config_filename).resolve()),
                'results_dir': str(member_dir),
                'work_dir': str(member_dir),
                'name': f'FF_{combo_dir.name}_{member_dir.name}',
                'ff_task': ff_task,
                'entry_point': entry_point,
                'finished_file': finished_file,
            }))

    if not chore_specs:
        raise ValueError(
            f"No ensemble member folder (with a {config_filename}) found under any combo subdirectory "
            f"of {ace_inputs_dir} -- has build_ff_inputs been run yet?")

    resources_kwargs = dict(num_tasks=cfg.get('num_tasks', 1),
                             cores_per_task=cfg.get('cores_per_task', 1),
                             gpus_per_task=cfg.get('gpus_per_task', 0))
    env = ace_fit_env()

    pipe = Pipeline(reserve_broker_node=False)

    @pipe.chore(name="fit_ace", **resources_kwargs, env=env, inherit_env=True)
    def fit_ace_chore(task_dict):
        """Thin chore wrapper -- the real fitting logic lives in FF/ace_fit.py."""
        return FFMatEnsemble.run_individual(task_dict)

    resources = Resources(**resources_kwargs, env=env, inherit_env=True)
    for combo_label, ff_dct in chore_specs:
        print(f"{stage_key}: submitting {combo_label}/{ff_dct['name']}")
        pipe.call("fit_ace", ff_dct, resources=resources)

    future = pipe.submit(log_delay=25)
    future.result()


# ---------------------------------------------------------------------------
# Stage: single_points_validation -- labels the raw validation POSCARs
# ---------------------------------------------------------------------------

def single_points_validation(config, stage_key='single_points_validation'):
    """
    Single-points every raw validation POSCAR (generate_validation_structures'
    own output -- real Materials Project structures, energies/forces not
    yet known) with the teacher MACE model, writing properties.json
    alongside each one. GAP FOUND AND FIXED THIS PASS: build_ff_inputs/
    prepare_ace_validation_structures (below) both already reference
    MACE/single_points/structures/validation as if it already existed with
    real energy_corrected/forces, but no prior pass ever produced it --
    generate_validation_structures only writes raw, unlabeled POSCARs to
    MACE/structures/validation. Every training tier gets this same
    single-point-labeling treatment (run_single_points above); validation
    needed it too and was missing it.

    Unlike run_single_points above, this does NOT unpack MD trajectory
    frames first -- validation structures are already final, static
    POSCARs (one structure per leaf directory), not MD trajectories -- so
    this dispatches _run_md_batch directly against the raw structures tree.
    """
    cfg = config[stage_key]
    _run_md_batch(cfg, stage_key, stage_key,
                  default_entry_point='run_ase_single_points', default_chunk_size=50,
                  require_in_file=False)


def single_points_isolated_elements(config, stage_key='single_points_isolated_elements'):
    """
    Single-points every generate_isolated_elements POSCAR with the teacher
    MACE model, writing properties.json alongside each one -- the same
    gap/fix single_points_validation (above) exists for, for the OTHER
    prerequisite build_ff_inputs/downselect_structures both already assume
    exists (MACE/single_points/structures/isolated_elements, read directly
    by EnsembleFFFit.potential.mace.build_ensemble_inputs.
    get_isolated_atom_e0s): before this stage existed, a user had to supply
    that tree by hand (there was no automated way to produce it at all,
    not even a wrong one -- unlike every other foundation_model-related gap
    this example has hit, see workflow_config.yaml's own
    &fine_tuned_mace_model comment, confirmed this session against a real
    interactive allocation).
    """
    cfg = config[stage_key]
    _run_md_batch(cfg, stage_key, stage_key,
                  default_entry_point='run_ase_single_points', default_chunk_size=50,
                  require_in_file=False)


# ---------------------------------------------------------------------------
# Stage: prepare_ace_validation_structures
# ---------------------------------------------------------------------------

def _relative_label(path_str, roots):
    """
    Relative path of `path_str` under whichever of `roots` (a single path,
    or a list) actually contains it, prefixed with that root's own
    basename so structures pooled from multiple roots can't collide in
    label space with each other.
    """
    path = Path(path_str).resolve()
    roots = [roots] if isinstance(roots, (str, Path)) else list(roots)
    for root in roots:
        root_resolved = Path(root).resolve()
        try:
            rel = path.relative_to(root_resolved)
        except ValueError:
            continue
        return str(Path(root_resolved.name) / rel)
    raise ValueError(f"{path} is not under any of {roots}")


def prepare_ace_validation_structures(config, stage_key='prepare_ace_validation_structures'):
    """
    One-time export of the real validation phase structures PLUS a random
    subsample of a training-set "overfitting sanity check" sample into a
    phase-labeled POSCAR + reference properties.json tree, then converts
    each POSCAR to LAMMPS data format (structure.lmp).

    Training-set sample built from the UNION of every combo's own resolved
    training_root (FF/subset_combination.resolve_subsets, applied once per
    combo in build_ff_inputs.training_subsets_by_combo then flattened/
    deduped) -- user's explicit choice (2026-11-13): different combos now
    have genuinely different training_root compositions (that's the whole
    point of the subset-combination experiment), so there's no single fixed
    "the" training tier to sample from the way the companion Pd-Se
    pipeline's own single-combo version did. This only affects the
    secondary training-sanity-check columns in the final ranking -- the
    real scored metric (the validation set) is sourced identically either
    way.
    """
    from ase.io import write as ase_write
    from EnsembleFFFit.molecular_dynamics.lammps.poscar_to_structure_lmp import poscar_to_structure_lmp
    from EnsembleFFFit.potential.ace.build_ace_dataset import build_ace_dataframe, compute_validation_floor
    from EnsembleFFFit.potential.mace.build_ensemble_inputs import get_isolated_atom_e0s
    from EnsembleFFFit.utilities.general import copy_and_transform_files, make_mirrored_rename_dest_path_fn
    from FF.subset_combination import resolve_subsets

    cfg = config[stage_key]
    build_cfg = config.get('build_ff_inputs', {})

    validation_root = cfg.get('validation_root') or build_cfg.get('validation_root')
    isolated_elements_root = cfg.get('isolated_elements_root') or build_cfg.get('isolated_elements_root')
    _require({'validation_root': validation_root, 'isolated_elements_root': isolated_elements_root},
              stage_key, ['validation_root', 'isolated_elements_root'])
    _require(cfg, stage_key, ['validation_dest_root', 'training_dest_root', 'lammps_recipe_source_dir'])

    combos = cfg.get('training_subsets_by_combo') or build_cfg.get('training_subsets_by_combo')
    if not combos:
        raise ValueError(f"{stage_key}: need training_subsets_by_combo, either set here or "
                          "inherited from build_ff_inputs.")
    training_root = []
    for combo in combos:
        for d in resolve_subsets(combo['subsets']):
            if d not in training_root:
                training_root.append(d)
    print(f"{stage_key}: training-set sanity-check sample built from the union of {len(combos)} "
          f"combo(s)' own training_root: {training_root}")

    max_abs_energy_per_atom = cfg.get('max_abs_energy_per_atom', build_cfg.get('max_abs_energy_per_atom', 50.0))
    max_abs_force_component = cfg.get('max_abs_force_component', build_cfg.get('max_abs_force_component', 50.0))
    stability_floor_margin = cfg.get('stability_floor_margin', build_cfg.get('stability_floor_margin', 0.5))
    training_sample_size = cfg.get('training_sample_size', 1000)
    seed = cfg.get('seed', 0)

    e0s = get_isolated_atom_e0s(isolated_elements_root)
    if not e0s:
        raise FileNotFoundError(f"No isolated-atom reference energies found under {isolated_elements_root}")

    validation_df = build_ace_dataframe(
        validation_root, e0s,
        max_abs_energy_per_atom=max_abs_energy_per_atom, max_abs_force_component=max_abs_force_component,
        include_path=True,
    )
    print(f"{stage_key}: {len(validation_df)} validation structure(s) under {validation_root}")

    stability_floor_per_atom = compute_validation_floor(
        validation_root, e0s,
        max_abs_energy_per_atom=max_abs_energy_per_atom, max_abs_force_component=max_abs_force_component,
    )
    training_df = build_ace_dataframe(
        training_root, e0s,
        max_abs_energy_per_atom=max_abs_energy_per_atom, max_abs_force_component=max_abs_force_component,
        stability_floor_per_atom=stability_floor_per_atom, stability_floor_margin=stability_floor_margin,
        include_path=True,
    )
    print(f"{stage_key}: {len(training_df)} training structure(s) under the union training_root "
          f"(after the same blow-up/stability filters build_ff_inputs applies)")

    if len(training_df) > training_sample_size:
        training_df = training_df.sample(n=training_sample_size, random_state=seed).reset_index(drop=True)
        print(f"{stage_key}: subsampled training to {training_sample_size} structure(s) (seed={seed})")

    for label, df, roots, dest_root in (
        ('validation', validation_df, validation_root, cfg['validation_dest_root']),
        ('training', training_df, training_root, cfg['training_dest_root']),
    ):
        reference_root = Path(dest_root).parent / 'reference'
        written = 0
        for _, row in df.iterrows():
            rel_label = _relative_label(row['path'], roots)
            out_dir = reference_root / rel_label
            out_dir.mkdir(parents=True, exist_ok=True)
            ase_write(str(out_dir / 'POSCAR'), row['ase_atoms'], format='vasp')

            # Unrotated, original frame -- same reasoning as the companion
            # pipeline's own version of this stage: this file stays
            # genuine, backend-agnostic ground truth; the LAMMPS-specific
            # frame rotation happens only at comparison time, in
            # rank_ace_validation below (rotate_reference_forces_to_lammps_frame).
            forces = row['forces']
            properties = {
                'energy': float(row['energy_corrected']),
                'fx': [float(v) for v in forces[:, 0]],
                'fy': [float(v) for v in forces[:, 1]],
                'fz': [float(v) for v in forces[:, 2]],
            }
            with open(out_dir / 'properties.json', 'w') as f:
                json.dump(properties, f, indent=4)
            written += 1
        print(f"{stage_key} ({label}): wrote {written} POSCAR+properties.json pair(s) under {reference_root}")

        dest_path_fn = make_mirrored_rename_dest_path_fn(str(reference_root), dest_root, 'structure.lmp')
        copy_and_transform_files(
            str(reference_root), str(dest_root), pattern='POSCAR',
            dest_path_fn=dest_path_fn,
            transform_fn=lambda src, dst: poscar_to_structure_lmp(str(src), str(dst), atom_style='atomic'),
        )

        recipe_dest_dir = Path(dest_root).parent
        recipe_dest_dir.mkdir(parents=True, exist_ok=True)
        recipe_source_dir = Path(cfg['lammps_recipe_source_dir'])
        recipe_files = cfg.get('lammps_recipe_files', ['in.single_point_ace', 'lammps_ace_single_point.py'])
        for name in recipe_files:
            src = recipe_source_dir / name
            if not src.is_file():
                raise FileNotFoundError(f"lammps_recipe_source_dir is missing {name}: {src}")
            shutil.copy2(src, recipe_dest_dir / name)
        print(f"{stage_key} ({label}): copied {len(recipe_files)} LAMMPS recipe file(s) "
              f"from {recipe_source_dir} into {recipe_dest_dir}")


# ---------------------------------------------------------------------------
# Stage: stage_ace_validation_force_fields
# ---------------------------------------------------------------------------

def _convert_yaml_to_yace(yaml_path, yace_path):
    """Converts a pyace B-basis potential (.yaml) to LAMMPS's own Ctilde-basis
    (.yace) format via pyace's own pace_yaml2yace CLI. Must run in-container."""
    yace_path = Path(yace_path)
    yace_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['pace_yaml2yace', str(yaml_path), '-o', str(yace_path)], check=True)


def _stage_ace_force_fields(sources, dest_roots, member_pattern='FF_*.yaml', target_name='ace.yace'):
    """
    Finds every fitted ACE potential under each of `sources` (a list of
    {label, source_root} dicts -- one entry per training-subset combo from
    build_ff_inputs' own nested ace_inputs_dir/<combo_label>/<member>/
    output), converts each to .yace, and copies the result into every
    listed dest_roots as <combo_label>_<member>/target_name.

    Takes the FULL combo list in one call (not one call per combo, the
    companion Pd-Se pipeline's own pattern of several near-duplicate
    stage-key config blocks) -- labeling each staged member
    combo_label_member avoids collisions, since every combo's own members
    are numbered independently starting from 0.
    """
    dest_roots = [Path(d) for d in dest_roots]
    staged_members = []
    skipped = []
    for source in sources:
        combo_label = source['label']
        source_root = Path(source['source_root'])
        if not source_root.is_dir():
            raise FileNotFoundError(
                f"_stage_ace_force_fields: source_root {source_root} (combo {combo_label!r}) does not "
                f"exist -- has build_ff_inputs been run yet?")
        for member_dir in sorted(p for p in source_root.iterdir() if p.is_dir()):
            matches = sorted(member_dir.glob(member_pattern))
            if len(matches) != 1:
                if matches:
                    skipped.append((member_dir, len(matches)))
                continue

            label = f"{combo_label}_{member_dir.name}"
            yace_cache = member_dir / target_name
            _convert_yaml_to_yace(matches[0], yace_cache)

            for dest_root in dest_roots:
                dest_path = dest_root / label / target_name
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(yace_cache, dest_path)
            staged_members.append(label)

    print(f"Converted+staged {len(staged_members)} fitted ACE potential(s) ({staged_members}) "
          f"from {[s['source_root'] for s in sources]} into {dest_roots}")
    if skipped:
        print(f"Skipped {len(skipped)} directory(ies) with != 1 '{member_pattern}' match:")
        for d, n in skipped:
            print(f"  {d}: {n} matches")


def stage_ace_validation_force_fields(config, stage_key='stage_ace_validation_force_fields'):
    """See _stage_ace_force_fields. Stages every training-subset combo's
    fitted potentials (converted to .yace) into ace_validation_single_points'
    own validation/training force_fields trees. Must run inside the
    container -- pace_yaml2yace needs to be on PATH."""
    cfg = config[stage_key]
    _require(cfg, stage_key, ['dest_roots'])

    build_cfg = config.get('build_ff_inputs', {})
    ace_inputs_dir = cfg.get('ace_inputs_dir') or build_cfg.get('ace_inputs_dir')
    combos = cfg.get('training_subsets_by_combo') or build_cfg.get('training_subsets_by_combo')
    if not ace_inputs_dir or not combos:
        raise ValueError(f"{stage_key}: need ace_inputs_dir and training_subsets_by_combo, either set "
                          "here or inherited from build_ff_inputs.")

    sources = [{'label': combo['label'], 'source_root': str(Path(ace_inputs_dir) / combo['label'])}
               for combo in combos]
    _stage_ace_force_fields(sources, cfg['dest_roots'],
                             cfg.get('member_pattern', 'FF_*.yaml'), cfg.get('target_name', 'ace.yace'))


# ---------------------------------------------------------------------------
# Stage: ace_validation_single_points
# ---------------------------------------------------------------------------

def _submit_ace_single_points_batch(sub_cfg, chore_name):
    """Shared submission logic for one (validation or training) ACE LAMMPS
    single-points batch. gpus_per_task defaults to 1 -- this chore's own
    driver activates Kokkos unconditionally, so it expects a real GPU."""
    from matensemble.pipeline import Pipeline
    from matensemble.model import Resources
    from EnsembleFFFit.base import MDMatEnsemble

    _require(sub_cfg, chore_name, ['run_directory', 'inputs_directory'])

    options = {'ffield': sub_cfg.get('ffield', 'ace.yace'),
               'in_file': sub_cfg.get('in_file'),
               'structure': sub_cfg.get('structure', 'structure.lmp'),
               'lammps_task': sub_cfg.get('lammps_task', 'lammps_ace_single_point.py')}
    options = {k: v for k, v in options.items() if v is not None}

    resources_kwargs = dict(num_tasks=sub_cfg.get('num_tasks', 1),
                             cores_per_task=sub_cfg.get('cores_per_task', 1),
                             gpus_per_task=sub_cfg.get('gpus_per_task', 1))

    md = MDMatEnsemble(sub_cfg['run_directory'], sub_cfg['inputs_directory'], **options)
    task_dicts = md.build_task_dicts(
        sub_cfg.get('lammps_task', 'lammps_ace_single_point.py'),
        sub_cfg.get('parent_levels', 3),
        sub_cfg.get('check_files', ['ffield']),
        sub_cfg.get('entry_point', 'run_lammps_single_points'),
        finished_file=sub_cfg.get('finished_file'),
    )
    if not task_dicts:
        raise ValueError(f"No (force field, structure) batches found for {chore_name} -- check "
                          f"run_directory/inputs_directory.")

    pipe = Pipeline(reserve_broker_node=False)

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


def ace_validation_single_points(config, stage_key='ace_validation_single_points'):
    """Runs LAMMPS/pair_style-pace single points for every staged ACE
    potential x every structure.lmp under both validation and training
    structure trees. Two separate, sequential Flux submissions within this
    one allocation -- a failure/empty-batch in one shouldn't block the
    other from at least being attempted."""
    cfg = config[stage_key]
    _require(cfg, stage_key, ['validation', 'training'])

    _submit_ace_single_points_batch(cfg['validation'], 'run_ace_val_lammps')
    _submit_ace_single_points_batch(cfg['training'], 'run_ace_train_lammps')


# ---------------------------------------------------------------------------
# Stage: rank_ace_validation
# ---------------------------------------------------------------------------

def _drop_first_run_segment(run_image_dct):
    """
    Strips the first '/'-separated segment off every "run" key of a
    {run: {image: props}} dict. CONFIRMED NECESSARY: MDMatEnsemble's own
    output structure mirrors inputs_directory's own structures/ subpath
    segment into every staged force field's own output tree, so
    parse_labeled_tree's own "run" key for the FF side comes out one
    segment longer than parse_reference_tree's "run" key for the reference
    side. Without stripping it here, assert_comparable_dicts would raise
    KeyError the first time it looked up an FF's own "run" key in the
    reference dict.
    """
    out = {}
    for run, images in run_image_dct.items():
        new_run = run.split('/', 1)[1] if '/' in run else ''
        out[new_run] = images
    return out


def rank_ace_validation(config, stage_key='rank_ace_validation'):
    """
    Parses the LAMMPS/pace single-point output from
    ace_validation_single_points plus the reference ground truth written by
    prepare_ace_validation_structures, and ranks every fitted ACE potential
    by a combined energy+force score -- calling
    EnsembleFFFit.analysis.best_force_field.rank_force_fields_combined
    DIRECTLY (validation_energy_weight/training_energy_weight both set,
    energy_per_atom=True, force_aggregation='atom_component_pooled'), not a
    hand-rolled reimplementation -- this is the actual payoff of this
    session's own EFF-core extensions to that function.

    RESOLVED NUMERICAL CAVEAT (was open in an earlier pass): the companion
    Pd-Se pipeline's own rank_ace_validation deliberately avoided
    rank_force_fields_combined's own DEFAULT force-pooling convention
    ("structure_mean" -- one RMSE per structure, averaged uniformly across
    structures), because it disagreed with pyace/tensorpotential's own
    fitting-time test_metrics.txt (which pools every force COMPONENT from
    every atom of every structure into one flat array before one RMSE --
    "atom_component_pooled", implicitly atom-count-weighted) by a real,
    confirmed ~25-35% margin. Rather than accept that mismatch or
    reimplement combination logic a third time, this session added
    force_aggregation as a real, verified-safe option on
    rank_force_fields_combined itself (see that function's and
    get_pooled_force_rmse's own docstrings) -- passing
    force_aggregation='atom_component_pooled' here makes this example's own
    val_force_dev/train_force_dev columns use the SAME pooling convention
    as fitting-time test_metrics.txt, directly comparable to it, with zero
    reimplementation.
    """
    from EnsembleFFFit.analysis.best_force_field import rank_force_fields_combined, rotate_reference_forces_to_lammps_frame
    from EnsembleFFFit.analysis.dict_parsers import parse_labeled_tree, parse_reference_tree, PropertiesOnlyParser

    cfg = config[stage_key]
    _require(cfg, stage_key,
              ['validation_ff_dir', 'validation_reference_root',
               'training_ff_dir', 'training_reference_root'])

    validation_ff_dct_raw = parse_labeled_tree(cfg['validation_ff_dir'], parser_cls=PropertiesOnlyParser)
    validation_reference_dct = {"DFT": parse_reference_tree(cfg['validation_reference_root'], parser_cls=PropertiesOnlyParser)}
    validation_ff_dct = {label: _drop_first_run_segment(run_dct) for label, run_dct in validation_ff_dct_raw.items()}

    training_ff_dct_raw = parse_labeled_tree(cfg['training_ff_dir'], parser_cls=PropertiesOnlyParser)
    training_reference_dct = {"DFT": parse_reference_tree(cfg['training_reference_root'], parser_cls=PropertiesOnlyParser)}
    training_ff_dct = {label: _drop_first_run_segment(run_dct) for label, run_dct in training_ff_dct_raw.items()}

    # Rotate the reference forces into LAMMPS's own canonical box frame at
    # comparison time -- see rotate_reference_forces_to_lammps_frame's own
    # docstring. Energy is untouched (rotation-invariant); only fx/fy/fz.
    validation_reference_dct["DFT"] = rotate_reference_forces_to_lammps_frame(
        validation_reference_dct["DFT"], cfg['validation_reference_root'])
    training_reference_dct["DFT"] = rotate_reference_forces_to_lammps_frame(
        training_reference_dct["DFT"], cfg['training_reference_root'])

    # No unit conversion -- ACE/pyace and LAMMPS's own `units metal` are
    # both eV/eV-per-Angstrom natively (confirmed by reading
    # in.single_point_ace: `units metal`, and build_ace_dataset.py's own
    # energy_corrected convention, also eV -- not assumed from an old
    # comment).
    table_lines, sorted_labels, combined_scores = rank_force_fields_combined(
        validation_ff_dct, validation_reference_dct,
        training_ff_dct, training_reference_dct,
        validation_force_weight=cfg.get('validation_force_weight', 1.0),
        training_force_weight=cfg.get('training_force_weight', 1.0),
        validation_energy_weight=cfg.get('validation_energy_weight', 1.0),
        training_energy_weight=cfg.get('training_energy_weight', 1.0),
        energy_per_atom=True,
        force_aggregation=cfg.get('force_aggregation', 'atom_component_pooled'),
    )
    print("\n".join(table_lines))

    dest_dir = cfg.get('dest_dir', os.path.dirname(cfg['validation_ff_dir']))
    os.makedirs(dest_dir, exist_ok=True)

    ranking_txt_path = cfg.get('ranking_txt_path', os.path.join(dest_dir, 'ranking.txt'))
    force_aggregation = cfg.get('force_aggregation', 'atom_component_pooled')
    header_line = (f"rank_force_fields_combined (EFF-core, energy_per_atom=True, "
                    f"force_aggregation={force_aggregation}) -- "
                    "units: eV / eV-per-Angstrom throughout, no conversion applied")
    print_and_write(table_lines, ranking_txt_path, header=header_line)

    # rank_force_fields_combined returns (lines, labels, scores) -- it does
    # NOT write a CSV itself; that's this stage's own job, same as the
    # companion pipeline's version.
    ranking_csv_path = cfg.get('ranking_csv_path', os.path.join(dest_dir, 'ranking.csv'))
    with open(ranking_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['rank', 'ff_label', 'combined_score'])
        for rank, (label, score) in enumerate(zip(sorted_labels, combined_scores), start=1):
            writer.writerow([rank, label, f"{score:.8f}"])
    print(f"Wrote ranking to {ranking_txt_path} and {ranking_csv_path}")


# ---------------------------------------------------------------------------
# Stage: select_ace_validation_force_field
# ---------------------------------------------------------------------------

def select_ace_validation_force_field(config, stage_key='select_ace_validation_force_field'):
    """
    Picks the top number_to_select ACE potential(s) from rank_ace_validation's
    own ranking.csv and copies each one's fitted FF_<member>.yaml forward to
    dest_root/<combo_label>/<member>/FF_<member>.yaml.

    Calls EnsembleFFFit.analysis.downselect_force_fields.select_and_copy
    DIRECTLY (NOT a hand-rolled reimplementation, unlike the companion
    Pd-Se pipeline's own version of this stage, which the Phase 1 audit
    incorrectly claimed already did this -- it didn't, confirmed by reading
    it) -- made possible by this session's own copy_files fix (glob instead
    of an exact-name match, so FF_<member>.yaml's per-member-varying name
    works). ranking.csv's own ff_label column uses the STAGED tree's label
    convention (combo_label_member, underscore-joined, to avoid collisions
    in stage_ace_validation_force_fields' own shared comparison tree) --
    select_and_copy instead needs to resolve against the ORIGINAL,
    un-staged fitted-potential tree (ace_inputs_dir/<combo_label>/<member>/,
    nested), so labels are reconstructed here as combo_label/member (a
    REAL path separator) by splitting each staged label back apart using
    the known combo labels from training_subsets_by_combo -- see
    _destage_label below.
    """
    from EnsembleFFFit.analysis.downselect_force_fields import select_and_copy

    cfg = config[stage_key]
    ranking_csv = cfg.get('ranking_csv') or os.path.join(
        config.get('rank_ace_validation', {}).get('dest_dir', '.'), 'ranking.csv')
    build_cfg = config.get('build_ff_inputs', {})
    fitted_potentials_root = cfg.get('fitted_potentials_root') or build_cfg.get('ace_inputs_dir')
    combos = cfg.get('training_subsets_by_combo') or build_cfg.get('training_subsets_by_combo')
    if not fitted_potentials_root or not combos:
        raise ValueError(f"{stage_key}: need fitted_potentials_root and training_subsets_by_combo, "
                          "either set here or inherited from build_ff_inputs.")
    dest_root = cfg.get('dest_root', 'MD/ace_validation/selected')
    number_to_select = cfg.get('number_to_select', 1)
    strategy = cfg.get('strategy', 'top')

    combo_labels = sorted((c['label'] for c in combos), key=len, reverse=True)

    def _destage_label(staged_label):
        """combo_label_member (staged-tree form) -> combo_label/member (nested-tree
        form) -- tries the longest combo label first so e.g. 'tier4_specific_1'
        doesn't wrongly match a shorter combo label that happens to be a prefix."""
        for combo_label in combo_labels:
            prefix = combo_label + '_'
            if staged_label.startswith(prefix):
                return f"{combo_label}/{staged_label[len(prefix):]}"
        raise ValueError(f"{stage_key}: staged label {staged_label!r} doesn't start with any known "
                          f"combo label {combo_labels} -- can't resolve it back to the original "
                          f"fitted-potential tree.")

    with open(ranking_csv) as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: float(r['combined_score']))
    sorted_labels = [_destage_label(r['ff_label']) for r in rows]

    selected = select_and_copy(
        validation_path=fitted_potentials_root, single_point_path=dest_root,
        validation_labels=sorted_labels, number_to_copy=number_to_select, strategy=strategy,
        patterns=(cfg.get('member_pattern', 'FF_*.yaml'),),
    )
    if not selected:
        raise RuntimeError(f"{stage_key}: nothing selected -- check {ranking_csv} and {fitted_potentials_root}.")

    report_lines = [f"Ranking source: {ranking_csv} ({len(rows)} candidate(s))",
                     f"Selected top {len(selected)} by combined_score via select_and_copy "
                     f"(strategy={strategy!r}):", ""]
    for label, old_path, new_path in selected:
        report_lines.append(f"  {label}: {old_path} -> {new_path}")
    selection_report_path = cfg.get('selection_report_path', os.path.join(dest_root, 'selection_report.txt'))
    print_and_write(report_lines, selection_report_path)
    print(f"Selected {len(selected)} ACE potential(s) -- see {selection_report_path}")


# ---------------------------------------------------------------------------
# Stage: MD Validation (structural stability) -- build_structural_stability_
# inputs, stage_structural_stability_force_fields, structural_stability_
# screening, check_structural_stability, stage_mace_reference_force_field,
# mace_structural_stability_screening.
#
# Ported from the companion Pd-Se pipeline's own analogous stages
# (run_pipeline/run_pipeline.py). Compares every staged ACE potential's
# short room-temperature NPT behavior against (a) each phase's own pristine
# reference structure and (b) the same protocol run with the ORIGINAL
# teacher MACE model (stage_mace_reference_force_field/
# mace_structural_stability_screening below) -- telling apart "this ACE fit
# doesn't preserve the motif" from "this NPT protocol is too aggressive for
# any potential at this scale/duration." See MD/structural_stability/
# structural_stability_checker.py's own module docstring for the actual
# RDF/z-density/MSD metrics and their relationship to the ReaxFF example's
# own, different coordination-number-based structural check.
# ---------------------------------------------------------------------------

_ACE_PAIR_STYLE_BLOCK = """pair_style pace product
pair_coeff * * ${ff_filename} ${elements}"""

_MACE_PAIR_STYLE_BLOCK = """pair_style mliap unified ${ff_filename} 0
pair_coeff * * ${elements}"""

_STRUCTURAL_STABILITY_IN_FILE_TEMPLATE = """# Generated by build_structural_stability_inputs (run_pipeline.py) --
# room-temperature ({room_temperature:.0f}K) NPT structural-stability check:
# does this potential preserve its known structural motifs under a short
# NPT run, or does it reorganize/degrade unphysically? Single-condition
# check, not a melting-point screen -- no temperature grid/ramp.
#
# dump d1/d2 are defined AFTER equilibration, immediately before the
# production run -- so md_run.traj/md_run_unwrapped.traj contain ONLY
# production-window frames, no minimize/equilibration-stage frames mixed
# in. This makes structural_stability_checker.py's own "first vs. last
# equilibrium_window_fraction of frames" baseline exact, not an
# approximation.

variable        seed                    equal {seed}

units metal
dimension 3
boundary p p p
atom_style atomic
box tilt large
processors * * *
newton on
neighbor 2.0 bin
neigh_modify every 1 delay 0 check yes

read_data ${{structure}}

{pair_style_block}

timestep        {timestep_ps}
thermo          {thermo_freq}
thermo_style    custom step temp pe etotal press lx ly lz vol density

#------------------- ENERGY MINIMIZATION (this force field's own equilibrium structure) ---------------------
# Each (ffield, structure, replicate) chore starts fresh from the one
# shared pristine structure.lmp and relaxes it under its OWN potential
# here, rather than relying on a separately pre-relaxed-per-potential
# structure upstream.
min_style cg
minimize 1.0e-2 1.0e-2 500 5000
min_style cg
minimize 1.0e-6 1.0e-4 2000 10000
write_data      ${{output_dir}}/Min.data

#------------------- NPT EQUILIBRATION AT {room_temperature:.0f}K (aniso) ------------------
velocity        all create {room_temperature:.0f} ${{seed}} dist gaussian
fix             equil all npt temp {room_temperature:.0f} {room_temperature:.0f} {temp_damp_ps} aniso 0.0 0.0 {press_damp_ps}
run             {equilibration_steps}
unfix           equil
write_data      ${{output_dir}}/data.equilibration

#------------------- PRODUCTION AT {room_temperature:.0f}K (aniso) -----------------------
# structural_stability_checker.py splits log.lammps on this exact marker
# line and only parses thermo rows after it for the production-window
# time-average.
print           "===PRODUCTION_START==="

dump            d1 all custom {dump_freq} ${{dump_file}} id type x y z fx fy fz
dump_modify     d1 element ${{elements}}
dump_modify     d1 sort id

dump            d2 all custom {dump_freq} ${{dump_unwrapped_file}} id type xu yu zu
dump_modify     d2 element ${{elements}}
dump_modify     d2 sort id

fix             production all npt temp {room_temperature:.0f} {room_temperature:.0f} {temp_damp_ps} aniso 0.0 0.0 {press_damp_ps}
run             {production_steps}
unfix           production
write_restart   ${{output_dir}}/production.res
write_data      ${{output_dir}}/data.production
"""


def _render_structural_stability_input(seed, cfg, pair_style_block=_ACE_PAIR_STYLE_BLOCK):
    timestep_fs = cfg.get('timestep_fs', 1.0)
    temp_damp_steps = cfg.get('temp_damp_steps', 100)
    press_damp_steps = cfg.get('press_damp_steps', 1000)
    timestep_ps = timestep_fs / 1000.0
    return _STRUCTURAL_STABILITY_IN_FILE_TEMPLATE.format(
        seed=seed,
        room_temperature=cfg.get('room_temperature_K', 300),
        pair_style_block=pair_style_block,
        timestep_ps=timestep_ps,
        temp_damp_ps=temp_damp_steps * timestep_ps,
        press_damp_ps=press_damp_steps * timestep_ps,
        thermo_freq=cfg.get('thermo_freq', 100),
        dump_freq=cfg.get('dump_freq', 100),
        equilibration_steps=cfg.get('equilibration_steps', 5000),
        production_steps=cfg.get('production_steps', 20000),
    )


def build_structural_stability_inputs(config):
    """
    One-time structure + recipe generation for the room-temperature
    structural-stability check. Writes one structure.lmp per phase in
    cfg['phases'] (a CubicSupercellTransformation supercell of that phase's
    own structure, ~supercell_kwargs['min_atoms']-sized, shared across
    every potential/replicate for that phase) under
    dest_root/structures/<phase_label>/, and one in.structural_stability
    (+ in.structural_stability_mace, same seed/protocol, different
    pair_style block -- what makes the ACE-vs-MACE comparison apples-to-
    apples) per replicate under dest_root/recipes/replicate_<k>/.

    cfg['phases'] values should point into this example's own
    generate_validation_structures output (MACE/structures/validation/...,
    not the companion pipeline's own finite_temperature/structures/
    validation/... path -- see this example's workflow_config.yaml).

    Replicate seeds are derived from cfg['seed'] (default 0) via
    random.Random, not independently random -- reproducible across
    re-generations of this same config.
    """
    from pymatgen.core.structure import Structure
    from pymatgen.transformations.advanced_transformations import CubicSupercellTransformation
    from EnsembleFFFit.molecular_dynamics.lammps.poscar_to_structure_lmp import structure_to_lmp

    cfg = config['build_structural_stability_inputs']
    _require(cfg, 'build_structural_stability_inputs', ['phases'])

    dest_root = cfg.get('dest_root', 'MD/structural_stability/lammps_inputs')
    supercell_kwargs = cfg.get('supercell_kwargs') or {'min_atoms': 3000, 'min_length': 20.0}

    written_structures = []
    for phase_label, source_structure in cfg['phases'].items():
        structure = Structure.from_file(source_structure)
        supercell = CubicSupercellTransformation(**supercell_kwargs).apply_transformation(structure)
        structure_dir = os.path.join(dest_root, 'structures', phase_label)
        os.makedirs(structure_dir, exist_ok=True)
        structure_lmp_path = os.path.join(structure_dir, 'structure.lmp')
        structure_to_lmp(supercell, structure_lmp_path, atom_style='atomic')
        written_structures.append(structure_lmp_path)
        print(f"{phase_label} ({source_structure}): {len(structure)} atoms -> supercell "
              f"{len(supercell)} atoms -> {structure_lmp_path}")

    replicates = cfg.get('replicates', 3)
    rng = random.Random(cfg.get('seed', 0))
    seeds = [rng.randint(1, 900_000_000) for _ in range(replicates)]

    written_recipes = []
    for k, seed in enumerate(seeds):
        recipe_dir = os.path.join(dest_root, 'recipes', f'replicate_{k}')
        os.makedirs(recipe_dir, exist_ok=True)

        in_file_path = os.path.join(recipe_dir, 'in.structural_stability')
        with open(in_file_path, 'w') as f:
            f.write(_render_structural_stability_input(seed, cfg, pair_style_block=_ACE_PAIR_STYLE_BLOCK))
        written_recipes.append(in_file_path)

        mace_in_file_path = os.path.join(recipe_dir, 'in.structural_stability_mace')
        with open(mace_in_file_path, 'w') as f:
            f.write(_render_structural_stability_input(seed, cfg, pair_style_block=_MACE_PAIR_STYLE_BLOCK))
        written_recipes.append(mace_in_file_path)

    print(f"Wrote {len(written_structures)} phase structure(s) and {len(written_recipes)} "
          f"replicate recipe(s) (ACE + MACE, {replicates} replicate(s) each) under {dest_root}")


def stage_structural_stability_force_fields(config):
    """
    Stages a small, hand-picked set of potentials (cfg['members'], an
    explicit [{source, label}, ...] list -- NOT a glob over one shared
    source_root, since these potentials live scattered across different
    FF/generation_1/<combo_label>/<member>/ directories, not one shared
    parent) into structural_stability_screening's own run_directory,
    converting each .yaml to .yace via the same _convert_yaml_to_yace
    helper stage_ace_validation_force_fields already uses.

    Already generic in the companion pipeline (confirmed by reading it
    directly) -- ported as-is, no changes needed.
    """
    cfg = config['stage_structural_stability_force_fields']
    _require(cfg, 'stage_structural_stability_force_fields', ['members', 'dest_roots'])

    target_name = cfg.get('target_name', 'ace.yace')
    dest_roots = [Path(d) for d in cfg['dest_roots']]

    staged = []
    for member in cfg['members']:
        label = member['label']
        source = Path(member['source'])
        yace_cache = source.parent / target_name
        _convert_yaml_to_yace(source, yace_cache)
        for dest_root in dest_roots:
            dest_path = dest_root / label / target_name
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(yace_cache, dest_path)
        staged.append(label)

    print(f"Converted+staged {len(staged)} hand-picked ACE potential(s) ({staged}) into {dest_roots}")


def structural_stability_screening(config):
    """
    Runs the room-temperature structural-stability check: every staged
    potential x every phase's structure.lmp x every replicate's
    in.structural_stability -- Kokkos-GPU LAMMPS/pair_style pace via
    MD/structural_stability/lammps_ace_structural_stability.py. Reuses
    _submit_ace_single_points_batch as-is (already generic over
    lammps_task/entry_point/ffield/structure/in_file/parent_levels).
    """
    cfg = config['structural_stability_screening']
    _require(cfg, 'structural_stability_screening', ['run_directory', 'inputs_directory'])
    _submit_ace_single_points_batch(cfg, 'run_structural_stability')


def check_structural_stability(config):
    """
    Walks structural_stability_screening's own run_directory for every
    completed (ff_label, phase, replicate) run and scores whether it
    preserved its known structural motifs, via
    MD/structural_stability/structural_stability_checker.py's own
    check_structural_stability entry point. stability_task/entry_point are
    dynamically imported (config-swappable, same convention used
    throughout this example for generation_task/entry_point) -- a
    different/extended metric, or a wholly different structural-stability
    concept (see that module's own note on the ReaxFF coordination-number
    precedent), is a config change, not a run_pipeline.py change.
    """
    from EnsembleFFFit.utilities.general import import_module_from_path

    cfg = config['check_structural_stability']
    _require(cfg, 'check_structural_stability', ['stability_task', 'inputs_directory'])
    run_directory = cfg.get('run_directory', 'MD/structural_stability/force_fields')
    inputs_directory = cfg['inputs_directory']

    module = import_module_from_path('structural_stability_task_module', cfg['stability_task'])
    entry_point = getattr(module, cfg.get('entry_point', 'check_structural_stability'))

    csv_path = cfg.get('csv_path', os.path.join(run_directory, 'structural_stability_summary.csv'))
    summary_rows, failures = entry_point(
        run_directory=run_directory,
        inputs_directory=inputs_directory,
        structure_filename=cfg.get('structure_filename', 'structure.lmp'),
        equilibrium_window_fraction=cfg.get('equilibrium_window_fraction', 0.2),
        rdf_cutoff=cfg.get('rdf_cutoff', 10.0),
        rdf_bins=cfg.get('rdf_bins', 200),
        rdf_far_fraction=cfg.get('rdf_far_fraction', 0.3),
        msd_fit_fraction=cfg.get('msd_fit_fraction', 0.5),
        dump_interval_ps=cfg.get('dump_interval_ps', 0.1),
        diffusive_slope_threshold=cfg.get('diffusive_slope_threshold', 0.2),
        diffusive_r_squared_threshold=cfg.get('diffusive_r_squared_threshold', 0.95),
        csv_path=csv_path,
    )
    n_failed = sum(len(d) for d in failures.values())
    print(f"Wrote structural stability summary ({len(summary_rows)} (ff_label, phase) row(s)) to {csv_path} "
          f"({n_failed} failed run(s) across {len(failures)} force field(s))")


def stage_mace_reference_force_field(config):
    """
    Converts the ORIGINAL teacher MACE model to LAMMPS's MLIAP format via
    mace_create_lammps_model's own CLI, then stages the result into
    structural_stability_screening's own run_directory as <label>/
    <target_name> -- a reference point to tell apart "this ACE fit doesn't
    preserve the motif" from "this NPT protocol is too aggressive for any
    potential at this scale/duration."

    Reuses teacher_preparation's own conversion: point this stage's
    source_model at the SAME path teacher_preparation.source_model already
    used (see workflow_config.yaml) -- the idempotency check below
    (converted_path already existing, written by teacher_preparation's own
    earlier run) means mace_create_lammps_model is NOT re-run here, this
    just stages the already-converted file. Same underlying tool, no
    duplicated conversion work.
    """
    cfg = config['stage_mace_reference_force_field']
    _require(cfg, 'stage_mace_reference_force_field', ['source_model', 'label', 'dest_roots'])

    source_model = Path(cfg['source_model'])
    label = cfg['label']
    target_name = cfg.get('target_name', 'model-mliap_lammps.pt')
    dest_roots = [Path(d) for d in cfg['dest_roots']]

    converted_path = source_model.parent / (source_model.name + '-mliap_lammps.pt')
    if not converted_path.is_file():
        subprocess.run(['mace_create_lammps_model', str(source_model), '--format=mliap'], check=True)
    if not converted_path.is_file():
        raise RuntimeError(f"mace_create_lammps_model did not produce the expected {converted_path}")

    for dest_root in dest_roots:
        dest_path = dest_root / label / target_name
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(converted_path, dest_path)

    print(f"Converted {source_model} -> {converted_path}, staged as {label}/{target_name} into {dest_roots}")


def mace_structural_stability_screening(config):
    """
    Runs the SAME room-temperature structural-stability check as
    structural_stability_screening, but for the single staged MACE
    reference potential (stage_mace_reference_force_field) against
    in.structural_stability_mace instead of in.structural_stability -- a
    separate submission (different driver/entry_point/ffield pattern)
    sharing the same run_directory/inputs_directory tree, so both land in
    one combined structural_stability_summary.csv with no analysis-script
    changes needed.
    """
    cfg = config['mace_structural_stability_screening']
    _require(cfg, 'mace_structural_stability_screening', ['run_directory', 'inputs_directory'])
    _submit_ace_single_points_batch(cfg, 'run_mace_structural_stability')


# Order matters for --stage all -- this is the actual pipeline sequence,
# each stage depending on the previous one's output. MD validation
# (structural stability) consumes select_ace_validation_force_field's own
# output but, matching the companion pipeline's own convention, stages ALL
# hand-picked best-by-validation members for comparison, not just the
# single final selection -- see stage_structural_stability_force_fields'
# own docstring.
STAGES = {
    'teacher_preparation': run_teacher_preparation,
    'generate_structures_tier1': generate_structures_tier1,
    'generate_validation_structures': generate_validation_structures,
    # single_points_validation: labels generate_validation_structures' own
    # raw POSCARs -- a gap found this pass, not present before (build_ff_
    # inputs/prepare_ace_validation_structures both already assumed this
    # existed).
    'single_points_validation': single_points_validation,
    # generate_isolated_elements/single_points_isolated_elements: the OTHER
    # gap found this pass -- build_ff_inputs/downselect_structures_tier2
    # both need MACE/single_points/structures/isolated_elements (isolated-
    # atom E0 reference energies) to already exist, and nothing produced it
    # before this. No tier number: unlike tier1-4, this isn't part of the
    # phase-specific/broad training-data story, it's a one-time prerequisite
    # these two stages exist to satisfy.
    'generate_isolated_elements': generate_isolated_elements,
    'single_points_isolated_elements': single_points_isolated_elements,
    'relax_structures': run_relaxation,
    'run_md_tier1': run_md_tier1,
    'run_single_points_tier1': run_single_points_tier1,
    'downselect_structures_tier2': downselect_structures_tier2,
    'run_md_tier2': run_md_tier2,
    'run_single_points_tier2': run_single_points_tier2,
    'generate_structures_tier3_defect': generate_structures_tier3_defect,
    'run_md_tier3': run_md_tier3,
    'run_single_points_tier3': run_single_points_tier3,
    'generate_structures_tier4_defect': generate_structures_tier4_defect,
    'generate_structures_tier4_motif': generate_structures_tier4_motif,
    'run_md_tier4_motif_300k': run_md_tier4_motif_300k,
    'run_md_tier4_motif_500k': run_md_tier4_motif_500k,
    'run_md_tier4_motif_700k': run_md_tier4_motif_700k,
    'run_md_tier4_motif_900k': run_md_tier4_motif_900k,
    'run_md_tier4_defect': run_md_tier4_defect,
    'run_single_points_tier4': run_single_points_tier4,
    # ACE ensemble fitting -- must come after every training-data-generation
    # stage above (it consumes their labeled output), not right after
    # teacher_preparation (an earlier version of this dict had that wrong).
    'build_ff_inputs': build_ff_inputs,
    'fit_and_validate': fit_and_validate,
    # Static validation + selection -- consumes fit_and_validate's own
    # fitted potentials.
    'prepare_ace_validation_structures': prepare_ace_validation_structures,
    'stage_ace_validation_force_fields': stage_ace_validation_force_fields,
    'ace_validation_single_points': ace_validation_single_points,
    'rank_ace_validation': rank_ace_validation,
    'select_ace_validation_force_field': select_ace_validation_force_field,
    # MD validation (structural stability) -- consumes select_ace_validation_
    # force_field's own output but, matching the companion pipeline's own
    # convention, stages ALL hand-picked best-by-validation members for
    # comparison, not just the single final selection.
    'build_structural_stability_inputs': build_structural_stability_inputs,
    'stage_structural_stability_force_fields': stage_structural_stability_force_fields,
    'structural_stability_screening': structural_stability_screening,
    'check_structural_stability': check_structural_stability,
    'stage_mace_reference_force_field': stage_mace_reference_force_field,
    'mace_structural_stability_screening': mace_structural_stability_screening,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", "-c", required=True, help="Path to the YAML workflow config file")
    parser.add_argument("--stage", "-s", required=True, choices=list(STAGES.keys()) + ['all'])
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.stage == 'all':
        for name, stage_fn in STAGES.items():
            print(f"\n{'=' * 20} Stage: {name} {'=' * 20}")
            stage_fn(config)
    else:
        STAGES[args.stage](config)


if __name__ == "__main__":
    main()
