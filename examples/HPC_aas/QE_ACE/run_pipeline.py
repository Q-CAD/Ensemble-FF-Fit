"""
Single entry point for the Pathfinder QE + pyACE + TorchSim pipeline. Only
the DFT stage (converge_dft_data) exists so far -- pyACE fitting and TorchSim
MD stages are added in later rounds of this build-out, following the same
Pipeline/@pipe.chore pattern as examples/Frontier/RMG_MACE_ASE/run_pipeline.py
and examples/Perlmutter/VASP_ReaxFF_LAMMPs/run_pipeline.py.

    python run_pipeline.py --config workflow_config.yaml --stage converge_dft_data

matensemble is only imported inside run_converge_dft_data, not at module load
time, so this file can still be inspected/linted without the Flux container.

converge_dft_data's Quantum-Espresso-specific environment/node-sizing logic
(below) is deliberately kept inline here rather than moved into
EnsembleFFFit -- it's tied to this one DFT backend/container, same reasoning
as why the RMG/VASP env-building helpers stay inline in their own
run_pipeline.py copies rather than living in the shared package.

This pipeline deliberately runs pw.x as a single MPI rank per chore for now
(num_tasks fixed at 1, see run_converge_dft_data) rather than sizing
multi-rank MPI jobs the way RMG's converge_dft_data does -- Pathfinder's
OpenMPI 5.0.5 bootstraps via PMIx (SLURM_MPI_TYPE=pmix), unlike Frontier's
Cray MPICH/PMI2 or Perlmutter's stack, and neither prior example's "bare
invocation, Flux owns launch semantics" convention has been tested against a
PMIx-based MPI stack yet. Revisit once the basic input/execute/parse
plumbing is confirmed working end-to-end against a real allocation.
"""
import argparse
import sys
from pathlib import Path

import yaml

# Pathfinder-specific runtime library path for the pw.x subprocess
# specifically -- NOT exported in the surrounding shell, only ever passed via
# a chore's/Resources' own `env`, matching RMG_LD_LIBRARY_PATH/
# VASP_LD_LIBRARY_PATH's own convention in the Frontier/Perlmutter examples.
# qe_dft.py reads this back out as QE_LD_LIBRARY_PATH and applies it as
# LD_LIBRARY_PATH ONLY to pw.x's own subprocess env, never to this chore
# process's own environment (see qe_dft.py's module docstring for why).
# Confirmed via `ldd $(which pw.x)` after `module load gcc/12.4.0
# openmpi/5.0.5 quantum-espresso/7.4-mpi-omp openblas/0.3.28` on Pathfinder
# -- re-verify if the module versions change.
# /host_lib64 (the real Rocky Linux /lib64, bound at container-launch time
# per matensemble_submission.sh) is DELIBERATELY LAST -- Spack's own
# self-consistent library tree should win for anything it provides; /host_lib64
# only fills genuine gaps. CONFIRMED (2026-09-16) as a real, necessary entry,
# not precautionary: pw.x failed with "error while loading shared libraries:
# libflexiblas.so.3: cannot open shared object file" until this was added --
# `ldd $(which pw.x)` on Pathfinder shows libflexiblas.so.3, libc.so.6,
# libm.so.6, libmvec.so.1, and the UCX libraries (libucp/libucs/libucm/libuct)
# all resolve from the host's plain /lib64, not anywhere under /software.
QE_LD_LIBRARY_PATH = (
    "/software/baseline/nsp/spack-envs/base-25.05/opt/gcc-12.4.0/openblas-0.3.28-j6vin5uxzhdkcfmzllwqfjfrrniv2vhk/lib:"
    "/software/baseline/nsp/spack-envs/base-25.05/opt/gcc-12.4.0/quantum-espresso-7.4-wwlqcxiopwbadoarm4v7obftaphxegaq/lib64:"
    "/software/baseline/nsp/spack-envs/base-25.05/opt/gcc-12.4.0/openmpi-5.0.5-ajpfcyc4knmqijf7z6bdiihugbld46zz/lib:"
    "/software/baseline/nsp/gcc/12.4.0/lib64:"
    "/host_lib64"
)


def qe_container_env(cores_per_task):
    """Env for the pw.x chore/task specifically -- see QE_LD_LIBRARY_PATH docstring.

    OMP_NUM_THREADS is derived from cores_per_task (not a second, separately
    hardcoded number) so the two can't quietly drift out of agreement, same
    convention as rmg_container_env in the Frontier example.

    OMPI_MCA_pml/OMPI_MCA_btl force OpenMPI's simplest, dependency-free
    single-process transport (ob1 point-to-point + self/sm byte-transport)
    instead of its default UCX preference. CONFIRMED (2026-09-16) necessary:
    without this, MPI_Init itself fails ("No components were able to be
    opened in the pml framework") -- OpenMPI's own UCX pml component needs
    its own separate transport-plugin directory (distinct from libucp.so.0
    etc. themselves, which QE_LD_LIBRARY_PATH's /host_lib64 entry already
    covers) that was never identified/bound. Since this pipeline's first
    pass only ever runs pw.x as a single MPI rank per chore (see module
    docstring), there's no real inter-process communication happening
    anyway -- forcing ob1/self,sm sidesteps needing to find that UCX plugin
    directory at all rather than chasing it down. Unlike QE_LD_LIBRARY_PATH,
    these are safe to set directly on the whole chore env (not just pw.x's
    own subprocess): nothing else in the chore process tree (Python, bash,
    matensemble) initializes MPI, so they're inert everywhere except pw.x
    itself.
    """
    from EnsembleFFFit.utilities.general import ensemble_fffit_pythonpath

    return {
        'PYTHONPATH': ensemble_fffit_pythonpath(),
        'OMP_NUM_THREADS': str(cores_per_task),
        'QE_LD_LIBRARY_PATH': QE_LD_LIBRARY_PATH,
        'OMPI_MCA_pml': 'ob1',
        'OMPI_MCA_btl': 'self,sm',
    }


def qe_resource_estimator(structure, input_args, options):
    """
    Node-count estimate for a QE chore: ceil(num_atoms / atoms_per_node),
    minimum 1. Deliberately simple compared to RMG's own processor-grid
    search (pyRMG.processor_grid) -- QE's own node/rank sizing here isn't
    governed by grid-divisibility constraints the way RMG's is. Passed as
    build_dft_dcts' resource_estimator (see base.py's DFTMatEnsemble
    docstring for why every non-RMG DFT backend should supply its own rather
    than this class growing backend-specific branches).
    """
    # atoms_per_node lives in the yaml directive (input_args, e.g.
    # single_point.yml's own 'atoms_per_node' key) -- NOT in options
    # (DFTMatEnsemble's own construction-time self.options, which never has
    # this key). Confirmed as a real bug during Stage 2 testing: reading it
    # from `options` silently fell back to the default every time.
    atoms_per_node = input_args.get('atoms_per_node', 100)
    return max(1, -(-len(structure) // atoms_per_node))  # ceil division


def _require(cfg, stage_name, keys):
    missing = [k for k in keys if cfg.get(k) is None]
    if missing:
        raise ValueError(f"Missing required key(s) {missing} under '{stage_name}' in the config file.")


# ---------------------------------------------------------------------------
# Stage: converge_dft_data
# ---------------------------------------------------------------------------

def run_converge_dft_data(config):
    """
    Submit one pw.x execution chore per (structure, single_point.yml) pair
    found under directory, each run as a single MPI rank (see module
    docstring for why) with OMP_NUM_THREADS=cores_per_task. Mirrors
    examples/Frontier/RMG_MACE_ASE/run_pipeline.py's run_converge_dft_data
    structure -- see its own docstring for why `directory` serves as both
    run_directory and inputs_directory, and why check_files should stay
    'qe_yaml' rather than 'structure_filename' for a multi-structure tree.
    """
    from matensemble.pipeline import Pipeline
    from matensemble.model import Resources
    from EnsembleFFFit.base import DFTMatEnsemble

    dft_cfg = config['converge_dft_data']
    _require(dft_cfg, 'converge_dft_data', ['directory', 'dft_task'])

    directory = dft_cfg['directory']
    dft_task = str(Path(dft_cfg['dft_task']).resolve())
    structure_filename = dft_cfg.get('structure_filename', 'POSCAR')
    qe_yaml_name = dft_cfg.get('qe_yaml_name', 'single_point.yml')
    entry_point = dft_cfg.get('entry_point', 'run_qe_calculation')
    check_files = dft_cfg.get('check_files', ['qe_yaml'])
    finished_file = dft_cfg.get('finished_file')
    cores_per_task = dft_cfg.get('cores_per_task', 1)
    max_tasks_per_job = dft_cfg.get('max_tasks_per_job', 1)
    # pw.x itself is CPU-only on Pathfinder (no GPU-enabled QE build yet), so
    # this defaults to 0 -- exposed as a config key rather than hardcoded so
    # switching to a GPU-enabled QE build later is a one-line config change,
    # not a code change.
    gpus_per_task = dft_cfg.get('gpus_per_task', 0)

    options = {
        'structure_filename': structure_filename,
        # DFTMatEnsemble.build_dft_dcts hardcodes recipe_keys=['dft_recipe']
        # internally (a required key -- it raises ValueError if missing, see
        # base.py) -- 'qe_yaml' is a SEPARATE key (same value) used only as
        # check_files' own proximity-matching anchor, mirroring rmg_dft.py's
        # 'rmg_yaml' key. NOTE: examples/Frontier/RMG_MACE_ASE/run_pipeline.py
        # only sets 'rmg_yaml', never 'dft_recipe' -- that example currently
        # doesn't satisfy this same requirement (likely a stale-relative-to-
        # base.py gap from whenever 'dft_recipe' was generalized in
        # anticipation of non-RMG backends; see pipeline/FRICTION_LOG.md).
        # Not fixed here since the existing examples aren't to be modified --
        # just making sure this new pipeline doesn't repeat the same gap.
        'dft_recipe': qe_yaml_name,
        'qe_yaml': qe_yaml_name,
    }

    dft = DFTMatEnsemble(directory, directory, **options)
    dft_dct_list = dft.build_dft_dcts(dft_task, check_files, entry_point,
                                       finished_file=finished_file,
                                       resource_estimator=qe_resource_estimator)

    if not dft_dct_list:
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
                f"Found {len(raw_task_dirs)} (structure, {qe_yaml_name}) pair(s) under {directory}, but "
                f"every one of them already has a '{finished_file}' match in its working directory -- "
                f"there's nothing new to run. Sample working director{'y' if len(sample) == 1 else 'ies'}: "
                f"{sample}{', ...' if len(raw_task_dirs) > 3 else ''}. Set finished_file to null (or a "
                f"pattern that doesn't already exist everywhere) to rerun them anyway."
            )
        raise ValueError(f"No (structure, {qe_yaml_name}) pairs found under {directory}")

    # reserve_broker_node=False: don't let Flux/MatEnsemble's own controller
    # monopolize an entire rank -- for the single-rank Flux instance this
    # pipeline launches (Pathfinder only allows single-node GPU jobs, so
    # there's no separate "leader" node), matensemble's own default (None)
    # would already share rank 0, but this is set explicitly so it stays
    # correct if this pipeline ever grows a multi-rank stage later (where
    # the default flips to reserving rank 0 entirely -- see
    # matensemble.pipeline.Pipeline's own docstring).
    pipe = Pipeline(reserve_broker_node=False)
    env = qe_container_env(cores_per_task)

    @pipe.chore(name="run_qe", num_tasks=1, cores_per_task=cores_per_task, gpus_per_task=gpus_per_task,
                env=env, inherit_env=True, mpi=False)
    def run_qe_chore(task_dict):
        """Thin chore wrapper -- the real pw.x-execution logic lives in DFTMatEnsemble.run_individual."""
        return DFTMatEnsemble.run_individual(task_dict)

    for dft_dct in dft_dct_list:
        # num_tasks fixed at 1 regardless of dft_dct['allocated_nodes'] --
        # see module docstring for why (single-MPI-rank pw.x, sidestepping
        # the open PMIx-under-Flux question until the rest of this stage is
        # confirmed working). max_tasks_per_job is accepted for forward
        # compatibility with a future multi-rank version of this stage, not
        # consulted yet.
        resources = Resources(num_tasks=1, cores_per_task=cores_per_task, gpus_per_task=gpus_per_task,
                               env=env, inherit_env=True, mpi=False)
        pipe.call("run_qe", dft_dct, resources=resources)

    future = pipe.submit(log_delay=25)
    future.result()


# ---------------------------------------------------------------------------
# Stage: build_ff_inputs
# ---------------------------------------------------------------------------

def run_build_ff_inputs(config):
    """
    Builds the ACE training dataset (.pckl.gzip, from converged QE data
    under DFT/training + the isolated-atom reference under
    DFT/isolated_elements), then a randomly-sampled ensemble of pyace
    input.yaml folders varying seed + kappa (fit.loss's energy-vs-force
    weight) -- mirrors the MACE example's own build_ff_inputs stage in
    shape (dataset construction, then ensemble-input construction, within
    one stage function), adapted to ACE's own single-self-contained-file-
    per-fit config shape rather than MACE's three separate train/test/
    config files.

    backend_config's n_workers is set explicitly here from THIS config's
    own cores_per_task (falling back to fine_tuning's own value if unset)
    -- baked into each ensemble member's input.yaml at THIS build time, not
    re-derived at fit time, so it must be kept in sync with fine_tuning's
    own cores_per_task by hand if either changes later (no automated
    mechanism enforcing that). See pipeline/FRICTION_LOG.md for why this
    matters: pyace's own parallel_mode='process' default (multiprocessing.
    cpu_count()) has no idea what subset of a node's cores Flux actually
    handed a given chore, and will happily oversubscribe/undersubscribe
    otherwise -- same class of concern as OMP_NUM_THREADS in
    qe_container_env above, just baked into a config file at ensemble-
    build time instead of a chore-level env var, since pyace reads
    n_workers from its own backend config, not an environment variable.
    """
    from EnsembleFFFit.potential.ace.build_ace_dataset import write_ace_dataset
    from EnsembleFFFit.potential.ace.build_ace_ensemble_inputs import build_ace_ensemble_inputs

    cfg = config['build_ff_inputs']
    _require(cfg, 'build_ff_inputs', ['dft_root', 'dataset_path', 'ace_inputs_dir'])

    dft_root = Path(cfg['dft_root'])
    training_root = dft_root / cfg.get('training_subpath', 'training')
    isolated_elements_root = dft_root / cfg.get('isolated_elements_subpath', 'isolated_elements')
    structure_filename = cfg.get('structure_filename', 'POSCAR')

    df, dataset_path = write_ace_dataset(
        run_directory=str(training_root),
        isolated_elements_directory=str(isolated_elements_root),
        output_path=cfg['dataset_path'],
        structure_filename=structure_filename,
    )
    print(f"Wrote ACE dataset with {len(df)} structure(s) to {dataset_path}")

    # Absolute, not write_ace_dataset's own (possibly relative) return value --
    # each ensemble folder's input.yaml embeds this path in data.filename, but
    # ace_fit.py chdirs into its own work_dir before GeneralACEFit reads it, so
    # a relative path here would resolve against the wrong directory at fit
    # time (confirmed: FileNotFoundError, pyace searching for the dataset
    # under FF/ace_dataset/<i>/ instead of FF/ace_dataset/).
    dataset_path = Path(dataset_path).resolve()

    cores_per_task = cfg.get('cores_per_task', config.get('fine_tuning', {}).get('cores_per_task', 1))

    written = build_ace_ensemble_inputs(
        dataset_path=dataset_path,
        output_dir=cfg['ace_inputs_dir'],
        elements=cfg.get('elements', ['Si']),
        seeds=tuple(cfg.get('seeds', (0, 1, 2))),
        kappa_grid_kwargs=cfg.get('kappa_grid_kwargs'),
        total_cap=cfg.get('total_cap', 10),
        sample_seed=cfg.get('seed', 0),
        backend_config={'parallel_mode': 'process', 'n_workers': cores_per_task},
    )
    print(f"Wrote {len(written)} ACE ensemble input folder(s) under {cfg['ace_inputs_dir']}")


# ---------------------------------------------------------------------------
# Stage: fit_and_validate
# ---------------------------------------------------------------------------

def run_fit_and_validate(config):
    """
    Submits one ACE-fitting chore per ensemble input folder (see
    build_ff_inputs) via FFMatEnsemble -- deliberately simpler than the MACE
    example's own fit_and_validate (no reactive validation-spawn strategy,
    no finite-temperature MD batch): this pipeline has no downstream MD/UQ
    story wired up for ACE yet, so this stage's only job is producing N
    fitted potentials, one per ensemble member.

    check_files=['dataset'] anchors on the single shared si.pckl.gzip under
    run_directory (matching MACE's own foundation_model-as-check_files-
    anchor convention); 'config' is the loose inputs_directory key,
    proximity-matched per ensemble folder under inputs_directory -- ACE
    only needs this one per-folder file (unlike MACE's three), since
    data.filename is already embedded in each folder's own input.yaml (see
    build_ace_ensemble_inputs.write_ace_input_yaml).
    """
    from matensemble.pipeline import Pipeline
    from matensemble.model import Resources
    from EnsembleFFFit.base import FFMatEnsemble
    from EnsembleFFFit.utilities.general import ensemble_fffit_pythonpath

    cfg = config['fine_tuning']
    _require(cfg, 'fine_tuning', ['run_directory', 'inputs_directory', 'ff_task'])

    options = {'dataset': cfg.get('dataset', 'si.pckl.gzip'), 'config': cfg.get('config', 'input.yaml')}
    ff_task = str(Path(cfg['ff_task']).resolve())
    entry_point = cfg.get('entry_point', 'run_ace_fit')
    check_files = cfg.get('check_files', ['dataset'])
    finished_file = cfg.get('finished_file')

    ff_matensemble = FFMatEnsemble(cfg['run_directory'], cfg['inputs_directory'], **options)
    ff_dct_list = ff_matensemble.build_ff_dcts(ff_task, check_files, entry_point, finished_file=finished_file)

    if not ff_dct_list:
        raise ValueError(
            f"No (dataset, config) pairs found -- check run_directory/inputs_directory in the "
            f"'fine_tuning' config section (has build_ff_inputs been run yet?)."
        )

    resources_kwargs = dict(num_tasks=cfg.get('num_tasks', 1),
                             cores_per_task=cfg.get('cores_per_task', 1),
                             gpus_per_task=cfg.get('gpus_per_task', 0))
    env = {'PYTHONPATH': ensemble_fffit_pythonpath()}

    pipe = Pipeline()

    @pipe.chore(name="fit_ace", **resources_kwargs, env=env, inherit_env=True)
    def fit_ace_chore(task_dict):
        """Thin chore wrapper -- the real fitting logic lives in whatever driver script
        fine_tuning.ff_task points at (see FFMatEnsemble.run_individual)."""
        return FFMatEnsemble.run_individual(task_dict)

    resources = Resources(**resources_kwargs, env=env, inherit_env=True)
    for ff_dct in ff_dct_list:
        pipe.call("fit_ace", ff_dct, resources=resources)

    future = pipe.submit(log_delay=25)
    future.result()


# ---------------------------------------------------------------------------
# Stage: molecular_dynamics
# ---------------------------------------------------------------------------

def run_molecular_dynamics(config):
    """
    Submits one MD chore per (fitted ACE potential, structure) pair --
    every fitted potential found under fine_tuning's own run_directory
    (FF/ace_dataset/<i>/FF_<i>.yaml, one per ensemble member from
    fit_and_validate) cross-producted with every structure found under
    structures_root.

    Deliberately uses MDMatEnsemble.build_flat_task_dicts, not
    build_task_dicts/build_lists' own proximity-matching machinery --
    CONFIRMED unusable here (2026-09-17): base.py's _collect_paths matches
    an options value by EXACT filename across run_directory's whole tree,
    but each ensemble member's fitted potential has a genuinely different
    literal filename (FF_0.yaml, FF_1.yaml, ... FF_5.yaml, from
    ace_fit.py's own f"{name}.yaml" convention) rather than one shared
    name repeated per folder the way FFMatEnsemble's dataset/config
    matching relies on. build_flat_task_dicts is exactly the "already a
    known, explicit list" escape hatch base.py itself documents for this
    shape (see its own docstring) -- called once per discovered potential
    file here instead of once total.

    Every MD run is single-core by design, not just by this stage's own
    default: pyace's native evaluator (MD/ace_md.py's PyACECalculator) has
    no internal OpenMP/pthread parallelism (confirmed via `ldd` on
    pyace/calculator*.so -- see pipeline/FRICTION_LOG.md), so
    cores_per_task here mainly buys concurrent chores (multiple ensemble
    members' MD runs at once), not a faster individual trajectory. Genuine
    multi-core speedup on one large-supercell trajectory needs LAMMPS's
    own ML-PACE pair_style (real MPI domain decomposition) -- not
    currently built into this container's LAMMPS, also logged there.
    """
    from matensemble.pipeline import Pipeline
    from matensemble.model import Resources
    from EnsembleFFFit.base import MDMatEnsemble
    from EnsembleFFFit.utilities.general import ensemble_fffit_pythonpath

    cfg = config['molecular_dynamics']
    _require(cfg, 'molecular_dynamics', ['ff_root', 'structures_root', 'md_task', 'output_root'])

    ff_root = Path(cfg['ff_root'])
    ff_glob = cfg.get('ff_glob', 'FF_*.yaml')
    structures_root = cfg['structures_root']
    structure_filename = cfg.get('structure_filename', 'POSCAR')
    md_recipe = cfg.get('md_recipe')
    md_task = str(Path(cfg['md_task']).resolve())
    entry_point = cfg.get('entry_point', 'run_ace_md')
    output_root = Path(cfg['output_root'])
    finished_file = cfg.get('finished_file', 'md_run.traj')

    # One glob per ensemble-member subfolder (FF/ace_dataset/<i>/FF_<i>.yaml)
    # rather than a single recursive glob -- keeps `member` (used below to
    # namespace each potential's own output subtree) tied to the same
    # subfolder convention fit_and_validate/ace_fit.py already writes into,
    # regardless of what ff_glob itself matches.
    potentials = sorted(ff_root.glob(f'*/{ff_glob}'))
    if not potentials:
        raise ValueError(
            f"No fitted potential(s) matching '{ff_glob}' found under {ff_root}/*/ -- "
            f"has fit_and_validate been run yet?"
        )

    md_dct_list = []
    for potential in potentials:
        member = potential.parent.name  # e.g. "0", "1", ... matching FF/ace_dataset/<i>/
        md_dct_list.extend(MDMatEnsemble.build_flat_task_dicts(
            structures_root=structures_root,
            foundation_model=str(potential),
            in_file=md_recipe,
            output_root=str(output_root / member),
            task_command=md_task,
            entry_point=entry_point,
            structure_filename=structure_filename,
            finished_file=finished_file,
        ))

    if not md_dct_list:
        raise ValueError(
            f"Found {len(potentials)} fitted potential(s) under {ff_root}, but every "
            f"structure under {structures_root} already has a '{finished_file}' match "
            f"in its corresponding output directory -- nothing new to run. Set "
            f"finished_file to null (or a pattern that doesn't already exist "
            f"everywhere) to rerun them anyway."
        )

    resources_kwargs = dict(num_tasks=cfg.get('num_tasks', 1),
                             cores_per_task=cfg.get('cores_per_task', 1),
                             gpus_per_task=cfg.get('gpus_per_task', 0))
    env = {'PYTHONPATH': ensemble_fffit_pythonpath()}

    pipe = Pipeline()

    @pipe.chore(name="run_ace_md", **resources_kwargs, env=env, inherit_env=True)
    def run_ace_md_chore(task_dict):
        """Thin chore wrapper -- the real MD logic lives in whatever driver script
        molecular_dynamics.md_task points at (see MDMatEnsemble.run_individual)."""
        return MDMatEnsemble.run_individual(task_dict)

    resources = Resources(**resources_kwargs, env=env, inherit_env=True)
    for md_dct in md_dct_list:
        pipe.call("run_ace_md", md_dct, resources=resources)

    future = pipe.submit(log_delay=25)
    future.result()


# ---------------------------------------------------------------------------
# Stage: copy_force_fields
# ---------------------------------------------------------------------------

def run_copy_force_fields(config):
    """
    Copies every fitted ACE potential (FF/ace_dataset/<i>/FF_<i>.yaml) into
    MD/single_points/force_fields/<i>/FF.yaml -- a uniform filename per
    subfolder, reusing the EXISTING <i> subfolder structure fit_and_validate
    already created (unlike the MACE example's own copy_force_fields, which
    derives a fresh subfolder name via a regex capture group on the source
    filename -- not needed here, since our source tree already has one
    subfolder per ensemble member; make_mirrored_rename_dest_path_fn mirrors
    that existing relative subtree as-is).

    Required, not cosmetic: MDMatEnsemble.build_task_dicts' own proximity
    matcher (_collect_paths) matches an options value by EXACT filename
    across the whole run_directory tree, so six differently-named files
    (FF_0.yaml ... FF_5.yaml) can't be matched at all -- confirmed the hard
    way while building the molecular_dynamics stage (see
    pipeline/FRICTION_LOG.md). A uniform name per folder is what makes
    single_points' own build_task_dicts call possible in the first place.

    Uses EnsembleFFFit.utilities.general's own copy_and_transform_files/
    make_mirrored_rename_dest_path_fn -- the same reusable utility the MACE
    example's own (reactive, inlined) force-field copy uses, not bespoke
    logic written for this pipeline.
    """
    from EnsembleFFFit.utilities.general import copy_and_transform_files, make_mirrored_rename_dest_path_fn

    cfg = config['copy_force_fields']
    _require(cfg, 'copy_force_fields', ['source_directory', 'target_directory'])

    source_directory = cfg['source_directory']
    target_directory = cfg['target_directory']
    pattern = cfg.get('pattern', r'FF_.*\.yaml')
    target_name = cfg.get('target_name', 'FF.yaml')

    dest_path_fn = make_mirrored_rename_dest_path_fn(source_directory, target_directory, target_name)
    copy_and_transform_files(source_directory, target_directory, pattern, dest_path_fn)


# ---------------------------------------------------------------------------
# Stage: mirror_training_structures
# ---------------------------------------------------------------------------

def run_mirror_training_structures(config):
    """
    Mirrors DFT/training's own completed leaves (POSCAR + properties.json,
    the DFT ground truth) into MD/single_points/ase_inputs/training -- same
    convention as the MACE example's own validation_mirroring (there
    mirroring DFT/validation instead, since that example has a genuine
    held-out validation set; this project doesn't yet, hence mirroring the
    training set itself here -- see this stage's own workflow_config.yaml
    comment). The mirrored properties.json isn't consumed by single_points
    itself (PyACECalculator computes its own from scratch) -- it's there
    for a later analysis.best_force_field-style RMSE-vs-DFT comparison,
    the same role it plays in the MACE example.
    """
    from EnsembleFFFit.utilities.general import mirror_completed_leaves

    cfg = config['mirror_training_structures']
    _require(cfg, 'mirror_training_structures', ['source_root', 'dest_root'])

    copied = mirror_completed_leaves(cfg['source_root'], cfg['dest_root'],
                                      tuple(cfg.get('filenames', ['POSCAR', 'properties.json'])))
    print(f"Mirrored {len(copied)} completed leaf(ves) from {cfg['source_root']} to {cfg['dest_root']}")


# ---------------------------------------------------------------------------
# Stage: single_points
# ---------------------------------------------------------------------------

def run_single_points(config):
    """
    Flat, non-reactive single-point submission -- every fitted ACE
    potential (MD/single_points/force_fields/<i>/FF.yaml, from
    copy_force_fields) cross-producted with every mirrored training
    structure (MD/single_points/ase_inputs/training/<name>/POSCAR, from
    mirror_training_structures), batched into ONE chore per force field
    covering both structures (parent_levels tuned to this project's own
    directory shape -- confirmed directly, not assumed from the MACE
    example's own parent_levels=8, which was tuned to a much deeper UQ-
    frame tree that doesn't apply here). Mirrors run_uq_single_points' own
    flat/non-reactive shape, not run_fit_and_validate's reactive
    fit-then-spawn strategy -- that doesn't apply here either, since
    fitting already happened as its own separate, already-completed stage.
    """
    from matensemble.pipeline import Pipeline
    from matensemble.model import Resources
    from EnsembleFFFit.base import MDMatEnsemble
    from EnsembleFFFit.utilities.general import ensemble_fffit_pythonpath

    cfg = config['single_points']
    _require(cfg, 'single_points', ['run_directory', 'inputs_directory'])

    options = {'ffield': cfg.get('ffield', 'FF.yaml'),
               'structure': cfg.get('structure', 'POSCAR')}

    resources_kwargs = dict(num_tasks=cfg.get('num_tasks', 1),
                             cores_per_task=cfg.get('cores_per_task', 1),
                             gpus_per_task=cfg.get('gpus_per_task', 0))
    env = {'PYTHONPATH': ensemble_fffit_pythonpath()}

    md = MDMatEnsemble(cfg['run_directory'], cfg['inputs_directory'], **options)
    task_dicts = md.build_task_dicts(
        cfg.get('single_point_task', 'ace_single_point.py'),
        cfg.get('parent_levels', 1),
        cfg.get('check_files', ['ffield']),
        cfg.get('entry_point', 'run_ase_single_points'),
        finished_file=cfg.get('finished_file'),
    )
    if not task_dicts:
        raise ValueError(
            "No (force field, structure) batches found -- check run_directory/inputs_directory in the "
            "'single_points' config section (have copy_force_fields/mirror_training_structures been run yet?)."
        )

    pipe = Pipeline()

    @pipe.chore(name="run_single_point", **resources_kwargs, env=env, inherit_env=True)
    def run_single_point_chore(task_dict):
        """Thin chore wrapper -- the real single-point logic lives in MDMatEnsemble.run_individual."""
        return MDMatEnsemble.run_individual(task_dict)

    resources = Resources(**resources_kwargs, env=env, inherit_env=True)
    for task_dict in task_dicts:
        pipe.call("run_single_point", task_dict, resources=resources)

    future = pipe.submit(log_delay=25)
    future.result()


STAGES = {
    'converge_dft_data': run_converge_dft_data,
    'build_ff_inputs': run_build_ff_inputs,
    'fit_and_validate': run_fit_and_validate,
    'molecular_dynamics': run_molecular_dynamics,
    'copy_force_fields': run_copy_force_fields,
    'mirror_training_structures': run_mirror_training_structures,
    'single_points': run_single_points,
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
