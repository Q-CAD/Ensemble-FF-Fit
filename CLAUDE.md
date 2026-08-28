# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

EnsembleFFFit coordinates data- and time-efficient fine-tuning of interatomic potentials/force-fields
(both physics-based and ML, e.g. JAX-ReaxFF, MACE, LAMMPS-based FFs) by fitting an *ensemble* of models
against ab initio (DFT) data, using adaptive asynchronous job scheduling (MatEnsemble, an external sibling
package) plus on-the-fly uncertainty quantification (UQ) to decide what new training data to generate
next. It now also drives the DFT convergence step itself (RMG, via `density_functional_theory/rmg/`), not
just the fitting/UQ side — see `examples/Frontier/RMG_MACE_ASE/` for the full loop. It targets HPC
clusters — historically NERSC Perlmutter (SLURM + Cray `cc`/`CC`/`ftn` compilers, CUDA GPUs), now also
OLCF Frontier (ROCm/AMD GPUs) via a MatEnsemble+Flux container. LAMMPS itself (with Python bindings) is
expected to be provided by that container on both systems, not built or pip-installed by this project.

Companion projects it depends on (not vendored here, resolved as normal `pyproject.toml` dependencies):
`MatEnsemble` (Flux-based async task execution; pinned as a version check against the container's
pre-installed copy, not fetched from git). `vaspflux` (DFT job orchestration), `parse2fit` (parses raw
DFT/MD output into FF training-set formats), and `HeteroBuilder` (importable as `vdW_structures`;
heterostructure generation) are currently commented out in `pyproject.toml` — deferred, not core
dependencies of the RMG/MACE/ASE pipeline this repo currently exercises.

## Install / environment setup

Two-step process — `pyproject.toml` alone can't express "fetch this package from a different index
depending on which GPU platform is present," so that part lives in a standalone script instead:

```bash
python install_gpu_torch.py                     # step 1: detect ROCm/CUDA, install the matching torch build
pip install -e ".[mace,cuda]"                    # step 2: backend extra + platform extra, combined as needed
```

See **README.md's Installation section** for the full set of extras (`mace`, `reaxff`, `torchsim` ×
`cuda`, `rocm`) and example combinations — not duplicated here. A few implementation notes worth knowing
if you're touching this area:

- `install_gpu_torch.py` fails loudly (non-zero exit, clear message) rather than guessing if it can't
  confidently detect the platform, or if it detects signals for *both* ROCm and CUDA. Its `PLATFORM_MAP`
  pins exact `torch`/`torchvision`/`torchaudio` versions per platform — verify these against
  `download.pytorch.org`'s wheel indices before trusting them if it's been a while (they were verified
  live when written, not assumed from memory; see the script's own header comment for the exact date/URLs
  checked). `cuda12.4` is still unvalidated against a real Perlmutter container build — see `TODO.md`.
- **The official LAMMPS Python module and `mpi4py` are deliberately not pip dependencies anywhere** — both
  need to link against whatever specific MPI/compiler LAMMPS itself was built against, which is
  container-specific (confirmed container-provided on Frontier; assumed but not yet confirmed on
  Perlmutter, see `TODO.md`). There is accordingly no `lammps` extra. Every driver script's `import
  lammps`/`import lammps.mliap` goes through
  `EnsembleFFFit.molecular_dynamics.pyMD.helpers.import_lammps()` / `import_lammps_mliap()`, which raise a
  clear, actionable error (rather than a bare `ModuleNotFoundError`) if the module isn't set up yet.
- `openequivariance` (the `rocm` extra's MACE accelerator) needs a working GCC 9+ and HIP toolchain *at
  pip-install time* to build its kernels — it isn't a prebuilt wheel like `cuequivariance` is.
- Registering the Jupyter kernel for the `notebooks` extra is a separate one-time imperative step that
  can't live in `pyproject.toml`: `python -m ipykernel install --user --name ensemblefffit --display-name
  "Ensemble-FF-Fit"`.

This supersedes a set of per-backend `build_*.sh` conda-env scripts that used to live at the repo root
(each created its own conda env, cloned sibling repos, and `pip install -e . --no-deps`'d everything). They
were retired once this extras-based approach existed; see git history (or the `main` branch, pre-merge) if
you need to recover exactly what they did for a specific backend/HPC quirk that isn't captured here or in
`TODO.md`.

## No test suite

There are no test files, `tests/` directory, or pytest config anywhere in this repo. Validate changes by
running the relevant CLI end-to-end, or against `examples/Frontier/RMG_MACE_ASE/` (the worked RMG -> MACE
-> ASE pipeline example — see its own `README.md`), rather than by writing/running unit tests, unless
you're explicitly asked to add one.

## Architecture

The package (`EnsembleFFFit/`) is organized around the ensemble-fitting *pipeline* — generate structures →
run ensemble force-field fits (via MatEnsemble) → quantify disagreement/error → pick new structures — and
each stage is a separate subpackage exposing argparse CLIs registered as console-scripts in `pyproject.toml`.

This layout is the result of a mid-2026 reorganization (structure-by-purpose rather than the previous
structure-by-legacy-name); see `TODO.md` for what's still deferred from that pass, and git history for the
detailed rationale behind specific renames/decisions if it's ever needed.

### `base.py` / `in_queue.py` — job orchestration core

This is the piece that actually drives async, adaptive ensemble fitting; everything else feeds it inputs
or consumes its outputs. Both live at the top of the `EnsembleFFFit` package (not inside a subpackage) —
`potential/`, `molecular_dynamics/`, and the MLIP-backend CLIs below all import directly from
`EnsembleFFFit.base`.

- **`base.py`** defines `MatEnsembleJob(ABC)`, with abstract `build_full_runs()`/`batch_by_parent()`.
  Shared logic: given a `run_directory` (where jobs/results live) and `inputs_directory` (template
  inputs), each subclass walks both trees, matches files by directory *proximity*
  (`_make_proximity_combinations`) or by a flat recipe-file cross-product, and batches individual tasks by
  common parent directory so one chore can process a whole set of related structures/seeds at once. There
  is no `run()`/`SuperFluxManager`/`poolexecutor` call anymore — each subclass instead exposes a static
  `run_individual(task_dict)` that a thin `@pipe.chore`-decorated wrapper function in the *submission
  script* calls; the submission script builds a `matensemble.pipeline.Pipeline`, registers chores via
  `pipe.call(...)`, and submits via `pipe.submit(...)` — see
  `examples/Frontier/RMG_MACE_ASE/run_pipeline.py` for the current working pattern end-to-end.
- Four concrete subclasses, one per backend, each supplying the sizing/task-arg logic `base.py` needs and
  its own `run_individual`:
  - `MDMatEnsemble` (in `base.py`; renamed from `LammpsMatEnsemble` and generalized) — covers ASE and
    TorchSim MD drivers as well as LAMMPS; `run_individual` dynamically imports the user-supplied driver
    script by path and dispatches to its named entry-point function.
  - `JaxReaxFFMatEnsemble` — converts an argparse namespace into per-task CLI-arg lists for JAX-ReaxFF.
  - `MACEMatEnsemble` — `build_mace_dcts` fans a single run-path into per-seed subdirectories so each
    ensemble member gets its own MACE fit; `run_individual` calls `mace.cli.run_train.run` directly. See
    `TODO.md` for the planned generalization of this class into a backend-agnostic `FFMatEnsemble`.
  - `DFTMatEnsemble` — the RMG DFT backend; `build_dft_dcts` sizes each structure's node/GPU footprint
    (see `density_functional_theory/rmg/` below), `run_individual` dispatches to a site-specific driver
    script (e.g. `rmg_dft.py`) by path/entry-point, same convention as `MDMatEnsemble`.
- **`in_queue.py`** is a separate SLURM-level helper (`sbatch` submission, `squeue` polling, sentinel-file
  based done/fail detection, auto-resubmission) operating one level above MatEnsemble's in-job Flux task
  distribution. It has no callers anywhere in this repo (its only caller was an example notebook that has
  since been removed) — a deletion candidate, not yet acted on.

### `density_functional_theory/rmg/` — RMG DFT backend

Backs `DFTMatEnsemble`. `rmg_calculator.py` is an ASE `Calculator` subclass wrapping the `rmg-gpu`/`rmg-cpu`
binary (bare `{rmg_executable} {rmg_name}` invocation by deliberate design — Flux/MatEnsemble owns launch
semantics for the surrounding chore, so this never wraps the command in its own `srun`/`mpirun`/`flux run`).
`rmg_input.py` builds RMG's own input-file format from a yaml recipe + structure, including
`compute_grid_and_resources` (grid sizing / node-count estimation — `processor_grid.py` holds the actual
grid-search logic). `pick_structure.py` resolves which structure file to actually run against at execution
time (a fresh `POSCAR` vs. a newer `rmg_input.*.log`/`rmg_input` left by a previous attempt in the same
working directory). `convergence.py`/`rmg_log.py` parse RMG's own log output for SCF convergence status.
`valence.py`/`forcefield.py` handle pseudopotential valence-electron counts and force-field-format output.
See `examples/Frontier/RMG_MACE_ASE/README.md` for the full container/build/launch story around actually
running `rmg-gpu` — that operational knowledge lives there, not here.

### `potential/{mace,reaxff}/` and `molecular_dynamics/pyMD/` — per-backend drivers

- **`potential/mace/mace_matensemble_cli.py`**, **`potential/reaxff/jaxreaxff_matensemble_cli.py`**, and
  **`molecular_dynamics/pyMD/lammps_matensemble_cli.py`** are deprecated, not-yet-deleted console scripts
  (their `pyproject.toml` registrations have already been removed) — they depended on `MatEnsembleJob.run()`,
  which no longer exists. The current pattern is the `Pipeline`/`@pipe.chore` one described above; see
  `TODO.md` for what still needs to happen before these files themselves are deleted.
- **`molecular_dynamics/pyMD/drivers/`** (`lammps_reaxff_cpu.py`, `lammps_mace_kokkos_gpu.py`,
  `ase_mace.py`, `torch_sim_mace.py`) and **`molecular_dynamics/pyMD/examples/*/inputs_directory/*.py`**
  (manually-kept-in-sync copies of those drivers, per MatEnsemble's convention of expecting the task script
  to live alongside the structure/force-field files inside `inputs_directory/`) predate the current
  `Pipeline`/chore pattern and haven't been exercised against it — likely stale relative to what
  `examples/Frontier/RMG_MACE_ASE/MD/*/ase_inputs/*.py` now demonstrates working. Flagged for review, not
  yet removed. `potential/mace/build_ensemble_inputs.py`/`write_training_xyz.py` are the current,
  actively-used MACE training-input builders.
- `molecular_dynamics/pyMD/helpers.py` also has the `import_lammps()`/`import_lammps_mliap()` guards
  described above.
- **Planned reorganization** (not yet done): mirror `density_functional_theory/`'s per-backend-folder
  layout — `molecular_dynamics/{ase,lammps,torchsim}/` instead of everything nested under the single
  `pyMD/` folder, dropping the `pyMD` name entirely. See `TODO.md`.

### `structures/` — training-structure generation CLIs

Each subfolder walks a tree of POSCARs and writes new structure variants:
`defects/` (vacancy/antisite/interstitial/substitution via `pymatgen.analysis.defects`),
`equation_of_state/` (volume-rescaled EoS points), `materials_project/` (`mp_query`, pulls structures
from the Materials Project API by mpid/chemsys), `substitutions/` (ionic-radius-guided element
substitution + volume prediction), `vdW_layers/` (interlayer-spacing sampling for vdW heterostructures,
depends on the external `HeteroBuilder`/`vdW_structures` package), `deviation_selection/` (UQ-driven:
consumes parsed ensemble single-point data, computes per-site energy/force variance across force fields,
writes out the highest/lowest-variance structures as the next round's training candidates — has known
limitations, see `TODO.md`).

### `analysis/` — ensemble scoring / best-FF selection

Pipeline: `dict_parsers.py` (generic ASE/VASP single-point ingestion) and `lammps_properties.py`
(`parse_single_points`, LAMMPS-specific ingestion into the same nested `{label: {run: {image: {...}}}}`
shape) parse raw run output → `variance.py` (`get_structures_scores`) scores *ensemble disagreement* per
MD image, driving which new structures get added to training → `best_force_field.py`
(`get_ff_deviations`/`rank_ff_scores`) scores each ensemble member's RMSE against DFT ground truth,
driving which force field is "best" → `downselect_force_fields.py` copies the top-ranked force fields
forward into the next stage. `cn_checker_cli.py` is an independent structural-QC tool (coordination-number
deviation via `pymatgen`'s `CrystalNN`), orthogonal to the energy/force analysis chain above — its
`pyproject.toml` console-script registration has been removed (deprecated CLI surface, not deleted).

### `utilities/` — misc CLIs and structure deduplication

`create_lammps_models_cli.py`/`formation_energy_lammps_runs.py`/`parse_vasp_aimd_cli.py` are standalone
helper CLIs. `copy_by_pattern_cli.py`'s console-script registration has been removed (same as
`cn_checker_cli.py` above) — the file remains, unregistered. `cluster_lammps_runs.py` featurizes
structures with matminer's `CrystalNNFingerprint`, builds a pairwise dissimilarity matrix, and
hierarchically clusters to pick representative structures per formula/cluster (reduces redundant training
data) — it's the one module here that's part of the generate→analyze pipeline rather than a standalone
tool. `create_lammps_models_cli.py` needs the `mace` extra despite being registered as a core console
script; see `TODO.md` for how that's handled, and for the plan to eventually move it under `potential/mace/`.

### `examples/`

`examples/Frontier/RMG_MACE_ASE/` is a full worked walkthrough of the current pipeline on OLCF Frontier —
RMG DFT convergence → MACE ensemble fitting/validation → ASE finite-temperature MD sampling → UQ-based
downselection of next-round DFT candidates — driven by a single `run_pipeline.py`/`workflow_config.yaml`
pair via the `Pipeline`/`@pipe.chore` pattern described above. It's a trimmed copy of a real, working run
(structures + driver/recipe scripts only, no run output, foundation model fetched separately — see its own
`README.md`), and is the closest thing this repo has to an integration test and to end-user-facing usage
documentation. Read it before changing pipeline-stage interfaces.

## Security note

`structures/materials_project/mp_query_cli.py` reads a Materials Project API key from a local
`api_key.yml` next to it; that filename is not covered by `.gitignore`, so watch for accidental commits
of real API keys if one gets created during development. (One such file already exists, checked in since
this repo's first commit, containing what looks like a real key — see git history if this needs rotating.)
