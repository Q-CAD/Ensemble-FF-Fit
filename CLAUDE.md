# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

EnsembleFFFit coordinates data- and time-efficient fine-tuning of interatomic potentials/force-fields by
fitting an *ensemble* of models against ab initio (DFT) data, using adaptive asynchronous job scheduling
(MatEnsemble, an external sibling package) plus on-the-fly uncertainty quantification (UQ) to decide what
new training data to generate next. Three FF-fitting backends are demonstrated end-to-end, all through the
same generic `FFMatEnsemble`/driver-script contract (see `base.py` below) rather than any backend-specific
class: **MACE** (`examples/Frontier/RMG_MACE_ASE/`), **JAX-ReaxFF** (`examples/Perlmutter/
VASP_ReaxFF_LAMMPs/`), and **ACE**/pyace (`examples/HPC_aas/QE_ACE/`). It also drives the DFT convergence
step itself, again via the same generic `DFTMatEnsemble`/driver-script contract across three DFT codes:
**RMG** (via the `pyRMG` package — see below), **VASP**, and **Quantum Espresso**. It targets HPC
clusters — historically NERSC Perlmutter (SLURM + Cray `cc`/`CC`/`ftn` compilers, CUDA GPUs), OLCF Frontier
(ROCm/AMD GPUs), and now also ORNL's Pathfinder HPC-as-a-Service offering (SLURM + Apptainer, CUDA GPUs,
Flux bootstrapped over PMIx rather than Cray PMI2) — via a MatEnsemble+Flux container on each. LAMMPS
itself (with Python bindings) is expected to be provided by that container where MD is actually run through
it, not built or pip-installed by this project; Pathfinder's own container does not currently have LAMMPS's
`ML-PACE` package built in, so ACE MD there goes through a plain-ASE driver instead (CPU-only, single-core
per trajectory — see `examples/HPC_aas/QE_ACE/README.md`'s pitfalls section).

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

See **README.md's Installation section** for the full set of extras (`mace`, `jaxreaxff`, `ace`,
`torchsim` × `cuda`, `rocm`) and example combinations — not duplicated here. A few implementation notes
worth knowing if you're touching this area:

- `ace` (pyace/pacemaker) has no PyPI release, pins a git ref, and compiles native CMake C++ extensions at
  pip-install time — cap parallelism explicitly (`CMAKE_BUILD_PARALLEL_LEVEL=2 pip install -e ".[ace]"`)
  on a shared/memory-constrained host, or its `setup.py`'s own `os.cpu_count()-1` default can exhaust
  memory. Deliberately does *not* pull in `tensorpotential` (pacemaker's GPU/TensorFlow evaluator) — its
  own `setup.py` requires Python <3.11, incompatible with this project's Python 3.12 containers; pyace's
  native, CPU-only evaluator (the default if `backend.evaluator` is omitted from a pacemaker `input.yaml`)
  is used instead everywhere in this repo.

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
  `EnsembleFFFit.molecular_dynamics.helpers.import_lammps()` / `import_lammps_mliap()`, which raise a
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
- Three concrete subclasses, one per backend, each supplying the sizing/task-arg logic `base.py` needs and
  its own `run_individual`. All three now embed the driver-script path (`*_task`) and its entry-point
  function name into every dict `build_*_dcts`/`build_task_dicts` returns, rather than leaving it for the
  caller to inject afterward — `MDMatEnsemble` set this convention originally; `DFTMatEnsemble` and
  `FFMatEnsemble` were retrofitted/designed to match it for consistency (see `TODO.md`).
  - `MDMatEnsemble` (in `base.py`; renamed from `LammpsMatEnsemble` and generalized) — covers ASE and
    TorchSim MD drivers as well as LAMMPS; `run_individual` dynamically imports the user-supplied driver
    script by path and dispatches to its named entry-point function.
  - `FFMatEnsemble` (renamed from `MACEMatEnsemble`) — `build_ff_dcts` fans a single run-path into
    per-seed subdirectories so each ensemble member gets its own fit; a clean, backend-agnostic wrapper —
    `run_individual` dynamically imports the user-supplied fitting driver script (`task_dict['ff_task']`)
    and dispatches to its named entry point, exactly like `MDMatEnsemble`/`DFTMatEnsemble`, rather than
    calling `mace.cli.run_train.run` directly itself. All the actual MACE-fitting logic now lives in
    `examples/Frontier/RMG_MACE_ASE/FF/mace_fit.py`, not here — porting to a different FF backend later
    means writing a new driver script with the same entry-point contract, no `base.py` changes. Two other
    backends already prove this out: JAX-ReaxFF (`JaxReaxFFMatEnsemble`, the old backend-specific class,
    is genuinely gone — but JAX-ReaxFF fitting itself is back, through this same generic contract, via
    `examples/Perlmutter/VASP_ReaxFF_LAMMPs/FF/jax_reaxff_fit.py`) and ACE/pyace (via
    `examples/HPC_aas/QE_ACE/FF/ace_fit.py`, calling `pyace.generalfit.GeneralACEFit` directly — see
    `EnsembleFFFit/potential/ace/` for the shared ensemble-input-building/dataset-conversion utilities
    those two examples' own `build_ff_inputs` stages call, analogous to `potential/mace/`'s).
  - `DFTMatEnsemble` — `build_dft_dcts` sizes each structure's node/GPU footprint, `run_individual`
    dispatches to a site-specific driver script (e.g. `rmg_dft.py`/`vasp_dft.py`/`qe_dft.py`) by
    path/entry-point, same convention as `MDMatEnsemble`/`FFMatEnsemble`. Three DFT backends are
    demonstrated: RMG (see the `pyRMG` section below), VASP (`examples/Perlmutter/VASP_ReaxFF_LAMMPs/`),
    and Quantum Espresso (`examples/HPC_aas/QE_ACE/`, via ASE's own `EspressoTemplate`/`EspressoProfile` —
    evaluated and rejected `pymatgen-io-espresso` first, since it was pre-alpha with no documented
    pseudopotential-resolution support at the time).
- **`in_queue.py`** is a separate SLURM-level helper (`sbatch` submission, `squeue` polling, sentinel-file
  based done/fail detection, auto-resubmission) operating one level above MatEnsemble's in-job Flux task
  distribution. It has no callers anywhere in this repo (its only caller was an example notebook that has
  since been removed) — a deletion candidate, not yet acted on.

**Why the three subclasses' input-passing conventions deliberately differ, not just historically drifted
apart** — each one's rigidity (or lack of it) tracks how much variation is actually expected across real
backends for that concern:
  - `DFTMatEnsemble.options` is fixed to exactly one recipe-file key (named per backend: `rmg_yaml`/
    `vasp_incar`/`qe_yaml`, whichever `DFTMatEnsemble.build_dft_dcts`'s caller sets as `dft_recipe`) plus
    `structure_filename`, full stop — but unlike `MDMatEnsemble`'s fixed shape below, this was never
    because only one DFT backend was expected: VASP and Quantum Espresso are now both demonstrated
    end-to-end (`examples/Perlmutter/VASP_ReaxFF_LAMMPs/`, `examples/HPC_aas/QE_ACE/`), and Gaussian (for
    molecular systems) remains a plausible future addition. It's fixed because "structure file +
    recipe/config file" keeps working as a generic contract *across* those codes — pymatgen/ASE already
    have solid input-generation support for VASP/QE/Gaussian-like codes, so a structure+config pair
    suffices for each without `DFTMatEnsemble` itself needing to change (confirmed twice now, not just
    once). RMG is the outlier here, not the norm: it's obscure enough that it needed genuinely bespoke,
    hand-written support (the `pyRMG` package, an optional dependency) rather than leaning on existing
    Python DFT tooling the way VASP/QE/Gaussian-like codes do. If a future DFT code turns out *not* to fit
    the structure+config shape, that's the point to revisit whether this class needs to generalize — not
    before.
  - `MDMatEnsemble.options` is backend-dependent (`ffield`/`in_file`/`control`/`structure` for LAMMPS, a
    different set for ASE/TorchSim) but still funnels into the same small, *fixed* positional shape at the
    `run_individual`/driver-script boundary (`ffield`, `structure`, `output`, `in_file` — four slots, always
    those four). That's deliberate too: the set of MD drivers this project expects to support (ASE, LAMMPS,
    TorchSim) is itself small and stable, so a fixed 4-slot contract is worth keeping rather than
    generalizing further.
  - `FFMatEnsemble.options` is fully caller-defined (whatever keys `build_ff_dcts`'s `check_files`/
    inputs-directory-keys end up being for whichever FF backend is configured), and its driver-script
    contract is correspondingly the loosest of the three: `run_individual` hands the whole resulting
    overrides dict to the driver script as *one* argument, rather than unpacking into DFT/MD's fixed
    positional-list shape (see the class docstring in `base.py`, and `FF/mace_fit.py`'s own docstring, for
    why unpacking into positional lists here would leak backend-specific key names back into `base.py`).
    This is the one class expected to see the most real variation across backends — different FF-fitting
    codes (MACE, JAX-ReaxFF, CHGNet, ...) have far more divergent input/hyperparameter shapes than DFT or MD
    codes typically do — so it's also the one where the input contract needed to flex the most.

One thing this does *not* cover: assembling MACE's own training-data ensemble (which DFT-converged
structures go into which numbered `train.xyz`/`test.xyz`/`config.yml` folder) is a *separate* concern from
anything `FFMatEnsemble` does, and always has been, on either side of this refactor — that combining logic
lives in `potential/mace/build_ensemble_inputs.py`/`write_training_xyz.py`, called directly by
`run_pipeline.py`'s `build_ff_inputs` stage, well before `fit_and_validate`/`FFMatEnsemble` ever runs.
`FFMatEnsemble.build_ff_dcts` only proximity-matches an already-built `mace_inputs/` tree against foundation
models; it doesn't know how that tree was assembled.

### RMG DFT backend — `pyRMG` (the `rmg` extra), not vendored in-tree

Backs `DFTMatEnsemble`. RMG-specific logic (calculator, input-file generation, processor-grid sizing, log
parsing, structure resolution) lives in the standalone `pyRMG` package, an optional dependency here (the
`rmg` extra — same pattern as `mace`), confirmed working end-to-end against the `RMG_MACE_ASE` example
pipeline. `base.py` and every `rmg_dft.py` driver copy import from `pyRMG` (`pyRMG.rmg_calculator`,
`pyRMG.rmg_input`, `pyRMG.pick_structure`, etc.), not a local copy.

`EnsembleFFFit/density_functional_theory/rmg/` now holds only `rmg_dft.py` — the package-level reference
copy of the driver script `DFTMatEnsemble.run_individual` dispatches to (analogous to
`molecular_dynamics/ase/ase_mace.py`), not RMG logic itself. A deployment's actual `dft_task` should point
at its own copy (e.g. `examples/Frontier/RMG_MACE_ASE/DFT/rmg_dft.py`), hand-kept-in-sync with this one,
same convention as the MD drivers.

`pyRMG.rmg_calculator` is an ASE `Calculator` subclass wrapping the `rmg-gpu`/`rmg-cpu` binary (bare
`{rmg_executable} {rmg_name}` invocation by deliberate design — Flux/MatEnsemble owns launch semantics for
the surrounding chore, so this never wraps the command in its own `srun`/`mpirun`/`flux run`).
`pyRMG.rmg_input` builds RMG's own input-file format from a yaml recipe + structure, including
`compute_grid_and_resources` (grid sizing / node-count estimation — `pyRMG.processor_grid` holds the
actual grid-search logic). `pyRMG.pick_structure` resolves which structure file to actually run against at
execution time (a fresh `POSCAR` vs. a newer `rmg_input.*.log`/`rmg_input` left by a previous attempt in
the same working directory). `pyRMG.convergence`/`pyRMG.rmg_log` parse RMG's own log output for SCF
convergence status. `pyRMG.valence`/`pyRMG.forcefield` handle pseudopotential valence-electron counts and
force-field-format output. See `examples/Frontier/RMG_MACE_ASE/README.md` for the full container/build/
launch story around actually running `rmg-gpu` — that operational knowledge lives there, not here.

### VASP and Quantum Espresso DFT backends — no bespoke package needed, unlike RMG

Both lean entirely on existing, mature Python DFT tooling rather than needing anything like `pyRMG`:
`EnsembleFFFit/density_functional_theory/vasp/vasp_dft.py` wraps `vaspflux`
(`examples/Perlmutter/VASP_ReaxFF_LAMMPs/`); `EnsembleFFFit/density_functional_theory/qe/qe_dft.py` uses
ASE's own `ase.calculators.espresso.EspressoTemplate`/`EspressoProfile` (`examples/HPC_aas/QE_ACE/`) — ASE
was picked over `pymatgen-io-espresso` after actually checking it first (pre-alpha, no PyPI release, no
documented pseudopotential-resolution support at the time). Both, like `rmg_dft.py`, are package-level
*reference* copies only — each deployment's own `dft_task` points at its own hand-kept-in-sync copy under
that example's own `DFT/` folder, same convention throughout.

### `potential/mace/` and `molecular_dynamics/{ase,lammps,torchsim}/` — per-backend drivers

`molecular_dynamics/` mirrors `potential/`/`density_functional_theory/`'s per-backend-folder convention —
`ase/`, `lammps/`, `torchsim/`, each holding that backend's per-task driver script(s), plus a shared
`molecular_dynamics/helpers.py` (`import_lammps()`/`import_lammps_mliap()` guards described above,
`parse_list`, `get_elements`, `make_prop_calculators`) used across all three. There used to be an
intermediate `pyMD/` folder (LAMMPS-flavored name nesting even the non-LAMMPS backends); it's gone —
confirmed via a full repo-wide import search that nothing outside `pyMD/` itself ever referenced it
(`base.py`'s `MDMatEnsemble.run_individual` dispatches to driver scripts via `import_module_from_path`, a
generic by-path loader, not a `pyMD` import).

- `lammps/lammps_reaxff_cpu.py`, `lammps/lammps_mace_kokkos_gpu.py`, `ase/ase_mace.py`,
  `torchsim/torch_sim_mace.py` are per-task Python entry points, invoked once per structure via
  `--lammps_task`/`--*_task` by whatever submission script builds the chore. They predate the current
  `Pipeline`/chore pattern and haven't been exercised against it — likely stale relative to what
  `examples/Frontier/RMG_MACE_ASE/MD/*/ase_inputs/*.py` now demonstrates working; flagged for review, not
  removed.
- `lammps/lammps_matensemble_cli.py` and `potential/mace/mace_matensemble_cli.py` (the deprecated,
  `MatEnsembleJob.run()`-dependent console scripts described in earlier revisions of this doc) have been
  deleted outright, along with every other console script that had no callers anywhere in this repo and no
  `pyproject.toml` registration left (see `TODO.md` for the full list and reasoning) — confirmed via a
  repo-wide import search before deleting anything, same as the `pyMD/` reorganization above. Still
  recoverable from the `main`/`Claude` branches if any of it turns out to be needed as reference later.
- `lammps/lammps_properties.py` (moved here from `analysis/`, since it's LAMMPS-specific) parses LAMMPS
  dump/log output into structures/single-point data (`get_atoms`, `parse_single_points`, `write_poscars`)
  and, merged in from the now-deleted `copy_by_pattern_cli.py` (its only real consumer),
  `get_atom_mapping_from_control` (maps LAMMPS dump placeholder elements back to real elements via a
  `pair_coeff`-parsed `.in.lammps` file) — genuinely reusable LAMMPS dump→POSCAR conversion capability, kept
  even though nothing in the current pipeline calls it yet.
- `potential/mace/build_ensemble_inputs.py`/`write_training_xyz.py` are the current, actively-used MACE
  training-input builders. `potential/mace/create_lammps_models_cli.py` (moved from `utilities/`, since
  it's MACE-specific) needs the `mace` extra despite being a core-registered console script; see `TODO.md`.
- `potential/ace/build_ace_dataset.py`/`build_ace_ensemble_inputs.py` are ACE's own equivalents —
  `build_ace_dataset.write_ace_dataset` converts converged DFT output into pyace's own `.pckl.gzip`
  training-DataFrame format (energy/forces/`energy_corrected`, the isolated-atom-E0-subtracted cohesive
  energy pyace actually fits against — see `examples/HPC_aas/QE_ACE/README.md` if a fitted potential's
  raw energies look wildly different from DFT's own raw totals; that offset is why),
  `build_ace_ensemble_inputs.build_ace_ensemble_inputs` samples the seed/`kappa` ensemble grid into
  numbered `input.yaml` folders, one shared dataset file per ensemble rather than MACE's three
  per-member files. `potential/reaxff/build_reaxff_ensemble_inputs.py` is JAX-ReaxFF's own analogous
  ensemble-input builder.

### `structures/` — training-structure generation CLIs

Each subfolder walks a tree of POSCARs and writes new structure variants:
`defects/` (vacancy/antisite/interstitial/substitution via `pymatgen.analysis.defects`),
`equation_of_state/` (volume-rescaled EoS points), `materials_project/` (`mp_query`, pulls structures
from the Materials Project API by mpid/chemsys), `substitutions/` (ionic-radius-guided element
substitution + volume prediction), `vdW_layers/` (interlayer-spacing sampling for vdW heterostructures,
depends on the external `HeteroBuilder`/`vdW_structures` package). `deviation_selection/` (a UQ-driven
structure-downselection tool) has been deleted — unreferenced anywhere, its role in the current pipeline is
filled by `analysis/variance.py`/`select_dft_candidates` instead; recoverable from `main`/`Claude` if
needed.

### `analysis/` — ensemble scoring / best-FF selection

Pipeline: `dict_parsers.py` (generic ASE/VASP single-point ingestion) parses raw run output →
`variance.py` (`get_structures_scores`) scores *ensemble disagreement* per MD image, driving which new
structures get added to training → `best_force_field.py` (`get_ff_deviations`/`rank_ff_scores`) scores
each ensemble member's RMSE against DFT ground truth, driving which force field is "best" →
`downselect_force_fields.py` copies the top-ranked force fields forward into the next stage.
(`lammps_properties.py`'s LAMMPS-specific ingestion moved to `molecular_dynamics/lammps/`, see above;
`cn_checker_cli.py`, an independent structural-QC tool, has been deleted — unreferenced anywhere.)

### `utilities/` — misc CLIs and structure deduplication

`create_lammps_models_cli.py` used to live here too — moved to `potential/mace/`, see above.
`copy_by_pattern_cli.py`, `cluster_lammps_runs.py`, `formation_energy_lammps_runs.py`, and
`parse_vasp_aimd_cli.py` have all been deleted — none were used anywhere in the current pipeline
(`copy_by_pattern_cli.py`'s one genuinely useful piece, `get_atom_mapping_from_control`, was merged into
`molecular_dynamics/lammps/lammps_properties.py` first, see above, rather than lost). All recoverable from
`main`/`Claude` if needed as reference for a future backend.

### `examples/`

Three full worked walkthroughs, each a trimmed copy of a real, working run (structures + driver/recipe
scripts only, no run output — see each one's own `README.md` for exactly what's excluded and why),
together the closest thing this repo has to an integration test and to end-user-facing usage
documentation. Read the relevant one before changing pipeline-stage interfaces.

- **`examples/Frontier/RMG_MACE_ASE/`** (OLCF Frontier, ROCm) — RMG DFT convergence → MACE ensemble
  fitting/validation → ASE finite-temperature MD sampling → UQ-based downselection of next-round DFT
  candidates. The most complete loop of the three (the only one with active-learning downselection wired
  up end-to-end); foundation model fetched separately, not committed (~80MB checkpoint).
- **`examples/Perlmutter/VASP_ReaxFF_LAMMPs/`** (NERSC Perlmutter, CUDA) — VASP DFT convergence →
  JAX-ReaxFF ensemble fitting/validation → LAMMPS finite-temperature MD + single-point validation.
  Demonstrates `DFTMatEnsemble`/`FFMatEnsemble` against a second DFT code and a second FF-fitting
  backend, both through the exact same generic contract MACE/RMG use.
  Multi-node launch (`launch_multi_node.slurm`) also lives here.
- **`examples/HPC_aas/QE_ACE/`** (ORNL Pathfinder, CUDA, Apptainer) — Quantum Espresso DFT convergence →
  pyACE ensemble fitting → CPU-only single-point re-evaluation against the DFT training set. A third DFT
  code and a third FF-fitting backend, same contract again. Deliberately doesn't attempt
  active-learning downselection or finite-temperature MD as a real demonstration yet — the training set
  is intentionally tiny (2 structures, a smoke test of the pipeline plumbing, not of FF quality), and its
  own README documents, with a real number, why that's currently too undertrained for MD to stay stable
  at all. All-CPU by design: Pathfinder's GPU partition is capacity-constrained, `pw.x` has no GPU build
  there yet, and pyACE's own GPU evaluator needs Python <3.11 (incompatible with this project's Python 3.12
  containers) — see its README's pitfalls section for what was actually checked (not assumed) before
  settling on CPU-only.

## Security note

`structures/materials_project/mp_query_cli.py` reads a Materials Project API key from a local
`api_key.yml` next to it; that filename is not covered by `.gitignore`, so watch for accidental commits
of real API keys if one gets created during development. (One such file already exists, checked in since
this repo's first commit, containing what looks like a real key — see git history if this needs rotating.)
