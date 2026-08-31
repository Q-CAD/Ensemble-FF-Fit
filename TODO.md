# TODO.md

Deferred items surfaced during the `Reformat.md` refactor and the subsequent branching-dependencies
`pyproject.toml` redesign, intentionally left unfixed by explicit decision. Each entry notes where the
decision was made for full context.

## Torch Sim compatibility

`EnsembleFFFit/molecular_dynamics/helpers.py`'s `make_prop_calculators` has `"kinetic_energy"`/
`"temperature"` branches that call `calc_kinetic_energy`/`calc_temperature` from `torch_sim.quantities`,
but that import is commented out at the top of the file. This is intentional for now — Torch Sim isn't
standardly available across the HPC container/runtime builds this package targets, and the new `torchsim`
extra in `pyproject.toml` (`torch-sim-atomistic`) is a separate opt-in for exactly this reason. Revisit
once Torch Sim is standardized in those environments; until then, calling `make_prop_calculators` with a
mapping that includes `"kinetic_energy"` or `"temperature"` will raise `NameError`.

## JAX-ReaxFF support removed from this branch (not ported into `FFMatEnsemble`)

`potential/reaxff/` (`jaxreaxff_matensemble_cli.py`, `JaxReaxFFMatEnsemble`) and the `reaxff`
`pyproject.toml` extra have been deleted outright on this branch, per explicit decision — the upstream
optimizer is deprecated by its own developers, and `FFMatEnsemble` (the `MACEMatEnsemble` generalization,
see below) is deliberately a clean wrapper with no backend-specific logic, so there was nothing worth
preserving here to generalize against. JAX-ReaxFF fitting may be needed again for a future paper; if so,
pull the relevant logic back from the `main`/`Claude` branches (where it still exists) rather than
reconstructing it from memory, and give it its own driver script under the `FFMatEnsemble` pattern rather
than reviving `JaxReaxFFMatEnsemble`/`MatEnsembleJob.run()`.

## `CLAUDE.md` Architecture section staleness — resolved

`CLAUDE.md`'s Architecture section (and the Install section) have been rewritten to describe the current
`structures/`/`potential/`/`molecular_dynamics/`/`utilities/`/`base.py`/`in_queue.py` layout, dropping the
now-fixed "known broken imports" list. `Reformat.md` (the Stage 1-3 execution plan/decision-log this info
was originally drafted for) has been deleted as redundant now that its still-relevant content lives here,
in `TODO.md`, and in git history — it was a plan document, not something meant to be a permanent fixture.

## mace: PyPI release vs. git ref

`pyproject.toml`'s `mace` extra installs `mace-torch` from its PyPI release rather than a
`git+https://github.com/ACEsuit/mace.git` reference to `main`, per an explicit decision — this avoids
unexpected deviations between installs. Context: the earlier `mace-freeze` fork
(`github.com/7radians/mace-freeze`), which used to be necessary for the frozen-layer fine-tuning approach,
is no longer needed since that functionality has since been merged upstream into `ACEsuit/mace`. If a
future project needs bleeding-edge `main`-branch MACE features not yet in a PyPI release, revisit this
(a pinned `git+...@<tag-or-commit>` reference would restore reproducibility without going back to a bare
`main` reference).

## mpi4py and the official LAMMPS Python module — not declared, container-dependent

Neither `mpi4py` nor the official LAMMPS Python module (`import lammps`) is declared as a dependency
anywhere in `pyproject.toml`. Both need to link against whatever specific MPI implementation LAMMPS itself
was built against (Cray MPICH via the `cc` compiler wrapper on Perlmutter today, per `build_lammps.sh`'s
`MPICC="cc -shared" pip install --no-binary=mpi4py ...`); a generic PyPI wheel risks being
ABI-incompatible. `EnsembleFFFit`'s own code never imports `mpi4py` directly — the current understanding,
now confirmed by hands-on Frontier container experience, is that MatEnsemble's containerized deployment
builds its own LAMMPS with Python bindings already linked in, so this project doesn't need to install
LAMMPS (or the `cuequivariance`/`cupy` stack that used to back its ML-IAP path) via pip at all — see the
"lammps extra retired" entry below. Perlmutter's containerized build hasn't been done yet, so it's not
100% confirmed the same holds there. Revisit if that turns out to need something explicit.
`EnsembleFFFit/molecular_dynamics/helpers.py`'s `import_lammps()`/`import_lammps_mliap()` raise a
clear error if the module isn't available in the meantime.

## `lammps` extra retired

`pyproject.toml` used to have a `lammps` extra carrying `cuequivariance`/`cuequivariance-torch`/
`cuequivariance-ops-torch-cu12`/`cupy-cuda12x` for LAMMPS's ML-IAP/Kokkos MACE-acceleration path. Removed
entirely (not even kept as an empty placeholder) — confirmed both Frontier and (presumably) Perlmutter
containers ship LAMMPS with Python bindings already built in, so installing anything LAMMPS-related via
this project's pip is out of scope. `cuequivariance*` moved to the new `cuda` extra (still needed there for
MACE's own CUDA acceleration, independent of LAMMPS); `cupy-cuda12x` was dropped rather than moved, since
its only identified consumer was the now-out-of-scope LAMMPS ML-IAP path — add it back if another use
turns up.

## GPU platform extras (`cuda`/`rocm`) and `install_gpu_torch.py`

Added as part of the extras-based `pyproject.toml` redesign. Two things are still open:
- The `cuda` extra's `install_gpu_torch.py` mapping stays on `cu124` (`torch==2.6.0`) for now, matching
  what the rest of the project already assumed — but this hasn't been validated against an actual
  Perlmutter container build (none has been done yet). Revisit the pin once that happens; it may need to
  move to a newer index (`cu126`/`cu128`) depending on what that build actually needs.
- `openequivariance` (the `rocm` extra's MACE accelerator, ROCm's counterpart to `cuequivariance`)
  requires a working GCC 9+ and HIP toolchain *at pip-install time* to build its kernels — it's not a
  prebuilt wheel like `cuequivariance` is. If a container build fails installing this extra, check compiler
  availability first before assuming something else is wrong.

## `create_lammps_models_cli.py` needs the `mace` extra despite being a core console script

`EnsembleFFFit/potential/mace/create_lammps_models_cli.py` (moved from `utilities/`, since it's
MACE-specific; registered as the `create_lammps_models` console script, which is always installed
regardless of which extras a user chose) needs `torch`/`e3nn`/`mace` to
actually run the conversion. Its imports are now deferred into a guarded helper
(`_import_mace_conversion_deps()`) that raises a clear `pip install -e ".[mace]"` message instead of a raw
`ModuleNotFoundError` if those aren't installed — so the console script exists for everyone but only
*works* for users who've also installed the `mace` extra. This is a reasonable stopgap; if more
core-registered scripts turn out to need extras-only dependencies, it may be worth a more general pattern
(or moving such scripts to be registered only when their extra is installed, though `[project.scripts]`
doesn't support that natively).

## `build_*.sh` scripts and `constraints.txt` deleted — Frontier validated, Perlmutter still open

The five `build_*.sh` scripts and `constraints.txt` have been deleted from this branch (they're
superseded by the extras-based `pyproject.toml` install + `install_gpu_torch.py`, and are still recoverable
from the `main` branch / git history if needed). Frontier testing has since been carried out end-to-end
(the RMG -> MACE -> ASE pipeline in `examples/Frontier/RMG_MACE_ASE/` is the result), so the new install path is
confirmed working there. Perlmutter/CUDA still hasn't been tried at all (see the `cu124` pin note above).
If Perlmutter testing turns up something the old scripts handled that the new path doesn't (e.g.
`build_lammps.sh`'s Kokkos/cmake flags, module loads, or the `sed` hack for an nvcc flag CMake used to
generate incorrectly), recover the relevant script from `main` rather than reconstructing it from memory.

## Undeclared runtime dependencies in `pyproject.toml` — resolved

Previously, `numpy`, `matminer`, `scikit-learn`, `scipy`, `tqdm`, and `pyyaml` (imported as `yaml`) were
imported by always-present modules (`analysis/`, `utilities/`, `structures/`) but missing from
`pyproject.toml`'s `dependencies`. All six were added to core `dependencies` as part of the
branching-dependencies rewrite (confirmed via a script that walks every `.py` file's imports in those
directories, plus `potential/mace` and `molecular_dynamics/{ase,lammps,torchsim}`, and diffs against
the declared dependencies/extras). `torch`/`e3nn`/`mace` were also found in
`potential/mace/create_lammps_models_cli.py`, but per the entry above, those are handled via a guarded
import rather than added to core.

## `MatEnsembleJob.sorting_function`/`generic_task_command` duplication — resolved

Both deleted outright, along with `get_tasks`, `dict_to_argv`/`dict_to_str_list`, `to_str_list`,
`construct_tasks`, `read_structure_from_lammps`, `get_python`, and `modify_write_paths` — confirmed unused
by anything outside the `*_matensemble_cli.py` scripts (see the new entry below; all three have since been
deleted outright), so there was nothing left to dedup once those scripts' fate was settled.

## `build_full_runs`/`batch_by_parent` v1/v2 duplication — resolved

`build_full_runs`/`batch_by_parent` are now `@abstractmethod`s on `MatEnsembleJob`, each concrete subclass
providing its own implementation under that single name (no more `_v2` suffix): `FFMatEnsemble` (then
`MACEMatEnsemble`; `JaxReaxFFMatEnsemble` also had its own copy before being removed, see below) carries
the flat proximity-matched logic, `MDMatEnsemble` (renamed from `LammpsMatEnsemble`, and generalized to
cover ASE/TorchSim as well as LAMMPS) carries the recipe-file cross-product logic.

## `MatEnsembleJob.run()`'s `SuperFluxManager`/`poolexecutor` call — resolved

`run()`/`dry_run()` deleted outright rather than rebuilt against the current `Chore`/`FluxManager` API.
Each backend now exposes a `run_individual(task_dict)` static method that dynamically imports a
user-supplied driver script by path and dispatches to its named entry-point function, and a thin
`@pipe.chore`-decorated wrapper function in the submission script calls it — see
`examples/Frontier/RMG_MACE_ASE/run_pipeline.py` for the working pattern. (`FFMatEnsemble`'s
`run_individual` used to call `mace.cli.run_train.run` directly instead of dispatching to a driver
script — see the `FFMatEnsemble` entry below for why/how that changed.) This is what made the three
`*_matensemble_cli.py` scripts' fate need deciding — see the new entry below.

## Unused/deprecated console scripts deleted from this branch — resolved

`potential/mace/mace_matensemble_cli.py` and `molecular_dynamics/lammps/lammps_matensemble_cli.py`
(`jaxreaxff_matensemble_cli.py` was deleted earlier along with the rest of `potential/reaxff/`, see above)
both depended solely on `MatEnsembleJob.run()` for execution, which no longer exists (see the entry
above) — real replacements already exist for each (`lammps_matensemble_cli.py`'s useful bits are
superseded by `MDMatEnsemble.build_lists`/`run_individual`; `mace_matensemble_cli.py`'s by
`FFMatEnsemble.build_ff_dcts`/`run_individual`, exercised by `examples/Frontier/RMG_MACE_ASE/run_pipeline.py`),
so rather than leaving them in place broken, both were deleted outright.

Also deleted, all confirmed to have zero callers anywhere in this repo and no `pyproject.toml`
registration (verified via a repo-wide import search before deleting anything, same discipline as the
`pyMD/`/`potential/reaxff/` removals above): `structures/deviation_selection/` (`deviation_selection_cli.py`
-- its role in the current pipeline is filled by `analysis/variance.py`/`select_dft_candidates` instead),
`utilities/cn_checker_cli.py` (actually lived in `analysis/`), `utilities/parse_vasp_aimd_cli.py`,
`utilities/cluster_lammps_runs.py`, `utilities/formation_energy_lammps_runs.py`, and
`utilities/copy_by_pattern_cli.py` -- the latter's one genuinely reusable piece,
`get_atom_mapping_from_control` (LAMMPS dump-file element mapping), was merged into
`molecular_dynamics/lammps/lammps_properties.py` first (also moved there from `analysis/`, its only other
real consumer, since it's LAMMPS-specific) rather than lost. All of the above are still recoverable from
the `main`/`Claude` branches if any of it turns out to be needed as reference when incorporating another
backend later.

## Move RMG logic back into pyRMG — resolved

`EnsembleFFFit/density_functional_theory/rmg/`'s logic (calculator, input-file generation, processor-grid
sizing, log parsing, etc.) has been reconciled into `pyRMG` (on its own `updated_MatEnsemble` branch,
including two new modules, `rmg_calculator.py`/`pick_structure.py`, that didn't exist there before) and
`pyRMG` added as the new `rmg` extra here, the way MACE is handled via its own upstream package.
`base.py` and every `rmg_dft.py` driver copy import from `pyRMG` instead of a local copy. Confirmed working
end-to-end by actually running the `RMG_MACE_ASE` example pipeline's `converge_dft_data` stage against the
`pyRMG` import path before deleting anything, per plan.

`EnsembleFFFit/density_functional_theory/rmg/` now holds only `rmg_dft.py` (the package-level reference
copy of the driver script, analogous to `molecular_dynamics/ase/ase_mace.py`) — the RMG logic files
themselves (`rmg_calculator.py`, `rmg_input.py`, `processor_grid.py`, `rmg_log.py`, `valence.py`,
`convergence.py`, `forcefield.py`, `pick_structure.py`) have been deleted from this repo entirely, since
they're no longer used by anything here; still recoverable from `pyRMG`'s git history (they were reconciled
from these exact files) or this repo's own history if ever needed.

## `FFMatEnsemble` replaces `MACEMatEnsemble` — resolved

`MACEMatEnsemble` renamed to `FFMatEnsemble`; `build_mace_dcts` renamed to `build_ff_dcts` (logic
unchanged in both cases -- both were already backend-agnostic, since neither hardcodes anything
MACE-specific: the option keys they proximity-match on are entirely caller-supplied). The one genuinely
MACE-specific piece was `run_individual`, which used to build a `mace` argparse `Namespace` and call
`mace.cli.run_train.run(args)` directly, inline in `base.py` -- that logic moved essentially verbatim to
`examples/Frontier/RMG_MACE_ASE/FF/mace_fit.py`'s `run_mace_fit(overrides)` entry point.
`FFMatEnsemble.run_individual` is now a thin dispatcher identical in shape to
`DFTMatEnsemble`/`MDMatEnsemble`'s: import the driver script named by `task_dict['ff_task']`, call the
function named by `task_dict['entry_point']`. Porting to a different FF backend later means writing a new
driver script with the same entry-point contract and pointing `ff_task` at it -- no `base.py` changes.

Along with this, all three `MatEnsembleJob` subclasses were standardized on `MDMatEnsemble`'s convention
for how a chore's driver-script path reaches `run_individual`: the `build_*_dcts`/`build_task_dicts` method
takes the task-script path and entry-point name as explicit parameters and embeds them into every returned
dict itself, rather than leaving the submission script to inject them afterward.
`DFTMatEnsemble.build_dft_dcts` was retrofitted from `(self, check_files, finished_file=None)` to
`(self, dft_task, check_files, entry_point, finished_file=None)` to match (the *content* of each built
dict is unchanged either way -- this only moves where `dft_task`/`entry_point` get set, `run_individual`
itself wasn't touched); `run_converge_dft_data` in `run_pipeline.py` lost its manual post-build injection
loop accordingly. `FFMatEnsemble.build_ff_dcts` was designed with this shape from the start.

## `molecular_dynamics/pyMD/` reorganized into `{ase,lammps,torchsim}/` — resolved

`molecular_dynamics/` now mirrors `potential/`/`density_functional_theory/`'s per-backend-folder
convention. The intermediate `pyMD/` folder (a LAMMPS-flavored name nesting even the non-LAMMPS backends)
is gone entirely; `helpers.py` moved to `molecular_dynamics/helpers.py` (shared across backends), and each
driver moved into its backend's own folder: `ase/ase_mace.py`, `lammps/lammps_reaxff_cpu.py`,
`lammps/lammps_mace_kokkos_gpu.py`, `torchsim/torch_sim_mace.py` (`lammps/lammps_matensemble_cli.py` also
moved here at the time, but has since been deleted outright, see the entry above). `pyMD/examples/` (worked
ReaxFF/LAMMPS, MACE/LAMMPS-Kokkos,
and MACE/ASE example datasets — POSCARs, LAMMPS data files, submit scripts, two model checkpoints) was
deleted outright rather than migrated — confirmed orphaned (nothing referenced it) and fully superseded by
`examples/Frontier/RMG_MACE_ASE/`; recoverable from git history if ever needed. Confirmed via a full
repo-wide import search before moving/deleting anything that nothing outside `pyMD/` itself ever
referenced it.
