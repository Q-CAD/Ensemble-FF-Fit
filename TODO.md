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

## JAX-ReaxFF `run_reaxff` missing `make_paths_list`

`EnsembleFFFit/potential/reaxff/jaxreaxff_matensemble_cli.py`'s `run_reaxff` calls
`JaxReaxFFMatEnsemble.run(...)` without the required `make_paths_list` argument, so it raises `TypeError`
on every invocation. Not fixed: JAX-ReaxFF's CUDA/jaxlib requirements conflict with MACE's in a shared
environment (this is exactly why `reaxff` and `mace` are separate `pyproject.toml` extras), and the
upstream optimizer is deprecated by its original developers — it may be removed entirely in a future
release rather than patched in place. Decide whether to fix or remove JAX-ReaxFF support before this path
is needed again.

## LAMMPS batching per-batch task count

`EnsembleFFFit/molecular_dynamics/lammps/lammps_matensemble_cli.py`'s `run_lammps` sizes each batch's task
count using only the first structure in that batch. Acceptable today because batching is typically used
for single-point runs where 1 GPU suffices regardless of atom count, but would undercount if batches ever
mix structures of meaningfully different sizes. Revisit if that usage pattern changes.

## `deviation_selection_cli.py` known limitations (left unfixed)

- `get_structures_scores`: keys per-image data by `image_key` alone across *all* MD trajectories — if two
  different trajectories reuse the same image index, their energies/forces will be silently merged.
- `rank_structures`: the `while True` loop stops at the first `IndexError` from `parse_single_points`,
  which fires as soon as **any one** trajectory in the tree is exhausted, not per-trajectory — frames from
  longer trajectories past that point are silently dropped.

Both were explicitly left as-is (not fixed) per a repo-author decision — this script was written as an
exploratory active-learning/UQ tool and may need a more thorough rework rather than a targeted patch.

## `get_reference_per_atom` single-force-field assumption

`EnsembleFFFit/utilities/cluster_lammps_runs.py` and `EnsembleFFFit/utilities/formation_energy_lammps_runs.py`
both have a `get_reference_per_atom` that assumes a single force field is being processed at a time (e.g.
one MACE variant); the dict structure supports multiple force-field keys, but per-element reference
energies are overwritten (not accumulated as a running minimum) across force fields if more than one is
ever passed in. Left as-is per the repo author's confirmation this was written for single-ffield use.
Related: both files' `parse_single_points`-adjacent code has hardcoded `ffield_labels`/`dump_index`
assumptions that would need generalizing alongside any multi-ffield fix.

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

`EnsembleFFFit/utilities/create_lammps_models_cli.py` (registered as the `create_lammps_models` console
script, which is always installed regardless of which extras a user chose) needs `torch`/`e3nn`/`mace` to
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
directories, plus `potential/reaxff`, `potential/mace`, and `molecular_dynamics/{ase,lammps,torchsim}`, and diffs against
the declared dependencies/extras). `torch`/`e3nn`/`mace` were also found in
`utilities/create_lammps_models_cli.py`, but per the entry above, those are handled via a guarded import
rather than added to core.

## `MatEnsembleJob.sorting_function`/`generic_task_command` duplication — resolved

Both deleted outright, along with `get_tasks`, `dict_to_argv`/`dict_to_str_list`, `to_str_list`,
`construct_tasks`, `read_structure_from_lammps`, `get_python`, and `modify_write_paths` — confirmed unused
by anything outside the three now-deprecated `*_matensemble_cli.py` scripts (see the new entry below), so
there was nothing left to dedup once those scripts' fate was settled.

## `build_full_runs`/`batch_by_parent` v1/v2 duplication — resolved

`build_full_runs`/`batch_by_parent` are now `@abstractmethod`s on `MatEnsembleJob`, each concrete subclass
providing its own implementation under that single name (no more `_v2` suffix): `MACEMatEnsemble` and
`JaxReaxFFMatEnsemble` each carry their own copy of the old non-`_v2` (flat proximity-matched) logic,
`MDMatEnsemble` (renamed from `LammpsMatEnsemble`, and generalized to cover ASE/TorchSim as well as LAMMPS)
carries the old `_v2` (recipe-file cross-product) logic. `JaxReaxFFMatEnsemble` got a direct copy rather
than a shared intermediate base class, given its likely eventual deprecation (see the entry below) — not
worth a permanent shared-base fixture for two classes where one is expected to go away.

## `MatEnsembleJob.run()`'s `SuperFluxManager`/`poolexecutor` call — resolved

`run()`/`dry_run()` deleted outright rather than rebuilt against the current `Chore`/`FluxManager` API.
Each backend now exposes a `run_individual(overrides)` static method (MACE: calls `mace.cli.run_train.run`
directly; MD: dynamically imports the user-supplied driver script by path and dispatches to its named
entry-point function) that a thin `@pipe.chore`-decorated wrapper function in the submission script calls —
see `examples/Frontier/RMG_MACE_ASE/run_pipeline.py` for the working pattern. This
is what made the three `*_matensemble_cli.py` scripts' fate need deciding — see the new entry below.

## Three `*_matensemble_cli.py` console scripts — deprecated, registrations removed, files not yet deleted

`potential/mace/mace_matensemble_cli.py`, `potential/reaxff/jaxreaxff_matensemble_cli.py`, and
`molecular_dynamics/lammps/lammps_matensemble_cli.py` all depended solely on `MatEnsembleJob.run()` for
execution, which no longer exists (see the entry above) — the intended replacement is the `Pipeline`/chore
pattern now exercised by `examples/Frontier/RMG_MACE_ASE/run_pipeline.py`. Their `pyproject.toml`
console-script registrations (`mace_matensemble`/`jaxreaxff_matensemble`/`lammps_matensemble`) have been
removed; the files themselves are left in place, broken, pending deletion once real replacements exist for
each: `lammps_matensemble_cli.py`'s useful bits are already superseded by
`MDMatEnsemble.build_lists`/`run_individual`; `mace_matensemble_cli.py`'s by
`MACEMatEnsemble.build_mace_dcts`/`run_individual`. `jaxreaxff_matensemble_cli.py` has not been exercised or
tested at all this pass (JAX-ReaxFF was out of scope) — it's a deprecation candidate given JAX-ReaxFF's
upstream-deprecated status, but there's a published paper tied to this workflow, and a dedicated Perlmutter
container for a JAX-ReaxFF fitting workflow may be built before this script is retired. Evaluate in that
context before deleting it outright.

`cn_checker`/`copy_by_pattern`'s console-script registrations have also been removed for the same reason
(unexercised by the current pipeline) — `cn_checker_cli.py`/`copy_by_pattern_cli.py` themselves are left in
place, unregistered, not deleted.

## Move RMG logic back into pyRMG (future work, not current)

`EnsembleFFFit/density_functional_theory/rmg/` currently carries RMG-specific logic directly in this
package (calculator, input-file generation, processor-grid sizing, log parsing, etc.). Longer-term, this
should move back into `pyRMG` as a proper standalone dependency, the way MACE/JAX-ReaxFF are handled via
their own upstream packages, rather than living in-tree here. Not being done now — flagged for future
tracking only.

## Create `FFMatEnsemble` to replace `MACEMatEnsemble` (future work, not current)

`MACEMatEnsemble` is currently MACE-specific, but most of its logic (fanning a run-path into per-seed
fitting subdirectories, matching foundation models against training-input folders, dispatching fits) isn't
inherently MACE-only. Create a generic `FFMatEnsemble` — structured like `DFTMatEnsemble`/`MDMatEnsemble`,
i.e. shared/conserved logic with a thin per-backend layer — so the same force-field-fitting caller can wrap
other MD codes' fitting workflows, not just MACE's. Not being done now — flagged for future tracking only.

## `molecular_dynamics/pyMD/` reorganized into `{ase,lammps,torchsim}/` — mostly resolved

Done: `molecular_dynamics/` now mirrors `potential/`/`density_functional_theory/`'s per-backend-folder
convention. The intermediate `pyMD/` folder (a LAMMPS-flavored name nesting even the non-LAMMPS backends)
is gone; `helpers.py` moved to `molecular_dynamics/helpers.py` (shared across backends), and each driver
moved into its backend's own folder: `ase/ase_mace.py`, `lammps/lammps_reaxff_cpu.py`,
`lammps/lammps_mace_kokkos_gpu.py`, `lammps/lammps_matensemble_cli.py` (still deprecated, see the entry
above — moving it didn't fix its already-broken `LammpsMatEnsemble` import, since that class was renamed to
`MDMatEnsemble`), `torchsim/torch_sim_mace.py`. Confirmed via a full repo-wide import search before moving
anything that nothing outside `pyMD/` itself ever referenced it.

Still open: `pyMD/examples/` (worked ReaxFF/LAMMPS, MACE/LAMMPS-Kokkos, and MACE/ASE example datasets —
POSCARs, LAMMPS data files, submit scripts, and two model checkpoints) wasn't moved or deleted yet — same
orphaned status as the drivers were (nothing currently references it), but it's real example content, not
just stale script copies, so it needs an explicit decide-and-migrate-or-drop pass rather than a mechanical
move.
