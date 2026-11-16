# MACE -> ACE -> LAMMPS: distilling a MACE teacher into an ACE ensemble on Perlmutter

This example distills an already-fine-tuned MACE foundation model (the "teacher") for
the Pd-Se system into a faster, more MD-friendly ACE potential (the "student"), fully
coordinated by a single script/config pair (`run_pipeline.py` / `workflow_config.yaml`)
via MatEnsemble/Flux:

1. **Teacher preparation**: convert your own fine-tuned MACE model to LAMMPS's MLIAP
   format once, up front. Fine-tuning the teacher itself is a separate, prior step this
   example does not repeat.
2. **Training-data generation**, 4 tiers, each MACE-labeled with the teacher: tier 1
   (broad, random pyxtal crystals across a composition sweep) -> downselect to the
   lowest-formation-energy structures per composition -> tier 2 (re-equilibrate those) ->
   tier 3 (broad, deliberately-unphysical defect supercells of every validation phase
   found) -> tier 4 (phase-specific defect + motif/volume-excursion structures for your
   actual target phases). Every tier is MD'd with the teacher, then single-pointed and
   metadata-tagged (stoichiometry, trajectory id, phase).
3. **ACE ensemble fitting** (`pyace`/`pacemaker`, via `TensorPotential`'s GPU/TensorFlow
   evaluator): one or more *named training-subset combinations* — e.g. "broad tier-1
   data" vs. "phase-specific tier-4 data" — each fit as its own small ensemble (seeds x
   a kappa grid), so you can compare how training-data composition itself affects the
   resulting ACE potentials, not just how fitting hyperparameters do.
4. **Static ACE validation + selection**: every fitted ensemble member is single-pointed
   (LAMMPS, `pair_style pace`) against real Pd-Se compounds (Materials Project) and
   against a training-set sanity-check sample, ranked by combined energy/force
   deviation, and the best member per combo is selected.
5. **MD structural-stability validation**: short, real room-temperature NPT runs
   (Kokkos-GPU LAMMPS) of the selected ACE potentials on each target phase, checked for
   RDF/z-density/MSD stability against both the pristine reference structure and the
   same protocol run with the original teacher MACE model — a force field that looks
   good statically but doesn't survive real dynamics is caught here, independent of its
   static ranking.

Commands/paths below are written as concrete examples from a real run; substitute your
own project/user/scratch paths throughout.

## 0. What's included vs. what you provide

This directory is a worked recipe, not a pre-run investigation — every generated
artifact (fitted/converted models, generated structures, run output, fitted ACE
potentials) is excluded (see `.gitignore`); you provide the one real input this example
can't ship, and regenerate everything else by actually running the stages below:

- **Your own fine-tuned MACE model.** This example does **not** perform MACE
  fine-tuning itself — `teacher_preparation.source_model` (and
  `stage_mace_reference_force_field.source_model`, the same file) must already point at
  a model you've fine-tuned elsewhere (e.g. the companion `VASP_MACE_ASE`/Frontier
  `RMG_MACE_ASE` active-learning examples in this repo).
- **A Materials Project API key**, read only from the `MP_API_KEY` environment
  variable — never from a committed file, and none is shipped with this example (see
  Prerequisites below).
- Everything else — the 4-tier structure-generation driver scripts, the ACE
  subset-combination logic (`FF/subset_combination.py`), the LAMMPS recipes, and the
  structural-stability checker — is real, runnable code, ready to point at your own
  model and MP key.

## 1. Prerequisites

- **A fine-tuned MACE model** (see item 0 above) — no default is provided deliberately;
  this must be your own file.
- **A Materials Project API key**, for `generate_validation_structures` (pulls real
  Pd-Se compounds to validate against). Get a free key at
  <https://materialsproject.org/api>, then:
  ```bash
  export MP_API_KEY=<your key>
  ```
  before running that stage. This key is read **only** from the `MP_API_KEY`
  environment variable — it is never read from a committed file, and no key is shipped
  with this example. Do not commit your own key anywhere in this repository.
- **Pd2Se3 and Pd3Se10 are NOT in the Materials Project** — CONFIRMED directly
  (this project's own `chemsys=[Pd-Se]` query, run for real against the live MP API):
  it returns 9 phases, never these 2. Both are real, experimentally-known compounds,
  just without a standard computed MP entry under this chemsys search. Every stage
  downstream of `generate_validation_structures` that references all 3 target phases
  (`generate_structures_tier4_defect`/`_motif`, `build_structural_stability_inputs`,
  `stage_mace_reference_force_field`'s validation set, etc.) expects a POSCAR for both
  of them at `MACE/structures/validation/Pd-Se/Pd2Se3/experimental/POSCAR` and
  `.../Pd3Se10/experimental/POSCAR` — you need to supply these two yourself (e.g. from
  your own DFT-relaxed structures or published CIFs) after
  `generate_validation_structures` runs; nothing in this example's own pipeline can
  produce them. Also run `relax_structures`/`run_md_tier1` (both already cover
  `MACE/structures/validation` in-place, not just the tier-1 training set) on them
  before `generate_structures_tier3_defect`/`tier4_defect`/`tier4_motif` — those
  generators require a `md_run.traj` next to each phase's POSCAR, confirming it's
  already relaxed/equilibrated, not the raw hand-supplied structure.
- **No separate TensorPotential checkout needed.** ACE fitting here uses
  `evaluator: tensorpot`'s GPU/TensorFlow backend, which `Dockerfile.ace` pip-installs
  directly from [Q-CAD/TensorPotential](https://github.com/Q-CAD/TensorPotential)
  (tag `v0.1.1-qcad1`) — a fork of upstream `ICAMS/TensorPotential` with a numpy-2.x
  compatibility fix and a relaxed `setup.py` (upstream's `python_requires='<3.11'`
  genuinely refuses to install on this image's Python 3.12 — see Pitfalls below).
  Licensed under the same Academic Software Licence (ASL) as upstream — non-commercial
  use, see that repo's own `LICENSE.md`.

## 2. Container build

```bash
podman-hpc build -f Dockerfile.ace -t matensemble:mace-ace-lammps .
podman-hpc migrate matensemble:mace-ace-lammps
```

`Dockerfile.ace` is adapted from the same NERSC Flux + MatEnsemble base this repo's
other Perlmutter examples use, plus CUDA torch + `mace-torch`/`cuequivariance` (the
teacher format) and `pyace` (the student format). A few choices worth knowing before
you rebuild it for yourself:

- **LAMMPS itself is not built here.** The base image (`nersc/flux:26.05`) already
  ships a working LAMMPS build with `ML-PACE` (ACE) and Kokkos-GPU compiled in —
  confirmed by direct inspection, no LAMMPS-related install step exists in any of this
  project's own Dockerfiles. Verify this holds for your own base image before relying
  on it, the way `VASP_ReaxFF_LAMMPs/README.md` step 1 verifies its own LAMMPS build.
- **`pyace` is pinned, not tracking upstream's default branch**, to commit
  `66c35ea6e5dbed2857d423952b922183206c140e` (`pyace 0.2.7+192.g66c35ea`) — confirmed,
  by direct inspection of a real running container, to be the exact revision every fit
  in this project has been validated against, including the numpy-2.x-compatibility fix
  in `TensorPotential` (see Pitfalls below). `python-ace` has no PyPI release and no
  API-stability guarantee between commits, so re-pin deliberately (with a fresh
  validation pass) rather than bumping opportunistically.
- **`ovito==3.15.0` is pinned explicitly**, needed only by this example's own
  `MD/structural_stability/structural_stability_checker.py` (RDF long-range-peak
  retention). It is deliberately installed **here**, in this example's own Dockerfile,
  not added to Ensemble-FF-Fit's own base dependencies — EFF core supports multiple FF
  backends and shouldn't carry a dependency only one example's own structural check
  needs (an explicit decision for this project).
- **TensorPotential is pip-installed fixed into the image**, from Q-CAD's fork, tag
  `v0.1.1-qcad1` (`pip install --no-deps "tensorpotential @ git+https://github.com/
  Q-CAD/TensorPotential.git@v0.1.1-qcad1"`). Unlike `pyace`, there's no reason to swap
  in a locally patched copy here — this pipeline uses it as a plain library — so baking
  it in avoids the env-var/bind-mount indirection a separate checkout would need.
  `--no-deps` is deliberate: its `install_requires` (scipy/tensorflow/numpy/pandas/ase)
  are already satisfied above by this image's own pinned versions.
- **Ensemble-FF-Fit itself is still not installed into the image.**
  `launch_multi_node.slurm` bind-mounts the checkout and puts it on `PYTHONPATH` at
  invocation time — only its dependencies (pymatgen, mp-api, ase, matminer,
  scikit-learn, ...) are baked into the image, so you can iterate on EnsembleFFFit
  itself without a rebuild.
- **pyxtal** (tier-1 structure generation) is installed from its GitHub source, same as
  `VASP_MACE_ASE`/`VASP_ReaxFF_LAMMPs`.

## 3. Why `launch_multi_node.slurm` specifically

Same multi-node-Flux-rendezvous problem, and same fix, as the other two Perlmutter
examples in this repo (`VASP_ReaxFF_LAMMPs/README.md` step 2 has the full explanation —
`--network=host`, explicit `SLURM_*`/`PALS_*`/`PMI_*` passthrough, PMI2 socket mounts,
Slingshot/xpmem device-node mounts, and a static `R.json` resource description instead
of live NVML/hwloc discovery). This copy is trimmed to drop every VASP-specific host
mount (no VASP stage exists in this pipeline) and adds one env var this example
specifically needs:

```bash
ENSEMBLE_FF_FIT="${ENSEMBLE_FF_FIT:-/path/to/your/Ensemble-FF-Fit}"
```

bind-mounted into the container and put on `PYTHONPATH`, ahead of a separate, complete
`flux` python install (`/usr/local/lib/python3.12/site-packages`) that has to come
*first* on `PYTHONPATH` — the image's default python only sees an incomplete, vendored
copy of the flux bindings (missing `_flux/_rlist.abi3.so`, which
`matensemble.manager.FluxManager._get_allocation_info` needs via `flux.resource.list`).
TensorPotential needs no entry here — it's pip-installed into the image itself (see
Container Build above).

`PYTHONPATH` is scoped to the one `env PYTHONPATH=... python run_pipeline.py ...`
command, not set at the container level, because setting it globally breaks `flux
start`'s own `rc1` bootstrap (confirmed in this project's own development:
`ModuleNotFoundError: No module named '_cffi_backend'` importing `_flux._core`) —
whatever's sensitive to a global `PYTHONPATH` is specific to how the broker process
sees its own environment during `rc1`, not python's module resolution in general.

Run it from inside this example's directory (`workflow_config.yaml`'s own keys are
relative to CWD, not to the script's location):

```bash
ENSEMBLE_FF_FIT=/path/to/your/Ensemble-FF-Fit \
CONFIG=workflow_config.yaml STAGE=all \
  sbatch launch_multi_node.slurm
```

`CONFIG`/`STAGE` default to a full one-loop run over `workflow_config.yaml`; both
default to `all`/`workflow_config.yaml` and can be overridden as shown, or by running
`bash launch_multi_node.slurm` directly from inside an already-active interactive
allocation (the `#SBATCH` lines are inert outside of `sbatch`).

## 4. The `scale` mechanism

A single top-level config key, `scale: smoke_test` (default) or `scale: full`, read by
each of the 4 training-data-generation stages' own `generate_structures` handler to
pick between that generator's own `generator_args_smoke_test`/`generator_args_full`
sub-block (a plain `generator_args` key, with no `_smoke_test`/`_full` suffix, is
ignored by this mechanism entirely). It controls **how many structures each generator
produces**, nothing else — it does **not** affect MD step counts: `nsteps`/`frequency`
in the `in.*.json` MD recipe files are already small enough for a quick smoke run, and
need to be bumped by hand (each recipe file says so in its own comment) for a genuine
full-scale run. Smoke-test defaults run the full pipeline end-to-end on a single node in
well under an hour; `full`'s own values are this project's actual production sweep.

## 5. Run the pipeline, stage by stage

Each stage is its own compute allocation/container invocation — run them in order (or
all in one process via `STAGE=all`, if one allocation covers every stage's needs):

```bash
python run_pipeline.py --config workflow_config.yaml --stage teacher_preparation
python run_pipeline.py --config workflow_config.yaml --stage generate_structures_tier1
python run_pipeline.py --config workflow_config.yaml --stage generate_validation_structures
python run_pipeline.py --config workflow_config.yaml --stage single_points_validation
python run_pipeline.py --config workflow_config.yaml --stage generate_isolated_elements
python run_pipeline.py --config workflow_config.yaml --stage single_points_isolated_elements
python run_pipeline.py --config workflow_config.yaml --stage relax_structures
python run_pipeline.py --config workflow_config.yaml --stage run_md_tier1
python run_pipeline.py --config workflow_config.yaml --stage run_single_points_tier1
python run_pipeline.py --config workflow_config.yaml --stage downselect_structures_tier2
python run_pipeline.py --config workflow_config.yaml --stage run_md_tier2
python run_pipeline.py --config workflow_config.yaml --stage run_single_points_tier2
python run_pipeline.py --config workflow_config.yaml --stage generate_structures_tier3_defect
python run_pipeline.py --config workflow_config.yaml --stage run_md_tier3
python run_pipeline.py --config workflow_config.yaml --stage run_single_points_tier3
python run_pipeline.py --config workflow_config.yaml --stage generate_structures_tier4_defect
python run_pipeline.py --config workflow_config.yaml --stage generate_structures_tier4_motif
python run_pipeline.py --config workflow_config.yaml --stage run_md_tier4_motif_300k
python run_pipeline.py --config workflow_config.yaml --stage run_md_tier4_motif_500k
python run_pipeline.py --config workflow_config.yaml --stage run_md_tier4_motif_700k
python run_pipeline.py --config workflow_config.yaml --stage run_md_tier4_motif_900k
python run_pipeline.py --config workflow_config.yaml --stage run_md_tier4_defect
python run_pipeline.py --config workflow_config.yaml --stage run_single_points_tier4
python run_pipeline.py --config workflow_config.yaml --stage build_ff_inputs
python run_pipeline.py --config workflow_config.yaml --stage fit_and_validate
python run_pipeline.py --config workflow_config.yaml --stage prepare_ace_validation_structures
python run_pipeline.py --config workflow_config.yaml --stage stage_ace_validation_force_fields
python run_pipeline.py --config workflow_config.yaml --stage ace_validation_single_points
python run_pipeline.py --config workflow_config.yaml --stage rank_ace_validation
python run_pipeline.py --config workflow_config.yaml --stage select_ace_validation_force_field
python run_pipeline.py --config workflow_config.yaml --stage build_structural_stability_inputs
python run_pipeline.py --config workflow_config.yaml --stage stage_structural_stability_force_fields
python run_pipeline.py --config workflow_config.yaml --stage structural_stability_screening
python run_pipeline.py --config workflow_config.yaml --stage check_structural_stability
python run_pipeline.py --config workflow_config.yaml --stage stage_mace_reference_force_field
python run_pipeline.py --config workflow_config.yaml --stage mace_structural_stability_screening
```

### 5.1 Teacher preparation

`teacher_preparation` converts your fine-tuned `source_model` to LAMMPS's MLIAP format
once (`dest_root: MACE/teacher/force_fields`, `target_name: model-mliap_lammps.pt`).
Every downstream MACE-labeling stage, and `stage_mace_reference_force_field` (section
5.5), reuses this same converted file.

### 5.2 Training data generation (tiers 1-4)

All 4 generator stages (`generate_structures_tier1`, `generate_structures_tier3_defect`,
`generate_structures_tier4_defect`, `generate_structures_tier4_motif`) dispatch through
one generic `generate_structures` function in `run_pipeline.py` — a new generator is a
config change (`generation_task`/`generation_entry_point`/`generator_args_*`), never a
`run_pipeline.py` change. `generate_validation_structures` (real Pd-Se compounds from
Materials Project, `MP_API_KEY`-gated) is a separate stage rather than a fifth
"generator" variant, since an MP query isn't a swappable structure-generator function
the same way the other four are.

Each tier follows the same MD-then-single-point pattern: `run_md_tier<N>*` runs
finite-temperature MD with the teacher MACE model (minimize -> 300K equilibrate ->
ramped melt for tiers 1/2/3; 4 independent fixed-temperature runs, 300/500/700/900K, for
tier 4's motif axis — one stage per temperature rather than one stage with a
temperature list, since `MDMatEnsemble`'s own dispatch takes exactly one `in_file` per
batch and each temperature needs its own `output_root` so reruns never clobber each
other). `run_single_points_tier<N>` then single-points every extracted MD frame with the
teacher, writing a `metadata.json` sidecar per structure (`tier`, `stoichiometry`,
`trajectory_id`, and `phase` — derived by matching each structure's own path against a
configured `phase_candidates` list, since nesting depth and phase-label strings vary by
tier; see `ase_mace.py`'s own `_path_derived_metadata` docstring — it **raises** rather
than silently guessing if nothing matches).

Between tier 1 and tier 2, `downselect_structures_tier2` runs a generalized
`downselect_structures` stage (not tier-2-specific — any already-labeled MD/single-point
output can be screened the same way): `group_by: [trajectory_id, stoichiometry]`
dedups to one lowest-energy structure per trajectory, then keeps the top
`n_per_group` by `energy_per_atom` per composition, then isotropically rescales the
survivors (`rescale_output.volume_fractions`) before tier 2's own re-equilibration MD.

Tier 3 and tier 4's defect axis share one generator
(`MACE/structures/generate_defect_structures.py`) but differ in scope on purpose: tier 3
omits `phase_labels` entirely, so it defect-substitutes *every* real phase found under
`validation_root` (broad coverage), while tier 4 restricts `phase_labels` to this
example's actual 3 target phases (`PdSe`, `Pd2Se3`, `Pd3Se10` by default). Because tier
3's phase set is therefore open-ended (whatever your own Materials Project query
returns), `in.single_points_tier3.json`'s own `phase_candidates` list has to be kept in
sync by hand if you point this example at a different `validation_root` or chemical
system (see Known Limitations below).

### 5.3 ACE ensemble fitting

`build_ff_inputs` fits one or more *named training-subset combinations*
(`training_subsets_by_combo`, a list), each its own small ensemble over a seed x kappa
grid — the smoke-test config ships two deliberately contrasting combos, `tier1_broad`
(trains only on tier 1's broad data) and `tier4_specific` (trains only on tier 4's
phase-specific data), so you can directly compare how training-data composition affects
the resulting ACE potentials. Each combo's `subsets` list (`{tier: ..., phases:
[...]}`) is resolved to real training directories by `FF/subset_combination.py`'s
`resolve_subsets`/`resolve_subset`, via `Path.rglob(f"Pd-Se/{phase}")`-style glob
matching rather than reading per-structure metadata — chosen because tier 3/tier 4 nest
phase labels at different depths, and a single phase can appear under multiple tier-4
subtrees (the defect axis plus 4 motif-temperature axes) at once. For each combo,
`build_ff_inputs` calls EFF-core's existing `write_ace_dataset`/
`build_ace_ensemble_inputs` unchanged, once per combo, writing a nested
`FF/generation_1/<combo_label>/<member>/` tree — this whole mechanism (`FF/
subset_combination.py` plus the per-combo loop in `build_ff_inputs`) is new,
pipeline-specific logic, not an EFF-core change.

`fit_and_validate` then submits one ACE-fitting chore per `(combo_label, member)`
folder, dispatching to `FF/ace_fit.py`'s `run_ace_fit` (ported as-is from the companion
Pd-Se pipeline — the nested combo/member directory structure is invisible to it, since
it only ever sees one `work_dir`/`results_dir` per chore). See Pitfalls below for two
real gotchas in that script worth knowing before you touch it.

### 5.4 Static ACE validation + selection

`prepare_ace_validation_structures` exports, once, a phase-labeled LAMMPS-format
reference tree: the real validation-phase structures, plus a training-set
overfitting-sanity-check sample built from the **union** of every combo's own resolved
training subsets (not just one combo's) — since different combos genuinely train on
different data, a single combo's training set wouldn't be a meaningful sanity check for
every fitted potential. `stage_ace_validation_force_fields` then stages every combo's
fitted potentials into both a `validation/force_fields` and `training/force_fields`
tree, labeled `<combo_label>_<member>/ace.yace` (underscore-joined, to avoid directory
collisions across combos in this shared comparison tree — a genuine generalization
built for this example, since the single-source version this was ported from only ever
staged one combo at a time). `ace_validation_single_points` runs one LAMMPS single-point
chore per staged potential x structure.

`rank_ace_validation` calls `EnsembleFFFit.analysis.best_force_field.
rank_force_fields_combined` directly — an EFF-core function extended this session with
`force_aggregation: atom_component_pooled` (the default here), which pools every force
component from every atom of every structure into one flat array before computing a
single RMSE. This matches `pyace`/`TensorPotential`'s own fitting-time
`test_metrics.txt` convention; the function's original default,
`structure_mean` (one RMSE per structure, then averaged), differs from that convention
by roughly 25-35% in practice — a real, previously-confirmed discrepancy, not a
theoretical one. Use `structure_mean` only if you specifically want this ranking's
force columns to match a *different* backend's own reporting convention instead.

`select_ace_validation_force_field` picks the top `number_to_select` member(s) by
combined score via EFF-core's `downselect_force_fields.select_and_copy`. This needed one
piece of reconciliation: `ranking.csv` uses the staged tree's labels
(`<combo_label>_<member>`, underscore-joined), but `select_and_copy` needs the original
nested fitted-potential tree's own `<combo_label>/<member>` path (real separator) to
locate the actual `FF_*.yaml` file — resolved by `_destage_label`, which tries the
longest known combo label first to avoid prefix ambiguity (e.g. `tier1` vs.
`tier1_broad`). `select_and_copy` itself was also generalized this session from
exact-filename matching to `glob.glob`-based matching, so it can resolve a
per-member-varying filename like `FF_1.yaml` — purely additive, since a literal
filename is just a no-wildcard glob matching only itself.

### 5.5 MD structural-stability validation

Short (by default: 5,000-step equilibration + 20,000-step production, both
**placeholders** — re-tune once you have a real per-step Kokkos/ACE GPU timing number
for your own ~3,000-atom cells) room-temperature (300K) NPT runs, via Kokkos-GPU LAMMPS,
of the selected ACE potential(s) on each target phase. `build_structural_stability_inputs`
builds supercells (`min_atoms`/`min_length`-gated) from this example's own validation
structures; `stage_structural_stability_force_fields` stages the specific fitted
members you choose (the smoke-test config's two entries, member `"0"` of each combo,
are illustrative placeholders — update them to match your own `rank_ace_validation`
results, not a real selection); `structural_stability_screening` runs the MD itself
(`pair_style pace`); `check_structural_stability` then compares each run's RDF
long-range-peak retention, z-density-profile modulation, and MSD diffusive-slope against
both the pristine reference structure and a threshold recalibrated for this project's
own PdSe/Pd2Se3/Pd3Se10 MSD-slope distribution (`diffusive_slope_threshold`/
`diffusive_r_squared_threshold` — not universal defaults, re-tune for your own system).

`stage_mace_reference_force_field` reuses `teacher_preparation`'s already-converted
MLIAP model (same `source_model` path) rather than re-running the MACE-to-LAMMPS
conversion, and `mace_structural_stability_screening` repeats the identical protocol
with `pair_style mliap unified` instead of `pace` — so `check_structural_stability`'s
same metrics can be compared directly between the distilled ACE potential and the
original MACE teacher on equal footing.

This whole structural-stability mechanism (`MD/structural_stability/
structural_stability_checker.py` plus the two LAMMPS driver scripts) is ported as-is
from the companion Pd-Se pipeline as genuinely pipeline-specific code, **not** added to
Ensemble-FF-Fit core — a deliberate choice (see item 0 and Container Build above re:
`ovito`). It's a deliberate, *not*-unified parallel to `VASP_ReaxFF_LAMMPs`'s own
different structural check (`cn_checker.py`, coordination-number deviation via
pymatgen's `CrystalNN`): same two-stage design (an independent dynamic-stability gate on
top of the static ranking), different physics, because the two example pipelines fit
genuinely different force-field formats.

## 6. Known limitations / deferred items

- **No per-training-subset weighting.** Every structure drawn from a combo's resolved
  `subsets` is weighted identically in that combo's dataset — there's no mechanism yet
  to, say, down-weight tier 3's deliberately-unphysical defect concentrations relative
  to tier 4's phase-specific data within the same combo.
- **No `warm_start_from` auto-resolution across generations.** `warm_start_potential`
  must be a literal path you set by hand; there's no equivalent yet of the companion
  pipeline's generation-chaining convenience for automatically picking up the previous
  generation's selected potential.
- **MD-staging member selection is manual.** `stage_structural_stability_force_fields.
  members` is a hand-maintained list of `{source, label}` entries — there's no stage
  that reads `rank_ace_validation`'s own ranking.csv and auto-selects the top member per
  combo for MD staging; you re-derive and update that list yourself after each real
  validation run.
- **Tier-3's `phase_candidates` list needs manual sync.** Because tier 3 deliberately
  covers *every* phase found under `validation_root` rather than a fixed set,
  `in.single_points_tier3.json`'s own `phase_candidates` list (currently the 9 phases
  this project's own Materials Project query returned) has to be updated by hand if you
  point this example at a different chemical system or a re-run query returns a
  different phase set.
- **Structural-stability equilibration/production step counts, and its diffusive-slope
  thresholds, are placeholders/recalibration targets**, not validated defaults — see
  section 5.5.

## 7. Pitfalls encountered getting this working (read before deviating)

- **`cores_per_task: 15`, not 16, for every single-node GPU stage**
  (`fit_and_validate`, `ace_validation_single_points`,
  `structural_stability_screening`, `mace_structural_stability_screening`). Flux offers
  `64 x N - 1` cores on an N-node allocation (one core stays with the broker) — on a
  single node that's 63 usable cores. `fit_and_validate`'s smoke-test config has exactly
  4 ensemble members total (2 combos x 2 seeds); at `cores_per_task: 16`, `63 / 16 =
  3.9`, so only 3 of 4 chores fit concurrently and one GPU sits idle. At
  `cores_per_task: 15`, `63 / 15 = 4.2`, and all 4 fit. The non-GPU
  training-data-generation stages keep `cores_per_task: 16` since they aren't trying to
  pack one chore per GPU on a single node the same way.
- **`FF/ace_fit.py`'s own resume-from-checkpoint logic reads from `work_dir`, not from
  any config key.** If a fit is interrupted mid-run (e.g. a walltime timeout), a
  resubmission of the exact same chore automatically resumes from whatever
  `interim_potential_<cycle>.yaml` is newest in that member's own `work_dir` — confirmed
  via `pyace.generalfit`'s own source: `interim_potential_<cycle>.yaml` is written on
  *every* iteration (a default callback), while `interim_potential_best_cycle.yaml` is
  only written once an entire fit's configured cycles finish, i.e. never for an
  interrupted run. The checkpoint glob (`interim_potential_[0-9]*.yaml`) deliberately
  excludes `interim_potential_ladder_step_*.yaml` (basis-growth-scheme-specific, not
  used by this example's single-shot, fixed-basis-size fits) and
  `interim_potential_best_cycle.yaml` by construction.
- **The resumed checkpoint overrides `cfg["potential"]` via `potential.filename`, not
  `potential.initial_potential`** — the same key `warm_start_potential` sets at
  build-input time, just decided here at *execution* time from whatever's actually on
  disk. A fresh member folder (no prior attempt) has no checkpoint file, so this is a
  no-op there and `input.yaml`'s own `potential` config is used unchanged. If you're
  writing your own resume/warm-start logic elsewhere in this pipeline, use the same key
  — pyace's basis-growth ("ladder") scheme uses a *different* config shape entirely and
  does not apply to this example's fits.
- **`ace_fit.py` `chdir`s into `work_dir` for the duration of each fit** (`GeneralACEFit`
  writes several working files — interim potentials, metrics — relative to the current
  directory, not to an explicit output path) and restores the original working
  directory afterward regardless of outcome. Any path you pass into `overrides` for a
  manual/standalone invocation should therefore be absolute, not relative to wherever
  you happen to launch the script from.
- **`structural_stability_screening`/`mace_structural_stability_screening`'s
  `parent_levels` has no hardcoded default in `run_pipeline.py`, deliberately** —
  confirm it against your own actually-staged tree with a real
  `MDMatEnsemble.build_task_dicts` dry run before trusting it. `2` is this example's own
  confirmed value for its tree shape (no per-temperature segment in the
  structural-stability tree, unlike the MD-training stages above); it can change if your
  own staged tree's depth differs, e.g. with a different number of potential labels at
  that level.
- **Multi-node Flux rendezvous needs every flag in `launch_multi_node.slurm`'s
  `PODMAN_ARGS`, and a static `R.json`, not live GPU discovery** — see section 3. If you
  write your own launch script instead of using this one, you will silently get *N*
  independent single-node Flux instances rather than one *N*-node instance;
  `flux resource list` inside the container is the fastest way to check whether the
  merge actually happened before running anything expensive.
- **Upstream `TensorPotential` genuinely will not `pip install` into this image at
  all**, let alone run correctly against a current TensorFlow/NumPy. Its own
  `setup.py` declares `python_requires='<3.11'` (this image is Python 3.12) —
  confirmed directly: `pip install "tensorpotential @ git+https://github.com/ICAMS/
  TensorPotential.git"` fails outright with `ERROR: Package 'tensorpotential' requires
  a different Python: 3.12.3 not in '<3.11'`, before even reaching its separate
  `numpy<=1.26.4`/`pandas<=2.0` pins (stale against this image's NumPy 2.x, pulled in
  by TensorFlow 2.21.0 regardless of that pin). Even past that, upstream's
  `FitMetrics.get_regularization_loss` doesn't handle NumPy 2.x (`float()` on a
  non-0-dimensional array raises unconditionally there, where it only warned before)
  — confirmed via a real chore failure, not a hypothetical. Upstream
  (`ICAMS/TensorPotential`) has no newer commit fixing either issue as of this fork.
  `Dockerfile.ace` instead pip-installs
  [Q-CAD/TensorPotential](https://github.com/Q-CAD/TensorPotential) tag `v0.1.1-qcad1`,
  which carries both fixes (the NumPy-2.x fix, and a relaxed `setup.py` with the
  `python_requires`/stale pins dropped) — see Container Build above. If you need to
  rebuild against a newer TensorPotential commit yourself, start from this fork's
  `develop` branch, not upstream directly.
