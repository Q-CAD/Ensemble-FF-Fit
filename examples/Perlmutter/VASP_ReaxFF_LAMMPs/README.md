# VASP -> ReaxFF -> LAMMPS: a fixed-dataset ReaxFF fitting/validation pipeline on Perlmutter

This example runs a full ReaxFF force-field fitting and validation investigation on
NERSC Perlmutter, coordinated by a single script/config pair (`run_pipeline.py` /
`workflow_config.yaml`) via MatEnsemble/Flux:

1. **VASP** DFT convergence on a tree of starting structures (bulk phases, defects,
   equation-of-state points, amorphous structures, isolated atoms).
2. **parse2fit** conversion of converged DFT results into ReaxFF's own `geo`/
   `trainset.in` training format.
3. **JAX-ReaxFF** ensemble fitting: a full cross product of parse2fit-generated
   training variants against a set of parameter-catalog "blocking schemes" (which
   groups of ReaxFF parameters — bonds, off-diagonal, angular, dihedral — are
   trainable vs. frozen in a given ensemble member).
4. **LAMMPS/ReaxFF** validation: static single-point re-evaluation of every fitted
   force field against DFT ground truth (both AIMD melting trajectories and the
   static training set), ranked by combined energy/force deviation.
5. **LAMMPS/ReaxFF (Kokkos GPU)** finite-temperature MD: short, real room-temperature
   NPT runs of every fitted force field on representative supercells, checked for
   coordination-number stability against the pre-MD structure — a force field that
   can't survive a few picoseconds of real dynamics on a phase of interest is
   flagged here, independent of how it scored on static single points.

Unlike the Frontier `RMG_MACE_ASE` example, this pipeline deliberately uses a
**strictly fixed training dataset** — an earlier UQ/active-learning loop (downselect
fitted force fields -> run single points on their own MD-sampled frames -> select the
most-disagreed-upon frames as next-round DFT candidates) was tried and then removed
entirely (not just disabled) once in practice the JAX-ReaxFF ensemble fits here
weren't judged reliable enough yet to drive that kind of active selection, and
parse2fit's own relative-energy handling is user-directed rather than fully
autonomous — both of which the active-learning loop would have needed. If you want
that loop back, the Frontier example still has a live version of the same pattern to
crib from.

Commands/paths below are written as concrete examples from a real run; substitute
your own project/user/scratch paths throughout.

## 0. What's included vs. what you provide

This directory is a trimmed copy of a real, working investigation, scoped down
deliberately at every stage to keep it small and to avoid shipping data that would
look authoritative but wouldn't actually match a fresh run:

- **`DFT/`**: `POSCAR` files and the two input YAMLs (`vdW_single_point.yml`, the VASP
  recipe; `reaxff_newest_kT.yml`, the parse2fit directive) only — **no
  `vasprun.xml`, no `properties.json`, no `POTCAR`**. Re-running `converge_dft_data`
  regenerates converged DFT results from these POSCARs; `DFT/vasprun_to_properties.py`
  then derives `properties.json` from the resulting `vasprun.xml`. Until you've done
  that, `parse2fit_generation` and everything downstream of it (fitting, validation,
  ranking, coordination-check) has nothing real to work with. Two directories from
  the original investigation are deliberately **not** included at all: `test_runs/`
  (an early "does VASP work through this container at all" smoke test, not part of
  the real dataset) and `pymatgen_pseudos/` (a local POTCAR cache — VASP
  pseudopotentials are licensed and can't be redistributed; you need your own copy,
  see step 5).
- **`FF/`**: only the seed `ffield` (`FF/generation_0/original/ffield`) and the full
  parameter catalog (`FF/params_template/params`), plus the two driver scripts
  (`build_reaxff_ensemble_inputs.py`, `jax_reaxff_fit.py`). **No fitted force fields,
  no parse2fit-generated training variants, no per-blocking-scheme fitting inputs**
  — all of that is downstream of your own DFT results and would look different from
  a fresh run anyway, so shipping the current investigation's copies would be
  actively misleading, not just bulky (the full set was ~82MB; this trimmed set is
  well under 100KB).
- **`MD/`**: recipe files only — LAMMPS `.in`/`control` files and driver scripts
  (`MD/finite_temperature/coordination_check/{cn_checker.py, lammps_inputs/}`,
  `MD/single_points/reaxff_validation/lammps_inputs/`) — **no staged force fields,
  no single-point/MD run output, no ranking or coordination-comparison result
  files**. The two former UQ-only directories
  (`MD/finite_temperature/uq/`, `MD/single_points/uq/`) are gone entirely along with
  the UQ loop itself.

Reproducing the pipeline means actually re-running every stage below, starting from
real VASP DFT convergence, not resuming from pre-computed results.

## 1. Build/obtain the Podman-HPC container

This example deliberately does **not** ship its own `Dockerfile.matensemble` —
that file is MatEnsemble/Perlmutter-specific, not something Ensemble-FF-Fit's own
scope covers, and it changes as Perlmutter's own container/module stack does. Fetch
the canonical one instead:

```bash
curl -L -o Dockerfile.matensemble \
    https://raw.githubusercontent.com/FredDude2004/MatEnsemble/main/containers/perlmutter/Dockerfile.matensemble
```

Then append this pipeline's own two extra dependencies (JAX-ReaxFF and parse2fit —
see step 4 for why these are baked into the image rather than installed live) to
the end of the file you just fetched:

```bash
cat >> Dockerfile.matensemble <<'EOF'

RUN /opt/basic/bin/python -m pip install \
    "jaxreaxff @ git+https://github.com/Q-CAD/JAX-ReaxFF.git@develop" \
    "parse2fit @ git+https://github.com/Q-CAD/parse2fit.git"
EOF
```

Then build:

```bash
podman-hpc build -f Dockerfile.matensemble -t matensemble:ff-fit .
```

Re-fetch the upstream file (re-running the `curl` above) and re-append the same
`RUN pip install` block whenever the upstream Dockerfile changes — don't just patch
your local copy in place, or it'll silently drift from whatever MatEnsemble/Perlmutter
actually recommend at the time.

`Dockerfile.matensemble`'s base image, `nersc/flux:26.05`, already ships a working
LAMMPS build with GPU/KOKKOS/REAXFF/PLUMED/KIM/ML-SNAP/ML-QUIP compiled in — you do
**not** need to separately build or merge in a LAMMPS layer. Verify this rather than
assume it before relying on it:

```bash
podman-hpc run --rm --gpu matensemble:ff-fit lmp -in /opt/lammps/examples/reaxff/AB/in.reaxff
```

If that runs cleanly (and `has_package('KOKKOS')`/`has_package('GPU')` both report
`True` — see step 3 below for how to check), the base image is sufficient; only fall
back to a custom-built/merged LAMMPS layer if a real gap turns up here.

For multi-node batch use, migrate the image after building:

```bash
podman-hpc migrate matensemble:ff-fit
```

VASP itself is **not** part of this container image at all — it's the host's NERSC
`vasp` module, bind-mounted in at container-launch time. See step 3.

## 2. Why `launch_multi_node.slurm` specifically

This isn't a generic "submit a Slurm job" script — it exists because getting
multiple Flux brokers (one per node) to merge into a single coordinated multi-node
Flux instance, rather than several independent single-node ones, needs several
things that aren't part of a default `podman-hpc run` and aren't obvious from a
single-node test:

- **`--network=host`, not the default (isolated, `pasta`-based) networking.**
  Without this, brokers on different nodes can't reach each other at all,
  regardless of what `srun --mpi=pmi2` sets up for the PMI2 rendezvous itself.
- **`-e SLURM_* -e PALS_* -e PMI_*`** — explicit environment-variable passthrough.
  `srun --mpi=pmi2` sets PMI rendezvous info in its own environment, but
  `podman-hpc` doesn't forward arbitrary host env vars into the container by
  default — without this, `flux start` inside the container never sees any of it
  and silently falls back to a standalone single-broker instance.
- **`-v /var/spool/slurmd -v /run/munge -v /run/nscd`** — PMI2's own
  authentication/step-communication sockets, needed for the rendezvous handshake to
  complete inside the container at all.
- **`-v /dev/cxi0-3 -v /dev/xpmem`** — the actual Slingshot NIC device nodes and
  xpmem shared-memory device. Binding only the `/opt/cray` *software* tree (library
  resolution) is not enough — the interconnect needs these *device nodes* at the
  kernel level for brokers on different nodes to actually talk to each other.
- **No live NVML/hwloc GPU discovery at all.** Instead, a static `R.json` resource
  description (one entry per allocated node, parsed from `SLURM_JOB_NODELIST`) plus
  a `resource.toml` (`noverify=true`) pointed at via `FLUX_CONF_DIR`. This is *why*
  a static description is used instead of the more obviously-correct-looking live
  discovery: live per-broker discovery only ever discovers each broker's own local
  node — there's no mechanism for two independently-discovered single-node views to
  merge into one, no matter how correct each one is individually.

If you write your own launch script instead of using this one, you'll silently get
N independent single-node Flux instances rather than one N-node instance, and
whichever stage you run will only ever see one node's worth of resources —
`flux resource list` inside the container is the fastest way to check whether the
merge actually happened before running anything expensive.

This script also assumes it's run from inside `run_pipeline/` itself (`workflow_config.yaml`'s
own keys are relative to CWD, not to the script's location — `-w $PWD` only does the
right thing if `$PWD` is `run_pipeline/` when the script starts). Works both as
`sbatch launch_multi_node.slurm` and as `bash launch_multi_node.slurm` from inside an
already-active interactive allocation.

## 3. Changing which VASP version/build is used

`vasp_std` is bind-mounted in from Perlmutter's own NERSC `vasp` module tree, not
baked into the container image, so switching versions is entirely a matter of
updating paths, not rebuilding anything:

1. **`DFT/vdW_single_point.yml`** — update both `vasp_executable` and `command`
   together (they must point at the same binary):
   ```yaml
   vasp_executable: "/global/common/software/nersc9/vasp/vasp/<version>-<cpu|gpu>/bin13/vasp_std"
   command: "/host_lib/ld-linux-x86-64.so.2 /global/common/software/nersc9/vasp/vasp/<version>-<cpu|gpu>/bin13/vasp_std"
   ```
   The `/host_lib/ld-linux-x86-64.so.2` prefix on `command` is required, not
   optional — `vasp_std` is RHEL/SLES-built, and its hardcoded ELF interpreter path
   has to resolve to the *real host* loader, not the container's own Ubuntu one at
   that same path. A bare invocation fails with a silent SIGSEGV that a plain `ldd`
   check won't catch (it never exercises deep runtime init like TLS setup, where
   the mismatch actually crashes) — requires `-v /lib64:/host_lib` at
   container-launch time (already in `launch_multi_node.slurm`'s `PODMAN_ARGS`).
2. **`run_pipeline.py`'s `VASP_LD_LIBRARY_PATH`** — re-verify every path in this
   string if the VASP module, container image, or NVHPC/Cray module versions
   change. The CPU and GPU builds need genuinely different NVHPC trees (confirmed
   via `readelf -d` on the actual binary, not assumed): the GPU build links against
   a *split* toolchain (`.../25.9/compilers/{lib,extras/qd/lib}`,
   `.../25.9/math_libs/lib64`, plus a separately-versioned
   `.../26.5/cuda/13.2/lib64` and `.../26.5/math_libs/13.2/lib64`), while a CPU
   build only needs the `26.5` tree.
3. **Available modules** (as of this writing): `vasp/5.4.4-cpu` (site default),
   `vasp/6.4.2-{cpu,gpu}`, `vasp/6.4.3-{cpu,gpu}`, `vasp/6.6.0-{cpu,gpu}`,
   `vasp-tpc/{5.4.4,6.4.2}-{cpu,gpu}`. **Watch out**: a bare `module load
   vasp/6.4.2` (no suffix) silently resolves to `-gpu`, not `-cpu` — always specify
   the suffix explicitly. These modules live under `modulefiles_hotfixes` and are
   actively patched, so re-verify rather than trust a cached path:
   ```bash
   module load vasp/6.4.2-gpu
   ldd $(which vasp_std)          # confirms the actual linked library tree
   module show vasp/6.4.2-gpu     # confirms the module's own env additions
   ```
4. **Container-launch-level requirements** — not settable from `vdW_single_point.yml`
   or `run_pipeline.py`, must already be present on whatever `podman-hpc run`
   invocation starts the chore's container (already wired into
   `launch_multi_node.slurm`'s `PODMAN_ARGS`, but worth knowing if you're adapting
   this for your own launch script): `--gpu`, `--group-add keep-groups` (`vasp_std`'s
   own directories are group-restricted on the host — without this, a rootless
   container can't even traverse into them, surfacing as a confusing "No such file
   or directory" rather than a permission error), and bind mounts for
   `/global/common/software/nersc9/vasp`,
   `/opt/nvidia/hpc_sdk/Linux_x86_64/{25.9,26.5}`, `/opt/cray`,
   `/global/common/software/nersc9/darshan/3.4.6-gcc-13.2.1`, and
   `/usr/lib64:/host_lib64`, `/lib64:/host_lib` (see step 2's note on why these two
   are mounted at non-conflicting paths rather than directly over their originals).
5. **Pseudopotentials.** `vdW_single_point.yml`'s `pseudopotentials_directory`
   currently points at a path this repo does **not** ship (see step 0 — VASP
   POTCARs are licensed). Point it at your own pymatgen-organized pseudopotential
   directory (`POT_LDA_PAW`/`POTPAW_PBE_54`-style layout) instead.

## 4. Installing Ensemble-FF-Fit, JAX-ReaxFF, and parse2fit

- **Ensemble-FF-Fit** (this repo) — a standard editable install:
  ```bash
  python install_gpu_torch.py
  pip install -e ".[mace,cuda]"
  ```
  (See the main repo README for the full extras list.)
- **JAX-ReaxFF and parse2fit are baked into the container image itself** (the
  `RUN pip install` block step 1 has you append to the fetched
  `Dockerfile.matensemble`), **not** installed live at container-launch time. This
  is a deliberate correction, not the original design: both used to be
  PYTHONPATH-injected from local, actively-edited clones, and an `Ensemble-FF-Fit`
  pip extra (`jaxreaxff`, still present in `pyproject.toml` as a reference for
  other, non-ephemeral environments) was tried as a replacement — but **confirmed
  broken for this specific container workflow** (2026-09, not just theoretical):
  `/opt/basic/lib/python3.12/site-packages` lives entirely inside the container's
  own filesystem, not any host-mounted path, so a live `pip install` inside one
  `podman-hpc run --rm` container vanishes the moment that container exits —
  invisible to every other chore's own separately-launched container. Baking the
  install into the Dockerfile itself (see step 1) is what actually persists it.
  JAX-ReaxFF's own `setup.py` pins its full GPU dependency chain
  (`jax[cuda12]==0.4.35`, `jax_md`, `dm-haiku`, `flax`, `optax`, ...) — an unpinned
  resolve drifts to newer releases that break at import time (confirmed via jax's
  own PyPI metadata) — so a plain `pip install` of the git URL already gets the
  exact right versions, no separate pinned-requirements script needed.
  Confirmed live (2026-09): a fresh `pip install` of both packages together, in one
  container session, installs cleanly and `import jaxreaxff.driver` / `import
  parse2fit` both succeed, with `dataclasses` still correctly resolving to the real
  stdlib module (not the shadowing bug an earlier `--target=deps`-based approach
  hit — that was specific to front-loading PYTHONPATH, and doesn't reproduce under
  a normal install). **Not yet independently confirmed**: a real GPU (`jax.devices()`
  reporting an actual device, not just a clean import) after a full image rebuild
  from this exact Dockerfile — the rebuild was stopped before finishing. Do that
  full rebuild-and-GPU-check once before relying on this for a real fitting run:
  ```bash
  podman-hpc build -f Dockerfile.matensemble -t matensemble:ff-fit .
  salloc -A <account>_g -C gpu --qos interactive -t 0:15:00 -N 1 --ntasks-per-node=1 --gpus-per-node=1
  podman-hpc run --rm --gpu matensemble:ff-fit /opt/basic/bin/python -c \
      'import jax, jaxreaxff.driver, parse2fit; print(jax.devices())'
  ```
  To update the pinned JAX-ReaxFF ref later (a new commit on its own `develop`
  branch, or a different tag entirely), edit the `@develop` in the `RUN pip install`
  line you appended to your local `Dockerfile.matensemble` (step 1) and rebuild —
  there's no other file to touch for this, since `run_pipeline.py` no longer does
  any JAX-ReaxFF/parse2fit path handling of its own.

**`DFT/reaxff_newest_kT.yml`** (the parse2fit directive) also needs attention
separately from the above: it has **46 occurrences** of an absolute path baked into
its `subtract`/`add`/`directory`/`output_directory` fields, one per
`runs_to_generate` entry. This isn't an oversight — `parse2fit`'s own path
resolution requires absolute paths in the directive file, it can't take relative
ones — but it does mean every one of those 46 needs updating to match wherever you
actually put this example:

```bash
sed -i 's|/pscratch/sd/r/rym/MatEnsemble/run_pipeline|<your absolute path to this directory>|g' DFT/reaxff_newest_kT.yml
```

## 5. Run the pipeline, stage by stage

Each stage is its own compute allocation/container invocation — run them in order
(or all in one process via `--stage all`, if one allocation covers every stage's
needs and you've already regenerated real DFT results for everything downstream to
work against):

```bash
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
```

`rank_reaxff_validation` writes a ranked comparison of every fitted force field
against DFT ground truth (relative energy + forces on the AIMD validation
trajectories, forces on the training set). `check_coordination_stability` writes a
second, independent ranking based on real short MD runs rather than static single
points — the two frequently disagree meaningfully (a force field that scores well
statically can still be dynamically unstable, and vice versa), which is the entire
point of running both.

## 6. Known limitations

- **The Dockerfile-baked JAX-ReaxFF/parse2fit install (step 4) is not yet
  independently confirmed with a real GPU.** A live install of both packages
  together, in one container session, was confirmed to work (clean imports,
  correct pinned versions, no `dataclasses`-shadowing regression) — but a full
  image rebuild from the updated `Dockerfile.matensemble` followed by a real
  `jax.devices()` GPU check was stopped before finishing. Do that check (step 4's
  own verification block) before relying on this for a real fitting run.
- **parse2fit is install-only scope, not debugged as part of this pipeline.**
  It hasn't been actively maintained and may have bugs; if you hit one, that's a
  parse2fit issue to raise/fix upstream, not something this pipeline's own code is
  expected to work around.
- **The coordination-stability check's metric (pymatgen `CrystalNN`, weighted
  coordination number) is a first pass, not a settled choice.** It reliably catches
  a full structural-motif breakdown but isn't the most descriptive metric available
  — see `MD/finite_temperature/coordination_check/cn_checker.py`'s own docstring
  for how to swap in a different structural descriptor without touching
  `run_pipeline.py` (it's a config-pointed, dynamically-imported driver script, same
  convention as `fine_tuning.ff_task`).
- **GPU VASP's own CUDA stack (13.2) and this container's LAMMPS-GPU stack
  (~12.9-class) are two separately-versioned CUDA trees co-existing in the same
  container** — confirmed working via the exact library paths in step 3, but this
  is inherently more fragile than a single-CUDA-version setup; re-verify the linked
  paths (`readelf -d`/`ldd`) whenever either the VASP module or the container's own
  CUDA/driver stack changes.

## 7. Pitfalls encountered getting this working (read before deviating)

- **`pair_style reaxff` on Kokkos GPU needs `newton on` + an explicit neighbor-list
  override, not Kokkos's own GPU default.** Kokkos's default for a GPU backend is
  `neigh full` + `newton off`; ReaxFF's Kokkos implementation
  (`pair_reaxff_kokkos.cpp`) calls straight through to the *base* (non-Kokkos)
  `PairReaxFF::init_style()`, which unconditionally requires `newton pair on` — a
  real conflict between Kokkos's own default and what this specific pair style
  needs, not a sign that ReaxFF can't run on GPU at all. Fix: `package kokkos neigh
  half neigh/qeq half newton on` before `newton on` in the `.in` file — `neigh half`
  auto-promotes to the GPU-safe `HALFTHREAD` variant whenever a GPU is in use (see
  `MD/finite_temperature/coordination_check/lammps_inputs/in.npt_room_temp`'s own
  comments for the full trace through LAMMPS's source that confirmed this).
- **Kokkos minimize needs a Kokkos-enabled `min_style` for *every* minimize call, no
  silent CPU fallback.** Unlike `pair_style`/`fix`, which auto-suffix cleanly with a
  CPU fallback if no `/kk` variant exists, `minimize` errors outright
  ("Must use a Kokkos-enabled min style") if a plain style is used while Kokkos is
  active. This build has no `fire/kk` (only `cg/kk`) — use `cg/kk` for both a loose
  and a tight minimization pass (via tolerance arguments, not a different
  `min_style`), not `fire` for the loose pass.
- **A LAMMPS `.in` file's `write_data`/`write_restart` with a bare relative filename
  writes into the *chore's own process working directory*, not the structure's own
  output directory** — unlike `dump`/`log`, which need (and get) an explicit
  absolute path via a LAMMPS variable set by the driver script. This is easy to miss
  because the chore still reports success and writes a plausible-looking
  `properties.json` — the actual NPT output just isn't where anything downstream
  expects it, and gets silently overwritten by the next structure in the same
  chore's own loop. Give every `write_data`/`write_restart` target an explicit
  `${output_dir}`-style variable, the same way `dump`'s `${dump_file}` already does.
- **`MDMatEnsemble.build_lists`' internal label ordering must be derived from the
  same list used to build the actual values, not re-derived from `self.options`'
  raw dict order.** If a caller's `options` dict happens to list `in_file` before
  `structure` (rather than after), the two independent derivations silently
  disagree and every task dict gets its `structure`/`in_file` values swapped — a
  real, previously-latent bug in `EnsembleFFFit/base.py`, not something specific to
  this example, but one that only ever manifested once a caller here passed both
  keys together in that particular order.
- **`pymatgen.io.vasp.outputs.Vasprun.final_structure` is not safe to assume matches
  a single-ionic-step vasprun.xml's own per-step structure.** It reads from a
  top-level `<structure name="finalpos">` block rather than the per-`<calculation>`
  one; for these per-frame AIMD single-point re-evaluations, that top-level block
  turned out to be a stale artifact shared across every frame in a trajectory, while
  `vasprun.ionic_steps[-1]['structure']` correctly varied frame to frame. Confirmed
  via a real trajectory (not assumed) — `DFT/vasprun_to_properties.py` derives
  `POSCAR` from `ionic_steps[-1]['structure']` specifically because of this, not
  from `vasprun.final_structure`.
- **`EnsembleFFFit.utilities.general.import_module_from_path` must register the
  loaded module in `sys.modules` before executing it**, or any driver script it
  loads that uses `multiprocessing.Pool` internally fails with a `PicklingError`
  the moment a worker process tries to unpickle a function defined in that module.
  Already fixed in `EnsembleFFFit/utilities/general.py`; worth knowing if you write
  a new driver script that wants to parallelize with `multiprocessing` rather than
  Flux chores.
- **An oversized `parent_levels` doesn't fail loudly — it silently collapses many
  intended chores into one.** `MDMatEnsemble.batch_by_parent`'s merge-child-paths
  step will happily merge across *different* force fields into a single chore if
  `parent_levels` walks up far enough past each force field's own directory,
  rather than erroring. Sweep `parent_levels` against the real staged tree (a
  direct `build_task_dicts` call, checking the resulting chore count and
  structures-per-chore) before trusting a value copied from a different stage with
  a different tree depth.
