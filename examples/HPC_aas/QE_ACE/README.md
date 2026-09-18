# QE -> pyACE: a CPU-only fitting pipeline on Pathfinder (ORNL)

This example runs a stress test of Ensemble-FF-Fit's abstractions on ORNL's
Pathfinder HPC-as-a-Service cluster, coordinated by a single script/config pair
(`run_pipeline.py` / `workflow_config.yaml`) via MatEnsemble/Flux:

1. **Quantum Espresso** (`pw.x`) DFT convergence on a small tree of bulk Si
   structures plus an isolated-atom reference (for cohesive-energy correction).
2. **pyACE**/pacemaker force-field fitting: a 6-member ensemble of ACE
   potentials, varying fitting seed and the energy/force loss balance
   (`kappa`), built from that DFT data.
3. Single-point re-evaluation of every fitted potential against the DFT
   training structures, as a first (CPU-only) sanity check of the whole
   pipeline before attempting anything more ambitious (finite-temperature MD,
   active-learning downselection, GPU acceleration).

Everything here is deliberately **CPU-only**: `pw.x` has no GPU-enabled build
on Pathfinder yet, pyACE's own GPU evaluator (`tensorpotential`) needs
Python <3.11 (incompatible with this project's Python 3.12 containers), and
Pathfinder's GPU partition has limited concurrent-GPU availability. None of
this pipeline's own compute steps need a GPU to run correctly at this scale.

Commands/paths below are written as concrete examples from a real run —
substitute your own project/user paths throughout (anywhere you see
`<your-project>`/`<you>`).

## 0. What's included vs. what you provide

This directory is a trimmed copy of a real, working run: only `POSCAR`
structure files and driver/recipe scripts are kept — all DFT/fitting/MD
*output* (converged energies/forces, the built `.pckl.gzip` training set and
sampled `input.yaml` ensemble, fitted potentials, single-point results) has
been stripped out, and the QE pseudopotential file isn't committed to the repo
either (binary, license/redistribution reasons — see step 4). Reproducing this
means actually re-running every stage below, not resuming from pre-computed
results.

## 1. Build the Apptainer container

**Build the writable sandbox somewhere other than `/scratch`.** `/scratch` on
Pathfinder is Lustre; a writable Apptainer sandbox kept there was observed
more than once to have its Python stdlib silently corrupted mid-session
(`encodings/` losing most of its files, `Fatal Python error:
init_fs_encoding` on the next import) with no single reproducing command
identified — plausibly a Lustre client-side caching issue under the kind of
many-small-files, frequent-write access pattern a live Python venv produces.
Building/keeping the sandbox under a `/projects/<proj>/proj-shared/<you>/...`
path (NFS-backed) instead has been stable across this whole pipeline's
development.

```bash
mkdir -p /projects/<your-project>/proj-shared/<you>/containers/pathfinder
cd /projects/<your-project>/proj-shared/<you>/containers/pathfinder

# Base image (Python 3.12 venv at /opt/venv, Flux 0.66.0 at /opt/flux, LAMMPS
# at /opt/lammps) -- ask your project's own maintainers where this is staged,
# or build one from Dockerfile.matensemble (containers/pathfinder/ in the
# MatEnsemble repo) if you don't have one yet.
apptainer build --sandbox matensemble_sandbox /path/to/matensemble_base.sif
```

Verify it's actually writable before doing anything else:

```bash
apptainer exec --fakeroot --writable matensemble_sandbox \
    /opt/venv/bin/python -c "import sys; print(sys.version)"
```

## 2. Install Ensemble-FF-Fit inside the sandbox

From an interactive shell in the sandbox (`apptainer shell --fakeroot
--writable matensemble_sandbox`, with this repo bound in — see the bind flags
in step 3):

```bash
apt-get update && apt-get install -y build-essential cmake liblua5.1-0 git
# liblua5.1-0 specifically -- flux-shell needs the *shared library*, not just
# the lua5.1 interpreter package; missing it blocks any real Flux-submitted
# chore, not just this pipeline's own stages (see pitfalls below).

python install_gpu_torch.py            # detects CUDA, installs the matching torch build
pip install -e ".[torchsim]"           # NOT the `cuda` extra -- that's MACE-specific
                                        # (cuequivariance), and silently drags torch to an
                                        # untested version if installed for a non-MACE pipeline

# pyace's own setup.py hardcodes os.cpu_count()-1 parallel compile jobs unless
# capped -- on a shared login node this exhausts memory (`cc1plus: out of
# memory`). CMAKE_BUILD_PARALLEL_LEVEL=2 is a safe, confirmed-working cap;
# raise it if you have a dedicated/larger allocation for this install step.
CMAKE_BUILD_PARALLEL_LEVEL=2 pip install -e ".[ace]"
```

See the main repo `README.md` for the full extras list if you need other
platform/backend combinations.

## 3. Launching the container for runtime

**This is an interactive protocol, not a batch submission** — every stage
below runs inside a single live Flux instance that you keep open in an
interactive shell for the duration of the pipeline, not something you
`sbatch` and walk away from.

```bash
# CPU-only:
interactive -N 1 -n 4 -c 1 --mem=32gb --time=1-00:00:00
# or GPU (not needed for anything in this example, but harmless if you have
# one anyway -- e.g. for a later TorchSim/GPU-MD stage):
interactive -N 1 -n 4 -c 1 --mem=32gb --time=1-00:00:00 -p gpu --gres=gpu:2

bash matensemble_submission.sh
```

`matensemble_submission.sh` starts a single-rank Flux instance inside the
container (Flux's own GPU/hwloc discovery doesn't see this system's GPUs even
with `--nv`, so GPUs are enumerated in a static, hand-written `R.json`
instead — automatic and CPU-vs-GPU-aware, see the script's own comments), then
drops you into that Flux instance's shell. From there:

```bash
flux resource list          # confirm cores (+ GPUs, if requested) show up
cd /opt/pipeline             # this example's own directory, bind-mounted here
/opt/venv/bin/python run_pipeline.py --config workflow_config.yaml --stage <stage>
```

Before running it, edit `matensemble_submission.sh`'s own `SANDBOX=` line to
point at wherever you built your sandbox in step 1 — that one value has no
portable default. `ENSEMBLE_FF_FIT`/`PIPELINE_DIR` are self-located from the
script's own path and don't need editing, even if you move this whole example
directory somewhere else.

## 4. Configure `workflow_config.yaml`

All paths in `workflow_config.yaml` are relative to this directory
(`examples/HPC_aas/QE_ACE`) — run every `run_pipeline.py` command from here
(or from `/opt/pipeline` inside the container, which is bind-mounted to this
same directory) so they resolve correctly.

**One-time setup: obtain the QE pseudopotential.** This example needs
`Si.pbe-n-kjpaw_psl.1.0.0.UPF` (PBE functional, Kresse-Joubert PAW, PSlibrary
v1.0.0 naming) under `DFT/qe_pseudos/`. It isn't committed to the repo
(`.gitignore`d — binary, and pseudopotential redistribution terms vary by
source). Get it from
[PSlibrary](https://www.quantum-espresso.org/pseudopotentials/ps-library) or
your own site's Quantum Espresso module documentation (Pathfinder's
`quantum-espresso` module may already point at a shared, site-provided
pseudopotential directory — check that first), then:

```bash
cp /path/to/Si.pbe-n-kjpaw_psl.1.0.0.UPF DFT/qe_pseudos/
```

`qe_dft.py` resolves pseudopotentials by a directory glob against each
structure's own elements, erroring loudly on an ambiguous/missing match
rather than guessing — if you swap in a different structure/element, place its
own pseudopotential here too.

Key knobs to check for your own allocation:

- `converge_dft_data.cores_per_task` / `build_ff_inputs.cores_per_task` /
  `fine_tuning.cores_per_task` — must match each other by hand where noted in
  their own comments (pyACE's own `n_workers` is baked into each ensemble
  member's `input.yaml` at `build_ff_inputs` time, not re-read from
  `fine_tuning` at fit time).
- `single_points.cores_per_task` / `molecular_dynamics.cores_per_task` are
  deliberately `1` and should stay that way regardless of allocation size —
  pyACE's own evaluator has no internal thread parallelism (confirmed via
  `ldd`; see pitfalls below), so more cores per chore only reduces how many
  chores fit on a node at once, never speeds up an individual one.

## 5. Run the pipeline, stage by stage

```bash
/opt/venv/bin/python run_pipeline.py --config workflow_config.yaml --stage converge_dft_data
/opt/venv/bin/python run_pipeline.py --config workflow_config.yaml --stage build_ff_inputs
/opt/venv/bin/python run_pipeline.py --config workflow_config.yaml --stage fit_and_validate
/opt/venv/bin/python run_pipeline.py --config workflow_config.yaml --stage copy_force_fields
/opt/venv/bin/python run_pipeline.py --config workflow_config.yaml --stage mirror_training_structures
/opt/venv/bin/python run_pipeline.py --config workflow_config.yaml --stage single_points
```

`converge_dft_data` needs `DFT/training/*` and `DFT/isolated_elements/Si`
converged before `build_ff_inputs` can build a training set from them; give it
several minutes per structure the first time (isolated-atom convergence in
particular needed several DFT-numerics adjustments to get reliably fast — see
pitfalls). `copy_force_fields`/`mirror_training_structures` are cheap,
non-Flux prep steps (plain file copies, no chore submission) for
`single_points`, not their own multi-minute stages.

A `molecular_dynamics` stage (CPU-only, plain-ASE Langevin/VelocityVerlet MD
via `MD/finite_temperature/`) also exists and was validated working
end-to-end, but is **not part of this walkthrough** — a real MD trajectory
with only 2 training structures behind the fit diverges within ~100-200 steps
regardless of starting temperature (confirmed, not assumed: the fit is
genuinely underdetermined at this training-set size, not a driver bug). Revisit
once the training set has grown past this smoke-test scale.

## 6. Pitfalls encountered getting this working (read before deviating)

- **`pw.x` runs as a single MPI rank per chore, deliberately.** Pathfinder's
  OpenMPI 5.0.5 bootstraps via PMIx (`SLURM_MPI_TYPE=pmix`); real multi-rank
  MPI cooperation *inside* a Flux-launched chore was tested directly and
  confirmed **not** to work yet (ranks silently fall back to independent
  singletons rather than actually communicating) — a genuine
  Flux/PMIx-interop gap, not a configuration mistake, and not chased further.
  `OMP_NUM_THREADS=cores_per_task` is still a real, working single-node
  parallelism lever in the meantime.
- **Even single-rank chores can hang indefinitely (not crash) if
  `inherit_env=True` leaks the outer Slurm allocation's own PMIx env vars**
  (`PMIX_SERVER_URI*`, `PMI_RANK`, etc.) into `pw.x`'s subprocess — it tries to
  rendezvous against the wrong (outer) PMIx server. Fixed by stripping every
  `PMI_`/`PMIX_`-prefixed env var before launching `pw.x` (see `qe_dft.py`).
- **`LD_LIBRARY_PATH` for `pw.x` must be set inline in the shell command
  string, never via `subprocess.run`'s `env=`.** `env=` also applies to the
  intermediate `/bin/sh` that `shell=True` spawns, and this container's
  `/bin/sh` needs a *different*, incompatible glibc than `pw.x` itself
  (`GLIBC_2.38 not found`) — it crashes before ever reaching `pw.x`. Scoping
  the assignment inline (`VAR=val cmd`, POSIX shell semantics) keeps it off
  `/bin/sh`'s own environment.
- **FlexiBLAS needs its config file and backend plugin bound at their exact
  host paths** (`/etc/flexiblasrc(.d)`, `/lib64/flexiblas`,
  `/usr/lib64/flexiblas`) — it resolves its plugin by bare filename through
  its own internal search path, not visible to `LD_LIBRARY_PATH` at all.
- **`flux-shell` needs `liblua5.1.so.0`**, not just the `lua5.1` interpreter
  package — missing it blocks *any* real Flux-submitted chore, silently
  enough to look like a hang rather than a missing-library error at first.
- **pyACE's own `GeneralACEFit` isn't re-exported at `pyace`'s top level** in
  the currently-pinned git ref — import from `pyace.generalfit` instead.
  Worth re-checking if `python-ace`'s pinned commit is ever updated.
- **pyACE's `backend.n_workers` (not `nworkers`) controls fitting
  parallelism** (`multiprocessing`-based, across independent structures) --
  confirm the exact key via `pyace.BACKEND_NWORKERS_KW` if this drifts. This
  is a *completely separate* axis from single-point/MD evaluation speed: the
  native `pyace` evaluator itself has no internal OpenMP/pthread parallelism
  at all (confirmed via `ldd` on its compiled extension) — a single MD/
  single-point run is bound to one core no matter how many a chore is given.
- **A relative `data.filename` baked into an ensemble member's `input.yaml`
  breaks once `ace_fit.py` `chdir()`s into its own `work_dir`** before
  fitting (needed so pacemaker's interim-potential/metrics files land
  alongside that run's own `input.yaml`). `build_ff_inputs` resolves this
  path to absolute before writing it, at build time — if you ever see a
  `FileNotFoundError` for the dataset from inside a fit chore, check whether
  something re-introduced a relative path here.
- **`MDMatEnsemble.build_task_dicts`'s proximity matcher needs an identical
  filename per folder**, not a glob — `ace_fit.py` names each ensemble
  member's fitted potential uniquely (`FF_0.yaml` ... `FF_5.yaml`), so
  `copy_force_fields` renames each to a uniform `FF.yaml` per subfolder before
  `single_points` can proximity-match across all of them at once.
- **A `Running` count that doesn't move for several minutes on a
  `single_points`-style stage is not on its own evidence of a hang.** Several
  independent chore processes all cold-starting Python and importing pyace's
  compiled extension (+ numpy/pandas/numexpr) at once from the same
  NFS-mounted sandbox contend on real NFSv4 file locks (`ps`/`/proc/<pid>/
  wchan` showing `nfs_set_open_stateid_locked`/`folio_wait_bit_common`) —
  confirmed to resolve on its own (once in ~2 minutes, once in ~9 minutes on a
  smaller/busier allocation) rather than being a true deadlock. Check
  `ps`'s cumulative CPU `TIME` staying at `0:00` across *every* worker before
  assuming something needs to be killed.
- **The isolated-atom QE reference energy needed several DFT-numerics fixes**
  to converge reliably: a 15 -> 10 Å vacuum box (memory), Davidson ->
  `diagonalization: cg` (crash), `degauss: 0.01` -> `0.1` (exact p-orbital
  degeneracy made `efermig` unable to bracket the Fermi level at the tighter
  smearing), `mixing_beta: 0.7` -> `0.3` (oscillating non-convergence), and
  `electron_maxstep: 300` (QE's own 100-iteration default cap was too low).
  None of this is specific to this one calculation — expect similar tuning for
  any other small/symmetric/degenerate isolated-atom reference.
