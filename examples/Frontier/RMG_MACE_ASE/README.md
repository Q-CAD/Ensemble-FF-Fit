# RMG -> MACE -> ASE: an active-learning pipeline on Frontier

This example reproduces a full active-learning loop on OLCF Frontier, coordinated by
a single script/config pair (`run_pipeline.py` / `workflow_config.yaml`) via
MatEnsemble/Flux:

1. **RMG** DFT convergence on a tree of starting structures (bulk relaxation,
   defects, equation-of-state points, isolated atoms) — and, from the second
   loop on, the DFT candidates the previous loop selected.
2. **MACE** force-field fitting (an ensemble of fine-tuned models, seeded from a
   foundation model) and validation against the DFT ground truth.
3. **ASE** finite-temperature MD sampling with the foundation model, then
   single-point re-evaluation of sampled frames across the whole ensemble.
4. Uncertainty-based downselection of the most-disagreed-upon frames as the next
   round's DFT candidates.

Everything below assumes Frontier (OLCF), Apptainer, and the AMD ROCm GPU stack.
Paths are written as concrete examples from a real run; substitute your own
project/user paths throughout. The Perlmutter
[`VASP_MACE_ASE`](../../Perlmutter/VASP_MACE_ASE/README.md) example is this
pipeline with VASP swapped in; the two are kept in step.

## 0. What's included vs. what you provide

This directory is a trimmed copy of a real, working run: only `POSCAR` structure
files and driver/recipe scripts are kept — all RMG/MACE/MD *output* has been
stripped out, and the MACE foundation-model checkpoint (`model.model`, ~80MB)
isn't committed — see step 5. Reproducing the pipeline means re-running every
stage below, not resuming from pre-computed results.

| File | What it's for |
|---|---|
| `build_image.slurm` | Pull a MatEnsemble Frontier image to a `.sif`; optionally build the derived image (step 1) |
| `matensemble-frontier.def` | Derived image: the pulled image plus the Python packages this example needs (step 1c) |
| `launch_flux.slurm` | Start Flux inside the image across the allocation and run the pipeline (step 4) |
| `run_pipeline.py`, `workflow_config.yaml` | The pipeline and its config (steps 5-7) |
| `workflow_config.smoke.yaml`, `make_smoke_copy.sh` | 1-node smoke test (step 6) |

## 1. Get the container image

### 1a. Pick the image

MatEnsemble publishes Frontier images as
`ghcr.io/freddude2004/matensemble:frontier-v<version>`. This example was last
validated with **`frontier-v0.5.8`** plus the derived image of step 1c (which
upgrades MatEnsemble inside it to 0.5.8 — the published tag itself still carries an
older MatEnsemble; see pitfalls). When moving to a newer tag, re-run step 1c and the
smoke test (step 6), and update this paragraph.

### 1b. Pull it into a `.sif`

```bash
mkdir -p /lustre/orion/<proj>/scratch/$USER/images
IMAGE_DIR=/lustre/orion/<proj>/scratch/$USER/images sbatch build_image.slurm
# IMAGE_TAG=frontier-v0.5.9 to pick another tag
```

This pulls on a node's NVMe (much faster than on Lustre; ~5-10 min), retries the
occasional proxy stream error, prints the image's MatEnsemble/Flux/torch versions,
and copies `matensemble-<tag>.sif` to `IMAGE_DIR`. No writable sandbox is needed:
everything below runs the `.sif` read-only. Edit `#SBATCH -A` for your project.

### 1c. Add Python packages: build the derived image

The published image has MatEnsemble, Flux and LAMMPS, but **not** torch, mace-torch,
openequivariance, or some of Ensemble-FF-Fit's own dependencies.
`matensemble-frontier.def` adds them (pinned), on top of the pulled `.sif`:

```bash
DEF=matensemble-frontier.def IMAGE_DIR=/lustre/orion/<proj>/scratch/$USER/images sbatch build_image.slurm
```

It reuses the `.sif` from step 1b if it's already in `IMAGE_DIR`, builds
`matensemble-<tag>-ensemblefffit.sif` with `apptainer build --fakeroot` (~5 min), and
fails the build — not your first chore — if torch isn't a ROCm build, MatEnsemble
isn't the expected version, MACE lacks `--freeze`, or an import fails. To add a
package, add it (pinned) to the `.def`'s pip lines and its import to the check at the
end, then rebuild.

**Why a derived image rather than an overlay or a writable sandbox.** Considered:

- *Derived image (chosen).* Reproducible from a file in this repo, immutable, and safe
  for every node of a multi-node job to read at once. Costs a ~5 min rebuild per
  change and a second ~14 GB `.sif`.
- *`pip install --target <lustre dir>`, put on `PYTHONPATH`* (`launch_flux.slurm`'s
  `EXTRA_SITE_PACKAGES`). Fine for trying a package out quickly; but nothing records
  what's in it, pip can't see the image's own packages when resolving (it can pull a
  second, wrong torch), and importing large packages from Lustre on many nodes is
  slow. Fold anything you keep into the `.def`.
- *Writable overlay / sandbox* (what this example used before). A writable overlay
  can only be mounted by one process at a time, so it can't serve a multi-node Flux
  instance; a sandbox needs `--fakeroot --writable` to change, takes ~30 min to
  rsync to Lustre, and drifts from any recorded recipe (the old sandbox's torch/MACE
  were installed by hand and were not in the image it came from).

Ensemble-FF-Fit and pyRMG themselves are *not* installed in the image:
`launch_flux.slurm` puts your checkouts on `PYTHONPATH`, so editing them never needs a
rebuild.

## 2. Get Ensemble-FF-Fit and pyRMG

Clone both somewhere on Lustre (they're bind-mounted into the container via Frontier's
default `/lustre/orion` bind) and set `ENSEMBLE_FF_FIT`/`PYRMG` in
`launch_flux.slurm` (or the environment) to their paths. pyRMG's repo is the `rmg`
extra in `pyproject.toml`. Nothing needs `pip install`.

## 3. Build the `rmg-gpu` executable natively

The container does *not* build RMG itself — `rmg-gpu` is built natively on
Frontier using the real Cray module stack, then run *inside* the container by
binding in the host paths it needs (step 4). Building inside the container from
scratch was tried first and abandoned: it surfaces a cascading series of missing
shared libraries (`libpmi` -> `libcxi` -> `libnl-3` -> `libfftw3f_omp` -> ...) that
are far easier to satisfy by reusing Frontier's own module system.

```bash
module load PrgEnv-gnu/8.6.0 gcc-native/13.2 cmake Core/24.00 bzip2 boost/1.85.0 \
    craype-x86-milan cray-fftw cray-hdf5-parallel craype-accel-amd-gfx90a rocm/6.3.1
export MPICH_GPU_SUPPORT_ENABLED=0

git clone <rmgdft-source-repo> rmgdft   # wherever you're keeping the RMG source
cd rmgdft
mkdir build-frontier-gpu && cd build-frontier-gpu
cmake .. -DRMG_HIP_ENABLED=1 -DHIP_PATH="/opt/rocm-6.3.1/"
make rmg-gpu -j 20 -k
```

Update `DFT/vdW_quench.yml`'s `rmg_executable`/`vdwdf_kernel_filepath` (both
`/path/to/...` placeholders) to point at your build and its bundled
`XC/vdW_kernel_table` (`make_smoke_copy.sh` does this for its scratch copy).

### 3b. Matching the container to your `rmg-gpu` build

**Rule: `rmg-gpu` must run against the same ROCm (and Cray PE) libraries it was built
with, not the image's.** The image ships its own ROCm (6.3.3) for torch/MACE; a binary
built against a different ROCm partly works against it, which is what makes this easy to
miss. Measured with the build above (`rocm/6.3.1`) on the isolated Bi atom, inside the
container:

| `rmg-gpu` run against | `davidson` | `multigrid` |
|---|---|---|
| image ROCm 6.3.3 | runs (Bi-Se cells converge) | dies at once: `rocBLAS error ... hipErrorInvalidDeviceFunction` in `hipblasDsyrkx` |
| host ROCm 6.3.1 (its build ROCm) | runs | converges (11 SCF steps, same energy as outside the container) |

`HSA_XNACK` makes no difference. Each solver calls different GPU library routines, so
"one mode works" says nothing about the other.

**Where it's set: one place.** `launch_flux.slurm`'s `RMG_ROCM` (default
`/opt/rocm-6.3.1`) is bind-mounted into the container and used — via the environment —
for the ROCm entry of `run_pipeline.py`'s `RMG_LD_LIBRARY_PATH`, which only the
`rmg-gpu` processes see. After building against another ROCm module, run with
`RMG_ROCM=/opt/rocm-<version>`. Cray PE libraries come from the `/opt/cray` bind (the
host's own); if your build uses other module libraries (e.g. a different boost from
`/sw/frontier`), add their directories to `RMG_LD_LIBRARY_PATH` — list what the binary
needs with `module load <your build modules>; ldd <rmg-gpu>`.

**Symptom → cause** (all seen while setting this example up):

| Symptom | Cause | Fix |
|---|---|---|
| `hipErrorInvalidDeviceFunction` from rocBLAS/hipBLAS | ROCm mismatch | `RMG_ROCM` = the ROCm `rmg-gpu` was built with |
| `error while loading shared libraries: ... cannot open shared object file` | library not bound / not on the path | add the bind (`launch_flux.slurm`) and directory (`RMG_LD_LIBRARY_PATH`) |
| `HSA exception: hsaKmtSVMSetAttr failed` then `hipHostMalloc failed`, exit 137 | out of GPU memory: too few GPUs for the cell (happens natively too) | more ranks/nodes (`max_tasks_per_job`, `allocated_nodes`) or smaller cells |
| `MPI_Init` abort `Unable to use a NIC_POLICY of 'NUMA'` (>1 node) | NIC policy | keep `MPICH_OFI_NIC_POLICY=BLOCK` (section 9) |
| `rmg-gpu` hangs at start with an empty log (>1 node) | HIP sees no GPUs after a multi-node `MPI_Init` | keep `FI_HMEM=system` (section 9) |

A 4-line `HSA exception: hsaKmtSVMSetAttr failed.` per rank *without* a following error
is harmless (seen with the image's ROCm; speed and results unaffected).

**Check after any rebuild** (a couple of minutes, one debug node): run the isolated Bi
atom through `converge_dft_data` once with each solver (`kohn_sham_solver:` in a copy of
the recipe, `directory: DFT/isolated_elements/Bi`) and compare with the numbers below.

**Choosing the solver.** Both now work in the container, single- and multi-node:

| BiSe cells, 1 node, 8 ranks (11-12 atoms) | `davidson` | `multigrid` |
|---|---|---|
| energies | — | same as davidson to < 0.003 meV |
| time per single point | 40-66 s | 53-81 s (1.2-1.9x slower) |
| isolated Bi | does not converge (energy oscillates ~3 Ha) | 11 SCF steps, 62 s (1 node) / 41 s (2 nodes) |
| isolated Se | 10 SCF steps, -256.198 eV | 7 SCF steps, **-253.766 eV — 2.43 eV higher** |

Davidson is faster for small cells like these; multigrid's near-linear scaling is what
pays off for the large systems RMG is meant for (hundreds to thousands of atoms on
large allocations) — the crossover depends on the system and wasn't measured here.
**Isolated atoms need care with either:** on Se, multigrid settled into the
higher-energy state Davidson passes through in its first steps (-9.3257 Ha) instead of
Davidson's -9.4151 Ha, so a multigrid E0 can be wrong while looking converged — and the
Bi E0 currently has only multigrid values to go on. Check isolated-atom energies before
using them as MACE E0s.

## 4. Launch: `launch_flux.slurm`

From this directory (config paths are relative to it):

```bash
sbatch launch_flux.slurm                                    # --stage all, workflow_config.yaml
CONFIG=my_config.yaml STAGE=converge_dft_data sbatch -N 2 launch_flux.slurm
FLUX_CMD="flux resource list" sbatch -N 2 launch_flux.slurm # check the Flux instance only
```

It also works as `bash launch_flux.slurm` inside an `salloc`. Edit its `#SBATCH`
lines (account, QOS, time, nodes) and its `SIF`/`ENSEMBLE_FF_FIT`/`PYRMG` defaults.
What it does:

```bash
srun -N <nodes> -n <nodes> --external-launcher --gpu-bind=closest --mpi=pmi2 --cpu-bind=none \
    apptainer exec --bind $RMG_ROCM/:$RMG_ROCM --bind /opt/cray/:/opt/cray --bind /usr/lib64/:/opt/cray_extra/lib64 \
                   --bind /var/spool/slurmd <sif> \
    flux start bash <initial program>
```

- That `srun` line is MatEnsemble's own Frontier recipe (`matensemble run`): one Flux
  broker per node, bootstrapped into one instance over Slurm's PMI2. `--mpi=pmi2` is
  only for that bootstrap (the site default, `cray_shasta`, isn't something Flux's
  broker speaks) — it says nothing about how the chores' own MPI jobs get wired.
  Commands must go through `srun`: run directly in an `salloc` shell they land in
  Slurm's extern step, with a minimal (often single-CPU) cpuset. `--cpu-bind=none`
  matters: with Slurm's default binding the instance only gets cores 10-63 of each
  node, and multi-node RMG runs ~25% slower (see step 5).
- The initial program prints `flux resource list` (expect N nodes, 56N cores, 8N
  GPUs) and then runs `run_pipeline.py` with `PYTHONPATH` set to your checkouts —
  scoped to that one command rather than the whole container.
- Binds: `$RMG_ROCM` (default `/opt/rocm-6.3.1`) is the host ROCm `rmg-gpu` was built
  against (step 3b); `/opt/cray` carries the Cray PE libraries `rmg-gpu` links (MPICH, FFTW, HDF5,
  LibSci, libfabric, PALS); `/usr/lib64` carries `libcxi.so.1` (Slingshot) and
  `libnl-3.so.200`; `/var/spool/slurmd` is what OLCF's own `apptainer-enable-mpi`
  module binds. `/lustre/orion` and `/sw/frontier` (Spack boost/bzip2) are already in
  Frontier's default Apptainer bind list. Keep the trailing slashes — binds were
  unreliable without them. The `/opt/cray_extra/lib64` mount point is created on the
  fly (Frontier's Apptainer has underlay enabled), so the `.sif` needs no changes.
- **Library path and OpenMP env are per chore, not global.** `run_pipeline.py`'s
  `rmg_container_env` gives only the `rmg-gpu` chores `RMG_LD_LIBRARY_PATH` (Cray
  dirs, then the container's own multiarch/ROCm/xpmem dirs, then Spack boost/bzip2,
  then the host `/usr/lib64` bind last so it only fills genuine gaps) and
  `OMP_PROC_BIND=false`. Setting either for the whole container broke `flux start`
  and `fi_info` in testing. A benign warning,
  `libcxi.so.1: libnl-3.so.200: no version information available`, is expected.
- `OMP_PROC_BIND=false` fixes a real bug: with `spread`/`threads`, GOMP collapses
  `rmg-gpu` onto a single CPU shortly after start (~240 s vs ~35 s per SCF step).
  `OMP_PLACES` must be unset rather than empty (an empty value crashes libgomp).

## 5. Configure `workflow_config.yaml`

Key knobs:

- `converge_dft_data.max_tasks_per_job` — RMG runs one MPI rank per GPU, 8 per
  node, so this caps the GPUs (and `ceil(ranks / 8)` nodes) any one RMG job may use;
  pyRMG's own estimate is clamped to it. **RMG jobs span nodes** (16 = 2 nodes in the
  default config); keep it within your allocation. Measured: a 25-atom Bi-Se single
  point converges in 269 s on 1 node and 173 s on 2, with energies agreeing to 1e-6
  eV; a 99-atom cell needs at least 2 nodes (on 1 it runs out of GPU memory).
- `converge_dft_data.cores_per_task: 7` (+ `controller_cores: 0`) — one whole
  7-core L3 group per rank, the layout native `srun` uses; `rmg_dft.py` then gives
  each rank the GCD closest to its cores. Other values straddle L3/NUMA groups and
  run ~25% slower (or put two ranks on one GCD).
- `fine_tuning.num_tasks` / `MD_*.num_tasks` — these stages are single-task/single-GPU
  by design; only RMG is multi-GPU.

**One-time setup: fetch the MACE foundation model.** MACE-OMat-0 (medium):

```bash
curl -L -o model.model \
    https://github.com/ACEsuit/mace-foundations/releases/download/mace_omat_0/mace-omat-0-medium.model
cp model.model FF/generation_0/mace-omat-medium/model.model                      # reference copy
cp model.model MD/finite_temperature/uq/generation_0/mace-omat-medium/model.model  # FT-MD model
mkdir -p FF/generation_1/mace-omat-medium
cp model.model FF/generation_1/mace-omat-medium/model.model                      # loop-1 fine-tuning seed
```

Keep `FF/generation_1/mace-omat-medium/` free of any other `model.model`:
`FFMatEnsemble` cross-products *every* `model.model` it finds under
`fine_tuning.run_directory` against every input folder. The pipeline deliberately
keeps separate physical copies rather than symlinks.

The MACE-fitting logic lives in `FF/mace_fit.py`, pointed at by
`fine_tuning.ff_task`/`entry_point` — `FFMatEnsemble` itself is a generic dispatcher
(same pattern as `converge_dft_data`'s `dft_task`/`rmg_dft.py`). For a different
foundation model, swap the URL above and update `fine_tuning.foundation_model`/
`finite_temperature_md.foundation_model` if the filename changes.

## 6. Run one loop — smoke-test first

`--stage all` runs every stage in order:

```bash
python run_pipeline.py --config workflow_config.yaml --stage converge_dft_data
python run_pipeline.py --config workflow_config.yaml --stage sample_ft_md_structures
python run_pipeline.py --config workflow_config.yaml --stage build_ff_inputs
python run_pipeline.py --config workflow_config.yaml --stage fit_and_validate
python run_pipeline.py --config workflow_config.yaml --stage downselect
python run_pipeline.py --config workflow_config.yaml --stage uq_single_points
python run_pipeline.py --config workflow_config.yaml --stage select_dft_candidates
```

(each as `STAGE=<name> sbatch launch_flux.slurm`). `select_dft_candidates` writes the
next round's DFT inputs to `DFT/uq/1`. Rerunning `--stage all` resumes after a
walltime limit: finished RMG runs, fits, MD runs and single points are skipped.

**Smoke test** — identical wiring, 12 small structures, a few minutes per stage:

```bash
bash make_smoke_copy.sh /lustre/orion/<proj>/scratch/$USER/smoke /path/to/rmgdft
cd /lustre/orion/<proj>/scratch/$USER/smoke
CONFIG=workflow_config.smoke.yaml sbatch launch_flux.slurm
```

`make_smoke_copy.sh` copies this directory with only both isolated atoms, six 11-atom
BiSe vacancies and four 12-atom BiSe EoS points under `DFT/`, fills in
`vdW_quench.yml`'s RMG paths, and places the foundation model.
`workflow_config.smoke.yaml` differs from `workflow_config.yaml` only in the keys its
header lists (3 fits × 2 epochs, 4 short MD runs, 2 models kept, 2 candidates,
`cores_per_task: 6`).

**Measured (smoke test, 1 node, 2026-10-23):** each 11-12 atom RMG single point took
40-66 s (8 GPUs, ~1.9 s per SCF step); MACE fitting, validation, FT-MD, UQ single
points and candidate selection then took about 4.5 min, and a rerun of `--stage all`
skipped every finished DFT run, fit and MD run. Known gaps: the isolated Bi atom
doesn't converge with `davidson` (pitfalls — pass `CONVERGED_BI=` to
`make_smoke_copy.sh`), and the loop-2 RMG run on the new candidates wasn't completed
in the smoke test (with the default `max_supercell_atoms: 100` they are ~99-atom
cells, which need 2+ nodes — see step 5).

## 7. The next loop

Run `--stage all` again after updating `workflow_config.yaml` for the new generation.
Every generation-specific location is tagged `# GEN` in the config; for loop 2:

| Key | Loop 1 | Loop 2 |
|---|---|---|
| `build_ff_inputs.extra_training_subpaths` | `[]` | `[uq/1]` (converts loop 1's new RMG runs into a `uq_1` training label) |
| `build_ff_inputs.always_include_labels` | `[]` | `[uq_1]` (every ensemble member trains on it, instead of it being sampled like any other label) |
| `fine_tuning.run_directory` | `FF/generation_1/mace-omat-medium` | `FF/generation_2/mace-omat-medium` (seed it with a `model.model` first, as in step 5) |
| `fine_tuning.inputs_directory` | `FF/generation_1/mace_inputs` | `FF/generation_2/mace_inputs` |
| `copy_force_fields.target_directory` | `MD/single_points/validation/generation_1/force_fields` | `.../generation_2/...` |
| `finite_temperature_md.structures_subpath` | `structures/generation_1` | `structures/generation_2` |
| `finite_temperature_md.output_directory` | `MD/finite_temperature/uq/generation_1/...` | `.../generation_2/...` |
| `MD_uq_single_points.run_directory` | `MD/single_points/uq/generation_1/force_fields` | `.../generation_2/...` |
| `MD_uq_single_points.inputs_directory` | `MD/single_points/uq/generation_1/ase_inputs` | `.../generation_2/...` |
| `dft_candidate_selection.dest_root` | `DFT/uq/1` | `DFT/uq/2` |

The rerun's first stage, `converge_dft_data`, runs RMG on the `DFT/uq/1` POSCARs
(everything already converged is skipped); `build_ff_inputs` then folds those results
in. Optionally also change `sample_ft_md_structures.seed` (different MD starting
structures) or point `finite_temperature_md.foundation_model` at a loop-1 fitted
model. Downstream path keys left `null` default to these, so nothing else needs
editing.

## 8. Pitfalls encountered getting this working (read before deviating)

- **The published `frontier-v0.5.8` image is not MatEnsemble 0.5.8, and has no
  torch/MACE.** Its `/opt/matensemble` is a 2026-07-08 checkout (reports 0.5.3;
  `Pipeline` has no `reserve_broker_node`), and torch, mace-torch and
  openequivariance aren't installed at all. The derived image (step 1c) fixes both;
  check `build_image.slurm`'s version printout whenever you change tags.
- **Two copies of the Flux Python bindings in the image.** The pip `flux-python` in
  `/opt/basic` lacks `_flux._rlist`, which `flux.resource` (and MatEnsemble's
  manager) needs; flux-core's own install in `/usr/local/lib/python3.12/site-packages`
  is complete. The derived image keeps only the complete one.
- **Chores run in their own working directory** (`matensemble_workflow-*/out/chore-*/`,
  not where you launched). Any path a chore or `Pipeline.strategy` resolves itself must
  be absolute — `run_fit_and_validate` makes `copy_and_spawn_md`'s paths absolute for
  this reason. Each chore's `stdout`/`stderr` are in that directory; the overall log is
  `matensemble_workflow-*/matensemble_workflow.log`.
- **Failed chores don't raise out of `future.result()`.** `converge_dft_data` checks
  every run's `properties.json` afterwards (warns per failure, raises if none
  converged) and `downselect` stops with a clear error on an empty validation tree.
- **A chore's `Resources.env['PYTHONPATH']` replaces the inherited value.**
  `chore_pythonpath()` keeps the launcher's `PYTHONPATH` after the EnsembleFFFit root —
  otherwise pyRMG (on `PYTHONPATH`, not in the image) would be unimportable in RMG
  chores.
- **Every `Pipeline` uses `reserve_broker_node=False`** (no node set aside for just the
  broker) — the Perlmutter convention.
- **mace-torch ≥ 0.3.15 for `freeze`** — `freeze: 6` (0.3.16) trains only the
  readouts. The derived image pins 0.3.16 and fails the build otherwise.
- **`HSA exception: hsaKmtSVMSetAttr failed` / `hipHostMalloc failed` means RMG ran out
  of memory** — a job given too few GPUs for its structure. Confirmed 2026-10-23: a
  99-atom Bi-Se cell on one node's 8 GPUs fails this way natively too (no container, no
  Flux); pyRMG's estimate for it was 14 nodes, clamped by `max_tasks_per_job`. Keep DFT
  candidates within what your `max_tasks_per_job` can hold (`max_supercell_atoms` in the
  FT-MD input caps their size), or allow multi-node RMG jobs.
- **Multigrid used to fail in the container** (`hipErrorInvalidDeviceFunction`) and was
  blamed on multigrid itself; the cause was running `rmg-gpu` against the image's ROCm
  instead of the one it was built with (step 3b). Both solvers work now; the recipe
  default stays `davidson`, faster for this example's small cells.
- **Isolated atoms:** Bi doesn't converge with `davidson`; with `multigrid` it does,
  but multigrid put isolated Se 2.43 eV above Davidson's (lower, so better) solution —
  step 3b. `converge_dft_data` uses one recipe for its whole `directory`, so to run the
  isolated atoms with a different solver, run it separately on
  `DFT/isolated_elements` with its own recipe copy (or pass `CONVERGED_BI=` to
  `make_smoke_copy.sh` for the smoke test).
- **`rmg_dft.py`'s subprocess call must use `shell=True`** and `close_fds=False` (the
  latter keeps Flux's PMI file descriptor). `shell=False` was confirmed, in a
  controlled A/B, to be an actual regression here.
- **RMG's `command` is a bare `{rmg_executable} {rmg_name}`** — no `srun`/`mpirun`/
  `flux run` wrapper. Flux owns launch semantics for every chore.
- **Multi-node RMG needs `MPICH_OFI_NIC_POLICY=BLOCK`, `FI_HMEM=system`, and the
  CPU/GPU layout above** — all already set; don't change them without reading
  section 9. Symptoms if they're lost: `MPI_Init` aborts on >1 node (NUMA policy), or
  `rmg-gpu` hangs at startup with an empty log because HIP sees no GPUs (FI_HMEM), or
  runs ~25-100% slower (layout). `Resources(mpi=False)` is correct.
- **MACE members with fewer training configs than `batch_size`** (common with small
  label combinations) used to die with `torch.cat(): expected a non-empty list of
  Tensors`; `FF/mace_fit.py` now shrinks `batch_size`/`valid_batch_size` to fit and
  says so in the chore's stderr.
- **`max_tasks_per_job` clamping uses ceiling, not floor, division** when computing a
  fallback node count — floor division can produce `Resources(num_tasks=0, ...)`.
- **A stray second `model.model` under `fine_tuning.run_directory` causes unwanted
  extra re-fits** — `FFMatEnsemble` cross-products every match against all input
  folders.
- **`MD_uq_single_points.parent_levels: 8`** matches
  `structures/Defects/<chemsys>/<formula>/<mp-id>/<defect_type>/<defect>/<frame>`; a
  different tree shape needs a different value, or models get merged into one chore.
- **Image pulls through the OLCF proxy occasionally fail** with
  `stream error ... PROTOCOL_ERROR`; `build_image.slurm` retries.

## 9. Multi-node RMG: what it takes, and how it was established

October 2026, on this branch. Kept here so the reasoning survives; the scripts used are
in git history (`mpi_tests/`, `INVESTIGATION_NOTES.md`).

**How a multi-rank RMG chore runs.** With `Resources(num_tasks=N, ...)`, Flux starts
N copies of the chore's Python function, one per rank; each copy of `DFT/rmg_dft.py`
writes the same inputs and launches its own `rmg-gpu`, and Flux's PMI joins those N
processes into one MPI job. So the driver's stdout appears N times per chore (in
`matensemble_workflow-*/out/chore-run_rmg-*/stdout`), but RMG prints one banner and
`N MPI processes/image`. Flux puts its own `libpmi` first on each task's library path,
so Cray MPICH needs no PMI changes (unlike VASP on Perlmutter) and `mpi=False` is fine.

**What was needed, bottom-up** (each found with a minimal MPI test built against
`rmg-gpu`'s own Cray MPICH, then confirmed with RMG):

| Symptom (2 nodes, inside the container) | Cause | Fix (where) |
|---|---|---|
| `MPI_Init` aborts on every rank: `Unable to use a NIC_POLICY of 'NUMA'. Rank N is not confined to a single NUMA node` (1 node always worked) | Flux tasks aren't bound to one NUMA domain | `MPICH_OFI_NIC_POLICY=BLOCK` (`run_pipeline.py`, `rmg_container_env`) |
| Ranks join, but HIP sees **0 GPUs** after `MPI_Init`; `rmg-gpu` hangs with an empty log; with GPU-aware MPI, `hsa_init ... HSA_STATUS_ERROR_OUT_OF_RESOURCES` | libfabric initialises its ROCm memory support during an inter-node `MPI_Init` (querying GPUs *before* `MPI_Init`, or `FI_HMEM=system`, both fix it) | `FI_HMEM=system` (same place); RMG uses host-buffer MPI |
| Works, but 1.6-2x slower per SCF step than native `srun` | the Flux instance lacked cores 1-9 of each node, 6-core slices straddled L3/NUMA groups, GPUs handed out in index order (most ranks on a far GCD) | brokers with `srun --cpu-bind=none` (`launch_flux.slurm`); `cores_per_task: 7` + `controller_cores: 0` (config); closest GCD per rank (`rmg_dft.py`, `RMG_GPU_CLOSEST_GCDS`) |

**Measured** (99-atom Bi54Se45 single point, davidson, 16 ranks on 2 nodes, same nodes):
native `srun` 86 s first step / 34.9 s per step after; container through Flux, with the
fixes, 87 s / 35.0 s; without the layout fixes 101-136 s / 42-61 s. A 25-atom Bi9Se16
single point, fully converged through `converge_dft_data`:

| | 1 node (8 ranks) | 2 nodes (16 ranks) |
|---|---|---|
| SCF steps | 26 | 29 |
| RMG total | 269 s | 173 s |
| per step | 10.4 s | 6.0 s |
| energy | -23172.17999939 eV | -23172.17999858 eV |

(energies within 1e-6 eV, forces within 2e-3 eV/A, stress within 5e-6).

**Ruled out along the way:** locked-memory limits (unlimited in Flux tasks),
`HSA_XNACK` (the image sets 1; 0 or unset changes nothing), each rank seeing all 8
GPUs (restricting it changes neither correctness nor speed), Slurm/PALS variables
leaking into the container (none reach Flux tasks), the container's ROCm version for the
HSA crash (that crash is RMG running out of GPU memory — it happens natively too).

**Still open:** why libfabric's ROCm initialisation only breaks HIP inside the
container (host vs container ROCm runtime is the obvious difference); pyRMG's node
estimate is far above what fits (14 nodes for the 99-atom cell, which runs on 2);
each rank logs `HSA exception: hsaKmtSVMSetAttr failed.` 4 times in the container
(harmless to speed or results). Multigrid in the container: works with the host ROCm
(step 3b); 2 nodes x 8 ranks via `converge_dft_data` converged the isolated Bi atom in
11 SCF steps / 41 s, the same energy as the native multigrid reference (36 s).
Also unchecked: that `rmg_dft.py`'s `stress_voigt` (taken from the ASE calculator, so
ASE's sign convention) matches the training data's once loop-2 candidates converge.

