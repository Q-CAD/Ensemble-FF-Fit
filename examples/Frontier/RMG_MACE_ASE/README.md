# RMG -> MACE -> ASE: an active-learning pipeline on Frontier

This example reproduces a full active-learning loop on OLCF Frontier, coordinated by
a single script/config pair (`run_pipeline.py` / `workflow_config.yaml`) via
MatEnsemble/Flux:

1. **RMG** DFT convergence on a tree of starting structures (bulk relaxation,
   defects, equation-of-state points, isolated atoms).
2. **MACE** force-field fitting (an ensemble of fine-tuned models, seeded from a
   foundation model) and validation against the DFT ground truth.
3. **ASE** finite-temperature MD sampling with the fitted/foundation models, then
   single-point re-evaluation of sampled frames across the whole ensemble.
4. Uncertainty-based downselection of the most-disagreed-upon frames as the next
   round's DFT candidates.

Everything below assumes Frontier (OLCF), an Apptainer container, and the AMD
ROCm GPU stack. Commands/paths are written as concrete examples from a real run;
substitute your own project/user paths throughout.

## 0. What's included vs. what you provide

This directory is a trimmed copy of a real, working run: only `POSCAR` structure
files and driver/recipe scripts are kept — all RMG/MACE/MD *output* (converged
energies/forces, fitted model checkpoints, trajectories, single-point results)
has been stripped out, and the MACE foundation-model checkpoint itself
(`model.model`, ~80MB) isn't committed to the repo at all — see step 5 for how to
fetch it. Reproducing the pipeline means actually re-running every stage below,
not resuming from pre-computed results.

## 1. Build/obtain the Apptainer container

The container image is pulled and unpacked on a Frontier NVMe-backed node (much
faster than doing it directly on Lustre), then the finished sandbox is moved to
Lustre scratch for actual use.

**a. Pull the `.sif` image on NVMe**, then copy it back to scratch:

```bash
#SBATCH -C nvme   # request an NVMe-backed node
NVME_DIR=/mnt/bb/$USER
mkdir -p "$NVME_DIR/apptainer_cache" "$NVME_DIR/apptainer_tmp"
export APPTAINER_CACHEDIR="$NVME_DIR/apptainer_cache"
export APPTAINER_TMPDIR="$NVME_DIR/apptainer_tmp"

cd "$NVME_DIR"
apptainer build matensemble.sif docker://ghcr.io/freddude2004/matensemble:frontier-v0.5.5
cp matensemble.sif /path/to/your/scratch/
```

**b. Build a writable sandbox from the `.sif`**, also on NVMe, then `rsync` it
back to scratch (capped parallelism avoids saturating the filesystem — safe to
re-run if it runs out of walltime, since `rsync` skips files that already match):

```bash
cd "$NVME_DIR"
apptainer build --sandbox matensemble_sandbox matensemble.sif

MAX_PARALLEL=6
find matensemble_sandbox -maxdepth 1 -mindepth 1 \( -type d -o -type f \) -print0 \
    | xargs -0 -n1 -P "$MAX_PARALLEL" -I{} \
      rsync -a --partial {} /path/to/your/scratch/matensemble_sandbox/
```

You now have a writable `matensemble_sandbox/` directory on scratch — this is
the `<sandbox>` referenced throughout the rest of this doc.

## 2. Install Ensemble-FF-Fit inside the sandbox

From inside an interactive shell in the sandbox (or via `apptainer exec ... bash`),
with this repo bound in (see the bind flags in step 4):

```bash
python install_gpu_torch.py          # detects ROCm, installs the matching torch build
pip install -e ".[mace,rocm]"        # MACE + ROCm-accelerated equivariant kernels
```

See the main repo `README.md` for the full extras list if you need other backends
(`reaxff`, `torchsim`, `cuda` for non-AMD platforms).

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

Built against `rocm/6.3.1` (the newest ROCm module Frontier's native stack offers)
even though the container ships `rocm-6.3.3` — same minor series, HIP/HSA ABI and
GPU kernel code-object format stay compatible, confirmed working in practice.

Update `DFT/vdW_quench.yml`'s `rmg_executable`/`vdwdf_kernel_filepath` (both
currently `/path/to/...` placeholders) to point at your actual build and its
bundled `XC/vdW_kernel_table`.

## 4. Launching the container for runtime

**One-time sandbox setup** (needs `--fakeroot --writable`, only ever for this):

```bash
apptainer exec --fakeroot --writable <sandbox> mkdir -p /opt/cray_extra/lib64
```

**Every launch** needs these binds — note the trailing slash on every source path,
binds are unreliable without it:

```bash
--bind /opt/cray/:/opt/cray --bind /sw/frontier/:/sw/frontier --bind /usr/lib64/:/opt/cray_extra/lib64
```

`/opt/cray` and `/sw/frontier` carry the real Cray PE libraries (MPICH, FFTW,
HDF5, LibSci, libfabric, pals) and the Spack-built boost/bzip2 RMG needs.
`/usr/lib64` carries `libcxi.so.1` (Slingshot NIC driver) and its dependency
`libnl-3.so.200`. The container's own ROCm/xpmem are used as-is (not bound).

**`LD_LIBRARY_PATH`, in this exact order** (container's own dirs first, so they
aren't shadowed by the host binds; host bind last, so it only fills genuine gaps):

```bash
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu"
export LD_LIBRARY_PATH="/opt/rocm-6.3.3/lib:/opt/xpmem/2.7.4/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="/sw/frontier/spack-envs/cpe24.03-cpu/opt/gcc-13.2/boost-1.85.0-3gvl6ws5xm7hrfkeyevgh45bv434ceol/lib:/sw/frontier/spack-envs/base/opt/linux-sles15-x86_64/gcc-7.5.0/bzip2-1.0.8-st7di5r4yikef76nw4xenvocycgp3god/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="/opt/cray/pe/lib64:/opt/cray/libfabric/2.3.1/lib64:/opt/cray/pals/1.8/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/opt/cray_extra/lib64"
```

(Adjust the versioned subpaths — `rocm-6.3.3`, `boost-1.85.0-...`, `libfabric/2.3.1`
— to whatever your container/module versions actually are.) A benign warning,
`libcxi.so.1: libnl-3.so.200: no version information available`, is expected and
harmless (the two builds differ in symbol-versioning style; the run proceeds
normally).

**OpenMP affinity env vars** (every launch) — without these, GOMP silently
collapses the process to a single CPU shortly after start regardless of what the
scheduler assigned, which was previously mistaken for library/ABI trouble but is
actually a separate affinity bug:

```bash
--env OMP_PROC_BIND=false --env OMP_PLACES= --env OMP_NUM_THREADS=7
```

**Must launch via `srun`, not a bare shell inside `salloc`** — commands run
directly in a `salloc` shell execute in Slurm's "extern" step, which only gets a
minimal (often single-CPU) cpuset regardless of the real allocation.

**`srun`/`flux start` flags**:

```bash
--ntasks=1 --cpus-per-task=7 --gpus-per-task=1 --cpu-bind=sockets --threads-per-core=1 --gpu-bind=closest --mpi=pmi2
```

`--mpi=pmi2` here is required for `srun` to bootstrap the *Flux broker* itself
(this MPICH build's PMI client doesn't speak Slurm's default `cray_shasta` MPI
plugin) — it is a different concern from anything MatEnsemble/Flux configures
internally once the broker is up (see the pitfalls list below).

**Putting it together** — start a Flux instance inside the container, on your
allocation:

```bash
env -u OMP_PLACES srun -N <nodes> -n <nodes> --external-launcher --mpi=pmi2 --pty \
    apptainer exec --bind /opt/cray/:/opt/cray --bind /sw/frontier/:/sw/frontier \
    --bind /usr/lib64/:/opt/cray_extra/lib64 <sandbox> flux start
```

From inside that Flux instance, run `python run_pipeline.py ...` (step 6) directly
— MatEnsemble/`fluxlet.py` submits every RMG/MACE/MD chore against this same
ambient Flux instance, you don't need to wrap each stage in its own `flux run`.

## 5. Configure `workflow_config.yaml`

All paths in the copied `workflow_config.yaml` are relative to this directory
(`examples/Frontier/RMG_MACE_ASE`) — run every `run_pipeline.py` command from here
so they resolve correctly. Key knobs to check for your allocation:

- `converge_dft_data.gpus_per_node` / `cores_per_task` / `max_tasks_per_job` — size
  these to your node's real GPU count and how many nodes you've allocated.
- `fine_tuning.num_tasks` / `MD_*.num_tasks` etc. — these MD/fitting stages are
  single-task/single-GPU by design; only the RMG stage is the multi-GPU one.

**One-time setup: fetch the MACE foundation model.** This example uses
MACE-OMat-0 (medium), downloaded directly from its GitHub release rather than
committed to this repo (keeps repo size sane — the checkpoint is ~80MB):

```bash
curl -L -o model.model \
    https://github.com/ACEsuit/mace-foundations/releases/download/mace_omat_0/mace-omat-0-medium.model
```

Place a copy at **both** of these paths — the pipeline deliberately keeps two
physical copies (see pitfalls below for why a shared/symlinked copy causes
problems):

```bash
cp model.model FF/generation_0/mace-omat-medium/model.model
cp model.model MD/finite_temperature/uq/generation_0/mace-omat-medium/model.model
```

Then, before running `fit_and_validate`, create `FF/generation_1/mace-omat-medium/`
and seed it with a third copy:

```bash
mkdir -p FF/generation_1/mace-omat-medium
cp model.model FF/generation_1/mace-omat-medium/model.model
```

`fine_tuning.run_directory` (`FF/generation_1/mace-omat-medium`) is where
`MACEMatEnsemble` looks for a seed model to fine-tune from; keep this copy
strictly under `generation_1/`, not alongside `generation_0`'s copy, or a stray
second `model.model` can trigger unwanted extra re-fits (see pitfalls below).

If you're fitting against a different foundation model instead, swap the
download URL above and update `fine_tuning.foundation_model`/
`finite_temperature_md.foundation_model` in `workflow_config.yaml` if the
filename changes.

## 6. Run the pipeline, stage by stage

Each stage is its own compute allocation/submission — run them in order (or all
in one process via `--stage all`, if your allocation covers every stage's needs):

```bash
python run_pipeline.py --config workflow_config.yaml --stage converge_dft_data
python run_pipeline.py --config workflow_config.yaml --stage sample_ft_md_structures
python run_pipeline.py --config workflow_config.yaml --stage build_ff_inputs
python run_pipeline.py --config workflow_config.yaml --stage fit_and_validate
python run_pipeline.py --config workflow_config.yaml --stage downselect
python run_pipeline.py --config workflow_config.yaml --stage uq_single_points
python run_pipeline.py --config workflow_config.yaml --stage select_dft_candidates
```

`select_dft_candidates` writes the next round's DFT inputs to
`DFT/uq/1` (kept as an empty directory in this example) — feed those back into
`converge_dft_data` to iterate the active-learning loop.

## 7. Pitfalls encountered getting this working (read before deviating)

- **RMG's `kohn_sham_solver` must be `"davidson"`, not `"multigrid"`, under this
  Flux/Apptainer setup.** `multigrid`'s real-space domain decomposition needs
  working halo-exchange communication between GPU domains; on this stack it fails
  with `HSA exception: hsaKmtSVMSetAttr failed` / `hipErrorInvalidDeviceFunction`
  errors, even at a single MPI task. This was mistaken for a launch-mechanism bug
  for a long time before being traced to the solver choice itself — if you ever
  need `multigrid` (e.g. it's what actually converges some difficult systems
  outside this container), expect to debug GPU cross-device communication
  specifically, not the launch/env layer described above.
- **`rmg_dft.py`'s subprocess call must use `shell=True`.** `shell=False` (with
  `shlex.split`) looks safer on paper (one fewer process hop) but was confirmed,
  in a controlled A/B against an equivalent working script, to be an actual
  regression here — not just a style choice.
- **RMG's own `command` should be a bare `{rmg_executable} {rmg_name}` invocation
  — no `srun`/`mpirun`/`flux run` wrapper.** Flux owns launch semantics for every
  chore; wrapping the command again on top of that conflicts with what the outer
  Flux chore already set up.
- **`Resources(mpi=...)` for the RMG chore should stay at its default (`False`).**
  It's tempting to set `mpi=True` (forcing Flux's `-o mpi=pmi2` shell option) by
  analogy with the `--mpi=pmi2` `srun` flag needed to bootstrap the Flux broker
  itself (step 4) — but that flag is for a completely different concern (getting
  `srun` to launch the *broker*), and forcing `pmi2` again *inside* an already-
  running Flux instance was tried and confirmed to not fix anything.
  `set_gpu_affinity` on `pipe.submit()` for this chore should likewise stay
  unset/`False`.
- **`max_tasks_per_job` clamping uses ceiling, not floor, division** when
  computing a fallback node count — floor division can produce
  `Resources(num_tasks=0, ...)`, which is invalid.
- **A stray second `model.model` under `fine_tuning.run_directory`'s ancestor
  directory causes unwanted extra re-fits** — `MACEMatEnsemble` recursively
  searches for *any* file named `foundation_model` and cross-products every match
  against all `mace_inputs` folders. Keep the one seed model's directory
  (`FF/generation_1/mace-omat-medium`) free of unrelated copies.
