# RMG_MACE_ASE on Frontier: investigation log

Running log of what was tried, what happened, and what it rules out, so the
work can be resumed later. Newest entries at the bottom of each section.
Confirmed findings also go into README.md's pitfalls; this file keeps the
dead ends and raw evidence too. Plan: `UPDATE_FROM_VASP_MACE_ASE.md` plus the
top-level `UPDATE_PLAN.md` it was derived from (outside the repo).

Branch: `frontier-rmg-update` (off `efffit_2`). Slurm account: `mat201`
(`lrn090` still shows in `sacctmgr` but is rejected at submit time,
2026-10-23).

## Starting state (2026-10-23, read-only survey)

- **The example was broken as committed.** Commit `21677e0` (2026-09-11)
  renamed `DFTMatEnsemble`'s recipe option to `dft_recipe` (required by
  `build_dft_dcts`, read by `run_individual`). This example still passed
  `rmg_yaml`, so `converge_dft_data` would raise before submitting anything.
- Old sandbox (`MatEnsemble2/matensemble_sandbox`, tag `frontier-v0.5.5`):
  MatEnsemble metadata says **0.5.3** (git 2026-07-08), but its `fluxlet.py`
  already sets `jobspec.cwd = chore.workdir` -- so the "chores run in their
  own directory" bug was live there too, not latent. mace-torch 0.3.16,
  torch 2.9.1+rocm6.3, openequivariance 0.6.8, flux-core 0.70.0, vanilla
  MPICH 3.4.3 at `/opt/mpich/3.4.3` (meant to be swapped for host Cray MPICH
  at runtime -- same `libmpi.so.12` soname), ROCm 6.3.3.
- Flux's own PMI client libraries are in the image at
  `/usr/local/lib/flux/libpmi{,2}.so.0` -- the directory Perlmutter's VASP fix
  put first on the binary's library path.
- **Earlier pipeline runs with `mpi=False` did produce multi-rank RMG jobs**:
  `MatEnsemble2/test/Consolidated_Pipeline/DFT/isolated_elements/Se` logged
  "8 MPI processes/image", `.../Bi` logged "Total PEs: 16" and completed.
  So Cray PMI *can* join ranks under Flux's default PMI. Unknown: whether
  those 16 ranks spanned 2 nodes or oversubscribed one.
- MatEnsemble's own Frontier recipe (`src/cli/matensemble-frontier`) is
  `srun -N n -n n --external-launcher --gpu-bind=closest --mpi=pmi2
  apptainer exec <sif> flux start ...`, after `module load
  olcf-container-tools apptainer-enable-mpi apptainer-enable-gpu`. Those
  modules bind `/opt/cray` and `/var/spool/slurmd` and put Cray MPICH's
  `lib-abi-mpich` first -- our hand-made binds are a subset of that.
- Host has `vasp/6.6.0-gpu`, `vasp/6.6.1-gpu`; user is in the `vasp` group.
  ROCm modules up to 7.x now exist (README said 6.3.1 was newest).
- Newer images published: `frontier-v0.5.6` ... `frontier-v0.5.9`. Chose
  `frontier-v0.5.8` to match Perlmutter's MatEnsemble 0.5.8 -- but see
  Phase 1: that tag doesn't actually contain MatEnsemble 0.5.8.

## Phase 1: example update

- **Image pull** (`build_image.slurm`, job 5744139): failed after 9.5 min,
  `stream error: stream ID 1; PROTOCOL_ERROR; received from peer` while
  fetching a layer through the OLCF proxy. Added a 4-attempt retry loop;
  the resubmission (5744210) succeeded first try in 8:56. Transient.
- **`frontier-v0.5.8` is not MatEnsemble 0.5.8.** Its `/opt/matensemble`
  is the same commit as the old v0.5.5 image (`6307e46`, 2026-07-08,
  reports 0.5.3); `Pipeline.__init__` takes only `basedir`/`registry`, so
  `Pipeline(reserve_broker_node=False)` would raise `TypeError`. The
  Frontier Dockerfile does `git clone --depth 1` of main, so the tags
  were presumably rebuilt from a cached layer. MatEnsemble main =
  `5d8be30` (2026-08-31, version 0.5.8 since `5371882`, 2026-08-18) -- what
  Perlmutter's Dockerfile.mace cloned on 2026-10-22.
- **The published image has no torch, mace-torch or openequivariance**
  (nor pymatgen-analysis-defects/mp-api/matminer). The old sandbox's copies
  (torch 2.9.1+rocm6.3, mace-torch 0.3.16, oeq 0.6.8) were installed into
  the writable sandbox by hand. So a derived image is required, not just a
  convenience: `matensemble-frontier.def` installs MatEnsemble@5d8be30,
  those three and the missing core deps, with import/version checks.
- Image otherwise: Python 3.12.3, flux-core/flux-python 0.70.0, ROCm 6.3.3,
  vanilla MPICH 3.4.3, mpi4py 4.1.2, LAMMPS python module importable,
  gcc 14.3.0, hipcc.
- Frontier's Apptainer (1.5.3) config already binds `/lustre/orion`,
  `/sw/frontier`, `/ccs/home`, and `/lib64:/host_lib64`, with
  `enable underlay = yes` -- so `--bind /usr/lib64/:/opt/cray_extra/lib64`
  works against a read-only .sif; the sandbox `mkdir` step is obsolete.
- **Chore PYTHONPATH** (found reading the code, not by a failure):
  `Resources.env['PYTHONPATH']` replaces the inherited value, and every
  chore set it to the Ensemble-FF-Fit root only. With pyRMG on PYTHONPATH
  instead of installed in the image, RMG chores could not have imported it.
  `chore_pythonpath()` appends the launcher's PYTHONPATH.
- **Derived build #1** (5744327, 4 min): every pip install succeeded, but
  the build-time check failed on `import flux.resource` --
  `No module named '_flux._rlist'`. The image's pip `flux-python` 0.70.0
  (in `/opt/basic`) is incomplete; flux-core's `make install` left a
  complete 0.70.0 set in `/usr/local/lib/python3.12/site-packages` (same
  split as the Perlmutter image). MatEnsemble's FluxManager imports
  `flux.resource.list`. Fix in the .def: uninstall the pip copy, add a
  `.pth` for the complete one. (The old sandbox had the same split and
  worked, presumably because `flux start` prepends its own python dir to
  the initial program's PYTHONPATH -- the build check runs outside Flux.)
- **Env check** (5744507, 1 node, 19 s): `flux resource list` = 1 node /
  56 cores / 8 GPUs (live hwloc discovery, no static R.json needed); derived
  image imports MatEnsemble 0.5.8, `flux.resource`, torch ROCm (8 devices),
  pyRMG and EnsembleFFFit from the launcher's PYTHONPATH; the
  `/usr/lib64 -> /opt/cray_extra/lib64` bind works on the read-only .sif.
  A plain `flux run -n8 -c6 -g1` (no affinity options) gives every task
  `CUDA_VISIBLE_DEVICES=0-7`, `ROCR_VISIBLE_DEVICES` unset, and the job's
  whole cpuset -- relevant for Phase 2's rank/GPU mapping.
- **Smoke run 1** (5744559, `--stage all`, smoke config): 10 of 12 RMG
  single points (Quench Electrons, davidson, 8 ranks / 8 GPUs, 6 cores
  each) converged in 41-66 s total, ~1.9 s per SCF step -- vs 37 s /
  1.6 s per step for the same BiSe vacancy0 in the old
  `MatEnsemble2/test/Consolidated_Pipeline` run. No slow-SCF regression.
  pyRMG's estimator wanted 40 ranks (5 nodes) even for these 11-12 atom
  cells (144 for volume_1.15); `max_tasks_per_job: 8` clamped all to 1 node.
- **Isolated Bi does not converge with davidson** (known case -- it was
  only ever converged with multigrid, outside the container). After 64 SCF
  steps the total energy swung between -77.6 and -80.5 Ha and twice reset
  to -77.632. Killed (`pkill` of the `rmg-gpu` processes on the node; the
  chore exited rc=1 and MatEnsemble moved on, no retry) and substituted the
  converged multigrid run from
  `MatEnsemble2/test/Full_Pipeline/run/DFT/isolated_elements/Bi` (same input
  apart from the solver and a 0.0013% larger box; E = -2112.4774 eV).
  The killed run's log/input are kept in `.../Bi/davidson_smoke1_killed/`.
  Implication: a fresh full run can't produce the Bi E0 with the current
  recipe -- either ship a converged Bi E0, use a per-structure recipe
  override for isolated atoms, or get multigrid working (Phase 2).
- Smoke run 1, rest of the loop: `sample_ft_md_structures` and
  `build_ff_inputs` fine (E0s: Bi -2112.4774 eV from the substituted run,
  Se -256.1982 eV). In `fit_and_validate` the 4 FT-MD chores finished, but
  all 3 MACE fits failed: `Batch size (10) is larger than the number of
  training data (5)`, surfacing as `torch.cat(): expected a non-empty list
  of Tensors` in `compute_avg_num_neighbors`. Smoke-only fix:
  `batch_size`/`valid_batch_size: 2` in `mace_overrides`. `downselect`'s new
  empty-tree check then stopped the run with its clear error, as intended.
  openequivariance JIT-builds its HIP extension on first use (no
  precompiled HIP build; torch < 2.10) into `~/.cache/torch_extensions/
  py312_cpu/libtorch_tp_jit` -- took ~20 s, shared across chores.
- Python's stdout was block-buffered into the Slurm file, so "Stage:"
  lines appeared only at exit; `launch_flux.slurm` now sets
  `PYTHONUNBUFFERED=1`.
- **Smoke run 2** (5744818, 4:36, resume after the batch-size fix):
  `converge_dft_data` skipped all 12 ("nothing new to run"), FT-MD skipped
  (4/4 done), 3 fits + validation single points ran, `downselect` ranked
  and kept 2 (`spread`), 88 MD frames unpacked, UQ single points, 2
  candidates written to `DFT/uq/1`. **Loop 1 complete.**
- The candidates are **99-atom** cells (Bi54Se45): `balanced_supercell
  (max_atoms=100)` (enabled in this branch, as in VASP_MACE_ASE) made a
  3x3 supercell of the 11-atom vacancy cell before MD.
- **Smoke run 3** (5744930, rerun of `--stage all`): only the 2 new
  `uq/1` POSCARs were submitted. Both RMG runs (davidson, 8 ranks, 1 node)
  died within ~1 min: `HSA exception: hsaKmtSVMSetAttr failed.` then
  `Error: hipHostMalloc failed. at LINE 1151 in .../RMG/Common/Kpoint.cpp`,
  exit 137; the stage then raised "all 2 RMG run(s) failed", as designed.
  **This is the same HSA error the README attributed to `multigrid` --
  here with davidson, so the solver can't be the whole story.** It is
  size-dependent (11-12 atom cells fine). Hypotheses: (a) a low
  RLIMIT_MEMLOCK inside Flux jobs limits pinned host memory
  (hipHostMalloc); (b) `HSA_XNACK=1`, set globally by the image for
  LAMMPS, changes ROCm's SVM host-memory registration for a binary not
  built for xnack. Test: `smoke/hsa_test/hsa_test.sh` (job 5744976).
- **HSA test 1** (5744976, `flux run -n8 -c6 -g1 rmg-gpu` directly, chore
  env): RLIMIT_MEMLOCK is `unlimited` both in Flux's initial program and
  inside Flux jobs -> **(a) ruled out**. With `HSA_XNACK=0` the run still
  logs `hsaKmtSVMSetAttr failed` x14 and dies (137) at ~60 s, just
  without the hipHostMalloc line -> **(b) ruled out** as the sole cause.
  Every rank listed all 8 GPUs in its RMG log. (That job sat in Slurm's
  "CG" state ~5+ min after finishing -- node epilog, not our processes.)
- **HSA test 2** (5745138): one visible GPU per rank, via
  `ROCR_VISIBLE_DEVICES=$FLUX_TASK_LOCAL_ID` (C) or Flux's
  `-o gpu-affinity=per-task` (D). RMG logs then list only `device 0`, but
  both fail identically -> **GPU visibility ruled out.**
- Next hypothesis: the container's ROCm 6.3.3 user-space runtime
  (`libhsa-runtime64`, which links the hsakmt thunk making the failing SVM
  ioctl) against the host's amdgpu/KFD driver, vs the host `rocm/6.3.1`
  rmg-gpu was built with. Test `rocm_test.slurm` (5745177): E native
  srun + modules; F srun + apptainer with host /opt/rocm-6.3.1 bound and
  first; G srun + apptainer with container ROCm (no Flux).
- **ROCm test** (5745177): E (native srun, build modules, host ROCm
  6.3.1, no container, no Flux) **also dies** -- `hipHostMalloc failed`,
  rc 137, ~35 s; F and G (srun + apptainer, host vs container ROCm) die
  the same way. **So it is not the container, Flux, or the ROCm version:
  the input doesn't fit in 8 GPUs.** pyRMG's estimator asked for 112 ranks
  (14 nodes) for these cells (Bi54Se45 = 1080 valence electrons; default
  `electrons_per_gpu: 10`), and `max_tasks_per_job: 8` clamped it to 1
  node. The HSA / hipHostMalloc lines are how RMG reports the failed
  allocation; 137 is the kill that follows.
  **Implication for the README's old pitfall** ("multigrid fails with
  hsaKmtSVMSetAttr, even at a single MPI task"): very possibly the same
  out-of-memory, not a multigrid/container incompatibility -- to re-test in
  Phase 2 with a structure that fits, and with enough nodes.
- To finish the generation-2 smoke on 1 node: `ase_mace_md.py` now takes
  the supercell cap from the MD input (`max_supercell_atoms`, default 100
  = unchanged behaviour); `in.mace.smoke.json` sets 24.
  `make_smoke_copy.sh` takes `CONVERGED_BI=<dir>` to pre-place the
  converged Bi run. Fresh copy `smoke/run2`, job 5745231.

## Phase 2: multi-node RMG

Test case: the 99-atom loop-1 candidate `smoke/run1/DFT/uq/1/1`
(Bi54Se45, davidson, Quench Electrons). From job 5745551 on, RMG tests use
a copy with `max_scf_steps = "3"` (`phase2/case_uq1_1_scf3`) -- enough to
show ranks join, memory fits and the per-step time, without converging.

- **Smoke run 4** (fresh `smoke/run2`, 5745231): 6 fresh RMG runs
  converged (40-57 s each) before it was cancelled on purpose to free the
  debug queue (1 submitted job per user) for multi-node tests.
- **`multinode_test.slurm`** (5745339, 2 nodes):
  - native `srun -N2 -n16 mpi_hello gpu`: one 16-rank job, host and
    GPU-buffer Allreduce OK; each rank sees 1 GPU (`--gpu-bind=closest`,
    ROCR set by Slurm); Cray MPICH loaded Cray's
    `/opt/cray/pe/pmi/6.1.15/lib/libpmi2.so.0.6.0`.
  - **native 2-node RMG on the 99-atom cell works**: 16 PEs (4x2x2), one
    banner, no memory errors (it died on 1 node). First SCF step 94 s
    (incl. setup), then 28-35 s/step; RMS[dV] 0.22 -> 0.066 in 5 steps;
    7 steps in the 420 s cap. At 1e-8 that's ~30-50 steps, ~15-25 min.
  - **container Flux instance spans both nodes**: `flux resource list` =
    2 nodes / 112 cores / 16 GPUs, one instance (first time on Frontier).
  - **`flux run -N2 -n16 mpi_hello` fails in MPI_Init on every process**,
    with Cray's or Flux's libpmi first, host or GPU buffers:
    `MPIDI_OFI_mpi_init_hook -> open_fabric -> MPIDI_CRAY_ofi_nic_assign_policy
    -> MPIDI_CRAY_ofi_get_nic_index: OFI invalid value for environment
    variable`, every process printing `[Rank 0]`. The 1-node pipeline runs
    with the same env did join (8 MPI processes). So: an env var Cray
    MPICH reads while picking a NIC is wrong/inconsistent under Flux on >1
    node -- suspects MPICH_OFI_NIC_POLICY (image sets BLOCK, chores NUMA)
    and the outer srun's SLURM_*/PALS_*/PMI_* leaking into the container.
    RMG-through-Flux was skipped (no PMI order joined).
  - Diagnostic `flux_mpi_diag.slurm` (5745551): env dump per node, 1-node
    control, 2 nodes x each NIC policy with MPICH_OFI_NIC_VERBOSE=2, and
    with the outer launcher's env stripped.
- **ROOT CAUSE of the multi-node MPI_Init failure (5745551): Cray MPICH's
  `MPICH_OFI_NIC_POLICY=NUMA`**, which `rmg_container_env` sets for every
  RMG chore. Under Flux (no per-task CPU binding) each task's cpuset is
  the whole node, and on >1 node MPICH aborts: `Unable to use a
  NIC_POLICY of 'NUMA'. Rank 12 is not confined to a single NUMA node.
  There are 4 numa_nodes detected`. Results, 2 nodes x 8 ranks via
  `flux run`, Cray libpmi first:
  | policy | result |
  |---|---|
  | NUMA | MPI_Init abort (above) |
  | GPU | abort: `Cannot determine NUMA node associated with user-selected GPU` |
  | **BLOCK** (image default) | **one 16-rank job, Allreduce OK**; NICs cxi0-3 used on each node |
  | **ROUND-ROBIN** | **one 16-rank job, Allreduce OK** |
  | NUMA, outer SLURM/PALS/PMI env stripped | same abort -- env leakage not involved |
  | NUMA, 1 rank per node | OK (2 ranks) |
  | NUMA, 1 node x 8 (control) | OK -- why every 1-node run worked |
  Also: no "Flux libpmi first" library-path change was needed -- but not
  because Cray's libpmi speaks Flux's PMI: /proc/self/maps in 5745571 shows
  every rank loaded `/usr/local/lib/flux/libpmi2.so.0.0.0` even with
  RMG_LD_LIBRARY_PATH (Cray first). Flux's shell puts its own libpmi
  directory first on each task's library path, i.e. Flux applies the
  Perlmutter fix itself. The env dump shows no `SLURM_*`/`PALS_*`
  vars inside Flux tasks at all; image env does leak in
  (`OMP_PROC_BIND=spread`, `OMP_PLACES=threads`, `HSA_XNACK=1`,
  `MPICH_GPU_SUPPORT_ENABLED=1`, `CUDA_VISIBLE_DEVICES=-1` from the
  broker's own srun step; Flux overrides CUDA_VISIBLE_DEVICES per job).
- **RMG via Flux on 2 nodes, BLOCK, 3-SCF-step cap** (5745571): the
  16 ranks of `mpi_hello` now join (host Allreduce OK, Flux libpmi2 loaded)
  -- **but every rank reports `gpu_count=0`**, the GPU-buffer variant logs
  `GTL_DEBUG: hsa_init ... HSA_STATUS_ERROR_OUT_OF_RESOURCES` on all 16, and
  rmg-gpu hung at startup (no log, empty stdout) until the 420 s timeout.
  Re-reading 5745551: in the same 2-node instance, a `-N1` job saw 8 GPUs
  per rank while every `-N2` job (even 1 rank per node) saw 0.
- **GPU diag** (5745716, no native steps first): in `-N2` Flux tasks
  `/dev/kfd` and all render nodes are present, `rocminfo` finds every GPU,
  `ulimit -u` ~2M, memlock unlimited -- yet `mpi_hello -N2` (2 or 16
  ranks) reports 0 GPUs after MPI_Init. **So MPI_Init in a job spanning
  nodes, inside the container, breaks HIP's GPU discovery.** Suspects:
  libfabric (CXI) initialising its ROCr HMEM support / ROCR cache monitor
  during inter-node MPI_Init, and the image's global `HSA_XNACK=1` (unset
  natively). Matrix test `flux_gpu_mpi_matrix.slurm` (5745756): hipfirst
  build, HSA_XNACK=0/unset, FI_HMEM=system,
  FI_MR_ROCR_CACHE_MONITOR_ENABLED=0; then RMG with the first that works.
- **Matrix result** (5745756, 2 nodes x 8 via Flux, BLOCK):
  | case | GPUs seen (of 16) | notes |
  |---|---|---|
  | baseline, GPU support 0 | 0 | host Allreduce OK |
  | baseline, GPU support 1 | 0 | MPI_Init aborts: GTL `hsa_init ... HSA_STATUS_ERROR_OUT_OF_RESOURCES` |
  | **hipfirst** (GPU query before MPI_Init), GPU support 0 | **16** | host Allreduce OK |
  | **hipfirst**, GPU support 1 | **16** | **GPU-buffer Allreduce OK across 2 nodes**; distinct PCI ids per rank |
  | HSA_XNACK=0, GPU support 0/1 | 0 / abort | XNACK ruled out |
  | HSA_XNACK unset, FI_HMEM=system, FI_MR_ROCR_CACHE_MONITOR_ENABLED=0 | inconclusive | empty output: runs queued behind a timed-out job until their own 90 s timeout |
  **So: initialising HIP before a multi-node MPI_Init fixes GPU visibility
  and GPU-aware MPI.** Unmodified-binary route: `mpi_tests/hip_preinit.c`
  -> `libhip_preinit.so`, an LD_PRELOAD shim whose constructor calls
  `hipInit(0)` before main. Test `flux_preinit_rmg.slurm` (5745915): hello
  + preload, RMG + preload on the 3-step-capped 99-atom case, and the 3
  inconclusive cases again with the Flux queue drained between runs.
- **Preload result** (5745915, 2 nodes): `mpi_hello` + preload -> 16/16
  ranks with a GPU, each node's 8 ranks on 8 distinct GCDs (PCI c1..de),
  host and GPU-buffer Allreduce OK. **rmg-gpu + preload runs as ONE 2-node
  job inside the container through Flux**: 16 PEs (4x2x2), one banner,
  8 devices per node; energies match the native 2-node run to ~1e-4 Ha and
  RMS[dV] exactly:
  | | native srun | container + Flux + preload |
  |---|---|---|
  | step 0 (incl. setup) | 85 s (94 s scf time) | 136 s (152 s) |
  | step 1 | 28.6 s, -4627.110131 Ha | 48.2 s, -4627.110279 Ha |
  | step 2 | 30.9 s, -4626.822909 Ha | 61.2 s, -4626.822979 Ha |
  ~1.6-2x slower per step. Each rank also logs `HSA exception:
  hsaKmtSVMSetAttr failed.` 4 times (64 total; native: 0) -- non-fatal here.
  The 3-step-capped run still hit the 480 s timeout in the post-SCF
  force/stress/output stage (log ends at the stress tensor).
- Retests with the queue drained: **`FI_HMEM=system` also fixes it**
  (16/16 GPUs, no preload; but that mpi_hello then hit its 90 s timeout
  after printing -- possible hang in MPI_Finalize, to check);
  `FI_MR_ROCR_CACHE_MONITOR_ENABLED=0` and HSA_XNACK unset: 0 GPUs.
  **Root cause narrowed to libfabric initialising its ROCr HMEM interface
  during an inter-node MPI_Init, before HIP.** (Why only in the container
  is still open: natively the same libfabric + Cray MPICH work -- the
  container's ROCm 6.3.3 runtime vs host 6.3.1 is the obvious difference.)
- Timing study `rmg_timing.slurm` (2-SCF-step cap), **job 1** (5746040):
  - `base` and `cpubind` ran as one 16-rank job but completed 0 SCF steps
    in their 300 s: the time went to **startup**. `base`: tasks started
    2 s after `flux run`, RMG's own `Start time` 3 min later. `cpubind`:
    tasks waited 2.7 min to start (the previous case's killed processes
    still releasing resources; 60 s drain too short), then 1.8 min to RMG
    start. Neither reached a steady-state step -- no per-step numbers.
  - `cpu-affinity=per-task` binds as intended: 6 cores + SMT siblings
    per rank (rank 0 10-15/74-79, rank 1 17-22/81-86).
  - `cpubind_numa`: MPI_Init abort in 28 s -- Flux's 6-core slices don't
    line up with NUMA domains, so NUMA policy is still unusable.
  - Lead: every rank sees all 8 GPUs under Flux (native: 1), so each of 8
    processes per node initialises 8 GPUs -- plausible source of the slow
    startup, the 4 hsaKmtSVMSetAttr warnings per rank, and slower steps.
- **Job 2** (5746452, 400 s per case, 180 s drain): `native`,
  `cpubind_gpubind` (one visible GPU per rank), `cpubind_hmem`
  (FI_HMEM=system instead of the preload); per-case launch/task/RMG-start
  timeline recorded.
- **Job 2 results** (5746452, same 2 nodes for all three):
  | case | step 0 | step 1 | total, exit |
  |---|---|---|---|
  | native srun -c7 --gpu-bind=closest | 83.9 s | 34.2 s | 342 s, rc 0 |
  | flux cpubind_gpubind (preload) | 101.1 s | 42.4 s | 368 s, rc 0 |
  | flux cpubind_hmem (FI_HMEM=system, no preload) | 101.2 s | 42.7 s | 378 s, rc 0 |
  vs unbound flux (5745915): 136 s / 48-61 s. **CPU binding closes most of
  the gap (now ~1.24x per step)**; one visible GPU per rank adds nothing
  measurable and doesn't remove the 4 hsaKmtSVMSetAttr warnings per rank.
  FI_HMEM=system is as fast as the preload, and RMG exits cleanly with it.
  Startup: tasks -> RMG `Start time` ~1:45 in both flux cases.
- **Why the rest of the gap** (per-rank bind files, node 1):
  native gives each rank one whole L3 group (1-7, 9-15, ..., 57-63) and
  its closest GCD (4,5,2,3,6,7,0,1). Under Flux: (1) the instance's cpuset
  starts at core 10 -- cores 1-9 missing, inherited from the brokers' srun
  step; (2) Flux's 6-core slices straddle L3/NUMA groups (rank 2:
  23,25-29; rank 3: 30-31,33-36); (3) gpu-affinity hands out GCD 0..7 in
  order, so e.g. rank 0 (cores 10-15, nearest GCD 5) gets GCD 0.
- **Job 3** (5746806): brokers started with `srun --cpu-bind=none`;
  `c7_closest` (-c7 per rank = one L3 group, ROCR_VISIBLE_DEVICES = GCD
  closest to the rank's first core), `c6_closest` (-c6, same GPU rule --
  what fits MatEnsemble's 56N-1 core budget), and native on the same nodes.
- **Job 3 results** (5746806, same 2 nodes): **gap closed.**
  | case | step 0 | step 1 | total |
  |---|---|---|---|
  | native srun -c7 --gpu-bind=closest | 86.3 s | 34.87 s | 355 s, rc 0 |
  | **flux c7_closest** (brokers --cpu-bind=none, -c7, per-task CPU affinity, closest GCD) | **87.1 s** | **34.98 s** | **335 s, rc 0** |
  | flux c6_closest | -- | -- | rc 137 at 61 s |
  With `--cpu-bind=none` the instance cpuset is the whole node
  (1-7,9-15,...,57-63 + siblings), and -c7 slices land exactly on the L3
  groups, giving every rank native's layout. c6_closest: 6-core slices put
  ranks 1 and 2 both in the 17-23 group, so both picked GCD 2 (two ranks
  on one GCD -> killed). So: 7 cores per rank, aligned to L3 groups.
  hsaKmtSVMSetAttr x4 per rank remains and is harmless to speed.
- MatEnsemble 0.5.8 core budget (manager.py): `controller_cores` is 0 on
  a multi-rank Flux instance and 1 on a 1-rank one (hence "56N-1" only
  ever on 1 node); `Pipeline(controller_cores=0)` lifts it on 1 node too.
- **Pipeline integration** (this branch): `launch_flux.slurm` brokers with
  `--cpu-bind=none`; `rmg_container_env` NIC policy BLOCK, `FI_HMEM=system`
  (chosen over the preload shim: same speed, clean exit, no extra binary;
  RMG uses host-buffer MPI), `RMG_GPU_CLOSEST_GCDS=4,5,2,3,6,7,0,1`;
  `rmg_dft.py` sets each rank's ROCR_VISIBLE_DEVICES from it
  (`closest_gpu_env`, package reference copy kept in sync); RMG Pipeline
  `controller_cores=0`; config `cores_per_task: 7`, `max_tasks_per_job: 16`
  (smoke: 8). Test: `phase2/pipe2node` -- `--stage converge_dft_data` on
  2 nodes, the two 99-atom uq/1 candidates with `max_scf_steps: 3`
  (5747038).
- **Pipeline result** (5747038): each candidate submitted as `2 node(s),
  16 MPI rank(s) x 7 core(s) x 1 GPU` (pyRMG's 112-rank estimate clamped
  by max_tasks_per_job=16); RMG logged `16 MPI processes/image`, one
  device per rank. uq/1/1: step 0 86.0 s, step 1 29.3 s, step 2 37.6 s
  (native 2-node: 85-86 / 28.6-34.9 / 30.9 s); uq/1/0: 87.9 / - / 37.5 s.
  **Multi-node RMG through the real pipeline runs at native speed.** Both
  chores then "failed" by design: rmg-gpu exited normally, but pyRMG's
  `read_results` raised `did not converge (Quench Electrons)` -- the
  3-step cap.
- 1- vs 2-node comparison, same structure, fully converged, through the
  pipeline: the 25-atom Bi9Se16 candidate (old Consolidated_Pipeline
  uq/1/44: 27 SCF steps, 275 s on 8 GPUs in the old sandbox) in
  `phase2/scaling_{1,2}node` (max_tasks_per_job 8 / 16).
- **Phase 2 done-criterion result** (5747152 / 5747197, pipeline
  `--stage converge_dft_data`, fully converged):
  | | 1 node | 2 nodes |
  |---|---|---|
  | MPI | 8 PEs (2x2x2) | 16 PEs (4x2x2) |
  | SCF steps | 26 | 29 |
  | RMG total | 269.0 s | 173.2 s |
  | mean per step | 10.35 s | 5.97 s (1.73x) |
  | energy | -23172.17999939 eV | -23172.17999858 eV |
  dE 8e-7 eV, max dF 1.7e-3 eV/A, max dstress 4.5e-6. (Old sandbox, same
  structure, 1 node: 275 s, 10.6 s/step.)

### Phase 2 summary -- what made multi-node RMG work in the container
1. `MPICH_OFI_NIC_POLICY=BLOCK`, not NUMA (NUMA aborts MPI_Init on >1
   node when Flux tasks aren't confined to one NUMA domain).
2. `FI_HMEM=system` (or `libhip_preinit.so` via LD_PRELOAD): otherwise
   libfabric's ROCr init during an inter-node MPI_Init leaves HIP with 0
   GPUs and rmg-gpu hangs at startup. Root cause of *why only in the
   container* not established (host vs container ROCm runtime suspected).
3. For speed: brokers started with `srun --cpu-bind=none`, 7 cores per
   rank with Flux per-task CPU affinity (one L3 group each), and each
   rank's GPU = closest GCD (rmg_dft.py `closest_gpu_env`).
Not needed: Perlmutter's "Flux libpmi first" (Flux does it itself),
`mpi=True`, static R.json, extra device binds. Ruled out along the way:
RLIMIT_MEMLOCK, HSA_XNACK, GPU visibility per rank, leaked SLURM/PALS env.
Separately: the HSA `hsaKmtSVMSetAttr`/`hipHostMalloc failed` crash is
RMG running out of GPU memory (too few GPUs for the cell), not a
container problem.

Still open: multigrid retest (may have been the same OOM); isolated-Bi
davidson non-convergence; pyRMG's estimator (14 nodes for 99 atoms) vs
what actually fits (2 nodes); `hsaKmtSVMSetAttr` x4 per rank in the
container (harmless to speed).

## Phase 3: VASP on Frontier
