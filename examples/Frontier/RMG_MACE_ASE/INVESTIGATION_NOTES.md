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
  `frontier-v0.5.8` (= MatEnsemble 0.5.8, what Perlmutter validated).

## Phase 1: example update

## Phase 2: multi-node RMG

## Phase 3: VASP on Frontier
