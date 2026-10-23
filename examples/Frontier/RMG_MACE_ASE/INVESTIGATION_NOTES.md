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

## Phase 2: multi-node RMG

## Phase 3: VASP on Frontier
