# Updating RMG_MACE_ASE from VASP_MACE_ASE, then multi-node RMG

**Audience:** whoever (person or coding agent) picks up the Frontier `RMG_MACE_ASE`
example next. It has two parts, to be done in order:

1. **Port** the fixes found while building and smoke-testing
   [`examples/Perlmutter/VASP_MACE_ASE`](../../Perlmutter/VASP_MACE_ASE/README.md)
   (Oct 2026), then **smoke-test this example on Frontier**.
2. **Investigate multi-node RMG**, using what got multi-rank and multi-node VASP working
   on Perlmutter as the starting point.

`VASP_MACE_ASE` is this example's pipeline with VASP swapped in, so its `run_pipeline.py`
is the reference for every change below. Diff the two files rather than re-deriving
anything. Read `examples/README.md` and this example's own README (especially its
pitfalls) before changing code, and agree on the plan with the person you're working for
before implementing.

---

## Part 1 — Port these changes, then smoke-test

Each item says what broke (or would break) and where the fix lives in
`VASP_MACE_ASE/run_pipeline.py`. Items 1–4 are bugs this example very likely has today,
because the code is shared. Confirm each one on Frontier rather than assuming it.

1. **Chores run in their own working directory, so strategy paths must be absolute.**
   With MatEnsemble 0.5.8, every chore (including the `Pipeline.strategy` function
   `copy_and_spawn_md`) executes inside `matensemble_workflow-*/out/chore-*/`, not the
   launch directory.
   - `copy_and_spawn_md` resolves `md_run_directory`/`md_inputs_directory` itself. With
     relative config paths it copied every fitted model *into the chore directory* and
     then failed with `Invalid task command .../ase_mace.py`. No validation single points
     ran, and `downselect` crashed with `not enough values to unpack`.
   - Fix: `os.path.abspath(...)` both paths in `run_fit_and_validate` before the strategy
     is defined.
   - Check first which MatEnsemble version the Frontier image has (`frontier-v0.5.5`
     at the time of writing). If it doesn't use per-chore directories, the bug is latent,
     but the fix is harmless.
2. **Failed chores don't raise out of `future.result()`.**
   - `converge_dft_data`: when every RMG chore fails or is rejected, the stage still
     "succeeds" and the error only appears later. Port the post-submission check that
     each working directory has `finished_file`: warn per failure, raise if none
     succeeded.
   - `downselect`: port the clear error for an empty validation tree.
   - Per-chore logs are `matensemble_workflow-*/out/chore-*/{stdout,stderr}`, and the
     overall log is `matensemble_workflow-*/matensemble_workflow.log`.
3. **Reruns of `--stage all` must skip finished work, not crash.** Three places currently
   raise when everything is already done:
   - `converge_dft_data` raises `ValueError` when every pair already has `finished_file`.
     Port the no-op.
   - `fit_and_validate` raises "no structures were found" when every FT-MD structure
     already has `properties.json`. It should raise only if no structures exist at all.
   - `uq_single_points` raises "No (force field, structure) batches found" when every
     pair is finished. It should skip.
4. **mace-torch must be ≥ 0.3.15 for `freeze: 6`.**
   - `build_ff_inputs.mace_overrides.freeze` needs `--freeze`, which 0.3.14 lacks. That
     is what MatEnsemble's own requirements installed on Perlmutter.
   - Check the Frontier image with
     `python -c "from mace.tools import build_default_arg_parser as p; p().parse_args(['--name','x','--freeze','6'])"`.
   - In 0.3.16, `freeze: 6` freezes the embedding, interaction and product weights and
     trains only the readouts.
5. **Closing the active-learning loop** (new capability, needed for multi-generation
   runs). This example's `build_ff_inputs` only reads `DFT/training/Defects`, so
   converged `DFT/uq/<n>` candidates never become training data. Port:
   - `extra_training_subpaths` (e.g. `[uq/1]`): converged runs are converted with
     `write_training_xyz` into a `uq_1` label;
   - `always_include_labels` (e.g. `[uq_1]`) so every member trains on the new data.
     Otherwise it's randomly sampled: only 1 of 3 smoke members drew it.
   - `_with_defaults` plus the `# GEN`-tagged config keys, so moving to generation N
     means editing ~10 keys. See `VASP_MACE_ASE/README.md` §5.
   - `rmg_dft.py` already writes `stress_voigt` in ASE's convention (from the ASE
     calculator), so unlike VASP no sign fix should be needed. Confirm it on one
     candidate.
6. **Core budget per node: total minus one for the broker** (a MatEnsemble behaviour).
   - On Perlmutter, Flux offered 64 × N − 1 cores on N nodes. A job needing a full
     node's cores in a 1-node allocation was rejected ("requires more resources than the
     allocation can provide").
   - On Frontier, check `flux resource list` and the `Free_cores=` in
     `matensemble_workflow.log` against `cores_per_task (7) × max_tasks_per_job (8) = 56`.
     If a 1-node smoke run's RMG chores are rejected, drop `cores_per_task` to 6 for the
     smoke config (and `rmg_num_threads` to match).
7. **Optional / to decide with the user:**
   - `Pipeline(reserve_broker_node=False)`, which is the Perlmutter convention.
   - A smoke config, following the shrink table in `VASP_MACE_ASE/README.md` §4:
     3 fits × 2 epochs, 4 short MD runs via an `in.mace.smoke.json`, 2 models kept,
     2 candidates.
   - Whether to enable `balanced_supercell` in `ase_mace_md.py`; it is active in
     VASP_MACE_ASE.

**Smoke test on Frontier** (before Part 2):
- Use a scratch copy of the example, since outputs are written in place.
- Run 1 node, smoke config, `--stage all`, then `--stage all` again (resume).
- Then a generation-2 config: bump the `# GEN` keys, seed
  `FF/generation_2/mace-omat-medium/model.model`, run `--stage all`.
- Expected:
  - loop 1 writes `DFT/uq/1/*/POSCAR`;
  - the rerun skips everything;
  - loop 2 converges those candidates with RMG, builds a `uq_1` label and writes
    `DFT/uq/2`.
- Report what actually happened, with log excerpts.

---

## Part 2 — Multi-node RMG

**Goal:** RMG chores that span many Frontier nodes inside one Flux instance, so users can
scale `allocated_nodes` up. Today `max_tasks_per_job: 8` effectively caps every RMG job at
one node, and this README's pitfalls record a long debugging history for multi-rank RMG.

**What made multi-rank/multi-node VASP work on Perlmutter** — verify which of these carry
over; don't assume:

1. **What a multi-task chore is.**
   - With `Resources(num_tasks=N, ...)`, Flux starts **N copies of the chore's Python
     function**, one per rank. Each copy runs the driver script and launches its own
     binary.
   - `mpi=True` makes MatEnsemble set Flux's `mpi=pmi2` shell option, and the N binaries
     then join one MPI job through Flux's PMI.
   - Evidence on Perlmutter: the driver's stdout appears N times, and the OUTCAR reports
     `running 8 mpi-ranks ... on 2 nodes`.
   - With `mpi=False`, VASP ran as N independent 1-rank jobs colliding in one
     directory. That's the "several startup banners" symptom this README's pitfalls
     describe for RMG.
2. **The binary must load Flux's PMI library, not Cray's.** Cray MPICH has a direct
   dependency on `libpmi.so.0`/`libpmi2.so.0`.
   - With `/opt/cray/pe/lib64` first on `LD_LIBRARY_PATH`, Cray's PMI client is loaded,
     doesn't understand Flux's `PMI_FD` rendezvous, and MPI_Init silently falls back to
     singleton mode.
   - The fix was putting `/usr/local/lib/flux` (Flux's `libpmi*`) **first** in the
     binary's library path.
   - `VASP_LD_LIBRARY_PATH` in `VASP_MACE_ASE/run_pipeline.py` has the full rationale.
     It is applied only to the binary's subprocess, never the chore's Python process.
   - Compare `RMG_LD_LIBRARY_PATH`: it puts `/opt/cray/pe/lib64` first and has no Flux
     PMI directory at all.
3. **The subprocess must keep the PMI file descriptor.** Both `vasp_dft.py` and this
   example's `rmg_dft.py` already launch with `close_fds=False`.
4. **Ranks are sized per GPU, and node count follows from it.**
   - VASP uses `num_tasks = nodes × ranks_per_node` with `gpus_per_task=1`. Flux can
     place at most 4 GPU ranks per Perlmutter node, so an 8-rank job necessarily spans
     2 nodes.
   - RMG already sizes `allocated_nodes × gpus_per_node` (8 per Frontier node). The
     limit is `max_tasks_per_job`, not the formula.
5. **One Flux instance must actually span the nodes.** On Perlmutter this took
   `launch_multi_node.slurm`'s static `R.json`, host networking, PMI env passthrough and
   device mounts. Confirm on Frontier with `flux resource list` inside the instance
   (expect N nodes, 8N GPUs) before running anything expensive.

**History to reconcile.** This README says `mpi=True` was tried for RMG and produced "a
single RMG startup banner instead of several", meaning the ranks *did* join one job, but
didn't fix the HSA / `hipErrorInvalidDeviceFunction` failures. Those were later traced to
`kohn_sham_solver: multigrid` rather than the launch layer, and `davidson` is now the
default. **So `mpi=True` was likely rejected for the wrong reason.** With `davidson` it
may now work. Test that first, before anything more elaborate.

**Suggested sequence** (smallest proof first; stop and report at each step):
1. Single node, `max_tasks_per_job: 8`, a structure known to converge. Compare
   `mpi=False` (current) against `mpi=True`.
   - Count startup banners and check the rank count RMG reports in its log.
   - Compare energy and wall time.
2. Check which `libpmi` the `rmg-gpu` process actually loaded, e.g.
   `cat /proc/<pid>/maps | grep -i pmi` from a debug chore, or `LD_DEBUG=libs`.
   - If it's Cray's, prepend the Frontier image's Flux `libpmi` directory to
     `RMG_LD_LIBRARY_PATH` (find it with `find / -name 'libpmi*.so*' 2>/dev/null` inside
     the container) and repeat step 1.
3. 2-node allocation, `flux resource list` showing 2 nodes / 16 GPUs, one structure with
   `allocated_nodes` forced to 2 (16 ranks).
   - Confirm 16 ranks on 2 nodes in RMG's log, and an energy matching the 1-node run of
     the same structure.
   - Mind the broker core (item 6): a job spanning every node may need one fewer core
     per rank.
4. Only then raise `max_tasks_per_job` and test larger node counts.
   - Make sure `pyRMG`'s `compute_grid_and_resources`/`allocated_nodes` consistency
     check agrees with what was actually requested (see the clamp comment in
     `run_converge_dft_data`).

Record every confirmed finding (and every dead end) in this example's README pitfalls,
the same way the existing entries are written. The same "N Python copies + Flux PMI"
mechanism is also the thing to check for multi-rank Quantum Espresso in
`examples/HPC_aas/QE_ACE`.
