# Pipeline checks suggested by the Frontier work (October 2026)

Updating `examples/Frontier/RMG_MACE_ASE` and adding `examples/Frontier/VASP` turned up
bugs and fragile spots in code these Perlmutter examples share. **Nothing in the
Perlmutter examples has been changed** — this file lists what to check, where, and what
the Frontier fix was, so each can be confirmed (or ruled out) on Perlmutter. Line numbers
are as of commit `82f766b` on branch `frontier-rmg-update`.

Legend: **likely** = the same code path, the bug should reproduce; **latent** = same code,
but this example's data/setup probably doesn't trigger it today.

## Summary

| # | Issue | `VASP_MACE_ASE` | `VASP_ReaxFF_LAMMPs` |
|---|---|---|---|
| 1 | MACE fit dies when a member has fewer training configs than `batch_size` | latent | n/a (JAX-ReaxFF) |
| 2 | MatEnsemble cloned unpinned in the Dockerfile | likely | likely |
| 3 | A chore's `PYTHONPATH` replaces the launcher's instead of extending it | latent | latent |
| 4 | `--stage all` rerun crashes when every FT-MD structure is already done | fixed | **likely** |
| 5 | `--stage all` rerun crashes when every single point is already done | fixed | **likely** |
| 6 | Failed DFT chores pass silently (no post-run `properties.json` check) | fixed | **likely** |
| 7 | VASP segfaults on large vacuum cells with the default stack limit | check | check |
| 8 | MPI library path replaced for the DFT binary must keep Flux's `libpmi` first | already done | already done |

## 1. MACE batch size larger than a member's data — latent

**What happened on Frontier.** In the RMG smoke test every MACE fit failed with

```
ERROR: Batch size (10) is larger than the number of training data (5)
...
ValueError: torch.cat(): expected a non-empty list of Tensors   (compute_avg_num_neighbors)
```

MACE's training loader drops incomplete batches, so a member whose sampled label
combination has fewer training configs than `batch_size` (default 10, from
`build_ensemble_inputs.DEFAULT_BASE_CONFIG`) gets an empty loader.

**Here.** `FF/mace_fit.py` is identical to the Frontier copy before the fix. This
example's labels are large (33-222 frames each), so the full config shouldn't hit it —
but a smoke config, a small `extra_training_subpaths` label used alone, or a new system
could.

**Frontier fix.** `examples/Frontier/RMG_MACE_ASE/FF/mace_fit.py`,
`_fit_batch_sizes_to_data`: before `run(args)`, count the configs in `train_file`
(and `valid_file`, if given), reproduce MACE's split
(`train = min(n - int(valid_fraction * n), n - 1)`, `valid = n - train`, as in
`mace.data.utils.random_train_valid_split`, mace-torch 0.3.16), and shrink
`batch_size`/`valid_batch_size` to fit, printing a note to the chore's stderr. A
drop-in copy works here unchanged.

**Check.** `grep -c '^[0-9]' FF/generation_*/mace_inputs/*/train.xyz` — any member
under about 12 configs (10 / (1 − 0.15)) is at risk.

## 2. Unpinned MatEnsemble clone — likely

**What happened on Frontier.** The published `ghcr.io/freddude2004/matensemble:frontier-v0.5.8`
image does **not** contain MatEnsemble 0.5.8: its `/opt/matensemble` is commit
`6307e46` (2026-07-08, reports 0.5.3), whose `Pipeline` has no `reserve_broker_node`.
The Frontier Dockerfile clones `main` with `--depth 1`, so the tag reflects whatever a
cached clone layer held when it was built. (That image also had no torch/MACE at all.)

**Here.** `Dockerfile.mace:24` and `../VASP_ReaxFF_LAMMPs/Dockerfile.ff-fit:12` both
`git clone --depth 1 https://github.com/Q-CAD/MatEnsemble.git`. A rebuild picks up
whatever `main` is that day — and a layer cache can silently keep an old one.

**Suggestion.** Pin it: `git clone https://github.com/Q-CAD/MatEnsemble.git && git -C
matensemble checkout 5d8be305f1` (main as of 2026-08-31, version 0.5.8 — what this
example's image actually got on 2026-10-22, and what the Frontier derived image pins),
and add a build-time check such as
`python -c "import importlib.metadata as m; assert m.version('matensemble') == '0.5.8'"`.
Bump both together deliberately.

## 3. Chore `PYTHONPATH` replaces the launcher's — latent

**What happened on Frontier.** `Resources(env={'PYTHONPATH': ensemble_fffit_pythonpath()})`
*replaces* the inherited `PYTHONPATH` in the chore rather than extending it. On Frontier
pyRMG is on the launcher's `PYTHONPATH` (not installed in the image), so RMG chores
would have lost it. Fixed with `chore_pythonpath()` = EnsembleFFFit root + the
launcher's `PYTHONPATH`.

**Here.** `run_pipeline.py:99, 121, 719` (and `../VASP_ReaxFF_LAMMPs/run_pipeline.py:238,
279, 309`) use `ensemble_fffit_pythonpath()` alone. `launch_multi_node.slurm` puts
`FLUX_PYTHON_BINDINGS` (`/usr/local/lib/python3.12/site-packages`, the complete Flux
Python bindings) on the pipeline's `PYTHONPATH` — chores therefore don't see it and
fall back to the image's incomplete `/opt/basic` copy. This works today, presumably
because nothing in a chore imports `flux.resource`; it would bite the first time a
chore needs a package that is only on the launcher's `PYTHONPATH`.

**Suggestion.** Copy `chore_pythonpath()` from
`examples/Frontier/RMG_MACE_ASE/run_pipeline.py` and use it in place of
`ensemble_fffit_pythonpath()` in every chore/strategy `env`.

## 4. FT-MD rerun crashes when all structures are done — `VASP_ReaxFF_LAMMPs` likely

**Here.** `../VASP_ReaxFF_LAMMPs/run_pipeline.py:1190` raises "no structures were found"
whenever `build_flat_task_dicts` returns nothing — which also happens when every
structure already has `finished_file`, i.e. on any resumed run. (`VASP_MACE_ASE` fixed
this, lines 806-819; the Frontier example got the same port.)

**Suggestion.** As in `VASP_MACE_ASE`: raise only if no `structure_filename` exists under
the structures root at all; otherwise print "skipping FT-MD" and continue.

**Check.** Run the FT-MD stage twice; the second run should be a no-op, not an error.

## 5. Single-point rerun crashes when all pairs are done — `VASP_ReaxFF_LAMMPs` likely

**Here.** `../VASP_ReaxFF_LAMMPs/run_pipeline.py:1513` raises "No (force field,
structure) batches found" when `build_task_dicts(..., finished_file=...)` is empty —
including when everything already finished. (`VASP_MACE_ASE` fixed this for
`uq_single_points`, lines 937-944.)

**Suggestion.** Same pattern: if a rebuild with `finished_file=None` finds batches,
print "nothing new to run, skipping" and return; raise only otherwise.

## 6. Failed DFT chores pass silently — `VASP_ReaxFF_LAMMPs` likely

**Here.** A failed or rejected chore doesn't raise out of `future.result()`.
`VASP_MACE_ASE`'s `converge_dft_data` checks every working directory for
`finished_file` afterwards (lines 364-379: warn per failure, raise if none converged);
`../VASP_ReaxFF_LAMMPs/run_pipeline.py`'s `run_converge_dft_data` (ends at line 573)
has no such check, so a stage where every VASP run failed "succeeds" and the error
surfaces stages later. The same holds for its other submit-and-`result()` stages.

**Suggestion.** Port the post-run check (it is ~10 lines). On Frontier it caught every
DFT failure in testing (out-of-memory, segfaults, capped SCF) at the stage that failed.

## 7. VASP segfault on large vacuum cells — check

**What happened on Frontier.** `vasp/6.6.1-gpu` (and 6.6.0) segfaulted right after
`EWALD` (before the first electronic step) on the isolated Pd and Se atoms (20 Å box,
this example's `DFT/isolated_elements` POSCARs and recipe) — natively too, on any rank
count including 1, with Γ-only k-points, with `vasp_gam`; `LREAL`, `NBANDS`, `ISPIN`,
`IVDW` and `KPAR` made no difference, and a 10 Å box ran. The cause was the stack limit:
with `ulimit -s unlimited` (default there: 300 MB) both atoms converge — natively, and
through the pipeline with the driver fix below (Flux tasks there have a 300 MB soft and
an unlimited hard stack limit, so raising the soft limit needs no privileges).

**Here.** These atoms ran fine on Perlmutter with `vasp/6.4.2-gpu`, so it may not apply.
If VASP ever dies with exit 139 and no electronic steps on a large cell, raise the stack
limit. The Frontier driver (`examples/Frontier/VASP/DFT/vasp_dft.py`,
`_unlimited_stack`) does it for `vasp_std` only, via `preexec_fn` in the `subprocess.run`
call — a one-function change to this example's `DFT/vasp_dft.py`.

**Check.** `ulimit -s` inside a chore (e.g. a one-line test chore), and whether the
default ever drops below what VASP needs for the largest vacuum cell you run.

## 8. Flux `libpmi` first on a replaced library path — already handled here

For reference, since it cost time on Frontier: `vasp_dft.py` *replaces*
`LD_LIBRARY_PATH` with `VASP_LD_LIBRARY_PATH` for `vasp_std`. Flux's shell normally
prepends its `libpmi` directory to every task's `LD_LIBRARY_PATH`, and replacing the
variable drops it — every rank then runs as its own 1-rank job (`running 1 mpi-ranks` in
`OUTCAR`, ranks colliding on `vaspout.h5`'s lock). This example's
`VASP_LD_LIBRARY_PATH` already starts with `/usr/local/lib/flux`, which is exactly the
fix; keep it first whenever that path is edited.

## Not applicable to Perlmutter (Frontier-only)

For completeness — these were real on Frontier but come from its MPI/ROCm stack, not
from shared pipeline code: `MPICH_OFI_NIC_POLICY=NUMA` aborting multi-node `MPI_Init`;
`FI_HMEM`/ROCm GPU visibility after a multi-node `MPI_Init`; running a host-built binary
against the image's ROCm instead of its build ROCm; CPU/GPU placement under Flux (Flux
hands out GPUs in index order — Perlmutter uses `set_gpu_affinity=True` with
`CUDA_VISIBLE_DEVICES`, which is fine with 4 equidistant GPUs per node). Details:
`examples/Frontier/RMG_MACE_ASE/README.md` steps 3b and 9, `examples/Frontier/VASP/README.md`.
