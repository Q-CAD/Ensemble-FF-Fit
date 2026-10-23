# Building a new workflow with Ensemble-FF-Fit

Read this before writing any new pipeline code. It's the fast path to a working deployment —
most of what you need already exists somewhere in this repo; the job is usually to find and
adapt it, not to write it from scratch.

## 1. Plan before you build

Don't start writing driver scripts or a `run_pipeline.py` on your own judgment call. First, with
the person you're working for:

- Identify which existing example (below) is the closest structural match to what they want —
  same DFT code, same FF-fitting backend, same MD backend, or close to it.
- Agree on the stage sequence explicitly (e.g. "generate structures → relax → finite-temperature
  MD → single points → build training set → fit ensemble → validate → select") *before*
  implementing even the first stage. Write it down as a short plan, not just in your own head.
- Agree on what's being reused as-is, what's being adapted, and what's genuinely new. If more
  than half the pipeline would be "genuinely new," that's worth flagging explicitly — it's a sign
  either the closest example is a poor match, or the new backend needs a real design conversation
  first, not just an implementation pass.
- Bias toward building and validating one stage at a time, smoke-testing on the smallest
  allocation that can prove it works (a single structure, a single ensemble member, a short
  step count) before scaling to the real run. Dry-run anything with a tunable batching parameter
  (e.g. `MDMatEnsemble.build_task_dicts`'s `parent_levels`) against the real staged tree rather
  than assuming a value from a different example carries over.

## 2. Conserve: reuse before you rewrite

- **A new backend is a new driver script, not a `base.py` change.** `MDMatEnsemble`/
  `FFMatEnsemble`/`DFTMatEnsemble` are stable, generic dispatchers — see `CLAUDE.md`'s own
  Architecture section for their fixed entry-point contracts. Supporting a new FF code, DFT code,
  or MD engine almost always means writing one new driver script matching the relevant contract,
  not modifying the dispatcher classes themselves.
- **Copy the closest example's driver script and adapt it**, rather than writing one from a blank
  file — e.g. a new LAMMPS pair-style driver should start from an existing
  `lammps_*_single_point.py`/`lammps_*_md.py`, not from reading `base.py` and improvising the
  contract.
- **Check `EnsembleFFFit`'s shared helpers before writing your own**:
  `molecular_dynamics/helpers.py` (`get_elements`, `import_lammps`/`import_lammps_mliap`,
  `parse_list`), `potential/{ace,mace}/build_*_ensemble_inputs.py` (seed/kappa ensemble sampling,
  dataset conversion), `analysis/` (`dict_parsers`, `variance`, `best_force_field`,
  `downselect_force_fields`). These already solve the common cross-cutting problems.
- **Copy the closest example's `run_pipeline.py`/`workflow_config.yaml` as your own project's
  starting skeleton.** The per-stage-function-registered-in-a-`STAGES`-dict pattern, invoked via
  `--stage <name>`, is itself a reusable template, not something to redesign per project.
- **New deployments usually live in their own project directory**, outside this repo, importing
  `EnsembleFFFit` as a dependency — not as a fourth folder under `examples/`, unless you're
  specifically contributing back a new reference example for others to learn from.

## 3. Pick an example

| Example | DFT | FF fitting | MD | Platform | Pick this if... |
|---|---|---|---|---|---|
| `Frontier/RMG_MACE_ASE/` | RMG | MACE | ASE | OLCF Frontier, ROCm | You want the most complete loop — the only one with active-learning downselection wired up end-to-end. |
| `Perlmutter/VASP_ReaxFF_LAMMPs/` | VASP | JAX-ReaxFF | LAMMPS | NERSC Perlmutter, CUDA | You want LAMMPS-based MD/validation and the multi-node Flux launch pattern (`launch_multi_node.slurm`). |
| `Perlmutter/VASP_MACE_ASE/` | VASP | MACE | ASE | NERSC Perlmutter, CUDA | You want `Frontier/RMG_MACE_ASE`'s loop with VASP, on CUDA/podman-hpc. |
| `Frontier/VASP/` | VASP | — | — | OLCF Frontier, ROCm/Apptainer | You want GPU VASP chores on Frontier (single- or multi-node) — the DFT stage only. |
| `HPC_aas/QE_ACE/` | Quantum Espresso | ACE/pyace | — | ORNL Pathfinder, CUDA/Apptainer | You want a minimal, all-CPU smoke-test pattern, or are fitting ACE specifically. |

Read the target example's own `README.md` end-to-end, including its pitfalls section, before
adapting anything from it.

## 4. Known pitfalls

- **A host-built DFT binary run inside a container must load the libraries it was built
  against** (its ROCm/CUDA, compiler runtime, MPI) — not the image's same-named ones. A
  partial mismatch can look like a solver-specific bug: on Frontier, RMG's davidson ran
  against the image's ROCm while its multigrid died with `hipErrorInvalidDeviceFunction`
  (`Frontier/RMG_MACE_ASE/README.md` step 3b). `ldd` the binary after loading its build
  modules, and bind/path exactly those.
- **Multi-node MPI under Flux on Frontier**: Cray MPICH's `MPICH_OFI_NIC_POLICY=NUMA`
  aborts `MPI_Init` on >1 node (Flux tasks aren't NUMA-confined) — use `BLOCK`; start
  the brokers with `srun --cpu-bind=none`; and if a driver *replaces* `LD_LIBRARY_PATH`
  for its binary, put Flux's `/usr/local/lib/flux` first again or every rank runs alone
  (`Frontier/RMG_MACE_ASE/README.md` section 9, `Frontier/VASP/README.md`).
- **MACE ensemble members can have fewer training configs than `batch_size`** (small
  label combinations); MACE then fails with `torch.cat(): expected a non-empty list of
  Tensors`. `Frontier/RMG_MACE_ASE/FF/mace_fit.py` shrinks the batch to fit.

- **Relative paths inside a per-member config file can resolve against the wrong directory.**
  Several fitting drivers (e.g. pyace's `GeneralACEFit`) `chdir()` into their own per-member work
  directory before reading config — a relative `data.filename`/similar then resolves against
  *that* directory, not wherever you built the config from. Always `.resolve()` paths before
  writing them into a config file a chore will read later.
