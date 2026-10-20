# TODO.md

Genuinely open items — not yet addressed by any build/example in this repo. Resolved/retired items have
been removed from this file as they're settled; see git history if the old context is ever needed.

## Torch Sim compatibility

`EnsembleFFFit/molecular_dynamics/helpers.py`'s `make_prop_calculators` has `"kinetic_energy"`/
`"temperature"` branches that call `calc_kinetic_energy`/`calc_temperature` from `torch_sim.quantities`,
but that import is commented out at the top of the file. This is intentional for now — Torch Sim isn't
standardly available across the HPC container/runtime builds this package targets, and the new `torchsim`
extra in `pyproject.toml` (`torch-sim-atomistic`) is a separate opt-in for exactly this reason. Revisit
once Torch Sim is standardized in those environments; until then, calling `make_prop_calculators` with a
mapping that includes `"kinetic_energy"` or `"temperature"` will raise `NameError`. No TorchSim example
exists anywhere in this repo yet either (see the QE_ACE entry below for why that example in particular
dropped it) — a real TorchSim example (most plausibly MACE-based, since TorchSim's own model zoo is built
around MACE-shaped GNN potentials) is still worth doing as its own thing.

## Perlmutter: direct pip install of Ensemble-FF-Fit + dependencies, not yet tried end-to-end

The extras-based `pyproject.toml` install path (`python install_gpu_torch.py` + `pip install -e
".[...]"`) has been validated end-to-end on Frontier (the RMG -> MACE -> ASE pipeline in
`examples/Frontier/RMG_MACE_ASE/`) but never via a fresh install on Perlmutter/CUDA — every Perlmutter
deployment so far (including a real, extensive Pd-Se/ACE+LAMMPS build) has run inside an already-built
container image, not through installing this repo from scratch. That Pd-Se work did directly confirm one
long-standing open question: a real Perlmutter container with LAMMPS's Python bindings (`import lammps`/
`import lammps.mliap`) already linked in, needing no pip install of `lammps`/`mpi4py` at all, does exist
and work the same way Frontier's does. Still open:
- Whether `install_gpu_torch.py`'s `cuda` extra mapping (pinned to `cu124`/`torch==2.6.0`) is still the
  right pin for a container actually built via this repo's own documented install path — the confirmation
  above came from a separately-built image, not from exercising `install_gpu_torch.py` itself.
- Whether the deleted `build_*.sh` scripts handled anything (e.g. `build_lammps.sh`'s Kokkos/cmake flags,
  module loads, or its `sed` hack for an nvcc flag CMake used to generate incorrectly) that the new
  extras-based path doesn't — recover the relevant script from `main`/git history rather than
  reconstructing it from memory if this turns up a real gap.

## Pathfinder QE/ACE pipeline (`examples/HPC_aas/QE_ACE/`) — still open

Surfaced building/validating this example end-to-end (see its own README's pitfalls section for the
already-resolved issues); these are the ones still genuinely open, not yet decided/fixed.

- **No TorchSim example anywhere in this repo yet**, despite `torchsim` being a real, installable extra.
  The original plan for this example was QE (DFT) -> pyACE (fitting) -> TorchSim (GPU MD) — TorchSim was
  dropped partway through once it became clear (checked directly, not assumed) that TorchSim's own model
  zoo has no ACE entry and no generic "wrap an arbitrary ASE calculator as a batched GPU model" bridge, so
  it couldn't do anything for ACE specifically that plain ASE + `PyACECalculator` doesn't already do,
  just without the GPU.
- **LAMMPS on Pathfinder's container has no `ML-PACE` package built in** — confirmed via
  `lammps.lammps().installed_packages`. This blocks the two things that would actually make ACE MD
  competitive with MACE-on-GPU for long-timescale/large-supercell dynamics: real MPI domain-decomposed
  parallelism (pyace's own native evaluator has no internal thread parallelism at all — confirmed via
  `ldd` — so a plain-ASE MD driver is single-core, full stop) and much larger systems than fit comfortably
  in one process's memory. Building LAMMPS here with `-DPKG_ML-PACE=yes` linked against pyace's own C++
  library is the natural next step before attempting a real (not toy-scale) ACE MD demonstration.
- **Multi-rank MPI under Flux on Pathfinder is untested/confirmed-not-working**, not just unconfigured —
  real multi-rank `pw.x` (multiple MPI ranks cooperating on one calculation) was tried directly and
  ranks silently fell back to independent singletons rather than actually communicating; a genuine
  Flux/PMIx-interop gap (Pathfinder's OpenMPI 5.0.5 bootstraps via PMIx, unlike Frontier's Cray MPICH/PMI2
  or Perlmutter's stack), not chased to a root cause. Blocks any DFT structure needing more than one
  node's worth of k-point-pool parallelism, and blocks a LAMMPS-ML-PACE MD demo from actually using more
  than one node too, once that's built.
- **The QE_ACE training set is a 2-structure smoke test, not a real force field.** Confirmed directly (not
  assumed): even a colder, shorter MD sanity check on the current 6-member ensemble diverges within
  ~100-200 steps regardless of starting temperature. There's also no held-out validation set yet — every
  `single_points` result checks a fit against data it was trained on, not generalization. Both need
  addressing (more DFT structures across more of Si's configuration space; a genuine train/validation
  split, mirroring `DFT/validation/` in the other two examples) before this pipeline demonstrates anything
  about force-field *quality*, as opposed to demonstrating that the *plumbing* works.
- **No GPU path exists for ACE anywhere in this repo.** pyace's own GPU evaluator (`tensorpotential`)
  needs Python <3.11, incompatible with every container this project currently builds (all Python 3.12).
  Getting GPU-accelerated ACE (fitting or MD) would need either a separate, older-Python sidecar
  environment just for that evaluator, or accepting CPU-only ACE as a permanent characteristic of this
  backend rather than a temporary gap — worth deciding explicitly rather than leaving implicit.
