# Ensemble-FF-Fit
![alt text](Ensemble-FF-Fit.png?raw=true)

This package allows **data** and **time efficient fine-tuning of physics-based as well as machine-learning interatomic potentials (MLIPS)**, including universal MLIPs, by using an ensemble approach to fit MLIPs to ab initio data using adaptive asynchronous job scheduling while incorporating UQ on-the-fly. to generate new training data to improve the force-fields. 

MatEnsemble is used to perform adaptive asynchronous job scheduling. DFT convergence is driven the same
way -- via RMG (PyRMG, https://github.com/RMGDFT/rmgdft), VASP, or Quantum Espresso, depending on the
example.

Three force-field-fitting backends are demonstrated end-to-end: **MACE**, **JAX-ReaxFF**, and **ACE**
(pyace/pacemaker) -- all through the same generic driver-script contract, so adding a fourth means
writing a new driver script, not changing this package. See `examples/` (one full worked pipeline per
backend/HPC-system combination) for what's actually been run and validated.

Future plans: Include support for universal ML-FFs such as CHGNET and M3GNet for quantum materials.  

## Installation

Install is a two-step process, because `pyproject.toml` cannot express "fetch this package from a
different index depending on what GPU hardware/drivers are present on this machine" -- that's exactly
what step 1 handles.

**Step 1: install the right GPU build of PyTorch for this machine.**

```bash
python install_gpu_torch.py                     # auto-detects ROCm or CUDA and installs the matching torch
# or, to skip detection and specify explicitly:
python install_gpu_torch.py --platform rocm6.3   # e.g. on Frontier
python install_gpu_torch.py --platform cuda12.4  # e.g. on Perlmutter
python install_gpu_torch.py --list-platforms     # see all supported platform keys
```

This fails loudly (non-zero exit, clear message) rather than guessing if it can't confidently detect your
platform, or if it detects signals for *both* ROCm and CUDA -- a wrong GPU build here causes silent,
hard-to-diagnose runtime failures rather than install-time errors, so an explicit error is much better
than a bad guess. If detection fails or picks the wrong thing, use `--platform` to override it.

**Step 2: install this package plus whichever force-field backend(s) and GPU-platform extra you need.**

The two are orthogonal -- pick one backend extra and one platform extra (or neither platform extra, if
you're CPU-only):

```bash
pip install -e ".[mace,cuda]"        # MACE, CUDA-accelerated (cuequivariance)
pip install -e ".[mace,rocm]"        # MACE, ROCm-accelerated (openequivariance)
pip install -e ".[mace]"             # MACE, no GPU acceleration library beyond torch itself
pip install -e ".[jaxreaxff]"        # JAX-ReaxFF fitting (Q-CAD/JAX-ReaxFF + parse2fit)
CMAKE_BUILD_PARALLEL_LEVEL=2 pip install -e ".[ace]"   # ACE/pyace fitting -- caps native-extension
                                                        # build parallelism; see CLAUDE.md for why
pip install -e ".[torchsim]"         # experimental Torch Sim MD driver
pip install -e ".[notebooks]"        # jupyter/py3dmol/ipykernel, for interactive/notebook work
pip install -e ".[dev]"              # pytest/black/ruff
pip install -e .                     # core only: structure generation + analysis, no MLIP backend
```

Extras can be combined, e.g. `pip install -e ".[mace,cuda,notebooks]"`. `ace` has no GPU-accelerated
counterpart to combine with `cuda`/`rocm` here -- see CLAUDE.md for why (pyace's own GPU evaluator,
`tensorpotential`, needs Python <3.11).

The official LAMMPS Python module itself is **not** installed by this package at all -- both Frontier and
Perlmutter containers are expected to provide LAMMPS with its Python bindings already built in. See
`TODO.md` if that assumption turns out to be wrong once containerized MatEnsemble+LAMMPS deployment is
fully worked out.

See `CLAUDE.md` for more on the project's architecture, and `TODO.md` for what's still open. Full worked
examples, one per backend/HPC-system combination:

- `examples/Frontier/RMG_MACE_ASE/README.md` -- RMG DFT -> MACE fitting -> ASE MD, on OLCF Frontier.
- `examples/Perlmutter/VASP_ReaxFF_LAMMPs/README.md` -- VASP DFT -> JAX-ReaxFF fitting -> LAMMPS MD, on
  NERSC Perlmutter.
- `examples/HPC_aas/QE_ACE/README.md` -- Quantum Espresso DFT -> pyACE fitting -> CPU-only single-point
  validation, on ORNL Pathfinder.
