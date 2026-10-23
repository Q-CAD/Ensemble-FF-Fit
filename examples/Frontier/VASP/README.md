# VASP on Frontier through MatEnsemble/Flux, in the MatEnsemble container

A short example: OLCF's GPU VASP module (`vasp/6.6.1-gpu`) run as MatEnsemble chores
inside the MatEnsemble Apptainer image, one MPI rank per GCD, on one or more nodes —
the `converge_dft_data` stage of the Perlmutter
[`VASP_MACE_ASE`](../../Perlmutter/VASP_MACE_ASE/README.md) example, ported to
Frontier. The rest of a VASP → MACE → ASE loop on Frontier is the same as
[`RMG_MACE_ASE`](../RMG_MACE_ASE/README.md)'s stages from `sample_ft_md_structures` on;
this example only establishes the DFT step: that VASP finds its GPUs, joins one MPI job
across nodes, and runs as fast as natively.

| File | What it's for |
|---|---|
| `launch_flux.slurm` | Start Flux in the image across the allocation and run the pipeline |
| `run_pipeline.py`, `workflow_config.yaml` | The `converge_dft_data` stage (VASP) and its config |
| `DFT/vasp_dft.py` | Driver: pymatgen input generation, runs `vasp_std`, writes `properties.json` (Perlmutter's, plus per-rank GPU selection) |
| `DFT/vdW_single_point.yml` | Recipe: MPRelaxSet + IVDW=12 static single point (Perlmutter's settings) |
| `DFT/isolated_elements/{Pd,Se}`, `DFT/amorphous/Pd18Se60` | Test structures: two isolated atoms and a 78-atom amorphous frame from the Perlmutter example's training data |

## 1. Container image

The same image as `RMG_MACE_ASE` — see its README step 1: `build_image.slurm` pulls the
MatEnsemble Frontier image and builds the derived image from
`matensemble-frontier.def`, which adds the Python packages these pipelines need
(pymatgen and pymatgen-analysis-defects for VASP input generation and parsing are
checked at build time, alongside torch/MACE). Nothing VASP-specific goes into the
image: VASP is the host module, bind-mounted in by `launch_flux.slurm`.

Set `SIF`/`ENSEMBLE_FF_FIT` in `launch_flux.slurm` (or the environment) to your
derived `.sif` and Ensemble-FF-Fit checkout.

## 2. VASP access and POTCARs

- You need to be in OLCF's `vasp` group (the install under `/sw/frontier/vasp` is
  group-restricted); `module avail vasp` shows the builds.
- POTCARs: Frontier ships VASP's PBE.54 set as `/sw/frontier/vasp/potentials/PBE/<symbol>/POTCAR`.
  pymatgen (`potcar_functional: PBE_54_W_HASH`) looks for a `POTPAW_PBE_54` directory,
  so make one once and point the recipe's `pseudopotentials_directory` at its parent:

  ```bash
  mkdir -p /lustre/orion/<proj>/scratch/$USER/vasp_pseudos
  ln -s /sw/frontier/vasp/potentials/PBE /lustre/orion/<proj>/scratch/$USER/vasp_pseudos/POTPAW_PBE_54
  ```

  pymatgen's hash check accepts these POTCARs (checked for Pd, Se).

## 3. Launch

```bash
STAGE=converge_dft_data sbatch -N 2 launch_flux.slurm   # from this directory
```

`launch_flux.slurm` is `RMG_MACE_ASE`'s launcher (one Flux broker per node, started with
`srun --external-launcher --mpi=pmi2 --cpu-bind=none`), with the host paths `vasp_std`
needs bound in:

| Bind | For |
|---|---|
| `/opt/cray` | Cray PE (MPICH, LibSci, FFTW, HDF5), CCE 20.0.2 runtime, libfabric, PALS |
| `/opt/rocm-6.4.2` | the ROCm `vasp_std` was built with (rocBLAS/rocFFT/rocSOLVER/RCCL) |
| `/opt/amdgpu` → `/opt/host_amdgpu`, `/opt/xpmem` → `/opt/host_xpmem` | host libdrm and xpmem — separate mount points, since the image has its own at those paths |
| `/sw/borg` | the Darshan library `vasp_std` links |
| `/usr/lib64` → `/opt/cray_extra/lib64` | `libcxi` and other host system libraries |

`run_pipeline.py`'s `VASP_LD_LIBRARY_PATH` lists those directories for `vasp_std`'s
own subprocess only (never Flux or Python), host libraries first, then the image's
glibc/libstdc++ (newer than the host's — so, unlike Perlmutter, no host `ld-linux`
loader trick is needed), host `/usr/lib64` last. When the VASP module changes, redo
`module load vasp/<version>; ldd $(which vasp_std)` and update the binds and that path.

## 4. Configure

`workflow_config.yaml`'s `converge_dft_data`:

- `ranks_per_node: 8`, `cores_per_task: 7`, `gpus_per_task: 1` — one rank per GCD, each
  on one whole 7-core L3 group; `DFT/vasp_dft.py` gives each rank the GCD closest to
  its cores (`VASP_GPU_CLOSEST_GCDS`), as native `srun --gpu-bind=closest` does.
  `controller_cores: 0` lets 8 × 7 fill a 1-node instance.
- `omp_num_threads: 1` (VASP's GPU build), `gpu_aware_mpi: true`
  (`MPICH_GPU_SUPPORT_ENABLED=1`).
- Node count per structure: the recipe's `nodes` (0 = `ceil(atoms / atoms_per_node)`),
  capped at `max_nodes_per_job`.

## 5. Measured

October 2026, `vasp/6.6.1-gpu`, one rank per GCD, same nodes for each native/container
pair. The 78-atom Pd18Se60 amorphous cell (21 electronic steps every time):

| | native `srun` | container, direct `flux run` | container, `converge_dft_data` |
|---|---|---|---|
| 1 node (8 ranks) | 211.6 s (loop 151.5 s) | 232.9 s (loop 157.4 s) | — |
| 2 nodes (16 ranks) | 167.1 s (loop 113.1 s) | 177.3 s (loop 113.8 s) | 190.2 s |
| free energy | -336.42512531 eV | same | same |

So the container matches native inside VASP's electronic loop (≤ 4% on 1 node, 0.6% on
2) and adds ~10-25 s of startup; the pipeline adds a little more (each of the 16 chore
copies writes the inputs with pymatgen before starting `vasp_std`). `properties.json`'s
energy is pymatgen's `Vasprun.final_energy` (-336.41535207 eV here, the σ→0 value), as
in the Perlmutter example. For scale: Perlmutter (`vasp/6.4.2-gpu`, 4 A100s) took 106 s
on its 80-atom loop candidates, 84 s on 2 nodes.

Isolated atoms (20 Å box, 10×10×10 k-mesh from the recipe's `grid_density: 1000`), 1
node, 8 ranks, with the stack limit raised (below):

| | native `srun` | container, `converge_dft_data` |
|---|---|---|
| Pd | 36 electronic steps, 189 s | 36 steps, 229 s, same energy |
| Se | 23 steps, 151 s | 23 steps, 183 s |

(Perlmutter: 225 s / 154 s.) Without the stack fix both segfault before the first
electronic step, natively and in the container.

## 6. Differences from the Perlmutter example, and pitfalls

- **Flux's `libpmi` first on `vasp_std`'s library path** (`/usr/local/lib/flux` at the
  head of `VASP_LD_LIBRARY_PATH`), as on Perlmutter. Flux's shell normally prepends it
  to every task's `LD_LIBRARY_PATH`, but `vasp_dft.py` replaces that variable for
  `vasp_std`; without it each rank runs as its own 1-rank job (`running 1 mpi-ranks` in
  `OUTCAR`, ranks failing on `vaspout.h5`'s file lock — `vhdf5.F` internal error). With
  it, `mpi=False` is enough (`running 16 mpi-ranks ... on 2 nodes`).
- **`MPICH_OFI_NIC_POLICY=BLOCK`**, not `NUMA`: under Flux, ranks aren't confined to one
  NUMA domain and Cray MPICH aborts `MPI_Init` on >1 node with `NUMA`
  (`RMG_MACE_ASE` README section 9).
- **GPU-aware MPI works across nodes as is**, because `vasp_std` runs entirely on the
  host ROCm it was built with. (RMG, while it ran on the image's ROCm, lost its GPUs
  after a multi-node `MPI_Init` and needed `FI_HMEM=system`; on its build ROCm it doesn't.)
- **`set_gpu_affinity=False`**: Flux's GPU affinity hands GPUs out in index order, which
  would clash with `vasp_dft.py`'s per-rank `ROCR_VISIBLE_DEVICES` (and put most ranks
  on a far GCD).
- **Stack size: `vasp_std` segfaults on large vacuum cells** (an isolated atom in a
  20 Å box: `Segmentation fault` right after `EWALD` in `OUTCAR`, before the first
  electronic step) with the default 300 MB stack limit — natively too, on any rank
  count (even 1), with Γ-only k-points, with `vasp_gam`, and with `vasp/6.6.0-gpu`;
  `LREAL`, `NBANDS`, `ISPIN`, `IVDW`, `KPAR` make no difference, a 10 Å box runs.
  With `ulimit -s unlimited` it converges. `vasp_dft.py` raises `vasp_std`'s stack
  limit to the hard limit before starting it — inside Flux tasks the soft limit is
  300 MB and the hard limit unlimited, so no privileges are needed (confirmed through
  the pipeline: both isolated atoms converge).
- **`HSA_XNACK=0`** for `vasp_std` (the image sets 1 for LAMMPS; native runs have it unset).
- Stress sign: `vasp_dft.py` negates pymatgen's (VASP, positive-compressive) stress into
  ASE/MACE's convention, as on Perlmutter.
