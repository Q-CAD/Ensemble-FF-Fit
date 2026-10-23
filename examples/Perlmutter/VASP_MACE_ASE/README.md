# VASP -> MACE -> ASE: a Pd-Se active-learning loop on Perlmutter

This example runs a full MACE fine-tuning / uncertainty-quantification loop for the
Pd-Se system on NERSC Perlmutter, coordinated by a single script/config pair
(`run_pipeline.py` / `workflow_config.yaml`) via MatEnsemble/Flux:

1. **VASP** isolated-atom calculations (Pd, Se) for MACE's E0s — and, from the second
   loop on, the DFT candidates the previous loop selected.
2. **MACE** ensemble fine-tuning of the MACE-OMat-0 (medium) foundation model: each
   ensemble member trains on a sampled combination of five pre-computed Pd-Se
   training subsets with its own energy/force/stress weights.
3. **Validation**: every fitted model is evaluated on held-out Pd-Se AIMD trajectories
   (VASP single points) and ranked by combined energy/force error.
4. **ASE** finite-temperature MD (foundation model) from frames sampled out of the
   training set, then single points of every MD frame with a *downselected subset*
   of the ranked ensemble.
5. **Uncertainty-based selection** of the frames the ensemble disagrees on most, written
   as POSCARs for the next loop's VASP calculations.

It is the Frontier [`RMG_MACE_ASE`](../../Frontier/RMG_MACE_ASE/README.md) pipeline with
VASP swapped in from [`VASP_ReaxFF_LAMMPs`](../VASP_ReaxFF_LAMMPs/README.md) (driver,
recipe, container launch). Read those two READMEs' pitfalls sections if you deviate
from what's here.

## 0. What's included vs. what you provide

- **`DFT/training/<label>/<label>.xyz`** — 623 VASP frames in five subsets: `amorphous`
  (126), `EoS` (110), `exfoliation` (33), `thermal_cluster` (132), `thermal_subbed`
  (222). Each subset is one combinable "label" when ensemble inputs are built.
- **`DFT/testing/Pd-Se/<composition>/<mp-id>.xyz`** — 8 AIMD trajectories, 147 VASP
  single-point frames (20 per trajectory, except Pd34Se11: 7 — the other 13 frames'
  `vasprun.xml` never finished). Used as every ensemble member's `test.xyz` and as
  the ranking reference.
- Both were converted once by **`DFT/prepare_pd_se_data.py`** from the original
  parse2fit/FitSNAP `.xyz` files and AIMD `vasprun.xml` files (paths in its docstring).
  The FitSNAP stress (bar, VASP sign) was converted to MACE/ASE's eV/Å³ and opposite
  sign; the script checks that convention against ASE's own `vasprun.xml` reader.
  All data used the same VASP settings as `DFT/vdW_single_point.yml` (MPRelaxSet +
  IVDW=12, static) — keep them consistent if you add data.
- **`DFT/isolated_elements/{Pd,Se}/POSCAR`** — single atoms in a 20 Å box, converged by
  the first stage.
- **Not included:** the MACE foundation model (step 3), VASP POTCARs (licensed — step
  2), and any run output.

## 1. Build the container

`Dockerfile.mace` is the VASP example's `Dockerfile.ff-fit` (NERSC Flux base +
MatEnsemble) without JAX-ReaxFF/parse2fit, plus CUDA torch, `mace-torch` and
cuequivariance. From this directory:

```bash
podman-hpc build -f Dockerfile.mace -t matensemble:mace .
podman-hpc migrate matensemble:mace
```

Ensemble-FF-Fit itself is *not* installed in the image: `launch_multi_node.slurm`
bind-mounts your checkout and puts it on `PYTHONPATH` (set `ENSEMBLE_FF_FIT` there,
or in your environment, if your checkout isn't at the default path). VASP isn't in
the image either — it's the host's `vasp/6.4.2-gpu` module, bind-mounted in.

## 2. VASP setup

- **`DFT/vdW_single_point.yml`**: `vasp_executable`/`command` point at
  `vasp/6.4.2-gpu`; `pseudopotentials_directory` must point at a pymatgen-layout POTCAR
  directory (`POTPAW_PBE_54/POTCAR.Pd.gz`, ...) you have access to, **under
  `$SCRATCH`, not `/global/homes`** (the latter doesn't bind-mount reliably into
  rootless containers on compute nodes).
- You need to be in NERSC's VASP user group (`vasp_std` is group-restricted).
- Switching VASP versions, and every container-launch flag VASP needs, is covered in
  the VASP example's README, step 3 — `launch_multi_node.slurm` here is that
  example's script with only the image name/config/stage changed.

## 3. Fetch the MACE foundation model

```bash
curl -L -o model.model \
    https://github.com/ACEsuit/mace-foundations/releases/download/mace_omat_0/mace-omat-0-medium.model
cp model.model FF/generation_0/mace-omat-medium/model.model                    # reference copy
cp model.model MD/finite_temperature/uq/generation_0/mace-omat-medium/model.model  # FT-MD model
mkdir -p FF/generation_1/mace-omat-medium
cp model.model FF/generation_1/mace-omat-medium/model.model                    # loop-1 fine-tuning seed
```

Keep `FF/generation_1/mace-omat-medium/` free of any other `model.model`:
`FFMatEnsemble` cross-products *every* `model.model` it finds under
`fine_tuning.run_directory` against every input folder.

## 4. Run one loop

The script works under `salloc` (interactive QOS, which only accepts `salloc`) or
`sbatch` (e.g. `regular` QOS — edit its `#SBATCH -A`/`--qos`/`-t`/`-N` lines for your
allocation). From this directory, e.g. on 4 interactive nodes:

```bash
salloc -A m2113_g -C gpu -q interactive -N 4 -t 04:00:00 --ntasks-per-node=1 \
    --gpus-per-node=4 --gpu-bind=closest bash launch_multi_node.slurm
# CONFIG=workflow_config.yaml STAGE=all are the defaults; override as env vars
```

That runs `--stage all`, i.e. these stages in order:

| Stage | What it does here |
|---|---|
| `converge_dft_data` | VASP on every `POSCAR` under `DFT/` without a `properties.json` — the two isolated atoms in loop 1 |
| `sample_ft_md_structures` | 20 frames sampled evenly from the amorphous/EoS/exfoliation/thermal_subbed training sets → FT-MD starting POSCARs |
| `build_ff_inputs` | 100 sampled (label combination × weights) MACE input folders; fixed AIMD `test.xyz`; E0s from the isolated atoms |
| `fit_and_validate` | 100 MACE fits; each finished fit's single points on all 147 AIMD frames; FT-MD with the foundation model (supercell + defect cycles) |
| `downselect` | ranks all fits against the AIMD DFT data, keeps 25 (`strategy: spread`) for UQ; unpacks MD frames |
| `uq_single_points` | the 25 kept models × every MD frame |
| `select_dft_candidates` | the 80 most-disagreed-upon, well-spread frames → `DFT/uq/1/<i>/POSCAR` |

Rerunning `--stage all` resumes after a walltime limit: finished VASP runs, fits, MD
runs and single points are all skipped (confirmed in the smoke test). Single stages
can also be run with `STAGE=<name>`.

**Smoke-test first.** Copy this directory somewhere scratch (outputs are written in
place), copy `workflow_config.yaml` to e.g. `workflow_config.smoke.yaml`, and shrink it —
identical wiring, just smaller:

| Key | Full | Smoke |
|---|---|---|
| `converge_dft_data.cores_per_task` | 16 | 15 (1 node — see pitfalls) |
| `sample_ft_md_structures.total` | 20 | 4 |
| `build_ff_inputs.total_cap` | 100 | 3 |
| `build_ff_inputs.mace_overrides.max_num_epochs` | (100, the default) | 2 |
| `finite_temperature_md.in_file` | `.../in.mace.json` | `.../in.mace.smoke.json` (100 steps, 1 cycle) |
| `downselect_force_fields.number_to_copy` | 25 | 2 |
| `dft_candidate_selection.total` | 80 | 2 |

```bash
CONFIG=workflow_config.smoke.yaml salloc -A m2113_g -C gpu -q interactive -N 1 -t 01:00:00 \
    --ntasks-per-node=1 --gpus-per-node=4 --gpu-bind=closest bash launch_multi_node.slurm
```

**Measured (smoke test, 1 node, 2026-10-22):** loop 1 ≈ 15 min, loop 2 ≈ 10 min. The
isolated-atom VASP runs took 154 s (Se) and 225 s (Pd), and the two 80-atom candidates
≈ 106 s each (4 A100s per run, so one at a time on a single node). A MACE fit spent ≈ 95 s on its first epoch (one-time warm-up)
and ≈ 2 s per epoch after that. **Full config, rough estimate (not measured):** 100
fits × 100 epochs is ~5-10 min per fit, i.e. roughly 1 h on 4 nodes (16 GPUs), with
FT-MD (20 runs × 30,000 steps) and UQ (25 models × every MD frame) sharing those GPUs.
One loop should fit a 4-hour interactive allocation on 4 nodes; the next loop's 80
VASP candidates add about 2 min each, 4 at a time on 4 nodes.

## 5. The next loop

Run `--stage all` again after updating `workflow_config.yaml` for the new generation.
Every generation-specific location is tagged `# GEN` in the config; for loop 2:

| Key | Loop 1 | Loop 2 |
|---|---|---|
| `fine_tuning.run_directory` | `FF/generation_1/mace-omat-medium` | `FF/generation_2/mace-omat-medium` (seed it with a `model.model` first, as in step 3) |
| `fine_tuning.inputs_directory` | `FF/generation_1/mace_inputs` | `FF/generation_2/mace_inputs` |
| `build_ff_inputs.extra_training_subpaths` | `[]` | `[uq/1]` (converts loop 1's new DFT runs into a `uq_1` training label) |
| `build_ff_inputs.always_include_labels` | `[]` | `[uq_1]` (every ensemble member trains on it — otherwise it's sampled like any other label, and in the smoke test only 1 of 3 members drew it) |
| `finite_temperature_md.structures_subpath` | `structures/generation_1` | `structures/generation_2` |
| `finite_temperature_md.output_directory` | `MD/finite_temperature/uq/generation_1/...` | `.../generation_2/...` |
| `copy_force_fields.target_directory` | `MD/single_points/validation/generation_1/force_fields` | `.../generation_2/...` |
| `MD_uq_single_points.run_directory` | `MD/single_points/uq/generation_1/force_fields` | `.../generation_2/...` |
| `MD_uq_single_points.inputs_directory` | `MD/single_points/uq/generation_1/ase_inputs` | `.../generation_2/...` |
| `dft_candidate_selection.dest_root` | `DFT/uq/1` | `DFT/uq/2` |

The rerun's first stage, `converge_dft_data`, runs VASP on the `DFT/uq/1` POSCARs
(everything already converged is skipped); `build_ff_inputs` then folds those results
in. Optionally also change `sample_ft_md_structures.seed` (different MD starting
frames) or point `finite_temperature_md.foundation_model` at a loop-1 fitted model.
Downstream path keys left `null` in the config default to these, so nothing else
needs editing.

## 6. Pitfalls (Perlmutter-specific, inherited from the VASP example unless noted)

- **Every `Pipeline` uses `reserve_broker_node=False`, every submit
  `set_cpu_affinity=False`.** The latter avoids flux-shell's live topology probe
  failing against the static `R.json` (`coreNNN not in topology`).
- **VASP chores need `mpi=True`** (PMI2 wiring), unlike RMG on Frontier — otherwise
  each rank runs as its own 1-rank job, colliding in one directory. Jobs are sized one
  MPI rank per GPU (`nodes × ranks_per_node`, `ranks_per_node: 4`), up to
  `max_nodes_per_job: 4`. **Multi-node VASP is confirmed** (2026-10-23): an 80-atom
  structure pinned to `nodes: 2` in a 2-node allocation ran as `8 mpi-ranks, with 15
  threads/rank, on 2 nodes` (8 GPUs) and reproduced the 1-node energy exactly
  (−389.40935981 eV) in 84 s vs 106 s. To pin a node count, set `nodes:` in
  `DFT/vdW_single_point.yml` (0 = estimate from `atoms_per_node`).
- **How a multi-rank chore actually runs:** Flux starts `num_tasks` copies of the
  chore's *Python* function, one per rank; each `DFT/vasp_dft.py` copy writes the same
  inputs and launches its own `vasp_std`, and Flux's PMI (`mpi=True`) joins those
  processes into one MPI job. So `vasp_dft.py`'s stdout appears once per rank, and
  every rank writes the same `properties.json` — harmless, but expected.
- **Flux offers 64 × N − 1 cores on N nodes** — one stays with the broker (a
  MatEnsemble behaviour also seen in other examples). 4 ranks × 16 cores fill a node,
  so a VASP job spanning *every* node of the allocation is rejected ("requires more
  resources than the allocation can provide"); that's why a 1-node smoke run needs
  `cores_per_task: 15`. The full config uses 16 and assumes the allocation has more
  nodes than its largest VASP job.
- **VASP's library path is passed as `VASP_LD_LIBRARY_PATH`**, applied only to
  `vasp_std`'s subprocess by `DFT/vasp_dft.py` — as `LD_LIBRARY_PATH` it segfaults the
  chore's own Python.
- **MACE/ASE chores prepend `/usr/lib64` to `LD_LIBRARY_PATH`** (`mace_container_env`)
  so torch finds the node's real NVIDIA driver instead of the image's CUDA compat shim.
- **cuequivariance, not openequivariance** (new here): the RMG example's `enable_oeq`
  is ROCm-oriented; this example sets `enable_cueq` for both fitting
  (`build_ff_inputs.mace_overrides`) and single points (`ase_mace.py`).
- **Stress sign** (new here): pymatgen/VASP stress is positive-compressive; ASE/MACE
  is the opposite. `DFT/vasp_dft.py` negates it (the VASP example's copy didn't, and
  flagged it unverified), so new `DFT/uq` results match the shipped training data.
- **Chores run in their own working directory** (new here, MatEnsemble 0.5.8):
  `matensemble_workflow-*/out/chore-*/`, not where you launched. Any path a chore or
  `Pipeline.strategy` resolves itself must be absolute — relative paths sent
  `copy_and_spawn_md`'s model copies into the chore directory. Failed chores also
  don't raise out of `future.result()`; `converge_dft_data` and `downselect` now check
  their own outputs and stop with a clear error instead. Each run's per-chore
  `stdout`/`stderr` are under that `matensemble_workflow-*/out/` directory.
- **mace-torch ≥ 0.3.15 for `freeze`** (new here) — MatEnsemble's requirements install
  0.3.14, which has no `--freeze`; `Dockerfile.mace` pins 0.3.16 (`freeze: 6` trains
  only the readouts) and fails the build if the option is missing. The image's torch
  is 2.11.0+cu128 (from MatEnsemble's requirements), not the cu126 build it requests.
- **`MD_uq_single_points.parent_levels: 4`** matches
  `structures/<label>/<frame>/<md_frame>`; 5 would merge different models into one
  chore. Re-check with a dry `build_task_dicts` call if that tree changes.
