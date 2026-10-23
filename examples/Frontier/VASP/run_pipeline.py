"""
GPU VASP on Frontier through MatEnsemble/Flux inside the MatEnsemble Apptainer
image: the `converge_dft_data` stage of examples/Perlmutter/VASP_MACE_ASE,
ported to Frontier (host vasp/6.6.1-gpu module, one MPI rank per GCD), run on
a small Pd-Se test set (two isolated atoms, one 78-atom amorphous cell). It is
the DFT half of a VASP -> MACE pipeline on Frontier; the MACE/ASE stages are
the same as examples/Frontier/RMG_MACE_ASE's and are not repeated here.

    python run_pipeline.py --config workflow_config.yaml --stage converge_dft_data

(normally via launch_flux.slurm, see README.md). Every already-converged
directory (properties.json present) is skipped, so rerunning resumes.

What differs from the Perlmutter version, and why (README.md has the
measurements):
- VASP_LD_LIBRARY_PATH holds the HOST libraries vasp_std was built against
  (CCE 20.0.2 runtime, ROCm 6.4.2, Cray PE, amdgpu, xpmem, darshan),
  bind-mounted by launch_flux.slurm -- applied to vasp_std's subprocess only.
- 8 ranks per node, 7 cores each (one L3 group), each rank on its closest
  GCD (DFT/vasp_dft.py, VASP_GPU_CLOSEST_GCDS); Flux's own GPU affinity is
  off, since it would hand GPUs out in index order.
- MPICH_OFI_NIC_POLICY=BLOCK (NUMA aborts MPI_Init on >1 node under Flux).
"""
import argparse
import glob
import os
import sys
from pathlib import Path

import yaml

# vasp/6.6.1-gpu's runtime libraries (from `ldd` after `module load
# vasp/6.6.1-gpu`), as seen inside the container with launch_flux.slurm's
# binds. FIRST: the image's Flux libpmi (/usr/local/lib/flux). Flux's shell
# normally prepends it to each task's LD_LIBRARY_PATH, which is how the
# ranks' Cray MPICH joins one job -- but vasp_dft.py *replaces*
# LD_LIBRARY_PATH with this value for vasp_std, dropping that prefix, and
# without it every rank comes up as its own 1-rank job (OUTCAR "running 1
# mpi-ranks", ranks colliding on vaspout.h5's lock). Same fix as the
# Perlmutter example. Then: host Cray PE / CCE runtime / ROCm 6.4.2 / libdrm (host /opt/amdgpu,
# mounted at /opt/host_amdgpu so the image's own copy stays visible to it) /
# xpmem (likewise /opt/host_xpmem) / darshan first; the container's own
# multiarch dirs next (its glibc/libstdc++ are newer than the host's); the
# host /usr/lib64 bind last, for what the container lacks (libcxi, ...).
# Passed as VASP_LD_LIBRARY_PATH, not LD_LIBRARY_PATH: DFT/vasp_dft.py
# applies it to vasp_std's own subprocess only, never to Python. Re-verify
# every path when the VASP module changes.
VASP_CCE = "/opt/cray/pe/cce/20.0.2"
VASP_DARSHAN = ("/sw/borg/spack-envs/cpe25.09-cpu/opt/cce-20.0.2/"
                "darshan-runtime-3.4.7-god5wjz7nfkvzvsjcrrzpa27r2qmvfh6/lib")
VASP_LD_LIBRARY_PATH = ":".join([
    "/usr/local/lib/flux",
    "/opt/cray/pe/lib64", "/opt/cray/pe/lib64/cce",
    f"{VASP_CCE}/cce/x86_64/lib", f"{VASP_CCE}/cce-clang/x86_64/lib",
    "/opt/cray/libfabric/2.3.1/lib64", "/opt/cray/pals/1.8/lib",
    "/opt/rocm-6.4.2/lib", "/opt/host_amdgpu/lib64", "/opt/host_xpmem/lib64", VASP_DARSHAN,
    "/usr/lib/x86_64-linux-gnu", "/lib/x86_64-linux-gnu",
    "/opt/cray_extra/lib64",
])

# GCD closest to each Frontier L3 group (cores 8k+1..8k+7, k = 0..7) -- what
# native `srun --gpu-bind=closest` picks.
FRONTIER_CLOSEST_GCDS = "4,5,2,3,6,7,0,1"


def chore_pythonpath():
    """This EnsembleFFFit checkout first, then the PYTHONPATH the pipeline was
    launched with -- a chore's Resources.env value replaces the inherited one."""
    from EnsembleFFFit.utilities.general import ensemble_fffit_pythonpath

    return os.pathsep.join(p for p in (ensemble_fffit_pythonpath(), os.environ.get('PYTHONPATH')) if p)


def vasp_container_env(omp_num_threads, gpu_aware_mpi):
    """Env for the vasp_std chore -- see VASP_LD_LIBRARY_PATH above.

    HSA_XNACK=0 matches a native run (the image sets 1, for LAMMPS).
    gpu_aware_mpi sets MPICH_GPU_SUPPORT_ENABLED, which VASP's GPU build uses
    to pass device buffers to MPI directly. VASP_POTENTIALS_DIR mirrors the
    module's own env (`module show vasp/6.6.1-gpu`) -- distinct from the
    recipe's pseudopotentials_directory, which pymatgen uses for POTCARs.
    """
    return {
        'PYTHONPATH': chore_pythonpath(),
        'OMP_NUM_THREADS': str(omp_num_threads),
        'VASP_LD_LIBRARY_PATH': VASP_LD_LIBRARY_PATH,
        'VASP_GPU_CLOSEST_GCDS': FRONTIER_CLOSEST_GCDS,
        'VASP_POTENTIALS_DIR': '/sw/frontier/vasp/potentials',
        'MPICH_GPU_SUPPORT_ENABLED': '1' if gpu_aware_mpi else '0',
        'MPICH_OFI_NIC_POLICY': 'BLOCK',
        'HSA_XNACK': '0',
    }


def _require(cfg, stage_name, keys):
    if cfg is None:
        raise ValueError(f"Missing required section '{stage_name}' in the config file.")
    missing = [k for k in keys if cfg.get(k) is None]
    if missing:
        raise ValueError(f"Missing required key(s) {missing} under '{stage_name}' in the config file.")


# ---------------------------------------------------------------------------
# Stage: converge_dft_data -- VASP
# ---------------------------------------------------------------------------

def vasp_resource_estimator(structure, input_args, options):
    """
    `resource_estimator` for DFTMatEnsemble.build_dft_dcts (same as
    examples/Perlmutter/VASP_MACE_ASE): the recipe's `nodes` (default 0)
    pins the node count when positive, otherwise ceil(n_atoms /
    atoms_per_node).
    """
    nodes = input_args.get('nodes', 0)
    if nodes and nodes > 0:
        return nodes
    atoms_per_node = input_args.get('atoms_per_node', options.get('atoms_per_node', 100))
    return max(1, -(-len(structure) // atoms_per_node))  # ceiling division


def run_converge_dft_data(config):
    """
    Submit one VASP chore per (structure, vasp_yaml) pair under `directory`,
    each sized from its own node estimate: num_tasks = nodes *
    ranks_per_node (8 -- one rank per GCD), cores_per_task (7) cores and one
    GPU per rank, OMP_NUM_THREADS = omp_num_threads (1, VASP's GPU build
    keeps the CPU threads idle). Flux starts num_tasks copies of the chore
    function, each launching its own vasp_std; Flux's PMI joins them into
    one MPI job (`running N mpi-ranks` in OUTCAR).

    `directory` serves as both run_directory and inputs_directory for
    DFTMatEnsemble; exactly one recipe file (vasp_yaml_name) is expected in
    the tree.
    """
    from matensemble.pipeline import Pipeline
    from matensemble.model import Resources
    from EnsembleFFFit.base import DFTMatEnsemble

    dft_cfg = config['converge_dft_data']
    _require(dft_cfg, 'converge_dft_data', ['directory', 'dft_task'])

    directory = dft_cfg['directory']
    dft_task = str(Path(dft_cfg['dft_task']).resolve())
    structure_filename = dft_cfg.get('structure_filename', 'POSCAR')
    vasp_yaml_name = dft_cfg.get('vasp_yaml_name', 'vdW_single_point.yml')
    entry_point = dft_cfg.get('entry_point', 'run_vasp_calculation')
    check_files = dft_cfg.get('check_files', ['dft_recipe'])
    finished_file = dft_cfg.get('finished_file')
    cores_per_task = dft_cfg.get('cores_per_task', 7)
    omp_num_threads = dft_cfg.get('omp_num_threads', 1)
    gpus_per_task = dft_cfg.get('gpus_per_task', 1)
    atoms_per_node = dft_cfg.get('atoms_per_node', 100)
    ranks_per_node = dft_cfg.get('ranks_per_node', 8)
    max_nodes_per_job = dft_cfg.get('max_nodes_per_job', 4)

    options = {
        'structure_filename': structure_filename,
        'dft_recipe': vasp_yaml_name,
        'atoms_per_node': atoms_per_node,
    }

    dft = DFTMatEnsemble(directory, directory, **options)
    dft_dct_list = dft.build_dft_dcts(dft_task, check_files, entry_point, finished_file=finished_file,
                                       resource_estimator=vasp_resource_estimator)

    if not dft_dct_list:
        recipe_keys, structure_keys = ['dft_recipe'], ['structure_filename']
        labels = check_files + structure_keys + recipe_keys
        _, raw_task_dirs = dft.build_full_runs(
            root0=dft.run_directory, files0=[dft.options[c] for c in check_files],
            root1=dft.inputs_directory, files1=[dft.options[k] for k in structure_keys],
            recipe_files=[dft.options[k] for k in recipe_keys],
            labels=labels, ordered_labels=labels, finished_file=None,
            run_directory=dft.run_directory, inputs_directory=dft.inputs_directory,
        )
        if raw_task_dirs and finished_file is not None:
            print(f"converge_dft_data: all {len(raw_task_dirs)} (structure, vasp_yaml) pair(s) under "
                  f"{directory} already have a '{finished_file}' match -- nothing new to run, skipping.")
            return
        raise ValueError(f"No (structure, vasp_yaml) pairs found under {directory}")

    # controller_cores=0: MatEnsemble otherwise keeps one core back on a
    # 1-node instance, so 8 ranks x 7 cores wouldn't fit there.
    pipe = Pipeline(reserve_broker_node=False, controller_cores=dft_cfg.get('controller_cores', 0))
    env = vasp_container_env(omp_num_threads, dft_cfg.get('gpu_aware_mpi', True))

    # mpi=False: Flux's shell wires every task to its PMI server either way
    # and puts its own libpmi first on the task's library path, so the ranks
    # join one MPI job (same as RMG on Frontier; Perlmutter's VASP needed
    # mpi=True for its PMI setup).
    @pipe.chore(name="run_vasp", num_tasks=1, cores_per_task=cores_per_task, gpus_per_task=gpus_per_task,
                env=env, inherit_env=True, mpi=False)
    def run_vasp_chore(task_dict):
        """Thin chore wrapper -- the real VASP-execution logic lives in DFTMatEnsemble.run_individual."""
        return DFTMatEnsemble.run_individual(task_dict)

    for dft_dct in dft_dct_list:
        task_dict = dict(dft_dct)
        nodes = dft_dct['allocated_nodes']
        if max_nodes_per_job is not None and nodes > max_nodes_per_job:
            print(f"WARNING: {task_dict['structure_filename']}: estimated {nodes} node(s) exceeds "
                  f"max_nodes_per_job={max_nodes_per_job}; running on {max_nodes_per_job}.", file=sys.stderr)
            nodes = max_nodes_per_job
            task_dict['allocated_nodes'] = nodes
        num_tasks = nodes * ranks_per_node
        print(f"converge_dft_data: {task_dict['working_directory']}: {nodes} node(s), {num_tasks} MPI rank(s) "
              f"x {cores_per_task} core(s) x {gpus_per_task} GPU(s)")
        resources = Resources(num_tasks=num_tasks, cores_per_task=cores_per_task, gpus_per_task=gpus_per_task,
                               env=env, inherit_env=True, mpi=False)
        pipe.call("run_vasp", task_dict, resources=resources)

    # set_cpu_affinity=True: per-task cores (one L3 group each with
    # cores_per_task 7). set_gpu_affinity=False: vasp_dft.py picks each
    # rank's GCD by topology instead of Flux's index order.
    future = pipe.submit(log_delay=25, set_gpu_affinity=False, set_cpu_affinity=True)
    future.result()

    # A failed chore doesn't raise out of future.result() -- check
    # finished_file directly: warn per failure, fail if nothing converged.
    if finished_file is not None:
        missing = [d['working_directory'] for d in dft_dct_list
                   if not glob.glob(os.path.join(d['working_directory'], finished_file))]
        for working_directory in missing:
            print(f"WARNING: converge_dft_data: no '{finished_file}' in {working_directory} -- that VASP "
                  f"run failed; see the matensemble_workflow-*/ logs.", file=sys.stderr)
        if len(missing) == len(dft_dct_list):
            raise RuntimeError(f"converge_dft_data: all {len(missing)} VASP run(s) failed (no '{finished_file}' "
                               f"written) -- see the matensemble_workflow-*/ logs.")


STAGES = {
    'converge_dft_data': run_converge_dft_data,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", "-c", required=True, help="Path to the YAML workflow config file")
    parser.add_argument("--stage", "-s", required=True, choices=list(STAGES.keys()) + ['all'])
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    for name, stage_fn in (STAGES.items() if args.stage == 'all' else [(args.stage, STAGES[args.stage])]):
        print(f"\n{'=' * 20} Stage: {name} {'=' * 20}")
        stage_fn(config)


if __name__ == "__main__":
    main()
