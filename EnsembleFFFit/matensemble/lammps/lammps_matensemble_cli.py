import argparse
import sys
import os
from pathlib import Path
from EnsembleFFFit.matensemble.base import LammpsMatEnsemble

def main():
    parser = argparse.ArgumentParser(description="Argument parser to run LAMMPs with Flux using Python")

    # Parse NoneType for dictionary 
    def none_or_str(value):
        if value == 'None':
            return None
        return value

    # Add arguments

    # Run and input directories
    parser.add_argument("--run_directory", "-rd", help="Path to the run directory tree", default='run_directory')
    parser.add_argument("--inputs_directory", "-id", help="Path to input file directory", default='inputs_directory')

    # Input files
    parser.add_argument("--check_files", "-cfs", nargs='+', 
                        help="Names of argparse keys ['ffield', 'in_lammps', 'control', 'structure', 'lammps_task'] to check for in the --run_directory", 
                        default=['ffield'])
    parser.add_argument("--ffield", "-ff", type=none_or_str, help="Name of force field file", default='ffield')
    parser.add_argument("--in_lammps", "-in", type=none_or_str, help="Name of the LAMMPs in file", default='in.matensemble')
    parser.add_argument("--control", "-c", type=none_or_str, help="Name of the LAMMPs control file", default='control') # Build out for contiuation jobs here
    parser.add_argument("--structure", "-s", type=none_or_str, help="Name of the .lmp file", default='structure.lmp') # Build out for continuation jobs here
    
    parser.add_argument("--lammps_task", "-lt", type=none_or_str, help="Name of the python script used to interface with LAMMPs", default='lammps_task.py')
    parser.add_argument("--lammps_task_order", "-lto", nargs='+', 
                        help="Order of system arguments to pass to --lammps_task, i.e., sys.argv[1] is 'ffield'", 
                        default=['ffield', 'in_lammps', 'control', 'structure'])
    parser.add_argument("--parent_levels", "-pl", type=int, help="Flag to set the parent directory levels for batching; batch by -pl parent directories above the run", default=0)

    # Execution options
    parser.add_argument("--atom_style", "-as", help="LAMMPs structure file atom style", type=str, default='charge')
    parser.add_argument("--atoms_per_task", "-apt", help="Atoms per task; passed as list", type=float, default=10)
    parser.add_argument("--cpus_per_task", "-cpt", help="CPUs per task", type=int, default=1)
    parser.add_argument("--gpus_per_task", "-gpt", help="GPUs per task", type=int, default=0)
    parser.add_argument("--dry_run", "-dry", help="Only print the structures to be run", action='store_true')

    args = parser.parse_args()
    run_lammps(args)
    
def run_lammps(args):
    # Construct an options dictionary and only keep keys without null values
    options = {'ffield': args.ffield, 
               'in_lammps': args.in_lammps,
               'control': args.control, 
               'structure': args.structure,
               'lammps_task': args.lammps_task,
               'atom_style': args.atom_style}
    options = {k: v for k, v in options.items() if v is not None}

    # Initialize the LAMMPs object
    lammps_matensemble = LammpsMatEnsemble(args.run_directory, args.inputs_directory, **options)
    
    # Generate the task command path command by checking the inputs directory
    lammps_task_command = os.path.abspath(os.path.join(args.inputs_directory, args.lammps_task))
    if not os.path.isfile(lammps_task_command):
        raise ValueError(f'Invalid task command {lammps_task_command}; file does not exist in {args.inputs_directory}!')

    # Split the files to be checked in --run_directory vs --input_directory
    inputs_directory_keys = [key for key in options.keys() if key not in args.check_files + ['lammps_task', 'atom_style']]
    
    # Generate combinations of run paths and task arguments
    task_arg_list, run_paths = lammps_matensemble.build_full_runs(root0=args.run_directory, files0=[options[c] for c in args.check_files], 
                                                                  root1=args.inputs_directory, files1=[options[k] for k in inputs_directory_keys],
                                                                  labels=args.check_files + inputs_directory_keys, 
                                                                  ordered_labels=args.lammps_task_order) 

    # Generate the tasks per run path based on the number of atoms in each structure
    structure_paths = [task_arg_list[i][args.lammps_task_order.index('structure')] for i in range(len(task_arg_list))]
    tasks = lammps_matensemble.get_tasks(structure_paths, atoms_per_task=args.atoms_per_task) 
  
    # Batch the runs based on the parent level
    task_arg_list, run_paths, make_paths = lammps_matensemble.batch_by_parent(task_arg_list, run_paths, args.check_files + inputs_directory_keys, args.parent_levels)
    structure_paths = [task_arg_list[i][args.lammps_task_order.index('structure')][0] for i in range(len(task_arg_list))]
    tasks = lammps_matensemble.get_tasks(structure_paths, atoms_per_task=args.atoms_per_task)

    # Execute the MatEnsemble call
    lammps_matensemble.run(dry_run=True if args.dry_run else False,
                           task_command=lammps_matensemble.generic_task_command(lammps_task_command), 
                           run_tasks=tasks,
                           cpus_per_task=args.cpus_per_task, 
                           gpus_per_task=args.gpus_per_task,
                           task_arg_list=task_arg_list, 
                           task_dir_list=run_paths, 
                           make_paths_list=make_paths)


if __name__ == '__main__':
    main()
