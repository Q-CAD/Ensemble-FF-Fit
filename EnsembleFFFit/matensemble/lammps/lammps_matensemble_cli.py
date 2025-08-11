import argparse
import sys
import os
from pathlib import Path
from EnsembleFFFit.matensemble.base import LammpsMatEnsemble

def main():
    parser = argparse.ArgumentParser(description="Argument parser to run LAMMPs with Flux using Python")

    # Add arguments

    # Run and input directories
    parser.add_argument("--run_directory", "-rd", help="Path to the run directory tree", default='run_directory')
    parser.add_argument("--inputs_directory", "-id", help="Path to input file directory", default='inputs_directory')

    # Input files
    parser.add_argument("--check_files", "-cfs", nargs='+', 
                        help="Names of argparse keys ['ffield', 'in_lammps', 'control', 'structure', 'lammps_task'] to check for in the --run_directory", 
                        default=['ffield'])
    parser.add_argument("--ffield", "-ff", help="Name of force field file", default='ffield')
    parser.add_argument("--in_lammps", "-in", help="Name of the LAMMPs in file", default='in.matensemble')
    parser.add_argument("--control", "-c", help="Name of the LAMMPs control file", default='control') # Build out for contiuation jobs here
    parser.add_argument("--structure", "-s", help="Name of the .lmp file", default='structure.lmp') # Build out for continuation jobs here
    
    parser.add_argument("--lammps_task", "-lt", help="Name of the python script used to interface with LAMMPs", default='lammps_task.py')
    parser.add_argument("--lammps_task_order", "-lto", nargs='+', 
                        help="Order of system arguments to pass to --lammps_task, i.e., sys.argv[1] is 'ffield'", 
                        default=['ffield', 'in_lammps', 'control', 'structure'])
    parser.add_argument("--batch", "-b", help="Flag to batch the runs based on parent directory; useful for single points", action='store_true')

    # Execution options
    parser.add_argument("--atom_style", "-as", help="LAMMPs structure file atom style", type=str, default='charge')
    parser.add_argument("--atoms_per_task", "-apt", help="Atoms per task; passed as list", type=float, default=10)
    parser.add_argument("--cpus_per_task", "-cpt", help="CPUs per task", type=int, default=1)
    parser.add_argument("--gpus_per_task", "-gpt", help="GPUs per task", type=int, default=0)
    parser.add_argument("--dry_run", "-dry", help="Only print the structures to be run", action='store_true')

    args = parser.parse_args()
    run_lammps(args)
    
def run_lammps(args):
    options = {'ffield': args.ffield, 
               'in_lammps': args.in_lammps,
               'control': args.control, 
               'structure': args.structure,
               'lammps_task': args.lammps_task,
               'atom_style': args.atom_style}
    
    # Initialize the LAMMPs object
    lammps_matensemble = LammpsMatEnsemble(args.run_directory, args.inputs_directory, **options)
    
    # Generate the task command path command by checking the inputs directory
    lammps_task_command = os.path.abspath(os.path.join(args.inputs_directory, args.lammps_task))
    if not os.path.isfile(lammps_task_command):
        raise ValueError(f'Invalid task command {lammps_task_command}; file does not exist!')

    # Split the files to be checked in --run_directory vs --input_directory
    inputs_directory_keys = [key for key in options.keys() if key not in args.check_files + ['lammps_task', 'atom_style']]
   
    # Generate the run paths and task arguments
    task_arg_list, run_paths = lammps_matensemble.build_full_runs(root0=args.run_directory, files0=[options[c] for c in args.check_files], 
                                                                  root1=args.inputs_directory, files1=[options[k] for k in inputs_directory_keys],
                                                                  labels=args.check_files + inputs_directory_keys, 
                                                                  ordered_labels=args.lammps_task_order) 
    
    # Generate the tasks per run path based on the number of atoms in each structure
    structure_paths = [task_arg_list[i][args.lammps_task_order.index('structure')] for i in range(len(task_arg_list))]
    tasks = lammps_matensemble.get_tasks(structure_paths, atoms_per_task=args.atoms_per_task) 
  
    # If batching is True (for many short runs, e.g., single-points or energy minimizations), batch the arguments
    if args.batch:
        task_arg_list, run_paths = lammps_matensemble.batch_by_parent(task_arg_list, run_paths, args.check_files + inputs_directory_keys , 'structure')
        structure_paths = [task_arg_list[i][args.lammps_task_order.index('structure')][0] for i in range(len(task_arg_list))]
        tasks = lammps_matensemble.get_tasks(structure_paths, atoms_per_task=args.atoms_per_task)
    
    # Execute the MatEnsemble call
    dry_run = True if args.dry_run else False
    lammps_matensemble.run(dry_run=dry_run,
                           task_command=lammps_matensemble.generic_task_command(lammps_task_command), 
                           run_tasks=tasks,
                           cpus_per_task=args.cpus_per_task, 
                           gpus_per_task=args.gpus_per_task,
                           task_arg_list=task_arg_list, 
                           task_dir_list=run_paths)


if __name__ == '__main__':
    main()
