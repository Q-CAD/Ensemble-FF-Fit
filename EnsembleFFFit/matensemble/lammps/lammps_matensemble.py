import argparse
import sys
from EnsembleFFFit.matensemble.base import LammpsMatEnsemble 

def main():
    parser = argparse.ArgumentParser(description="Argument parser to run LAMMPs with Flux using template files")

    # Add arguments

    # Run and input directories
    parser.add_argument("--run_directory", "-rd", help="Path to the run directory tree", default='run_directory')
    parser.add_argument("--inputs_directory", "-id", help="Path to input file directory", default='inputs_directory')

    # Input files
    parser.add_argument("--check_file", "-cf", help="Name of options key to check for in --run_directory", default='ffield')
    parser.add_argument("--ffield", "-ff", help="Name of force field file", default='ffield')
    parser.add_argument("--in_lammps", "-in", help="Name of the LAMMPs in file", default='in.matensemble')
    parser.add_argument("--control", "-c", help="Name of the LAMMPs control file", default='control') # Build out for contiuation jobs here
    parser.add_argument("--structure", "-s", help="Name of the .lmp file", default='structure.lmp') # Build out for continuation jobs here
    parser.add_argument("--lammps_task", "-lt", help="Name of the python script used to interface with LAMMPs", default='lammps_task.py')

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
               'in': args.in_lammps,
               'control': args.control, 
               'structure': args.structure,
               'lammps_task': args.lammps_task,
               'atom_style': args.atom_style}
               
    lammps_matensemble = LammpsMatEnsemble(args.run_directory, args.inputs_directory, **options)
    run_paths = lammps_matensemble.get_run_paths(args.check_file, append_check_file=False) 
    options_order = ["ffield", "in", "control", "structure"] # Must correspond with sys.argv order for Python submission script
    task_arg_list = lammps_matensemble.get_task_args_list(run_paths, args.check_file, options_order)
    structure_paths = [task_arg_list[i][options_order.index('structure')] for i in range(len(task_arg_list))]
    tasks = lammps_matensemble.get_tasks(structure_paths, atoms_per_task=args.atoms_per_task)    

    lammps_list = lammps_matensemble.get_task_args_list(run_paths, args.check_file, ['lammps_task'], include_root=False)

    if all(lmp == lammps_list[0] for lmp in lammps_list):
        task_command = lammps_matensemble.generic_task_command(lammps_list[0][0])
    else:
        print('Not all {args.lammps_task} are the same; exiting')
        sys.exit(1)

    if args.dry_run:
        lammps_matensemble.dry_run(run_paths, tasks, args.cpus_per_task, args.gpus_per_task)
        #print(task_command)
        #print(task_arg_list)
    else:
        lammps_matensemble.run(task_list=[i for i in range(len(run_paths))], 
                               task_command=task_command, 
                               run_tasks=tasks,
                               cpus_per_task=args.cpus_per_task, 
                               gpus_per_task=args.gpus_per_task,
                               task_arg_list=task_arg_list, 
                               task_dir_list=run_paths)

if __name__ == '__main__':
    main()
