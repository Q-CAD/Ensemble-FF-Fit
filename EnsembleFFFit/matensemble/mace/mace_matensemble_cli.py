import argparse
from copy import deepcopy
import os
from EnsembleFFFit.matensemble.base import MACEMatEnsemble

def main():
  # create parser
  parser = argparse.ArgumentParser(description='MACE refitting driver')
  
  # MatEnsemble arguments
  parser.add_argument("--run_directory", "-rd", help="Path to the run directory tree", default='run_directory')
  parser.add_argument("--inputs_directory", "-id", help="Path to input file directory", default='inputs_directory')

  parser.add_argument("--check_files", "-cfs", nargs='+',
                        help="Names of argparse keys [foundation_model, config] to check for in the --run_directory",
                        default=['foundation_model'])

  parser.add_argument('--foundation_model', metavar='filename', type=str, default="model.model", help='Initial MACE force field file')
  parser.add_argument('--train_file', metavar='filename', type=str, default="train.xyz", help='Training .xyz file')
  parser.add_argument('--test_file', metavar='filename', type=str, default="test.xyz", help='Testing .xyz file')
  parser.add_argument('--config', metavar='filename', type=str, default="config.yml", help='Configuration yaml')

  parser.add_argument("--cpus_per_task", "-cpt", help="CPUs per task", type=int, default=16)
  parser.add_argument("--gpus_per_task", "-gpt", help="GPUs per task", type=int, default=1)
  parser.add_argument("--fits_per_runpath", "-fpr", help="Number of MACE fits for each runpath", type=int, default=1)
  parser.add_argument("--dry_run", "-dry", help="Only print the structures to be run", action='store_true') 

  args = parser.parse_args()
  run_mace(args)

def run_mace(args):
  # Generate the options dictionary for MatEnsembleJob object initilization
  matensemble_arguments = ['run_directory', 'inputs_directory', 'check_files', 
                           'cpus_per_task', 'gpus_per_task', 'fits_per_runpath', 'dry_run']

  options = {'foundation_model': args.foundation_model,
             'config': args.config, 
             'train_file': args.train_file, 
             'test_file': args.test_file}

  # Initialize the JaxReaxFF object
  mace_matensemble = MACEMatEnsemble(args.run_directory, args.inputs_directory, **options)

  # Split the files to be checked in --run_directory vs --input_directory
  inputs_directory_keys = [key for key in options.keys() if key not in args.check_files]

  # Generate the task arguments, run paths and task numbers
  labels = args.check_files + inputs_directory_keys
  task_arg_list, run_paths = mace_matensemble.build_full_runs(root0=args.run_directory, files0=[options[c] for c in args.check_files],
                                                                  root1=args.inputs_directory, files1=[options[k] for k in inputs_directory_keys],
                                                                  labels=labels, ordered_labels=labels) # No ordering needed
  
  # Generate the task_arg_list and run_paths with separate run directories for different starting seeds
  task_arg_list, run_paths = mace_matensemble.construct_tasks(task_arg_list, run_paths, args.fits_per_runpath)
  
  # Create the arguments dictionary and yield the list of list of argparse argument strings
  task_arg_strs = mace_matensemble.to_str_list(labels=labels, 
                                               task_arg_list=task_arg_list, 
                                               run_paths=run_paths) 
  # Generate tasks per run based on user arguments
  tasks = mace_matensemble.get_tasks(run_paths)

  # Execute the MatEnsemble call
  dry_run = True if args.dry_run else False
  mace_matensemble.run(dry_run=dry_run,
                           task_command='mace_run_train',
                           run_tasks=tasks,
                           cpus_per_task=args.cpus_per_task,
                           gpus_per_task=args.gpus_per_task,
                           task_arg_list=task_arg_strs,
                           task_dir_list=run_paths)

if __name__ == '__main__':
  main()
