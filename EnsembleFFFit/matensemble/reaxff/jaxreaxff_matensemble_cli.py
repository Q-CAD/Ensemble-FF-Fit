import argparse
from copy import deepcopy
import os
from frozendict import frozendict
from EnsembleFFFit.matensemble.base import JaxReaxFFMatEnsemble

class SmartFormatter(argparse.ArgumentDefaultsHelpFormatter):
  def _split_lines(self, text, width):
    if text.startswith('R|'):
      return text[2:].splitlines()  
    return argparse.ArgumentDefaultsHelpFormatter._split_lines(self, text, width)

def build_float_range_checker(min_v, max_v):
  '''
  Returns a function that can be used to validate fiven FP value
  withing the allowed range ([min_v, max_v])
  '''
  def range_checker(arg):
    try:
      val = float(arg)
    except ValueError:    
      raise argparse.ArgumentTypeError("Value must be a floating point number")
    if val < min_v or val > max_v:
      raise argparse.ArgumentTypeError("Value must be in range [" + str(min_v) + ", " + str(max_v)+"]")
    return val
  return range_checker

def main():
  # create parser
  parser = argparse.ArgumentParser(description='JAX-ReaxFF driver',
                                   formatter_class=SmartFormatter)
  
  # MatEnsemble arguments
  parser.add_argument("--run_directory", "-rd", help="Path to the run directory tree", default='run_directory')
  parser.add_argument("--inputs_directory", "-id", help="Path to input file directory", default='inputs_directory')

  parser.add_argument("--check_files", "-cfs", nargs='+',
                        help="Names of argparse keys ['init_FF', 'params', 'geo', 'train_file', 'valid_file', 'valid_geo_file'] to check for in the --run_directory",
                        default=['init_FF'])

  parser.add_argument("--cpus_per_task", "-cpt", help="CPUs per task", type=int, default=16)
  parser.add_argument("--gpus_per_task", "-gpt", help="GPUs per task", type=int, default=1)
  parser.add_argument("--fits_per_runpath", "-fpr", help="Number of JaxReaxFF fits for each runpath", type=int, default=4)
  parser.add_argument("--dry_run", "-dry", help="Only print the structures to be run", action='store_true') 

  # From the Jax-ReaxFF package jaxreaxff executable. Default inputs: inital force field, parameters, geo and trainset files
  parser.add_argument('--init_FF', metavar='filename',
      type=str,
      default="ffield",
      help='Initial force field file')
  parser.add_argument('--params', metavar='filename',
      type=str,
      default="params",
      help='Parameters file')
  parser.add_argument('--geo', metavar='filename',
      type=str,
      default="geo",
      help='Geometry file')
  parser.add_argument('--train_file', metavar='filename',
      type=str,
      default="trainset.in",
      help='Training set file')
  parser.add_argument('--use_valid', metavar='boolean',
      type=bool,
      default=False,
      help='Flag indicating whether to use validation data (True/False)')
  parser.add_argument('--valid_file', metavar='filename',
      type=str,
      default="validset.in",
      help='Validation set file (same format as trainset.in)')
  parser.add_argument('--valid_geo_file', metavar='filename',
      type=str,
      default="valid_geo",
      help='Geo file for the validation data')
  # optimization related parameters
  parser.add_argument('--opt_method', metavar='method',
      choices=['L-BFGS-B', 'SLSQP'],
      type=str,
      default='L-BFGS-B',
      help='Optimization method - "L-BFGS-B" or "SLSQP"')
  parser.add_argument('--num_trials', metavar='number',
      type=int,
      default=1,
      help='R|Number of trials (Population size).\n' +
      'If set to <= 0, provided force field will be evaluated w/o any training (init_FF).')
  parser.add_argument('--num_steps', metavar='number',
      type=int,
      default=5,
      help='Number of optimization steps per trial')
  parser.add_argument('--init_FF_type', metavar='init_type',
      choices=['random', 'educated', 'fixed'],
      default='fixed',
      help='R|How to start the trials from the given initial force field.\n' +
      '"random": Sample the parameters from uniform distribution between given ranges.\n'
      '"educated": Sample the parameters from a narrow uniform distribution centered at given values.\n'
      '"fixed": Start from the parameters given in "init_FF" file')
  parser.add_argument('--random_sample_count', metavar='number',
      type=int,
      default=0,
      help='R|Before the optimization starts, uniforms sample the paramater space.\n' +
      'Select the best sample to start the training with, only works with "random" inital start.\n' +
      'if set to 0, no random search step will be skipped. ')
  # energy minimization related parameters
  parser.add_argument('--num_e_minim_steps', metavar='number',
      type=int,
      default=0,
      help='Number of energy minimization steps')
  parser.add_argument('--e_minim_LR', metavar='init_LR',
      type=float,
      default=5e-4,
      help='Initial learning rate for energy minimization')
  parser.add_argument('--end_RMSG', metavar='end_RMSG',
      type=float,
      default=1.0,
      help='Stopping condition for E. minimization')
  # output related options
  parser.add_argument('--out_folder', metavar='folder',
      type=str,
      default="outputs",
      help='Folder to store the output files')
  parser.add_argument('--save_opt', metavar='option',
      choices=['all', 'best'],
      default="best",
      help='R|"all" or "best"\n' +
      '"all": save all of the trained force fields\n' +
      '"best": save only the best force field')
  parser.add_argument('--cutoff2', metavar='cutoff',
      type=float,
      default=0.001,
      help='BO-cutoff for valency angles and torsion angles')
  parser.add_argument('--max_num_clusters', metavar='max # clusters',
      type=int,
      default=10,
      choices=range(1, 16),
      help='R|Max number of clusters that can be used\n' +
           'High number of clusters lowers the memory cost\n' +
           'However, it increases compilation time,especially for cpus')
  parser.add_argument('--perc_noise_when_stuck', metavar='percentage',
      type=build_float_range_checker(0.0, 0.1),
      default=0.04,
      help='R|Percentage of the noise that will be added to the parameters\n' +
           'when the optimizer is stuck.\n' +
           'param_noise_i = (param_min_i, param_max_i) * perc_noise_when_stuck\n' +
           'Allowed range: [0.0, 0.1]')
  parser.add_argument('--seed', metavar='seed',
      type=int,
      default=0,
      help='Seed value')

  args = parser.parse_args()
  run_reaxff(args)

def run_reaxff(args):
  # Generate the options dictionary for MatEnsembleJob object initilization
  matensemble_arguments = ['run_directory', 'inputs_directory', 'check_files', 
                           'cpus_per_task', 'gpus_per_task', 'fits_per_runpath', 'dry_run']

  options = {'init_FF': args.init_FF,
             'params': args.params,
             'geo': args.geo,
             'train_file': args.train_file}

  if args.use_valid:
    options['valid_file'] = args.valid_file 
    options['valid_geo_file'] = args.valid_geo_file

  # Initialize the JaxReaxFF object
  jaxreaxff_matensemble = JaxReaxFFMatEnsemble(args.run_directory, args.inputs_directory, **options)

  # Split the files to be checked in --run_directory vs --input_directory
  inputs_directory_keys = [key for key in options.keys() if key not in args.check_files]

  # Generate the task arguments, run paths and task numbers
  labels = args.check_files + inputs_directory_keys
  task_arg_list, run_paths = jaxreaxff_matensemble.build_full_runs(root0=args.run_directory, files0=[options[c] for c in args.check_files],
                                                                  root1=args.inputs_directory, files1=[options[k] for k in inputs_directory_keys],
                                                                  labels=labels, ordered_labels=labels) # No ordering needed

  # Create the arguments dictionary and yield the list of list of argparse argument strings
  task_arg_strs = jaxreaxff_matensemble.dict_to_str_list(d=args, labels=labels, 
                                                         task_arg_list=task_arg_list, 
                                                         ignore_list=matensemble_arguments)
  
  # Generate tasks per run based on user arguments
  tasks = jaxreaxff_matensemble.get_tasks(run_paths, args.fits_per_runpath)

  # Execute the MatEnsemble call
  dry_run = True if args.dry_run else False
  jaxreaxff_matensemble.run(dry_run=dry_run,
                           task_command='jaxreaxff',
                           run_tasks=tasks,
                           cpus_per_task=args.cpus_per_task,
                           gpus_per_task=args.gpus_per_task,
                           task_arg_list=task_arg_strs,
                           task_dir_list=run_paths)


if __name__ == '__main__':
  main()
