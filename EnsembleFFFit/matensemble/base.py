from abc import ABC, abstractmethod
from pymatgen.io.lammps.data import LammpsData
import numpy as np
import os
import warnings
import sys

class MatEnsembleJob(ABC):
    def __init__(self, run_directory, inputs_directory, **kwargs):
        self.run_directory = run_directory
        self.inputs_directory = inputs_directory
        self.options = kwargs

    @abstractmethod
    def sorting_function(self, paths): pass

    @abstractmethod
    def get_tasks(self, paths): pass

    def get_run_paths(self, check_file, append_check_file=True):
        '''Check for check_file in roots of self.run_directory; if so, append path'''
        run_paths = []
        for root, _, _ in os.walk(self.run_directory):
            check_file_path = os.path.join(root, check_file)
            if os.path.exists(check_file_path):
                if append_check_file == True:
                    run_paths.append(os.path.abspath(check_file_path))
                else:
                    run_paths.append(os.path.abspath(root))
        return run_paths

    def get_task_args_list(self, run_path_roots, check_file, options_keys, include_root=True, abs_path=True):
        ''' Determines a task arg list based on passed options keys and run directories'''
        task_arg_list = [[] for i in range(len(run_path_roots))]
        
        for i, root in enumerate(run_path_roots):
            for option_key in options_keys:
                root_path = os.path.join(root, self.options[option_key])
                input_path = os.path.join(self.inputs_directory, self.options[option_key])
                if os.path.exists(root_path):
                    if include_root:
                        task_arg_list[i].append(root_path)
                    else:
                        task_arg_list[i].append(self.options[option_key])
                elif os.path.exists(input_path):
                    if abs_path:
                        task_arg_list[i].append(os.path.abspath(input_path))
                    else:
                        task_arg_list[i].append(input_path)
                else:
                    print(f'Required {option_key} file {self.options[option_key]} does not exist in {self.run_directory} or {self.input_directory}; exiting')
                    sys.exit(1)
        return task_arg_list 

    def get_python(self):
        try:
            python_exe = os.path.join(os.environ['CONDA_PREFIX'], 'bin', 'python')
        except KeyError:
            python_exe = 'python'
        return python_exe

    def dry_run(self, paths, tasks, cpus_per_task, gpus_per_task):
        for i, path in enumerate(paths):
            print(f'path: {paths[i]}, tasks: {tasks[i]}\n')
        print(f'Total tasks = {np.sum(tasks)}; cpus_per_task={cpus_per_task}; gpus_per_task={gpus_per_task}')

    def run(self, task_list, task_command, run_tasks, 
                  cpus_per_task, gpus_per_task, 
                  task_arg_list, task_dir_list, 
                  write_restart_freq=1000000, buffer_time=1):
        
        from matensemble.matfluxGen import SuperFluxManager
        master = SuperFluxManager(gen_task_list=task_list,
                                  gen_task_cmd=task_command,
                                  ml_task_cmd=None,
                                  tasks_per_job=run_tasks,
                                  cores_per_task=cpus_per_task,
                                  gpus_per_task=gpus_per_task,
                                  write_restart_freq=1000000)
        master.poolexecutor(task_arg_list=task_arg_list,
                            buffer_time=1,
                            task_dir_list=task_dir_list)
        return 


class LammpsMatEnsemble(MatEnsembleJob):
    def __init__(self, run_directory, inputs_directory, **kwargs):
        super().__init__(run_directory, inputs_directory, **kwargs)
        ''' LAMMPS self.options keys for generic_task_command are "ffield", "in", "control" and "structure" '''

    def read_structure_from_lammps(self, lmp_file_path):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return LammpsData.from_file(lmp_file_path, atom_style=self.options['atom_style']).structure

    def sorting_function(self, path):
        ''' Sort by structure length '''
        structure = self.read_structure_from_lammps(path)
        return len(structure)

    def get_tasks(self, structure_paths, atoms_per_task=10):
        ''' Set based on structure length '''
        structures = [self.read_structure_from_lammps(path) for path in structure_paths]
        return [max(np.floor(len(s)/atoms_per_task).astype(int), 1) for i, s in enumerate(structures)]

    def generic_task_command(self, python_file):
        ''' Builds a generic task command for the LAMMPs python interface '''
        python_exe = self.get_python()
        return ''.join([python_exe] + [' '] + [python_file]) 

class JaxReaxFFMatEnsemble(MatEnsembleJob):
    def __init__(self, run_directory, inputs_directory, **kwargs):
        super().__init__(run_directory, inputs_directory, **kwargs)

    def sorting_function(self, path):
        return str.lower

    def get_tasks(self, paths, tasks_per_path):
        return [tasks_per_path for path in paths]

    def dict_to_argv(self, d):
        """
        Turn a dict of {option_name: value} into a flat list of CLI args:
          {"foo": "bar", "baz": 1, "flag": True}
        → ["--foo", "bar", "--baz", "1", "--flag"]
        Boolean True→ include the flag, False→ omit it.
        """
        argv = []
        for k, v in d.items():
            flag = f" --{k}"
            if isinstance(v, bool):
                if v:
                    argv.append(flag)
            else:
                argv.extend([flag + ' ', str(v)])
        return argv

    def generic_task_command(self, args_dct):
        jaxreaxff_exe = os.path.join(os.environ['CONDA_PREFIX'], 'bin', 'jaxreaxff')
        return ''.join([jaxreaxff_exe] + self.dict_to_argv(args_dct))
