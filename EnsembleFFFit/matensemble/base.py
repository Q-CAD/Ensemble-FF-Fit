from abc import ABC, abstractmethod
from pymatgen.io.lammps.data import LammpsData
import numpy as np
from itertools import product
from copy import deepcopy
from pathlib import Path
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

    def _collect_paths(self, root: str, names: list[str]) -> dict[str, list[str]]:
        d = {n: [] for n in names}
        for dp, _, files in os.walk(root):
            for f in files:
                if f in d:
                    d[f].append(os.path.abspath(os.path.join(dp, f)))
        return d

    def _common_prefix(self, parts1: list[str], parts2: list[str]) -> int:
        """How many leading path‐components do two split paths share?"""
        i = 0
        for a, b in zip(parts1, parts2):
            if a == b:
                i += 1
            else:
                break
        return i

    def _make_proximity_combinations(self, root: str, names: list[str]) -> list[list[str]]:
        """
        Like before: pick the name with the most hits as “anchor”,
        then for each anchor-path choose nearest matches for the others.
        """
        paths = self._collect_paths(root, names)
        # sanity
        for n in names:
            if not paths[n]:
                raise FileNotFoundError(f"{n!r} not found under {root!r}")

        # choose anchor = the key with max occurrences
        anchor = max(names, key=lambda n: len(paths[n]))
        combos = []
        # pre-split into parts
        split = {n: [p.split(os.sep) for p in paths[n]] for n in names}

        for a_path, a_parts in zip(paths[anchor], split[anchor]):
            row = []
            for n in names:
                if n == anchor:
                    row.append(a_path)
                else:
                    # pick the occurrence of n with largest common-prefix with this anchor
                    candidates = zip(paths[n], split[n])
                    best = max(candidates, key=lambda tup: self._common_prefix(a_parts, tup[1]))[0]
                    row.append(best)
            combos.append(row)
        return combos

    def _reorder_combos(self, combos: list[list[str]],
                   labels: list[str],
                   ordered_labels: list[str]
                  ) -> list[list[str]]:
        """
        Given:
          - combos:        e.g. [['p1','p2','p3'], ['q1','q2','q3'], …]
          - labels:        e.g. ['b','c','a']  # the meaning of each position
          - ordered_labels: e.g. ['a','b','c']
        Returns:
          - reordered:    e.g. [['p3','p1','p2'], …]
        """
        reordered = []
        for combo in combos:
            # build a mapping from the old labels to the corresponding paths
            m = dict(zip(labels, combo))
            # then reassemble in the desired order
            new_combo = [m[label] for label in ordered_labels]
            reordered.append(new_combo)
        return reordered

    def build_full_runs(self, root0: str, files0: list[str], 
                        root1: str, files1: list[str], 
                        labels: list[str], ordered_labels: list[str]):
        """
        Returns a list of dicts, each with keys
          'file0','file1','file2','file3','run_dir'
        one entry per each file0 occurrence.
        """
        combos0 = self._make_proximity_combinations(root0, files0)
        combos1 = self._make_proximity_combinations(root1, files1)

        combos_both, task_dirs = [], []
        for combo0 in combos0:
            for combo1 in combos1:
                combo_both = combo0 + combo1
                combos_both.append(combo_both)

                # Now solve for the run directory
                sec_parts = {f: f.split(os.sep) for f in combo1}
                longest_file = max(sec_parts, key=lambda f: len(sec_parts[f]))
                lp = sec_parts[longest_file]
                p0 = combo0[0].split(os.sep)
                c = self._common_prefix(p0, lp)
                # divergent tail from the long path
                tail = lp[c+1:-1] # ignore root1 and base filename
                parent0 = os.path.dirname(combo0[0])
                task_dirs.append(os.path.join(parent0, *tail))

        reordered_combos_both = self._reorder_combos(combos_both, labels, ordered_labels)

        return reordered_combos_both, task_dirs 

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

    def run(self, dry_run, task_command, run_tasks, 
                  cpus_per_task, gpus_per_task, 
                  task_arg_list, task_dir_list, 
                  write_restart_freq=1000000, buffer_time=1):

        if dry_run:
            self.dry_run(task_dir_list, run_tasks, cpus_per_task, gpus_per_task)
        else:
            from matensemble.matfluxGen import SuperFluxManager
            
            # Make a task list
            task_list=[i for i in range(len(run_tasks))]

            master = SuperFluxManager(gen_task_list=task_list,
                                  gen_task_cmd=task_command,
                                  ml_task_cmd=None,
                                  tasks_per_job=run_tasks,
                                  cores_per_task=cpus_per_task,
                                  gpus_per_task=gpus_per_task,
                                  write_restart_freq=1000000)
            
            # Make the task directories if they do not exist
            for task_dir in task_dir_list:
                os.makedirs(task_dir, exist_ok=True)

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
            flag = f"--{k}"
            if isinstance(v, bool):
                if v:
                    argv.append(flag)
            else:
                argv.extend([flag, str(v)])
        return argv

    def dict_to_str_list(self, d, labels, task_arg_list, ignore_list):
        task_arg_strs = []
        args_dct = {k: v for k, v in vars(d).items() if k not in ignore_list}
        for i, task_arg in enumerate(task_arg_list):
            task_arg_dct = deepcopy(args_dct)
            for j, arg in enumerate(task_arg):
                task_arg_dct[labels[j]] = arg
            task_arg_str = self.dict_to_argv(task_arg_dct)
            task_arg_strs.append(task_arg_str)
        
        return task_arg_strs

    def generic_task_command(self, python_file):
        python_exe = self.get_python()
        return ''.join([python_exe] + [' '] + [python_file])
