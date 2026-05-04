from abc import ABC, abstractmethod
from pymatgen.io.lammps.data import LammpsData
from pymatgen.io.ase import AseAtomsAdaptor
from ase.io import read
import numpy as np
from itertools import product
from copy import deepcopy
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple
import os
import glob
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
        name_set = set(names)
        for dp, dirs, files in os.walk(root, topdown=True):
            found_here = False
            for f in files:
                if f in name_set:
                    d[f].append(os.path.abspath(os.path.join(dp, f)))
                    found_here = True
            if found_here:
                dirs.clear()  # don't descend into subdirectories of this directory
        '''
        d = {n: [] for n in names}
        for dp, _, files in os.walk(root):
            for f in files:
                if f in d:
                    d[f].append(os.path.abspath(os.path.join(dp, f)))
        '''
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
        if not os.path.isdir(root):
            raise FileNotFoundError(f"{root} is not a valid directory!")

        paths = self._collect_paths(root, names)
        # sanity
        for n in names:
            if not paths[n]:
                raise FileNotFoundError(f"{n} not found under {root}")

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
                        labels: list[str], ordered_labels: list[str], 
                        finished_file: str | None = None):
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

                # Solve for the run directory
                sec_parts = {f: f.split(os.sep) for f in combo1}
                longest_file = max(sec_parts, key=lambda f: len(sec_parts[f]))
                lp = sec_parts[longest_file]
                p0 = combo0[0].split(os.sep)
                c = self._common_prefix(p0, lp)

                # Divergent tail from the long path
                tail = lp[c+1:-1] # ignore root1 and base filename
                parent0 = os.path.dirname(combo0[0])
                task_dir = os.path.join(parent0, *tail)

                # Check existence of finished_file in task_dir
                combo_both = combo0 + combo1
                if os.path.isdir(task_dir) and finished_file is not None:
                    pattern = os.path.join(task_dir, finished_file)
                    if glob.glob(pattern):
                        continue # finished_file pattern already written

                task_dirs.append(task_dir)
                combos_both.append(combo_both)

        reordered_combos_both = self._reorder_combos(combos_both, labels, ordered_labels)

        return reordered_combos_both, task_dirs

    def build_full_runs_v2(self, root0: str, files0: list[str],
                    root1: str, files1: list[str],
                    recipe_files: list[str],
                    labels: list[str], ordered_labels: list[str],
                    run_directory: str, 
                    inputs_directory: str,
                    finished_file: str | None = None):
        """
        root0/files0     - run directory files, proximity matched
        root1/files1     - inputs directory structure files, proximity matched
                           within their own subtree
        root1/recipe_files - recipe files (e.g. ase.json), cross-producted with
                             all structure combos
        """
        combos0 = self._make_proximity_combinations(root0, files0)

        # Proximity match structure files within inputs directory
        structure_combos = self._make_proximity_combinations(root1, files1)

        # All occurrences of each recipe file, to be cross-producted
        recipe_paths = self._collect_paths(root1, recipe_files)
        for n in recipe_files:
            if not recipe_paths[n]:
                raise FileNotFoundError(f"{n} not found under {root1}")

        import itertools
        recipe_combos = [list(combo) for combo in itertools.product(
            *[recipe_paths[n] for n in recipe_files]
        )]

        combos_both, task_dirs = [], []
        run_directory_name = Path(run_directory).name
        inputs_directory_name = Path(inputs_directory).name

        for combo0 in combos0:
            for struct_combo in structure_combos:
                for recipe_combo in recipe_combos:

                    # combo1 is the structure files + recipe files combined
                    combo1 = struct_combo + recipe_combo

                    # Always derive task_dir from the structure file, not the recipe
                    longest_file = max(struct_combo, key=lambda f: len(f.split(os.sep)))
                    rel = os.path.relpath(os.path.dirname(longest_file), root1)
                    #longest_file = max(combo1, key=lambda f: len(f.split(os.sep)))
                    #rel = os.path.relpath(os.path.dirname(longest_file), root1)
                    parent0 = os.path.dirname(combo0[0])
                    task_dir = os.path.join(parent0, rel)

                    combo_both = combo0 + combo1
                    mod_task_dir = self.modify_single_run_path(combo_both, 
                                                               task_dir, 
                                                               run_directory_name,
                                                               inputs_directory_name)
                    
                    if os.path.isdir(mod_task_dir) and finished_file is not None:
                        pattern = os.path.join(mod_task_dir, finished_file)
                        if glob.glob(pattern):
                            continue

                    task_dirs.append(mod_task_dir)
                    combos_both.append(combo_both)

        reordered_combos_both = self._reorder_combos(combos_both, labels, ordered_labels)
        return reordered_combos_both, task_dirs

    def modify_single_run_path(self, task_arg, run_path,
                             run_directory_name, inputs_directory_name):
        """
        Apply modify_write_paths logic to a single task_arg/run_path pair.
        Returns the modified run_path, or the original if no modification needed.
        """
        in_lammps_path_parts = Path(task_arg[-1]).parent.parts # Could break here

        if inputs_directory_name not in in_lammps_path_parts:
            return run_path  # can't modify, return original

        add_index = in_lammps_path_parts.index(inputs_directory_name)
        remaining = in_lammps_path_parts[add_index+1:]
        to_add = os.path.join(*remaining) if remaining else ""

        if not to_add:
            return run_path

        run_path_parts = Path(run_path).parts
        if run_directory_name not in run_path_parts:
            return run_path  # can't modify, return original

        where_add_index = run_path_parts.index(run_directory_name)
        base = Path(*run_path_parts[:where_add_index+1])
        tail_parts = run_path_parts[where_add_index+1:]
        tail = Path(*tail_parts) if tail_parts else Path()

        return str(base / to_add / tail)

    def batch_by_parent_v2(self, tasks, run_paths, labels, parent_levels=0):

        if parent_levels == 0:
            batched_tasks = []
            new_run_paths = []
            all_labels = labels + ['run_path']
            for i, run_path in enumerate(run_paths):
                batch = [[tasks[i][j]] for j in range(len(labels))]
                batch.append([run_path])
                batched_tasks.append(batch)
                new_run_paths.append(run_path)
            return batched_tasks, new_run_paths, run_paths

        sep = os.sep

        def get_parent_str(path, n):
            """Extract parent n levels up using string ops."""
            parts = path.split(sep)
            end = len(parts) - n
            if end <= 0:
                return sep
            return sep.join(parts[:end])

        all_labels = labels + ['run_path']
        groups = defaultdict(lambda: {label: [] for label in all_labels})

        for i, run_path in enumerate(run_paths):
            parent = get_parent_str(run_path, parent_levels)
            for j, label in enumerate(labels):
                groups[parent][label].append(tasks[i][j])
            groups[parent]['run_path'].append(run_path)

        def merge_child_paths_fast(dct):
            # Sort by depth (fewest separators = highest in tree = parent first)
            sorted_paths = sorted(dct.keys(), key=lambda p: p.count(sep))
            out = {}

            for path in sorted_paths:
                # Check if any existing key is a prefix of this path
                # Add sep to avoid /a/b matching /a/bc
                parent = next(
                    (p for p in out if path.startswith(p + sep) or path == p),
                    None
                )
                if parent is not None:
                    for k, v in dct[path].items():
                        out[parent].setdefault(k, []).extend(v)
                else:
                    out[path] = {k: list(v) for k, v in dct[path].items()}

            return out

        groups_merged = merge_child_paths_fast(groups)

        batched_tasks = []
        new_run_paths = []
        for parent, contents in groups_merged.items():
            use_batch = [contents[label] for label in all_labels]
            batched_tasks.append(use_batch)
            new_run_paths.append(parent)

        return batched_tasks, new_run_paths, run_paths

    def batch_by_parent(self, tasks, run_paths, labels, parent_levels=1):
        """
        Given tasks = [(ffield1, struct_path1), (ffield2, struct_path2), …],
        group them by the parent directory of each run_path defined by parent_levels.

        Returns: [
            [[structA, structB, …], [ffieldA, ffieldB, …]],
            [[structC, structD, …], [ffieldC, ffieldD, …]],
            …
        ]
        """
        def get_parent(path, parent_levels):
            p = Path(path)
            for _ in range(parent_levels):
                p = p.parent
            return p

        def merge_child_paths(dct):
            out = {}

            # Sort so parents come before children
            for path in sorted(dct, key=lambda p: Path(p).parts):
                path_obj = Path(path)
                parent = next((p for p in out if path_obj.is_relative_to(p)), None)

                if parent:
                    # Merge into parent
                    for k, v in dct[path].items():
                        out[parent].setdefault(k, []).extend(v)
                else:
                    # Copy new parent entry
                    out[path] = {k: list(v) for k, v in dct[path].items()}

            return out

        groups = defaultdict(lambda: {label: [] for label in labels + ['run_path']})

        # Group the tasks by parent directory
        for i, run_path in enumerate(run_paths):
            parent = get_parent(run_path, parent_levels)
            for j, label in enumerate(labels):
                groups[parent][label].append(tasks[i][j])
            groups[parent]['run_path'].append(run_paths[i])

        # Merge the parent directories by super-parents
        groups_merged = merge_child_paths(groups)

        # Build the final output in arbitrary parent‐directory order:
        batched_tasks = []
        new_run_paths = []
        for parent, contents in groups_merged.items():
            use_batch = []
            for label in labels + ['run_path']: # Add run path to arguments here
                use_batch.append(contents[label])
            batched_tasks.append(use_batch)
            new_run_paths.append(parent)

        return batched_tasks, new_run_paths, run_paths

    def dict_to_argv(self, d, bool_arg=True):
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
                if v and bool_arg: # Pass argument as bool
                    argv.extend([flag, str(v)])
                else:
                    argv.append(flag) # Treat as single flag
            else:
                argv.extend([flag, str(v)])
        return argv

    def get_python(self):
        try:
            python_exe = os.path.join(os.environ['CONDA_PREFIX'], 'bin', 'python')
        except KeyError:
            python_exe = 'python'
        return python_exe

    def dry_run(self, paths, task_command, tasks, cpus_per_task, gpus_per_task):
        print(f'Task Command: {task_command}\n')
        for i, path in enumerate(paths):
            print(f'path: {paths[i]}, tasks: {tasks[i]}\n')
        print(f'Total tasks = {int(np.sum(tasks))}; cpus_per_task={cpus_per_task}; gpus_per_task={gpus_per_task}')

    def run(self, dry_run, task_command, run_tasks, 
                  cpus_per_task, gpus_per_task, 
                  task_arg_list, task_dir_list,
                  make_paths_list, 
                  write_restart_freq=1000000, buffer_time=1):

        if dry_run:
            self.dry_run(task_dir_list, task_command, run_tasks, cpus_per_task, gpus_per_task)
        else:
            from matensemble.manager import SuperFluxManager
            
            # Make a task list
            task_list=[i for i in range(len(run_tasks))]

            master = SuperFluxManager(gen_task_list=task_list,
                                  gen_task_cmd=task_command,
                                  tasks_per_job=run_tasks,
                                  cores_per_task=cpus_per_task,
                                  gpus_per_task=gpus_per_task,
                                  write_restart_freq=1000000)
            
            # Make directories for outputs if they do not exist
            for make_path in make_paths_list:
                os.makedirs(make_path, exist_ok=True)

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
            try: # For lammps-data format
                return LammpsData.from_file(lmp_file_path, atom_style=self.options['atom_style']).structure
            except KeyError: # For lammps-dump-text format
                try:
                    aaa = AseAtomsAdaptor()
                    return aaa.get_structure(read(lmp_file_path, format='lammps-dump-text'))
                except StopIteration: # For other structure formats
                    return aaa.get_structure(read(lmp_file_path))

    def modify_write_paths(self, task_arg_list, run_paths, run_directory, inputs_directory):
        run_directory_name = Path(run_directory).name
        inputs_directory_name = Path(inputs_directory).name

        mod_run_paths = []

        for i, task_arg in enumerate(task_arg_list):
            in_lammps_path_parts = Path(task_arg[1]).parent.parts

            if inputs_directory_name not in in_lammps_path_parts:
                raise ValueError(f"'{inputs_directory_name}' not found in path: {task_arg[1]}")
            add_index = in_lammps_path_parts.index(inputs_directory_name)

            remaining = in_lammps_path_parts[add_index+1:]
            to_add = os.path.join(*remaining) if remaining else ""

            if to_add:
                run_path_parts = Path(run_paths[i]).parts

                if run_directory_name not in run_path_parts:
                    raise ValueError(f"'{run_directory_name}' not found in path: {run_paths[i]}")
                where_add_index = run_path_parts.index(run_directory_name)

                base = Path(*run_path_parts[:where_add_index+1])
                tail_parts = run_path_parts[where_add_index+1:]
                tail = Path(*tail_parts) if tail_parts else Path()
                mod_run_path = str(base / to_add / tail)
                mod_run_paths.append(mod_run_path)
            else:
                mod_run_paths.append(run_paths[i])

        return mod_run_paths

    def sorting_function(self, path):
        ''' Sort by structure length '''
        structure = self.read_structure_from_lammps(path)
        return len(structure)

    def get_tasks(self, structure_paths, atoms_per_task=10):
        ''' Set based on structure length '''
        structures = [self.read_structure_from_lammps(path) for path in structure_paths]
        return [max(np.floor(len(s)/atoms_per_task).astype(int), 1) for i, s in enumerate(structures)]

    def generic_task_command(self, python_file, user_command=''):
        ''' Builds a generic task command for the LAMMPs python interface '''
        if user_command:
            python_exe = user_command
        else:
            python_exe = self.get_python()
        return f"{python_exe.strip()} {python_file.strip()}" 

class JaxReaxFFMatEnsemble(MatEnsembleJob):
    def __init__(self, run_directory, inputs_directory, **kwargs):
        super().__init__(run_directory, inputs_directory, **kwargs)

    def sorting_function(self, path):
        return str.lower

    def get_tasks(self, paths, tasks_per_path):
        return [tasks_per_path for path in paths]

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

    def generic_task_command(self):
        pass

class MACEMatEnsemble(MatEnsembleJob):
    def __init__(self, run_directory, inputs_directory, **kwargs):
        super().__init__(run_directory, inputs_directory, **kwargs)

    def sorting_function(self, path):
        return str.lower

    def get_tasks(self, paths):
        return [1 for path in paths]

    def construct_tasks(self, task_arg_list, run_paths, fits_per_runpath, random=False, finished_file=None, upper=10000):
        new_task_arg_list = []
        new_run_paths = []

        for i, run_path in enumerate(run_paths):
            if random:
                seeds = np.random.choice(np.arange(0, upper), size=fits_per_runpath, replace=False)
            else:
                seeds = [i for i in range(fits_per_runpath)]

            for j, seed in enumerate(seeds):
                new_run_path = os.path.join(run_path, str(seed))
                if os.path.isdir(new_run_path) and finished_file is not None:
                    pattern = os.path.join(new_run_path, finished_file)
                    if glob.glob(pattern):
                        continue # finished_file pattern already written
                
                new_task_arg_list.append(task_arg_list[i]) # Same inputs here
                new_run_paths.append(new_run_path)

        return new_task_arg_list, new_run_paths

    def to_str_list(self, labels, task_arg_list, run_paths):
        # Create the base argument string for MACE force field fitting
        task_arg_strs = []
        for i, run_path in enumerate(run_paths):
            seed = Path(run_path).name
            name = f'MACE_{seed}'
            task_arg_dct = {'name': name, 'seed': seed}
            for j, label in enumerate(labels):
                task_arg_dct[label] = task_arg_list[i][j]
            task_arg_str = self.dict_to_argv(task_arg_dct)
            task_arg_strs.append(task_arg_str)

        return task_arg_strs

    def generic_task_command(self):
        pass
