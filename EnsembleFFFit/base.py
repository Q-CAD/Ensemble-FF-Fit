from abc import ABC, abstractmethod
from pathlib import Path
from collections import defaultdict
import itertools
import os
import glob

from EnsembleFFFit.utilities.general import import_module_from_path


class MatEnsembleJob(ABC):
    def __init__(self, run_directory, inputs_directory, **kwargs):
        self.run_directory = run_directory
        self.inputs_directory = inputs_directory
        self.options = kwargs

    @abstractmethod
    def build_full_runs(self, *args, **kwargs):
        """Abstract hook: construct the (task_arg_list, run_paths) mapping for this backend's run/inputs layout."""
        pass

    @abstractmethod
    def batch_by_parent(self, *args, **kwargs):
        """Abstract hook: group constructed tasks by parent directory for this backend's batching needs."""
        pass

    def _collect_paths(self, root: str, names: list[str]) -> dict[str, list[str]]:
        """
        Walk `root` and collect absolute paths of any file whose name is in `names`;
        once a directory containing *every* target filename is found, stop
        descending into its subdirectories (that directory is a complete match, so
        nothing relevant can be nested below it). A directory with only some of the
        target filenames does not prune the walk -- the remaining names may still
        live deeper below it (e.g. shared train/test files near the root with
        per-variant config files nested underneath). Returns `{name: [paths]}`.
        """
        d = {n: [] for n in names}
        name_set = set(names)
        for dp, dirs, files in os.walk(root, topdown=True):
            found_here = {f for f in files if f in name_set}
            for f in found_here:
                d[f].append(os.path.abspath(os.path.join(dp, f)))
            if found_here == name_set:
                dirs.clear()  # complete match; nothing relevant can be nested below
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


class MDMatEnsemble(MatEnsembleJob):
    """
    Supports MD-driven backends whose run construction needs a recipe file
    (e.g. LAMMPS `.in`/control files, or an ASE run config) cross-producted
    against structure files, distinct from a single flat check-file per run
    (that's MACEMatEnsemble's shape) -- currently used for LAMMPS, ASE, and
    (in principle) TorchSim MD drivers.
    """

    def __init__(self, run_directory, inputs_directory, **kwargs):
        super().__init__(run_directory, inputs_directory, **kwargs)
        """ MD self.options keys are backend-dependent, e.g. "ffield"/"in_file"/"control"/"structure" for LAMMPS """

    def build_full_runs(self, root0: str, files0: list[str],
                    root1: str, files1: list[str],
                    recipe_files: list[str],
                    labels: list[str], ordered_labels: list[str],
                    run_directory: str,
                    inputs_directory: str,
                    finished_file: str | None = None):
        """
        root0/files0        - run directory files, proximity matched
        root1/files1         - inputs directory structure files, proximity matched
                                within their own subtree
        root1/recipe_files   - recipe files (e.g. ase.json), cross-producted with
                                all structure combos
        labels/ordered_labels - `labels` names each position in the raw combo
                                (run files + structure files + recipe files, in
                                that concatenation order), and `ordered_labels`
                                gives the order to reassemble them into for output.
        run_directory/inputs_directory - passed through to `_modify_single_run_path`
                                so the derived task_dir can be corrected to reflect
                                where, under `inputs_directory`, the matched recipe
                                file actually lives.
        finished_file        - if given, skip any combo whose (modified) task_dir already
                                contains a file matching this glob pattern.

        Returns (reordered_combos, task_dirs): reordered_combos is a list of path
        lists ordered per `ordered_labels`, and task_dirs is the parallel list of
        derived task directories.
        """
        combos0 = self._make_proximity_combinations(root0, files0)

        # Proximity match structure files within inputs directory
        structure_combos = self._make_proximity_combinations(root1, files1)

        # All occurrences of each recipe file, to be cross-producted
        recipe_paths = self._collect_paths(root1, recipe_files)
        for n in recipe_files:
            if not recipe_paths[n]:
                raise FileNotFoundError(f"{n} not found under {root1}")

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
                    parent0 = os.path.dirname(combo0[0])
                    task_dir = os.path.join(parent0, rel)

                    combo_both = combo0 + combo1
                    if recipe_combo:
                        mod_task_dir = self._modify_single_run_path(recipe_combo,
                                                                   task_dir,
                                                                   run_directory_name,
                                                                   inputs_directory_name)
                    else:
                        mod_task_dir = task_dir

                    if os.path.isdir(mod_task_dir) and finished_file is not None:
                        pattern = os.path.join(mod_task_dir, finished_file)
                        if glob.glob(pattern):
                            continue

                    task_dirs.append(mod_task_dir)
                    combos_both.append(combo_both)

        reordered_combos_both = self._reorder_combos(combos_both, labels, ordered_labels)
        return reordered_combos_both, task_dirs

    def _modify_single_run_path(self, recipe_arg, run_path,
                             run_directory_name, inputs_directory_name):
        """
        Correct `run_path` for the location of the recipe file, which
        `build_full_runs`'s own task_dir formula deliberately ignores (it
        derives task_dir from the structure file only). `recipe_arg` is the
        recipe-file combo for this task -- callers should skip calling this
        entirely when there is no recipe file, rather than passing some other
        file positionally (that file's location is already fully accounted for
        in `run_path`, and re-inserting it here would double it up). Returns the
        modified run_path, or the original if no modification needed.
        """
        recipe_path_parts = Path(recipe_arg[-1]).parent.parts

        if inputs_directory_name not in recipe_path_parts:
            return run_path  # can't modify, return original

        add_index = recipe_path_parts.index(inputs_directory_name)
        remaining = recipe_path_parts[add_index+1:]
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

    def batch_by_parent(self, tasks, run_paths, labels, parent_levels=0):
        """
        Group tasks by the directory `parent_levels` levels above each run_path (or
        verbatim per-path grouping if `parent_levels==0`), merging any resulting
        groups where one path is an ancestor of another.

        Returns `(batched_tasks, new_run_paths, run_paths)`.
        """
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

    def build_lists(self, lammps_task, parent_levels, check_files, finished_file=None):
        """
        Build the (task_arg_list, run_paths, make_paths, task_command,
        batch_labels) needed to submit one chore per (or one chore per batch
        of) MD run, using `self.run_directory`/`self.inputs_directory`/
        `self.options` (set by `__init__`). `lammps_task` names the
        user-authored driver script living under `self.inputs_directory`
        (e.g. an ASE or LAMMPS execution script) -- kept general across MD
        backends by construction (proximity matching + recipe/structure
        cross-product), not tied to any one backend's file layout beyond
        which `self.options` keys are present. `finished_file` (a glob
        pattern, e.g. "properties.json") skips any run whose task_dir already
        contains a match -- see `build_full_runs`.
        """
        task_command = os.path.abspath(os.path.join(self.inputs_directory, lammps_task))
        if not os.path.isfile(task_command):
            raise ValueError(f'Invalid task command {task_command}; file does not exist in {self.inputs_directory}!')

        # Split the files to be checked in --run_directory vs --input_directory
        inputs_directory_keys = [key for key in self.options.keys() if key not in check_files + ['lammps_task', 'atom_style']]

        # in_file is always the recipe file; everything else is a structure file
        recipe_keys = [k for k in inputs_directory_keys if k == 'in_file']
        structure_keys = [k for k in inputs_directory_keys if k != 'in_file']

        # Generate combinations of run paths and task arguments
        task_arg_list, run_paths = self.build_full_runs(
            root0=self.run_directory,
            files0=[self.options[c] for c in check_files],
            root1=self.inputs_directory,
            files1=[self.options[k] for k in structure_keys],
            recipe_files=[self.options[k] for k in recipe_keys],
            labels=check_files + structure_keys + recipe_keys,
            ordered_labels=check_files + structure_keys + recipe_keys,
            finished_file=finished_file,
            run_directory=self.run_directory,
            inputs_directory=self.inputs_directory
        )

        # Batch the runs based on the parent level
        batch_labels = check_files + inputs_directory_keys
        task_arg_list, run_paths, make_paths = self.batch_by_parent(task_arg_list, run_paths, batch_labels, parent_levels)

        return task_arg_list, run_paths, make_paths, task_command, batch_labels

    @staticmethod
    def run_individual(task_dict):
        """
        Import the user-supplied MD driver script (`task_dict['task_command']`)
        by path and dispatch to its entry-point function (named by
        `task_dict['entry_point']`, since the function being called is a
        general, configurable choice -- not hardcoded to any one driver's
        function name). The module name is derived from the driver script's
        own filename (extension stripped), not hardcoded either, so this
        works for any user-authored driver script, not just one specific one.
        """
        module_name = Path(task_dict['task_command']).stem
        driver = import_module_from_path(module_name, task_dict['task_command'])
        entry_point = getattr(driver, task_dict['entry_point'])
        return entry_point(task_dict['ffield'], task_dict['structure'], task_dict['output'])


class JaxReaxFFMatEnsemble(MatEnsembleJob):
    def __init__(self, run_directory, inputs_directory, **kwargs):
        super().__init__(run_directory, inputs_directory, **kwargs)

    def build_full_runs(self, root0: str, files0: list[str],
                        root1: str, files1: list[str],
                        labels: list[str], ordered_labels: list[str],
                        finished_file: str | None = None):
        """
        Cross-product every proximity-matched combo from root0/files0 with every
        proximity-matched combo from root1/files1 (minus any combo whose derived
        task_dir already contains a `finished_file` match), and derive a run/task
        directory for each surviving combo.

        Returns (reordered_combos, task_dirs): reordered_combos is a list of path
        lists ordered per `ordered_labels`, and task_dirs is the parallel list of
        derived task directories.
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

    def batch_by_parent(self, tasks, run_paths, labels, parent_levels=1):
        """
        Given tasks = [(ffield1, struct_path1), (ffield2, struct_path2), …],
        group them by the parent directory of each run_path defined by parent_levels.

        Returns: [
            [[structA, structB, …], [ffieldA, ffieldB, …]],
            [[structC, structD, …], [ffieldC, ffieldD, …]],
            …
        ]
        (Illustrative only — the actual per-group ordering of inner lists follows
        the caller-supplied `labels` list, not a fixed struct/ffield order.)
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


class MACEMatEnsemble(MatEnsembleJob):
    def __init__(self, run_directory, inputs_directory, **kwargs):
        super().__init__(run_directory, inputs_directory, **kwargs)

    def build_full_runs(self, root0: str, files0: list[str],
                        root1: str, files1: list[str],
                        labels: list[str], ordered_labels: list[str],
                        finished_file: str | None = None):
        """
        Cross-product every proximity-matched combo from root0/files0 with every
        proximity-matched combo from root1/files1 (minus any combo whose derived
        task_dir already contains a `finished_file` match), and derive a run/task
        directory for each surviving combo.

        Returns (reordered_combos, task_dirs): reordered_combos is a list of path
        lists ordered per `ordered_labels`, and task_dirs is the parallel list of
        derived task directories.
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

    def batch_by_parent(self, tasks, run_paths, labels, parent_levels=1):
        """
        Given tasks = [(ffield1, struct_path1), (ffield2, struct_path2), …],
        group them by the parent directory of each run_path defined by parent_levels.

        Returns: [
            [[structA, structB, …], [ffieldA, ffieldB, …]],
            [[structC, structD, …], [ffieldC, ffieldD, …]],
            …
        ]
        (Illustrative only — the actual per-group ordering of inner lists follows
        the caller-supplied `labels` list, not a fixed struct/ffield order.)
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

    def build_mace_dcts(self, check_files, finished_file=None):
        """
        Build the list of per-run MACE override dicts (foundation_model/config/
        train_file/test_file plus results_dir/work_dir/name) needed to submit
        one chore per fit, using `self.run_directory`/`self.inputs_directory`/
        `self.options` (set by `__init__`). `finished_file` (a glob pattern,
        e.g. "MACE_*.model") skips any run whose task_dir already contains a
        match -- see `build_full_runs`.
        """
        inputs_directory_keys = [key for key in self.options.keys() if key not in check_files]
        labels = check_files + inputs_directory_keys

        task_arg_list, run_paths = self.build_full_runs(
            root0=self.run_directory, files0=[self.options[c] for c in check_files],
            root1=self.inputs_directory, files1=[self.options[k] for k in inputs_directory_keys],
            labels=labels, ordered_labels=labels, finished_file=finished_file
        )

        task_arg_dct_list = [dict(zip(labels, task_arg)) for task_arg in task_arg_list]
        run_path_arg_list = [{'results_dir': run_path, 'work_dir': run_path, 'name': f'MACE_{str(i)}'} for i, run_path in enumerate(run_paths)]
        mace_arg_dct_list = [{**task_dct, **run_dct} for task_dct, run_dct in zip(task_arg_dct_list, run_path_arg_list, strict=True)]

        return mace_arg_dct_list

    @staticmethod
    def run_individual(overrides):
        """
        Run a single MACE fit from a per-run `overrides` dict (foundation_model/
        config/train_file/test_file/results_dir/work_dir/name/...). Imports
        `mace` lazily -- this method is the only thing in this module that
        needs the `mace` extra installed, so importing EnsembleFFFit.base
        itself doesn't require it.
        """
        from mace.cli.run_train import run
        from mace.tools import build_default_arg_parser

        name = overrides.get("name", "MatEnsemble")
        config_path = overrides.get("config")

        initial_args = ["--name", name]
        if config_path:
            initial_args += ["--config", config_path]

        args = build_default_arg_parser().parse_args(initial_args)

        for key, value in overrides.items():
            if key in ("name", "config", "finished_file"):
                continue  # already handled above / below
            setattr(args, key, value)

        work_path = overrides.get("work_dir")
        if work_path:
            os.makedirs(work_path, exist_ok=True)

        # `finished_file` here is deliberately an execution-time skip, not a
        # build-time filter (contrast build_mace_dcts's own finished_file
        # param) -- a caller relying on this chore's *completion* to trigger
        # further work (e.g. a Pipeline.strategy processing chore) needs the
        # chore to still run and succeed even when the fit itself was already
        # done, rather than never being submitted at all.
        finished_file = overrides.get("finished_file")
        already_done = bool(finished_file and work_path and glob.glob(os.path.join(work_path, finished_file)))
        if not already_done:
            run(args)

        # results_dir/name are included (not just status) so callers watching
        # this chore's completion (e.g. a Pipeline.strategy processing chore)
        # can locate the fitted model file directly, without re-walking the
        # run_directory -- MACE writes it to f"{results_dir}/{name}.model".
        return {"status": "complete", "results_dir": overrides.get("results_dir"), "name": name}
