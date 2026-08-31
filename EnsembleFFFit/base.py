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
    (that's FFMatEnsemble's shape) -- currently used for LAMMPS, ASE, and
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
    def combine_task_dicts(task_dicts):
        """
        Merge a list of same-shaped task_dicts (as returned by
        build_task_dicts) into a single task_dict covering all of their
        ffield/structure/output[/in_file] entries -- for callers that want
        everything in one chore regardless of how many batches
        parent_levels produced, without needing a parent_levels value tuned
        to a specific, uniform tree depth (e.g. a per-fit-variant caller
        scoped to a run_directory containing just that one variant, where
        the structure tree's depth/shape isn't known in advance). Returns
        None if `task_dicts` is empty.
        """
        if not task_dicts:
            return None

        combined = {
            'task_command': task_dicts[0]['task_command'],
            'entry_point': task_dicts[0]['entry_point'],
            'ffield': [v for td in task_dicts for v in td['ffield']],
            'structure': [v for td in task_dicts for v in td['structure']],
            'output': [v for td in task_dicts for v in td['output']],
        }
        if 'in_file' in task_dicts[0]:
            combined['in_file'] = [v for td in task_dicts for v in td['in_file']]

        return combined

    def build_task_dicts(self, lammps_task, parent_levels, check_files, entry_point, finished_file=None):
        """
        Wrap build_lists, flattening each resulting batch into the
        ready-to-submit task_dict shape run_individual expects (ffield/
        structure/output[/in_file] lists, plus task_command/entry_point) --
        this is the flattening pattern callers used to hand-roll themselves
        (once scoped to a single batch, once across many batches), now done
        once here regardless of how many batches parent_levels produces.
        Returns a list of task_dicts, one per batch.
        """
        task_arg_list, run_paths, make_paths, task_command, batch_labels = self.build_lists(
            lammps_task, parent_levels, check_files, finished_file=finished_file)

        ffield_idx = batch_labels.index('ffield')
        structure_idx = batch_labels.index('structure')
        output_idx = len(batch_labels)
        in_file_idx = batch_labels.index('in_file') if 'in_file' in batch_labels else None

        task_dicts = []
        for batch in task_arg_list:
            task_dict = {
                'task_command': task_command,
                'entry_point': entry_point,
                'ffield': batch[ffield_idx],
                'structure': batch[structure_idx],
                'output': batch[output_idx],
            }
            if in_file_idx is not None:
                task_dict['in_file'] = batch[in_file_idx]
            task_dicts.append(task_dict)

        return task_dicts

    @staticmethod
    def build_flat_task_dicts(structures_root, foundation_model, in_file, output_root,
                              task_command, entry_point, structure_filename="POSCAR",
                              finished_file=None):
        """
        One task_dict per structure found under structures_root, each
        cross-producted against a single fixed foundation_model and a
        single fixed in_file/recipe -- for cases where the structures are
        already a known, flat, explicit list (e.g. pre-sampled MD starting
        structures) rather than something build_lists' proximity-matching
        machinery needs to discover. Deliberately bypasses build_lists/
        _modify_single_run_path, which would otherwise insert
        structures_root's own subpath (e.g. a "structures/" segment) into
        the derived output path. `finished_file` (a glob pattern) skips any
        structure whose output_dir already contains a match.
        """
        struct_dirs = sorted(
            dirpath for dirpath, _, files in os.walk(structures_root) if structure_filename in files
        )

        task_dicts = []
        for struct_dir in struct_dirs:
            rel = os.path.relpath(struct_dir, structures_root)
            output_dir = os.path.join(output_root, rel)
            if finished_file and os.path.isdir(output_dir) and glob.glob(os.path.join(output_dir, finished_file)):
                continue
            task_dicts.append({
                'task_command': task_command,
                'entry_point': entry_point,
                'ffield': [foundation_model],
                'structure': [os.path.join(struct_dir, structure_filename)],
                'output': [output_dir],
                'in_file': [in_file],
            })
        return task_dicts

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

        Passes a 4th list, `in_file`, alongside ffield/structure/output --
        e.g. an ASE run's config yaml, or a LAMMPS input file with variables
        to set. `build_lists` already discovers/proximity-matches 'in_file'
        (and 'control') as option keys, but previously nothing threaded that
        match through to actual execution; this closes that gap. Defaults to
        a same-length list of None if the caller's task_dict never set
        'in_file' at all, so drivers that don't need a recipe file (e.g.
        ase_mace.py's single points) keep working unchanged -- they just
        need to accept (and can ignore) this 4th parameter now.
        """
        module_name = Path(task_dict['task_command']).stem
        driver = import_module_from_path(module_name, task_dict['task_command'])
        entry_point = getattr(driver, task_dict['entry_point'])
        in_file = task_dict.get('in_file', [None] * len(task_dict['ffield']))
        return entry_point(task_dict['ffield'], task_dict['structure'], task_dict['output'], in_file)


class FFMatEnsemble(MatEnsembleJob):
    """
    Supports force-field-fitting backends. Matches a single flat check-file
    per run (e.g. a foundation model) against inputs_directory folders --
    distinct from DFTMatEnsemble/MDMatEnsemble's recipe/structure
    cross-product, since a fit's "recipe" (train/test/config files) is
    itself just another proximity-matched file here, not a separate axis to
    cross with.

    Deliberately the most input-shape-flexible of the three concrete
    MatEnsembleJob subclasses -- an intentional divergence, not drift to
    reconcile:
    - DFTMatEnsemble.options is fixed to exactly 'rmg_yaml' +
      'structure_filename' -- not because RMG is the only DFT backend
      expected (VASP/Quantum Espresso/possibly Gaussian are all planned),
      but because structure+recipe is expected to keep working as a generic
      contract across those codes, which pymatgen/ASE already have solid
      input-generation support for. RMG is the outlier that needed bespoke,
      hand-written support (the pyRMG package, an optional dependency --
      see the 'rmg' extra) specifically because it's obscure enough to lack
      that kind of Python tooling.
    - MDMatEnsemble.options is backend-dependent but still funnels into a
      small, fixed positional shape at the driver-script boundary
      (ffield/structure/output/in_file) -- the set of MD drivers expected
      here (ASE, LAMMPS, TorchSim) is itself small and stable.
    - This class's options are fully caller-defined (whatever keys
      build_ff_dcts's check_files/inputs-directory-keys end up being), and
      run_individual passes the whole resulting overrides dict to the
      driver script as one argument rather than unpacking into positional
      lists like DFTMatEnsemble.run_individual/MDMatEnsemble.run_individual
      do -- unpacking into fixed positional slots here would mean hardcoding
      backend-specific key names (e.g. MACE's 'foundation_model'/
      'train_file') back into this generic class. Different FF-fitting
      codes (MACE, JAX-ReaxFF, CHGNet, ...) have far more divergent
      input/hyperparameter shapes than DFT or MD codes typically do, so this
      is the one class expected to see real variation across backends, and
      the driver-script contract needed to flex accordingly. See
      examples/Frontier/RMG_MACE_ASE/FF/mace_fit.py's own docstring for the
      concrete (MACE) case.
    """

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

    def build_ff_dcts(self, ff_task, check_files, entry_point, finished_file=None):
        """
        Build the list of per-run FF-fitting override dicts (whatever
        backend-specific option keys the caller passed to `__init__` --
        e.g. foundation_model/config/train_file/test_file for MACE -- plus
        results_dir/work_dir/name/ff_task/entry_point) needed to submit one
        chore per fit, using `self.run_directory`/`self.inputs_directory`/
        `self.options` (set by `__init__`). `ff_task`/`entry_point` name the
        backend-specific driver script and its entry-point function --
        embedded into every returned dict here (not left for the caller to
        inject afterward), matching `MDMatEnsemble.build_task_dicts`'s
        convention rather than `DFTMatEnsemble`'s old one. `finished_file`
        (a glob pattern, e.g. "MACE_*.model") skips any run whose task_dir
        already contains a match -- see `build_full_runs`.
        """
        inputs_directory_keys = [key for key in self.options.keys() if key not in check_files]
        labels = check_files + inputs_directory_keys

        task_arg_list, run_paths = self.build_full_runs(
            root0=self.run_directory, files0=[self.options[c] for c in check_files],
            root1=self.inputs_directory, files1=[self.options[k] for k in inputs_directory_keys],
            labels=labels, ordered_labels=labels, finished_file=finished_file
        )

        task_arg_dct_list = [dict(zip(labels, task_arg)) for task_arg in task_arg_list]
        run_path_arg_list = [{'results_dir': run_path, 'work_dir': run_path, 'name': f'FF_{str(i)}',
                               'ff_task': ff_task, 'entry_point': entry_point}
                              for i, run_path in enumerate(run_paths)]
        ff_arg_dct_list = [{**task_dct, **run_dct} for task_dct, run_dct in zip(task_arg_dct_list, run_path_arg_list, strict=True)]

        return ff_arg_dct_list

    @staticmethod
    def run_individual(task_dict):
        """
        Import the user-supplied FF-fitting driver script (`task_dict['ff_task']`)
        by path and dispatch to its entry-point function (named by
        `task_dict['entry_point']`) -- same dynamic-import-and-dispatch
        mechanism as MDMatEnsemble.run_individual/DFTMatEnsemble.run_individual.
        The entry point receives the remaining override dict (foundation_model/
        config/train_file/test_file/results_dir/work_dir/name/... for MACE,
        or whatever a different backend's driver needs) as a single argument --
        unlike MD/DFT's positional-list convention, an FF fit is already
        one-per-chore, so there's nothing to batch/unpack here.
        """
        module_name = Path(task_dict['ff_task']).stem
        driver = import_module_from_path(module_name, task_dict['ff_task'])
        entry_point = getattr(driver, task_dict['entry_point'])
        overrides = {k: v for k, v in task_dict.items() if k not in ('ff_task', 'entry_point')}
        return entry_point(overrides)


class DFTMatEnsemble(MatEnsembleJob):
    """
    Supports DFT backends (currently just RMG) whose run construction needs a
    recipe file (an RMG input YAML) cross-producted against structure files
    (POSCAR/CONTCAR) -- the same recipe/structure-cross-product shape as
    MDMatEnsemble, reused here rather than inherited from it since the other
    concrete backend in this module (FFMatEnsemble) already duplicates its
    build_full_runs/batch_by_parent rather than sharing a common non-abstract
    base for them.
    """

    def __init__(self, run_directory, inputs_directory, **kwargs):
        super().__init__(run_directory, inputs_directory, **kwargs)
        """ DFT self.options keys: 'rmg_yaml' (recipe), a structure-filename key
        (e.g. 'structure_filename': 'POSCAR'), plus scalar (non-file) config:
        'dft_task', 'entry_point', 'pseudopotentials_directory', 'gpus_per_node',
        'electrons_per_gpu', 'grid_divisibility_exponent'. """

    def build_full_runs(self, root0: str, files0: list[str],
                    root1: str, files1: list[str],
                    recipe_files: list[str],
                    labels: list[str], ordered_labels: list[str],
                    run_directory: str,
                    inputs_directory: str,
                    finished_file: str | None = None):
        """
        Identical shape to MDMatEnsemble.build_full_runs -- see there for the
        parameter-by-parameter explanation. root0/files0 anchors on
        run_directory files, root1/files1 proximity-matches structure files
        under inputs_directory, recipe_files (the RMG input YAML) is
        cross-producted against every structure combo, and finished_file
        skips any combo whose derived task_dir already has a match.
        """
        combos0 = self._make_proximity_combinations(root0, files0)
        structure_combos = self._make_proximity_combinations(root1, files1)

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

                    combo1 = struct_combo + recipe_combo

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
        """Identical to MDMatEnsemble._modify_single_run_path -- see there."""
        recipe_path_parts = Path(recipe_arg[-1]).parent.parts

        if inputs_directory_name not in recipe_path_parts:
            return run_path

        add_index = recipe_path_parts.index(inputs_directory_name)
        remaining = recipe_path_parts[add_index+1:]
        to_add = os.path.join(*remaining) if remaining else ""

        if not to_add:
            return run_path

        run_path_parts = Path(run_path).parts
        if run_directory_name not in run_path_parts:
            return run_path

        where_add_index = run_path_parts.index(run_directory_name)
        base = Path(*run_path_parts[:where_add_index+1])
        tail_parts = run_path_parts[where_add_index+1:]
        tail = Path(*tail_parts) if tail_parts else Path()

        return str(base / to_add / tail)

    def batch_by_parent(self, tasks, run_paths, labels, parent_levels=0):
        """Identical to MDMatEnsemble.batch_by_parent -- see there."""
        if parent_levels == 0:
            batched_tasks = []
            new_run_paths = []
            for i, run_path in enumerate(run_paths):
                batch = [[tasks[i][j]] for j in range(len(labels))]
                batch.append([run_path])
                batched_tasks.append(batch)
                new_run_paths.append(run_path)
            return batched_tasks, new_run_paths, run_paths

        sep = os.sep

        def get_parent_str(path, n):
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
            sorted_paths = sorted(dct.keys(), key=lambda p: p.count(sep))
            out = {}

            for path in sorted_paths:
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

    def build_dft_dcts(self, dft_task, check_files, entry_point, finished_file=None):
        """
        Build the list of per-run DFT override dicts needed to submit one
        chore per RMG calculation, using `self.run_directory`/
        `self.inputs_directory`/`self.options` (set by `__init__`).
        `dft_task`/`entry_point` name the DFT driver script and its
        entry-point function -- embedded into every returned dict here (not
        left for the caller to inject afterward), matching
        `MDMatEnsemble.build_task_dicts`/`FFMatEnsemble.build_ff_dcts`'s
        convention. `finished_file` (a glob pattern, e.g. "forcefield.xml")
        skips any run whose task_dir already contains a match -- see
        `build_full_runs`.

        Deliberately deals only in file paths, never a resolved Structure/
        Atoms object: the structure this dict points at (via whichever
        structure-filename option was configured, e.g. POSCAR) may be stale
        by the time the chore actually executes (a prior RMG run in the same
        directory may have since produced a newer rmg_input.*.log or
        rmg_input) -- the driver script resolves the authoritative structure
        at execution time via pick_structure.pick_best_structure, not here.

        Also computes 'allocated_nodes', a build-time node-count *estimate*
        (via rmg_input.compute_grid_and_resources against whichever structure
        file happens to be on disk right now) purely to size each chore's
        Resources at submission time. This is only a starting point: if the
        caller changes the chore's actual node allocation for any reason, it
        must update this key to match what it actually requested, since
        RMG.write_input's consistency check (run against the *authoritative*
        structure, at execution time) compares its own fresh recomputation
        against exactly this value and raises on mismatch.

        Unlike MDMatEnsemble.build_lists (which accepts any number of
        backend-dependent structure-ish option keys), the recipe/structure
        keys here are fixed to exactly 'rmg_yaml' and 'structure_filename' --
        the driver script needs to unambiguously recover the bare filename
        `pick_structure.pick_best_structure` expects (via
        os.path.basename(task_dct['structure_filename'])), which a
        generic multi-key scan (as MD uses) can't guarantee.
        """
        import yaml
        from pymatgen.core import Structure
        from pyRMG.rmg_input import compute_grid_and_resources

        if 'rmg_yaml' not in self.options or 'structure_filename' not in self.options:
            raise ValueError("DFTMatEnsemble.options must include 'rmg_yaml' and 'structure_filename'.")

        recipe_keys = ['rmg_yaml']
        structure_keys = ['structure_filename']

        labels = check_files + structure_keys + recipe_keys
        task_arg_list, run_paths = self.build_full_runs(
            root0=self.run_directory, files0=[self.options[c] for c in check_files],
            root1=self.inputs_directory, files1=[self.options[k] for k in structure_keys],
            recipe_files=[self.options[k] for k in recipe_keys],
            labels=labels, ordered_labels=labels, finished_file=finished_file,
            run_directory=self.run_directory, inputs_directory=self.inputs_directory,
        )

        rmg_name = self.options.get('rmg_name', 'rmg_input')
        pseudopotentials_directory = self.options.get('pseudopotentials_directory', '')
        gpus_per_node = self.options.get('gpus_per_node', 8)
        electrons_per_gpu = self.options.get('electrons_per_gpu', 10)
        grid_divisibility_exponent = self.options.get('grid_divisibility_exponent', 3)

        dft_dct_list = []
        for task_arg, run_path in zip(task_arg_list, run_paths, strict=True):
            task_dct = dict(zip(labels, task_arg))
            rmg_yaml = task_dct['rmg_yaml']
            structure_path = task_dct['structure_filename']

            structure = Structure.from_file(structure_path)
            with open(rmg_yaml, 'r') as f:
                input_args = yaml.safe_load(f)

            # This class discovers structure_filename via self.options (this
            # method's own caller-supplied value, e.g. from --structure_filename)
            # -- but rmg_dft.py resolves it independently at execution time by
            # reading the *same-named key straight out of this same yaml*,
            # deliberately, so it stays self-sufficient for standalone testing
            # without a Pipeline/chore wired around it. Those are two separate
            # sources of truth for the same concept; if they disagree, every
            # task built here will silently discover files under one name but
            # then fail deep inside a chore ("No usable structure found ...")
            # under the other. Catching the mismatch here, before any chore is
            # submitted, is a lot clearer than that failure mode.
            yaml_structure_filename = input_args.get('structure_filename', 'POSCAR')
            if yaml_structure_filename != self.options['structure_filename']:
                raise ValueError(
                    f"{rmg_yaml} sets structure_filename={yaml_structure_filename!r}, but this run was "
                    f"built with structure_filename={self.options['structure_filename']!r} (e.g. via "
                    f"--structure_filename) -- rmg_dft.py will use the yaml's value at execution time, "
                    f"not this one, so every task built here would fail there. Update the yaml to match, "
                    f"or pass --structure_filename to match the yaml."
                )

            _, allocated_nodes = compute_grid_and_resources(
                structure, input_args, target_nodes=0, gpus_per_node=gpus_per_node,
                electrons_per_gpu=electrons_per_gpu,
                grid_divisibility_exponent=grid_divisibility_exponent,
                pseudopotentials_directory=pseudopotentials_directory,
            )

            dft_dct_list.append({
                **task_dct,
                'working_directory': os.path.normpath(run_path),
                'rmg_name': rmg_name,
                'pseudopotentials_directory': pseudopotentials_directory,
                'gpus_per_node': gpus_per_node,
                'electrons_per_gpu': electrons_per_gpu,
                'grid_divisibility_exponent': grid_divisibility_exponent,
                'allocated_nodes': allocated_nodes,
                'dft_task': dft_task,
                'entry_point': entry_point,
            })

        return dft_dct_list

    @staticmethod
    def run_individual(task_dict):
        """
        Import the user-supplied DFT driver script (`task_dict['dft_task']`)
        by path and dispatch to its entry-point function (named by
        `task_dict['entry_point']`) -- the same dynamic-import-and-dispatch
        mechanism as MDMatEnsemble.run_individual, and the same
        list-of-positional-args calling convention (e.g. entry_point(ffield,
        structure, output) there): here, entry_point(working_directory,
        rmg_yaml), each wrapped in a singleton list since build_dft_dcts
        (unlike MD's batch_by_parent) never batches multiple jobs into one
        task_dict. Everything else the driver needs (pseudopotentials_dir via
        RMG's own 'pseudo_dir' keyword, gpus_per_node/electrons_per_gpu/
        grid_divisibility_exponent/rmg_name/rmg_executable/command/
        structure_filename/allocated_nodes) is read directly out of the
        rmg_yaml file by the driver script itself -- see
        test/RMG_testing/rmg_dft.py.
        """
        module_name = Path(task_dict['dft_task']).stem
        driver = import_module_from_path(module_name, task_dict['dft_task'])
        entry_point = getattr(driver, task_dict['entry_point'])
        return entry_point([task_dict['working_directory']], [task_dict['rmg_yaml']])
