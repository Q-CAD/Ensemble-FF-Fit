"""
Compares each fitted ReaxFF potential's LAMMPS single-point energies
against DFT ground truth using the exact same relative-energy combination
logic (subtract/add/get_divisors) that parse2fit's own
reaxff_newest_kT.yml/reaxff_validation.yml already define for ReaxFF
training -- see Paper_Analysis.md for the full problem this solves.
ReaxFF is fit to relative energies only, so comparing absolute DFT vs.
ReaxFF energies is meaningless; this reuses parse2fit's own combination
math (parse2fit.core.entries.ReaxEntry.get_relative_energy) against both
data sources, rather than re-deriving a separate, possibly-inconsistent
scheme.

Does NOT use parse2fit.io.parsers/ParserFactory (the vasprun.xml-specific
layer) -- both DFT and LAMMPS structures here already have a precomputed
properties.json (confirmed under DFT/training/ and DFT/validation/aimd/,
and under every MD/single_points/reaxff_validation/{training,aimd}/
force_fields/<run>/structures/ leaf) in the same {"energy","fx","fy","fz"}
shape, so no DFT-code-specific parsing is needed at all -- just a POSCAR
(for composition/site_counts, used by get_divisors) and properties.json's
"energy" key (DFT: eV; LAMMPS "real" units: kcal/mol -- ReaxEntry's own
_correct_units() converts the DFT side automatically, same mechanism
parse2fit's own VaspParser path already relies on).
"""
import json
import os
from multiprocessing import Pool, cpu_count

import yaml
from pymatgen.core.structure import Structure
from tqdm import tqdm

from parse2fit.core.entries import ReaxEntry
from parse2fit.core.properties import Energy

DFT_UNIT = 'eV'
LAMMPS_UNIT = 'kcal/mol'


def _is_parseable(directory):
    return (os.path.exists(os.path.join(directory, 'POSCAR'))
            and os.path.exists(os.path.join(directory, 'properties.json')))


def _walk_parseable(root):
    """
    Every directory under `root` (itself included) containing a POSCAR +
    properties.json -- mirrors parse2fit.io.readwrite.RW._get_all_paths's
    walk_roots=True behavior for a YAML entry's "directories" list (walk
    everything, keep whatever actually parses).
    """
    found = []
    for dirpath, _, _ in os.walk(os.path.normpath(root), topdown=True):
        if _is_parseable(dirpath):
            found.append(dirpath)
    return found


def _load_structure_and_energy(directory, unit):
    structure = Structure.from_file(os.path.join(directory, 'POSCAR'))
    with open(os.path.join(directory, 'properties.json')) as f:
        energy_value = json.load(f)['energy']
    return structure, Energy(value=energy_value, unit=unit)


def _make_reax_entry(label, structure, energy):
    """
    A direct ReaxEntry(structure=..., energy=...) used to crash:
    ReaxEntry.__init__ called self._correct_units() (which reads
    self.units_dct) BEFORE self.units_dct was assigned a few lines later
    in that same __init__ -- a real bug in parse2fit itself (confirmed:
    masked in parse2fit's own usage, which always constructs an empty
    ReaxEntry() first and populates it via .from_dict() afterward, so
    self.units_dct already exists by the time _correct_units() runs a
    second time inside from_dict). FIXED upstream now (parse2fit's own
    develop branch, commit efc7786) -- this still goes through
    ReaxEntry(label=...).from_dict(...) rather than the direct
    constructor anyway, since it costs nothing and stays correct even
    against an older pinned parse2fit build that predates that fix.
    """
    entry = ReaxEntry(label=label)
    entry.from_dict({'structure': structure, 'energy': energy})
    return entry


def _compute_run_relative_energies(task):
    """
    Worker: every relative-energy comparison for ONE force-field run,
    independent of every other run -- same reasoning as cn_checker.py's
    own per-force-field multiprocessing split (compute_cn_diff): the
    actual bottleneck here is ReaxEntry construction (_make_reax_entry,
    O(n_structures x n_runs) total calls), and a given target structure's
    divisors/signs/stoichiometric combination are already resolved from
    the DFT side (passed in, not recomputed here) -- so each run only
    needs to read ITS OWN properties.json energies and construct ITS OWN
    ReaxEntry objects, no shared mutable state, no cross-run dependency.
    _compute_total_energy is called on whichever ReaxEntry happens to be
    first in reax_objs purely to invoke the (unbound, self-independent --
    confirmed directly against parse2fit.core.entries.ReaxEntry's own
    source, not assumed) method -- same thing the original sequential code
    did by calling it on target_dft instead.

    Structures are re-parsed from each path's own POSCAR here (one
    re-parse per unique path per worker, not per (structure, run) pair)
    rather than pickled in from the parent process -- avoids shipping
    pymatgen Structure objects across the process boundary for every one
    of potentially hundreds of runs; a POSCAR re-read is cheap next to
    constructing+comparing ReaxEntry objects for every run.

    task: (run, force_field_root, dft_root, items), items a list of (key,
    target_path, add_paths, subtract_paths, signs, divisors) tuples, key
    an opaque (entry_name, target_relpath) pair used only to route each
    result back to the right slot in RelativeEnergyComparison.compute's
    own results dict.

    Returns (run, {key: total_energy}).
    """
    run, force_field_root, dft_root, items = task
    structure_cache = {}
    entry_cache = {}

    def ff_entry(path):
        path = os.path.normpath(path)
        if path not in entry_cache:
            if path not in structure_cache:
                structure_cache[path] = Structure.from_file(os.path.join(path, 'POSCAR'))
            relpath = os.path.relpath(path, dft_root)
            ff_path = os.path.join(force_field_root, relpath)
            with open(os.path.join(ff_path, 'properties.json')) as f:
                energy_value = json.load(f)['energy']
            entry_cache[path] = _make_reax_entry(path, structure_cache[path],
                                                  Energy(value=energy_value, unit=LAMMPS_UNIT))
        return entry_cache[path]

    results = {}
    for key, target_path, add_paths, subtract_paths, signs, divisors in items:
        ff_reax_objs = ([ff_entry(target_path)]
                        + [ff_entry(p) for p in add_paths]
                        + [ff_entry(p) for p in subtract_paths])
        results[key] = ff_reax_objs[0]._compute_total_energy(ff_reax_objs, divisors, signs)
    return run, results


class RelativeEnergyComparison:
    """
    Builds ground-truth (DFT) and per-force-field (LAMMPS) ReaxEntry
    objects for every structure referenced by a parse2fit-format YAML
    (reaxff_newest_kT.yml / reaxff_validation.yml), then computes each
    named entry's relative energy the same way parse2fit itself does when
    building ReaxFF's own training set (see
    parse2fit.io.readwrite.ReaxRW._get_property_weights's own 'energy'
    branch, which this mirrors).

    get_divisors is solved ONCE per entry, against the DFT-side ReaxEntry
    objects only, then the resulting divisor/sign combination is reused
    for every force-field run -- the divisor search
    (ReaxEntry._get_divisors_to_write) is purely a function of structure
    composition (site_counts), not energy value, so solving it separately
    per run would be redundant AND risks different runs landing on
    different (each individually valid) divisor combinations, making the
    comparison apples-to-oranges across the ensemble.
    """

    def __init__(self, yaml_path, dft_root, force_field_roots):
        """
        dft_root: DFT/training or DFT/validation -- matches
        workflow_config.yaml's rank_reaxff_validation.{training,validation}
        _reference_root convention; the YAML's own absolute paths are
        expected to live under this root.

        force_field_roots: {run_label: path to that run's own mirrored
        "structures" root}, e.g. {'reaxff_run_0_angular_only':
        '.../training/force_fields/reaxff_run_0_angular_only/structures'}.
        """
        with open(yaml_path) as f:
            self.config = yaml.safe_load(f)
        self.dft_root = os.path.normpath(dft_root)
        self.force_field_roots = force_field_roots

        self._dft_entry_cache = {}
        self._ff_entry_cache = {run: {} for run in force_field_roots}

    def _dft_entry(self, dft_path):
        dft_path = os.path.normpath(dft_path)
        if dft_path not in self._dft_entry_cache:
            structure, energy = _load_structure_and_energy(dft_path, DFT_UNIT)
            self._dft_entry_cache[dft_path] = _make_reax_entry(dft_path, structure, energy)
        return self._dft_entry_cache[dft_path]

    def _ff_entry(self, run, dft_path):
        dft_path = os.path.normpath(dft_path)
        cache = self._ff_entry_cache[run]
        if dft_path not in cache:
            relpath = os.path.relpath(dft_path, self.dft_root)
            ff_path = os.path.join(self.force_field_roots[run], relpath)
            # Same atoms as the DFT side -- only the energy source differs,
            # so reuse the already-parsed structure rather than requiring a
            # structure file under the LAMMPS output leaf (it doesn't have
            # one; see this module's own docstring).
            structure = self._dft_entry(dft_path).structure
            with open(os.path.join(ff_path, 'properties.json')) as f:
                energy_value = json.load(f)['energy']
            energy = Energy(value=energy_value, unit=LAMMPS_UNIT)
            cache[dft_path] = _make_reax_entry(dft_path, structure, energy)
        return cache[dft_path]

    def _resolve_group_paths(self, group_dct):
        """
        (target_paths, add_paths, subtract_paths) for one input_paths
        entry. directories are walked (one target per parseable
        subdirectory discovered); add/subtract are literal single paths,
        never walked -- matches
        parse2fit.io.readwrite.RW._path_root_dictionary exactly.
        """
        target_paths = []
        for directory in group_dct.get('directories', []):
            target_paths.extend(_walk_parseable(directory))

        energy_dct = group_dct.get('energy', {})
        add_paths = [os.path.normpath(p) for p in energy_dct.get('add', [])] if isinstance(energy_dct, dict) else []
        subtract_paths = [os.path.normpath(p) for p in energy_dct.get('subtract', [])] if isinstance(energy_dct, dict) else []
        return target_paths, add_paths, subtract_paths

    def compute(self, num_processes=None):
        """
        Returns {entry_name: {target_relpath: {'expression': str, 'values':
        {'DFT': value, run_label: value, ...}}}} -- one relative-energy
        value per target structure discovered under that entry's
        directories, per data source, all in kcal/mol (ReaxEntry's own
        native unit -- see _make_reax_entry's docstring; _correct_units()
        normalizes both the DFT (eV-sourced) and LAMMPS (already kcal/mol)
        sides to this same unit, so these values are directly comparable
        as-is). target_relpath is relative to dft_root, matching the
        LAMMPS-side mirroring convention, so results can be joined against
        EnsembleFFFit.analysis.dict_parsers's own {label: {run: {image:
        props}}} shape downstream. 'expression' is the human-readable
        combination (structure paths, divisors, +/- signs) used for every
        source -- same ingredients as parse2fit's own trainset.in ENERGY
        section (see ReaxEntry._relative_energy_substring), just full
        precision and without a training weight, for inspection/plotting
        rather than fitting.

        Entries with neither 'add' nor 'subtract' set are skipped
        entirely -- reference-only entries that exist to be pointed at by
        OTHER entries' add/subtract lists, never written as their own
        relative-energy line (matches
        parse2fit.io.readwrite.ReaxRW._get_property_weights's own
        `if add_objects or subtract_objects` check).

        num_processes (optional, default os.cpu_count()): the per-run
        ReaxEntry construction + total-energy computation (the actual
        bottleneck -- one ReaxEntry per (target structure, force-field
        run), O(n_structures x n_runs), previously fully sequential) is
        parallelized ACROSS RUNS via multiprocessing.Pool (see
        _compute_run_relative_energies) -- one task per force-field run
        (typically far more runs than cores, so every core stays busy),
        not one task per (structure, run) pair (which, at hundreds of
        structures x hundreds of runs, would be far too fine-grained --
        dispatch overhead would dominate). The DFT-side divisor/sign
        resolution below stays sequential either way -- cheap, and each
        target's divisors must be resolved before any run's work for that
        target can even be defined.
        """
        results = {}
        tasks_by_run = {run: [] for run in self.force_field_roots}

        for name, group_dct in self.config.get('input_paths', {}).items():
            if not isinstance(group_dct, dict):
                continue
            energy_dct = group_dct.get('energy', {})
            if not isinstance(energy_dct, dict):
                continue

            target_paths, add_paths, subtract_paths = self._resolve_group_paths(group_dct)
            if not add_paths and not subtract_paths:
                continue

            get_divisors = energy_dct.get('get_divisors', False)
            add_dft = [self._dft_entry(p) for p in add_paths]
            subtract_dft = [self._dft_entry(p) for p in subtract_paths]

            entry_results = {}
            for target_path in tqdm(target_paths, desc=name, unit="structure", leave=False):
                target_path = os.path.normpath(target_path)
                if target_path in add_paths + subtract_paths:
                    continue  # self-referential walk hit; see docstring above

                target_dft = self._dft_entry(target_path)
                _, signs, divisors, dft_energy = target_dft.get_relative_energy(
                    add=add_dft, subtract=subtract_dft, get_divisors=get_divisors)
                if dft_energy.value is None:
                    continue  # no valid divisor combination found for this target

                target_relpath = os.path.relpath(target_path, self.dft_root)
                per_source = {'DFT': dft_energy.value}

                # Same reax_objs/signs/divisors ordering DFT-side
                # get_relative_energy() just resolved -- the expression is
                # identical for every data source (same structures, same
                # combination), only the per-object energy value differs,
                # so it's built once here rather than per run below.
                dft_reax_objs = [target_dft] + add_dft + subtract_dft
                expression = ' '.join(
                    f"{sign} {os.path.relpath(obj.label, self.dft_root)}/{divisor}"
                    for obj, sign, divisor in zip(dft_reax_objs, signs, divisors)
                    if divisor != 0
                )

                entry_results[target_relpath] = {'expression': expression, 'values': per_source}

                key = (name, target_relpath)
                for run in self.force_field_roots:
                    tasks_by_run[run].append((key, target_path, add_paths, subtract_paths, signs, divisors))

            if entry_results:
                results[name] = entry_results

        pool_tasks = [(run, self.force_field_roots[run], self.dft_root, items)
                      for run, items in tasks_by_run.items() if items]
        if pool_tasks:
            n_procs = min(num_processes or cpu_count(), len(pool_tasks))
            with Pool(processes=n_procs) as pool:
                for run, run_results in tqdm(pool.imap_unordered(_compute_run_relative_energies, pool_tasks),
                                              total=len(pool_tasks), desc="force-field runs", unit="run"):
                    for (entry_name, target_relpath), value in run_results.items():
                        results[entry_name][target_relpath]['values'][run] = value

        return results


def write_relative_energy_details(per_entry_results, ff_dir, run_labels, filename="relative_energy_comparison.csv"):
    """
    Writes one file per force-field run, into ff_dir/<run>/filename (that
    run's own directory, alongside its ffield/structures/ -- e.g.
    MD/single_points/reaxff_validation/training/force_fields/
    reaxff_run_0_angular_only/relative_energy_comparison.csv), listing
    every (entry_name, target_relpath) relative-energy comparison for
    that run: the full combination expression (see RelativeEnergyComparison.
    compute's own docstring), that run's predicted relative energy, the
    DFT ground truth for the same expression, and their absolute
    difference -- all in kcal/mol. Full float precision, not rounded to
    ReaxFF's own 3-sig-fig trainset.in convention -- this is for
    inspection/plotting, not for feeding back into a fit.
    """
    rows_by_run = {run: [] for run in run_labels}
    for entry_name, target_dct in tqdm(per_entry_results.items(), desc="writing relative energy details", unit="entry"):
        for target_relpath, detail in target_dct.items():
            dft_value = detail['values']['DFT']
            for run in run_labels:
                ff_value = detail['values'][run]
                deviation = abs(ff_value - dft_value)
                rows_by_run[run].append(
                    f"{entry_name},{target_relpath},{detail['expression']},"
                    f"{ff_value:.6f},{dft_value:.6f},{deviation:.6f}"
                )

    header = "entry_name,target,expression,predicted_kcal_mol,dft_kcal_mol,abs_deviation_kcal_mol"
    for run, rows in rows_by_run.items():
        out_path = os.path.join(ff_dir, run, filename)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'w') as f:
            f.write(header + "\n")
            f.write("\n".join(rows) + ("\n" if rows else ""))


def build_relative_energy_dcts(training_yaml, training_reference_root, training_ff_dir,
                                validation_yaml, validation_reference_root, validation_ff_dir,
                                num_processes=None):
    """
    Runs RelativeEnergyComparison for both the training (reaxff_newest_kT.yml)
    and validation (reaxff_validation.yml) YAMLs, writes per-run detail
    files via write_relative_energy_details (see its own docstring), and
    reshapes each result into {label: {entry_name: {target_relpath:
    {"relative_energy": value}}}} -- one dict per label ("DFT" plus every
    force-field run discovered under the corresponding *_ff_dir) -- so the
    result slots into EnsembleFFFit.analysis.best_force_field's existing
    {label: {run: {image: props}}} convention, treating entry_name as
    "run" and target_relpath as "image".

    num_processes: passed straight through to RelativeEnergyComparison.compute
    (see its own docstring) -- default os.cpu_count().

    Returns (training_relative_energy_dct, validation_relative_energy_dct).
    """
    def reshape(per_entry_results, run_labels):
        out = {label: {} for label in ['DFT'] + run_labels}
        for entry_name, target_dct in per_entry_results.items():
            for target_relpath, detail in target_dct.items():
                for label, value in detail['values'].items():
                    out[label].setdefault(entry_name, {})[target_relpath] = {'relative_energy': value}
        return out

    def discover_runs(ff_dir):
        return sorted(d for d in os.listdir(ff_dir) if os.path.isdir(os.path.join(ff_dir, d)))

    training_runs = discover_runs(training_ff_dir)
    training_comparison = RelativeEnergyComparison(
        training_yaml, training_reference_root,
        {run: os.path.join(training_ff_dir, run, 'structures') for run in training_runs})
    training_results = training_comparison.compute(num_processes=num_processes)
    write_relative_energy_details(training_results, training_ff_dir, training_runs)
    training_relative_energy_dct = reshape(training_results, training_runs)

    validation_runs = discover_runs(validation_ff_dir)
    validation_comparison = RelativeEnergyComparison(
        validation_yaml, validation_reference_root,
        {run: os.path.join(validation_ff_dir, run, 'structures') for run in validation_runs})
    validation_results = validation_comparison.compute(num_processes=num_processes)
    write_relative_energy_details(validation_results, validation_ff_dir, validation_runs)
    validation_relative_energy_dct = reshape(validation_results, validation_runs)

    return training_relative_energy_dct, validation_relative_energy_dct


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-yaml', default='DFT/reaxff_newest_kT.yml')
    parser.add_argument('--training-reference-root', default='DFT/training')
    parser.add_argument('--training-ff-dir', default='MD/single_points/reaxff_validation/training/force_fields')
    parser.add_argument('--validation-yaml', default='DFT/reaxff_validation.yml')
    parser.add_argument('--validation-reference-root', default='DFT/validation')
    parser.add_argument('--validation-ff-dir', default='MD/single_points/reaxff_validation/aimd/force_fields')
    args = parser.parse_args()

    training_dct, validation_dct = build_relative_energy_dcts(
        args.training_yaml, args.training_reference_root, args.training_ff_dir,
        args.validation_yaml, args.validation_reference_root, args.validation_ff_dir)

    for set_name, dct in [('Training', training_dct), ('Validation', validation_dct)]:
        print(f"\n=== {set_name} ===")
        dft_entries = dct.get('DFT', {})
        n_entries = sum(len(targets) for targets in dft_entries.values())
        print(f"{len(dft_entries)} named entries, {n_entries} target structures, "
              f"{len(dct) - 1} force field run(s) parsed")
        for entry_name, targets in dft_entries.items():
            for target_relpath, props in targets.items():
                print(f"  {entry_name} / {target_relpath}: DFT={props['relative_energy']:.6f} kcal/mol")
