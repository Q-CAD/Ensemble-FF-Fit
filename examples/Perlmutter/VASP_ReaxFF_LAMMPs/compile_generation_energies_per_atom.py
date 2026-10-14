"""
Compiles the relative-energy comparison (predicted vs. DFT, already in
kcal/mol -- see MD/single_points/reaxff_validation/
relative_energy_comparison.py) for the SELECTED force field from EVERY
fitting generation found on disk (generation_0, the unfit seed, through
whichever generation_N is the latest to have a selection_report.txt/
winner -- auto-discovered, not hardcoded, so this keeps working as more
generations are added by future pipeline runs) into one table per subset
(training, validation), adding genuine per-atom energy columns alongside
the existing per-"formula-unit" ones.

Why this is needed (not just a re-read of energy_deviations_ranked_
{training,validation}.csv): predicted_kcal_mol/dft_kcal_mol there are NOT
per atom. They're computed via parse2fit's own get_divisors logic
(ReaxEntry.get_relative_energy, see relative_energy_comparison.py's own
docstring), which picks a per-ENTRY divisor for each structure in the
combination so that the overall combination's composition balances to
zero (a legitimate stoichiometric equation) -- not a fixed, dataset-wide
atom count. E.g. two 135-atom amorphous Bi2Se3 structures (composition
Bi54Se81, divisor=27) end up reported PER Bi2Se3 FORMULA UNIT (5 atoms),
not per the full 135-atom cell; naively dividing by 135 would be wrong
there. Meanwhile an equation-of-state entry comparing one structure
against itself at a different volume (no stoichiometric reduction
needed, divisor=1 on both sides) DOES end up reported per the structure's
own full, unreduced atom count -- so there is no single constant divisor
that works across every row, in either subset; it must be read per-row
from that row's own "expression" column. Validation (AIMD single-image
vs. that same trajectory's own frame 0) follows the identical rule --
confirmed directly against real data: every validation row's add/
subtract target shares the target's own composition, so divisor_target
there is consistently the trajectory supercell's own full-composition
GCD (e.g. /72 for a 360-atom Bi2Se3 AIMD cell -> 5 atoms/unit), not
necessarily 1 -- so it needs the exact same per-row treatment as
training, not a shortcut.

The general rule (derived directly from parse2fit.core.entries.ReaxEntry.
get_relative_energy/_get_divisors_to_write -- confirmed by reading the
source, not assumed): the "expression" column is always built as
"+ <target_relpath>/<divisor_target> <sign> <ref_relpath>/<divisor_ref> ...",
target ALWAYS first (reax_objs = [self] + add + subtract, in that order,
and self's own divisor is never zeroed out -- see _get_divisors_to_write).
So:

    atoms_per_unit = natoms(target POSCAR) / divisor_target

is the number of atoms the row's reported relative energy actually
corresponds to, where divisor_target is parsed off the expression's own
first term. Dividing predicted_kcal_mol/dft_kcal_mol by atoms_per_unit
gives a true per-atom energy, directly comparable across entries/
structures of any supercell size -- the right quantity for convex-hull /
phase-diagram-style plotting (matches pymatgen's own per-atom PDEntry
convention).

Run this from the pipeline's own working directory (wherever
workflow_config.yaml/FF/MD/DFT live -- e.g. run_pipeline/ on Perlmutter,
or this example's own directory once it's been run end-to-end), inside
the matensemble:ff-fit container (needs pandas + pymatgen; no
jax/jaxreaxff import, so CPU-only is fine):

    python3 compile_generation_energies_per_atom.py

Writes compiled_generation_energies_per_atom_training.csv and
compiled_generation_energies_per_atom_validation.csv in the current
directory.
"""
import glob
import os
import re

import pandas as pd
import yaml
from pymatgen.core.structure import Structure

EV_TO_KCAL_MOL = 23.060548867


def _discover_generation_winners():
    """
    (generation label, ff_label) for every generation_N found under FF/ --
    generation_0 (the unfit seed) plus every later generation's own
    select_next_generation_force_field winner. Auto-discovered rather
    than hardcoded: for each FF/generation_N/, the winner is whichever
    IMMEDIATE child directory has its own energy_deviations_ranked_
    training.csv (written by write_generation_diagnostics in run_
    pipeline.py) -- NOT by filtering on directory-name patterns like
    "parse2fit*"/"reaxff_inputs", since those are convention, not
    contract, and a generation_N folder can otherwise contain a lot else
    besides the winner (leftover parse2fit_*/reaxff_inputs sibling dirs,
    occasionally a large nested candidate tree inside the winner's own
    directory from an earlier pipeline layout) -- the marker file is the
    one thing guaranteed to exist only in the winner's own directory.
    Raises if any generation_N doesn't have EXACTLY one such child, so a
    malformed/partial generation is a loud error, not a silent skip or a
    wrong guess.
    """
    gen_dirs = sorted(
        (p for p in glob.glob('FF/generation_*') if os.path.isdir(p)),
        key=lambda p: int(re.search(r'generation_(\d+)$', p).group(1)))
    if not gen_dirs:
        raise RuntimeError("No FF/generation_* directories found -- run the pipeline's "
                            "build_ff_inputs/fit_and_validate/select_next_generation_force_field "
                            "stages first.")

    winners = []
    for gen_dir in gen_dirs:
        generation = os.path.basename(gen_dir)
        matches = glob.glob(os.path.join(gen_dir, '*', 'energy_deviations_ranked_training.csv'))
        if len(matches) != 1:
            raise RuntimeError(
                f"{gen_dir}: expected exactly one winner subdirectory containing "
                f"energy_deviations_ranked_training.csv, found {len(matches)}: {matches!r}")
        ff_label = os.path.basename(os.path.dirname(matches[0]))
        winners.append((generation, ff_label))
    return winners


def _load_reference_roots():
    """
    {'training': ..., 'validation': ...} DFT reference roots, read
    straight from workflow_config.yaml's own rank_reaxff_validation
    section (training_reference_root/validation_reference_root) rather
    than hardcoded -- these are the SAME roots every "target"/"structure"
    path in energy_deviations_ranked_*.csv is relative to, since that's
    exactly what parse_reference_tree(cfg['...reference_root'], ...) was
    pointed at when rank_reaxff_validation originally produced them.
    """
    with open('workflow_config.yaml') as f:
        config = yaml.safe_load(f)
    cfg = config['rank_reaxff_validation']
    return {'training': cfg['training_reference_root'], 'validation': cfg['validation_reference_root']}


# First term of "expression" is always the target itself (sign always
# '+', divisor never zeroed) -- see entries.py's _get_divisors_to_write:
# reax_objs = [self] + add + subtract, self appended first with
# all_divisors.append(self._composition_divisors(...)) (no zero option).
_FIRST_TERM_RE = re.compile(r'^\+\s+\S+/(\d+)')

# First term of "expression" is always the target itself (sign always
# '+', divisor never zeroed) -- see entries.py's _get_divisors_to_write:
# reax_objs = [self] + add + subtract, self appended first with
# all_divisors.append(self._composition_divisors(...)) (no zero option).
_FIRST_TERM_RE = re.compile(r'^\+\s+\S+/(\d+)')


def _target_divisor(expression):
    m = _FIRST_TERM_RE.match(expression.strip())
    if not m:
        raise ValueError(f"Couldn't parse target divisor from expression: {expression!r}")
    return int(m.group(1))


def compile_subset(subset, dft_root, generations):
    natoms_cache = {}

    def natoms(target_relpath):
        if target_relpath not in natoms_cache:
            poscar_path = os.path.join(dft_root, target_relpath, 'POSCAR')
            natoms_cache[target_relpath] = len(Structure.from_file(poscar_path))
        return natoms_cache[target_relpath]

    frames = []
    for generation, ff_label in generations:
        csv_path = f'FF/{generation}/{ff_label}/energy_deviations_ranked_{subset}.csv'
        df = pd.read_csv(csv_path)
        df['generation'] = generation
        df['ff_label'] = ff_label

        divisors = df['expression'].map(_target_divisor)
        atoms = df['target'].map(natoms)
        atoms_per_unit = atoms / divisors

        df['target_divisor'] = divisors
        df['target_natoms'] = atoms
        df['atoms_per_unit'] = atoms_per_unit

        df['predicted_kcal_mol_per_atom'] = df['predicted_kcal_mol'] / atoms_per_unit
        df['dft_kcal_mol_per_atom'] = df['dft_kcal_mol'] / atoms_per_unit
        df['abs_deviation_kcal_mol_per_atom'] = (df['predicted_kcal_mol_per_atom']
                                                  - df['dft_kcal_mol_per_atom']).abs()

        df['predicted_eV_per_atom'] = df['predicted_kcal_mol_per_atom'] / EV_TO_KCAL_MOL
        df['dft_eV_per_atom'] = df['dft_kcal_mol_per_atom'] / EV_TO_KCAL_MOL

        frames.append(df)

    compiled = pd.concat(frames, ignore_index=True)
    cols = ['generation', 'ff_label', 'entry_name', 'target', 'expression',
            'target_divisor', 'target_natoms', 'atoms_per_unit',
            'predicted_kcal_mol', 'dft_kcal_mol', 'abs_deviation_kcal_mol',
            'predicted_kcal_mol_per_atom', 'dft_kcal_mol_per_atom', 'abs_deviation_kcal_mol_per_atom',
            'predicted_eV_per_atom', 'dft_eV_per_atom']
    return compiled[cols]


def _report_atoms_per_unit_spread(compiled):
    # Sanity spot-check: how many distinct atoms_per_unit values show up,
    # and a couple of examples -- confirms this ISN'T a single constant
    # (the whole point of doing this per-row rather than dividing by one
    # fixed atom count).
    seen = {}
    for _, row in compiled.iterrows():
        seen.setdefault(row['atoms_per_unit'], row['entry_name'])
    for apu, example in sorted(seen.items()):
        print(f"  {apu:g} atoms/unit  (e.g. entry_name={example!r})")


if __name__ == '__main__':
    generations = _discover_generation_winners()
    print(f"Discovered {len(generations)} generation(s): "
          + ", ".join(f"{g}={label}" for g, label in generations))
    print()

    for subset, dft_root in _load_reference_roots().items():
        compiled = compile_subset(subset, dft_root, generations)
        out_path = f'compiled_generation_energies_per_atom_{subset}.csv'
        compiled.to_csv(out_path, index=False)
        print(f"Wrote {len(compiled)} row(s) across {compiled['generation'].nunique()} generation(s) to {out_path}")
        print(f"Distinct atoms_per_unit values found in {subset} (entry_name -> example):")
        _report_atoms_per_unit_spread(compiled)
        print()
