"""
Build a pyace-compatible training dataset (.pckl.gzip) from completed QE
single-point runs (structure_filename + properties.json pairs, the same
output shape rmg_dft.py/vasp_dft.py/qe_dft.py all produce).

Deliberately ACE-specific (pandas DataFrame with ase_atoms/energy/forces/
energy_corrected columns, pyace's own documented training-data schema --
see pipeline/FRICTION_LOG.md for the research trail confirming it), not
conserved DFT-agnostic logic -- other force-field backends have their own
conventions (see potential/mace/write_training_xyz.py for the MACE/extxyz
equivalent, which this module's own leaf-walking loop mirrors closely).

pyace ships its own `pace_collect` CLI for auto-converting VASP output, but
that's VASP-OUTCAR-specific -- nothing accepts pre-parsed properties.json
the way this project's own DFT drivers already produce it, hence this
module rather than shelling out to pace_collect.
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from ase.calculators.singlepoint import SinglePointCalculator
from ase.stress import voigt_6_to_full_3x3_stress
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor


def build_ace_dataframe(run_directory, e0s, structure_filename="POSCAR",
                         properties_filename="properties.json",
                         max_abs_energy_per_atom=50.0, max_abs_force_component=50.0,
                         stability_floor_per_atom=None, stability_floor_margin=0.5,
                         include_path=False):
    """
    Walk `run_directory` (a single path, OR -- ADDED 2026-10-12 -- a list/
    tuple of paths, e.g. [.../training, .../training_2], each walked and
    pooled into one combined dataframe; meant for adding a second, separate
    training-structure tier -- see downselect_low_energy.py's own module
    docstring -- without merging its files into the original tree) for
    every leaf containing both structure_filename and properties_filename,
    building one pyace-schema row per leaf:
    ase_atoms (ASE Atoms, with a SinglePointCalculator attached for
    convenience -- not required by pyace's own schema, which reads energy/
    forces from the separate columns below, but harmless and matches how
    ASE round-trips real calculation output), energy (eV), forces
    ([N,3] eV/Angstrom), energy_corrected (energy minus the sum of each
    atom's isolated-atom reference energy from `e0s`).

    `e0s` is {atomic_number: energy} -- the same shape and the same
    potential.mace.build_ensemble_inputs.get_isolated_atom_e0s function
    already produces it with (that function is generic over DFT output
    shape, not MACE-specific, despite living in the mace/ subpackage --
    reused here rather than duplicated).

    Any structure containing an element missing from e0s is skipped (with a
    printed warning), not silently included with a wrong/missing reference
    -- energy_corrected would be meaningless without a real E0 for every
    element present.

    max_abs_energy_per_atom / max_abs_force_component (None disables either
    check): a generic sanity filter against physically-impossible outliers
    -- CONFIRMED (2026-10-07) a real, not hypothetical, failure mode: a
    structure with a single pair of atoms 0.12 Angstrom apart (two
    different species essentially coincident) produced an energy of
    -2.2 MILLION eV and forces of 727,000 eV/Angstrom from an otherwise-
    healthy foundation model, evaluated far outside its training
    distribution. Root cause traced to an upstream MD safety check
    (max_energy_deviation_per_atom in a finite-temperature MD driver)
    correctly detecting and aborting an unstable trajectory, but the
    specific frame that triggered the abort still being written to that
    trajectory and carried through unchanged by downstream frame-extraction
    -- this filter is a generic, backend-of-origin-agnostic safety net
    against that (and any other future source of similarly-corrupted
    training data), not a fix to that specific upstream gap. Defaults
    (50 eV/atom, 50 eV/Angstrom) are deliberately generous -- well beyond
    even an unusually high-energy but genuinely physical configuration
    (typical cohesive energies for solids are a few eV/atom at most) --
    meant to catch only true, many-orders-of-magnitude pathologies like
    the one above, not to second-guess legitimately unusual but real data.

    stability_floor_per_atom / stability_floor_margin: ADDED (2026-10-07) a
    second, complementary sanity check -- a structure can look numerically
    sane (no blow-up, passes the check above) while still being quietly
    wrong: a foundation model spuriously over-stabilizing a structure it's
    extrapolating on, rather than obviously failing on it. Physical
    reasoning (standard practice in computational structure search, e.g.
    convex-hull sanity-checking): a random/synthetic structure isn't
    expected to be MORE stable (more negative formation energy per atom)
    than the best real, known phase for this system -- if one is, that's
    far more likely a prediction artifact than a genuine new discovery.
    stability_floor_per_atom is the reference value to compare against
    (typically the minimum formation energy/atom found across a SEPARATE,
    real/known-phase validation set -- see compute_validation_floor below,
    not computed from this same run_directory, which would be circular);
    any row whose own energy_corrected/n_atoms falls below
    (stability_floor_per_atom - stability_floor_margin) is excluded. Margin
    default (0.5 eV/atom) absorbs ordinary prediction noise around the true
    floor rather than cutting exactly at it. Both args None/unset disables
    this check entirely (e.g. when building the validation set itself,
    which defines the floor rather than being checked against it).

    include_path (ADDED 2026-10-08, default False): when True, an extra
    'path' column (str(root), the leaf directory each row's structure/
    properties files were read from) is included. Off by default so
    pyace's own data-loading code -- which reads whatever columns the
    saved dataset actually has -- never sees a column it wasn't written
    for from any EXISTING caller (write_ace_dataset doesn't request this);
    meant for callers that need to trace a row back to its source
    structure (e.g. potential.ace.downselect_low_energy, which groups rows
    by composition/original-MD-trajectory before ranking them).

    Returns a pandas DataFrame with columns [ase_atoms, energy, forces,
    energy_corrected] (plus 'path' if include_path=True), in walk order.
    df.attrs['filter_counts'] is a dict ({'considered', 'missing_e0',
    'blowup_outliers', 'below_stability_floor', 'kept'}) -- stashed as
    DataFrame metadata (not a second return value) so this stays a drop-in
    replacement for every existing caller that just does
    `df = build_ace_dataframe(...)`.
    """
    run_directories = [run_directory] if isinstance(run_directory, (str, Path)) else list(run_directory)

    rows = []
    skipped = []
    outliers = []
    floor_outliers = []
    n_considered = 0
    for run_directory_single in run_directories:
        for root, _, files in os.walk(run_directory_single):
            root = Path(root)
            struct_path = root / structure_filename
            props_path = root / properties_filename
            if not (struct_path.exists() and props_path.exists()):
                continue

            n_considered += 1
            with open(props_path) as f:
                props = json.load(f)

            structure = Structure.from_file(struct_path)
            atoms = AseAtomsAdaptor.get_atoms(structure)

            atomic_numbers = atoms.get_atomic_numbers()
            missing = sorted(set(int(z) for z in atomic_numbers if int(z) not in e0s))
            if missing:
                from pymatgen.core import Element
                missing_symbols = [Element.from_Z(z).symbol for z in missing]
                skipped.append((str(root), missing_symbols))
                continue

            reference_energy = sum(e0s[int(z)] for z in atomic_numbers)
            energy_corrected = props['energy'] - reference_energy

            forces = np.column_stack([props['fx'], props['fy'], props['fz']])

            epa = energy_corrected / len(atoms)
            max_f = float(np.abs(forces).max()) if forces.size else 0.0
            if max_abs_energy_per_atom is not None and abs(epa) > max_abs_energy_per_atom:
                outliers.append((str(root), f"energy_corrected/atom={epa:.3g} eV/at"))
                continue
            if max_abs_force_component is not None and max_f > max_abs_force_component:
                outliers.append((str(root), f"max force component={max_f:.3g} eV/Angstrom"))
                continue
            if stability_floor_per_atom is not None and epa < (stability_floor_per_atom - stability_floor_margin):
                floor_outliers.append((str(root), f"energy_corrected/atom={epa:.3g} eV/at "
                                                   f"(floor={stability_floor_per_atom:.3g}, "
                                                   f"margin={stability_floor_margin:.3g})"))
                continue

            stress = (
                voigt_6_to_full_3x3_stress(np.array(props['stress_voigt']))
                if 'stress_voigt' in props else None
            )
            atoms.calc = SinglePointCalculator(
                atoms, energy=props['energy'], forces=forces, stress=stress,
            )

            row = {
                'ase_atoms': atoms,
                'energy': props['energy'],
                'forces': forces,
                'energy_corrected': energy_corrected,
            }
            if include_path:
                row['path'] = str(root)
            rows.append(row)

    run_directories_label = run_directories[0] if len(run_directories) == 1 else run_directories
    if skipped:
        print(f"WARNING: skipped {len(skipped)} run(s) under {run_directories_label} with no isolated-atom "
              f"reference energy available:")
        for path, symbols in skipped:
            print(f"  {path}: missing E0 for {symbols}")

    if outliers:
        print(f"WARNING: excluded {len(outliers)} run(s) under {run_directories_label} as physically-impossible "
              f"outliers (energy/atom or force component beyond the configured sanity bound):")
        for path, reason in outliers[:20]:
            print(f"  {path}: {reason}")
        if len(outliers) > 20:
            print(f"  ... and {len(outliers) - 20} more")

    if floor_outliers:
        print(f"WARNING: excluded {len(floor_outliers)} run(s) under {run_directories_label} as more stable than "
              f"the configured stability floor (likely a quiet foundation-model extrapolation error, not "
              f"a genuine new phase -- see build_ace_dataframe's own docstring):")
        for path, reason in floor_outliers[:20]:
            print(f"  {path}: {reason}")
        if len(floor_outliers) > 20:
            print(f"  ... and {len(floor_outliers) - 20} more")

    columns = ['ase_atoms', 'energy', 'forces', 'energy_corrected']
    if include_path:
        columns.append('path')
    df = pd.DataFrame(rows, columns=columns)
    df.attrs['filter_counts'] = {
        'considered': n_considered,
        'missing_e0': len(skipped),
        'blowup_outliers': len(outliers),
        'below_stability_floor': len(floor_outliers),
        'kept': len(rows),
    }
    return df


def compute_validation_floor(validation_directory, e0s, structure_filename="POSCAR",
                              properties_filename="properties.json",
                              max_abs_energy_per_atom=50.0, max_abs_force_component=50.0):
    """
    Build the validation DataFrame (build_ace_dataframe, same blow-up
    filter applied -- an unfiltered validation outlier would corrupt the
    floor itself) and return the minimum energy_corrected/atom found
    anywhere in it: the most stable known real phase in the validation set,
    used as the stability_floor_per_atom argument to build_ace_dataframe's
    own training-side filter. Deliberately a bare float, not the DataFrame
    itself -- the caller only ever needs this one number from the
    validation side; use build_ace_dataframe directly if the validation
    DataFrame itself is also needed (e.g. its own filter_counts).

    Raises RuntimeError if the validation set is empty after filtering --
    there is no sensible floor without at least one real reference phase.
    """
    df = build_ace_dataframe(validation_directory, e0s, structure_filename=structure_filename,
                              properties_filename=properties_filename,
                              max_abs_energy_per_atom=max_abs_energy_per_atom,
                              max_abs_force_component=max_abs_force_component)
    if df.empty:
        raise RuntimeError(
            f"No usable validation structures found under {validation_directory} -- can't compute a "
            f"stability floor without at least one known real phase to compare against."
        )
    nat = df['ase_atoms'].apply(len)
    epa = df['energy_corrected'] / nat
    return float(epa.min())


def write_ace_dataset(run_directory, isolated_elements_directory, output_path,
                       structure_filename="POSCAR", properties_filename="properties.json",
                       max_abs_energy_per_atom=50.0, max_abs_force_component=50.0,
                       validation_directory=None, stability_floor_margin=0.5,
                       validation_output_path=None):
    """
    Convenience wrapper: resolve isolated-atom E0s from
    isolated_elements_directory, build the training DataFrame from
    run_directory (a single path, or -- passed straight through to
    build_ace_dataframe unchanged -- a list/tuple of paths, e.g. a second
    training tier like .../training_2 added alongside .../training; see
    build_ace_dataframe's own docstring for the max_abs_energy_per_atom/
    max_abs_force_component sanity filter, and for the stability-floor
    filter enabled by passing validation_directory here), and save it as a
    gzip-compressed pickle at output_path -- pyace's own documented on-disk
    format (df.to_pickle(path, compression='gzip', protocol=4)).

    validation_directory=None (default) disables the stability-floor check
    entirely, matching this function's pre-existing behavior for callers
    that don't pass it (e.g. QE_ACE/Frontier, unaffected by this addition).

    validation_output_path: ADDED (2026-10-08) -- when given (and
    validation_directory is also given), the validation DataFrame (blow-up-
    filtered the same way the training set is, NOT stability-floor-filtered
    against itself -- that filter only ever applies to the training side)
    is ALSO saved as its own gzip-compressed pickle at this path, in the
    exact same pyace on-disk schema as the training dataset. This is meant
    to be passed as `data.test_filename` to every ensemble member's own
    input.yaml (see build_ace_ensemble_inputs' own test_filename param) so
    pyace/pacemaker computes genuine held-out validation-set RMSE/MAE each
    iteration (written to test_metrics.txt in each member's work_dir) --
    confirmed via pyace.preparedata.ACEDataset and pyace.fitadapter's own
    source to be backend-agnostic (works identically for the pyace and
    tensorpot evaluators) -- rather than needing a separate, external
    LAMMPS-single-point-based comparison step. Building this dataframe once
    here (rather than via a second, separate compute_validation_floor call)
    avoids re-walking validation_directory twice for the same floor/dataset
    information.

    df.attrs['validation_dataset_path'] is the resolved path this was
    written to (None if validation_output_path wasn't given) -- stashed as
    DataFrame metadata, not a third return value, for the same
    backward-compatibility reason df.attrs['filter_counts'] already is: every
    existing caller doing `df, output_path = write_ace_dataset(...)` keeps
    working unchanged.

    Returns (dataframe, output_path) for the caller's own reporting/METADATA
    needs -- this function doesn't write a METADATA sidecar itself, since
    (unlike build_mace_ensemble_inputs) it produces exactly one dataset
    file, not N labeled ensemble folders each needing their own provenance
    record.
    """
    from EnsembleFFFit.potential.mace.build_ensemble_inputs import get_isolated_atom_e0s

    e0s = get_isolated_atom_e0s(isolated_elements_directory, properties_filename=properties_filename)
    if not e0s:
        raise FileNotFoundError(
            f"No isolated-atom reference energies found under {isolated_elements_directory} -- "
            f"has the isolated-atom DFT calculation been run yet?"
        )

    stability_floor_per_atom = None
    resolved_validation_output_path = None
    if validation_directory is not None:
        validation_df = build_ace_dataframe(
            validation_directory, e0s, structure_filename=structure_filename,
            properties_filename=properties_filename,
            max_abs_energy_per_atom=max_abs_energy_per_atom,
            max_abs_force_component=max_abs_force_component,
        )
        if validation_df.empty:
            raise RuntimeError(
                f"No usable validation structures found under {validation_directory} -- can't compute a "
                f"stability floor without at least one known real phase to compare against."
            )
        nat = validation_df['ase_atoms'].apply(len)
        epa = validation_df['energy_corrected'] / nat
        stability_floor_per_atom = float(epa.min())
        print(f"Stability floor from {validation_directory}: {stability_floor_per_atom:.4g} eV/atom "
              f"(margin {stability_floor_margin:.3g} eV/atom)")

        if validation_output_path is not None:
            resolved_validation_output_path = Path(validation_output_path).resolve()
            resolved_validation_output_path.parent.mkdir(parents=True, exist_ok=True)
            validation_df.to_pickle(resolved_validation_output_path, compression='gzip', protocol=4)
            print(f"Wrote ACE validation dataset with {len(validation_df)} structure(s) to "
                  f"{resolved_validation_output_path}")

    df = build_ace_dataframe(run_directory, e0s, structure_filename=structure_filename,
                              properties_filename=properties_filename,
                              max_abs_energy_per_atom=max_abs_energy_per_atom,
                              max_abs_force_component=max_abs_force_component,
                              stability_floor_per_atom=stability_floor_per_atom,
                              stability_floor_margin=stability_floor_margin)
    if df.empty:
        raise RuntimeError(f"No usable training structures found under {run_directory}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(output_path, compression='gzip', protocol=4)
    df.attrs['validation_dataset_path'] = (
        str(resolved_validation_output_path) if resolved_validation_output_path else None
    )

    return df, output_path
