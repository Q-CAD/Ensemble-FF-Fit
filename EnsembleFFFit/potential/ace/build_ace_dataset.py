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
                         properties_filename="properties.json"):
    """
    Walk `run_directory` for every leaf containing both structure_filename
    and properties_filename, building one pyace-schema row per leaf:
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

    Returns a pandas DataFrame with columns [ase_atoms, energy, forces,
    energy_corrected], in walk order.
    """
    rows = []
    skipped = []
    for root, _, files in os.walk(run_directory):
        root = Path(root)
        struct_path = root / structure_filename
        props_path = root / properties_filename
        if not (struct_path.exists() and props_path.exists()):
            continue

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

        forces = np.column_stack([props['fx'], props['fy'], props['fz']])
        stress = (
            voigt_6_to_full_3x3_stress(np.array(props['stress_voigt']))
            if 'stress_voigt' in props else None
        )
        atoms.calc = SinglePointCalculator(
            atoms, energy=props['energy'], forces=forces, stress=stress,
        )

        rows.append({
            'ase_atoms': atoms,
            'energy': props['energy'],
            'forces': forces,
            'energy_corrected': props['energy'] - reference_energy,
        })

    if skipped:
        print(f"WARNING: skipped {len(skipped)} run(s) under {run_directory} with no isolated-atom "
              f"reference energy available:")
        for path, symbols in skipped:
            print(f"  {path}: missing E0 for {symbols}")

    return pd.DataFrame(rows, columns=['ase_atoms', 'energy', 'forces', 'energy_corrected'])


def write_ace_dataset(run_directory, isolated_elements_directory, output_path,
                       structure_filename="POSCAR", properties_filename="properties.json"):
    """
    Convenience wrapper: resolve isolated-atom E0s from
    isolated_elements_directory, build the training DataFrame from
    run_directory, and save it as a gzip-compressed pickle at output_path --
    pyace's own documented on-disk format
    (df.to_pickle(path, compression='gzip', protocol=4)).

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

    df = build_ace_dataframe(run_directory, e0s, structure_filename=structure_filename,
                              properties_filename=properties_filename)
    if df.empty:
        raise RuntimeError(f"No usable training structures found under {run_directory}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(output_path, compression='gzip', protocol=4)

    return df, output_path
