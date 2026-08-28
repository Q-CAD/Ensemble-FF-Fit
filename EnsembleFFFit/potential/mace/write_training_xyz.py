"""
Write completed single-point DFT runs to an extxyz file MACE can train on.

Deliberately MACE-specific (not conserved DFT-agnostic logic, per the general
distinction the rest of this package draws) -- other force-field backends
will need their own parsing/writing conventions instead of this one. Reuses
ASE's own extxyz writer rather than hand-formatting the header line, so the
Lattice=/Properties=/energy=/stress= syntax matches ASE's own convention by
construction rather than by manual replication.
"""
import json
import os
from pathlib import Path

import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write
from ase.stress import voigt_6_to_full_3x3_stress
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor


def write_training_xyz(run_directory, output_xyz_path, structure_filename="POSCAR",
                        properties_filename="properties.json"):
    """
    Walk `run_directory` for every leaf containing both `structure_filename`
    and `properties_filename` (rmg_dft.py's output shape: 'energy'/'fx'/'fy'/
    'fz', plus optional 'stress_voigt' -- ASE Voigt-6, eV/Angstrom^3, absent
    if that run's yaml didn't set stress: True), and append each as one
    extxyz frame to `output_xyz_path`.

    Returns the list of leaf directories actually written, in walk order --
    the caller is responsible for turning that into a METADATA record of
    provenance, since this function only writes the .xyz itself.
    """
    written = []
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

        forces = np.column_stack([props['fx'], props['fy'], props['fz']])
        stress = (
            voigt_6_to_full_3x3_stress(np.array(props['stress_voigt']))
            if 'stress_voigt' in props else None
        )
        atoms.calc = SinglePointCalculator(
            atoms, energy=props['energy'], forces=forces, stress=stress,
        )

        write(output_xyz_path, atoms, format='extxyz', append=True)
        written.append(str(root))

    return written
