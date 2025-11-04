import sys
import json
import os
import numpy as np
from ase.io import write
from pymatgen.core import Composition
from EnsembleFFFit.analysis.properties import parse_single_points

def get_reference_per_atom(dct):
    # --- Step 1: compute per-atom reference energies ---
    ref_per_atom = {}
    for ffield, sub in dct.items():
        for element, entries in sub.items():
            energies = []
            for subkey, props in entries.items():
                atoms = props["atoms"]
                e_tot = props["energy"]
                n_atoms = len(atoms)
                energies.append(e_tot / n_atoms)
            ref_per_atom[element] = np.min(energies)
        print("Reference per-atom energies:", ref_per_atom)
    return ref_per_atom

def get_formation_energy_data(dct, ref_dct):
    formation_data = []
    for ffield, sub in dct.items():
        for formula, entries in sub.items():
            for subkey, props in entries.items():
                atoms = props["atoms"]
                e_tot = props["energy"]

                # Get composition (dict of element -> count)
                symbols, counts = np.unique(atoms.get_chemical_symbols(), return_counts=True)
                s_symbols = [str(s) for s in symbols]
                composition = dict(zip(s_symbols, counts))
                real_formula = atoms.get_chemical_formula()
                
                # Total number of atoms in the system
                n_total = len(atoms)

                # Reference energy for that composition
                e_ref = 0.0
                for elem, count in composition.items():
                    if elem not in list(ref_dct.keys()):
                        raise KeyError(f"Missing reference for element {elem}")
                    e_ref += ref_dct[elem] * count

                # Formation energy per atom
                try:
                    e_form = (e_tot - e_ref) / n_total
                    formation_data.append((formula, real_formula, subkey, atoms, e_form))
                except TypeError:
                    print(f'Energy of {e_tot} for {formula}')

    # --- Step 3: sort results ---
    formation_data.sort(key=lambda x: x[-1])

    return formation_data

if __name__ == '__main__':
    references_path = sys.argv[1]
    phases_path = sys.argv[2]

    reference_dct = parse_single_points(references_path, dump_index=-1, ffield_label=(-5, -3))
    structure_dct = parse_single_points(phases_path, dump_index=-1, ffield_label=(-5, -3))

    reference_per_atom = get_reference_per_atom(reference_dct)
    formation_sorted = get_formation_energy_data(structure_dct, reference_per_atom)

    count = 0
    for label, formula, subkey, atoms, e_form in formation_sorted:
        print(f"{label:15s} {formula:15s} {Composition(formula).reduced_formula:15s} Formation energy = {e_form:.4f} eV/atom")

        # Write each to an output path
        dirs = os.path.join(os.getcwd(), 'output', str(count), str(formula))
        os.makedirs(dirs, exist_ok=True)
        write(filename=os.path.join(dirs, 'POSCAR'), images=[atoms], format='vasp')

        # Write MACE energy to a text file
        with open(os.path.join(dirs, "energy.json"), "w") as f:
            json.dump({'formation_energy': e_form}, f, indent=2)

        count += 1

