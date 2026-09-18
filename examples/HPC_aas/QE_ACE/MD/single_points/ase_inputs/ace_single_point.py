"""
Run single-point (energy/forces, no MD) evaluations using pyace's own ASE
Calculator (PyACECalculator), one call per (force field, structure) pair,
mirroring examples/Frontier/RMG_MACE_ASE/MD/single_points/validation/
ase_inputs/ase_mace.py's own run_ase_single_points as closely as possible
(same function name/signature/return shape) -- that script is the
established single-point precedent this project's driver scripts are meant
to stay consistent with, not MD/ace_md.py's own finite-temperature driver
(0 MD steps isn't the same thing as a dedicated single-point script).

Dispatched via MDMatEnsemble.run_individual's standard 4-list contract
(ffield_list, structure_list, output_list, in_file_list) -- in_file_list is
accepted for signature consistency but unused, same as ase_mace.py's own
docstring already explains (single points don't need a recipe file).

CPU-only, same rationale as MD/ace_md.py: PyACECalculator has no GPU/device
argument at all.
"""
import json
import os
import sys

from ase.io import read, write

from EnsembleFFFit.utilities.general import parse_list


def run_ase_single_points(ff_list, struct_list, output_list, in_file_list=None):
    """
    For each (force field, structure, output) triple, load the ACE
    potential and structure, run a single-point energy/force calculation,
    and write the structure (as POSCAR) and computed properties into
    `output` -- same steps/output shape as ase_mace.py's own
    run_ase_single_points.
    """
    from pyace.asecalc import PyACECalculator

    n = len(ff_list)
    assert all(len(lst) == n for lst in (struct_list, output_list)), "All lists must be same length"

    for ff, struct, output in zip(ff_list, struct_list, output_list):
        os.makedirs(output, exist_ok=True)
        property_dictionary = {}

        # 1a) Load the ACE potential
        calculator = PyACECalculator(ff)

        # 1b) Load the structure file
        init_conf = read(struct)

        # 1c) Write the structure file
        write(filename=os.path.join(output, 'POSCAR'), images=init_conf)

        # 2) Compute the potential energy
        init_conf.calc = calculator
        energy = init_conf.get_potential_energy()
        forces = init_conf.get_forces()

        # 3) Write the energy to the output path
        property_dictionary['energy'] = float(energy)
        property_dictionary['fx'] = [float(forces[i][0]) for i in range(len(forces))]
        property_dictionary['fy'] = [float(forces[i][1]) for i in range(len(forces))]
        property_dictionary['fz'] = [float(forces[i][2]) for i in range(len(forces))]

        with open(os.path.join(output, 'properties.json'), "w") as f:
            json.dump(property_dictionary, f, indent=4)

    return {"status": "complete"}


if __name__ == "__main__":
    run_ase_single_points(parse_list(sys.argv[1]), parse_list(sys.argv[2]), parse_list(sys.argv[3]),
                          parse_list(sys.argv[4]) if len(sys.argv) > 4 else None)
