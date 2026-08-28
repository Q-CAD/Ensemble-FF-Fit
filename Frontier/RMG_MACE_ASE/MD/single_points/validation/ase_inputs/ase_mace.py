import sys
import os
import torch
from mace.calculators import MACECalculator
from EnsembleFFFit.utilities.general import parse_list
from ase.io import read, write
import json


def run_ase_single_points(ff_list, struct_list, output_list, in_file_list=None):
    """
    For each (force field, structure, output) triple, load the MACE model and
    structure, run a single-point energy/force calculation, and write the
    structure (as POSCAR) and computed properties into `output`.

    `in_file_list` is accepted for signature consistency with
    MDMatEnsemble.run_individual's dispatch convention (which now always
    passes a 4th list) but unused here -- single points don't need a recipe
    file the way e.g. finite-temperature MD does.
    """
    n = len(ff_list)
    assert all(len(lst) == n for lst in (struct_list, output_list)), "All lists must be same length"

    for ff, struct, output in zip(ff_list, struct_list, output_list):
        os.makedirs(output, exist_ok=True)
        property_dictionary = {}

        # 1a) Load the MACE model
        calculator = MACECalculator(model_path=ff, enable_oeq=True,
                                    device="cuda" if torch.cuda.is_available() else "cpu")

        # 1b) Load the structure file
        init_conf = read(struct)

        # 1c) Write the structure file
        write(filename=os.path.join(output, 'POSCAR'), images=init_conf)

        # 2) Compute the potential energy
        init_conf.set_calculator(calculator)
        energy = init_conf.get_potential_energy()
        forces = init_conf.get_forces()

        # 3) Write the energy to the output path
        property_dictionary['energy'] = energy
        property_dictionary['fx'] = [forces[i][0] for i in range(len(forces))]
        property_dictionary['fy'] = [forces[i][1] for i in range(len(forces))]
        property_dictionary['fz'] = [forces[i][2] for i in range(len(forces))]

        with open(os.path.join(output, 'properties.json'), "w") as f:
            json.dump(property_dictionary, f, indent=4)

    return {"status": "complete"}


if __name__ == "__main__":
    run_ase_single_points(parse_list(sys.argv[1]), parse_list(sys.argv[2]), parse_list(sys.argv[3]),
                          parse_list(sys.argv[4]) if len(sys.argv) > 4 else None)
