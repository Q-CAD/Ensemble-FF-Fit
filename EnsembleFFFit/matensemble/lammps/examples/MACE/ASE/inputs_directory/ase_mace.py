import sys
import os
import torch
import gc
from mace.calculators import MACECalculator
from EnsembleFFFit.matensemble.lammps.helpers import parse_list
from ase.io import read
from ase.io import Trajectory
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
import ase.units as units
import json

if __name__ == "__main__":

    ff_list       = parse_list(sys.argv[1]) # Force field file paths
    input_list    = parse_list(sys.argv[2]) # Input file paths
    struct_list   = parse_list(sys.argv[3]) # Structure file paths
    output_list   = parse_list(sys.argv[4]) # LAMMPs output write paths

    n = len(ff_list)
    assert all(len(lst) == n for lst in (input_list, struct_list, output_list)), "All lists must be same length"

    for idx, (ff, inp, struct, output) in enumerate(zip(ff_list, input_list, struct_list, output_list), start=0):
        
        # 1a) Load the MACE model
        calculator = MACECalculator(model_path=ff, device="cuda" if torch.cuda.is_available() else "cpu")

        # 1b) Load the structure file
        init_conf = read(struct)

        # 2) Initialize the calculation
        init_conf.set_calculator(calculator)
        with open(inp) as fh: 
            cfg = json.load(fh)
        MaxwellBoltzmannDistribution(init_conf, temperature_K=cfg['temperature'])

        # --- MD integrator
        dt = 1.0 * units.fs
        dyn = VelocityVerlet(init_conf, dt)

        # 3) Run the MD and update the status of property_dictionary
        property_dict = {}

        def update_status():
            step = dyn.get_number_of_steps()
            energy = init_conf.get_potential_energy()
            forces = init_conf.get_forces()
            property_dictionary = {
                'energy': float(energy),
                'fx': [float(f[0]) for f in forces],
                'fy': [float(f[1]) for f in forces],
                'fz': [float(f[2]) for f in forces],
            }
            property_dict[step] = property_dictionary

        # attach the callback to run every 1000 steps
        dyn.attach(update_status, 1000)

        # --- trajectory and attachments
        traj = Trajectory(os.path.join(output, 'md_run.traj'), 'w', init_conf)
        dyn.attach(traj.write, 1000, init_conf)

        # Run the MD
        dyn.run(cfg['nsteps'])

        # close trajectory file
        traj.close()

        # --- after run: write the property_dict to disk once ---
        output_name = os.path.join(output, 'properties.json')
        with open(output_name, "w") as f:
            json.dump(property_dict, f, indent=4)

        # --- after run: free Python memory ---
        if torch.cuda.is_available():
            del calculator, init_conf, dyn  # remove large objects
            gc.collect()                # free Python memory
            torch.cuda.empty_cache()    # release unreferenced GPU memory back to CUDA driver
