import sys
import os
import torch
import gc
from torch_sim.models.mace import MaceModel
from torch_sim import static, integrate
from torch_sim.integrators import nvt_langevin
from EnsembleFFFit.matensemble.lammps.helpers import parse_list
from EnsembleFFFit.matensemble.lammps.helpers import make_prop_calculators
from ase.io import read
import json

if __name__ == "__main__":

    ff_list       = parse_list(sys.argv[1]) # Force field file paths
    input_list    = parse_list(sys.argv[2]) # Input file paths
    struct_list   = parse_list(sys.argv[3]) # Structure file paths
    output_list   = parse_list(sys.argv[4]) # TorchSim output write paths

    n = len(ff_list)
    assert all(x == ff_list[0] for x in ff_list), "Not all force fields are the same for each batch"
    assert all(x == input_list[0] for x in input_list), "Not all inputs are the same for each batch"
    assert all(len(lst) == n for lst in (input_list, struct_list, output_list)), "All lists must be same length"

    # 1a) Load the MACE model
    model = torch.load(ff_list[0], map_location="cuda" if torch.cuda.is_available() else "cpu")
    mace_model = MaceModel(model=model) #, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # 1b) Read the atoms objects and configuration dictionary
    init_confs = [read(struct) for struct in struct_list]
    trajectory_files = [os.path.join(output, f"md_run.h5md") for i, output in enumerate(output_list)]
    with open(input_list[0]) as fh:
        cfg = json.load(fh)

    # 3) Run the MD
    mapping = {'potential_energy': cfg['frequency'], 
               'kinetic_energy': cfg['frequency'], 
               'forces': cfg['frequency']}

    prop_calculators = make_prop_calculators(mapping, debug=False)
    
    final_state = integrate(system=init_confs,
                            model=mace_model,
                            n_steps=cfg['nsteps'],
                            timestep=cfg['timestep'], # in Metal units
                            temperature=cfg['temperature'],
                            integrator=nvt_langevin,
                            trajectory_reporter=dict(filenames=trajectory_files,
                                                     state_frequency=cfg['frequency'],  # snapshot write-out
                                                     prop_calculators=prop_calculators))

    # --- after run: free Python memory ---
    if torch.cuda.is_available():
        del mace_model, final_state  # remove large objects
        gc.collect()                # free Python memory
        torch.cuda.empty_cache()    # release unreferenced GPU memory back to CUDA driver
