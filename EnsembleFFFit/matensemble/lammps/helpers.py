import ast
from pymatgen.io.lammps.data import LammpsData
#from torch_sim.quantities import calc_kinetic_energy, calc_temperature

def parse_list(arg):
    """
    Try to parse `arg` as a Python literal list via ast.literal_eval.
    If that fails, fall back to simple comma-splitting.
    """
    try:
        val = ast.literal_eval(arg)
        if isinstance(val, list):
            return val
        # if it parsed to something else, keep going to split
    except (ValueError, SyntaxError):
        pass
    # fallback
    return arg.split(',')

def get_elements(structure_path, styles=['full', 'charge', 'atomic']):
    """
    Determine which elements are present in each structure.
    Used to set the lammps pair_coefficient flags.
    """
    for style in styles:
        try:
            ld = LammpsData.from_file(structure_path, atom_style=style)
        except ValueError:
            continue
        elements = ''
        for i, element in enumerate(ld.structure.elements):
            elements += str(element)
            if i != len(ld.structure.elements) - 1:
                elements += ' '
        return elements
    return None

def make_prop_calculators(mapping):
    """
    Given a dict of { name: freq }, return the prop_calculators dict
    where each name is wired up to the correct lambda for MaceModel.
    Supported names: 'potential_energy', 'kinetic_energy', 'temperature', 'forces'
    """
    pc = {}
    for name, freq in mapping.items():
        if name == "potential_energy":
            func = lambda s, m: m(s)["energy"]
        elif name == "forces":
            func = lambda s, m: m(s)["forces"].cpu()
        elif name == "kinetic_energy":
            func = lambda s, m: calc_kinetic_energy(
                momenta=s.momenta,
                masses=s.masses,
                velocities=None
            ).unsqueeze(0)
        elif name == "temperature":
            func = lambda s, m: calc_temperature(
                momenta=s.momenta,
                masses=s.masses,
                velocities=None,
            ).unsqueeze(0)
        else:
            raise ValueError(f"Unknown prop name: {name!r}")
        pc[freq] = pc.get(freq, {})
        pc[freq][name] = func
    return pc
