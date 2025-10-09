import ast
from pymatgen.io.lammps.data import LammpsData

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
