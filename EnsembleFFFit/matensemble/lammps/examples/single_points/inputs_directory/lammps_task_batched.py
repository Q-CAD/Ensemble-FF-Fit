import lammps
import lammps.mliap
import torch
from pymatgen.io.lammps.data import LammpsData
import sys
import ast
from pathlib import Path

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

ff_list       = parse_list(sys.argv[1])
input_list    = parse_list(sys.argv[2])
struct_list   = parse_list(sys.argv[3])

n = len(ff_list)
assert all(len(lst) == n for lst in (input_list, struct_list)), "All lists must be same length"

lmp = lammps.lammps(cmdargs=['-k', 'on', 'g', '4', '-sf', 'kk', 
                      '-pk', 'kokkos', 'neigh', 'half', 
                      'newton', 'off', '-echo', 'both', 
                      "-log", "none", "-screen", "os.devnull"])
lammps.mliap.activate_mliappy_kokkos(lmp)

for idx, (ff, inp, struct) in enumerate(zip(ff_list, input_list, struct_list), start=0):
    # 1) Open a new log file
    image_name = Path(struct).parent.name
    lmp.command(f"log log_{image_name}.lammps")

    # 2) Pass in your filenames
    if ff:
        lmp.command(f"variable ff_filename string {ff}")
    if struct:
        lmp.command(f"variable structure string {struct}")

    # 3) Determine the pair coefficient
    elements = get_elements(struct)
    lmp.command(f'variable elements string "{elements}"')

    # 4) Run the LAMMPS input
    lmp.file(inp)

    # 5) Clear for the next iteration
    lmp.command("clear")

# Final cleanup
lmp.close()
