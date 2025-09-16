import sys
from lammps import lammps
from pymatgen.io.lammps.data import LammpsData

ff_filename   = sys.argv[1]
lmp_input     = sys.argv[2]
control_filename = sys.argv[3]
structure     = sys.argv[4]

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

lmp = lammps()
lmp.command(f"variable structure string {structure}")
lmp.command(f"variable ff_filename string {ff_filename}")
lmp.command(f"variable control_filename string {control_filename}")
elements = get_elements(structure)
lmp.command(f'variable elements string "{elements}"')
lmp.file(lmp_input)
lmp.close()
