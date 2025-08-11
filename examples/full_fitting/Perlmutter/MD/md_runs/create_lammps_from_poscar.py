from pymatgen.core.structure import Structure
from pymatgen.io.lammps.data import LammpsData
from pymatgen.transformations.advanced_transformations import CubicSupercellTransformation
from pathlib import Path
import sys
import os

def main():
    from_poscar = sys.argv[1]
    structure = Structure.from_file(from_poscar)
    to_lammps = sys.argv[2]
    make_supercell = eval(sys.argv[3])

    p = os.path.abspath(Path(to_lammps).parent)
    os.makedirs(p, exist_ok=True)
    if make_supercell:
        cst = CubicSupercellTransformation(max_atoms=370, allow_orthorhombic=True, max_length=50)
        structure = cst.apply_transformation(structure)
    structure_to_lammps(structure, to_lammps)
    return

def structure_to_lammps(structure, write_path, charge_dct={'Bi': 1, 'Se': -0.667}):
    structure.add_site_property("charge", [charge_dct[str(s.specie)] for s in structure])
    ld = LammpsData.from_structure(structure)
    ld.write_file(filename=write_path, distance=8, charge=5)
    return

if __name__ == '__main__':
    main()        
