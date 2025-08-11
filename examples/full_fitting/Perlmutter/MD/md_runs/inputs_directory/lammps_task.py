import sys
from lammps import lammps

ff_filename   = sys.argv[1]
lmp_input     = sys.argv[2]
control_filename = sys.argv[3]
structure     = sys.argv[4]

lmp = lammps()
lmp.command(f"variable structure string {structure}")
lmp.command(f"variable ff_filename string {ff_filename}")
lmp.command(f"variable control_filename string {control_filename}")
lmp.file(lmp_input)
lmp.close()
