"""
Run a single LAMMPS single-point/MD task via the LAMMPS Python interface.

Expects four positional CLI args (force-field file, LAMMPS input script, control
file, structure file), passes them to the LAMMPS input script as string variables,
and executes it via `lmp.file(lmp_input)`. Used as a per-task command dispatched by
MatEnsemble (see EnsembleFFFit.molecular_dynamics.pyMD.lammps_matensemble_cli).
"""
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
