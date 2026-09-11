"""
Driver script for a single-point LAMMPS ReaxFF energy/force evaluation, run
as a MatEnsemble chore. Dispatched dynamically by MDMatEnsemble.run_individual
(see Ensemble-FF-Fit/EnsembleFFFit/base.py) via task_dict['task_command']/
task_dict['entry_point'] -- same convention as MD/*/ase_inputs/ase_mace*.py
(now MD/*/lammps_inputs/), kept in the example rather than the installed
package since it's a site-specific driver. Loosely adapted from
EnsembleFFFit/molecular_dynamics/lammps/lammps_reaxff_cpu.py, which predates
the current Pipeline/chore pattern and used a 5-argument (ff, control, in,
struct, output) CLI shape rather than MDMatEnsemble's fixed 4-slot
convention -- see this module's own docstring below for how `control` is
handled instead.

Genuinely new: there was no existing single-point LAMMPS/ReaxFF template to
adapt (only in.matensemble's finite-temperature MD cycle existed) --
in.single_point (this driver's default in_file) is new too, see its own
header comment.

Same loading convention as ase_mace.py: a plain entry function taking
MDMatEnsemble's fixed 4-slot (ffield, structure, output, in_file)
positional-list convention, runnable standalone via
`python lammps_reaxff_single_point.py '["ff1", ...]' '["struct1", ...]' '["out1", ...]' '["in1", ...]'`
using parse_list to parse each sys.argv entry.

`control` (the LAMMPS reaxff control file -- cutoffs, tabulation
granularity) is NOT one of MDMatEnsemble's 4 fixed slots (that contract is
documented as deliberately stable in CLAUDE.md, and adding a 5th slot was
weighed against this and rejected -- see Perlmutter_Pipeline_Wiring.md).
Discovered instead as a fixed sibling of `in_file` (i.e. living in the same
directory as the .in script) rather than of `ffield` -- the architecturally
better fit of the two "sibling" options considered: control configures *how
the simulation runs* (a recipe concern, same category as in_file), not
*which force field is used* (ffield's concern), and its contents don't vary
per force-field variant -- so this also means control never needs to be
copied alongside each per-variant force field the way ffield itself does
(see run_pipeline.py's copy_and_spawn_md), unlike the sibling-of-ffield
option originally floated.

Unlike the reference lammps_reaxff_cpu.py (which never wrote properties.json
at all -- its only output was LAMMPS's own log/dump files), this driver
extracts energy/forces directly via the LAMMPS Python module's
lmp.get_thermo()/lmp.numpy.extract_atom() and writes properties.json in the
same energy/fx/fy/fz shape ase_mace.py/vasp_dft.py do, so downstream
analysis code (best_force_field.py, variance.py, ...) doesn't need to know
which MD/DFT backend produced a given properties.json. This also sidesteps
the log-parsing fragility EnsembleFFFit/molecular_dynamics/lammps/
lammps_properties.py's own docstring flags. UNVERIFIED: the exact
lmp.numpy.extract_atom("f")/lmp.get_thermo("pe") call shapes are inferred
from the general LAMMPS Python API, not confirmed against a real compiled
LAMMPS+Python build (Perlmutter's compute nodes were degraded for the
entirety of this pipeline's development) -- check these first against the
actual `lammps` module docstrings/behavior before trusting this blindly.
"""
import os
import sys
import json

import numpy as np

from EnsembleFFFit.molecular_dynamics.helpers import parse_list, get_elements, import_lammps

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_IN_FILE = os.path.join(_THIS_DIR, 'in.single_point')


def run_lammps_single_points(ff_list, struct_list, output_list, in_file_list=None):
    """
    Named for what actually runs here (LAMMPS, not ASE) -- MD_single_points/
    MD_uq_single_points' entry_point config keys name this function
    explicitly, so there's no need to keep an ase_mace.py-derived name
    around just for drop-in consistency with that other driver.
    """
    n = len(ff_list)
    assert all(len(lst) == n for lst in (struct_list, output_list)), "All lists must be same length"
    if in_file_list is None:
        in_file_list = [None] * n
    assert len(in_file_list) == n, "in_file_list must be the same length as ff_list"

    lammps = import_lammps()
    lmp = lammps.lammps(cmdargs=["-log", "none", "-screen", "none"])

    for ff, struct, output, in_file in zip(ff_list, struct_list, output_list, in_file_list):
        os.makedirs(output, exist_ok=True)

        in_file = in_file or _DEFAULT_IN_FILE
        control = os.path.join(os.path.dirname(in_file), 'control')

        lmp.command(f"log {os.path.join(output, 'log.lammps')}")
        lmp.command(f"variable ff_filename string {ff}")
        lmp.command(f"variable structure string {struct}")
        lmp.command(f"variable control_filename string {control}")

        elements = get_elements(struct)
        lmp.command(f'variable elements string "{elements}"')

        lmp.file(in_file)

        # Extract final energy/forces directly from the running LAMMPS
        # instance -- avoids re-parsing the log file it just wrote (see this
        # module's docstring for why). in.single_point ends with `run 0`,
        # which computes forces/energy for the current configuration without
        # any timestepping -- a true single point, matching ase_mace.py's
        # own behavior of never relaxing before evaluating.
        #
        # extract_atom("f")/extract_atom("id") return ALL atoms LAMMPS is
        # currently tracking -- local (owned) AND ghost (periodic images
        # needed for the reaxff cutoff's neighbor list), not just the
        # structure's own real atoms. CONFIRMED (2026-09-05): a 35-atom
        # structure's data file, on a small/thin cell against a 13 A
        # cutoff, produced ~2237 ghost atoms and an unsliced array of
        # ~2272 entries -- writing all of them would silently corrupt
        # properties.json with thousands of extra, non-physical "atoms".
        # LAMMPS's own internal atom-vector layout guarantees local atoms
        # occupy indices [0, nlocal) with ghosts always appended after (a
        # structural invariant every pair style relies on, not something
        # that varies by build/version) -- slicing to the first `natoms`
        # entries BEFORE sorting isolates the real atoms correctly. This
        # has to happen before sorting by id, not after: ghost atoms share
        # the same atom-id as the real atom they're a periodic image of,
        # so a plain id-sort over the full (real+ghost) array can't
        # reliably separate them (np.argsort's default quicksort isn't
        # even stable across equal keys) -- only the structural local-
        # atoms-come-first guarantee can. get_natoms() gives the correct
        # count to slice to here specifically because every chore in this
        # pipeline runs num_tasks=1 (one MPI rank per structure) -- for
        # that case, LAMMPS's own global atom total exactly equals nlocal,
        # since there's no other rank for atoms to be split across.
        #
        # Once correctly reduced to just the natoms real atoms, sorting by
        # LAMMPS atom-id is still needed separately -- extract_atom("f")'s
        # own row order isn't guaranteed to match atom-id order even among
        # real atoms (LAMMPS can reorder atoms internally, e.g. spatial
        # sorting for performance -- neither in.single_point nor
        # in.matensemble disables this via atom_modify sort). np.argsort on
        # the id array handles the LAMMPS (1-indexed) vs Python (0-indexed)
        # shift automatically: sorting ids [1..natoms] into ascending order
        # puts atom-id 1's force at array position 0, exactly matching
        # POSCAR's own 0-indexed site order -- no manual index arithmetic
        # needed.
        energy = lmp.get_thermo("pe")
        natoms = lmp.get_natoms()
        ids = np.asarray(lmp.numpy.extract_atom("id"))[:natoms]
        forces = np.asarray(lmp.numpy.extract_atom("f"))[:natoms]
        forces = forces[np.argsort(ids)]

        property_dictionary = {'energy': float(energy)}
        property_dictionary['fx'] = [float(forces[i][0]) for i in range(natoms)]
        property_dictionary['fy'] = [float(forces[i][1]) for i in range(natoms)]
        property_dictionary['fz'] = [float(forces[i][2]) for i in range(natoms)]

        with open(os.path.join(output, 'properties.json'), "w") as f:
            json.dump(property_dictionary, f, indent=4)

        lmp.command("clear")

    lmp.close()
    return {"status": "complete"}


if __name__ == "__main__":
    run_lammps_single_points(
        parse_list(sys.argv[1]), parse_list(sys.argv[2]), parse_list(sys.argv[3]),
        parse_list(sys.argv[4]) if len(sys.argv) > 4 else None,
    )
