"""
Driver script for a single-point LAMMPS/pair_style-pace (ACE) energy/force
evaluation, run as a MatEnsemble chore -- dispatched dynamically by
MDMatEnsemble.run_individual (see Ensemble-FF-Fit/EnsembleFFFit/base.py) via
task_dict['task_command']/task_dict['entry_point'], same convention as
MD/single_points/reaxff_validation/lammps_inputs/lammps_reaxff_single_point.py
in the VASP_ReaxFF_LAMMPs example this was adapted from. Kept in this
project's own run_pipeline tree (not the installed Ensemble-FF-Fit package),
same reasoning as FF/ace_fit.py -- a site-specific driver, not portable
package logic.

Genuinely new relative to the ReaxFF precedent, not just renamed:
  - pair_style pace, not reaxff -- no qeq/reax fix (ACE has no charge
    equilibration step at all), atom_style atomic (not charge).
  - Activates LAMMPS's KOKKOS package via cmdargs (-k on g 1 -sf kk -pk
    kokkos neigh half newton on), not just inherited from the chore's own
    GPU resource allocation -- CONFIRMED (2026-10-13) against this exact
    container by reading pair_pace_kokkos.cpp's own init_style() directly:
    it requires `newton pair on` and errors if the Kokkos neighbor list is
    `full` ("Must use half neighbor list style with pair pace/kk"), so
    `neigh half` is required, not just a reasonable default -- same class
    of requirement the VASP_ReaxFF_LAMMPs README's own Kokkos pitfalls
    section documents for pair_style reaxff/kk, confirmed independently
    here for pace/kk rather than assumed to carry over. The `-k on g 1 -sf
    kk -pk kokkos neigh half newton on` cmdargs combination itself was
    smoke-tested directly against this container's own `lmp` binary (got
    past KOKKOS's own GPU-device-count check, only failing afterward on
    this login node's lack of physical GPU hardware -- expected there, not
    a flag-syntax problem).
  - get_elements(struct)'s returned element-symbol ORDER is REQUIRED here,
    not just convenient -- CONFIRMED empirically (2026-10-13, not assumed):
    pair_coeff's element arguments map LAMMPS atom types to ACE species BY
    NAME (LAMMPS's standard Pair::map_element2type, confirmed via
    pair_pace.cpp's own coeff()), so correctness depends on get_elements
    reporting the SAME per-atom-type symbol order the structure.lmp file
    actually encodes -- which a direct smoke test confirmed does NOT
    always match poscar_to_structure_lmp's own returned ff_elements list
    (one real validation structure: get_elements gave "Pd Se",
    poscar_to_structure_lmp's own return value gave "Se Pd" -- only the
    get_elements order reproduced pyace's own PyACECalculator reference
    energy to 5 significant figures; the other order was off by several
    eV). Always re-derive elements from get_elements(struct) at execution
    time, never from poscar_to_structure_lmp's return value or any other
    cached/assumed ordering.

Same loading convention as lammps_reaxff_single_point.py: a plain entry
function taking MDMatEnsemble's fixed 4-slot (ffield, structure, output,
in_file) positional-list convention, runnable standalone via
`python lammps_ace_single_point.py '["ff1", ...]' '["struct1", ...]' '["out1", ...]' '["in1", ...]'`
using parse_list to parse each sys.argv entry. No `control` sibling file
unlike the ReaxFF driver -- ACE's potential file carries everything
pair_style pace needs (cutoffs, basis), no separate tabulation/cutoff
recipe file.
"""
import os
import sys
import json

import numpy as np

from EnsembleFFFit.molecular_dynamics.helpers import parse_list, get_elements, import_lammps

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_IN_FILE = os.path.join(_THIS_DIR, 'in.single_point_ace')

# -k on g 1: one GPU per chore, matching this stage's own gpus_per_task: 1
# resource request. -sf kk: auto-suffix pair_style (and everything else
# that has a /kk variant) with /kk, so in.single_point_ace's own plain
# `pair_style pace` line runs on Kokkos without needing a separate
# Kokkos-specific .in file. -pk kokkos neigh half newton on: the one
# non-default Kokkos setting pair_pace_kokkos.cpp's own init_style()
# actually requires (see this module's own docstring) -- NOT optional,
# confirmed via source and a real smoke test of this exact flag
# combination, not guessed from documentation alone.
_KOKKOS_CMDARGS = ["-k", "on", "g", "1", "-sf", "kk", "-pk", "kokkos", "neigh", "half", "newton", "on"]


def run_lammps_single_points(ff_list, struct_list, output_list, in_file_list=None):
    """
    Named to match lammps_reaxff_single_point.py's own entry-point name
    (both drivers are dispatched the same way, by
    task_dict['entry_point'] == 'run_lammps_single_points') -- the actual
    pair style/units/elements logic is what differs, not the function
    name or its signature.
    """
    n = len(ff_list)
    assert all(len(lst) == n for lst in (struct_list, output_list)), "All lists must be same length"
    if in_file_list is None:
        in_file_list = [None] * n
    assert len(in_file_list) == n, "in_file_list must be the same length as ff_list"

    lammps = import_lammps()
    lmp = lammps.lammps(cmdargs=_KOKKOS_CMDARGS + ["-log", "none", "-screen", "none"])

    for ff, struct, output, in_file in zip(ff_list, struct_list, output_list, in_file_list):
        os.makedirs(output, exist_ok=True)

        in_file = in_file or _DEFAULT_IN_FILE

        lmp.command(f"log {os.path.join(output, 'log.lammps')}")
        lmp.command(f"variable ff_filename string {ff}")
        lmp.command(f"variable structure string {struct}")

        elements = get_elements(struct)
        lmp.command(f'variable elements string "{elements}"')

        lmp.file(in_file)

        # Same ghost-atom slicing / atom-id sort as
        # lammps_reaxff_single_point.py -- see that driver's own docstring
        # for the full reasoning (extract_atom returns local+ghost atoms,
        # local atoms always occupy indices [0, nlocal) by LAMMPS's own
        # structural invariant; get_natoms() is the real atom count only
        # because every chore here runs num_tasks=1, one MPI rank per
        # structure).
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
