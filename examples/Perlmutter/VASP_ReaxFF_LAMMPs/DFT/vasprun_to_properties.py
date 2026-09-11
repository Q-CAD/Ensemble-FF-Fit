"""
Reconstructs POSCAR + properties.json from already-completed vasprun.xml
files, for DFT/training and DFT/validation subtrees that were condensed
down to just vasprun.xml for storage (see this repo's own DFT/vasp_dft.py,
which writes both files at run time -- this script produces the identical
POSCAR/properties.json shape after the fact, from vasprun.xml alone, for
runs that predate properties.json parsing or had their POSCAR/
properties.json stripped when condensed).

Walks a directory tree looking for vasprun.xml files, and for each one:
- Skips it if a properties.json already exists there (unless --overwrite),
  matching this pipeline's own finished_file convention everywhere else.
- Skips (with a warning, not a hard failure -- this can process hundreds of
  runs in one pass, per DFT/training's/DFT/validation's own size) any
  vasprun.xml whose run did not converge.
- Writes POSCAR from the LAST ionic step's own <calculation>-level structure
  (final_step['structure']), not any pre-existing POSCAR/POSCAR.orig that
  might already sit alongside it -- some directories in this tree (e.g.
  defects/.../vacancy0) were NOT condensed and already have their own
  POSCAR, but that POSCAR is the PRE-relaxation input structure, not the
  geometry the final energy/forces below were actually computed at. Always
  deriving POSCAR from the final ionic step instead keeps POSCAR and
  properties.json describing the exact same geometry -- the whole point of
  writing them as a pair -- rather than silently mismatching an existing
  pre-relaxation POSCAR against the relaxed-geometry forces/energy that
  follow.
  NOTE (confirmed 2026-09): deliberately NOT vasprun.final_structure here.
  That property returns vasprun.structures[-1], which pymatgen overwrites
  from the top-level <structure name="finalpos"> block rather than the
  per-<calculation> one. For DFT/validation's per-frame single-point
  vasprun.xml files (single-ionic-step re-evaluations of individual AIMD
  snapshots), that top-level initialpos/finalpos block is IDENTICAL across
  every frame of a trajectory (a stale artifact of however these per-frame
  files were produced), while final_step['structure'] -- read straight off
  that one calculation's own <structure> block -- correctly varies frame to
  frame. Verified this does NOT regress DFT/training's genuine relaxations
  (there, final_structure and ionic_steps[-1]['structure'] agree exactly),
  so this is a strict fix, not a special case.
- Writes properties.json in the exact same shape DFT/vasp_dft.py's own
  run_vasp_calculation writes it (energy/fx/fy/fz, plus stress_voigt if
  present) -- see that function for the convention this mirrors.

Usage: python vasprun_to_properties.py <root_directory> [--overwrite]
"""
import argparse
import json
import os

from pymatgen.io.vasp.outputs import Vasprun

KBAR_TO_EV_PER_A3 = 6.241509074e-4  # same constant/convention as vasp_dft.py


def find_vasprun_dirs(root_directory, vasprun_filename='vasprun.xml'):
    for dirpath, _, files in os.walk(root_directory):
        if vasprun_filename in files:
            yield dirpath


def convert_one(directory, vasprun_filename='vasprun.xml', overwrite=False):
    """
    Returns 'converted', 'skipped_done', or 'skipped_not_converged'.
    Raises on a genuine parse error -- the caller decides whether to log
    and continue past that or treat it as fatal.
    """
    properties_path = os.path.join(directory, 'properties.json')
    if os.path.exists(properties_path) and not overwrite:
        return 'skipped_done'

    vasprun_path = os.path.join(directory, vasprun_filename)
    # parse_potcar_file=False: this tree was condensed for storage and many
    # directories no longer have a sibling POTCAR -- nothing this script
    # needs (structure/energy/forces) comes from POTCAR anyway.
    # parse_dos/parse_eigen/parse_projected_eigen=False: skip the slowest
    # parts of a large vasprun.xml when only ionic-step data is needed.
    vasprun = Vasprun(vasprun_path, parse_potcar_file=False, parse_dos=False,
                       parse_eigen=False, parse_projected_eigen=False)
    # Electronic (SCF) convergence only -- NOT vasprun.converged, which also
    # requires converged_ionic. CONFIRMED via a real file in this tree
    # (validation/aimd/2000K/.../single_image/0/vasprun.xml): it carries
    # NSW=334/IBRION=0 (leftover MD flags from the original AIMD run's
    # INCAR) but only ran 1 ionic step (a deliberate single-point
    # recalculation of one snapshot) -- converged_electronic=True,
    # converged_ionic=False, so the blanket .converged check would wrongly
    # skip every such snapshot in this tree. Ionic convergence is a
    # relaxation concept; it doesn't apply to a deliberate single-point
    # evaluation, so checking it here would reject good data, not bad data.
    if not vasprun.converged_electronic:
        return 'skipped_not_converged'

    final_step = vasprun.ionic_steps[-1]
    forces = final_step['forces']

    property_dictionary = {'energy': vasprun.final_energy}
    property_dictionary['fx'] = [forces[i][0] for i in range(len(forces))]
    property_dictionary['fy'] = [forces[i][1] for i in range(len(forces))]
    property_dictionary['fz'] = [forces[i][2] for i in range(len(forces))]

    # Sign convention relative to rmg_dft.py's ASE-Voigt output not
    # verified here -- see vasp_dft.py's own stress_voigt comment, same
    # caveat applies to this identical conversion.
    stress = final_step.get('stress')
    if stress is not None:
        property_dictionary['stress_voigt'] = [
            stress[0][0] * KBAR_TO_EV_PER_A3, stress[1][1] * KBAR_TO_EV_PER_A3, stress[2][2] * KBAR_TO_EV_PER_A3,
            stress[1][2] * KBAR_TO_EV_PER_A3, stress[0][2] * KBAR_TO_EV_PER_A3, stress[0][1] * KBAR_TO_EV_PER_A3,
        ]

    final_step['structure'].to(fmt='poscar', filename=os.path.join(directory, 'POSCAR'))
    with open(properties_path, "w") as f:
        json.dump(property_dictionary, f, indent=4)

    return 'converted'


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('root_directory', help='Directory tree to walk for vasprun.xml files')
    parser.add_argument('--vasprun-filename', default='vasprun.xml',
                         help='Filename to look for under root_directory (default: vasprun.xml)')
    parser.add_argument('--overwrite', action='store_true',
                         help='Re-parse and overwrite POSCAR/properties.json even where properties.json already exists')
    args = parser.parse_args()

    counts = {'converted': 0, 'skipped_done': 0, 'skipped_not_converged': 0, 'errored': 0}
    errored_dirs = []

    for directory in find_vasprun_dirs(args.root_directory, args.vasprun_filename):
        try:
            result = convert_one(directory, args.vasprun_filename, args.overwrite)
        except Exception as e:
            counts['errored'] += 1
            errored_dirs.append(directory)
            print(f"ERROR parsing {directory}: {e}")
            continue
        counts[result] += 1
        if result != 'skipped_done':
            print(f"{result}: {directory}")

    print()
    print(f"Done: {counts['converted']} converted, {counts['skipped_done']} already had properties.json, "
          f"{counts['skipped_not_converged']} not converged, {counts['errored']} errored.")
    if errored_dirs:
        print("Errored directories:")
        for d in errored_dirs:
            print(f"  {d}")


if __name__ == "__main__":
    main()
