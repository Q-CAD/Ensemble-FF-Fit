"""
Driver script for finite-temperature LAMMPS ReaxFF MD, run as a MatEnsemble
chore via MDMatEnsemble.build_flat_task_dicts/run_individual (see
Ensemble-FF-Fit/EnsembleFFFit/base.py), dispatched by task_dict['entry_point']
= 'run_finite_temperature_md'. Adapted from
EnsembleFFFit/molecular_dynamics/lammps/lammps_reaxff_cpu.py (the reference
driver, which predates the current Pipeline/chore pattern and used a
5-argument CLI shape) into MDMatEnsemble's fixed 4-slot (ffield, structure,
output, in_file) convention, with the same control-as-sibling-of-in_file
fix as lammps_reaxff_single_point.py -- see that module's docstring for the
rationale.

Runs the existing in.matensemble recipe (minimize -> NPT room-temperature ->
NVT melt -> NVT anneal) essentially unchanged via a single `lmp.file(inp)`
call, matching the reference driver's own structure -- this section's scope
per Perlmutter_Pipeline_Wiring.md was "write the LAMMPS driver and input
scripts... in a pattern similar to how ASE was used", not re-architecting
in.matensemble's cycle to match ase_mace_md.py's Python-orchestrated
minimize/low-T/high-T/defect-mutation loop feature-for-feature -- that
would be new scope, not a port, and is intentionally NOT attempted here.

Writes an ASE-format `md_run.traj` (the exact filename/shape
EnsembleFFFit.utilities.general.unpack_trajectory_frames expects), matching
ase_mace_md.py's own convention -- CONFIRMED (2026-09-05) via a direct,
real end-to-end test (not assumed): in.matensemble's `dump` command writes
to a single, fixed filename (dump.all, no "*" wildcard -- LAMMPS appends
each snapshot to one file instead of writing one-file-per-snapshot when
the filename has no "*"), sorted by atom-id at the source
(`dump_modify ... sort id`); once the run completes, this driver reads
that one file back via ase.io.read(..., format='lammps-dump-text',
specorder=...) and writes it out as md_run.traj, then deletes the
intermediate dump.all. Confirmed directly: LAMMPS's `dump` command (unlike
extract_atom(), see the ghost-atom note below) only ever writes local/owned
atoms, never ghosts, so no separate nlocal-slicing is needed for this path.
This also solves a real, separate problem: on Perlmutter/HPC in general,
scratch filesystems impose a hard total-file-count quota, which
incentivizes writing few, larger files over many small ones -- the
one-file-per-snapshot approach this replaced would have produced hundreds
of dump_N.dump files per structure across a real FT-MD run.

RESTART SAFETY: CONFIRMED directly (both empirically tested, not assumed)
that neither LAMMPS's own `dump` command nor ase.io.write's .traj writer
append to a pre-existing file at the same path -- both cleanly overwrite/
truncate on open, even when the new run produces fewer frames than an old,
killed-at-walltime attempt left behind. This driver still explicitly
removes any stale dump.all/md_run.traj at the start of processing each
structure regardless, as an explicit, self-documenting safeguard rather
than relying on that implicit library behavior -- doing this
unconditionally (not gated on checking for properties.json itself) is
safe because MDMatEnsemble.build_flat_task_dicts' own finished_file
filtering already guarantees properties.json does NOT exist for any
structure that reaches this loop at all (otherwise it wouldn't be in
ff_list/struct_list/output_list to begin with).

This driver DOES write a properties.json (energy/forces at the final
configuration only, via the same lmp.get_thermo()/lmp.numpy.extract_atom()
technique as lammps_reaxff_single_point.py) so the chore has a
machine-checkable completion signal (finished_file) and a same-shaped
output as every other driver in this pipeline, even though the full
per-step trace ase_mace_md.py's properties.json carries isn't reproduced.
"""
import os
import sys
import json

import numpy as np
import ase.io

from EnsembleFFFit.molecular_dynamics.helpers import parse_list, get_elements, import_lammps

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_IN_FILE = os.path.join(_THIS_DIR, 'in.matensemble')


def run_finite_temperature_md(ff_list, struct_list, output_list, in_file_list):
    n = len(ff_list)
    assert all(len(lst) == n for lst in (struct_list, output_list, in_file_list)), \
        "All lists must be same length"

    lammps = import_lammps()
    # -k on g 1 -sf kk: enable Kokkos with 1 GPU per rank and auto-suffix
    # every eligible style (pair_style reaxff -> reaxff/kk, fix qeq/reax ->
    # qeq/reax/kk -- see in.npt_room_temp's own comment) to actually use
    # the GPU this chore's own gpus_per_task=1 already reserves. CONFIRMED
    # necessary (2026-09), not cosmetic: every run up to this point used
    # the plain CPU-only styles regardless of gpus_per_task -- reserving a
    # GPU per chore did nothing on its own, since nothing in this driver
    # ever asked LAMMPS to use one. Verified these flags parse and
    # correctly engage Kokkos's CUDA backend (fails only at the
    # no-GPU-on-this-node device-count check, as expected, on a CPU-only
    # login node -- the real test is a real GPU allocation).
    lmp = lammps.lammps(cmdargs=["-k", "on", "g", "1", "-sf", "kk", "-log", "none", "-screen", "none"])

    for ff, struct, output, in_file in zip(ff_list, struct_list, output_list, in_file_list):
        os.makedirs(output, exist_ok=True)

        # Remove any stale dump.all/md_run.traj left by a previous attempt
        # that was killed (e.g. walltime) before properties.json got
        # written -- see this module's own docstring for why this is safe
        # to do unconditionally here, and why it's not strictly required
        # (both LAMMPS's dump and ase.io.write already overwrite cleanly on
        # their own) but kept anyway as an explicit safeguard.
        dump_path = os.path.join(output, 'dump.all')
        traj_path = os.path.join(output, 'md_run.traj')
        for stale_path in (dump_path, traj_path):
            if os.path.exists(stale_path):
                os.remove(stale_path)

        in_file = in_file or _DEFAULT_IN_FILE
        control = os.path.join(os.path.dirname(in_file), 'control')

        lmp.command(f"log {os.path.join(output, 'log.lammps')}")
        # No "*" wildcard -- every snapshot appends to this one file instead
        # of LAMMPS writing a separate dump_<timestep>.dump per snapshot
        # (see this module's docstring for why: HPC scratch filesystems'
        # total-file-count quotas).
        lmp.command(f"variable dump_file string {dump_path}")
        lmp.command(f"variable ff_filename string {ff}")
        lmp.command(f"variable structure string {struct}")
        lmp.command(f"variable control_filename string {control}")
        # CONFIRMED necessary (2026-09): in.npt_room_temp's write_data/
        # write_restart commands (Min.data, npt_relax.res, data.npt_relax)
        # use bare relative filenames, so without an explicit absolute
        # destination they land in whatever this chore's own process CWD
        # happens to be (the chore's own MatEnsemble/Flux working
        # directory) -- NOT this structure's own `output` directory, unlike
        # dump_file/log above which already get an explicit absolute path.
        # Confirmed via a real completed chore: data.npt_relax/Min.data/
        # npt_relax.res were all sitting in the chore's own working
        # directory, getting silently overwritten by each successive
        # structure in this same loop (only the last structure's data
        # survived), and never found by cn_checker.py's own directory walk
        # under force_fields/<ff>/structures/... at all.
        lmp.command(f"variable output_dir string {output}")

        elements = get_elements(struct)
        lmp.command(f'variable elements string "{elements}"')

        try:
            lmp.file(in_file)

            # Convert the single, appended LAMMPS dump file into the ASE-format
            # md_run.traj unpack_trajectory_frames expects -- see this module's
            # docstring for the confirmed mechanics (order=True default sorts
            # by atom-id, no ghost atoms since LAMMPS's own dump command only
            # ever writes local/owned atoms, specorder maps LAMMPS's numeric
            # types back to element symbols using the exact same type-ordering
            # convention poscar_to_structure_lmp already established when
            # structure.lmp was generated). `elements` is already computed
            # above for pair_coeff -- same "Bi Se"-style string, just split()
            # into the list specorder expects.
            frames = ase.io.read(dump_path, index=':', format='lammps-dump-text',
                                  specorder=elements.split())
            ase.io.write(traj_path, frames)
            os.remove(dump_path)

            # Final-configuration energy/forces only -- see this module's
            # docstring for why the full per-step trace isn't reproduced here.
            #
            # extract_atom("f")/extract_atom("id") return ALL atoms LAMMPS is
            # tracking -- local (owned) AND ghost (periodic images needed for
            # the reaxff cutoff's neighbor list), not just the structure's own
            # real atoms -- CONFIRMED (2026-09-05, in the single-point driver,
            # same underlying LAMMPS mechanism applies here) via a real
            # comparison against log.lammps's own reported Nghost count. Slicing
            # to the first `natoms` (from get_natoms(), correct here because
            # this pipeline always runs num_tasks=1) entries BEFORE sorting is
            # required, not optional: LAMMPS's atom-vector layout guarantees
            # local atoms occupy indices [0, nlocal) with ghosts appended after
            # (a structural invariant, not build/version-dependent), but ghost
            # atoms share the same atom-id as the real atom they're an image
            # of, so sorting by id first (before slicing) can't reliably
            # separate them -- see lammps_reaxff_single_point.py's docstring
            # for the full reasoning. Even more important here than in a
            # single-point `run 0`: this driver runs thousands of actual MD
            # steps (minimize/NPT/NVT), so LAMMPS's own internal spatial atom
            # reordering (neither in.matensemble nor any fix here disables
            # atom_modify sort) is far more likely to have actually triggered
            # by the time these final forces are read, making the subsequent
            # id-sort (np.argsort, handling the LAMMPS 1-indexed vs Python
            # 0-indexed shift automatically) just as necessary as the slice.
            energy = lmp.get_thermo("pe")
            natoms = lmp.get_natoms()
            ids = np.asarray(lmp.numpy.extract_atom("id"))[:natoms]
            forces = np.asarray(lmp.numpy.extract_atom("f"))[:natoms]
            forces = forces[np.argsort(ids)]

            property_dictionary = {'energy': float(energy)}
            property_dictionary['fx'] = [float(forces[i][0]) for i in range(natoms)]
            property_dictionary['fy'] = [float(forces[i][1]) for i in range(natoms)]
            property_dictionary['fz'] = [float(forces[i][2]) for i in range(natoms)]
            property_dictionary['_status'] = 'complete'
        except Exception as e:
            # A LAMMPS run-time error (e.g. "Non-numeric pressure -
            # simulation unstable") raises a clean Python exception here
            # (lammps.core.py's own ExceptionCheck), not a hard crash --
            # CONFIRMED (2026-09) via a real chore's own stderr. Treated as
            # a genuine, recordable OUTCOME (this force field is simply
            # unstable for this structure under real dynamics -- exactly
            # the kind of motif breakdown this check exists to catch), not
            # an infra failure to retry: write a failed-status
            # properties.json instead of letting the exception propagate,
            # so (a) build_task_dicts' own finished_file=properties.json
            # gating sees this structure as done and never re-submits it,
            # and (b) check_coordination_stability can distinguish "ran
            # fine" from "physically unstable" downstream (see cn_checker.
            # py's own comparison_paths) instead of just silently missing
            # data.npt_relax with no explanation. Deliberately continues
            # the loop (does not re-raise) so one unstable structure
            # doesn't abort the rest of this chore's own force field x
            # structure batch.
            property_dictionary = {'_status': 'failed', 'error': str(e)}
            with open(os.path.join(output, 'properties.json'), "w") as f:
                json.dump(property_dictionary, f, indent=4)
            lmp.command("clear")
            continue

        with open(os.path.join(output, 'properties.json'), "w") as f:
            json.dump(property_dictionary, f, indent=4)

        lmp.command("clear")

    lmp.close()
    return {"status": "complete"}


if __name__ == "__main__":
    run_finite_temperature_md(parse_list(sys.argv[1]), parse_list(sys.argv[2]),
                               parse_list(sys.argv[3]), parse_list(sys.argv[4]))
