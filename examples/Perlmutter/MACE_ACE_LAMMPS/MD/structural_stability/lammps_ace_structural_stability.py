"""
Driver for the room-temperature structural-stability check
(MD/finite_temperature/structural_stability), run as a MatEnsemble chore via
MDMatEnsemble.build_task_dicts/run_individual -- same 4-slot (ffield,
structure, output, in_file) contract as
MD/single_points/ace_validation/lammps_inputs/lammps_ace_single_point.py,
which this reuses the Kokkos cmdargs/get_elements-ordering/ghost-atom-
slicing conventions from directly (see that module's own docstring for the
full reasoning on each of those -- not re-derived here). Also adapted from
MatEnsemble/melting_pipeline/MD/single_melt/lammps_reaxff_melt_screen.py for
the two-dump-stream (wrapped + unwrapped)/marker-based production-thermo
parsing machinery.

What's different here, and why:

1. pair_style pace (ACE) via the same _KOKKOS_CMDARGS as
   lammps_ace_single_point.py, not reaxff/kk -- no qeq/reax fix, no control
   file. Minimize + a single room-temperature NPT equilibration + a single
   room-temperature NPT production stage -- no temperature ramp (this is a
   one-condition stability check, not a melting-point screen).

2. TWO dump streams (wrapped id/type/x/y/z/fx/fy/fz -> md_run.traj, and
   unwrapped id/type/xu/yu/zu -> md_run_unwrapped.traj), same reasoning as
   lammps_reaxff_melt_screen.py's own docstring (MSD needs true unwrapped
   positions; ASE's lammps-dump-text reader can't disambiguate wrapped vs.
   unwrapped coordinates in one combined dump). UNLIKE that reference,
   in.structural_stability (see run_build_structural_stability_inputs)
   defines both dumps AFTER the equilibration stage, immediately before the
   production run -- so md_run.traj/md_run_unwrapped.traj here contain ONLY
   production-window frames, no minimize/equilibration frames mixed in.
   structural_stability_checker.py's own "first vs. last
   equilibrium_window_fraction of frames" baseline is therefore EXACT, not
   the reference's documented approximation.

3. properties.json carries both the PRODUCTION-window thermo trace/
   time-average (same marker-split convention as
   lammps_reaxff_melt_screen.py) AND the EQUILIBRATION-stage thermo trace's
   first/last rows, as a direct PE/atom and TotEng/atom drift check over
   that short pre-production stage -- a literal answer to "does
   timestep_fs cause integrator energy drift", not reconstructable from the
   production-only window alone.
"""
import os
import re
import sys
import json

import numpy as np
import ase.io

from EnsembleFFFit.molecular_dynamics.helpers import parse_list, get_elements, import_lammps

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# Same Kokkos requirement as lammps_ace_single_point.py -- see that
# module's own docstring for why each of these flags is required, not just
# a reasonable default (confirmed via pair_pace_kokkos.cpp's own
# init_style()).
_KOKKOS_CMDARGS = ["-k", "on", "g", "1", "-sf", "kk", "-pk", "kokkos", "neigh", "half", "newton", "on"]

_PRODUCTION_MARKER = "===PRODUCTION_START==="

# Same thermo-header regex as lammps_reaxff_melt_screen.py (LAMMPS
# left-pads this header line with whitespace for column alignment) -- one
# shared thermo_style custom column layout is set once, globally, in
# in.structural_stability, so there's exactly one layout to handle.
_THERMO_HEADER_RE = re.compile(r'^\s*(Step\s+\S.*)$', re.MULTILINE)


def _parse_thermo_blocks(text, columns):
    """Returns a list of per-block row-lists, one block per thermo table
    (header line + numeric rows, terminated by 'Loop time' or end of text)
    found in `text` -- see this module's own docstring for why a block is
    bounded this way (LAMMPS emits one such table per `run`/`minimize`
    command, all sharing the same thermo_style custom column layout)."""
    blocks = []
    for header_match in _THERMO_HEADER_RE.finditer(text):
        header_cols = header_match.group(1).split()
        block_start = header_match.end()
        block_end = text.find('Loop time', block_start)
        block = text[block_start:block_end if block_end != -1 else len(text)]
        rows = []
        for line in block.strip().splitlines():
            parts = line.split()
            if len(parts) != len(header_cols):
                continue  # not a data row (e.g. a WARNING line interleaved by LAMMPS)
            try:
                values = [float(p) for p in parts]
            except ValueError:
                continue
            rows.append(dict(zip(header_cols, values)))
        if rows:
            missing = [c for c in columns if c not in rows[0]]
            if not missing:
                blocks.append(rows)
    return blocks


def _parse_log(log_path, columns):
    """Returns (equilibration_rows, production_rows) -- equilibration_rows
    is the LAST thermo block found BEFORE the production marker (the
    `fix equil`/`run` stage; minimize's own earlier block(s) are skipped,
    since those represent 0K-ish relaxation toward this potential's own
    equilibrium structure, not real room-temperature dynamics), and
    production_rows is every row from every thermo block found AFTER the
    marker (there is exactly one `fix production`/`run` stage in
    in.structural_stability, so this is equivalent to that one block)."""
    with open(log_path) as f:
        text = f.read()

    marker_idx = text.rfind(_PRODUCTION_MARKER)
    if marker_idx == -1:
        raise RuntimeError(f"{log_path}: no {_PRODUCTION_MARKER!r} marker found -- "
                            f"in.structural_stability should print it immediately before the production run")

    equilibration_blocks = _parse_thermo_blocks(text[:marker_idx], columns)
    if not equilibration_blocks:
        raise RuntimeError(f"{log_path}: no equilibration-stage thermo rows found before {_PRODUCTION_MARKER!r}")
    equilibration_rows = equilibration_blocks[-1]

    production_blocks = _parse_thermo_blocks(text[marker_idx:], columns)
    production_rows = [row for block in production_blocks for row in block]
    if not production_rows:
        raise RuntimeError(f"{log_path}: found {_PRODUCTION_MARKER!r} but no thermo rows after it")

    return equilibration_rows, production_rows


def _equilibrium_window_average(rows, columns, window_fraction):
    n = max(1, int(round(len(rows) * window_fraction)))
    window = rows[-n:]
    return {c: float(np.mean([r[c] for r in window])) for c in columns}


def run_structural_stability(ff_list, struct_list, output_list, in_file_list,
                              thermo_columns=('Step', 'Temp', 'PotEng', 'TotEng', 'Press',
                                              'Lx', 'Ly', 'Lz', 'Volume', 'Density'),
                              equilibrium_window_fraction=0.2):
    n = len(ff_list)
    assert all(len(lst) == n for lst in (struct_list, output_list, in_file_list)), \
        "All lists must be same length"

    lammps = import_lammps()
    lmp = lammps.lammps(cmdargs=_KOKKOS_CMDARGS + ["-log", "none", "-screen", "none"])

    for ff, struct, output, in_file in zip(ff_list, struct_list, output_list, in_file_list):
        os.makedirs(output, exist_ok=True)

        log_path = os.path.join(output, 'log.lammps')
        dump_path = os.path.join(output, 'dump.all')
        dump_unwrapped_path = os.path.join(output, 'dump_unwrapped.all')
        traj_path = os.path.join(output, 'md_run.traj')
        traj_unwrapped_path = os.path.join(output, 'md_run_unwrapped.traj')
        for stale_path in (dump_path, dump_unwrapped_path, traj_path, traj_unwrapped_path, log_path):
            if os.path.exists(stale_path):
                os.remove(stale_path)

        lmp.command(f"log {log_path}")
        lmp.command(f"variable dump_file string {dump_path}")
        lmp.command(f"variable dump_unwrapped_file string {dump_unwrapped_path}")
        lmp.command(f"variable ff_filename string {ff}")
        lmp.command(f"variable structure string {struct}")
        lmp.command(f"variable output_dir string {output}")

        elements = get_elements(struct)
        lmp.command(f'variable elements string "{elements}"')

        try:
            lmp.file(in_file)

            frames = ase.io.read(dump_path, index=':', format='lammps-dump-text',
                                  specorder=elements.split())
            ase.io.write(traj_path, frames)
            os.remove(dump_path)

            unwrapped_frames = ase.io.read(dump_unwrapped_path, index=':', format='lammps-dump-text',
                                            specorder=elements.split())
            ase.io.write(traj_unwrapped_path, unwrapped_frames)
            os.remove(dump_unwrapped_path)

            energy = lmp.get_thermo("pe")
            natoms = lmp.get_natoms()
            ids = np.asarray(lmp.numpy.extract_atom("id"))[:natoms]
            forces = np.asarray(lmp.numpy.extract_atom("f"))[:natoms]
            forces = forces[np.argsort(ids)]

            equilibration_rows, production_rows = _parse_log(log_path, thermo_columns)
            equilibrium_average = _equilibrium_window_average(
                production_rows, thermo_columns, equilibrium_window_fraction)

            seed = int(lmp.extract_variable("seed"))

            property_dictionary = {
                'energy': float(energy),
                'seed': seed,
                'equilibration_thermo_first': equilibration_rows[0],
                'equilibration_thermo_last': equilibration_rows[-1],
                'production_thermo_trace': production_rows,
                'equilibrium_window_fraction': equilibrium_window_fraction,
                'equilibrium_average': equilibrium_average,
            }
            property_dictionary['fx'] = [float(forces[i][0]) for i in range(natoms)]
            property_dictionary['fy'] = [float(forces[i][1]) for i in range(natoms)]
            property_dictionary['fz'] = [float(forces[i][2]) for i in range(natoms)]
            property_dictionary['_status'] = 'complete'
        except Exception as e:
            # A genuine recordable outcome (this potential went unstable at
            # this phase/replicate), not necessarily an infra failure --
            # same reasoning as lammps_reaxff_melt_screen.py's own handling.
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
    run_structural_stability(parse_list(sys.argv[1]), parse_list(sys.argv[2]),
                              parse_list(sys.argv[3]), parse_list(sys.argv[4]))
