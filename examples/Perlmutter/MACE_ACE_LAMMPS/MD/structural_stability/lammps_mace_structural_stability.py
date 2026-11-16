"""
Driver for the room-temperature structural-stability check, MACE-via-LAMMPS
variant -- same NPT protocol/dump/log-marker machinery as
lammps_ace_structural_stability.py (see that module's own docstring), run
against the SAME structure.lmp/replicate seeds via in.structural_stability_mace
(see run_build_structural_stability_inputs), but pair_style mliap unified
instead of pair_style pace. Exists as a REFERENCE POINT: does the original
MACE foundation model (the one that labeled every bit of ACE's own training
data) preserve PdSe/Pd2Se3/Pd3Se10's known structural motifs under this same
protocol, or does ACE's own apparent breakdown of e.g. Pd2Se3's vdW layering
just reflect a generically-too-aggressive NPT setup rather than something
specific to ACE's own potential-energy surface?

Confirmed (2026-10-19) this container's LAMMPS has ML-IAP (Kokkos/Python-
coupled "unified" interface) compiled in via a direct symbol check against
liblammps.so.0 -- no native pair_style mace exists here. Adapted from
/pscratch/sd/r/rym/efffit_example/run_pipeline's own lammps_mace_md.py/
in.nvt_mace (that pipeline's own confirmed-working MACE-via-mliap recipe),
combined with lammps_ace_structural_stability.py's own two-dump-stream/
marker-split machinery (that reference only needed one dump and no
equilibration-stage drift capture, since it wasn't doing this kind of
structural-stability check).

Genuinely different from lammps_ace_structural_stability.py, not just a
renamed copy:

1. torch and mace.calculators.lammps_mliap_mace MUST be imported before
   LAMMPS/Kokkos starts (confirmed in the reference driver this is adapted
   from -- LAMMPS's own linked libraries otherwise get loaded first and
   shadow the Python packages' bundled copies: libnccl ->
   "libtorch_cuda.so: undefined symbol: ncclDevCommCreate" at
   activate_mliappy_kokkos, libhdf5 -> unpickling the model at `pair_style
   mliap unified` imports mace.data -> h5py -> "ValueError: Not a
   datatype"). Both reasons are import-order bugs, not Kokkos-cmdargs
   issues -- the cmdargs themselves are otherwise the same shape as
   lammps_ace_structural_stability.py's own _KOKKOS_CMDARGS.

2. import_lammps_mliap(lammps) + lammps.mliap.activate_mliappy_kokkos(lmp)
   must run once, right after the lammps.lammps() instance is created and
   before any `lmp.file(...)` call -- the actual ML-IAP/Kokkos activation
   step `pair_style pace`'s plain Kokkos cmdargs don't need.

3. The model file itself (ff_filename here) must already be in MLIAP format
   (a `*.model-mliap_lammps.pt` produced by `mace_create_lammps_model
   <model> --format=mliap`, NOT the original ASE/.model checkpoint) -- see
   run_stage_mace_reference_force_field's own docstring for the conversion
   step. `pair_style mliap unified ${ff_filename} 0` takes that file
   directly; `pair_coeff * * ${elements}` carries no filename of its own
   (contrast pair_style pace's `pair_coeff * * ${ff_filename} ${elements}`).
"""
import os
import re
import sys
import json

# MUST precede any lammps/Kokkos import -- see this module's own docstring.
import torch  # noqa: F401
import mace.calculators.lammps_mliap_mace  # noqa: F401

import numpy as np
import ase.io

from EnsembleFFFit.molecular_dynamics.helpers import parse_list, get_elements, import_lammps, import_lammps_mliap

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# Same shape as lammps_ace_structural_stability.py's own _KOKKOS_CMDARGS,
# just "newton on neigh half" ordering to match the confirmed-working
# reference this was adapted from (LAMMPS -pk keyword/value pairs are
# order-independent, so this is cosmetic, but kept identical to the
# confirmed recipe rather than reordered without re-testing).
_KOKKOS_CMDARGS = ["-k", "on", "g", "1", "-sf", "kk", "-pk", "kokkos", "newton", "on", "neigh", "half"]

_PRODUCTION_MARKER = "===PRODUCTION_START==="
_THERMO_HEADER_RE = re.compile(r'^\s*(Step\s+\S.*)$', re.MULTILINE)


def _parse_thermo_blocks(text, columns):
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
                continue
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
    with open(log_path) as f:
        text = f.read()

    marker_idx = text.rfind(_PRODUCTION_MARKER)
    if marker_idx == -1:
        raise RuntimeError(f"{log_path}: no {_PRODUCTION_MARKER!r} marker found -- "
                            f"in.structural_stability_mace should print it immediately before the production run")

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


def run_mace_structural_stability(ff_list, struct_list, output_list, in_file_list,
                                   thermo_columns=('Step', 'Temp', 'PotEng', 'TotEng', 'Press',
                                                   'Lx', 'Ly', 'Lz', 'Volume', 'Density'),
                                   equilibrium_window_fraction=0.2):
    n = len(ff_list)
    assert all(len(lst) == n for lst in (struct_list, output_list, in_file_list)), \
        "All lists must be same length"

    lammps = import_lammps()
    import_lammps_mliap(lammps)
    lmp = lammps.lammps(cmdargs=_KOKKOS_CMDARGS + ["-log", "none", "-screen", "none"])
    lammps.mliap.activate_mliappy_kokkos(lmp)

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
    run_mace_structural_stability(parse_list(sys.argv[1]), parse_list(sys.argv[2]),
                                   parse_list(sys.argv[3]), parse_list(sys.argv[4]))
