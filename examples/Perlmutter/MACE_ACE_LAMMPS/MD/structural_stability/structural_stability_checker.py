"""
Room-temperature structural-stability check for MD/finite_temperature/
structural_stability -- adapted from MatEnsemble/melting_pipeline/MD/
single_melt/melt_checker.py, dropping everything that was specific to a
temperature-grid melting-point SCREEN (there is no temperature grid here --
just one 300K condition per (ff_label, phase, replicate)), and combines a
structural (RDF long-range-peak retention + z-density-profile modulation,
via ovito) and dynamical (MSD on the unwrapped trajectory) signal, plus a
direct equilibration-stage PE/TotEng drift check, into one per-(ff_label,
phase) summary row, replicate-aggregated.

Dynamically imported by run_pipeline.py's check_structural_stability,
the same config-swappable convention as melt_checker.py's own
check_melting_screening (stability_task/entry_point in workflow_config.yaml)
-- a different/extended structural metric is a config change, not a
run_pipeline.py change.

PIPELINE-SPECIFIC BY DELIBERATE CHOICE, not an oversight: this needs
`ovito` (RadialDistributionFunctionModifier, see _rdf_long_range_peak
below), which stays OUT of Ensemble-FF-Fit core on purpose (kept as this
example's own dependency in Dockerfile.ace, pinned to the version
confirmed working in this project's own container -- EFF core supports
multiple FF backends and shouldn't carry a dependency only one example's
own structural check needs).

RELATED PRECEDENT, not unified with this pass: examples/Perlmutter/
VASP_ReaxFF_LAMMPs/MD/finite_temperature/coordination_check/cn_checker.py
implements an analogous but genuinely different structural-stability
concept for ReaxFF -- coordination-number deviation (pymatgen's CrystalNN)
per element, vs. this module's RDF/z-density/MSD signals -- used there as
an explicit go/no-go GATE on top of that example's own combined RMSE
ranking, not folded into the score, the same two-stage design (validation
score + separate structural gate) this example uses. The two aren't
unified into one shared function here: the underlying physics is too
different (coordination topology vs. radial/density/dynamical signals) to
force through one interface without guessing at a premature abstraction
from a single additional data point. Worth factoring a shared tree-
walking/reference-comparison scaffold out of EnsembleFFFit.analysis once a
THIRD backend's own structural check exists to generalize from -- not
attempted with only these two.

Per-signal design notes:

- Structural (RDF + z-density profile) and dynamical (MSD): computed from
  lammps_ace_structural_stability.py's own md_run.traj/md_run_unwrapped.traj
  -- UNLIKE melt_checker.py's own documented approximation (its dump
  streams ran for the WHOLE script, mixing minimize/equilibration frames
  into the production-window signal, since a step-based split of one
  continuous stream was fragile against minimize's own non-fixed iteration
  count), in.structural_stability (see run_build_structural_stability_inputs
  in run_pipeline.py) only starts dumping AFTER the equilibration stage,
  immediately before production -- so every frame in these two files is
  already production-only.

- RDF uses ovito's RadialDistributionFunctionModifier (confirmed present,
  3.15.0, same as the reference) via ovito.io.ase.ase_to_ovito, one frame
  at a time, manually averaged across each window. The z-density-profile
  modulation (QL/vdW-gap-style layering, relevant for Pd2Se3) uses the same
  per-window-averaged frames: a plain numpy histogram of wrapped
  z-coordinates, std-of-bin-counts as the modulation-amplitude proxy.

  TWO baselines are reported for both signals, not one -- a real bug found
  (2026-10-19) via direct visual inspection: the ORIGINAL design only
  compared each run's own LATE production window against its own EARLY
  production window ("self" baseline below), on the reasoning that
  dumping only starts after equilibration makes that comparison exact (no
  approximation needed, unlike melt_checker.py's own cross-temperature
  baseline). That reasoning is still correct as far as it goes, but it has
  a blind spot: if a phase's real structural motifs (e.g. Pd2Se3's vdW
  layering) collapse DURING the 5ps pre-production equilibration stage --
  plausible, and confirmed by eye to actually be happening here -- the
  "early window" baseline is ALREADY collapsed by the time dumping starts,
  so self-vs-self reports "no further change" (~1.0) even though the true
  starting structure was completely different. Comparing against the
  PRISTINE reference structure (reference_structure, read once per phase
  from inputs_directory/structures/<phase_label>/structure_filename -- the
  exact structure.lmp shared across every potential/replicate for that
  phase) catches this; self-vs-self is kept alongside it only as a
  complementary "is it still actively drifting at the very end of
  production" signal, not as the primary degradation check anymore.

  A THIRD comparison -- Min.data (this run's own post-minimization,
  pre-equilibration structure, written by the driver before any thermal MD
  runs at all) vs. the same pristine reference -- is also reported, per
  explicit request: does minimization ALONE (relaxing the shared DFT/MACE-
  derived structure under this potential's own 0K energy landscape, no
  thermal motion involved yet) already distort known bonding environments,
  independent of anything NPT/300K does afterward? This is diagnostic only
  for now (not folded into any pass/fail judgment) -- worth a close look
  once a given potential's production-stage numbers look reasonable, to
  make sure minimization itself isn't already the actual source of
  whatever structural change shows up.

- Dynamical (MSD): from md_run_unwrapped.traj's own LAST
  equilibrium_window_fraction of frames (true unwrapped positions, so a
  boundary-crossing atom doesn't look like it teleported). Reference frame
  is this window's own first frame; MSD(t) is the atom-averaged squared
  displacement from there, with a line fit to only the latter
  msd_fit_fraction (default 0.5) of that curve -- a solid gives a
  near-zero slope with a poor linear fit (saturating), a liquid/disordered
  structure a clearly positive slope with a good linear fit. Both the
  fitted slope (Angstrom^2/ps) and the fit's R^2 are reported.

- Thermodynamic (equilibration-stage energy drift): properties.json's own
  equilibration_thermo_first/equilibration_thermo_last give PE/atom and
  TotEng/atom at the start and end of the short pre-production NPT
  equilibration stage directly -- a literal answer to "does timestep_fs
  cause integrator energy drift" over that stage, reported per run (not
  just replicate-averaged in the CSV) since a single badly-drifting
  replicate is itself the finding, not something to average away.
"""
import csv
import json
import os
from pathlib import Path

import numpy as np
import ase.io
import ase.data

from EnsembleFFFit.molecular_dynamics.helpers import get_elements


def _read_lammps_structure(path):
    """
    Reads a LAMMPS data file (structure.lmp or a driver-written Min.data,
    both atom_style atomic) as an ASE Atoms object, with the correct
    atomic numbers per LAMMPS atom type -- ase.io's own lammps-data reader
    needs an explicit Z_of_type mapping (it can't infer element identity
    from atom_style atomic alone, which carries no species information
    besides a bare integer type), derived here from get_elements' own
    atom-type-ordered element string (the same helper every LAMMPS driver
    in this pipeline already uses for pair_coeff ordering, so this is
    guaranteed to agree with however the file was actually written).
    """
    elements = get_elements(str(path)).split()
    Z_of_type = {i + 1: ase.data.atomic_numbers[el] for i, el in enumerate(elements)}
    # style='atomic' explicitly -- Min.data (written by LAMMPS's own
    # write_data, unlike structure.lmp which this project writes directly)
    # carries an "Atoms # atomic/kk" style comment (Kokkos-suffixed, from
    # -sf kk auto-suffixing atom_style at runtime) that ASE's own auto-
    # detection doesn't recognize ("Style atomic/kk not supported") --
    # bypass that auto-detection entirely by naming the style directly,
    # since every LAMMPS data file this pipeline writes is atom_style
    # atomic regardless of any Kokkos suffix.
    return ase.io.read(str(path), format='lammps-data', Z_of_type=Z_of_type, style='atomic')


def _iter_completed_runs(run_directory):
    """
    Walk run_directory for every leaf with a properties.json (written by
    lammps_ace_structural_stability.py), parsing (ff_label, phase_label,
    replicate) out of its own path -- run_directory/recipes/replicate_<k>/
    <ff_label...>/structures/<phase_label>/ (same MDMatEnsemble
    _modify_single_run_path path-shape convention as the melting_pipeline
    reference's own single-melt tree, just missing the reference's extra
    T<temp>K segment since there's no temperature grid here). Returns
    (runs, failures); failures is {ff_label: {"replicate_<k>/<phase_label>":
    error_message}} for every leaf whose properties.json recorded
    _status="failed".
    """
    run_directory = Path(run_directory)
    failures = {}
    runs = []
    for root, _, files in os.walk(run_directory):
        if 'properties.json' not in files:
            continue
        root = Path(root)
        parts = root.relative_to(run_directory).parts
        if len(parts) < 4 or parts[0] != 'recipes' or 'structures' not in parts:
            continue  # not a real (recipe, ff_label, structure) leaf
        replicate_part = parts[1]
        struct_idx = parts.index('structures')
        ff_label = '/'.join(parts[2:struct_idx])
        phase_label = '/'.join(parts[struct_idx + 1:])

        with open(root / 'properties.json') as f:
            props = json.load(f)
        if props.get('_status') != 'complete':
            if props.get('_status') == 'failed':
                failures.setdefault(ff_label, {})[f"{replicate_part}/{phase_label}"] = props.get('error', 'unknown error')
            continue

        runs.append({
            'ff_label': ff_label,
            'phase_label': phase_label,
            'replicate': replicate_part,
            'path': root,
            'properties': props,
        })
    return runs, failures


def _rdf_long_range_peak(frames, cutoff, bins, far_fraction=0.3):
    from ovito.io.ase import ase_to_ovito
    from ovito.pipeline import Pipeline, StaticSource
    from ovito.modifiers import RadialDistributionFunctionModifier

    r = None
    g_sum = None
    for atoms in frames:
        data = ase_to_ovito(atoms)
        pipeline = Pipeline(source=StaticSource(data=data))
        pipeline.modifiers.append(RadialDistributionFunctionModifier(cutoff=cutoff, number_of_bins=bins))
        rdf = pipeline.compute().tables['coordination-rdf'].xy()
        if r is None:
            r = rdf[:, 0]
            g_sum = rdf[:, 1].copy()
        else:
            g_sum += rdf[:, 1]
    g_mean = g_sum / len(frames)
    far_mask = r >= far_fraction * cutoff
    peak_height = float(g_mean[far_mask].max()) if far_mask.any() else float(g_mean.max())
    return peak_height


def _z_modulation_amplitude(frames, n_bins=100):
    """Std of a wrapped-z-coordinate histogram, averaged across frames --
    a layered/motif-bearing crystal gives a high value; a disordered,
    homogenized structure gives a low one."""
    amps = []
    for atoms in frames:
        c = atoms.cell[2, 2]
        z = atoms.get_positions()[:, 2] % c
        hist, _ = np.histogram(z, bins=n_bins, range=(0, c))
        amps.append(float(hist.std()))
    return float(np.mean(amps))


def _msd_fit(frames_unwrapped, msd_fit_fraction):
    """MSD(t) (numpy, atom-averaged squared displacement from this
    window's own first frame) plus a linear fit to the latter
    msd_fit_fraction of that curve -- see this module's own docstring."""
    ref = frames_unwrapped[0].get_positions()
    msd = np.array([float(np.mean(np.sum((atoms.get_positions() - ref) ** 2, axis=1)))
                    for atoms in frames_unwrapped])

    n = len(msd)
    fit_start = max(1, int(round(n * (1 - msd_fit_fraction))))
    if n - fit_start < 2:
        return 0.0, 0.0
    x = np.arange(fit_start, n, dtype=float)
    y = msd[fit_start:]
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(slope), float(r_squared)


def check_structural_stability(run_directory, inputs_directory, structure_filename='structure.lmp',
                                equilibrium_window_fraction=0.2, rdf_cutoff=10.0,
                                rdf_bins=200, rdf_far_fraction=0.3, msd_fit_fraction=0.5,
                                dump_interval_ps=0.1, diffusive_slope_threshold=0.01,
                                diffusive_r_squared_threshold=0.8,
                                csv_path='structural_stability_summary.csv'):
    """
    Entry point run_pipeline.py's run_check_structural_stability calls.
    inputs_directory/structure_filename locate each phase's own pristine
    reference structure (inputs_directory/structures/<phase_label>/
    structure_filename, written once by build_structural_stability_inputs,
    shared across every potential/replicate for that phase) -- see this
    module's own docstring for why that reference, not just each run's own
    early-production window, is now the primary baseline.
    """
    runs, failures = _iter_completed_runs(run_directory)
    if not runs:
        raise RuntimeError(f"No completed structural-stability runs found under {run_directory}")

    reference_cache = {}

    def _reference(phase_label):
        if phase_label not in reference_cache:
            ref_path = Path(inputs_directory) / 'structures' / phase_label / structure_filename
            reference_cache[phase_label] = _read_lammps_structure(ref_path)
        return reference_cache[phase_label]

    reference_rdf_cache = {}
    reference_z_cache = {}

    def _reference_rdf(phase_label):
        if phase_label not in reference_rdf_cache:
            reference_rdf_cache[phase_label] = _rdf_long_range_peak(
                [_reference(phase_label)], rdf_cutoff, rdf_bins, rdf_far_fraction)
        return reference_rdf_cache[phase_label]

    def _reference_z(phase_label):
        if phase_label not in reference_z_cache:
            reference_z_cache[phase_label] = _z_modulation_amplitude([_reference(phase_label)])
        return reference_z_cache[phase_label]

    for r in runs:
        traj = ase.io.read(r['path'] / 'md_run.traj', index=':')
        unwrapped = ase.io.read(r['path'] / 'md_run_unwrapped.traj', index=':')
        n_window = max(1, int(round(len(traj) * equilibrium_window_fraction)))
        n_window_u = max(1, int(round(len(unwrapped) * equilibrium_window_fraction)))

        # Every frame here is already production-only (see this module's
        # own docstring). early_window/late_window still feed the
        # complementary self-vs-self "still drifting at the very end"
        # signal; reference_rdf/reference_z (outside the loop, cached per
        # phase) are the PRIMARY baseline now.
        early_window = traj[:n_window]
        late_window = traj[-n_window:]
        late_window_u = unwrapped[-n_window_u:]

        eq = r['properties'].get('equilibrium_average', {})
        natoms = len(traj[0])
        pe_total = eq.get('PotEng', r['properties']['energy'])

        r['pe_per_atom'] = pe_total / natoms
        r['volume_per_atom'] = eq.get('Volume', float('nan')) / natoms

        ref_rdf = _reference_rdf(r['phase_label'])
        ref_z = _reference_z(r['phase_label'])
        late_rdf = _rdf_long_range_peak(late_window, rdf_cutoff, rdf_bins, rdf_far_fraction)
        late_z = _z_modulation_amplitude(late_window)
        r['rdf_vs_reference_fraction'] = late_rdf / ref_rdf if ref_rdf else float('nan')
        r['z_vs_reference_fraction'] = late_z / ref_z if ref_z else float('nan')

        early_rdf = _rdf_long_range_peak(early_window, rdf_cutoff, rdf_bins, rdf_far_fraction)
        early_z = _z_modulation_amplitude(early_window)
        r['rdf_self_fraction'] = late_rdf / early_rdf if early_rdf else float('nan')
        r['z_self_fraction'] = late_z / early_z if early_z else float('nan')

        # Diagnostic only (see this module's own docstring) -- does
        # minimization ALONE, before any thermal MD, already distort the
        # shared reference structure under this potential's own 0K energy
        # landscape?
        min_data_path = r['path'] / 'Min.data'
        if min_data_path.is_file():
            min_atoms = _read_lammps_structure(min_data_path)
            min_rdf = _rdf_long_range_peak([min_atoms], rdf_cutoff, rdf_bins, rdf_far_fraction)
            min_z = _z_modulation_amplitude([min_atoms])
            r['rdf_after_minimize_vs_reference_fraction'] = min_rdf / ref_rdf if ref_rdf else float('nan')
            r['z_after_minimize_vs_reference_fraction'] = min_z / ref_z if ref_z else float('nan')
        else:
            r['rdf_after_minimize_vs_reference_fraction'] = float('nan')
            r['z_after_minimize_vs_reference_fraction'] = float('nan')

        slope, r_squared = _msd_fit(late_window_u, msd_fit_fraction)
        r['msd_slope_A2_per_ps'] = slope / dump_interval_ps
        r['msd_r_squared'] = r_squared
        r['msd_diffusive'] = bool(slope / dump_interval_ps > diffusive_slope_threshold
                                   and r_squared > diffusive_r_squared_threshold)

        eq_first = r['properties']['equilibration_thermo_first']
        eq_last = r['properties']['equilibration_thermo_last']
        r['equilibration_pe_drift_per_atom'] = (eq_last['PotEng'] - eq_first['PotEng']) / natoms
        r['equilibration_totE_drift_per_atom'] = (eq_last['TotEng'] - eq_first['TotEng']) / natoms

    groups = {}
    for r in runs:
        groups.setdefault((r['ff_label'], r['phase_label']), []).append(r)

    def _group_mean(group_runs, key):
        return float(np.mean([r[key] for r in group_runs]))

    summary_rows = []
    for (ff_label, phase_label), group_runs in sorted(groups.items()):
        msd_slopes = [r['msd_slope_A2_per_ps'] for r in group_runs]
        msd_r2s = [r['msd_r_squared'] for r in group_runs]

        summary_rows.append({
            'ff_label': ff_label,
            'phase_label': phase_label,
            'n_replicates': len(group_runs),
            'pe_per_atom_eV': _group_mean(group_runs, 'pe_per_atom'),
            'pe_per_atom_std': float(np.std([r['pe_per_atom'] for r in group_runs])),
            'volume_per_atom_A3': _group_mean(group_runs, 'volume_per_atom'),
            'volume_per_atom_std': float(np.std([r['volume_per_atom'] for r in group_runs])),
            'rdf_vs_reference_fraction_mean': _group_mean(group_runs, 'rdf_vs_reference_fraction'),
            'z_vs_reference_fraction_mean': _group_mean(group_runs, 'z_vs_reference_fraction'),
            'rdf_self_fraction_mean': _group_mean(group_runs, 'rdf_self_fraction'),
            'z_self_fraction_mean': _group_mean(group_runs, 'z_self_fraction'),
            'rdf_after_minimize_vs_reference_fraction_mean': _group_mean(group_runs, 'rdf_after_minimize_vs_reference_fraction'),
            'z_after_minimize_vs_reference_fraction_mean': _group_mean(group_runs, 'z_after_minimize_vs_reference_fraction'),
            'msd_slope_A2_per_ps_mean': float(np.mean(msd_slopes)),
            'msd_r_squared_mean': float(np.mean(msd_r2s)),
            'fraction_replicates_diffusive': float(np.mean([r['msd_diffusive'] for r in group_runs])),
            'equilibration_pe_drift_per_atom_eV_mean': _group_mean(group_runs, 'equilibration_pe_drift_per_atom'),
            'equilibration_totE_drift_per_atom_eV_mean': _group_mean(group_runs, 'equilibration_totE_drift_per_atom'),
            'equilibration_totE_drift_per_atom_eV_max_abs': float(max(abs(r['equilibration_totE_drift_per_atom']) for r in group_runs)),
        })

    os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    return summary_rows, failures


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Standalone CLI for check_structural_stability (see this module's own docstring)")
    parser.add_argument("--run_directory", "-rd", default='run_directory')
    parser.add_argument("--inputs_directory", "-id", required=True)
    parser.add_argument("--csv_path", "-o", default='structural_stability_summary.csv')
    parser.add_argument("--equilibrium_window_fraction", type=float, default=0.2)
    parser.add_argument("--rdf_cutoff", type=float, default=10.0)
    parser.add_argument("--rdf_bins", type=int, default=200)
    parser.add_argument("--msd_fit_fraction", type=float, default=0.5)
    parser.add_argument("--dump_interval_ps", type=float, default=0.1)
    args = parser.parse_args()

    check_structural_stability(
        run_directory=args.run_directory, inputs_directory=args.inputs_directory, csv_path=args.csv_path,
        equilibrium_window_fraction=args.equilibrium_window_fraction,
        rdf_cutoff=args.rdf_cutoff, rdf_bins=args.rdf_bins,
        msd_fit_fraction=args.msd_fit_fraction, dump_interval_ps=args.dump_interval_ps,
    )
