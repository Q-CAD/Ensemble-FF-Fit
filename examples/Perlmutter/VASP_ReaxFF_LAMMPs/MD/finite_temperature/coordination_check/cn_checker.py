"""
Coordination-number (CN) deviation check for the finite-temperature MD
coordination-stability screen (see run_pipeline.py's
run_check_coordination_stability, which dynamically imports this module by
path via EnsembleFFFit.utilities.general.import_module_from_path, the same
config-driven-swappable-driver convention used for fine_tuning.ff_task/
converge_dft_data.dft_task/MDMatEnsemble's own lammps_task).

Ported from run_pipeline/logic_locations/LAMMPs/cn_checker.py (the original
standalone script), refactored into a plain function
(check_coordination_stability) with explicit keyword arguments instead of
an argparse Namespace, so run_pipeline.py can call it directly -- the
underlying near-neighbor algorithm is pymatgen's CrystalNN.

Per-element deviation aggregation defaults to the MEAN absolute per-site
CN delta (agg='mean'; 'median' and the original 'rms', i.e.
np.linalg.norm, are also available) -- 2026-10, switched away from a pure
RMS/L2 norm over all sites of an element, which let a single borderline
site (e.g. one whose nearest-neighbor count flips due to a CrystalNN/
Voronoi neighbor-inclusion threshold crossing) dominate the whole
per-element number even when every other site was essentially unchanged.
distance_cutoffs is also now exposed (default (0.5, 1), CrystalNN's own
default) -- CrystalNN penalizes neighbor distances beyond
covalent-radius-sum + distance_cutoffs[0] with a smooth cosine taper down
to zero weight at covalent-radius-sum + distance_cutoffs[1]; widening this
window widens that smooth taper (fewer neighbors sitting exactly at the
cliff edge) but does NOT prevent the Voronoi tessellation itself from
discretely adding/dropping a facet, which is a separate, harder-to-avoid
source of jumpiness inherent to any Voronoi-based neighbor list. This is deliberately kept swappable,
not folded into run_pipeline.py itself: per JaxReaxFF_Integration_Plan.md's
own note (2026-09), CrystalNN's default weighted-CN worked reasonably well
at catching a full motif breakdown but wasn't the most descriptive metric,
and a different structural descriptor (or non-default CrystalNN settings)
may replace this later -- doing so means pointing
check_coordination_stability.coordination_task at a new script with a
matching check_coordination_stability(...) entry point, no
run_pipeline.py change.

The CLI entry point (main(), via `python cn_checker.py ...`) is kept for
standalone debugging against a single already-completed run tree, same
usage as the original script.
"""
import argparse
import json
import os
from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np
from tqdm import tqdm
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.io.lammps.data import LammpsData


def aggregate_cn_deltas(deltas, agg='mean'):
    """
    Collapse one element's per-site CN deltas (candidate - reference) into
    a single deviation number. 'mean'/'median' use the mean/median
    absolute per-site delta (robust to a single outlier site flipping its
    CN); 'rms' reproduces the original np.linalg.norm (L2) behavior, which
    lets one outlier site dominate the whole element's reported deviation.
    """
    deltas = np.asarray(deltas, dtype=float)
    if agg == 'rms':
        return float(np.linalg.norm(deltas))
    if agg == 'mean':
        return float(np.mean(np.abs(deltas)))
    if agg == 'median':
        return float(np.median(np.abs(deltas)))
    raise ValueError(f"Unknown agg {agg!r}, expected 'mean', 'median', or 'rms'")


def compute_cn_diff(use_args):
    """Helper function to compute the coordination number difference."""
    cnn, ref_cn_dct, ff_label, md_name, candidate_structure, use_weights, agg = use_args

    site_els = [str(candidate_structure[i].specie.element) for i in range(len(candidate_structure))]
    unique_site_els = list(np.unique(site_els))
    unique_site_dct = {el: None for el in unique_site_els}

    # Assume that the site elements/ordering in the reference and the
    # candidate structure line up index-for-index (both are derived from
    # the exact same starting structure.lmp -- LAMMPS never adds/removes/
    # reorders atoms across an NPT run when read/written with sort_id).
    for uel in unique_site_els:
        matched_els_is = [i for i in range(len(candidate_structure)) if str(candidate_structure[i].specie.element) == uel]
        matched_els_cns = [cnn.get_cn(candidate_structure, i, use_weights=use_weights) for i in matched_els_is]
        ref_els_cns = [ref_cn_dct[md_name][i] for i in matched_els_is]
        unique_site_dct[uel] = aggregate_cn_deltas(np.subtract(matched_els_cns, ref_els_cns), agg=agg)

    return (ff_label, md_name, unique_site_dct)


def get_average_coordination_deviation(use_weights, ref_path_dct, path_dictionary,
                                        distance_cutoffs=(0.5, 1), agg='mean'):
    print('Constructing reference coordination numbers...')
    cnn = CrystalNN(weighted_cn=use_weights, distance_cutoffs=distance_cutoffs)
    ref_cn_dct = {}
    for md_name, structure in ref_path_dct.items():
        ref_cn_dct[md_name] = [cnn.get_cn(structure, site_ind, use_weights=use_weights) for site_ind in range(len(structure))]

    args_list = [
        (cnn, ref_cn_dct, p_dct['ff_label'], p_dct['name'], p_dct['structure'], use_weights, agg)
        for p_dct in path_dictionary.values()
    ]

    print('Performing multiprocessing analysis...')
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap(compute_cn_diff, args_list), total=len(args_list)))

    # {force_field_label: {structure_name: {element: norm_diff}}}
    structure_dct = {}
    for ff_label, md_name, norm_dct in results:
        structure_dct.setdefault(ff_label, {})[md_name] = norm_dct

    return structure_dct


def comparison_paths(run_directory, inputs_directory, check_file, structure, atom_style, oxi_dct):
    """
    Builds (1) ref_dct: {md_name: pymatgen Structure} from every
    structure file found under inputs_directory, and (2) path_dictionary:
    {root: {ff_label, name (md_name), structure}} from every check_file
    found under run_directory.

    md_name is the FULL path of each structure's own directory, relative
    to inputs_directory (e.g. "mp_bulk/bulk_sp/Bi-Se/Bi2Se3/mp-23164/
    volume_1") -- NOT just the leaf directory name. This matters here
    specifically because sample_ft_md_structures' named_paths all happen
    to share the same leaf directory name ("volume_1"), so a leaf-name-only
    key (the original standalone script's own convention, ported
    unchanged at first -- fixed here, 2026-09, once the coordination
    table surfaced it) would silently collide all three named structures
    into one reference entry.

    ff_label is derived from run_directory's own first path segment below
    each check_file's directory (e.g. "reaxff_run_5_bonds_and_angular"),
    with the run_directory's own "structures" segment (mirrored in by
    MDMatEnsemble.build_full_runs from inputs_directory's own leaf dirname
    -- CONFIRMED via a direct build_task_dicts test, 2026-09, not assumed)
    stripped so the remaining path matches ref_dct's own md_name exactly.

    Also returns (3) failures: {ff_label: {md_name: error_message}} for
    every leaf directory that has a properties.json recording
    _status="failed" (written by lammps_reaxff_md.py when the MD run
    itself raised, e.g. "Non-numeric pressure - simulation unstable") but
    no check_file -- a genuinely unstable force field/structure combo, not
    a data-plumbing gap. A leaf with NEITHER check_file NOR properties.json
    is still silently skipped (not yet run / stage not finished), same as
    before -- only a leaf whose own driver explicitly recorded failure is
    reported here.
    """
    inputs_directory = Path(inputs_directory)
    run_directory = Path(run_directory)

    print('Building reference dictionary...')
    ref_dct = {}
    for root, _, _ in os.walk(inputs_directory):
        structure_path = os.path.join(root, structure)
        if os.path.exists(structure_path):
            md_name = str(Path(root).relative_to(inputs_directory))
            ld = LammpsData.from_file(structure_path, atom_style=atom_style, sort_id=True)
            ref_dct[md_name] = ld.structure.add_oxidation_state_by_element(oxi_dct)

    print('Finding comparison paths...')
    path_dictionary = {}
    failures = {}
    for root, _, _ in os.walk(run_directory):
        check_structure_path = os.path.join(root, check_file)
        properties_path = os.path.join(root, 'properties.json')

        if not os.path.exists(check_structure_path) and not os.path.exists(properties_path):
            continue  # not yet run / stage not finished -- nothing to report

        rel_parts = Path(root).relative_to(run_directory).parts
        if len(rel_parts) < 2:
            continue  # not a real (ff_combo, structures, ...) leaf
        ff_label = rel_parts[0]
        # rel_parts[1] is the mirrored "structures" segment (see this
        # function's own docstring); rel_parts[2:] is the same
        # structure-relative path ref_dct's md_name uses.
        md_name = str(Path(*rel_parts[2:])) if len(rel_parts) > 2 else ''

        if not os.path.exists(check_structure_path):
            with open(properties_path) as f:
                props = json.load(f)
            if props.get('_status') == 'failed':
                failures.setdefault(ff_label, {})[md_name] = props.get('error', 'unknown error')
            continue

        if md_name not in ref_dct:
            print(f"  WARNING: no reference structure for {root} (md_name={md_name!r}) -- skipping")
            continue
        ld = LammpsData.from_file(check_structure_path, atom_style=atom_style, sort_id=True)
        path_dictionary[root] = {
            'ff_label': ff_label,
            'name': md_name,
            'structure': ld.structure.add_oxidation_state_by_element(oxi_dct),
        }

    return ref_dct, path_dictionary, failures


def check_coordination_stability(run_directory, inputs_directory, check_file='data.npt_relax',
                                  structure='structure.lmp', atom_style='charge', oxi_dct=None,
                                  use_weights=True, json_file='comparison.json',
                                  distance_cutoffs=(0.5, 1), agg='mean'):
    """
    Entry point run_pipeline.py's run_check_coordination_stability calls.

    run_directory: finite_temperature_md_batch's own run_directory --
    walked for every <ff_combo>/<structure_rel>/check_file (the LAMMPS
    data file written by write_data at the end of the room-temperature
    NPT stage, e.g. data.npt_relax).
    inputs_directory: sample_ft_md_structures' own dest_root (or its
    lammps_inputs_directory) -- walked for every <structure_rel>/structure
    (the pre-MD structure.lmp each named structure started from).
    oxi_dct: REQUIRED (e.g. {"Bi": 3, "Se": -2} for Bi2Se3) -- CrystalNN's
    weighted CN needs oxidation states assigned on both sides for a
    meaningful comparison; there's no safe silent default across
    arbitrary compositions.
    distance_cutoffs: passed straight to CrystalNN (default (0.5, 1), its
    own default) -- see this module's docstring for what widening it does
    and does not fix.
    agg: per-element deviation aggregation, see aggregate_cn_deltas
    ('mean' default, 'median', or 'rms' for the original L2-norm
    behavior).

    Returns (deviation_dct, failures): deviation_dct is {ff_label:
    {md_name: {element: norm_diff}}} (ff_label a bare directory name e.g.
    "reaxff_run_5_bonds_and_angular", md_name the structure's own path
    relative to inputs_directory e.g. "mp_bulk/bulk_sp/Bi-Se/Bi2Se3/
    mp-23164/volume_1" -- see comparison_paths' own docstring); failures
    is {ff_label: {md_name: error_message}} for every MD run that itself
    recorded _status="failed" (see comparison_paths). Both get written to
    json_file as {"deviations": deviation_dct, "failures": failures}.
    """
    if not oxi_dct:
        raise ValueError("check_coordination_stability requires oxi_dct (e.g. {'Bi': 3, 'Se': -2})")

    ref_dct, path_dictionary, failures = comparison_paths(run_directory, inputs_directory, check_file,
                                                           structure, atom_style, oxi_dct)
    deviation_dct = get_average_coordination_deviation(use_weights, ref_dct, path_dictionary,
                                                        distance_cutoffs=distance_cutoffs, agg=agg)

    os.makedirs(os.path.dirname(json_file) or '.', exist_ok=True)
    with open(json_file, 'w') as f:
        json.dump({'deviations': deviation_dct, 'failures': failures}, f, indent=4)

    return deviation_dct, failures


def main():
    parser = argparse.ArgumentParser(description="Standalone CLI for check_coordination_stability (see this module's own docstring)")
    parser.add_argument("--run_directory", "-rd", help="Path to the run directory tree", default='run_directory')
    parser.add_argument("--inputs_directory", "-id", help="Path to input file directory", default='inputs_directory')
    parser.add_argument("--check_file", "-cf", help="Name of LAMMPs data file to check for in the --run_directory", default='data.npt_relax')
    parser.add_argument("--structure", "-s", type=str, help="Name of the .lmp file", default='structure.lmp')
    parser.add_argument("--json_file", "-jf", type=str, help="Name of the .json file", default='comparison.json')
    parser.add_argument("--atom_style", "-as", help="LAMMPs structure file atom style", type=str, default='charge')
    parser.add_argument("--oxi_dct", "-od", help="Pymatgen oxidation dictionary in .json format, e.g., '{\"Bi\":3,\"Se\":-2}'", type=json.loads)
    parser.add_argument("--use_weights", "-uw", help="Use weights for pymatgen's CN analysis", type=bool, default=True)
    parser.add_argument("--distance_cutoffs", "-dc", help="CrystalNN distance_cutoffs as two floats", type=float, nargs=2, default=(0.5, 1))
    parser.add_argument("--agg", "-a", help="Per-element deviation aggregation: mean, median, or rms", type=str, default='mean')
    args = parser.parse_args()

    check_coordination_stability(
        run_directory=args.run_directory, inputs_directory=args.inputs_directory,
        check_file=args.check_file, structure=args.structure, atom_style=args.atom_style,
        oxi_dct=args.oxi_dct, use_weights=args.use_weights, json_file=args.json_file,
        distance_cutoffs=tuple(args.distance_cutoffs), agg=args.agg,
    )


if __name__ == '__main__':
    main()
