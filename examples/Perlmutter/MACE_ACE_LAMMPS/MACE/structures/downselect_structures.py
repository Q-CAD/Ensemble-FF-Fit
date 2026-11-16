"""
General-purpose structure screening/downselection, generalized from the
companion Pd-Se pipeline's own tier-2-specific
MACE/finite_temperature/generate_structures/downselect_low_energy.py.

That version only ever fed one particular training tier and grouped
structures by hardcoding a 3-level directory-path convention
(<composition>/<trajectory-subpath>/<frame>). Neither limitation is
fundamental: ACE fitting is sensitive to outliers in MACE-labeled data
(blown-up/over-stabilized configurations), and that sensitivity isn't
specific to any one tier -- any already-computed MD/single-point output
can benefit from the same screening. This version takes an arbitrary
`training_root` and groups by explicit METADATA fields (read from each
leaf's own metadata.json sidecar, written by run_single_points -- see
run_pipeline.py's own run_single_points docstring) instead of path
structure, so it isn't tied to any one tier's directory layout.

Reuses EnsembleFFFit.potential.ace.build_ace_dataset.build_ace_dataframe/
compute_validation_floor DIRECTLY rather than a hand-rolled, weaker
ranking -- important, not cosmetic: naively ranking raw single-point
energies by "lowest energy" would otherwise preferentially select exactly
the kind of foundation-model over-stabilization artifact the
stability-floor filter exists to screen out elsewhere. Applying the SAME
blow-up/stability-floor filters here means this downselection can only
ever pick from structures that would also have survived build_ff_inputs'
own dataset-quality screening.

`rank_by`: only 'energy_per_atom' is implemented for now (matches this
project's own first-pass choice to keep screening to low-temperature
stability, not build out multiple criteria yet) -- passing anything else
raises ValueError rather than silently falling back, so a future
force-magnitude or structural criterion is an explicit, visible addition
here, not a silent no-op.

`group_by`: a priority-ordered list of metadata keys. The FIRST key is the
dedup level (keep only each group's own best-ranked representative); the
REMAINING keys (if any) are the final selection level (keep the top
`n_per_group` representatives within each of those groups). E.g.
group_by=['trajectory_id', 'stoichiometry'] reproduces the original
tier-2 behavior (one winner per trajectory, then top-N of those per
composition) -- but any metadata field(s) work, not just those two.
group_by with only one entry skips the dedup step and goes straight to
top-N-per-group.

`rescale_output`: optional {'volume_fractions': [...]} -- if given, chains
the same isotropic rescale_volume used elsewhere in this example as a
post-step, writing len(volume_fractions) copies per selected structure (as
tier 2 does, to seed a new MD tier at different starting volumes). If
omitted (the default), this stage just writes the selected structures
unscaled -- the plain "screen out outliers" use case.
"""
import json
from pathlib import Path

from pymatgen.core import Structure


def rescale_volume(structure, volume_fraction):
    """Isotropic cell rescale -- same convention used by every generate_* driver in this example."""
    scaling = volume_fraction ** (1 / 3)
    new_matrix = structure.lattice.matrix * scaling
    return Structure(lattice=new_matrix, species=structure.species, coords=structure.frac_coords,
                      coords_are_cartesian=False)


def _read_metadata(leaf_dir, metadata_filename):
    metadata_path = leaf_dir / metadata_filename
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"{metadata_path} does not exist -- downselect_structures needs every leaf's own "
            f"metadata.json sidecar (written by run_single_points) to group by metadata fields.")
    with open(metadata_path) as f:
        return json.load(f)


def downselect_structures(training_root, isolated_elements_directory, validation_root, output_root,
                           group_by, n_per_group, rank_by='energy_per_atom',
                           rescale_output=None,
                           structure_filename="POSCAR", properties_filename="properties.json",
                           metadata_filename="metadata.json",
                           max_abs_energy_per_atom=50.0, max_abs_force_component=50.0,
                           stability_floor_margin=0.5):
    """
    Screens every structure under training_root with the same blow-up/
    stability-floor filters build_ff_inputs uses, ranks survivors by
    `rank_by` within the groups `group_by` defines (see this module's own
    docstring for the two-level grouping convention), keeps the best
    `n_per_group` per final group, and writes each selected structure
    (optionally rescaled at several volume fractions, see `rescale_output`)
    under output_root.

    Returns {"selected": int, "written": [str, ...], "summary":
    {group_key: {"available": int, "selected": int}}}.
    """
    from EnsembleFFFit.potential.ace.build_ace_dataset import build_ace_dataframe, compute_validation_floor
    from EnsembleFFFit.potential.mace.build_ensemble_inputs import get_isolated_atom_e0s

    if rank_by != 'energy_per_atom':
        raise ValueError(
            f"downselect_structures: rank_by={rank_by!r} is not implemented yet -- only "
            f"'energy_per_atom' is supported in this first pass (see this module's own docstring).")
    if not group_by:
        raise ValueError("downselect_structures: group_by must be a non-empty list of metadata keys.")

    e0s = get_isolated_atom_e0s(isolated_elements_directory, properties_filename=properties_filename)
    if not e0s:
        raise FileNotFoundError(
            f"No isolated-atom reference energies found under {isolated_elements_directory} -- "
            f"has the isolated-atom single-point calculation been run yet?"
        )

    stability_floor_per_atom = compute_validation_floor(
        validation_root, e0s, structure_filename=structure_filename,
        properties_filename=properties_filename,
        max_abs_energy_per_atom=max_abs_energy_per_atom,
        max_abs_force_component=max_abs_force_component,
    )
    print(f"downselect_structures: stability floor from {validation_root}: "
          f"{stability_floor_per_atom:.4g} eV/atom (margin {stability_floor_margin:.3g} eV/atom)")

    df = build_ace_dataframe(
        training_root, e0s, structure_filename=structure_filename,
        properties_filename=properties_filename,
        max_abs_energy_per_atom=max_abs_energy_per_atom,
        max_abs_force_component=max_abs_force_component,
        stability_floor_per_atom=stability_floor_per_atom,
        stability_floor_margin=stability_floor_margin,
        include_path=True,
    )
    counts = df.attrs['filter_counts']
    print(f"downselect_structures: {counts['kept']}/{counts['considered']} structure(s) survived the "
          f"same blow-up/stability-floor filters build_ff_inputs uses "
          f"({counts['blowup_outliers']} blow-up, {counts['below_stability_floor']} below floor)")

    dedup_key, *final_keys = group_by

    # Level 1 (dedup): one winner (lowest energy_per_atom) per dedup_key value.
    best_per_dedup_group = {}  # dedup_value -> (epa, leaf_dir, metadata)
    for _, row in df.iterrows():
        leaf_dir = Path(row['path']).resolve()
        metadata = _read_metadata(leaf_dir, metadata_filename)
        dedup_value = metadata.get(dedup_key)
        if dedup_value is None:
            raise KeyError(f"{leaf_dir / metadata_filename} has no '{dedup_key}' field (group_by[0]).")

        epa = row['energy_corrected'] / len(row['ase_atoms'])
        if dedup_value not in best_per_dedup_group or epa < best_per_dedup_group[dedup_value][0]:
            best_per_dedup_group[dedup_value] = (epa, leaf_dir, metadata)

    # Level 2 (final selection): top n_per_group per tuple(final_keys), or
    # per dedup_key itself if group_by had only one entry.
    def _final_key(metadata):
        if final_keys:
            return tuple(metadata.get(k) for k in final_keys)
        return None  # single-level grouping: everything is one group

    by_final_group = {}
    for epa, leaf_dir, metadata in best_per_dedup_group.values():
        by_final_group.setdefault(_final_key(metadata), []).append((epa, leaf_dir))

    output_root = Path(output_root)
    written = []
    summary = {}
    for group_key, entries in sorted(by_final_group.items(), key=lambda kv: str(kv[0])):
        entries.sort(key=lambda e: e[0])
        selected = entries[:n_per_group]
        summary[str(group_key)] = {'available': len(entries), 'selected': len(selected)}

        for rank, (epa, leaf_dir) in enumerate(selected):
            structure = Structure.from_file(leaf_dir / structure_filename)
            group_label = "_".join(str(g) for g in group_key) if group_key is not None else "all"
            if rescale_output and rescale_output.get('volume_fractions'):
                for vf in rescale_output['volume_fractions']:
                    rescaled = rescale_volume(structure, vf)
                    pct_label = f"{round(vf * 100):d}pct"
                    out_dir = output_root / group_label / str(rank) / pct_label
                    out_dir.mkdir(parents=True, exist_ok=True)
                    rescaled.to(filename=str(out_dir / structure_filename), fmt="poscar")
                    written.append(str(out_dir))
            else:
                out_dir = output_root / group_label / str(rank)
                out_dir.mkdir(parents=True, exist_ok=True)
                structure.to(filename=str(out_dir / structure_filename), fmt="poscar")
                written.append(str(out_dir))

    n_selected = sum(s['selected'] for s in summary.values())
    print(f"downselect_structures: selected {n_selected} structure(s) across {len(summary)} "
          f"group(s) (grouped by {group_by}), wrote {len(written)} output structure(s) under {output_root}")
    for group_key, s in sorted(summary.items()):
        flag = "" if s['selected'] == n_per_group else f" (capped -- only {s['available']} available)"
        print(f"  {group_key}: {s['selected']}/{n_per_group} selected{flag}")

    return {"selected": n_selected, "written": written, "summary": summary}
