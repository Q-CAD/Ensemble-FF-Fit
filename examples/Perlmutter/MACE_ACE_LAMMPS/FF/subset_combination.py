"""
Resolves a list of {tier, phases?} subset specs (this example's own
training_subsets_by_combo config, see run_pipeline.py's own build_ff_inputs
docstring) into a flat list of directory paths -- exactly the shape
EnsembleFFFit.potential.ace.build_ace_dataset.write_ace_dataset/
build_ace_dataframe's own `training_root` argument already accepts (a
single path, OR a list of paths, each walked and pooled unweighted; see
that module's own docstring). No EFF-core change needed for this at all --
write_ace_dataset already supports being handed a list assembled however
the caller likes; this module is just that assembly, done once per combo
in the fitting stage's own outer loop.

Resolution approach -- GLOB, not a metadata.json read, and deliberately so:
{tier: X} with no `phases` resolves directly to that tier's own
single-point-labeled root (TIER_ROOTS below -- the exact same structures_
root/output_root paths this example's own run_single_points_tier{1,2,3,4}
stages already write to). {tier: X, phases: [...]} needs to find the
SPECIFIC per-phase subtrees within that tier's root -- tier 3 and tier 4
nest phase labels at DIFFERENT depths (tier 4's defect/motif axes add a
leading "defect"/"motif_<T>k" segment tier 3 doesn't have -- confirmed
directly from workflow_config.yaml's own run_md_tier3/run_md_tier4_defect/
run_md_tier4_motif_300k output_root values), and a given phase can appear
under MULTIPLE such subtrees at once (tier 4's 3 target phases each appear
under both its defect axis and all 4 of its motif-temperature axes).
Path.rglob(f"Pd-Se/{phase}") finds every one of these subtrees in one call,
regardless of how many or how deep, using the SAME "Pd-Se/<phase>" path
convention the structure generators themselves already write (see
generate_defect_structures.py/generate_motif_structures.py's own
docstrings) and ase_mace.py's own _path_derived_metadata already matches
against for its 'phase' metadata field -- one convention, not two. This
needs no running MD/single-point data to already exist (pure directory-
glob, not a per-structure read), unlike a metadata.json-based filter,
which would need to read every leaf's own sidecar just to decide which
directories to pass to write_ace_dataset at all.
"""
from pathlib import Path

# Tier label -> single-point-labeled structures root, matching this
# example's own run_single_points_tier{1,2,3,4} stage config exactly
# (workflow_config.yaml's own structures_root/output_root values for each).
TIER_ROOTS = {
    "tier1": "MACE/single_points/structures/training",
    "tier2": "MACE/single_points/structures/training_2",
    "tier3": "MACE/single_points/structures/training_3",
    "tier4": "MACE/single_points/structures/training_4",
}


def resolve_subset(spec):
    """
    Resolves one {tier, phases?} spec to a list of directory path strings.
    Raises KeyError for an unknown tier (fail loudly, not a silent empty
    subset) and ValueError if `phases` is given but a phase has no matching
    subtree under that tier's root (same "don't silently resolve to
    nothing" reasoning as ase_mace.py's own _path_derived_metadata).
    """
    tier = spec["tier"]
    if tier not in TIER_ROOTS:
        raise KeyError(f"subset_combination: unknown tier {tier!r} -- must be one of {list(TIER_ROOTS)}.")
    root = Path(TIER_ROOTS[tier])

    phases = spec.get("phases")
    if not phases:
        return [str(root)]

    resolved = []
    for phase in phases:
        matches = sorted(root.rglob(f"Pd-Se/{phase}"))
        if not matches:
            raise ValueError(
                f"subset_combination: no subtree matching 'Pd-Se/{phase}' found under {root} "
                f"(tier={tier!r}) -- check that this phase's structures were actually generated "
                f"for this tier, and that the phase name matches exactly (case-sensitive).")
        resolved.extend(str(m) for m in matches)
    return resolved


def resolve_subsets(specs):
    """Resolves a list of {tier, phases?} specs to one flat, deduplicated list of directory paths."""
    directories = []
    for spec in specs:
        for d in resolve_subset(spec):
            if d not in directories:
                directories.append(d)
    return directories
