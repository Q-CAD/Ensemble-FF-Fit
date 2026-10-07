"""
Generate a diverse ensemble of ReaxFF fitting-input folders by
cross-producting parse2fit's own geo/trainset.in variants (see
run_pipeline.py's run_parse2fit_generation -- each
parse2fit_root/<output_format>_run_<i>/{geo,trainset.in}) against a params
"blocking scheme" spec describing which ReaxFF parameter SECTIONS are
active (uncommented, trainable) vs frozen (commented out, `#`-prefixed)
for each ensemble member -- the ReaxFF analog of MACE's "freeze" config
knob (see build_ff_inputs' mace_overrides in run_pipeline.py), but working
at the level of individual params-file lines rather than model layers,
since ReaxFF has no equivalent concept of "layers" to freeze.

No automated version of this existed before this pipeline. Confirmed by
recovering the deleted JaxReaxFFMatEnsemble/potential/reaxff/ code from
Ensemble-FF-Fit's git history (origin/Claude branch) before writing this --
that code's ensemble diversity came from hand-authored params files per
composition pair (Bi-Bi/params, Bi-Se/params, Se-Se/params), organized by
directory-naming convention ("angular", "angular_twobody"), not generated
variants. This module is genuine new design, not a port -- see
Perlmutter_Pipeline_Wiring.md/JaxReaxFF_Integration_Plan.md for the open
questions it answers.

DESIGN, revised (2026-09-09) from an earlier name-substring-matching
scheme to section-NUMBER-based grouping instead, per direct guidance: each
params-catalog line's leading integer (its first whitespace-separated
field) is ReaxFF's own parameter-type marker -- 1=General, 2=Atom,
3=Bond, 4=Off-diagonal, 5=Angular, 6=Dihedral in this catalog's own
convention (confirmed by reading the full catalog file, not assumed) --
and a blocking scheme is just {ensemble_member_label: [section_number,
...]}. This replaced an earlier group_patterns approach (matching each
line's trailing `!`-comment field, e.g. "p_boc1", against caller-supplied
substrings) for good reason, confirmed by reading the FULL catalog (not
just its General-parameters section): naming conventions are wildly
inconsistent across sections (General's "p_boc1" vs Atom's "cov.r" vs
Bond's "pbo5" vs Angular's "p(val1)" -- note parentheses, not the
underscore Angular's own name would suggest), Off-diagonal's comments are
freeform description text with no parameter-name token at all, and
Dihedral lines have NO `!`-comment field whatsoever -- making them
structurally unreachable by any comment-matching scheme regardless of
which substrings were supplied. The leading section number is present and
unambiguous on every single parameter line, comment or no comment,
sidestepping all of this at once.

BOUND HANDLING, confirmed necessary (2026-09-09) rather than assumed:
JAX-ReaxFF's own jaxreaxff.helper.read_parameter_file parses every
non-commented line's 5th/6th columns as bare floats -- an ACTIVE line
whose catalog bounds are the literal placeholder "n/a" (meaning "not
given/unbounded" in this catalog's own convention) makes driver.py crash
immediately on `float("n/a")` the moment it reads that params file. "n/a"
only ever worked because read_parameter_file skips `#`-commented lines
entirely, without parsing their bounds at all -- it was never actually a
supported "unbounded, but still trainable" state. So any line this module
activates that the catalog leaves unbounded gets a wide numeric placeholder
range instead (get_ffield_parameter_values's caller-supplied
unbounded_half_width around that parameter's own current force-field
value, or around 0.0 if no force field was given) -- the closest safe
equivalent to "unbounded" JAX-ReaxFF can actually parse. Separately, if a
seed force field is given, every activated line's bounds are also widened
(never narrowed) just enough to include that force field's own actual
starting value for that parameter, when the catalog's own bounds don't
already -- otherwise the optimizer would start outside its own declared
search range. Some Bond-section lines in this catalog omit bounds
entirely (fewer than 6 fields, not even "n/a n/a") -- those are still
correctly grouped by section number, but their bounds are left exactly as
written (bound-widening needs a real low/high pair to widen), and if
activated, JAX-ReaxFF's own read_parameter_file will silently skip them
(its own len(fields) < 6 check) rather than actually optimize them -- a
pre-existing catalog-completeness gap, not something this module can fix
on the catalog's behalf.

Output shape: output_dir/<parse2fit_run_name>_<label>/{geo, trainset.in,
params, METADATA} -- one folder per (parse2fit run, blocking_scheme label)
combination (a full cross product, by design -- see
JaxReaxFF_Integration_Plan.md's sign-off on this). This shape matters, not
just style: FFMatEnsemble.build_ff_dcts proximity-matches inputs_directory's
own files by directory nearness (_make_proximity_combinations) -- with
geo/trainset.in/params/METADATA all sitting together in one combo folder,
every combo's own files always pair correctly regardless of how many
combos exist (the same self-contained-per-variant shape already confirmed
to proximity-match correctly for finite_temperature_md/MD_uq_single_points
structures elsewhere in this pipeline). `ffield` (the shared, unvarying
seed force field -- the thing that actually gets fitted, distinct from
`params`, which just controls which of its numbers are allowed to vary,
and whose CURRENT values this module reads for bound-widening) is
deliberately NOT copied into each combo folder: it belongs in
fine_tuning.run_directory (see run_pipeline/workflow_config.yaml), matching
FFMatEnsemble's check_files-anchored single-seed-file convention -- the
same role MACE's foundation_model plays under fine_tuning.run_directory,
not fine_tuning.inputs_directory.
"""
import glob
import itertools
import math
import os
import random
import shutil


def tag_params_catalog(catalog_path):
    """
    Read `catalog_path` (a params file with every candidate line present --
    this catalog's own lines may already be `#`-commented or not, it makes
    no difference here, since active-vs-frozen membership is decided fresh
    by this module, not inherited from the catalog's own comment state),
    and return a list of (raw_line, section_or_None, parsed_fields_or_None,
    comment_field_or_None) tuples in file order:

    - section is the line's own leading integer (ReaxFF's parameter-type
      marker: 1=General, 2=Atom, 3=Bond, 4=Off-diagonal, 5=Angular,
      6=Dihedral in this catalog's own convention) whenever the line's
      content starts with >=3 whitespace-separated integer fields
      (section, index1, index2) -- None for blank lines and section-header
      comment lines (e.g. "# General parameters") that aren't parameter
      lines at all.
    - parsed_fields is (section, index1, index2, sensitivity, low_str,
      high_str) when the line has >=6 such leading fields (enough for
      bound-widening) -- None otherwise, e.g. for the several Bond-section
      lines in this catalog that omit bounds entirely (see this module's
      own docstring). section can be set even when parsed_fields is None.
    - comment_field is the line's own trailing `!`-text (for METADATA
      description text only, never used for grouping any more -- see this
      module's own docstring for why) -- None when the line has no `!` at
      all (true for every Dihedral-section line in this catalog).
    """
    tagged = []
    with open(catalog_path) as f:
        lines = f.readlines()

    for line in lines:
        stripped = line.rstrip('\n').strip()
        if not stripped:
            tagged.append((line, None, None, None))
            continue

        content_and_comment = stripped.lstrip('#').strip()
        if '!' in content_and_comment:
            content, comment_field = content_and_comment.split('!', 1)
        else:
            content, comment_field = content_and_comment, None

        if not content.strip():
            tagged.append((line, None, None, comment_field))
            continue

        fields = content.split()
        section = None
        parsed = None
        if len(fields) >= 3:
            try:
                section, index1, index2 = int(fields[0]), int(fields[1]), int(fields[2])
            except ValueError:
                section = None  # not a parameter line (e.g. a header/description-only comment)

            if section is not None and len(fields) >= 6:
                try:
                    sensitivity = float(fields[3])
                    parsed = (section, index1, index2, sensitivity, fields[4], fields[5])
                except ValueError:
                    parsed = None  # bounds/sensitivity aren't cleanly numeric -- section is still known

        tagged.append((line, section, parsed, comment_field))

    return tagged


def get_ffield_parameter_values(ffield_path, section_index_tuples, cutoff2=0.001):
    """
    Returns (values, valid_keys):
    - values: {(section, index1, index2): float_value} for every requested
      tuple that this force field actually has, using JAX-ReaxFF's OWN
      force-field-loading + parameter-mapping pipeline -- CONFIRMED
      (2026-09-09) this exact recipe reproduces driver.py's own force-
      field-loading path (see jaxreaxff/driver.py's build_arg_parser()/
      run(), around the "Force field is read" print), so parameter values
      looked up here are the same ones driver.py itself would start
      optimizing from, not a reimplementation of ReaxFF's file format
      guessed from examples.
    - valid_keys: the FULL set of (section, index1, index2) keys this
      force field's own params_to_indices mapping actually supports, not
      just the requested ones. CONFIRMED NECESSARY (2026-09-09), not a
      defensive nice-to-have -- checked directly against a real crash:
      jaxreaxff.helper.map_params does a bare `index_map[key]` with no
      fallback, so activating ANY catalog line whose (section, index1,
      index2) isn't a real key here crashes driver.py outright
      (KeyError) the moment it tries to fit, regardless of what bounds
      that line has. This is not rare or confined to one line -- checked
      directly against this exact catalog/force field: General (10
      entries), Atom (18), Bond (12, uniformly across every bond pair --
      e.g. (3, i, 6)'s "13corr" for every i), and Dihedral (ALL 6, no
      exceptions) each contain catalog lines with no real mapping at
      all; only Off-diagonal and Angular are fully covered.
      write_params_variant/build_reaxff_ensemble_inputs use this to force
      any such line frozen regardless of which blocking_scheme section
      it nominally belongs to.

    Lazily imports jax/jax_md/jaxreaxff -- this module's other functions
    (tag_params_catalog/write_params_variant with ffield_values=None) have
    no jax dependency at all, so only call this when bound-widening/
    validating against a real seed force field is actually wanted.

    Forces the CPU platform (JAX_PLATFORMS, set before jax is ever
    imported by this process) -- this lookup is a handful of array reads
    from one small force field, not worth risking the libcuda.so.1
    real-driver-vs-compat-shim version mismatch already confirmed twice
    elsewhere in this pipeline (lammps_container_env()/
    jax_reaxff_container_env()) for something this cheap. Only takes
    effect if jax hasn't already been imported by the calling process with
    a different platform selected -- fine here since this stage never
    otherwise imports jax itself.

    Any requested tuple not present in `valid_keys` is silently omitted
    from `values`, not raised -- callers fall back to the catalog's own
    bounds (or the unbounded placeholder) for those, and separately use
    `valid_keys` itself to decide whether a line should even be
    activatable at all.
    """
    os.environ.setdefault('JAX_PLATFORMS', 'cpu')
    import jax.numpy as jnp
    from jax_md.reaxff.reaxff_forcefield import ForceField
    from jax_md.reaxff.reaxff_helper import read_force_field
    from jaxreaxff.helper import map_params, get_params

    force_field = read_force_field(ffield_path, cutoff2=cutoff2, dtype=jnp.float64)
    force_field = ForceField.fill_off_diag(force_field)
    force_field = ForceField.fill_symm(force_field)

    valid_keys = set(force_field.params_to_indices.keys())
    available = [t for t in section_index_tuples if t in valid_keys]

    values = {}
    if available:
        # sensitivity/low/high are irrelevant for a value lookup --
        # map_params only uses positions 0-2 to look up the ForceField
        # attribute/index, positions 3-5 just ride along into its own
        # return shape unchanged.
        dummy = [(section, index1, index2, 1.0, 0.0, 0.0) for section, index1, index2 in available]
        mapped = map_params(dummy, force_field.params_to_indices)
        computed = get_params(force_field, [m[0] for m in mapped])
        values = {t: float(v) for t, v in zip(available, computed)}

    return values, valid_keys


def sample_parameter_subsets(tagged_lines, valid_param_keys, section=5, n_variants=75,
                              min_params=1, max_params=6, stratify=True, rng_seed=None,
                              label_prefix=None):
    """
    Returns {label: [(section, index1, index2), ...]} -- n_variants randomly
    sampled subsets of `section`'s own catalog lines, each of size k (k
    ranging over min_params..max_params), in the SAME shape write_params_
    variant's `active_sections` already accepts for fine-grained (single-
    line) activation -- so the result can be passed straight through as a
    blocking_scheme to build_reaxff_ensemble_inputs, no other changes
    needed. Added (2026-10) after a real investigation found that
    activating an entire section at once (the original, coarse-grained
    scheme) let ReaxFF's angular force constants move 7-18x their starting
    value during fitting, severely degrading LAMMPS-reax/c agreement
    relative to the unfitted seed potential -- this caps how many
    parameters move at once instead, to isolate which ones are safe to
    free.

    Candidate pool is every `section` line with real parsed bounds (parsed
    is not None) AND a real JAX-ReaxFF parameter mapping (its (section,
    index1, index2) key is in valid_param_keys -- see
    get_ffield_parameter_values; activating a line with no real mapping
    would otherwise crash driver.py with a KeyError once fitting runs,
    same reasoning as write_params_variant's own valid_param_keys check).

    If stratify (the default), k cycles evenly through min_params..
    max_params (k = min_params + (i % span) for variant i) so every count
    is represented an equal, or as-equal-as-n_variants-allows, number of
    times -- deliberate, not incidental: reading "does performance degrade
    as more parameters are freed" cleanly out of the resulting ensemble
    needs even coverage of each count, which isn't guaranteed if k is
    drawn freely at random instead (set stratify=False for that).

    Never generates two variants with the exact identical sampled subset.
    If n_variants is >= the total number of distinct subsets actually
    available (sum of C(len(pool), k) for k in min_params..max_params --
    e.g. the Off-diagonal section's catalog has only 4 real JAX-ReaxFF-
    mapped parameters, 14 distinct subsets total for min_params=1,
    max_params=3, far fewer than a 75-variant request), every distinct
    subset is returned exactly once (exhaustive enumeration) instead of
    padding out to n_variants with duplicate re-fits of the same subset,
    which would waste compute without adding information. Otherwise, falls
    back to random sampling (regenerating on a duplicate draw, up to 1000
    attempts before giving up and accepting the duplicate -- only
    reachable right at the edge of the distinct-subset count).
    """
    pool = sorted({parsed[0:3] for _, _, parsed, _ in tagged_lines
                   if parsed is not None and parsed[0] == section and parsed[0:3] in valid_param_keys})
    if not pool:
        raise ValueError(f"No section-{section} catalog lines with real bounds and a valid "
                          f"JAX-ReaxFF parameter mapping -- nothing to sample from.")

    label_prefix = label_prefix or f"section{section}_sample"
    span = max_params - min_params + 1
    k_range = [k for k in range(min_params, max_params + 1) if k <= len(pool)]
    total_distinct = sum(math.comb(len(pool), k) for k in k_range)

    if n_variants >= total_distinct:
        if n_variants > total_distinct:
            print(f"NOTE: sample_parameter_subsets({label_prefix!r}): requested n_variants={n_variants} "
                  f"but only {total_distinct} distinct subset(s) exist for section {section} with "
                  f"{len(pool)} valid parameter(s) and min_params={min_params}, max_params={max_params} "
                  f"-- returning all {total_distinct} exhaustively instead of padding with duplicates.")
        scheme = {}
        i = 0
        for k in k_range:
            for subset in itertools.combinations(pool, k):
                scheme[f"{label_prefix}_{i:03d}_k{k}"] = list(subset)
                i += 1
        return scheme

    rng = random.Random(rng_seed)
    scheme = {}
    seen_subsets = set()
    for i in range(n_variants):
        k = (min_params + (i % span)) if stratify else rng.randint(min_params, max_params)
        k = min(k, len(pool))

        attempts = 0
        while True:
            subset = tuple(sorted(rng.sample(pool, k)))
            attempts += 1
            if subset not in seen_subsets or attempts > 1000:
                break
        seen_subsets.add(subset)

        scheme[f"{label_prefix}_{i:03d}_k{k}"] = list(subset)

    return scheme


def write_params_variant(tagged_lines, active_sections, output_path,
                          ffield_values=None, unbounded_half_width=1e4,
                          valid_param_keys=None):
    """
    Write a `params` file to `output_path`: every tagged line written active
    (uncommented) if EITHER its own section is in `active_sections` (whole-
    section activation, e.g. [4, 5] for Off-diagonal + Angular -- the
    original, coarse-grained scheme) OR its own (section, index1, index2)
    key is itself an element of `active_sections` (fine-grained, single-
    line activation -- what sample_parameter_subsets produces) -- the two
    forms of membership don't conflict (ints vs. 3-tuples), so the same set
    can mix both if ever needed, though a given call typically uses one or
    the other. Every other line (including section=None lines) written
    frozen (commented, `#`-prefixed).

    valid_param_keys, if given (see get_ffield_parameter_values), forces
    any line whose (section, index1, index2) isn't a real, optimizable
    JAX-ReaxFF parameter to stay frozen regardless of whether its section
    is in active_sections -- CONFIRMED NECESSARY, not defensive (see
    get_ffield_parameter_values' own docstring): several catalog lines per
    section have no real mapping at all, and activating one crashes
    driver.py with a raw KeyError the moment it tries to fit. Lines with
    parsed=None (missing bounds entirely) are never checked against this
    -- they can't reach JAX-ReaxFF's own optimizer regardless, since its
    own read_parameter_file skips any line with fewer than 6 fields
    before map_params ever runs.

    For every ACTIVE line with cleanly parsed bound fields (see
    tag_params_catalog), the written bounds are:
    - left as originally catalogued, if both are numeric AND (no
      ffield_values were given, OR that parameter isn't in ffield_values,
      OR the actual force-field value already falls within them);
    - widened (never narrowed) just enough to include the actual
      force-field value, with a small margin so the optimizer doesn't
      start exactly on a bound, when ffield_values gives a value outside
      the catalogued range;
    - replaced with a wide placeholder range centered on the force-field
      value (or 0.0 if none given/available) of half-width
      unbounded_half_width, whenever either side is the catalog's "n/a"
      placeholder -- see this module's own docstring for why "n/a" is
      never safe to leave on an active line at all, independent of
      whether bound-widening against a specific force field was requested.

    Returns (active_descriptions, skipped_unsupported_descriptions) --
    both lists of description strings (comment field when present, else
    "section index1 index2"), in file order, for METADATA.
    """
    active_sections = set(active_sections)
    active_descriptions = []
    skipped_unsupported = []
    lines_out = []

    for raw_line, section, parsed, comment_field in tagged_lines:
        stripped = raw_line.rstrip('\n')
        is_active = (section is not None and section in active_sections) or (
            parsed is not None and parsed[0:3] in active_sections)

        if is_active and parsed is not None and valid_param_keys is not None:
            if parsed[0:3] not in valid_param_keys:
                is_active = False
                skipped_unsupported.append(
                    (comment_field or f"section {parsed[0]} index ({parsed[1]},{parsed[2]})").strip())

        if is_active and parsed is not None:
            section_p, index1, index2, sensitivity, low_str, high_str = parsed
            low = None if low_str.strip().lower() == 'n/a' else float(low_str)
            high = None if high_str.strip().lower() == 'n/a' else float(high_str)
            value = ffield_values.get((section_p, index1, index2)) if ffield_values else None

            needs_rewrite = (low is None) or (high is None) or (
                value is not None and (value < low or value > high))
            if needs_rewrite:
                base = value if value is not None else 0.0
                if low is None:
                    low = base - unbounded_half_width
                if high is None:
                    high = base + unbounded_half_width
                if value is not None:
                    margin = 0.01 * max(abs(high - low), 1.0)
                    if value < low:
                        low = value - margin
                    if value > high:
                        high = value + margin
                comment_suffix = f"  !{comment_field}" if comment_field is not None else ""
                stripped = f"{section_p} {index1} {index2}  {sensitivity:g}  {low:.6g} {high:.6g}{comment_suffix}"

        if is_active:
            if comment_field is not None:
                active_descriptions.append(comment_field.strip())
            elif parsed is not None:
                active_descriptions.append(f"section {parsed[0]} index ({parsed[1]},{parsed[2]})")
            else:
                active_descriptions.append(stripped.strip())

        body = stripped.lstrip('#') if stripped.lstrip().startswith('#') else stripped
        if is_active:
            lines_out.append(body.lstrip() + '\n')
        else:
            lines_out.append((('#' + body.lstrip()) if body.strip() else stripped) + '\n')

    with open(output_path, 'w') as f:
        f.writelines(lines_out)

    if skipped_unsupported:
        print(f"NOTE: {output_path}: {len(skipped_unsupported)} catalog line(s) requested by "
              f"section membership have no real JAX-ReaxFF parameter mapping -- kept frozen.")

    return active_descriptions, skipped_unsupported


def write_metadata(combo_dir, parse2fit_run_dir, blocking_label, active_sections,
                    active_descriptions, skipped_unsupported=()):
    """Plain-text summary of which geo/trainset.in this combo uses and which
    parameters were allowed to optimize -- same spirit as the MACE pipeline's
    own METADATA (which training .xyz files + weights went into a given
    ensemble member), adapted to ReaxFF's own params shape rather than
    matching its exact fields, since MACE has no equivalent of a params
    blocking scheme. skipped_unsupported records anything requested by
    section membership but forced frozen anyway because JAX-ReaxFF itself
    has no real mapping for it (see write_params_variant's own docstring)
    -- kept here, not just printed during generation, so it stays
    auditable per combo after the fact."""
    lines = [
        f"parse2fit source: {parse2fit_run_dir}",
        f"blocking-scheme label: {blocking_label}",
        f"active parameter sections: {sorted(active_sections)}",
        f"active parameters (n={len(active_descriptions)}):",
    ]
    lines.extend(f"  {desc}" for desc in active_descriptions)
    if skipped_unsupported:
        lines.append(f"requested but not a real JAX-ReaxFF parameter, kept frozen "
                     f"(n={len(skipped_unsupported)}):")
        lines.extend(f"  {desc}" for desc in skipped_unsupported)
    with open(os.path.join(combo_dir, 'METADATA'), 'w') as f:
        f.write('\n'.join(lines) + '\n')


def build_reaxff_ensemble_inputs(parse2fit_root, catalog_params_path,
                                  blocking_scheme, output_dir, ffield_path=None,
                                  unbounded_half_width=1e4, seed_runs=None):
    """
    Cross-products every parse2fit-generated geo/trainset.in variant under
    parse2fit_root (one per f"{output_format}_run_{i}" folder -- see
    parse2fit's own readwrite.py/run_pipeline.py's run_parse2fit_generation)
    against every blocking_scheme label ({label: [section_number_or_(section,
    index1,index2)_key, ...]} -- see write_params_variant/
    sample_parameter_subsets for the two forms), writing output_dir/
    <parse2fit_run_name>_<label>/{geo, trainset.in, params, METADATA} for
    each combination. Returns the list of written combo directories.

    seed_runs (optional): restrict to specific parse2fit run directories by
    basename (e.g. ['reaxff_run_0', 'reaxff_run_1']) instead of every one
    discovered under parse2fit_root -- for exploring a parameter-sampling
    scheme against a handful of seeds before committing to the full set.

    ffield_path is effectively required for a correct result, not merely
    helpful for bound-widening -- see get_ffield_parameter_values' own
    docstring: without it, there's no way to filter out the catalog lines
    that have no real JAX-ReaxFF parameter mapping at all, and activating
    one of those crashes driver.py outright once fitting actually runs
    (confirmed via a real KeyError, not theoretical).
    """
    tagged_lines = tag_params_catalog(catalog_params_path)

    ffield_values = None
    valid_param_keys = None
    if ffield_path is not None:
        section_index_tuples = {p[0:3] for _, _, p, _ in tagged_lines if p is not None}
        ffield_values, valid_param_keys = get_ffield_parameter_values(ffield_path, section_index_tuples)
    else:
        print("WARNING: build_reaxff_ensemble_inputs called with no ffield_path -- cannot filter "
              "out catalog lines with no real JAX-ReaxFF parameter mapping (see "
              "get_ffield_parameter_values' own docstring); activating one of those will crash "
              "driver.py once fitting runs.")

    parse2fit_runs = sorted(
        d for d in glob.glob(os.path.join(parse2fit_root, '*'))
        if os.path.isdir(d) and os.path.exists(os.path.join(d, 'geo'))
        and os.path.exists(os.path.join(d, 'trainset.in'))
    )
    if seed_runs is not None:
        wanted = set(seed_runs)
        parse2fit_runs = [d for d in parse2fit_runs if os.path.basename(d) in wanted]
        missing = wanted - {os.path.basename(d) for d in parse2fit_runs}
        if missing:
            raise FileNotFoundError(f"seed_runs requested but not found under {parse2fit_root}: {sorted(missing)}")
    if not parse2fit_runs:
        raise FileNotFoundError(
            f"No geo+trainset.in folders found under {parse2fit_root} -- did you run "
            f"the parse2fit_generation stage first?"
        )

    os.makedirs(output_dir, exist_ok=True)
    written = []
    for run_dir in parse2fit_runs:
        run_name = os.path.basename(run_dir)
        for label, active_sections in blocking_scheme.items():
            combo_dir = os.path.join(output_dir, f"{run_name}_{label}")
            os.makedirs(combo_dir, exist_ok=True)
            shutil.copy2(os.path.join(run_dir, 'geo'), os.path.join(combo_dir, 'geo'))
            shutil.copy2(os.path.join(run_dir, 'trainset.in'), os.path.join(combo_dir, 'trainset.in'))
            active_descriptions, skipped_unsupported = write_params_variant(
                tagged_lines, active_sections, os.path.join(combo_dir, 'params'),
                ffield_values=ffield_values, unbounded_half_width=unbounded_half_width,
                valid_param_keys=valid_param_keys,
            )
            write_metadata(combo_dir, run_dir, label, active_sections, active_descriptions, skipped_unsupported)
            written.append(combo_dir)

    return written
