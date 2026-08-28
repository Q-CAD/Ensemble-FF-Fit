"""
Build a randomly-sampled ensemble of MACE training-input folders from a pool
of per-composition (or other label) .xyz files crossed against an
energy/force/stress weight grid.

Ported from an older, notebook-driven Perlmutter implementation
(test/Full_Pipeline/linking_logic_excerpts/combine_xyzs.py) into importable,
argparse-free functions per the standing bar for this refactor: agent-legible
signatures, no hidden global state, callable directly from a pipeline script.

Deliberately MACE-specific (config.yml shape, E0s-by-atomic-number), not
conserved DFT-agnostic logic -- other force-field backends need their own
equivalent.
"""
import itertools
import os
import random
import shutil
import tempfile
from pathlib import Path

import yaml


def get_label_combos(labels, always_include=()):
    """
    Every non-empty subset of `labels` (the full powerset minus the empty
    set), each with `always_include` appended. This is the same combinatorial
    shape as the old combine_xyzs.py's get_data_combos, generalized to work
    over any label set (compositions here; the old script used structure-type
    labels like EoS/Defect/Slab) -- for 6 labels this is 63 combinations, not
    a small number, which is why sample_ensemble_configs below exists rather
    than materializing every combo directly.
    """
    always_include = tuple(always_include)
    choosable = [label for label in labels if label not in always_include]

    combos = []
    for r in range(1, len(choosable) + 1):
        combos.extend(itertools.combinations(choosable, r))
    if not combos:
        return [always_include] if always_include else []
    return [combo + always_include for combo in combos]


def get_weight_grid(energy_range=(1.0, 100.0), energy_num=3,
                     force_range=(1.0, 100.0), force_num=3,
                     stress_range=(1.0, 100.0), stress_num=1):
    """
    Cartesian product of linspace'd energy/force/stress weight ranges, e.g.
    the default (3, 3, 1) gives 9 combinations -- the same three-axis
    parametrization the old combine_xyzs.py used (there hardcoded), now with
    each axis's range/count exposed as a handle rather than fixed in place.
    Returns a list of {'energy_weight':..., 'forces_weight':..., 'stress_weight':...} dicts.
    """
    def _linspace(lo, hi, n):
        if n == 1:
            return [float(lo)]
        step = (hi - lo) / (n - 1)
        return [float(lo + i * step) for i in range(n)]

    energy_weights = _linspace(*energy_range, energy_num)
    force_weights = _linspace(*force_range, force_num)
    stress_weights = _linspace(*stress_range, stress_num)

    return [
        {"energy_weight": e, "forces_weight": f, "stress_weight": s}
        for e, f, s in itertools.product(energy_weights, force_weights, stress_weights)
    ]


def sample_ensemble_configs(data_combos, weight_combos, total_cap, seed):
    """
    Flat cartesian product of data_combos x weight_combos, then a seeded
    random sample capped at `total_cap` (or the full product, if smaller).
    Each returned (data_combo, weight_combo) pair becomes one self-contained
    numbered mace_inputs folder -- no further nesting by weight, since each
    pair already pins exactly one weight choice.
    """
    full_product = list(itertools.product(data_combos, weight_combos))
    rng = random.Random(seed)
    n = min(total_cap, len(full_product))
    return rng.sample(full_product, n)


def concatenate_xyz_files(xyz_paths, output_path):
    """Concatenate extxyz files verbatim (each is already a complete, valid
    sequence of frames) into one combined file at output_path."""
    with open(output_path, "w") as out:
        for path in xyz_paths:
            with open(path) as f:
                out.write(f.read())


def get_isolated_atom_e0s(isolated_elements_directory, properties_filename="properties.json"):
    """
    Walk `isolated_elements_directory` for completed single-atom runs (one
    subdirectory per element, e.g. isolated_elements/Bi/properties.json) and
    return {atomic_number: energy}, the shape MACE's config.yml E0s key
    expects. Element symbol -> atomic number is resolved via the run
    directory's own name, matching the isolated_elements/<Symbol>/ convention.
    """
    import json
    from pymatgen.core import Element

    e0s = {}
    for entry in sorted(Path(isolated_elements_directory).iterdir()):
        if not entry.is_dir():
            continue
        props_path = entry / properties_filename
        if not props_path.exists():
            continue
        with open(props_path) as f:
            energy = json.load(f)["energy"]
        atomic_number = Element(entry.name).Z
        e0s[atomic_number] = energy

    return e0s


DEFAULT_BASE_CONFIG = {
    "energy_key": "energy",
    "forces_key": "forces",
    "stress_key": "stress",
    "device": "cuda",
    "batch_size": 10,
    "max_num_epochs": 100,
    "num_samples_pt": 300,
    "swa": False,
    "model": "MACE",
    "multiheads_finetuning": False,
    "valid_fraction": 0.15,
    "enable_oeq": True,
}


def write_mace_config(path, weight_combo, e0s, base_config=None):
    """
    Write one config.yml: DEFAULT_BASE_CONFIG above (matching the reference
    test/multi_MACE_stages/generation_0/FF/mace_inputs example), with
    `base_config` layered on top -- pass only the keys you want to add or
    override (e.g. {"freeze": 6}), not a full config from scratch -- followed
    by this specific weight_combo and e0s, which always win last.
    """
    config = dict(DEFAULT_BASE_CONFIG)
    if base_config:
        config.update(base_config)
    config.update(weight_combo)
    config["E0s"] = dict(e0s)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)


def build_mace_ensemble_inputs(training_xyz_by_label, validation_xyz_paths, output_dir,
                                e0s, total_cap=100, seed=0, weight_grid_kwargs=None,
                                always_include_labels=(), base_config=None,
                                config_filename="config.yml"):
    """
    Materialize the randomly-sampled ensemble: numbered folders 0..N-1 under
    `output_dir`, each with train.xyz (concatenation of whichever composition
    labels that folder's sampled data_combo picked), test.xyz (the fixed
    concatenation of every path in validation_xyz_paths -- the same for every
    folder, not sampled), config.yml (this folder's sampled weight_combo +
    e0s), and METADATA (which source .xyz paths went into train.xyz/test.xyz).

    training_xyz_by_label: {label: xyz_path} -- one path per composition (or
    whatever label set is in play), e.g. {"Bi2Se3": ".../Bi2Se3.xyz", ...}.
    validation_xyz_paths: list of xyz paths making up the fixed test.xyz.

    Returns the list of folder paths actually written.
    """
    weight_grid_kwargs = weight_grid_kwargs or {}
    labels = list(training_xyz_by_label.keys())
    data_combos = get_label_combos(labels, always_include=always_include_labels)
    weight_combos = get_weight_grid(**weight_grid_kwargs)

    sampled = sample_ensemble_configs(data_combos, weight_combos, total_cap, seed)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Built in a temp file, never directly inside output_dir: a stray
    # top-level test.xyz there would give MACEMatEnsemble's proximity
    # matcher one more test.xyz match than train.xyz/config.yml matches,
    # making test.xyz the anchor and creating a spurious extra fitting task
    # (confirmed -- it duplicates one folder's results_dir with a mismatched
    # test_file, an actual collision between two chores writing to the same
    # directory).
    with tempfile.TemporaryDirectory() as tmp_dir:
        fixed_test_xyz = Path(tmp_dir) / "test.xyz"
        concatenate_xyz_files(validation_xyz_paths, fixed_test_xyz)

        written = []
        for i, (data_combo, weight_combo) in enumerate(sampled):
            folder = output_dir / str(i)
            folder.mkdir(parents=True, exist_ok=True)

            train_paths = [training_xyz_by_label[label] for label in data_combo]
            train_xyz = folder / "train.xyz"
            concatenate_xyz_files(train_paths, train_xyz)

            test_xyz = folder / "test.xyz"
            shutil.copy2(fixed_test_xyz, test_xyz)

            write_mace_config(folder / config_filename, weight_combo, e0s, base_config=base_config)

            with open(folder / "METADATA", "w") as f:
                f.write(f"train.xyz: combined from labels {list(data_combo)}\n")
                for label, path in zip(data_combo, train_paths):
                    f.write(f"  {label}: {path}\n")
                f.write(f"test.xyz: fixed validation set, combined from {len(validation_xyz_paths)} path(s)\n")
                for path in validation_xyz_paths:
                    f.write(f"  {path}\n")
                f.write(f"weight_combo: {weight_combo}\n")

            written.append(str(folder))

    return written
