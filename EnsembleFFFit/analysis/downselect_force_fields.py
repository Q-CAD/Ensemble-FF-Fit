"""
Downselect a subset of already-ranked force fields and copy them to a new
location -- e.g. picking which validated ensemble members go on to run UQ
single points, out of everything that was fit and validated.

Ported from an older Perlmutter notebook implementation; kept in
Ensemble-FF-Fit (unlike the mp-id structure sampler for FT-MD, which the
project explicitly wants to keep as an easily user-edited standalone script)
since the three selection strategies here are generic, reusable mechanics --
not site-specific logic -- the same category as
EnsembleFFFit.structures.deviation_selection's select_structures.
"""
import os
import random
import shutil

import numpy as np


def copy_files(source_dir, dest_dir, patterns):
    """Copy every file directly under source_dir whose name is in `patterns`
    into dest_dir (created if needed), preserving the filename."""
    os.makedirs(dest_dir, exist_ok=True)
    copied = []
    for name in patterns:
        src = os.path.join(source_dir, name)
        if os.path.exists(src):
            dst = os.path.join(dest_dir, name)
            shutil.copy2(src, dst)
            copied.append(dst)
    return copied


def select_and_copy(validation_path, single_point_path, validation_labels,
                     number_to_copy=25, strategy="top", patterns=("model.model",), seed=None):
    """
    Pick `number_to_copy` entries out of `validation_labels` per `strategy`,
    and copy `patterns` (e.g. the fitted model file) from each selected
    entry's directory under `validation_path` into the matching directory
    under `single_point_path`.

    `validation_labels` must already be sorted by validation score (best
    first) -- this function only selects *indices* into that list, it
    doesn't do any ranking itself (see EnsembleFFFit.analysis.best_force_field
    .rank_ff_scores for that step). Each label is treated as a relative path
    (e.g. the mirrored per-variant subtree name from build_ff_inputs.py's
    numbered mace_inputs folders), joined onto validation_path/single_point_path.

    strategy:
      - "top":    the first number_to_copy labels (i.e. the best-scoring ones)
      - "spread": number_to_copy indices evenly spaced across the full sorted
                  range, from best to worst -- deliberately includes low- and
                  mid-ranked members too, for ensemble diversity rather than
                  just picking the single best cluster
      - "random": a uniform random sample of number_to_copy indices, seeded
                  for reproducibility if `seed` is given

    Returns the list of (label, source_dir, dest_dir) actually copied.
    """
    n = len(validation_labels)
    if strategy == "top":
        indices = list(range(min(number_to_copy, n)))
    elif strategy == "spread":
        indices = sorted(set(int(i) for i in np.linspace(0, n - 1, number_to_copy)))
    elif strategy == "random":
        rng = random.Random(seed)
        indices = rng.sample(range(n), min(number_to_copy, n))
    else:
        raise ValueError(f"Unknown strategy {strategy!r} -- expected 'top', 'spread', or 'random'")

    selected = []
    for i in indices:
        label = validation_labels[i]
        old_path = os.path.join(validation_path, label)
        new_path = os.path.join(single_point_path, label)
        copy_files(old_path, new_path, patterns)
        selected.append((label, old_path, new_path))

    return selected
