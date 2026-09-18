"""
Build a randomly-sampled ensemble of pyace/pacemaker fitting-input folders,
varying seed and fit.loss's energy/force balance (kappa) across members --
same conceptual shape as potential.mace.build_ensemble_inputs' weight grid
(get_weight_grid/sample_ensemble_configs), adapted to ACE's own config
shape rather than MACE's separate train.xyz/test.xyz/config.yml files.

Base potential/fit/backend content is adapted, minimally, from
example_scripts/ACE/input.yaml (a worked ethanol/revMD17 example) -- the
basis size (functions.nradmax_by_orders/lmax_by_orders) is deliberately
much smaller here than that example's, since this project's own dataset
(two structures, ten atomic environments total, as of this writing) has far
fewer data points than a full multi-element ACE basis has fitting
parameters -- see this module's own build_ace_ensemble_inputs docstring.

backend.evaluator is 'pyace' (pyace's native, CPU-only evaluator), not
'tensorpot' -- tensorpotential (the GPU-accelerated, TensorFlow-based
evaluator the example's own comment recommends) can't be installed in this
project's Python 3.12 containers at all (see pipeline/FRICTION_LOG.md).
"""
import itertools
import os
import random
from pathlib import Path

import yaml

# Deliberately small relative to example_scripts/ACE/input.yaml's own
# functions block (nradmax_by_orders: [15, 3, 2, 2, 1], a 3-element,
# body-order-5 basis meant for a much larger dataset) -- body order up to
# 2 (pair + 3-body terms only) keeps the fitting-parameter count in a
# plausible range for this project's current, very small training set.
# Revisit (grow this) once the training set itself grows past a
# smoke-test scale -- see the "only two structures" question this module
# was written to answer.
DEFAULT_POTENTIAL_CONFIG = {
    "deltaSplineBins": 0.001,
    "embeddings": {
        "ALL": {
            "npot": "FinnisSinclairShiftedScaled",
            "fs_parameters": [1, 1, 1, 0.5],
            "ndensity": 2,
        },
    },
    "bonds": {
        "ALL": {
            "radbase": "ChebExpCos",
            "radparameters": [5.25],
            "rcut": 4,
            "dcut": 0.01,
            "NameOfCutoffFunction": "cos",
            "core-repulsion": [0.0, 5.0],
        },
    },
    "functions": {
        "ALL": {
            "nradmax_by_orders": [4, 2],
            "lmax_by_orders": [0, 1],
        },
    },
}

# L1_coeffs/L2_coeffs given a small nonzero default (the example's own base
# leaves both at 0) -- with more basis functions than data points at this
# project's current training-set scale, a little regularization keeps the
# least-squares problem better-posed rather than leaving it fully
# underdetermined. kappa (energy-vs-force loss balance) is deliberately
# NOT set here -- it's this ensemble's own varied axis, see get_kappa_grid.
DEFAULT_FIT_CONFIG = {
    "optimizer": "BFGS",
    "maxiter": 100,
    "fit_cycles": 1,
}

# parallel_mode='serial' (pyace.paralleldataexecutor.ParallelDataExecutor.
# MODE_SERIAL, a real, documented mode -- not a workaround using an
# unsupported value), not 'process'. CONFIRMED (2026-09-17) necessary on
# this login node, not just a simplification for a tiny dataset: 'process'
# mode's ProcessPoolExecutor failed to os.fork() even a single additional
# worker process ("OSError: [Errno 12] Cannot allocate memory"), reproduced
# identically whether left at its own default (multiprocessing.cpu_count()
# = 32 workers here) or capped via the correct config key
# (backend.n_workers -- NOT "nworkers", confirmed via
# pyace.BACKEND_NWORKERS_KW's own string value -- set to 2). Since even 2
# workers couldn't be forked, this isn't really an over-parallelization
# problem at all (contrast pyace's own CMAKE_BUILD_PARALLEL_LEVEL install-
# time issue, where reducing the count DID fix it) -- more likely the same
# unpredictable, fluctuating login-node resource pressure already seen
# elsewhere in this pipeline (see the isolated-atom QE convergence saga in
# this same log). 'serial' sidesteps subprocess forking entirely, which
# also happens to be the right choice at this training set's current tiny
# scale regardless (no real parallelism benefit from multiple processes
# over a literal handful of structures) -- revisit toward 'process' (with
# an explicit, modest n_workers) once both the training set and the
# execution environment (a real allocation, not the login node) justify it.
DEFAULT_BACKEND_CONFIG = {
    "evaluator": "pyace",
    "parallel_mode": "serial",
}


def get_kappa_grid(kappa_range=(0.5, 0.99), kappa_num=3):
    """
    Linspace of `kappa` values -- fit.loss's energy-vs-force weighting
    balance (1.0 = energy-only, 0.0 = forces-only), ACE's analog of MACE's
    separate energy_weight/forces_weight pair. Returns a list of floats.
    """
    lo, hi = kappa_range
    if kappa_num == 1:
        return [float(lo)]
    step = (hi - lo) / (kappa_num - 1)
    return [float(lo + i * step) for i in range(kappa_num)]


def sample_ensemble_configs(seeds, kappa_values, total_cap, sample_seed):
    """
    Flat cartesian product of seeds x kappa_values, then a seeded random
    sample capped at `total_cap` (or the full product, if smaller) -- same
    shape as potential.mace.build_ensemble_inputs.sample_ensemble_configs.
    """
    full_product = list(itertools.product(seeds, kappa_values))
    rng = random.Random(sample_seed)
    n = min(total_cap, len(full_product))
    return rng.sample(full_product, n)


def write_ace_input_yaml(path, elements, dataset_path, seed, kappa,
                          potential_config=None, fit_config=None, backend_config=None):
    """
    Write one self-contained pacemaker input.yaml: DEFAULT_POTENTIAL_CONFIG/
    DEFAULT_FIT_CONFIG/DEFAULT_BACKEND_CONFIG above, each with the caller's
    own overrides layered on top (pass only the keys you want to add/change,
    matching write_mace_config's own convention), then this member's own
    `elements`/`seed`/`data.filename`/`fit.loss.kappa` written in last.
    """
    potential = dict(DEFAULT_POTENTIAL_CONFIG)
    if potential_config:
        potential.update(potential_config)
    potential["elements"] = list(elements)

    fit = dict(DEFAULT_FIT_CONFIG)
    if fit_config:
        fit.update(fit_config)
    fit["loss"] = {
        "kappa": kappa, "L1_coeffs": 1e-8, "L2_coeffs": 1e-8,
        "w1_coeffs": 0, "w2_coeffs": 0, "w0_rad": 0, "w1_rad": 0, "w2_rad": 0,
    }

    backend = dict(DEFAULT_BACKEND_CONFIG)
    if backend_config:
        backend.update(backend_config)

    full_config = {
        "seed": seed,
        "potential": potential,
        "data": {"filename": str(dataset_path)},
        "fit": fit,
        "backend": backend,
    }

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(full_config, f, default_flow_style=False, sort_keys=False)


def build_ace_ensemble_inputs(dataset_path, output_dir, elements,
                               seeds=(0, 1, 2), kappa_grid_kwargs=None,
                               total_cap=10, sample_seed=0,
                               potential_config=None, fit_config=None, backend_config=None,
                               input_filename="input.yaml"):
    """
    Materialize the randomly-sampled ensemble: numbered folders 0..N-1 under
    `output_dir`, each with input.yaml (this member's own sampled
    (seed, kappa) pair, referencing the SHARED dataset_path -- not copied
    per folder, matching how MACE's foundation_model is a single shared
    check_files-anchored file rather than duplicated into every mace_inputs
    folder) and METADATA (which seed/kappa this folder used).

    dataset_path is NOT validated to exist here (mirrors
    build_mace_ensemble_inputs' own division of labor -- dataset-building
    is a separate step, see build_ace_dataset.write_ace_dataset) -- a
    missing dataset only fails once pacemaker/GeneralACEFit actually tries
    to read it, at fit time.

    Returns the list of folder paths written.
    """
    kappa_grid_kwargs = kappa_grid_kwargs or {}
    kappa_values = get_kappa_grid(**kappa_grid_kwargs)
    sampled = sample_ensemble_configs(seeds, kappa_values, total_cap, sample_seed)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for i, (seed, kappa) in enumerate(sampled):
        folder = output_dir / str(i)
        folder.mkdir(parents=True, exist_ok=True)

        write_ace_input_yaml(
            folder / input_filename, elements, dataset_path, seed, kappa,
            potential_config=potential_config, fit_config=fit_config, backend_config=backend_config,
        )

        with open(folder / "METADATA", "w") as f:
            f.write(f"dataset: {dataset_path}\n")
            f.write(f"seed: {seed}\n")
            f.write(f"kappa (energy-vs-force loss weight): {kappa}\n")

        written.append(str(folder))

    return written
