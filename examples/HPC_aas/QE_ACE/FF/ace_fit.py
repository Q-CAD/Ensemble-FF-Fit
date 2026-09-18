"""
Driver script for running a single pyace/pacemaker ACE fit as a MatEnsemble
chore. Dispatched dynamically by FFMatEnsemble.run_individual (see
Ensemble-FF-Fit/EnsembleFFFit/base.py) via task_dict['ff_task']/
task_dict['entry_point'] -- same convention as FF/mace_fit.py, kept here
(not in the installed package) since it's a site-specific driver, not
portable package logic -- mirrors mace_fit.py's own precedent exactly
(MACE's own fitting driver has no package-level reference copy either,
unlike the DFT drivers, which do -- see CLAUDE.md's own note on this).

Calls pyace's own GeneralACEFit class directly (in-process), the same
Python API pacemaker's own CLI (bin/pacemaker) calls internally --
confirmed by reading bin/pacemaker's own source, not assumed. Mirrors
mace_fit.py's own convention of calling mace.cli.run_train.run(args)
directly rather than shelling out to a CLI. GeneralACEFit itself is
imported from pyace.generalfit, NOT re-exported at pyace's own top level in
the currently-pinned git ref (confirmed 2026-09-17) -- see
pipeline/FRICTION_LOG.md.

`config` (an overrides dict key, matching FFMatEnsemble's own per-run
config file convention) is a self-contained pacemaker input.yaml -- see
potential.ace.build_ace_ensemble_inputs.write_ace_input_yaml -- with its
own embedded data.filename already pointing at the shared dataset, so this
script never needs to separately consult any other overrides key for the
dataset path; it stays runnable standalone.

Still runnable standalone for manual testing, same as mace_fit.py -- via a
single JSON *dict* argument (parsed with json.loads), matching this
script's actual per-fit shape:
`python ace_fit.py '{"config": "...", "results_dir": "...", "work_dir": "...", "name": "FF_0"}'`
"""
import glob
import json
import os
import sys

import yaml


def run_ace_fit(overrides):
    """
    Run a single ACE fit from a per-run `overrides` dict (config/
    results_dir/work_dir/name/...). Imports pyace lazily so this script is
    only ever loaded (via FFMatEnsemble.run_individual's
    import_module_from_path) inside a chore that actually has the `ace`
    extra installed.

    `finished_file` here is deliberately an execution-time skip, not a
    build-time filter (matches mace_fit.py's own reasoning) -- a caller
    relying on this chore's completion to trigger further work still needs
    the chore to run and succeed even when the fit was already done.
    """
    config_path = os.path.abspath(overrides["config"])
    name = overrides.get("name", "ACE")
    work_dir = overrides.get("work_dir")
    results_dir = overrides.get("results_dir", work_dir)
    finished_file = overrides.get("finished_file")

    already_done = bool(finished_file and results_dir and glob.glob(os.path.join(results_dir, finished_file)))
    if already_done:
        return {"status": "complete", "results_dir": results_dir, "name": name}

    from pyace.generalfit import GeneralACEFit

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if work_dir:
        os.makedirs(work_dir, exist_ok=True)

    # GeneralACEFit/pacemaker's own CLI write several working files
    # (interim potentials, metrics) relative to the current directory --
    # chdir into work_dir for the duration of the fit so those land
    # alongside this run's own input.yaml/METADATA, then restore cwd
    # regardless of outcome.
    cwd = os.getcwd()
    try:
        if work_dir:
            os.chdir(work_dir)
        general_fit = GeneralACEFit(
            potential_config=cfg["potential"],
            fit_config=cfg["fit"],
            data_config=cfg["data"],
            backend_config=cfg["backend"],
            seed=cfg.get("seed"),
        )
        general_fit.fit()
        general_fit.set_core_rep(general_fit.target_bbasisconfig)
        general_fit.save_optimized_potential(f"{name}.yaml")
    finally:
        os.chdir(cwd)

    # results_dir/name included (not just status) so callers watching this
    # chore's completion can locate the fitted potential file directly --
    # written to f"{work_dir}/{name}.yaml", matching MACE's own
    # f"{results_dir}/{name}.model" convention (work_dir/results_dir are
    # the same value in FFMatEnsemble's own build_ff_dcts, same as MACE's).
    return {"status": "complete", "results_dir": results_dir, "name": name}


if __name__ == "__main__":
    run_ace_fit(json.loads(sys.argv[1]))
