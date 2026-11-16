"""
Driver script for running a single pyace/pacemaker ACE fit as a MatEnsemble
chore. Dispatched dynamically by FFMatEnsemble.run_individual (see
Ensemble-FF-Fit/EnsembleFFFit/base.py) via task_dict['ff_task']/
task_dict['entry_point'] -- same convention as FF/mace_fit.py in the other
Perlmutter examples, kept here (not in the installed package) since it's a
site-specific driver, not portable package logic.

Calls pyace's own GeneralACEFit class directly (in-process), the same
Python API pacemaker's own CLI (bin/pacemaker) calls internally.
GeneralACEFit itself is imported from pyace.generalfit, NOT re-exported at
pyace's own top level in the pinned git ref this example's Dockerfile.ace
installs (see that file's own comment for the exact pinned commit).

`config` (an overrides dict key, matching FFMatEnsemble's own per-run
config file convention) is a self-contained pacemaker input.yaml -- see
EnsembleFFFit.potential.ace.build_ace_ensemble_inputs.write_ace_input_yaml
-- with its own embedded data.filename already pointing at the shared
dataset, so this script never needs to separately consult any other
overrides key for the dataset path; it stays runnable standalone.

Still runnable standalone for manual testing via a single JSON *dict*
argument (parsed with json.loads), matching this script's actual per-fit
shape:
`python ace_fit.py '{"config": "...", "results_dir": "...", "work_dir": "...", "name": "FF_0"}'`

Ported as-is from the companion Pd-Se pipeline's own FF/ace_fit.py -- no
changes needed here for this example's nested <combo_label>/<member>/
nesting from the ACE Ensemble Fitting stage: this script only ever sees
one work_dir/results_dir per chore, so nesting depth above that is
invisible to it.
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
    build-time filter -- a caller relying on this chore's completion to
    trigger further work still needs the chore to run and succeed even
    when the fit was already done.

    Resume-from-checkpoint: if work_dir already has an
    interim_potential_<cycle>.yaml from a PREVIOUS, interrupted attempt at
    this exact member (e.g. a walltime timeout mid-fit), this run resumes
    from it instead of input.yaml's own potential config, automatically,
    with no config/flag needed. Confirmed via pyace/generalfit.py's own
    source which file is actually "most recent progress":
      - interim_potential_<cycle>.yaml (save_interim_potential_callback) is
        registered as a DEFAULT callback (self.callbacks =
        [save_interim_potential_callback] at GeneralACEFit construction)
        and fires on EVERY iteration via callback_hook -- always the live,
        latest snapshot of this exact fit's progress.
      - interim_potential_best_cycle.yaml is written only once
        cycle_fitting's own loop finishes ALL configured cycles -- NEVER
        written at all if killed mid-cycle, which is exactly what a
        walltime timeout does.
      - interim_potential_ladder_step_*.yaml is ladder-scheme-specific
        (basis-growth runs only) -- doesn't apply to this example's
        single-shot, same-basis-size fits, and the glob below's
        [0-9]-after-underscore pattern excludes it (and
        interim_potential_best_cycle.yaml) by construction.
    Overrides cfg["potential"] the same way the fitting stage's own
    warm_start_potential does (potential.filename, not
    potential.initial_potential), just decided here at execution time from
    whatever's actually on disk, so a resubmitted fit automatically
    continues each member from where IT left off rather than every member
    restarting from the same shared, generation-level warm-start potential
    again. A fresh member folder (no prior attempt) has no such file, so
    this is a no-op there -- input.yaml's own potential config is used
    unchanged.
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

        checkpoints = glob.glob(os.path.join(work_dir, "interim_potential_[0-9]*.yaml"))
        if checkpoints:
            resume_from = max(checkpoints, key=os.path.getmtime)
            print(f"ace_fit: {name} resuming from its own checkpoint {resume_from} "
                  f"(found in work_dir, overriding input.yaml's own potential config)", file=sys.stderr)
            cfg["potential"] = {"filename": os.path.abspath(resume_from), "reset": False}

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
    # written to f"{work_dir}/{name}.yaml".
    return {"status": "complete", "results_dir": results_dir, "name": name}


if __name__ == "__main__":
    run_ace_fit(json.loads(sys.argv[1]))
