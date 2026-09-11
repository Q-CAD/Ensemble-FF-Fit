"""
Driver script for running a single JAX-ReaxFF fit as a MatEnsemble chore.
Dispatched dynamically by FFMatEnsemble.run_individual (see
Ensemble-FF-Fit/EnsembleFFFit/base.py) via task_dict['ff_task']/
task_dict['entry_point'] -- same convention as FF/mace_fit.py, kept in the
example rather than the installed package since it's a site-specific
driver, not portable package logic (see mace_fit.py's own docstring for
why FFMatEnsemble.run_individual hands the whole overrides dict to this
kind of driver as one argument rather than unpacking into positional
slots).

Targets JAX-ReaxFF/jaxreaxff/driver.py specifically, NOT driver_v2.py --
driver.py's CLI takes --init_FF/--params/--geo/--train_file directly
(matching Perlmutter_Pipeline_Wiring.md's ffield/params/geo+trainset.in
conceptual mapping exactly); driver_v2.py instead expects a pre-built
.pickle dataset via --data_file, which nothing in this pipeline currently
produces (confirmed by inspecting both files -- see driver_v2.py's own
build_arg_parser()/run() split, done for future flexibility, but unused by
this driver today). Both driver.py and driver_v2.py were given the same
build_arg_parser()/run(args) split (mirroring mace.cli.run_train's
Namespace-based entry point) specifically so either can be called
in-process like this, without shelling out to them as subprocesses.

Runnable standalone for manual testing, same as mace_fit.py -- via a single
JSON *dict* argument (parsed with json.loads), matching this script's
per-fit shape:
`python jax_reaxff_fit.py '{"init_FF": "...", "params": "...", "geo": "...", "train_file": "...", "out_folder": "...", "name": "FF_0"}'`

Override keys are deliberately driver.py's own argparse attribute names
(init_FF/params/geo/train_file/valid_file/valid_geo_file/opt_method/
num_trials/num_steps/...), not a friendlier renamed set -- `setattr(args,
key, value)` below only does the right thing if `key` already matches a
real attribute name on the parsed Namespace; a translation layer between
"nicer" override key names and driver.py's actual attribute names would be
an easy place to introduce a silent mismatch (e.g. setting an unrelated
new `args.ffield` attribute instead of the real `args.init_FF`) that
nothing would catch without actually running a fit. See
workflow_config.yaml's fine_tuning section for how these get set.
"""
import os
import sys
import json
import glob


def run_jax_reaxff_fit(overrides):
    """
    Run a single JAX-ReaxFF fit from a per-run `overrides` dict
    (init_FF/params/geo/train_file/out_folder/results_dir/work_dir/name/...).
    Imports jaxreaxff.driver lazily so this script is only ever loaded (via
    FFMatEnsemble.run_individual's import_module_from_path) inside a chore
    that actually has JAX-ReaxFF's dependencies installed.
    """
    from jaxreaxff.driver import build_arg_parser, run

    name = overrides.get("name", "MatEnsemble")

    # Start from driver.py's own defaults (parse_args([]) -- no argv, just
    # every --flag's default value), then override with whatever this run's
    # overrides dict actually sets -- same two-step shape as mace_fit.py's
    # build_default_arg_parser().parse_args(initial_args) + setattr loop.
    args = build_arg_parser().parse_args([])

    for key, value in overrides.items():
        if key in ("name", "results_dir", "work_dir", "finished_file", "ff_task", "entry_point"):
            continue  # handled separately below / not a driver.py CLI flag
        setattr(args, key, value)

    # driver.py writes fitted force fields to f"{args.out_folder}/new_FF_...";
    # out_folder defaults to a plain "outputs" (driver.py's own --out_folder
    # default) unless overrides sets it explicitly -- point it at this run's
    # own directory so concurrent ensemble members don't collide.
    out_folder = overrides.get("out_folder") or overrides.get("work_dir")
    if out_folder:
        args.out_folder = out_folder
        os.makedirs(out_folder, exist_ok=True)

    # `finished_file` here is deliberately an execution-time skip, not a
    # build-time filter -- same reasoning as mace_fit.py's own finished_file
    # handling (see that docstring): a caller relying on this chore's
    # *completion* to trigger further work needs the chore to still run and
    # succeed even when the fit itself was already done.
    finished_file = overrides.get("finished_file")
    already_done = bool(finished_file and args.out_folder and glob.glob(os.path.join(args.out_folder, finished_file)))
    if not already_done:
        run(args)

    return {"status": "complete", "results_dir": overrides.get("results_dir"), "name": name}


if __name__ == "__main__":
    run_jax_reaxff_fit(json.loads(sys.argv[1]))
