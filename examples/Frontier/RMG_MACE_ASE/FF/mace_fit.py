"""
Driver script for running a single MACE fit as a MatEnsemble chore.
Dispatched dynamically by FFMatEnsemble.run_individual (see
Ensemble-FF-Fit/EnsembleFFFit/base.py) via task_dict['ff_task']/
task_dict['entry_point'] -- same convention as DFT/rmg_dft.py and
MD/*/ase_inputs/ase_mace*.py, kept in the example rather than the installed
package since it's a site-specific driver, not portable package logic.

Unlike those two (which take parallel lists of per-structure inputs, a
convention baked into DFTMatEnsemble/MDMatEnsemble's own run_individual
because RMG/MD each have a small, fixed number of positional slots --
working_directory+rmg_yaml, or ffield+structure+output+in_file -- that those
two backend-specific classes are allowed to know about), run_mace_fit takes
a single `overrides` dict instead. FFMatEnsemble.run_individual is
deliberately backend-agnostic (see EnsembleFFFit/base.py) -- it has no fixed
set of key names to unpack into positional lists the way DFT/MD's
run_individual does, since a different FF backend's overrides could
legitimately use different keys. An FF fit is also already one-per-chore
(FFMatEnsemble.build_ff_dcts never batches multiple fits into one
task_dict), so there's nothing to batch here either way.

Still runnable standalone for manual testing, same as rmg_dft.py/
ase_mace_md.py -- just via a single JSON *dict* argument (parsed with
json.loads) instead of parse_list's JSON *list* argument, matching this
script's actual per-fit shape:
`python mace_fit.py '{"foundation_model": "...", "config": "...", "train_file": "...", "test_file": "...", "results_dir": "...", "work_dir": "...", "name": "FF_0"}'`

All the actual MACE-fitting logic below (building a mace argparse Namespace,
the finished_file execution-time skip, the results_dir/name return shape)
is unchanged from what used to live inline in FFMatEnsemble.run_individual
(back when that class was MACEMatEnsemble) -- only its location moved, so
porting to a different FF backend means writing a new driver script with the
same run_<backend>_fit(overrides) entry-point contract, not touching
EnsembleFFFit/base.py at all.
"""
import os
import sys
import json
import glob


def run_mace_fit(overrides):
    """
    Run a single MACE fit from a per-run `overrides` dict (foundation_model/
    config/train_file/test_file/results_dir/work_dir/name/...). Imports
    `mace` lazily so this script is only ever loaded (via
    FFMatEnsemble.run_individual's import_module_from_path) inside a chore
    that actually has the `mace` extra installed.
    """
    from mace.cli.run_train import run
    from mace.tools import build_default_arg_parser

    name = overrides.get("name", "MatEnsemble")
    config_path = overrides.get("config")

    initial_args = ["--name", name]
    if config_path:
        initial_args += ["--config", config_path]

    args = build_default_arg_parser().parse_args(initial_args)

    for key, value in overrides.items():
        if key in ("name", "config", "finished_file"):
            continue  # already handled above / below
        setattr(args, key, value)

    work_path = overrides.get("work_dir")
    if work_path:
        os.makedirs(work_path, exist_ok=True)

    # `finished_file` here is deliberately an execution-time skip, not a
    # build-time filter (contrast FFMatEnsemble.build_ff_dcts's own
    # finished_file param) -- a caller relying on this chore's *completion*
    # to trigger further work (e.g. a Pipeline.strategy processing chore)
    # needs the chore to still run and succeed even when the fit itself was
    # already done, rather than never being submitted at all.
    finished_file = overrides.get("finished_file")
    already_done = bool(finished_file and work_path and glob.glob(os.path.join(work_path, finished_file)))
    if not already_done:
        run(args)

    # results_dir/name are included (not just status) so callers watching
    # this chore's completion (e.g. a Pipeline.strategy processing chore)
    # can locate the fitted model file directly, without re-walking the
    # run_directory -- MACE writes it to f"{results_dir}/{name}.model".
    return {"status": "complete", "results_dir": overrides.get("results_dir"), "name": name}


if __name__ == "__main__":
    run_mace_fit(json.loads(sys.argv[1]))
