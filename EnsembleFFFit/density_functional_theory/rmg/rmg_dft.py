"""
Driver script for running RMG DFT calculations as MatEnsemble chores.
Dispatched dynamically by DFTMatEnsemble.run_individual (see
Ensemble-FF-Fit/EnsembleFFFit/base.py) via task_dict['dft_task']/
task_dict['entry_point']. This is the package-level reference copy of the
driver -- analogous to molecular_dynamics/ase/ase_mace.py -- kept here so
there's a canonical, importable-by-path template independent of any one
example. A deployment's actual dft_task should point at its own copy (e.g.
examples/Frontier/RMG_MACE_ASE/DFT/rmg_dft.py), hand-copied from here (or
vice versa) rather than importing this file directly, matching the existing
convention documented in CLAUDE.md for the MD drivers.

Imports from pyRMG (an optional dependency -- the `rmg` extra), not a local
copy: pick_structure/rmg_input/rmg_calculator's actual logic lives there now,
not in this package.

A plain entry function taking parallel lists of inputs, runnable standalone
via `python rmg_dft.py '["working_dir1", ...]' '["rmg1.yaml", ...]'` using
parse_list to parse each sys.argv entry -- same loading convention as
ase_mace.py.

Unlike ase_mace.py's ffield/structure/output triple, only two lists are
needed here (working_directory, rmg_yaml) -- everything else (rmg_name,
rmg_executable, command, structure_filename, gpus_per_node, electrons_per_gpu,
grid_divisibility_exponent, allocated_nodes) is read directly out of each
rmg_yaml file below, as ordinary extra top-level YAML keys alongside RMG's
own keywords. gpus_per_node/electrons_per_gpu/grid_divisibility_exponent are
popped out by rmg_input.compute_grid_and_resources before the rest of the
yaml's contents get written into the generated rmg_input file, so they're
safe to add there -- unlike those three, RMG itself has no notion of
rmg_name/rmg_executable/command/structure_filename/allocated_nodes, so those
four are read directly here and never touch RMGInput at all.
pseudopotentials_directory is deliberately not read here: it's already
handled internally by RMGInput.from_yaml via the genuine 'pseudo_dir'/
'pseudopotential' RMG keywords, which should already be in the same yaml.

Resolves the structure to actually run against at execution time (via
pick_best_structure), not from any static path baked in ahead of time: a
previous RMG run in the same working_directory may since have produced a
newer rmg_input.*.log or rmg_input than whatever structure_filename points at.

RMG's own default `command` (a bare `{rmg_executable} {rmg_name}` invocation,
no srun/mpirun/flux-run wrapper) is confirmed working for genuinely
multi-task/multi-GPU chores (num_tasks=8 on Frontier) -- Flux/MatEnsemble
owns launch semantics for the surrounding chore via the chore's own
Resources, so wrapping the command again here would conflict with that. The
subprocess call below deliberately uses shell=True (not shell=False/shlex),
confirmed via a controlled A/B against an otherwise-identical script that
shell=False is an actual regression here, not just a style choice. See
examples/Frontier/RMG_MACE_ASE/README.md's pitfalls section for the full
story (this took a long debugging pass to pin down -- the eventual root
cause was RMG's `kohn_sham_solver` needing to be "davidson", not
"multigrid", under this Flux/Apptainer setup, not the launch mechanism at
all).
"""
import os
import subprocess
import sys
import json

import numpy as np
import yaml
from ase.calculators.calculator import all_changes
from pymatgen.io.ase import AseAtomsAdaptor

from EnsembleFFFit.utilities.general import parse_list
from pyRMG.pick_structure import pick_best_structure
from pyRMG.rmg_input import RMGInput
from pyRMG.rmg_calculator import RMG


def run_rmg_calculation(working_directory_list, rmg_yaml_list):
    n = len(working_directory_list)
    assert len(rmg_yaml_list) == n, "working_directory_list and rmg_yaml_list must be the same length"

    for working_directory, rmg_yaml in zip(working_directory_list, rmg_yaml_list):
        with open(rmg_yaml, 'r') as f:
            yaml_args = yaml.safe_load(f)

        rmg_name = yaml_args.get('rmg_name', 'rmg_input')
        rmg_executable = yaml_args.get('rmg_executable', 'rmg-gpu')
        command = yaml_args.get('command')
        structure_filename = yaml_args.get('structure_filename', 'POSCAR')
        allocated_nodes = yaml_args.get('allocated_nodes')

        structure, source = pick_best_structure(
            working_directory, structure_filename=structure_filename, rmg_name=rmg_name)

        if allocated_nodes is None:
            # No node count pinned in the yaml -- fall back to a fresh
            # estimate against whichever structure pick_best_structure just
            # resolved, exactly like DFTMatEnsemble.build_dft_dcts does at
            # task-construction time, so this script is self-sufficient for
            # standalone testing without a Pipeline/chore wired up around it.
            probe = RMGInput.from_yaml(yaml_path=rmg_yaml, structure_path=None,
                                        structure_obj=structure, target_nodes=0)
            allocated_nodes = probe.target_nodes

        atoms = AseAtomsAdaptor.get_atoms(structure)
        # Carried explicitly (rather than relying on AseAtomsAdaptor round-tripping
        # site_properties) so RMG.write_input can recover them via atoms.arrays --
        # see that class's docstring for the contract this fulfills.
        # pymatgen's site_properties come back as plain Python lists, but
        # Atoms.set_array checks a.flags['C_CONTIGUOUS'] without converting --
        # needs an actual ndarray first.
        if 'selective_dynamics' in structure.site_properties:
            atoms.set_array('selective_dynamics', np.array(structure.site_properties['selective_dynamics']))
        if 'magnetic_properties' in structure.site_properties:
            atoms.set_array('magnetic_properties', np.array(structure.site_properties['magnetic_properties']))

        calc = RMG(
            rmg_yaml=rmg_yaml,
            allocated_nodes=allocated_nodes,
            directory=working_directory,
            rmg_name=rmg_name,
            rmg_executable=rmg_executable,
            command=command,
        )
        # Bypassing atoms.calc/get_potential_energy()'s normal write_input ->
        # execute() -> read_results() cycle here -- ASE's own execute() spawns
        # rmg-gpu via a plain subprocess call with Python's default
        # close_fds=True, which silently drops any inherited file descriptor
        # beyond stdin/stdout/stderr. Under Flux, PMI_Init needs the open
        # socket PMI_FD points at (not just the env var, which does survive
        # close_fds=True since it's just a string) to actually rendezvous --
        # losing it here is the leading explanation for PMI_Init returning -1
        # only when launched through a MatEnsemble chore (an extra
        # process-subprocess hop versus the validated direct `flux run
        # rmg-gpu` case, where Flux execs the binary itself with no
        # intervening subprocess boundary to lose anything across).
        # write_input/read_results are reused as-is -- only the middle
        # execute() step is replaced, with close_fds=False so PMI_FD (and
        # anything else Flux opened for this process) survives into rmg-gpu.
        #
        # shell=True: matches test/RMG_testing/GPU_run/rmg_dft.py (the
        # earlier, confirmed-working version of this same driver) exactly.
        # shell=False (splitting calc.command via shlex instead) was tried
        # here on the theory that shell=True's extra /bin/sh -c hop could be
        # dropping PMI_FD -- that was never actually the cause (PMI_FD was
        # separately confirmed present and still a live, open socket at this
        # exact point even under shell=False, via os.fstat), and switching to
        # shell=False is the one concrete, controlled difference found
        # between a still-failing Consolidated_Pipeline run and an otherwise
        # equivalent (same env, same resources, same task count) GPU_run
        # rmg_pipeline.py run that succeeds. calc.command is always a bare
        # "{rmg_executable} {rmg_name}" (see rmg_calculator.py -- no shell
        # metacharacters by construction), so shell=True is safe here.
        calc.write_input(atoms, properties=['energy', 'forces'], system_changes=all_changes)

        result = subprocess.run(
            calc.command, shell=True, cwd=calc.directory, close_fds=False,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"rmg-gpu failed (exit {result.returncode}) in {calc.directory}:\n{result.stdout}"
            )

        calc.read_results()
        energy = calc.results['energy']
        forces = calc.results['forces']
        stress = calc.results.get('stress')  # Voigt-6, eV/Angstrom^3 -- absent if the yaml didn't set stress: True

        print(f"{working_directory}: structure from {source}, target_nodes={calc.target_nodes}")

        property_dictionary = {}
        property_dictionary['energy'] = energy
        property_dictionary['fx'] = [forces[i][0] for i in range(len(forces))]
        property_dictionary['fy'] = [forces[i][1] for i in range(len(forces))]
        property_dictionary['fz'] = [forces[i][2] for i in range(len(forces))]
        if stress is not None:
            # ASE Voigt-6 order: xx, yy, zz, yz, xz, xy -- eV/Angstrom^3
            property_dictionary['stress_voigt'] = list(stress)

        with open(os.path.join(working_directory, 'properties.json'), "w") as f:
            json.dump(property_dictionary, f, indent=4)

    return {"status": "complete"}


if __name__ == "__main__":
    run_rmg_calculation(parse_list(sys.argv[1]), parse_list(sys.argv[2]))
