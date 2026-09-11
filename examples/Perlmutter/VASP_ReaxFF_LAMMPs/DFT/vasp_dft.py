"""
Driver script for running VASP DFT calculations as MatEnsemble chores.
Dispatched dynamically by DFTMatEnsemble.run_individual (see
Ensemble-FF-Fit/EnsembleFFFit/base.py) via task_dict['dft_task']/
task_dict['entry_point'] -- this file is deliberately outside the
Ensemble-FF-Fit git repo, since it's a site-specific test driver rather than
portable package logic (same convention as DFT/rmg_dft.py, the Frontier/RMG
driver this replaces for the Perlmutter VASP_ReaxFF_LAMMPS pipeline).

Same loading convention as rmg_dft.py: a plain entry function taking
parallel lists of inputs, runnable standalone via
`python vasp_dft.py '["working_dir1", ...]' '["vasp1.yml", ...]'` using
parse_list to parse each sys.argv entry.

Unlike rmg_dft.py (which reads rmg_name/rmg_executable/command/etc. out of
the yaml because RMGInput.from_yaml needs them at input-generation time
too), VASP's input generation (pymatgen's VaspInputSet subclasses) only
needs user_incar_settings/user_kpoints_settings -- everything else
(vasp_executable, command, potcar_functional, pseudopotentials_directory,
band_multiplier, nodes/atoms_per_node, input_set) is read directly here as
ordinary top-level yaml keys, mirroring logic_locations/VASP/generate_vasp_flux_cli.py's
CLI-argument shape but as yaml config instead of argparse flags (this driver
runs one job per chore, launched by Flux -- there's no separate
submission-script generation step the way that SLURM-native script had).

input_set (yaml key, default "MPRelaxSet") names a
pymatgen.io.vasp.sets.VaspInputSet subclass by its plain class name,
resolved dynamically via getattr rather than a single hardcoded import --
see _resolve_input_set below. Any subclass sharing MPRelaxSet's
(structure, user_incar_settings=, user_kpoints_settings=,
user_potcar_functional=) constructor shape works (MPStaticSet,
MPScanRelaxSet, MITRelaxSet, MVLRelax52Set, MPMetalRelaxSet, ...);
specialized sets needing extra prev-run-derived arguments (MPNonSCFSet,
MPHSEBSSet, ...) are not supported by this generic mechanism.

Structure resolution: prefers CONTCAR (a continuation of a previous
relaxation attempt in the same working_directory) over structure_filename
(e.g. POSCAR), mirroring generate_vasp_flux_cli.py's call_write_vaspin --
deliberately simpler than RMG's pick_structure.pick_best_structure (no
RMG-log-derived structure concept applies here).

pseudopotentials_directory is set via pymatgen.core.SETTINGS at runtime
(a process-local override, not a persistent ~/.pmgrc.yaml/PMG_VASP_PSP_DIR
env var) -- per Perlmutter_Pipeline_Wiring.md's flagged open design
question, an explicit yaml key survives container rebuilds/migrations and
is visible in one place, unlike container-launch-time env vars or an
in-container persistent config file that likely won't survive a
`podman-hpc migrate`.

`command` mirrors RMG's own convention (see rmg_dft.py's UNVERIFIED note):
a bare `{vasp_executable}` invocation by default, no srun/mpirun/flux run
wrapper -- Flux owns launch semantics for every chore. Overridable via the
yaml's own 'command' key if that turns out not to be correct in practice.
UNVERIFIED against a real allocation -- Perlmutter's compute nodes were
degraded for the entirety of this pipeline's development (see
Perlmutter_Build_Order.md); re-check this the first time a real VASP chore
actually runs.

VASP's own LD_LIBRARY_PATH/module-derived environment (Cray MPICH/libSci/
libfabric, NVHPC compiler runtime -- see this pipeline's earlier `ldd
$(which vasp_std)` inspection, and vasp_container_env in run_pipeline.py) is
passed in via the chore's own env as VASP_LD_LIBRARY_PATH, NOT the literal
LD_LIBRARY_PATH -- CONFIRMED (2026-09-03) that the literal name, applied to
this whole chore process (not just vasp_std's own subprocess), segfaults a
bare `python3 -c "print(1)"` (rc=139) via the exact same host-library-
shadowing bug VASP_LD_LIBRARY_PATH's own docstring in run_pipeline.py
describes for the container-launch level, just one layer deeper (it
contains /host_lib64:/host_lib -- real host libraries, including the
host's own libc.so.6, appropriate for vasp_std specifically, wrong for this
Python process's own dynamic linking). run_vasp_calculation below reads
VASP_LD_LIBRARY_PATH back out of os.environ and applies it as
LD_LIBRARY_PATH ONLY to vasp_std's own subprocess env=, never to this
process's own environment.
"""
import os
import subprocess
import sys
import json
from copy import deepcopy

import yaml
import pymatgen.io.vasp.sets as vasp_input_sets
from pymatgen.core import SETTINGS
from pymatgen.core.structure import Structure
from pymatgen.io.vasp.sets import VaspInputSet
from pymatgen.io.vasp.outputs import Vasprun

from EnsembleFFFit.utilities.general import parse_list

# 1 kBar = 1e8 Pa = 1e8 / 1.602176634e-19 / 1e30 eV/Angstrom^3
KBAR_TO_EV_PER_A3 = 6.241509074e-4


def _resolve_input_set(input_set_name, vasp_yaml):
    """
    Resolve a pymatgen.io.vasp.sets.VaspInputSet subclass by plain class
    name (e.g. "MPRelaxSet", "MPStaticSet") -- a dynamic getattr lookup
    instead of a single hardcoded `from pymatgen.io.vasp.sets import
    MPRelaxSet`, so a vasp_yaml can select a different input set via its
    own `input_set` key. issubclass-checked (not just "does the module have
    an attribute by this name") so a typo or a non-input-set name fails
    loudly here rather than surfacing as a confusing constructor-signature
    error further down.
    """
    input_set = getattr(vasp_input_sets, input_set_name, None)
    if not (isinstance(input_set, type) and issubclass(input_set, VaspInputSet)):
        raise ValueError(
            f"{vasp_yaml} sets input_set={input_set_name!r}, but pymatgen.io.vasp.sets has no "
            f"VaspInputSet subclass by that name. Common options: MPRelaxSet (default), MPStaticSet, "
            f"MPScanRelaxSet, MITRelaxSet, MVLRelax52Set, MPMetalRelaxSet -- anything sharing "
            f"MPRelaxSet's (structure, user_incar_settings=, user_kpoints_settings=, "
            f"user_potcar_functional=) constructor shape."
        )
    return input_set


def _pick_vasp_structure(working_directory, structure_filename):
    """
    Prefer a CONTCAR left by a previous attempt in this same
    working_directory (continuation) over the original structure_filename
    (e.g. POSCAR) -- mirrors generate_vasp_flux_cli.py's
    call_write_vaspin/write_vaspin continuation logic, simplified to a
    single execution-time resolution (no separate submission-script pass).
    """
    contcar_path = os.path.join(working_directory, 'CONTCAR')
    if os.path.isfile(contcar_path) and os.path.getsize(contcar_path) > 0:
        try:
            return Structure.from_file(contcar_path), contcar_path
        except Exception:
            pass  # empty/malformed CONTCAR shouldn't abort the run -- fall through to structure_filename
    structure_path = os.path.join(working_directory, structure_filename)
    return Structure.from_file(structure_path), structure_path


def _get_nbands(nelect, nions, multiplier):
    """Same formula as generate_vasp_flux_cli.py's get_nbands."""
    return int(max(nelect / 2 + nions / 2, nelect * multiplier))


def run_vasp_calculation(working_directory_list, vasp_yaml_list):
    n = len(working_directory_list)
    assert len(vasp_yaml_list) == n, "working_directory_list and vasp_yaml_list must be the same length"

    for working_directory, vasp_yaml in zip(working_directory_list, vasp_yaml_list):
        with open(vasp_yaml, 'r') as f:
            yaml_args = yaml.safe_load(f)

        vasp_executable = yaml_args.get('vasp_executable', 'vasp_std')
        command = yaml_args.get('command', vasp_executable)
        structure_filename = yaml_args.get('structure_filename', 'POSCAR')
        potcar_functional = yaml_args.get('potcar_functional', 'PBE_54_W_HASH')
        band_multiplier = yaml_args.get('band_multiplier', 0.6)
        pseudopotentials_directory = yaml_args.get('pseudopotentials_directory')
        user_incar_settings = deepcopy(yaml_args.get('user_incar_settings', {}))
        user_kpoints_settings = yaml_args.get('user_kpoints_settings', {'grid_density': 1000})
        input_set_name = yaml_args.get('input_set', 'MPRelaxSet')
        InputSet = _resolve_input_set(input_set_name, vasp_yaml)

        if pseudopotentials_directory:
            SETTINGS["PMG_VASP_PSP_DIR"] = pseudopotentials_directory

        structure, source = _pick_vasp_structure(working_directory, structure_filename)

        base_set = InputSet(structure, user_potcar_functional=potcar_functional)
        if 'NBANDS' not in user_incar_settings:
            user_incar_settings['NBANDS'] = _get_nbands(base_set.nelect, len(structure), band_multiplier)

        relax_set = InputSet(structure, user_incar_settings=user_incar_settings,
                              user_kpoints_settings=user_kpoints_settings,
                              user_potcar_functional=potcar_functional)
        relax_set.potcar.write_file(os.path.join(working_directory, 'POTCAR'))
        relax_set.kpoints.write_file(os.path.join(working_directory, 'KPOINTS'))
        relax_set.incar.write_file(os.path.join(working_directory, 'INCAR'))
        structure.to(fmt='poscar', filename=os.path.join(working_directory, 'POSCAR'))

        print(f"{working_directory}: structure from {source}")

        # vasp_std's own env is built here, scoped to JUST this subprocess --
        # NOT inherited as-is from this whole chore process's own os.environ.
        # VASP_LD_LIBRARY_PATH (see vasp_container_env's docstring in
        # run_pipeline.py) is deliberately passed under that name, not the
        # literal LD_LIBRARY_PATH, at the chore-env level specifically so it
        # can't affect this Python process's own dynamic linking (confirmed
        # to segfault a bare `python3 -c "print(1)"` if it does) -- applied
        # to vasp_std's own subprocess env as LD_LIBRARY_PATH here, and only
        # here, since vasp_std (unlike this Python process) genuinely needs
        # it to resolve its Cray MPICH/NVHPC runtime dependencies.
        vasp_env = dict(os.environ)
        if 'VASP_LD_LIBRARY_PATH' in vasp_env:
            vasp_env['LD_LIBRARY_PATH'] = vasp_env['VASP_LD_LIBRARY_PATH']

        # Bare command, no srun/mpirun/flux run wrapper -- Flux owns launch
        # semantics for every chore (same convention as RMG's calc.command,
        # see rmg_dft.py). shell=True to match rmg_dft.py's precedent;
        # command is always either a bare executable name/path or whatever
        # the yaml's own 'command' key set, never built from untrusted input.
        result = subprocess.run(
            command, shell=True, cwd=working_directory, close_fds=False,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            env=vasp_env,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"vasp_std failed (exit {result.returncode}) in {working_directory}:\n{result.stdout}"
            )

        vasprun = Vasprun(os.path.join(working_directory, 'vasprun.xml'))
        if not vasprun.converged:
            raise RuntimeError(f"VASP run in {working_directory} did not converge (vasprun.xml).")

        final_step = vasprun.ionic_steps[-1]
        forces = final_step['forces']

        property_dictionary = {'energy': vasprun.final_energy}
        property_dictionary['fx'] = [forces[i][0] for i in range(len(forces))]
        property_dictionary['fy'] = [forces[i][1] for i in range(len(forces))]
        property_dictionary['fz'] = [forces[i][2] for i in range(len(forces))]

        stress = final_step.get('stress')
        if stress is not None:
            # Unit conversion (kBar -> eV/Angstrom^3) is a solid physical
            # constant; the SIGN convention relative to rmg_dft.py's ASE-Voigt
            # output (xx, yy, zz, yz, xz, xy) is NOT verified here -- VASP and
            # ASE are known to use opposite stress sign conventions in
            # general, but confirming the exact sign for this pymatgen/VASP
            # version combination needs a real run to check against a known
            # reference, which wasn't possible while Perlmutter's compute
            # nodes were degraded. Treat any stress-based downstream result
            # (e.g. cell-relaxation-driven training data) as unverified until
            # checked against a hand-computed reference.
            property_dictionary['stress_voigt'] = [
                stress[0][0] * KBAR_TO_EV_PER_A3, stress[1][1] * KBAR_TO_EV_PER_A3, stress[2][2] * KBAR_TO_EV_PER_A3,
                stress[1][2] * KBAR_TO_EV_PER_A3, stress[0][2] * KBAR_TO_EV_PER_A3, stress[0][1] * KBAR_TO_EV_PER_A3,
            ]

        with open(os.path.join(working_directory, 'properties.json'), "w") as f:
            json.dump(property_dictionary, f, indent=4)

    return {"status": "complete"}


if __name__ == "__main__":
    run_vasp_calculation(parse_list(sys.argv[1]), parse_list(sys.argv[2]))
