"""
Driver script for running Quantum Espresso (pw.x) DFT calculations as
MatEnsemble chores. Dispatched dynamically by DFTMatEnsemble.run_individual
(see Ensemble-FF-Fit/EnsembleFFFit/base.py) via task_dict['dft_task']/
task_dict['entry_point'] -- same convention as rmg_dft.py/vasp_dft.py. This
is the package-level reference copy (analogous to
density_functional_theory/rmg/rmg_dft.py, molecular_dynamics/ase/ase_mace.py)
-- a deployment's actual dft_task should point at its own copy (e.g.
examples/Pathfinder/.../DFT/qe_dft.py), hand-kept-in-sync with this one, same
convention as the other DFT/MD drivers.

Same loading convention as rmg_dft.py/vasp_dft.py: a plain entry function
taking parallel lists of inputs, runnable standalone via
`python qe_dft.py '["working_dir1", ...]' '["single_point1.yml", ...]'`
using parse_list to parse each sys.argv entry.

Input generation and output parsing go through ASE's
ase.calculators.espresso.EspressoTemplate/EspressoProfile directly -- NOT
pymatgen-io-espresso. pymatgen-io-espresso was evaluated and rejected: as of
2026-09 its input-generation side ("Caffeinator") is pre-alpha/work-in-progress
with no documented pseudopotential-resolution support, and it isn't published
on PyPI. ASE is already a core Ensemble-FF-Fit dependency and has mature,
documented QE I/O instead -- pymatgen-io-espresso's PWxml output parser (kept
API-compatible with pymatgen's own Vasprun) may still be worth revisiting
later purely for output parsing, but isn't needed here since ASE's own
EspressoTemplate.read_results already covers energy/forces/stress.

EspressoTemplate.execute() is deliberately bypassed here (it just calls
profile.run(), ASE's own subprocess launch) in favor of a bare
subprocess.run(..., close_fds=False) call -- same reasoning as rmg_dft.py's
own bypass of RMG's execute(): Flux owns launch semantics for the surrounding
chore, and close_fds=True (subprocess.run's default) silently drops any
inherited file descriptor an MPI rendezvous may need (see rmg_dft.py's own
PMI_FD finding). write_input/read_results are reused as-is -- only the
middle execute() step is replaced, matching rmg_dft.py's own pattern exactly.

Pseudopotential resolution: ASE does not auto-resolve pseudopotential
filenames per element (EspressoTemplate.write_input expects an explicit
{element: filename} dict). If the yaml's own 'pseudopotentials' key doesn't
name a given element, this resolves it by globbing
'{pseudopotentials_directory}/{Element}*.[uU][pP][fF]' -- errors loudly
(rather than guessing) if that glob finds zero or more than one match for a
given element.

Container-launch-level requirement, NOT handled here: pw.x (built via
Pathfinder's `module load quantum-espresso/7.4-mpi-omp`) is a Rocky Linux 9
binary; running it inside this project's Debian-based Apptainer container
needs the container launched with a bind mount of the host's real /lib64 at
a non-conflicting path (e.g. --bind /lib64:/host_lib64) and 'command' set to
invoke pw.x via that real loader explicitly (e.g.
"/host_lib64/ld-linux-x86-64.so.2 <path-to-pw.x>") -- exactly like
vasp_dft.py's own documented ELF-interpreter-mismatch fix for VASP on
Perlmutter (a bare invocation risks a silent SIGSEGV, not a clear error, since
Debian ships a different loader at that same path). Also needs the module's
own library tree bound in -- confirmed via `ldd $(which pw.x)` that Pathfinder's
entire QE dependency chain (fftw, hdf5, scalapack, lapack, openmpi, hwloc,
pmix, libevent, libxml2, ...) resolves under one single root,
/software/baseline/nsp/spack-envs/..., so one `--bind /software:/software`
should cover it.

QE_LD_LIBRARY_PATH (read back out of os.environ below, applied to pw.x's own
subprocess only) mirrors vasp_dft.py's own VASP_LD_LIBRARY_PATH convention --
deliberately NOT the literal LD_LIBRARY_PATH at the chore-env level, since
applying a Rocky-Linux-appropriate library path to this whole Python process
(rather than just pw.x's own subprocess) risks the same host-library-shadowing
segfault vasp_dft.py's docstring describes for VASP.

UNVERIFIED, needs empirical testing against a real allocation: whether
'command' wants a bare invocation (relying on Flux to own MPI launch
semantics, the RMG/VASP precedent) or an explicit mpirun wrapper --
Pathfinder's OpenMPI 5.0.5 bootstraps via PMIx (SLURM_MPI_TYPE=pmix), unlike
Frontier's Cray MPICH/PMI2 or Perlmutter's stack, and neither prior example's
convention has been tested against a PMIx-based MPI stack. This pipeline's
first pass deliberately runs pw.x as a single MPI rank per chore (see
run_pipeline.py's converge_dft_data stage) to sidestep this question until
the basic input/execute/parse plumbing is confirmed working end-to-end.
"""
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml
from ase.calculators.espresso import EspressoProfile, EspressoTemplate
from ase.io import read as ase_read

from EnsembleFFFit.utilities.general import parse_list


def _resolve_pseudopotentials(elements, pseudo_dir, explicit=None):
    """
    Build the {element: filename} dict EspressoTemplate.write_input expects.
    Uses the yaml's own explicit mapping for any element it names; every
    other element is resolved by globbing pseudo_dir for a
    case-insensitively-matched '<Element>[._-]*.upf' or bare '<Element>.upf'
    file. Errors loudly (rather than guessing) if that glob finds zero or
    more than one match for a given element -- an ambiguous pseudopotential
    directory should be fixed by the caller, not silently resolved to an
    arbitrary choice.
    """
    explicit = explicit or {}
    pseudopotentials = {}
    for element in elements:
        if element in explicit:
            pseudopotentials[element] = explicit[element]
            continue
        matches = sorted(set(
            glob.glob(os.path.join(pseudo_dir, f"{element}.[uU][pP][fF]")) +
            glob.glob(os.path.join(pseudo_dir, f"{element}[._-]*.[uU][pP][fF]"))
        ))
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one pseudopotential for element {element!r} under {pseudo_dir}, "
                f"found {len(matches)}: {matches}. Set an explicit 'pseudopotentials: {{{element}: "
                f"<filename>}}' entry in the yaml directive to disambiguate."
            )
        pseudopotentials[element] = os.path.basename(matches[0])
    return pseudopotentials


def run_qe_calculation(working_directory_list, qe_yaml_list):
    n = len(working_directory_list)
    assert len(qe_yaml_list) == n, "working_directory_list and qe_yaml_list must be the same length"

    for working_directory, qe_yaml in zip(working_directory_list, qe_yaml_list):
        with open(qe_yaml, 'r') as f:
            yaml_args = yaml.safe_load(f)

        structure_filename = yaml_args.get('structure_filename', 'POSCAR')
        qe_executable = yaml_args.get('qe_executable', 'pw.x')
        command = yaml_args.get('command', qe_executable)
        pseudopotentials_directory = yaml_args['pseudopotentials_directory']
        explicit_pseudos = yaml_args.get('pseudopotentials', {})
        input_data = yaml_args.get('input_data', {})
        kpts = yaml_args.get('kpts')
        kspacing = yaml_args.get('kspacing')
        koffset = yaml_args.get('koffset', (0, 0, 0))

        directory = Path(working_directory)
        structure_path = directory / structure_filename
        atoms = ase_read(structure_path)

        elements = sorted(set(atoms.get_chemical_symbols()))
        pseudopotentials = _resolve_pseudopotentials(elements, pseudopotentials_directory, explicit_pseudos)

        template = EspressoTemplate()
        profile = EspressoProfile(command=command, pseudo_dir=pseudopotentials_directory)

        parameters = dict(
            input_data=input_data,
            pseudopotentials=pseudopotentials,
            kpts=kpts,
            kspacing=kspacing,
            koffset=koffset,
        )
        parameters = {k: v for k, v in parameters.items() if v is not None}

        template.write_input(profile, directory, atoms, parameters, properties=['energy', 'forces', 'stress'])

        # pw.x's own LD_LIBRARY_PATH is set INLINE within the shell command
        # string below (POSIX `VAR=val cmd` prefix syntax), applying ONLY to
        # pw.x itself -- NOT via subprocess.run's own env= parameter.
        # CONFIRMED (2026-09-16) as a real, necessary distinction, not a
        # style choice: shell=True spawns /bin/sh first to interpret the
        # redirection below, and env= would apply to THAT /bin/sh too -- the
        # container's own (Debian) /bin/sh crashes trying to load the host's
        # (Rocky) newer glibc from /host_lib64 if QE_LD_LIBRARY_PATH ends up
        # in LD_LIBRARY_PATH at the whole-subprocess level ("version
        # `GLIBC_2.38' not found (required by /bin/sh)"). This is the exact
        # same class of bug the Perlmutter VASP_ReaxFF_LAMMPs pipeline's own
        # launch_multi_node.slurm documents hitting and fixing (there, for
        # every forked child of the container's own shell, not just one
        # binary); vasp_dft.py's own env=vasp_env (with LD_LIBRARY_PATH set
        # the same way this code used to) carries the identical latent risk
        # -- apparently just not triggered there, likely a difference in
        # host/container glibc version pairing on that system, not an
        # intentional fix already in place.
        qe_ld_library_path = os.environ.get('QE_LD_LIBRARY_PATH', '')

        # Strip any PMI/PMIx environment before launching pw.x. CONFIRMED
        # (2026-09-16) as a real, necessary fix, not precautionary: when this
        # chore runs through a real Flux job (inherit_env=True on its own
        # Resources), flux-shell passes the WHOLE ambient environment
        # through to the launched task -- including Slurm's own PMIx server
        # info for the OUTER srun/apptainer invocation that bootstrapped the
        # Flux broker itself (PMIX_SERVER_URI2/3, PMIX_NAMESPACE=
        # slurm.pmix.<jobid>.0, PMI_RANK, PMI_SIZE, PMI_FD, ...) -- a
        # completely different PMIx context than anything Flux's own
        # job-shell provides for tasks IT launches. pw.x's OpenMPI runtime,
        # seeing these, tries to rendezvous via that (wrong, outer) PMIx
        # server/namespace and hangs indefinitely waiting for a peer that
        # can never appear there -- confirmed via a real stuck chore (6+
        # minutes, zero bytes written to espresso.pwo) that only ever
        # printed its startup banner and converged once these vars were
        # removed. This is unrelated to the (separate, still-unresolved)
        # multi-rank PMIx interop gap -- OMPI_MCA_pml=ob1/OMPI_MCA_btl=
        # self,sm alone does not prevent OpenMPI from attempting PMI-based
        # rendezvous in the first place if it sees PMI/PMIx env vars at all;
        # stripping them here forces immediate, unambiguous singleton mode
        # regardless of what leaked through from the surrounding launch.
        qe_env = {k: v for k, v in os.environ.items() if not k.startswith(('PMI_', 'PMIX_'))}

        # shell=True for the stdout/stderr redirection pw.x needs (it writes
        # its main output to stdout, not a fixed-name file the way RMG/VASP
        # do) -- also matches rmg_dft.py/vasp_dft.py's own shell=True
        # precedent. close_fds=False matches rmg_dft.py's own reasoning
        # (preserve inherited file descriptors) -- harmless here since the
        # PMI_FD *environment variable* (not just the fd itself) is what
        # anything would actually look for, and that's stripped above.
        full_command = (
            f'LD_LIBRARY_PATH="{qe_ld_library_path}" {command} '
            f'-in {template.inputname} > {template.outputname} 2> {template.errorname}'
        )
        result = subprocess.run(
            full_command, shell=True, cwd=working_directory, close_fds=False,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            env=qe_env,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"pw.x failed (exit {result.returncode}) in {working_directory}:\n{result.stdout}"
            )

        results = template.read_results(directory)
        energy = results['energy']
        forces = results.get('forces')
        stress = results.get('stress')  # ASE Voigt-6 order, eV/Angstrom^3 -- absent unless tstress=True

        print(f"{working_directory}: pw.x converged, energy={energy}")

        property_dictionary = {'energy': energy}
        if forces is not None:
            property_dictionary['fx'] = [forces[i][0] for i in range(len(forces))]
            property_dictionary['fy'] = [forces[i][1] for i in range(len(forces))]
            property_dictionary['fz'] = [forces[i][2] for i in range(len(forces))]
        if stress is not None:
            property_dictionary['stress_voigt'] = list(stress)

        with open(directory / 'properties.json', 'w') as f:
            json.dump(property_dictionary, f, indent=4)

    return {"status": "complete"}


if __name__ == "__main__":
    run_qe_calculation(parse_list(sys.argv[1]), parse_list(sys.argv[2]))
