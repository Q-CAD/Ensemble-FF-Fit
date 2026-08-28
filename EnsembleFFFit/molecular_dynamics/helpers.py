import ast
from pymatgen.io.lammps.data import LammpsData
#from torch_sim.quantities import calc_kinetic_energy, calc_temperature


def parse_list(arg):
    """
    Try to parse `arg` as a Python literal list via ast.literal_eval.
    If that fails, fall back to simple comma-splitting.
    """
    try:
        val = ast.literal_eval(arg)
        if isinstance(val, list):
            return val
        # if it parsed to something else, keep going to split
    except (ValueError, SyntaxError):
        pass
    # fallback
    return arg.split(',')


def get_elements(structure_path, styles=['full', 'charge', 'atomic']):
    """
    Determine which elements are present in each structure.
    Used to set the lammps pair_coefficient flags.
    """
    for style in styles:
        try:
            ld = LammpsData.from_file(structure_path, atom_style=style)
        except ValueError:
            continue
        elements = ''
        for i, element in enumerate(ld.structure.elements):
            elements += str(element)
            if i != len(ld.structure.elements) - 1:
                elements += ' '
        return elements
    return None


def import_lammps():
    """
    Import the official LAMMPS Python module, raising a clear, actionable error
    if it isn't available.

    Unlike this project's other dependencies, `lammps` is not published on PyPI
    in the normal sense -- it's built from the LAMMPS source tree after LAMMPS
    itself is compiled, via either `make install-python` (installs a wheel built
    from that specific compiled LAMMPS into this environment) or by pointing
    PYTHONPATH at LAMMPS's `python/` directory and LD_LIBRARY_PATH at the
    directory containing the compiled LAMMPS shared library. Neither can be
    expressed as a normal pip dependency, so `pyproject.toml`'s `lammps` extra
    intentionally does not list `lammps` itself -- see TODO.md.
    """
    try:
        import lammps
    except ImportError as e:
        raise ImportError(
            "Could not import the official LAMMPS Python module ('import lammps' "
            "failed). This is expected if LAMMPS hasn't been built and linked into "
            "this environment yet -- pip cannot install it as a normal dependency. "
            "Either (a) run `make install-python` from your compiled LAMMPS build "
            "directory, or (b) set PYTHONPATH to LAMMPS's `python/` directory and "
            "LD_LIBRARY_PATH to the directory containing the compiled LAMMPS shared "
            "library. See TODO.md for details."
        ) from e
    return lammps


def import_lammps_mliap(lammps_module):
    """
    Import `lammps.mliap` (the Kokkos/MLIAP Python-coupling submodule used to run
    MACE potentials inside LAMMPS), raising a clear error if it's missing.

    If the base `lammps` module imports fine but this fails, the LAMMPS build
    most likely wasn't compiled with the `MLIAP_ENABLE_PYTHON=yes` and
    `PKG_ML-IAP=yes` cmake flags (the retired build_lammps.sh's cmake invocation
    had these set -- recover it from git history / the main branch for reference).
    """
    try:
        import lammps.mliap
    except ImportError as e:
        raise ImportError(
            "Could not import lammps.mliap. The base 'lammps' module imported "
            "successfully, but the mliap submodule is missing -- this LAMMPS build "
            "likely wasn't compiled with MLIAP_ENABLE_PYTHON=yes and PKG_ML-IAP=yes."
        ) from e
    return lammps_module.mliap


def make_prop_calculators(mapping):
    """
    Given a dict of { name: freq }, return the prop_calculators dict
    where each name is wired up to the correct lambda for MaceModel.
    Supported names: 'potential_energy', 'kinetic_energy', 'temperature', 'forces'
    """
    pc = {}
    for name, freq in mapping.items():
        if name == "potential_energy":
            func = lambda s, m: m(s)["energy"]
        elif name == "forces":
            func = lambda s, m: m(s)["forces"].cpu()
        elif name == "kinetic_energy":
            func = lambda s, m: calc_kinetic_energy(
                momenta=s.momenta,
                masses=s.masses,
                velocities=None
            ).unsqueeze(0)
        elif name == "temperature":
            func = lambda s, m: calc_temperature(
                momenta=s.momenta,
                masses=s.masses,
                velocities=None,
            ).unsqueeze(0)
        else:
            raise ValueError(f"Unknown prop name: {name!r}")
        pc[freq] = pc.get(freq, {})
        pc[freq][name] = func
    return pc
