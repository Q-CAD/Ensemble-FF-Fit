import argparse
import copy
import os


def _import_mace_conversion_deps():
    """
    Import torch/e3nn/mace, raising a clear error if they aren't installed.

    Unlike the rest of `utilities/`, this module needs the `mace` extra's
    dependencies even though `create_lammps_models` is registered as a core
    console script -- guard the import here rather than at module level so the
    rest of this file (and `pip install .` without the `mace` extra) stays usable.
    """
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    try:
        import torch
        from e3nn.util import jit
        from mace.calculators import LAMMPS_MACE
        from mace.calculators.lammps_mliap_mace import LAMMPS_MLIAP_MACE
        from mace.cli.convert_e3nn_cueq import run as run_e3nn_to_cueq
    except ImportError as e:
        raise ImportError(
            "create_lammps_models requires torch/e3nn/mace, which the core "
            "EnsembleFFFit install does not pull in -- install the 'mace' extra "
            'first: pip install -e ".[mace]"'
        ) from e
    return torch, jit, LAMMPS_MACE, LAMMPS_MLIAP_MACE, run_e3nn_to_cueq


def parse_args():
    """
    Parse CLI args for converting a MACE model into a LAMMPS-consumable model.
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--models_path",
        type=str,
        help="Path to the directory with models to be converted to LAMMPS",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Name of model to be converted to LAMMPs",
    )
    parser.add_argument(
        "--head",
        type=str,
        nargs="?",
        help="Head of the model to be converted to LAMMPS",
        default=None,
    )
    parser.add_argument(
        "--dtype",
        type=str,
        nargs="?",
        help="Data type of the model to be converted to LAMMPS",
        default="float64",
    )
    parser.add_argument(
        "--format",
        type=str,
        help="Old libtorch format, or new mliap format",
        default="mliap",
    )
    return parser.parse_args()


def select_head(model):
    """
    Prompt the user to choose which model head to use when the model exposes
    multiple heads; auto-selects the single head automatically if only one exists.
    """
    if hasattr(model, "heads"):
        heads = model.heads
    else:
        heads = [None]

    if len(heads) == 1:
        print(f"Only one head found in the model: {heads[0]}. Skipping selection.")
        return heads[0]

    print("Available heads in the model:")
    for i, head in enumerate(heads):
        print(f"{i + 1}: {head}")

    # Ask the user to select a head
    selected = input(
        f"Select a head by number (1-{len(heads)}), or press Enter to proceed without specifying one: "
    )

    if selected.isdigit() and 1 <= int(selected) <= len(heads):
        return heads[int(selected) - 1]
    if selected == "":
        print("No head selected. Proceeding without specifying a head.")
        return None
    print(f"No valid selection made. Defaulting to the last head: {heads[-1]}")
    return heads[-1]


def main():
    """
    Walk `models_path` for files named `model_name`, load each with torch,
    optionally cast dtype, convert to cueq/mliap format if requested, select a
    head, wrap in the LAMMPS MACE calculator class, and save the resulting
    LAMMPS-consumable model next to the source.
    """
    torch, jit, LAMMPS_MACE, LAMMPS_MLIAP_MACE, run_e3nn_to_cueq = _import_mace_conversion_deps()

    args = parse_args()
    model_path = args.models_path  # takes model name as command-line input
    check_name = args.model_name
    base_name = check_name.split('.', 1)[0]
    for root, _, _ in os.walk(model_path):
        check_model_path = os.path.join(root, check_name)
        if os.path.exists(check_model_path):
            print(f"Creating {args.format} model for {check_model_path}")
            model = torch.load(
            check_model_path,
            map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            )
            if args.dtype == "float64":
                model = model.double().to("cpu")
            elif args.dtype == "float32":
                print("Converting model to float32, this may cause loss of precision.")
                model = model.float().to("cpu")

            if args.format == "mliap":
                # Enabling cuequivariance by default. TODO: switch?
                model = run_e3nn_to_cueq(copy.deepcopy(model))
                model.lammps_mliap = True

            if args.head is None:
                head = select_head(model)
            else:
                head = args.head
                print(
                    f"Selected head: {head} from command line in the list available heads: {model.heads}"
                )

            lammps_class = LAMMPS_MLIAP_MACE if args.format == "mliap" else LAMMPS_MACE
            lammps_model = (
                lammps_class(model, head=head) if head is not None else lammps_class(model)
            )
            if args.format == "mliap":
                torch.save(lammps_model, os.path.join(root, base_name + "-mliap.pt"))
            else:
                lammps_model_compiled = jit.compile(lammps_model)
                lammps_model_compiled.save(os.path.join(root, base_name + ".pt"))


if __name__ == "__main__":
    main()
