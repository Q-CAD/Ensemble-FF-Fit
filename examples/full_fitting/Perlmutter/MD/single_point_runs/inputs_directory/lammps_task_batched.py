from lammps import lammps
import sys
import ast
from pathlib import Path

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

ff_list       = parse_list(sys.argv[1])
input_list    = parse_list(sys.argv[2])
control_list  = parse_list(sys.argv[3])
struct_list   = parse_list(sys.argv[4])

n = len(ff_list)
assert all(len(lst) == n for lst in (control_list, input_list, struct_list)), "All lists must be same length"

lmp = lammps(cmdargs=["-log", "none", "-screen", "os.devnull"])
#lmp.command("log none")

for idx, (ff, ctrl, inp, struct) in enumerate(zip(ff_list, control_list, input_list, struct_list), start=0):
    # 1) Open a new log file
    image_name = Path(struct).parent.name
    lmp.command(f"log log_{image_name}.lammps")

    # 2) Pass in your filenames
    lmp.command(f"variable ff_filename string {ff}")
    lmp.command(f"variable control_filename string {ctrl}")
    lmp.command(f"variable structure string {struct}")

    # 3) Run the LAMMPS input
    lmp.file(inp)

    # 4) Clear for the next iteration
    lmp.command("clear")

# Final cleanup
lmp.close()
