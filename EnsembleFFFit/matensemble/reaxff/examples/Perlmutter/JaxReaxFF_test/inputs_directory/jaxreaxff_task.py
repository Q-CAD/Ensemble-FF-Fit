#!/usr/bin/env python3
import sys
import subprocess
import os

def main():
    # Build the command: jaxreaxff <all the args you passed>
    cmd = ["jaxreaxff"] + sys.argv[1:]

    # (Optional) print for debugging:
    print("Running:", " ".join(cmd))

    # Actually run it
    #result = subprocess.run(cmd)
    #return result.returncode
    os.execvp(cmd[0], cmd)

if __name__ == "__main__":
    #sys.exit(main())
    main()
