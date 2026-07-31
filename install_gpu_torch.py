#!/usr/bin/env python3
"""
Detect the GPU platform (ROCm or CUDA) on this machine and install the matching
torch/torchvision/torchaudio build.

`pyproject.toml` cannot express "use a different package index for this one
dependency" or "pick a version based on what hardware/drivers are present" --
PEP 508 environment markers only cover interpreter/OS attributes pip can
introspect itself, not external GPU stack state. So this lives here instead, as
step 1 of a two-step install (step 2 is `pip install -e ".[mace,cuda]"` or
".[mace,rocm]"", etc. -- see README.md).

Run standalone, before installing this package or any of its extras:
    python install_gpu_torch.py                # auto-detect
    python install_gpu_torch.py --platform rocm6.3
    python install_gpu_torch.py --platform cuda12.4
    python install_gpu_torch.py --list-platforms
"""
import argparse
import glob
import re
import subprocess
import sys

# Verified against https://download.pytorch.org/whl/<index>/{torch,torchvision,torchaudio}/
# on 2026-07-24 -- these are the highest versions actually published for each index at that
# time (both indices predate PyTorch's current latest release and are no longer getting new
# builds, so "highest available" is stable, not a moving target -- but re-verify before trusting
# this if it's been a long time, since PyTorch doesn't publish a wheel for every point release
# and retires old indices over time). Update PLATFORM_MAP deliberately, don't leave it stale.
PLATFORM_MAP = {
    "rocm6.3": {
        "torch": "2.9.1",
        "torchvision": "0.24.1",
        "torchaudio": "2.9.1",
        "index_url": "https://download.pytorch.org/whl/rocm6.3",
    },
    "cuda12.4": {
        "torch": "2.6.0",
        "torchvision": "0.21.0",
        "torchaudio": "2.6.0",
        "index_url": "https://download.pytorch.org/whl/cu124",
    },
}


class DetectionError(Exception):
    """Raised when the GPU platform can't be confidently detected."""


def _run(cmd):
    """Run `cmd`, returning stdout on success or None if the command doesn't exist/fails."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return result.stdout if result.returncode == 0 else None


def detect_rocm():
    """
    Look for ROCm via three independent signals: a versioned /opt/rocm-* install
    path, /opt/rocm/.info/version, and `rocminfo` on PATH. Returns a "rocmX.Y"
    string if at least one signal fires and none disagree, else None.

    Raises DetectionError if signals disagree with each other, rather than
    silently trusting one over another.
    """
    found = {}

    for path in sorted(glob.glob("/opt/rocm-*")):
        m = re.search(r"/opt/rocm-(\d+\.\d+)", path)
        if m:
            found["path glob (/opt/rocm-*)"] = m.group(1)

    try:
        with open("/opt/rocm/.info/version") as f:
            m = re.search(r"(\d+\.\d+)", f.read())
            if m:
                found["/opt/rocm/.info/version"] = m.group(1)
    except OSError:
        pass

    rocminfo_out = _run(["rocminfo"])
    if rocminfo_out:
        m = re.search(r"ROCm Version:\s*(\d+\.\d+)", rocminfo_out)
        if m:
            found["rocminfo"] = m.group(1)

    if not found:
        return None

    versions = set(found.values())
    if len(versions) > 1:
        raise DetectionError(
            f"Conflicting ROCm versions detected from different signals: {found}. "
            "Refusing to guess -- use --platform to specify explicitly."
        )
    return f"rocm{versions.pop()}"


def detect_cuda():
    """
    Look for CUDA via two independent signals: `nvidia-smi` (driver-reported CUDA
    version) and `nvcc --version` (toolkit version). Returns a "cudaX.Y" string
    if at least one signal fires and none disagree, else None.

    Deliberately does not try `import torch` -- torch isn't installed yet at
    this point; detecting the platform is what determines which torch to
    install in the first place.

    Raises DetectionError if signals disagree with each other.
    """
    found = {}

    smi_out = _run(["nvidia-smi"])
    if smi_out:
        m = re.search(r"CUDA Version:\s*(\d+\.\d+)", smi_out)
        if m:
            found["nvidia-smi"] = m.group(1)

    nvcc_out = _run(["nvcc", "--version"])
    if nvcc_out:
        m = re.search(r"release (\d+\.\d+)", nvcc_out)
        if m:
            found["nvcc --version"] = m.group(1)

    if not found:
        return None

    versions = set(found.values())
    if len(versions) > 1:
        raise DetectionError(
            f"Conflicting CUDA versions detected from different signals: {found}. "
            "Refusing to guess -- use --platform to specify explicitly."
        )
    return f"cuda{versions.pop()}"


def resolve_platform_key(detected):
    """
    Map a raw "rocmX.Y"/"cudaX.Y" detection string to the closest supported key
    in PLATFORM_MAP (exact match required for the major.minor pair we detect --
    PyTorch doesn't publish a wheel for every point release, so silently
    rounding to "the nearest one we have" is exactly the wrong-guess failure
    mode this script is meant to avoid).
    """
    if detected in PLATFORM_MAP:
        return detected
    return None


def fail(message):
    """Print a clear error to stderr and exit non-zero -- never a bare traceback."""
    print(f"ERROR: {message}", file=sys.stderr)
    print(
        f"\nSupported --platform values: {', '.join(sorted(PLATFORM_MAP))}",
        file=sys.stderr,
    )
    sys.exit(1)


def install(platform_key):
    spec = PLATFORM_MAP[platform_key]
    packages = [
        f"torch=={spec['torch']}",
        f"torchvision=={spec['torchvision']}",
        f"torchaudio=={spec['torchaudio']}",
    ]
    cmd = [
        sys.executable, "-m", "pip", "install",
        *packages,
        "--index-url", spec["index_url"],
    ]
    print(f"Installing for platform '{platform_key}': {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        fail(f"pip install failed (exit {result.returncode}) -- see output above.")


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument(
        "--platform",
        choices=sorted(PLATFORM_MAP),
        default=None,
        help="Skip auto-detection and install this platform explicitly.",
    )
    parser.add_argument(
        "--list-platforms",
        action="store_true",
        help="Print supported --platform values and exit.",
    )
    args = parser.parse_args()

    if args.list_platforms:
        for key, spec in sorted(PLATFORM_MAP.items()):
            print(f"{key}: torch=={spec['torch']} (index: {spec['index_url']})")
        return

    if args.platform:
        install(args.platform)
        return

    try:
        rocm = detect_rocm()
        cuda = detect_cuda()
    except DetectionError as e:
        fail(str(e))
        return

    if rocm and cuda:
        fail(
            f"Both ROCm ({rocm}) and CUDA ({cuda}) were detected on this machine. "
            "Refusing to guess which one you want -- use --platform to specify explicitly."
        )
    elif rocm:
        key = resolve_platform_key(rocm)
        if key is None:
            fail(
                f"Detected ROCm {rocm[4:]}, but no PyTorch wheel index is mapped for it. "
                "Use --platform to specify a supported version explicitly, or add this "
                "version to PLATFORM_MAP once you've confirmed a matching wheel exists."
            )
        install(key)
    elif cuda:
        key = resolve_platform_key(cuda)
        if key is None:
            fail(
                f"Detected CUDA {cuda[4:]}, but no PyTorch wheel index is mapped for it. "
                "Use --platform to specify a supported version explicitly, or add this "
                "version to PLATFORM_MAP once you've confirmed a matching wheel exists."
            )
        install(key)
    else:
        fail(
            "Could not detect ROCm or CUDA on this machine (checked /opt/rocm-*, "
            "/opt/rocm/.info/version, rocminfo, nvidia-smi, nvcc --version). "
            "Use --platform to specify explicitly if you know this machine has a GPU stack "
            "that just wasn't detected -- see --list-platforms for supported values."
        )


if __name__ == "__main__":
    main()
