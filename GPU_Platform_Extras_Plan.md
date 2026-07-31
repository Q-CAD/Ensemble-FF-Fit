# Plan: ROCm/CUDA platform extras + detection script

This is a **plan**, matching the format used for `Reformat.md` earlier in this project — nothing has
been implemented yet. Answer the open questions inline (like `Reformat.md`'s `**Answer**` tags) and I'll
execute. Source brief: `pyproject_gpu_platform_extras.md`.

---

## Research findings (fetched live this session — my training data is stale for this ecosystem)

Before designing anything, I fetched current data from pytorch.org/PyPI rather than rely on memory,
per the brief's explicit instruction. Results, with dates, so you can judge how fresh they still are:

- **PyTorch's current install-selector page** (`pytorch.org/get-started/locally/`) lists supported
  platforms as: CPU, CUDA 11.8, CUDA 12.6, CUDA 12.8, **ROCm 6.3**. This page is JS-rendered (an
  interactive selector), so I could not reliably scrape the *exact* pinned `torch`/`torchvision`/
  `torchaudio` version triple it generates per platform — only the platform list itself.
- **Latest stable releases on PyPI** (as of this session): `torch` 2.13.0 (Jul 8, 2026), `torchvision`
  0.28.0 (Jul 8, 2026 — same day as torch, consistent with their usual coordinated release cadence),
  `torchaudio` **2.11.0 (Mar 23, 2026 — four months older, and off the torch==torchaudio version-lockstep
  convention that used to hold)**. That torchaudio gap is exactly the kind of thing I don't want to paper
  over — I'm flagging it rather than assuming 2.13.0 exists for torchaudio too. **This whole matrix needs
  a live re-check at execution time** (or a copy-paste from the actual interactive selector, which is more
  reliable than my scraping) before any version gets hardcoded.
- **`cuequivariance-torch` is confirmed CUDA-only** (PyPI page: only `cuequivariance-ops-torch-cu12` and
  `-cu13` variants exist; description says "CUDA accelerated equivariant operations"; current version
  0.10.0). **No ROCm build exists.**
- **`cupy` is confirmed CUDA-only** on PyPI (`cupy-cuda12x`, version 14.1.1; no ROCm-named variant
  listed).

**Implication for the design below:** the `cuequivariance*`/`cupy-cuda12x` stack currently sitting inside
this project's `mace` and `lammps` extras (from last session's work) is CUDA-exclusive with no ROCm
equivalent I could find. That's the single biggest structural consequence of this whole brief — see
Question 1 below, since I want to confirm this against your actual hands-on Frontier experience before
restructuring around it.

---

## Proposed design

### 1. Confirming there's no better TOML-native mechanism (per your explicit ask to flag one if I know of one)

I don't know of one, and this conclusion is consistent with a decision we already made in the
`pyproject_branching_deps.md` round: PEP 508 environment markers only cover interpreter/OS attributes pip
can introspect itself, not external hardware/driver state. The only per-package-index mechanisms I'm
aware of are `uv`'s `[tool.uv.sources]` and Poetry's `[tool.poetry.source]` — both toolchain-specific
extensions, not something expressible in a plain `pyproject.toml` consumed by stock `pip`. You'd already
decided against adopting `uv` broadly for the analogous mace/torch-index problem, citing Perlmutter
module/availability concerns — I'm treating that as still the operative decision here rather than
re-opening it, unless you want to revisit given ROCm adds a second platform to the same problem.

### 2. Detection/install script

**Language: Python.** Justification: the rest of this codebase is 100% Python; the logic needed here
(structured mapping table, `argparse` override flag, cross-checking multiple detection signals, clear
error formatting) is exactly what Python is good at and bash is awkward at. Contrast with `build_*.sh`,
which is bash specifically because it orchestrates `conda create`/`git clone`/`cmake` — none of which this
script needs to do.

**Location: repo root, standalone** (e.g. `install_gpu_torch.py`), run directly
(`python install_gpu_torch.py [--platform rocm6.3]`) — **not** registered as a `pyproject.toml` console
script. Reasoning: this has to run *before* any extras (or even the base package) are necessarily
installed, mirroring how `build_*.sh` scripts are invoked today (standalone, no prior install step). A
console-script entry would only become runnable after `pip install -e .` succeeds, which adds an
unnecessary ordering dependency for a "run this literally first" tool. Flagging as Question 6 below since
it's a real choice, not a forced one.

**Detection logic:**
- ROCm: check `/opt/rocm-*` (glob, since your container has it at a versioned path like
  `/opt/rocm-6.3.3/`) *and* `/opt/rocm/.info/version` if present *and* `rocminfo` on `PATH` if available —
  cross-check whichever signals exist rather than trusting the first one found, and fail (not guess) if
  they disagree.
- CUDA: `nvidia-smi` (parse the driver's reported CUDA version) and `nvcc --version` if available, same
  cross-check-and-fail-on-disagreement approach. Deliberately does **not** try `import torch` — as the
  brief notes, torch isn't installed yet at this point, that's what we're about to fix.
- Both detected, or neither detected → fail loudly with a clear message pointing at `--platform`, never
  silently pick CPU or one of the two.

**Mapping table:** a small explicit dict inside the script (not a separate data file, for simplicity —
flag if you'd rather it be external/JSON for easier updating without touching code), each entry citing
`pytorch.org/get-started/locally/` and the date it was last verified, e.g.:
```python
# Verified against https://pytorch.org/get-started/locally/ on <date> -- re-check before trusting this
# if it's been a while; PyTorch doesn't ship a wheel for every point release.
PLATFORM_MAP = {
    "rocm6.3":  {"torch": "?", "torchvision": "?", "torchaudio": "?", "index_url": "https://download.pytorch.org/whl/rocm6.3"},
    "cuda12.6": {"torch": "?", "torchvision": "?", "torchaudio": "?", "index_url": "https://download.pytorch.org/whl/cu126"},
    ...
}
```
The `?`s are deliberate — see Questions 2/3, I'm not filling these in from a scrape I can't fully verify.

**Manual override:** `--platform <key>`, validated against `PLATFORM_MAP`'s keys (not freely constructed
into a URL) — an unrecognized value fails loudly and lists the valid keys.

**Install invocation:** `subprocess.run([sys.executable, "-m", "pip", "install", ...])` — using
`sys.executable` specifically (not a bare `pip` on `PATH`) so it always targets the environment the script
itself is running in, avoiding a mismatch if multiple Pythons/environments are on `PATH`.

**Failure mode:** clear `print(..., file=sys.stderr)` + `sys.exit(1)`, not an uncaught exception traceback
— matches "fails loudly and clearly," not just "fails."

### 3. `pyproject.toml` restructuring

Given the `cuequivariance`/`cupy` finding above, I'm proposing to **move** the CUDA-specific packages
currently inside `mace` and `lammps` out into the new `cuda` extra, leaving `mace`/`lammps` as
backend-only extras that get combined with `cuda` or `rocm` at install time — this is what "orthogonal
dimensions" implies architecturally, but it's a real restructuring of what I built last session, so I want
to confirm before doing it (Questions 1 and 4).

- **`cuda` extra (new):** `cuequivariance>0.6.0`, `cuequivariance-torch>0.6.0`,
  `cuequivariance-ops-torch-cu12>0.6.0` (moved from `mace`+`lammps`), `cupy-cuda12x` (moved from `lammps`).
- **`rocm` extra (new):** empty, or near-empty, pending Question 1 — I found no ROCm equivalent for any
  of the above.
- **`mace` extra:** becomes just `mace-torch` (backend package only). Comment updated to point at
  `install_gpu_torch.py` instead of the generic "install torch from the index yourself" note.
- **`lammps` extra:** would become **empty** once its cuequivariance/cupy content moves to `cuda` — see
  Question 4 on whether to keep it as an empty placeholder (matching how we handled the LAMMPS-python-module
  case last session) or retire the name entirely.
- **`reaxff` extra:** proposing to leave as CUDA-only / untouched by this change (it already pins
  `jax[cuda12]`, jax's own platform-extra syntax) — given JAX-ReaxFF's deprecated status per `TODO.md`,
  I'm assuming ROCm support for it is out of scope unless you say otherwise (Question 5).
- **`torchsim` extra:** unaffected structurally (still just `torch-sim-atomistic`), but its comment about
  excluding torch should get the same pointer to `install_gpu_torch.py`.

Two-step install becomes:
```bash
python install_gpu_torch.py            # detects platform, installs the right torch/torchvision/torchaudio
pip install -e ".[mace,cuda]"          # or ".[mace,rocm]", ".[lammps,cuda]", etc.
```
This note goes in `CLAUDE.md`'s install section (alongside the extras table already there) rather than a
separate README section, to keep install instructions in one place — flag if you'd rather it lived in
README.md specifically.

### 4. Sanity check

I don't have ROCm or CUDA hardware in this environment (this sandbox is macOS/Darwin) — the "actually
running it on a GPU box" half of the brief's ask isn't something I can do from here; that part has to
happen on Frontier/Perlmutter by you. What I *can* do, and will as part of implementation:
- Simulate the no-GPU-stack-at-all path (mock away `/opt/rocm*`, `nvidia-smi`, `nvcc`) and confirm the
  script exits non-zero with a clear message rather than defaulting to CPU torch or crashing with a raw
  traceback.
- Unit-test the mapping-table lookup and `--platform` override validation against fake inputs (valid key,
  invalid key, conflicting detection signals).

---

## Open questions

1. **(Most consequential)** Does your Frontier build use *any* ROCm-specific acceleration package for
   MACE (something playing the role `cuequivariance`/`cupy` play on CUDA), or does ROCm MACE run without
   that acceleration layer entirely? This determines whether the `rocm` extra has real content or starts
   genuinely empty. **Answer** The current build doesn't have any acceleration installed. However, and in a parallel manner to CUDA cuequivariance, with ROCm we could install openequivariance; MACE supports openequivariance acceleration, which would likely be useful for fine-tuning and running MD with ASE. Please consider this idea and let me know if this is possible to add into the pyproject.toml. 
2. Your brief's example command pins `torch==2.10.0`/`torchvision==0.25.0`/`torchaudio==2.10.0` against
   `--index-url .../rocm7.1`. PyTorch's current stable install page lists **ROCm 6.3**, not 7.1, as of
   this session — is `rocm7.1` from a newer/nightly PyTorch build you used on Frontier, or a placeholder
   in the brief? I'd rather anchor the mapping table to whatever you've actually run successfully than to
   my scrape of a JS-rendered page. **Answer** The current OLCF docs pip install rocm7.1 as an example, but the most recent container build has rocm6.3.3 in /opt/rocm-6.3.3/bin. The version contained in /opt will likely change as the supported Frontier stack changes in the future, but for now anchoring to rocm6.3 or similar might be the best approach. 
3. For the `cuda` side: keep targeting `cu124` (matching the existing `mace`/`torchsim` comments from last
   session), or move to a newer index (`cu126`/`cu128`, per what's currently listed)? **Answer** Keep cu124 for now, as I haven't built the Perlmutter container that uses cuda, so I'm not sure what the most recent version of this supports. But we can add a note to the TODO.md describing this as a potential issue. 
4. Once `cuequivariance`/`cupy` move to `cuda`, the `lammps` extra would be empty. Keep it as an empty
   placeholder (documentation/signposting value), or retire the extra name entirely? **Answer** We should retire the name entirely. The Frontier, and I'm assuming Perlmutter, containers already have lammps installed with Python bindings. So there isn't any need to do the installation from within the pyproject.toml. 
5. Confirm: `reaxff` stays CUDA-only / ROCm out of scope for it, given its deprecated status? **Answer** Yes, let's keep it this way for now. I was able to get jax-reaxff running on ROCm awhile ago, but it is much slower, which might be moot regardless once it's deprecated. 
6. Script location/registration: standalone at repo root (my recommendation, matches `build_*.sh`
   convention), or a registered `pyproject.toml` console script (only runnable post-`pip install -e .`)? **Answer** I think you can go with your recommendation to start here. We can document the full pipeline for installation in the README.md.
7. Mapping table as an inline Python dict in the script (my default), or a separate JSON/YAML file for
   easier updating without touching code? **Answer** Let's stick with your default for now, and see how it performs. 

Nothing executed yet — once these are answered I'll implement the script, the `pyproject.toml` changes,
and the `CLAUDE.md`/comment updates, then run the sanity checks described above.
