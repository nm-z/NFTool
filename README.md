# NFTool — Generalized Deep-Learning Regression Tool

> **This is the `claude-rewrite` branch** — a generalized model-registry refactor
> of the original monolithic NFTool, verified end-to-end on **Linux and Windows 11**
> (29/29 tests pass on both; trains MLP, CNN, and a new ResNetMLP through one
> branch-free engine). The full Tauri desktop app lives on the `windows` branch.

## Install + run from GitHub (one command)

**Windows (PowerShell):**

```powershell
irm https://raw.githubusercontent.com/nm-z/NFTool/claude-rewrite/install.ps1 | iex
```

**Linux / macOS:**

```bash
curl -fsSL https://raw.githubusercontent.com/nm-z/NFTool/claude-rewrite/install.sh | bash
```

Each downloads this branch, installs a real Python + (on Windows) the VC++ runtime
PyTorch needs, creates a venv with CPU PyTorch + the ML stack, runs the test suite,
then trains the model — from nothing, on a clean machine. Verified on an isolated
Windows 11 VM (see `proof/`).

---

NFTool trains regression models (Optuna hyperparameter search + early stopping)
and emits diagnostic reports. This branch contains the original single-file
desktop app (`NFTool_V3_071125A.py`) **and** a generalized, cross-platform model
layer (`nf/`) that replaces the two hardcoded architectures with a registry so
any neural network can be plugged in without touching the training loop.

> **Why the refactor:** the original branched on a `"NN"`/`"CNN"` string in ~12
> places (see `REVIEW.md`). Adding a third architecture meant editing the build
> function, the Optuna objective, the data path, the eval path, and the report
> writer. Now it is one self-contained module + one registry line.

---

## One command — set up everything and run the model

These force a clean environment (venv + CPU PyTorch + full ML stack), run the
test suite, then train every registered architecture on the bundled dataset.

**Linux / macOS:**

```bash
bash setup_and_run.sh
```

**Windows (PowerShell):** installs a real Python *and* the VC++ runtime PyTorch
needs, automatically:

```powershell
powershell -ExecutionPolicy Bypass -File setup_and_run.ps1
```

Pass any `demo.py` args through, e.g. a quick single-architecture run:

```bash
bash setup_and_run.sh --arch MLP --trials 10 --epochs 80
```

(The CNN is the slow architecture — it convolves over the full 3204-feature
sequence on CPU; use `--arch MLP` or `--arch ResNetMLP` for a fast run.)

---

## What's here

| Path | Purpose |
|------|---------|
| `nf/models/` | The architecture **registry** — `MLP`, `CNN`, and a new `ResNetMLP` |
| `nf/engine.py` | One architecture-agnostic training loop + Optuna search |
| `nf/data.py` | CSV load / scale / split |
| `nf/compat.py` | Cross-platform helpers (Unicode console, paths, device) |
| `demo.py` | Headless end-to-end runner / smoke test |
| `nf_gui.py` | Tkinter GUI whose controls are generated **from the registry** |
| `tests/` | Pytest suite incl. simulated-Windows (cp1252) tests |
| `REVIEW.md` | Full code-review findings (`file:line` + severity) |
| `ARCHITECTURES.md` | How to add a new architecture in 3 steps |

---

## Quick start

### Linux / macOS

```bash
python -m venv .venv && . .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install numpy pandas scikit-learn optuna pytest

python demo.py --schema                 # list architectures + hyperparameters
python demo.py --trials 5 --epochs 50   # train MLP, CNN, ResNetMLP on the sample data
python -m pytest tests -q               # run the test suite
python nf_gui.py                        # launch the registry-driven GUI
```

### Windows (PowerShell)

```powershell
py -m venv .venv ; .\.venv\Scripts\Activate.ps1
py -m pip install --index-url https://download.pytorch.org/whl/cpu torch
py -m pip install numpy pandas scikit-learn optuna pytest

py demo.py --schema
py demo.py --trials 5 --epochs 50
py -m pytest tests -q
py nf_gui.py
```

The exact same code runs on both. `win_run.ps1` (used by the Windows VM in CI)
automates the Windows path end to end.

---

## Cross-platform notes

The generalized layer is OS-agnostic by construction:

- **Unicode console** — `nf/compat.enable_utf8_console()` + `cprint()` make
  output safe on a legacy Windows cp1252 console, where the original script's
  emoji `print()`s raise `UnicodeEncodeError`. Covered by
  `tests/test_compat.py` (which *simulates* a cp1252 console on any OS).
- **Paths** — `pathlib` / `os.path` throughout; no hardcoded separators.
- **Device** — CUDA is used when available, CPU otherwise, on every platform.
- **No POSIX-only calls** — no signals, `fork`, or Unix-only libraries in `nf/`.
- **Optional Graphviz** — model-graph rendering (`torchviz`) is treated as
  optional, since the `dot` binary is usually absent on Windows.

CI matrix: the suite is run on Linux (native) and on Windows 11 (KVM VM) using
the same `tests/` and `demo.py`. The Windows run is fully automated by
`win_run.ps1`.

**Two real Windows-deployment gotchas the VM testing surfaced** (handled by
`win_run.ps1`, worth knowing if you deploy by hand):

1. **The Microsoft Store Python stub.** A fresh Windows 11 ships a
   `python.exe` alias in `…\WindowsApps\` that is *not* Python — it just opens
   the Store. Naive `where python` finds it and every call exits 9009. The
   bootstrap rejects any interpreter under `\WindowsApps\` and installs a real
   Python (winget, with a python.org fallback).
2. **PyTorch needs the Visual C++ Redistributable.** The CPU `torch` wheel
   fails to import on a bare Win11 with `OSError: [WinError 126] … c10.dll`
   because `vcruntime140_1.dll` is missing. The bootstrap installs
   `vc_redist.x64` before importing torch.

---

## Adding an architecture

See **`ARCHITECTURES.md`**. Short version:

1. Write `nf/models/my_arch.py` subclassing `ArchitectureSpec` and call `register(...)`.
2. Add one import line to `nf/models/__init__.py`.
3. `python demo.py --arch MyArch` — it trains through the same engine and appears
   in `--schema` / the GUI automatically.
