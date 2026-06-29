"""Cross-platform compatibility helpers (Linux + Windows + macOS).

The original monolith has a few habits that crash or misbehave off Linux:

* It prints emoji (🟢, ⛔, 🌟, …). On a legacy Windows console (code page 1252)
  ``print("🟢")`` raises ``UnicodeEncodeError`` and kills the run.
* It depends on ``torchviz`` (needs the Graphviz ``dot`` binary), which is
  usually absent on Windows.
* It builds report paths with a mix of separators.

This module centralises the fixes so every entry point (CLI, GUI, engine) gets
the same safe behaviour regardless of OS. Nothing here imports torch at module
load, so a stdlib-only frontend can use ``device_str``/paths without pulling in
the ML stack.
"""

from __future__ import annotations

import os
import sys
import platform
from pathlib import Path


IS_WINDOWS = os.name == "nt" or platform.system() == "Windows"


# ---------------------------------------------------------------------------
# Console / Unicode
# ---------------------------------------------------------------------------
def enable_utf8_console() -> None:
    """Best-effort: make stdout/stderr tolerate Unicode on every platform.

    On modern Python (3.7+) ``reconfigure`` exists; we ask for UTF-8 and, if the
    underlying console still can't encode a glyph, fall back to
    ``backslashreplace`` instead of raising. Safe to call more than once and on
    streams that don't support it.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
        except (AttributeError, ValueError, OSError):
            pass


# Emoji -> ASCII fallbacks, used when the active encoding can't represent them.
_ASCII_FALLBACKS = {
    "🟢": "[GPU]", "🔴": "[CPU]", "⏸️": "[pause]", "⛔": "[stop]", "⛔️": "[stop]",
    "🚀": "->", "🚧": "[wip]", "📊": "[stats]", "📈": "[trend]", "📁": "[file]",
    "📂": "[open]", "📄": "[doc]", "🌟": "*", "⚠️": "[warn]", "❌": "[x]",
    "✅": "[ok]", "🔍": "[search]", "🏆": "[best]", "🖥️": "[host]", "⏱️": "[time]",
    "🧪": "[test]", "🔁": "[loop]", "🧠": "[model]",
}


def ascii_safe(text: str) -> str:
    """Return ``text`` guaranteed printable on the current stdout encoding.

    If stdout is already UTF-8 the string is returned unchanged. Otherwise emoji
    are swapped for ASCII tags and anything still unencodable is dropped, so a
    Windows cp1252 console never raises mid-run.
    """
    enc = (getattr(sys.stdout, "encoding", None) or "utf-8")
    try:
        text.encode(enc)
        return text
    except (UnicodeEncodeError, LookupError):
        for glyph, repl in _ASCII_FALLBACKS.items():
            text = text.replace(glyph, repl)
        return text.encode(enc, errors="ignore").decode(enc, errors="ignore")


def cprint(*args, **kwargs) -> None:
    """``print`` that can never die on a non-UTF console."""
    sep = kwargs.pop("sep", " ")
    msg = sep.join(ascii_safe(str(a)) for a in args)
    try:
        print(msg, **kwargs)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "ignore").decode("ascii"), **kwargs)


# ---------------------------------------------------------------------------
# Filesystem
# ---------------------------------------------------------------------------
def results_dir(base: str | os.PathLike, timestamp: str) -> Path:
    """OS-correct results directory (uses the platform separator)."""
    return Path(base) / "Training Reports" / timestamp


def ensure_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Compute device (torch imported lazily so this stays stdlib-safe)
# ---------------------------------------------------------------------------
def device_str() -> str:
    """Return ``"cuda"`` / ``"cpu"`` without importing torch unless available."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"
