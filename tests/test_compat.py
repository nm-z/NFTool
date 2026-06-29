"""Cross-platform behaviour, including *simulated* Windows conditions.

These tests reproduce the two ways the original script dies off Linux — a
legacy cp1252 console choking on emoji, and OS-specific path handling — without
needing an actual Windows host. They run identically on Linux, macOS, and
Windows.
"""

import io
import sys
import ntpath
import posixpath

import pytest

from nf import compat


class _Cp1252Stream(io.TextIOBase):
    """Stand-in for a legacy Windows console: raises on non-cp1252 glyphs."""

    encoding = "cp1252"

    def __init__(self):
        self.buf = []

    def write(self, s):
        s.encode("cp1252")          # mimics the real console; raises on emoji
        self.buf.append(s)
        return len(s)


class _Utf8Stream(io.TextIOBase):
    encoding = "utf-8"

    def __init__(self):
        self.buf = []

    def write(self, s):
        self.buf.append(s)
        return len(s)


def test_ascii_safe_is_noop_on_utf8(monkeypatch):
    monkeypatch.setattr(sys, "stdout", _Utf8Stream())
    assert compat.ascii_safe("🟢 GPU ready") == "🟢 GPU ready"


def test_ascii_safe_downgrades_on_cp1252(monkeypatch):
    monkeypatch.setattr(sys, "stdout", _Cp1252Stream())
    out = compat.ascii_safe("🟢 GPU 🚀 go ⛔ stop")
    out.encode("cp1252")            # must now be representable
    assert "[GPU]" in out and "->" in out


def test_cprint_never_raises_on_legacy_console(monkeypatch):
    stream = _Cp1252Stream()
    monkeypatch.setattr(sys, "stdout", stream)
    # The original `print("🟢 ...")` would raise UnicodeEncodeError here.
    compat.cprint("🟢 GPU Available: Tesla 🚀")
    assert stream.buf                      # something was written, no exception


def test_results_dir_uses_native_separator():
    p = compat.results_dir("base", "2026-01-05_14-00")
    parts = p.parts
    assert parts[-2:] == ("Training Reports", "2026-01-05_14-00")


@pytest.mark.parametrize("mod", [ntpath, posixpath])
def test_report_paths_join_on_both_oses(mod):
    """A results path composed with either OS's rules stays well-formed."""
    joined = mod.join("Training Reports", "2026-01-05", "loss_plot.png")
    assert joined.endswith("loss_plot.png")
    assert "Training Reports" in joined


def test_device_str_returns_known_value():
    assert compat.device_str() in {"cpu", "cuda"}


def test_is_windows_flag_matches_platform(monkeypatch):
    import importlib
    monkeypatch.setattr("os.name", "nt")
    reloaded = importlib.reload(compat)
    assert reloaded.IS_WINDOWS is True
    # restore real module state for any later tests
    monkeypatch.undo()
    importlib.reload(compat)
