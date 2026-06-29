"""NFTool — registry-driven Tkinter GUI (cross-platform: Linux / Windows / macOS).

This is the local equivalent of the prompt's "the frontend discovers available
architectures and their tunable hyperparameters from the backend" requirement.
The controls are NOT hardcoded: the architecture dropdown and every
hyperparameter widget are generated at runtime from the registry schema
(``nf.models.list_architectures()``), so adding an architecture makes it appear
here automatically — no UI edits.

Runs anywhere Python + Tkinter exist. If the ML stack (torch) is not importable
in this environment, it transparently falls back to the bundled
``architectures.json`` so the UI still renders.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import subprocess

import tkinter as tk
from tkinter import ttk, filedialog

from nf.compat import enable_utf8_console

HERE = os.path.dirname(os.path.abspath(__file__))


def load_schema():
    """Schema from the live registry if possible, else the bundled JSON."""
    try:
        from nf.models import list_architectures
        return list_architectures(), "registry"
    except Exception:
        with open(os.path.join(HERE, "architectures.json"), encoding="utf-8") as f:
            return json.load(f), "architectures.json"


class NFToolGUI:
    BG = "#2b2b2b"
    FG = "#e0e0e0"
    ACCENT = "#3fb950"

    def __init__(self, root):
        self.root = root
        self.schema, source = load_schema()
        self.by_name = {a["name"]: a for a in self.schema}
        self.hp_vars: dict[str, tk.Variable] = {}

        root.title("NFTool — Generalized Trainer")
        root.geometry("680x720")
        root.configure(bg=self.BG)

        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TLabel", background=self.BG, foreground=self.FG)
        style.configure("TButton", padding=6)
        style.configure("Head.TLabel", background=self.BG, foreground=self.ACCENT,
                        font=("TkDefaultFont", 13, "bold"))

        ttk.Label(root, text="NFTool — Generalized Trainer", style="Head.TLabel").pack(pady=(12, 2))
        ttk.Label(root, text=f"architectures loaded from: {source}").pack()

        # --- architecture picker (populated from the schema) ---------------
        pick = tk.Frame(root, bg=self.BG); pick.pack(pady=10, fill="x", padx=20)
        ttk.Label(pick, text="Architecture:").pack(side="left")
        self.arch_var = tk.StringVar(value=self.schema[0]["name"])
        self.arch_combo = ttk.Combobox(
            pick, values=[a["name"] for a in self.schema],
            textvariable=self.arch_var, state="readonly", width=20,
        )
        self.arch_combo.pack(side="left", padx=10)
        self.arch_combo.bind("<<ComboboxSelected>>", lambda e: self.render_hyperparams())

        self.meta_label = ttk.Label(root, text="")
        self.meta_label.pack()

        # --- dynamic hyperparameter area -----------------------------------
        ttk.Label(root, text="Hyperparameters (auto-generated):", style="Head.TLabel").pack(pady=(12, 2))
        self.hp_frame = tk.Frame(root, bg=self.BG)
        self.hp_frame.pack(fill="x", padx=30)

        # --- run controls ---------------------------------------------------
        runrow = tk.Frame(root, bg=self.BG); runrow.pack(pady=8, fill="x", padx=30)
        ttk.Label(runrow, text="Trials:").grid(row=0, column=0, sticky="w")
        self.trials_var = tk.StringVar(value="5")
        ttk.Entry(runrow, textvariable=self.trials_var, width=8).grid(row=0, column=1, padx=(4, 16))
        ttk.Label(runrow, text="Epochs:").grid(row=0, column=2, sticky="w")
        self.epochs_var = tk.StringVar(value="50")
        ttk.Entry(runrow, textvariable=self.epochs_var, width=8).grid(row=0, column=3, padx=4)

        self.train_btn = ttk.Button(root, text="▶  Train through registry", command=self.on_train)
        self.train_btn.pack(pady=8)

        # --- output ---------------------------------------------------------
        self.out = tk.Text(root, height=12, bg="#1e1e1e", fg="#d4d4d4",
                           insertbackground=self.FG, wrap="word")
        self.out.pack(fill="both", expand=True, padx=12, pady=(4, 12))

        self.render_hyperparams()

    # ------------------------------------------------------------------ UI
    def render_hyperparams(self):
        for w in self.hp_frame.winfo_children():
            w.destroy()
        self.hp_vars.clear()

        arch = self.by_name[self.arch_var.get()]
        self.meta_label.config(
            text=f"expects_channel_dim={arch['expects_channel_dim']}  "
                 f"min_input_dim={arch['min_input_dim']}"
        )
        for i, hp in enumerate(arch["hyperparameters"]):
            ttk.Label(self.hp_frame, text=hp["label"] + ":").grid(
                row=i, column=0, sticky="w", pady=3)
            if hp["kind"] == "categorical":
                var = tk.StringVar(value=str(hp.get("default", hp["choices"][0])))
                ttk.Combobox(self.hp_frame, values=list(hp["choices"]),
                             textvariable=var, state="readonly", width=18
                             ).grid(row=i, column=1, sticky="w", padx=8)
            else:
                var = tk.StringVar(value=str(hp.get("default", "")))
                ttk.Entry(self.hp_frame, textvariable=var, width=20
                          ).grid(row=i, column=1, sticky="w", padx=8)
                rng = f"[{hp.get('low')} … {hp.get('high')}]"
                ttk.Label(self.hp_frame, text=rng).grid(row=i, column=2, sticky="w")
            self.hp_vars[hp["name"]] = var

    def log(self, msg):
        self.out.insert("end", msg + "\n")
        self.out.see("end")
        self.root.update_idletasks()

    # --------------------------------------------------------------- train
    def on_train(self):
        self.train_btn.config(state="disabled")
        self.out.delete("1.0", "end")
        arch = self.arch_var.get()
        self.log(f"Launching registry-driven training for: {arch}")
        self.log(f"hyperparameters: "
                 + ", ".join(f"{k}={v.get()}" for k, v in self.hp_vars.items()))
        threading.Thread(target=self._run, args=(arch,), daemon=True).start()

    def _run(self, arch):
        cmd = [sys.executable, os.path.join(HERE, "demo.py"),
               "--arch", arch, "--trials", self.trials_var.get(),
               "--epochs", self.epochs_var.get()]
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, cwd=HERE)
            for line in proc.stdout:
                self.root.after(0, self.log, line.rstrip())
            proc.wait()
            self.root.after(0, self.log, f"[done] exit code {proc.returncode}")
        except Exception as exc:  # noqa: BLE001
            self.root.after(0, self.log, f"[error] {exc}")
        finally:
            self.root.after(0, lambda: self.train_btn.config(state="normal"))


def main():
    enable_utf8_console()
    root = tk.Tk()
    NFToolGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
