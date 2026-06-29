"""End-to-end demo of the generalized model layer — no Tkinter, no GUI.

Runs headless so it can be used as a smoke test:

    python demo.py                      # all architectures, real dataset
    python demo.py --arch ResNetMLP     # one architecture
    python demo.py --schema             # just print the /architectures schema
    python demo.py --trials 5 --epochs 60

It proves the extension path: MLP, CNN, and the brand-new ResNetMLP all train
through the *same* engine, selected purely by name from the registry.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import optuna
from sklearn.metrics import mean_absolute_error, r2_score

import torch

from nf.compat import enable_utf8_console
from nf.data import load_csv_dataset
from nf.engine import SearchConfig, run_search, train
from nf.models import get_spec, list_architectures

enable_utf8_console()                 # make stdout Unicode-safe on Windows consoles
optuna.logging.set_verbosity(optuna.logging.WARNING)

HERE = os.path.dirname(os.path.abspath(__file__))
PRED = os.path.join(HERE, "dataset", "Predictors_2025-04-15_10-43_Hold-2.csv")
TARG = os.path.join(HERE, "dataset", "9_10_24_Hold_02_targets.csv")
OPTIMIZERS = ["Adam"]


def evaluate(spec, model, X, y, device):
    model.eval()
    with torch.no_grad():
        preds = model(spec.prepare_inputs(X).to(device)).cpu().numpy().flatten()
    y = np.asarray(y).flatten()
    return r2_score(y, preds), mean_absolute_error(y, preds)


def run_one(arch: str, ds, trials: int, epochs: int, seed: int, device):
    spec = get_spec(arch)
    cfg = SearchConfig(
        architecture=arch,
        optimizer_choices=OPTIMIZERS,
        patience=25,
        num_epochs=epochs,
        seed=seed,
        # A short demo benefits from a more effective LR range than the original
        # GUI's 1e-6..1e-3 default. The registry/engine are untouched; this is
        # just the demo's search configuration.
        lr_range=(1e-4, 5e-3),
    )
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = run_search(ds.data, cfg, sampler=sampler, n_trials=trials, device=device)

    best = max((t for t in study.trials if "r2" in t.user_attrs),
               key=lambda t: t.user_attrs["r2"])
    # Retrain the winning config and score it on the held-out test set.
    hp = dict(best.params)
    model, _, _ = train(
        spec, ds.data, hp,
        optimizer_name=hp["optimizer"], lr=hp["lr"],
        patience=25, num_epochs=epochs, device=device,
    )
    test_r2, test_mae = evaluate(spec, model, ds.X_test, ds.y_test, device)
    return {
        "arch": arch,
        "n_params": sum(p.numel() for p in model.parameters()),
        "best_val_r2": best.user_attrs["r2"],
        "test_r2": test_r2,
        "test_mae": test_mae,
        "best_hp": {k: hp[k] for k in hp},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default=None, help="single architecture (default: all)")
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--schema", action="store_true", help="print /architectures schema and exit")
    args = ap.parse_args()

    if args.schema:
        print(json.dumps(list_architectures(), indent=2))
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("Registered architectures:",
          ", ".join(s["name"] for s in list_architectures()))

    ds = load_csv_dataset(PRED, TARG)
    print(f"Dataset: input_dim={ds.input_dim}, "
          f"train={len(ds.data.X_train)} val={len(ds.data.X_val)} test={len(ds.X_test)}\n")

    archs = [args.arch] if args.arch else [s["name"] for s in list_architectures()]
    rows = []
    for arch in archs:
        print(f"=== {arch}: searching {args.trials} trials x {args.epochs} epochs ===")
        rows.append(run_one(arch, ds, args.trials, args.epochs, args.seed, device))
        r = rows[-1]
        print(f"    -> params={r['n_params']:,}  val R2={r['best_val_r2']:.4f}  "
              f"test R2={r['test_r2']:.4f}  test MAE={r['test_mae']:.6f}\n")

    print("================ SUMMARY ================")
    print(f"{'arch':<10} {'params':>10} {'val R2':>9} {'test R2':>9} {'test MAE':>12}")
    for r in rows:
        print(f"{r['arch']:<10} {r['n_params']:>10,} {r['best_val_r2']:>9.4f} "
              f"{r['test_r2']:>9.4f} {r['test_mae']:>12.6f}")


if __name__ == "__main__":
    main()
