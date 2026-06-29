"""Architecture-agnostic training engine.

There is exactly one training loop and one Optuna objective in here, and neither
contains a single ``if model == "CNN"`` branch. Every architecture-specific
decision is delegated to its :class:`~nf.models.base.ArchitectureSpec`:

* what knobs to tune          -> ``spec.hyperparameter_space()``
* how to build the net        -> ``spec.build(...)``
* how to shape the inputs     -> ``spec.prepare_inputs(...)``
* what input sizes are legal  -> ``spec.validate_input_dim(...)``

Common *training* knobs that the original code tuned identically for every model
(optimizer, learning rate, the unused ``alpha``) live here, in :func:`common_space`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import mean_absolute_error, r2_score

from .models import HParam, get_spec


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------
@dataclass
class SplitData:
    """Raw (un-shaped) numpy splits. The spec decides the final tensor shape."""
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray

    @property
    def input_dim(self) -> int:
        return self.X_train.shape[1]


# ---------------------------------------------------------------------------
# The batch-size invariant, enforced in ONE place
# ---------------------------------------------------------------------------
def resolve_batch_size(n_samples: int, requested: int | None) -> int | None:
    """Return a safe batch size, or ``None`` for full-batch training.

    The original GUI/UI tried to keep ``batch_size >= 2`` so that a future
    ``BatchNorm1d`` layer would never see a single-row batch. We enforce that
    invariant here, engine-side, so it holds for *every* architecture and is not
    duplicated in UI code or re-derived per model.
    """
    if requested is None:
        return None                      # full-batch (the original behaviour)
    bs = max(2, int(requested))          # never below 2
    bs = min(bs, n_samples)
    return bs


# ---------------------------------------------------------------------------
# Optimizer construction (LBFGS needs the closure dance)
# ---------------------------------------------------------------------------
def build_optimizer(name: str, params, lr: float) -> torch.optim.Optimizer:
    cls = torch.optim.LBFGS if name == "LBFGS" else getattr(torch.optim, name)
    return cls(params, lr=lr)


# ---------------------------------------------------------------------------
# The single training loop
# ---------------------------------------------------------------------------
def train(
    spec,
    data: SplitData,
    hp: dict[str, Any],
    *,
    optimizer_name: str,
    lr: float,
    patience: int,
    num_epochs: int = 200,
    output_dim: int = 1,
    device: torch.device | None = None,
    batch_size: int | None = None,
    on_epoch=None,
):
    """Train any architecture. Returns ``(model, best_val_loss, history)``.

    Full-batch by default, matching the original NFTool training dynamics
    exactly. Pass ``batch_size`` to opt into safe minibatching.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec.validate_input_dim(data.input_dim)

    X_train = spec.prepare_inputs(data.X_train).to(device)
    X_val = spec.prepare_inputs(data.X_val).to(device)
    y_train = torch.as_tensor(data.y_train, dtype=torch.float32).reshape(-1, output_dim).to(device)
    y_val = torch.as_tensor(data.y_val, dtype=torch.float32).reshape(-1, output_dim).to(device)

    model = spec.build(data.input_dim, output_dim, hp).to(device)
    optimizer = build_optimizer(optimizer_name, model.parameters(), lr)
    criterion = nn.MSELoss()

    bs = resolve_batch_size(len(X_train), batch_size)
    loader = None
    if bs is not None:
        ds = torch.utils.data.TensorDataset(X_train, y_train)
        # drop_last keeps the BatchNorm-safe invariant: no stray size-1 final batch.
        loader = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True)

    best_val_loss = float("inf")
    best_state = model.state_dict()
    counter = 0
    history = {"train": [], "val": [], "r2": [], "mae": []}

    for epoch in range(num_epochs):
        model.train()
        if optimizer_name == "LBFGS":
            def closure():
                optimizer.zero_grad()
                loss = criterion(model(X_train), y_train)
                loss.backward()
                return loss
            train_loss = optimizer.step(closure).item()
        elif loader is not None:
            batch_losses = []
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())
            train_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")
        else:
            optimizer.zero_grad()
            loss = criterion(model(X_train), y_train)
            loss.backward()
            optimizer.step()
            train_loss = loss.item()

        model.eval()
        with torch.no_grad():
            val_output = model(X_val)
            val_loss = criterion(val_output, y_val).item()
            val_preds = val_output.cpu().numpy().flatten()
            val_targets = y_val.cpu().numpy().flatten()

        if (
            np.isnan(val_preds).any() or np.isnan(val_targets).any()
            or np.isinf(val_preds).any() or np.isinf(val_targets).any()
            or np.abs(val_preds).max() > 1e6 or np.abs(val_targets).max() > 1e6
        ):
            # Same guard as the original: a blown-up trial is unusable.
            return model, float("inf"), history

        r2_val = r2_score(val_targets, val_preds)
        mae_val = mean_absolute_error(val_targets, val_preds)
        history["train"].append(train_loss)
        history["val"].append(val_loss)
        history["r2"].append(r2_val)
        history["mae"].append(mae_val)
        if on_epoch:
            on_epoch(epoch, {"train": train_loss, "val": val_loss, "r2": r2_val, "mae": mae_val})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break

    model.load_state_dict(best_state)
    return model, best_val_loss, history


# ---------------------------------------------------------------------------
# Search space = common training knobs + architecture knobs
# ---------------------------------------------------------------------------
def common_space(
    optimizer_choices: Sequence[str],
    lr_range=(1e-6, 1e-3),
    alpha_range=(0.0, 0.0),
) -> list[HParam]:
    """Knobs the original tuned identically for every architecture."""
    return [
        HParam("optimizer", "categorical", choices=tuple(optimizer_choices),
               default=optimizer_choices[0], label="Optimizer"),
        HParam("lr", "float", low=lr_range[0], high=lr_range[1], log=True,
               default=1e-4, label="Learning Rate"),
        HParam("alpha", "float", low=alpha_range[0], high=alpha_range[1],
               default=0.0, label="Alpha"),
    ]


def full_space(architecture: str, optimizer_choices, lr_range=(1e-6, 1e-3),
               alpha_range=(0.0, 0.0)) -> list[HParam]:
    """The complete tunable space for an architecture: common + arch-specific.

    This is what the engine searches and what a UI would render. No branching on
    the architecture name — we just concatenate the spec's own knobs.
    """
    spec = get_spec(architecture)
    return common_space(optimizer_choices, lr_range, alpha_range) + spec.hyperparameter_space()


# ---------------------------------------------------------------------------
# Optuna objective + search driver — architecture-agnostic
# ---------------------------------------------------------------------------
@dataclass
class SearchConfig:
    architecture: str
    optimizer_choices: Sequence[str]
    patience: int = 100
    num_epochs: int = 200
    output_dim: int = 1
    batch_size: int | None = None
    lr_range: tuple[float, float] = (1e-6, 1e-3)
    alpha_range: tuple[float, float] = (0.0, 0.0)
    seed: int | None = None
    bound_overrides: dict[str, tuple] | None = None   # name -> (low, high)


def _seed_everything(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_objective(data: SplitData, cfg: SearchConfig, device=None):
    spec = get_spec(cfg.architecture)
    space = full_space(cfg.architecture, cfg.optimizer_choices, cfg.lr_range, cfg.alpha_range)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def objective(trial):
        t0 = time.time()
        if cfg.seed is not None:
            _seed_everything(cfg.seed)
            trial.set_user_attr("seed", cfg.seed)

        # Suggest every knob straight from the declarative space.
        hp = {p.name: p.suggest(trial, cfg.bound_overrides) for p in space}

        model, val_loss, history = train(
            spec, data, hp,
            optimizer_name=hp["optimizer"], lr=hp["lr"],
            patience=cfg.patience, num_epochs=cfg.num_epochs,
            output_dim=cfg.output_dim, device=device, batch_size=cfg.batch_size,
        )

        # Final validation metrics for ranking + reporting.
        X_val = spec.prepare_inputs(data.X_val).to(device)
        model.eval()
        with torch.no_grad():
            preds = model(X_val).cpu().numpy().flatten()
        targets = np.asarray(data.y_val).flatten()
        r2_val = r2_score(targets, preds)
        mae_val = mean_absolute_error(targets, preds)

        trial.set_user_attr("r2", r2_val)
        trial.set_user_attr("mae", mae_val)
        trial.set_user_attr("val_loss", val_loss)
        trial.set_user_attr("train_loss", history.get("train", [None])[-1] if history.get("train") else None)
        trial.set_user_attr("duration_sec", time.time() - t0)
        return val_loss

    return objective


def run_search(data: SplitData, cfg: SearchConfig, *, sampler=None,
               n_trials: int | None = None, time_limit: int | None = None,
               device=None):
    """Drive an Optuna study with the architecture-agnostic objective."""
    import optuna
    study = optuna.create_study(direction="minimize", sampler=sampler)
    objective = make_objective(data, cfg, device=device)
    if time_limit:
        study.optimize(objective, timeout=time_limit, show_progress_bar=False)
    else:
        study.optimize(objective, n_trials=n_trials)
    return study
