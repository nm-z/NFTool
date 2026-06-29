"""Architecture registry core: the uniform interface every model plugs into.

The whole point of this module is that the training engine, the Optuna search,
and any UI never need to know whether they are dealing with an MLP, a CNN, or
something that doesn't exist yet. They talk to an :class:`ArchitectureSpec`
through three things only:

* ``build(input_dim, output_dim, hp)``  -> a ``torch.nn.Module``
* ``hyperparameter_space()``            -> the tunable knobs (for Optuna *and* UI)
* ``prepare_inputs(X)``                 -> shapes raw 2-D data the way the net wants

Add an architecture = write one module that subclasses ``ArchitectureSpec`` and
calls ``register(MySpec())``. Nothing else in the codebase changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Hyperparameter descriptor
# ---------------------------------------------------------------------------
# A single source of truth for every tunable knob. It is consumed in two places:
#   1. The Optuna objective, via ``suggest(trial)`` -> a concrete value.
#   2. The "/architectures" schema (``as_schema``) so a UI can render a control
#      with the right type, range, and default WITHOUT hardcoding model names.
@dataclass(frozen=True)
class HParam:
    name: str
    kind: str                       # "int" | "float" | "categorical"
    low: float | None = None
    high: float | None = None
    log: bool = False               # sample on a log scale (learning rate, etc.)
    step: int | float | None = None
    choices: tuple[Any, ...] | None = None
    default: Any = None
    label: str | None = None        # human-friendly label for a UI

    def suggest(self, trial, overrides: dict[str, tuple] | None = None) -> Any:
        """Ask an Optuna trial for a value for this knob.

        ``overrides`` lets the caller (e.g. the GUI ranges) substitute the
        (low, high) bounds at runtime while keeping the knob's identity/type.
        """
        lo, hi = self.low, self.high
        if overrides and self.name in overrides:
            lo, hi = overrides[self.name]
        if self.kind == "int":
            return trial.suggest_int(self.name, int(lo), int(hi), step=int(self.step or 1))
        if self.kind == "float":
            return trial.suggest_float(self.name, float(lo), float(hi), log=self.log)
        if self.kind == "categorical":
            return trial.suggest_categorical(self.name, list(self.choices))
        raise ValueError(f"Unknown HParam kind: {self.kind!r}")

    def as_schema(self) -> dict[str, Any]:
        """JSON-serialisable description for the /architectures endpoint / UI."""
        d = {k: v for k, v in asdict(self).items() if v is not None}
        d["label"] = self.label or self.name.replace("_", " ").title()
        return d


# ---------------------------------------------------------------------------
# The architecture interface
# ---------------------------------------------------------------------------
class ArchitectureSpec:
    """Base class for every pluggable architecture.

    Subclasses set :attr:`name` and implement :meth:`build` and
    :meth:`hyperparameter_space`. Everything else has a sensible default so a
    minimal new architecture is genuinely a few lines (see ``resnet_mlp.py``).
    """

    name: str = "base"
    #: Minimum number of input features the architecture can accept. The engine
    #: enforces this centrally so the check is not buried inside ``__init__``.
    min_input_dim: int = 1
    #: Set by CNN-like specs that need a (N, channels, length) input tensor.
    expects_channel_dim: bool = False

    def build(self, input_dim: int, output_dim: int, hp: dict[str, Any]) -> nn.Module:
        raise NotImplementedError

    def hyperparameter_space(self) -> list[HParam]:
        raise NotImplementedError

    # -- input shaping ------------------------------------------------------
    def prepare_inputs(self, X: np.ndarray) -> torch.Tensor:
        """Turn a raw (N, features) array into the tensor this net expects.

        Default: pass through as a 2-D float tensor (MLP-style). CNN-style specs
        set ``expects_channel_dim = True`` and get a (N, 1, features) tensor.
        This replaces the old scattered ``preprocess_for_cnn`` / reshape logic.
        """
        t = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        if self.expects_channel_dim:
            if t.ndim == 2:
                t = t.unsqueeze(1)              # (N, F) -> (N, 1, F)
            elif not (t.ndim == 3 and t.shape[1] == 1):
                raise ValueError(f"{self.name}: unexpected input shape {tuple(t.shape)}")
        return t

    def validate_input_dim(self, input_dim: int) -> None:
        if input_dim < self.min_input_dim:
            raise ValueError(
                f"{self.name}: input has {input_dim} features but needs "
                f">= {self.min_input_dim}."
            )

    # -- schema for the /architectures endpoint / UI ------------------------
    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "expects_channel_dim": self.expects_channel_dim,
            "min_input_dim": self.min_input_dim,
            "hyperparameters": [hp.as_schema() for hp in self.hyperparameter_space()],
        }


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------
REGISTRY: dict[str, ArchitectureSpec] = {}


def register(spec: ArchitectureSpec) -> ArchitectureSpec:
    """Register an architecture under its ``name``. Returns the spec so it can
    be used as a decorator-ish one-liner at import time."""
    if not getattr(spec, "name", None) or spec.name == "base":
        raise ValueError("ArchitectureSpec must define a unique, non-empty `name`.")
    if spec.name in REGISTRY:
        raise ValueError(f"Architecture {spec.name!r} is already registered.")
    REGISTRY[spec.name] = spec
    return spec


def get_spec(name: str) -> ArchitectureSpec:
    try:
        return REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(REGISTRY)) or "<none>"
        raise KeyError(f"Unknown architecture {name!r}. Registered: {known}")


def list_architectures() -> list[dict[str, Any]]:
    """The backend answer to ``GET /architectures`` — every architecture and
    its hyperparameter schema, so a UI can render controls dynamically."""
    return [spec.schema() for spec in REGISTRY.values()]
