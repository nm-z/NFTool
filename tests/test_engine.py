"""Engine behaves identically for every architecture, with no model branches."""

import numpy as np
import pytest
import torch

from nf.engine import (
    SplitData, SearchConfig, train, run_search, resolve_batch_size, full_space,
)
from nf.models import get_spec, list_architectures


def _synthetic(n=60, d=32, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype("float32")
    w = rng.standard_normal(d)
    y = (X @ w + 0.01 * rng.standard_normal(n)).astype("float32")
    cut = n // 2
    return SplitData(X[:cut], y[:cut], X[cut:], y[cut:])


ALL_ARCHS = [a["name"] for a in list_architectures()]


@pytest.mark.parametrize("arch", ALL_ARCHS)
def test_prepare_inputs_shape(arch):
    spec = get_spec(arch)
    X = np.zeros((5, 32), dtype="float32")
    t = spec.prepare_inputs(X)
    if spec.expects_channel_dim:
        assert t.shape == (5, 1, 32)        # conv nets get a channel dim
    else:
        assert t.shape == (5, 32)


@pytest.mark.parametrize("arch", ALL_ARCHS)
def test_build_and_forward(arch):
    spec = get_spec(arch)
    hp = {p.name: p.default for p in spec.hyperparameter_space()}
    model = spec.build(32, 1, hp)
    x = spec.prepare_inputs(np.zeros((4, 32), dtype="float32"))
    out = model(x)
    assert out.shape[0] == 4


@pytest.mark.parametrize("arch", ALL_ARCHS)
def test_train_runs_and_history_is_aligned(arch):
    spec = get_spec(arch)
    data = _synthetic()
    hp = {p.name: p.default for p in spec.hyperparameter_space()}
    model, best_val, hist = train(
        spec, data, hp, optimizer_name="Adam", lr=1e-3,
        patience=5, num_epochs=8,
    )
    assert np.isfinite(best_val)
    # The old code appended train-loss twice; here every series is epoch-aligned.
    n = len(hist["train"])
    assert n == len(hist["val"]) == len(hist["r2"]) == len(hist["mae"]) >= 1


def test_cnn_min_input_dim_enforced_centrally():
    spec = get_spec("CNN")
    tiny = SplitData(np.zeros((4, 8), "float32"), np.zeros(4, "float32"),
                     np.zeros((4, 8), "float32"), np.zeros(4, "float32"))
    hp = {p.name: p.default for p in spec.hyperparameter_space()}
    with pytest.raises(ValueError):
        train(spec, tiny, hp, optimizer_name="Adam", lr=1e-3, patience=2, num_epochs=2)


def test_batch_size_invariant():
    assert resolve_batch_size(100, None) is None      # full-batch preserved
    assert resolve_batch_size(100, 1) == 2            # never below 2 (BatchNorm-safe)
    assert resolve_batch_size(3, 64) == 3             # never exceed sample count


def test_safe_minibatch_path_drops_last():
    spec = get_spec("MLP")
    data = _synthetic(n=64, d=16)
    hp = {p.name: p.default for p in spec.hyperparameter_space()}
    model, best_val, hist = train(
        spec, data, hp, optimizer_name="Adam", lr=1e-3,
        patience=3, num_epochs=4, batch_size=8,
    )
    assert np.isfinite(best_val)


def test_full_space_is_common_plus_arch():
    space = full_space("MLP", optimizer_choices=["Adam", "SGD"])
    names = [p.name for p in space]
    assert names[:3] == ["optimizer", "lr", "alpha"]   # common training knobs first
    assert "num_layers" in names and "layer_size" in names


def test_run_search_smoke_for_new_architecture():
    """The brand-new ResNetMLP trains through the same search driver."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    data = _synthetic()
    cfg = SearchConfig(architecture="ResNetMLP", optimizer_choices=["Adam"],
                       patience=3, num_epochs=5, seed=1)
    study = run_search(data, cfg, n_trials=2)
    assert len(study.trials) == 2
    assert all("r2" in t.user_attrs for t in study.trials)
