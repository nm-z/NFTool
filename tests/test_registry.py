"""Registry / factory behaviour — the heart of the generalization."""

import pytest

from nf.models import REGISTRY, get_spec, list_architectures, register
from nf.models.base import ArchitectureSpec, HParam


def test_builtin_architectures_registered():
    assert {"MLP", "CNN", "ResNetMLP"} <= set(REGISTRY)


def test_get_spec_unknown_raises_with_helpful_message():
    with pytest.raises(KeyError) as exc:
        get_spec("Nope")
    assert "Registered:" in str(exc.value)


def test_list_architectures_is_json_serialisable():
    import json
    schema = list_architectures()
    json.dumps(schema)  # must not raise
    names = {a["name"] for a in schema}
    assert {"MLP", "CNN", "ResNetMLP"} <= names
    for arch in schema:
        assert arch["hyperparameters"]                 # every arch exposes knobs
        for hp in arch["hyperparameters"]:
            assert {"name", "kind", "label"} <= set(hp)


def test_cnn_advertises_channel_dim_and_min_input():
    cnn = next(a for a in list_architectures() if a["name"] == "CNN")
    assert cnn["expects_channel_dim"] is True
    assert cnn["min_input_dim"] == 16


def test_register_rejects_duplicate_and_unnamed():
    with pytest.raises(ValueError):
        register(get_spec("MLP"))           # already registered

    class Nameless(ArchitectureSpec):
        pass
    with pytest.raises(ValueError):
        register(Nameless())


def test_hparam_schema_roundtrip():
    hp = HParam("lr", "float", low=1e-6, high=1e-3, log=True, default=1e-4)
    s = hp.as_schema()
    assert s["kind"] == "float" and s["log"] is True and s["label"] == "Lr"


def test_hparam_suggest_uses_overrides():
    # A tiny fake trial records what bounds it was asked for.
    class FakeTrial:
        def __init__(self):
            self.calls = {}

        def suggest_int(self, name, lo, hi, step=1):
            self.calls[name] = (lo, hi)
            return lo

        def suggest_float(self, name, lo, hi, log=False):
            self.calls[name] = (lo, hi)
            return lo

        def suggest_categorical(self, name, choices):
            return choices[0]

    hp = HParam("layer_size", "int", low=1, high=1024)
    t = FakeTrial()
    hp.suggest(t, overrides={"layer_size": (8, 8)})
    assert t.calls["layer_size"] == (8, 8)
