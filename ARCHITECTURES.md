# NFTool — Generalized Model Layer

The model layer was hardcoded to two nets (`RegressionNet` / `CNNRegressionNet`)
with `"NN"`/`"CNN"` string branches in ~12 places. It is now a **registry /
factory**: the training engine, Optuna search, and (future) UI talk to every
architecture through one uniform interface and never branch on model type.

```
nf/
├── models/
│   ├── base.py        # ArchitectureSpec interface + HParam + REGISTRY/register
│   ├── mlp.py         # "MLP"      (== the original RegressionNet, unchanged behaviour)
│   ├── cnn.py         # "CNN"      (== the original CNNRegressionNet, unchanged behaviour)
│   └── resnet_mlp.py  # "ResNetMLP" (NEW — proves the extension path)
├── engine.py          # ONE training loop + ONE Optuna objective, no model branches
└── data.py            # CSV load / scale / split
demo.py                # headless end-to-end runner & smoke test
```

## The interface

Every architecture is an `ArchitectureSpec` (`nf/models/base.py`) with:

| Member | Purpose |
|--------|---------|
| `name` | unique key in the registry (`"MLP"`, `"CNN"`, …) |
| `build(input_dim, output_dim, hp)` | return a `torch.nn.Module` |
| `hyperparameter_space()` | list of `HParam` — drives **both** Optuna search **and** the UI schema |
| `prepare_inputs(X)` *(optional)* | shape raw `(N, features)` data (set `expects_channel_dim=True` for conv nets) |
| `min_input_dim` *(optional)* | minimum feature count, enforced centrally by the engine |

`HParam(name, kind, low, high, log, choices, default, label)` is the single
source of truth for a knob. `hp.suggest(trial)` produces an Optuna value;
`hp.as_schema()` produces the JSON a UI renders.

## `GET /architectures` analog

`nf.models.list_architectures()` returns every architecture and its
hyperparameter schema, so a frontend can render controls dynamically instead of
hardcoding sliders:

```bash
python demo.py --schema
```

---

## How to add an architecture in 3 steps

**1. Write a spec module** — `nf/models/my_arch.py`:

```python
import torch.nn as nn
from .base import ArchitectureSpec, HParam, register

class MyNet(nn.Module):
    def __init__(self, input_dim, width, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, width), nn.GELU(),
                                 nn.Linear(width, output_dim))
    def forward(self, x):
        return self.net(x)

class MyArchSpec(ArchitectureSpec):
    name = "MyArch"
    def build(self, input_dim, output_dim, hp):
        return MyNet(input_dim, int(hp["width"]), output_dim)
    def hyperparameter_space(self):
        return [HParam("width", "int", low=16, high=512, default=128)]

register(MyArchSpec())
```

**2. Register it** — add one import line to `nf/models/__init__.py`:

```python
from . import my_arch   # noqa: F401  registers "MyArch"
```

**3. Use it** — by name, nowhere else changes:

```bash
python demo.py --arch MyArch          # trains it through the same engine
python demo.py --schema               # it now appears in /architectures
```

No edits to `engine.py`, the Optuna objective, the data pipeline, or any UI. A
conv-style net just sets `expects_channel_dim = True` (and optionally
`min_input_dim`) and the engine handles the `(N, 1, features)` reshape for it.

`nf/models/resnet_mlp.py` is exactly such an addition — it was **not** in the
original NFTool and was added purely via these three steps.
