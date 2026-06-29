"""NFTool — generalized model layer.

Public surface:
    nf.models.list_architectures()   -> the /architectures schema
    nf.engine.run_search(...)        -> architecture-agnostic Optuna search
    nf.engine.train(...)             -> the single training loop
    nf.data.load_csv_dataset(...)    -> faithful data loading/splitting
"""

from . import models, engine, data  # noqa: F401

__all__ = ["models", "engine", "data"]
