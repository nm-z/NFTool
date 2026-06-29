"""NFTool model registry.

Importing this package registers every built-in architecture. To add your own,
create a module here that calls ``register(MySpec())`` and add it to the import
list below (see ARCHITECTURES.md).
"""

from .base import (
    ArchitectureSpec,
    HParam,
    REGISTRY,
    register,
    get_spec,
    list_architectures,
)

# Importing each module runs its ``register(...)`` call as a side effect.
from . import mlp        # noqa: F401  registers "MLP"
from . import cnn        # noqa: F401  registers "CNN"
from . import resnet_mlp  # noqa: F401  registers "ResNetMLP"

__all__ = [
    "ArchitectureSpec",
    "HParam",
    "REGISTRY",
    "register",
    "get_spec",
    "list_architectures",
]
