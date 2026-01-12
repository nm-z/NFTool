import time
from collections.abc import Callable

import numpy as np
import optuna
import torch
from sklearn.metrics import mean_absolute_error, r2_score
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from src.data.processing import preprocess_for_cnn
from src.models.architectures import model_factory


def train_model(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    input_size: int,
    config: dict,
    device: torch.device,
    patience: int,
    num_epochs: int = 200,
    batch_size: int = 32,
    gpu_throttle_sleep: float = 0.1,
    check_stop: Callable[[], bool] | None = None,
    on_epoch_end: Callable[[int, int, float, float, float], None] | None = None,
    checkpoint_callback: Callable[[nn.Module, float, float, float], None] | None = None,
) -> tuple[nn.Module | None, float, dict]:
    """Train a dense neural network and return (model, best_val_loss, history)."""
    if config is None:
        return None, float("inf"), {"train": [], "val": [], "r2": [], "mae": []}

    model = model_factory("NN", input_size, config, device)
    optimizer_class = getattr(optim, config["optimizer"])
    optimizer = optimizer_class(model.parameters(), lr=config["lr"], weight_decay=1e-5)
    criterion = nn.MSELoss()

    train_dataset = TensorDataset(x_train, y_train)
    actual_batch_size = min(batch_size, max(1, len(x_train)))
    train_loader = DataLoader(train_dataset, batch_size=actual_batch_size, shuffle=True)

    best_val_loss = float("inf")
    best_model_state = None
    history = {"train": [], "val": [], "r2": [], "mae": []}
    counter = 0

    for epoch in range(num_epochs):
        if check_stop and check_stop():
            break

        model.train()
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())

        avg_epoch_loss = epoch_loss / max(1, len(train_loader))

        if check_stop and check_stop():
            break

        model.eval()
        with torch.no_grad():
            val_output = model(x_val)
            val_loss = float(criterion(val_output, y_val).item())
            preds = val_output.cpu().numpy().flatten()
            targets = y_val.cpu().numpy().flatten()
            r2 = float(r2_score(targets, preds))
            mae = float(mean_absolute_error(targets, preds))

        history["train"].append(avg_epoch_loss)
        history["val"].append(val_loss)
        history["r2"].append(r2)
        history["mae"].append(mae)

        if on_epoch_end:
            on_epoch_end(epoch, num_epochs, avg_epoch_loss, val_loss, r2)

        if gpu_throttle_sleep > 0:
            time.sleep(gpu_throttle_sleep)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
            counter = 0
            if checkpoint_callback:
                checkpoint_callback(model, val_loss, r2, mae)
        else:
            counter += 1
            if counter >= patience:
                break

    if best_model_state:
        model.load_state_dict(best_model_state)
    return model, best_val_loss, history


def train_cnn_model(
    x_train_np: np.ndarray | torch.Tensor,
    y_train_np: np.ndarray | torch.Tensor,
    x_val_np: np.ndarray | torch.Tensor,
    y_val_np: np.ndarray | torch.Tensor,
    config: dict,
    device: torch.device,
    patience: int,
    num_epochs: int = 200,
    batch_size: int = 16,
    gpu_throttle_sleep: float = 0.1,
    check_stop: Callable[[], bool] | None = None,
    on_epoch_end: Callable[[int, int, float, float, float], None] | None = None,
    checkpoint_callback: Callable[[nn.Module, float, float, float], None] | None = None,
) -> tuple[nn.Module | None, float, dict]:
    """Train a CNN model. Inputs can be numpy arrays or torch tensors."""
    x_train = preprocess_for_cnn(x_train_np).to(device)
    x_val = preprocess_for_cnn(x_val_np).to(device)
    y_train = torch.tensor(y_train_np, dtype=torch.float32).unsqueeze(1).to(device)
    y_val = torch.tensor(y_val_np, dtype=torch.float32).unsqueeze(1).to(device)

    model = model_factory("CNN", x_train.shape[2], config, device)
    optimizer_class = getattr(optim, config["optimizer"])
    optimizer = optimizer_class(model.parameters(), lr=config["lr"], weight_decay=1e-4)
    criterion = nn.MSELoss()

    train_dataset = TensorDataset(x_train, y_train)
    actual_batch_size = min(batch_size, max(1, len(x_train)))
    train_loader = DataLoader(train_dataset, batch_size=actual_batch_size, shuffle=True)

    best_val_loss = float("inf")
    best_model_state = None
    history = {"train": [], "val": [], "r2": [], "mae": []}
    counter = 0

    for epoch in range(num_epochs):
        if check_stop and check_stop():
            break

        model.train()
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())

        avg_epoch_loss = epoch_loss / max(1, len(train_loader))

        if check_stop and check_stop():
            break

        model.eval()
        with torch.no_grad():
            val_output = model(x_val)
            val_loss = float(criterion(val_output, y_val).item())
            preds = val_output.cpu().numpy().flatten()
            targets = y_val.cpu().numpy().flatten()
            r2 = float(r2_score(targets, preds))
            mae = float(mean_absolute_error(targets, preds))

        history["train"].append(avg_epoch_loss)
        history["val"].append(val_loss)
        history["r2"].append(r2)
        history["mae"].append(mae)

        if on_epoch_end:
            on_epoch_end(epoch, num_epochs, avg_epoch_loss, val_loss, r2)

        if gpu_throttle_sleep > 0:
            time.sleep(gpu_throttle_sleep)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
            counter = 0
            if checkpoint_callback:
                checkpoint_callback(model, val_loss, r2, mae)
        else:
            counter += 1
            if counter >= patience:
                break

    if best_model_state:
        model.load_state_dict(best_model_state)
    return model, best_val_loss, history


class Objective:
    """Optuna objective wrapper that trains a model for a given trial.

    The object holds training/validation data and configuration and is callable
    by Optuna's study.
    """

    def __init__(
        self,
        model_choice: str,
        x_train: np.ndarray | torch.Tensor,
        y_train: np.ndarray | torch.Tensor,
        x_val: np.ndarray | torch.Tensor,
        y_val: np.ndarray | torch.Tensor,
        device: torch.device,
        patience: int,
        params: dict,
        on_checkpoint: Callable[[int, nn.Module, float, float, float], None]
        | None = None,
    ) -> None:
        """Initialize objective with dataset, device, and hyperparameter bounds."""
        self.model_choice = model_choice
        self.X_train = x_train
        self.y_train = y_train
        self.X_val = x_val
        self.y_val = y_val
        self.device = device
        self.patience = patience
        self.params = params
        self.on_checkpoint = on_checkpoint

    def __call__(self, trial: optuna.trial.Trial) -> float:
        """Execute a single Optuna trial and return the final loss."""
        if self.params.get("check_stop") and self.params["check_stop"]():
            trial.study.stop()
            msg = "Training aborted by user."
            raise optuna.exceptions.TrialPruned(msg)

        optimizer = trial.suggest_categorical("optimizer", self.params["optimizers"])
        lr = trial.suggest_float(
            "lr",
            self.params["lr_min"],
            self.params["lr_max"],
            log=True,
        )
        dropout = trial.suggest_float(
            "dropout",
            self.params["drop_min"],
            self.params["drop_max"],
        )

        def local_checkpoint(
            model: nn.Module,
            loss: float,
            r2: float,
            mae: float,
        ) -> None:
            """Local wrapper to call the provided on_checkpoint callback."""
            if self.on_checkpoint:
                self.on_checkpoint(trial.number, model, loss, r2, mae)

        if self.model_choice == "NN":
            n_layers = trial.suggest_int(
                "num_layers",
                self.params["n_layers_min"],
                self.params["n_layers_max"],
            )
            l_size = trial.suggest_int(
                "layer_size",
                self.params["l_size_min"],
                self.params["l_size_max"],
            )
            config = {
                "layers": [l_size] * n_layers,
                "dropout": dropout,
                "lr": lr,
                "optimizer": optimizer,
            }
            _, loss, history = train_model(
                torch.tensor(self.X_train, dtype=torch.float32).to(self.device),
                torch.tensor(self.y_train, dtype=torch.float32)
                .unsqueeze(1)
                .to(self.device),
                torch.tensor(self.X_val, dtype=torch.float32).to(self.device),
                torch.tensor(self.y_val, dtype=torch.float32)
                .unsqueeze(1)
                .to(self.device),
                self.X_train.shape[1],
                config,
                self.device,
                self.patience,
                num_epochs=self.params.get("max_epochs", 200),
                gpu_throttle_sleep=self.params.get("gpu_throttle_sleep", 0.1),
                on_epoch_end=self.params.get("on_epoch_end"),
                checkpoint_callback=local_checkpoint,
            )
        else:
            n_conv = trial.suggest_int(
                "num_conv_blocks",
                self.params.get("conv_blocks_min", 1),
                self.params.get("conv_blocks_max", 5),
            )
            base_filters = trial.suggest_int(
                "base_filters",
                self.params["l_size_min"],
                self.params["l_size_max"],
            )
            h_dim = trial.suggest_int(
                "hidden_dim",
                int(self.params["h_dim_min"]),
                int(self.params["h_dim_max"]),
            )

            cap_min = self.params.get("cnn_filter_cap_min", 512)
            cap_max = self.params.get("cnn_filter_cap_max", 512)
            current_cap = trial.suggest_int("cnn_filter_cap", cap_min, cap_max)

            conv_layers = [
                {
                    "out_channels": min(base_filters * (2**i), current_cap),
                    "kernel": self.params.get("kernel_size", 3),
                    "pool": 2,
                }
                for i in range(n_conv)
            ]
            config = {
                "conv_layers": conv_layers,
                "hidden_dim": h_dim,
                "dropout": dropout,
                "lr": lr,
                "optimizer": optimizer,
            }
            _, loss, history = train_cnn_model(
                self.X_train,
                self.y_train,
                self.X_val,
                self.y_val,
                config,
                self.device,
                self.patience,
                num_epochs=self.params.get("max_epochs", 200),
                gpu_throttle_sleep=self.params.get("gpu_throttle_sleep", 0.1),
                on_epoch_end=self.params.get("on_epoch_end"),
                checkpoint_callback=local_checkpoint,
            )

        if history["r2"]:
            trial.set_user_attr("r2", history["r2"][-1])
            trial.set_user_attr("mae", history["mae"][-1])

        return loss


def run_optimization(
    study_name: str,
    n_trials: int,
    timeout: int | None,
    objective_func: Callable[..., float],
) -> optuna.study.Study:
    """Create and run an Optuna study using the provided objective.

    Returns the completed Study object.
    """
    study = optuna.create_study(study_name=study_name, direction="minimize")
    study.optimize(objective_func, n_trials=n_trials, timeout=timeout)
    return study
