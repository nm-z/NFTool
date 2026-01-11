import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, r2_score
import optuna

from src.models.architectures import model_factory
from src.data.processing import preprocess_for_cnn

def train_model(X_train, y_train, X_val, y_val, input_size, config, device, patience, 
                num_epochs=200, batch_size=32, gpu_throttle_sleep=0.1, check_stop=None, on_epoch_end=None):
    if config is None:
        return None, float("inf"), {'train': [], 'val': [], 'r2': [], 'mae': []}
    
    model = model_factory("NN", input_size, config, device)
    optimizer_class = getattr(optim, config['optimizer'])
    optimizer = optimizer_class(model.parameters(), lr=config['lr'], weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    train_dataset = TensorDataset(X_train, y_train)
    # Adjust batch size for small datasets
    actual_batch_size = min(batch_size, len(X_train))
    train_loader = DataLoader(train_dataset, batch_size=actual_batch_size, shuffle=True)
    
    best_val_loss = float('inf')
    best_model_state = None
    history = {'train': [], 'val': [], 'r2': [], 'mae': []}
    counter = 0

    for epoch in range(num_epochs):
        if check_stop and check_stop():
            break

        model.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_epoch_loss = epoch_loss / len(train_loader)

        if check_stop and check_stop():
            break

        model.eval()
        with torch.no_grad():
            val_output = model(X_val)
            val_loss = criterion(val_output, y_val).item()
            preds = val_output.cpu().numpy().flatten()
            targets = y_val.cpu().numpy().flatten()
            r2 = r2_score(targets, preds)
            mae = mean_absolute_error(targets, preds)

        history['train'].append(avg_epoch_loss)
        history['val'].append(val_loss)
        history['r2'].append(r2)
        history['mae'].append(mae)

        if on_epoch_end:
            on_epoch_end(epoch, num_epochs, avg_epoch_loss, val_loss, r2)

        if gpu_throttle_sleep > 0:
            time.sleep(gpu_throttle_sleep)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            counter = 0
        else:
            counter += 1
            if counter >= patience: break

    if best_model_state:
        model.load_state_dict(best_model_state)
    return model, best_val_loss, history

def train_cnn_model(X_train_np, y_train_np, X_val_np, y_val_np, config, device, patience, 
                   num_epochs=200, batch_size=16, gpu_throttle_sleep=0.1, check_stop=None, on_epoch_end=None):
    X_train = preprocess_for_cnn(X_train_np).to(device)
    X_val = preprocess_for_cnn(X_val_np).to(device)
    y_train = torch.tensor(y_train_np, dtype=torch.float32).unsqueeze(1).to(device)
    y_val = torch.tensor(y_val_np, dtype=torch.float32).unsqueeze(1).to(device)

    model = model_factory("CNN", X_train.shape[2], config, device)
    optimizer_class = getattr(optim, config['optimizer'])
    # Add weight decay for regularization
    optimizer = optimizer_class(model.parameters(), lr=config['lr'], weight_decay=1e-4)
    criterion = nn.MSELoss()

    train_dataset = TensorDataset(X_train, y_train)
    # Use smaller batches for 108 samples to get more updates per epoch
    actual_batch_size = min(batch_size, len(X_train))
    train_loader = DataLoader(train_dataset, batch_size=actual_batch_size, shuffle=True)

    best_val_loss = float('inf')
    best_model_state = None
    history = {"train": [], "val": [], "r2": [], "mae": []}
    counter = 0

    for epoch in range(num_epochs):
        if check_stop and check_stop():
            break

        model.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_epoch_loss = epoch_loss / len(train_loader)

        if check_stop and check_stop():
            break

        model.eval()
        with torch.no_grad():
            val_output = model(X_val)
            val_loss = criterion(val_output, y_val).item()
            preds = val_output.cpu().numpy().flatten()
            targets = y_val.cpu().numpy().flatten()
            r2 = r2_score(targets, preds)
            mae = mean_absolute_error(targets, preds)

        history['train'].append(avg_epoch_loss)
        history['val'].append(val_loss)
        history['r2'].append(r2)
        history['mae'].append(mae)

        if on_epoch_end:
            on_epoch_end(epoch, num_epochs, avg_epoch_loss, val_loss, r2)

        if gpu_throttle_sleep > 0:
            time.sleep(gpu_throttle_sleep)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            counter = 0
        else:
            counter += 1
            if counter >= patience: break

    if best_model_state:
        model.load_state_dict(best_model_state)
    return model, best_val_loss, history

class Objective:
    def __init__(self, model_choice, X_train, y_train, X_val, y_val, device, patience, params):
        self.model_choice = model_choice
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.device = device
        self.patience = patience
        self.params = params

    def __call__(self, trial):
        if self.params.get('check_stop') and self.params['check_stop']():
            trial.study.stop()
            raise optuna.exceptions.TrialPruned("Training aborted by user.")

        optimizer = trial.suggest_categorical("optimizer", self.params['optimizers'])
        lr = trial.suggest_float("lr", self.params['lr_min'], self.params['lr_max'], log=True)
        dropout = trial.suggest_float("dropout", self.params['drop_min'], self.params['drop_max'])
        
        if self.model_choice == "NN":
            n_layers = trial.suggest_int("num_layers", self.params['n_layers_min'], self.params['n_layers_max'])
            l_size = trial.suggest_int("layer_size", self.params['l_size_min'], self.params['l_size_max'])
            config = {'layers': [l_size] * n_layers, 'dropout': dropout, 'lr': lr, 'optimizer': optimizer}
            _, loss, history = train_model(
                torch.tensor(self.X_train, dtype=torch.float32).to(self.device),
                torch.tensor(self.y_train, dtype=torch.float32).unsqueeze(1).to(self.device),
                torch.tensor(self.X_val, dtype=torch.float32).to(self.device),
                torch.tensor(self.y_val, dtype=torch.float32).unsqueeze(1).to(self.device),
                self.X_train.shape[1], config, self.device, self.patience
            )
        else:
            n_conv = trial.suggest_int("num_conv_blocks", 1, 5)
            base_filters = trial.suggest_int("base_filters", self.params['l_size_min'], self.params['l_size_max'])
            h_dim = trial.suggest_int("hidden_dim", int(self.params['h_dim_min']), int(self.params['h_dim_max']))
            
            # Use dynamic filter cap if provided in params, otherwise default to a high value
            cap_min = self.params.get('cnn_filter_cap_min', 512)
            cap_max = self.params.get('cnn_filter_cap_max', 512)
            current_cap = trial.suggest_int("cnn_filter_cap", cap_min, cap_max)
            
            conv_layers = [{'out_channels': min(base_filters * (2**i), current_cap), 'kernel': 3, 'pool': 2} for i in range(n_conv)]
            config = {'conv_layers': conv_layers, 'hidden_dim': h_dim, 'dropout': dropout, 'lr': lr, 'optimizer': optimizer}
            _, loss, history = train_cnn_model(self.X_train, self.y_train, self.X_val, self.y_val, config, self.device, self.patience)
        
        if history['r2']:
            trial.set_user_attr("r2", history['r2'][-1])
            trial.set_user_attr("mae", history['mae'][-1])
            
        return loss

def run_optimization(study_name, n_trials, timeout, objective_func):
    study = optuna.create_study(study_name=study_name, direction="minimize")
    study.optimize(objective_func, n_trials=n_trials, timeout=timeout)
    return study
