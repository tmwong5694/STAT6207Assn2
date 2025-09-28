# Advanced non-recurrent training script for sequence regression using PyTorch
# - Trains a CNN-based regressor (no LSTM/RNN/Transformer) on X.npy, y.npy
# - Compares different values of one hyperparameter (hidden_size)
# - Evaluates on X_test.npy, y_test.npy and saves plots to advanced_results.pdf
# - Runs inference on X_test2.npy and saves predictions to a2_test.json
# - Saves a summary to advanced_results.json
#
# Notes on shapes:
#   * X may be (N, T) or (N, T, F); we convert to (N, T, F)
#   * y may be (N,), (N,1) for per-sequence regression, or (N, T) / (N, T,1) for per-timestep
#     - For per-sequence targets, the model produces (N, 1)
#     - For per-timestep targets, the model produces (N, T, 1)
#
# This script avoids any recurrent or attention-based modules and uses 1D convolutions.

import os
import json
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "PyTorch is required for this script. Please install requirements first (see requirements.txt).\n"
        f"Import error: {e}"
    )

try:
    import matplotlib
    matplotlib.use("Agg")  # headless-safe backend for saving to files
    import matplotlib.pyplot as plt
except Exception as e:
    raise SystemExit(
        "matplotlib is required for plotting. Please install requirements first (see requirements.txt).\n"
        f"Import error: {e}"
    )


# -----------------------------
# Config and utilities
# -----------------------------

@dataclass
class Config:
    data_dir: str = os.path.join(os.path.dirname(__file__), "data")
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-5
    max_epochs: int = 15  # keep modest for quick experimentation
    num_layers: int = 3
    kernel_size: int = 5
    dropout: float = 0.1
    val_ratio: float = 0.1
    seed: int = 42
    num_workers: int = 0
    hidden_size_grid: Tuple[int, ...] = (32, 64, 128)  # hyperparameter to compare
    plot_pdf_path: str = os.path.join(os.path.dirname(__file__), "advanced_results.pdf")
    results_json_path: str = os.path.join(os.path.dirname(__file__), "advanced_results.json")
    a2_json_path: str = os.path.join(os.path.dirname(__file__), "a2_test.json")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device():
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# -----------------------------
# Data loading and preprocessing
# -----------------------------

def load_arrays_all(data_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X = np.load(os.path.join(data_dir, "X.npy"))
    y = np.load(os.path.join(data_dir, "y.npy"))
    X_te = np.load(os.path.join(data_dir, "X_test.npy"))
    y_te = np.load(os.path.join(data_dir, "y_test.npy"))
    X_te2 = np.load(os.path.join(data_dir, "X_test2.npy"))
    return X, y, X_te, y_te, X_te2


def ensure_3d_X(X: np.ndarray) -> np.ndarray:
    # Ensure shape (N, T, F)
    if X.ndim == 3:
        return X
    if X.ndim == 2:
        return X[..., None]
    raise ValueError(f"X must be 2D or 3D, got shape {X.shape}")


def make_targets(y: np.ndarray) -> Tuple[np.ndarray, int, bool]:
    # Returns (y_processed, out_dim, seq_to_seq)
    if y.ndim == 1:
        y = y[:, None]
    elif y.ndim == 3 and y.shape[-1] == 1:
        y = y.squeeze(-1)  # (N, T)
    # After this, y is either (N, 1) for per-seq, or (N, T) for per-step
    if y.ndim == 2 and y.shape[1] == 1:
        out_dim = 1
        seq_to_seq = False
    elif y.ndim == 2:  # (N, T)
        out_dim = 1
        seq_to_seq = True
        y = y[..., None]  # (N, T, 1)
    else:
        raise ValueError(f"Unsupported y shape {y.shape}. Expected (N,), (N,1), or (N,T)[,1].")
    return y, out_dim, seq_to_seq


def fit_normalizer(X_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Compute mean/std over (N, T)
    mu = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
    return mu, std


def apply_normalizer(X: np.ndarray, mu: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mu) / std


def train_val_split_tensor(tensor: torch.Tensor, val_ratio: float) -> Tuple[torch.Tensor, torch.Tensor]:
    n = tensor.shape[0]
    n_val = max(1, int(n * val_ratio)) if n > 10 else max(1, int(n * 0.2))
    idx = torch.randperm(n)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return tensor[train_idx], tensor[val_idx]


# -----------------------------
# CNN model (no recurrence)
# -----------------------------

class CNNRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, kernel_size: int, dropout: float, out_dim: int, seq_to_seq: bool):
        super().__init__()
        self.seq_to_seq = seq_to_seq
        layers: List[nn.Module] = []
        in_ch = input_size  # treat feature dim as channels for Conv1d
        for i in range(num_layers):
            out_ch = hidden_size
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding="same"))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_ch = out_ch
        self.backbone = nn.Sequential(*layers)
        if seq_to_seq:
            # map hidden_size -> out_dim per timestep
            self.head_seq = nn.Conv1d(in_ch, out_dim, kernel_size=1)
            self.head_final = None
        else:
            # global average pool over time then linear to out_dim
            self.head_seq = None
            self.head_final = nn.Linear(in_ch, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        x = x.transpose(1, 2)  # -> (B, F, T)
        h = self.backbone(x)   # (B, H, T)
        if self.seq_to_seq:
            y = self.head_seq(h)           # (B, out_dim, T)
            y = y.transpose(1, 2)          # -> (B, T, out_dim)
            return y
        else:
            # global average pooling over time dimension
            h_avg = h.mean(dim=-1)         # (B, H)
            y = self.head_final(h_avg)     # (B, out_dim)
            return y


# -----------------------------
# Training and evaluation
# -----------------------------

def run_training_for_hidden_size(hidden_size: int, cfg: Config, device, X_t, y_t, Xval_t, yval_t, out_dim: int, seq_to_seq: bool) -> Dict:
    input_size = X_t.shape[-1]
    model = CNNRegressor(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=cfg.num_layers,
        kernel_size=cfg.kernel_size,
        dropout=cfg.dropout,
        out_dim=out_dim,
        seq_to_seq=seq_to_seq,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    train_ds = torch.utils.data.TensorDataset(X_t, y_t)
    val_ds = torch.utils.data.TensorDataset(Xval_t, yval_t)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    train_losses: List[float] = []
    val_losses: List[float] = []

    best_val = float("inf")
    best_state = None

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * xb.size(0)
        train_loss = running / len(train_ds)
        train_losses.append(train_loss)

        model.eval()
        with torch.no_grad():
            running = 0.0
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                running += loss.item() * xb.size(0)
            val_loss = running / max(1, len(val_ds))
        val_losses.append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch % max(1, cfg.max_epochs // 5) == 0 or epoch == 1 or epoch == cfg.max_epochs:
            print(f"[hidden_size={hidden_size}] Epoch {epoch:03d}/{cfg.max_epochs} | train MSE={train_loss:.6f} | val MSE={val_loss:.6f}")

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "hidden_size": hidden_size,
        "model": model,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_mse": float(best_val),
    }


def evaluate_model(model: nn.Module, device, X: torch.Tensor, y: torch.Tensor) -> Tuple[float, np.ndarray]:
    criterion = nn.MSELoss()
    model.eval()
    with torch.no_grad():
        yhat = model(X.to(device))
        mse = criterion(yhat, y.to(device)).item()
        yhat_np = yhat.detach().cpu().numpy()
    return mse, yhat_np


# -----------------------------
# Orchestration
# -----------------------------

def main(cfg: Config) -> Dict:
    set_seed(cfg.seed)
    device = select_device()

    # Load and shape data
    X, y, X_te, y_te, X_te2 = load_arrays_all(cfg.data_dir)
    X = ensure_3d_X(X)
    X_te = ensure_3d_X(X_te)
    X_te2 = ensure_3d_X(X_te2)
    y, out_dim, seq_to_seq = make_targets(y)
    y_te, _, _ = make_targets(y_te)

    # Fit normalizer on train only, apply to others
    mu, std = fit_normalizer(X)
    X = apply_normalizer(X, mu, std)
    X_te = apply_normalizer(X_te, mu, std)
    X_te2 = apply_normalizer(X_te2, mu, std)

    # Torch tensors
    X_t = torch.from_numpy(X.astype(np.float32))
    y_t = torch.from_numpy(y.astype(np.float32))
    Xte_t = torch.from_numpy(X_te.astype(np.float32))
    yte_t = torch.from_numpy(y_te.astype(np.float32))
    Xte2_t = torch.from_numpy(X_te2.astype(np.float32))

    # Train/Val split
    Xtr_t, Xval_t = train_val_split_tensor(X_t, cfg.val_ratio)
    ytr_t, yval_t = train_val_split_tensor(y_t, cfg.val_ratio)

    # Hyperparameter sweep over hidden_size
    runs: List[Dict] = []
    for hs in cfg.hidden_size_grid:
        res = run_training_for_hidden_size(
            hidden_size=hs,
            cfg=cfg,
            device=device,
            X_t=Xtr_t,
            y_t=ytr_t,
            Xval_t=Xval_t,
            yval_t=yval_t,
            out_dim=out_dim,
            seq_to_seq=seq_to_seq,
        )
        runs.append(res)

    # Select best by validation MSE
    best_run = min(runs, key=lambda r: r["best_val_mse"])
    best_model: nn.Module = best_run["model"]
    best_hs = best_run["hidden_size"]

    # Evaluate on test
    test_mse, yhat_te_np = evaluate_model(best_model, device, Xte_t, yte_t)

    # Inference on X_test2 for submission JSON
    best_model.eval()
    with torch.no_grad():
        yhat_te2 = best_model(Xte2_t.to(device)).detach().cpu().numpy()

    # Prepare plots and save to PDF
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: Validation MSE vs hidden_size
    hs_vals = [r["hidden_size"] for r in runs]
    val_mses = [r["best_val_mse"] for r in runs]
    axes[0].plot(hs_vals, val_mses, marker="o")
    axes[0].set_xlabel("hidden_size")
    axes[0].set_ylabel("Best Val MSE")
    axes[0].set_title("Hyperparameter comparison")
    axes[0].grid(True, alpha=0.3)

    # Right: Predictions visualization on test
    if not seq_to_seq:
        y_true = yte_t.squeeze(-1).cpu().numpy()
        y_pred = yhat_te_np.squeeze(-1)
        axes[1].scatter(y_true, y_pred, s=10, alpha=0.6)
        minv = float(min(y_true.min(), y_pred.min()))
        maxv = float(max(y_true.max(), y_pred.max()))
        axes[1].plot([minv, maxv], [minv, maxv], 'r--', linewidth=1)
        axes[1].set_xlabel("True")
        axes[1].set_ylabel("Pred")
        axes[1].set_title(f"Test scatter (MSE={test_mse:.4f}, hs={best_hs})")
    else:
        y_true = yte_t[0].squeeze(-1).cpu().numpy()
        y_pred = yhat_te_np[0].squeeze(-1)
        axes[1].plot(y_true, label="true")
        axes[1].plot(y_pred, label="pred")
        axes[1].set_xlabel("Timestep")
        axes[1].set_ylabel("Value")
        axes[1].set_title(f"Test sequence (sample 0) MSE={test_mse:.4f}, hs={best_hs}")
        axes[1].legend()

    plt.tight_layout()
    fig.savefig(cfg.plot_pdf_path, dpi=150)

    # Save advanced results JSON
    results = {
        "device": str(device),
        "input_shape": list(X.shape),
        "target_shape": list(y.shape),
        "test_input_shape": list(X_te.shape),
        "test_target_shape": list(y_te.shape),
        "seq_to_seq": bool(seq_to_seq),
        "hidden_size_grid": list(cfg.hidden_size_grid),
        "val_mse_by_hidden_size": {str(r["hidden_size"]): float(r["best_val_mse"]) for r in runs},
        "best_hidden_size": int(best_hs),
        "test_mse": float(test_mse),
        "plot_pdf_path": cfg.plot_pdf_path,
    }
    with open(cfg.results_json_path, "w") as f:
        json.dump(results, f, indent=2)

    # Save predictions for X_test2 to a2_test.json
    # Shape handling: if seq_to_seq, yhat_te2 is (N, T, 1) -> save list; else (N, 1) -> save float
    submit: Dict[str, object] = {}
    if not seq_to_seq:
        vals = yhat_te2.squeeze(-1).tolist()
        for i, v in enumerate(vals):
            submit[str(i)] = float(v)
    else:
        vals = yhat_te2.squeeze(-1).tolist()  # list of lists
        for i, arr in enumerate(vals):
            submit[str(i)] = arr

    with open(cfg.a2_json_path, "w") as f:
        json.dump(submit, f, indent=2)

    print(f"Saved PDF plot to {cfg.plot_pdf_path}")
    print(f"Saved results to {cfg.results_json_path}")
    print(f"Saved predictions to {cfg.a2_json_path}")

    return {
        **results,
        "a2_json_path": cfg.a2_json_path,
    }


if __name__ == '__main__':
    cfg = Config()
    _ = main(cfg)
