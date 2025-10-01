# Basic LSTM training script for sequence regression using PyTorch
# - Loads X.npy, y.npy for training
# - Evaluates on X_test.npy, y_test.npy and reports MSE
# - Saves learning curves to basic_results.png and also shows them
#
# The script is defensive about data shapes:
#   * X may be (N, T) or (N, T, F); it will reshape to (N, T, F)
#   * y may be (N,), (N, 1), (N, T), or (N, T, 1)
#     - If y is per-sequence (N or N,1): we use last hidden state
#     - If y is per-timestep (N,T or N,T,1): we use sequence outputs

import os
import json
import random
from dataclasses import dataclass
from typing import Tuple

import numpy as np

try:
    import torch
except Exception as e:  # pragma: no cover - provide clearer error
    raise SystemExit(
        "PyTorch is required for this script. Please install requirements first (see requirements.txt).\n"
        f"Import error: {e}"
    )

try:
    import matplotlib.pyplot as plt
except Exception as e:
    raise SystemExit(
        "matplotlib is required for plotting. Please install requirements first (see requirements.txt).\n"
        f"Import error: {e}"
    )


@dataclass
class Config:
    data_dir: str = os.path.join(os.path.dirname(__file__), "data")
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-5
    max_epochs: int = 30
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.0
    val_ratio: float = 0.1
    seed: int = 42
    num_workers: int = 0  # set >0 if using a faster CPU dataloader
    plot_path: str = os.path.join(os.path.dirname(__file__), "basic_results.png")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device():
    # Prefer Apple Silicon MPS if available, else CUDA, else CPU
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_arrays(data_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.load(os.path.join(data_dir, "X.npy"))
    y = np.load(os.path.join(data_dir, "y.npy"))
    x_te = np.load(os.path.join(data_dir, "X_test.npy"))
    y_te = np.load(os.path.join(data_dir, "y_test.npy"))
    return x, y, x_te, y_te


def ensure_3d_X(X: np.ndarray) -> np.ndarray:
    # Ensure shape (N, T, F)
    if X.ndim == 3:
        return X
    if X.ndim == 2:
        return X[..., None]
    raise ValueError(f"X must be 2D or 3D, got shape {X.shape}")


def normalize(X_train: np.ndarray, X_other: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Standardize features using train statistics only
    # X shape: (N, T, F)
    mu = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
    return (X_train - mu) / std, (X_other - mu) / std


class LSTMRegressor(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float, out_dim: int, seq_to_seq: bool):
        super().__init__()
        self.seq_to_seq = seq_to_seq
        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = torch.nn.Linear(hidden_size, out_dim)

    def forward(self, x):
        # x: (B, T, F)
        out, (h_n, c_n) = self.lstm(x)
        if self.seq_to_seq:
            # out: (B, T, H) -> (B, T, out_dim)
            return self.head(out)
        else:
            # Use last hidden state (num_layers last): h_n[-1] shape (B, H)
            last = h_n[-1]
            return self.head(last)


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


def train_val_split_tensor(tensor: torch.Tensor, val_ratio: float) -> Tuple[torch.Tensor, torch.Tensor]:
    n = tensor.shape[0]
    n_val = max(1, int(n * val_ratio)) if n > 10 else max(1, int(n * 0.2))
    idx = torch.randperm(n)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return tensor[train_idx], tensor[val_idx]


# -----------------------------
# Data holder with caching
# -----------------------------
class SequenceData:
    """Loads and holds train/test arrays once, with normalization and target shaping.

    Attributes
    - X, y: normalized train inputs and targets (numpy)
    - X_te, y_te: normalized test inputs and targets (numpy)
    - mu, std: normalization params computed from X
    - out_dim, seq_to_seq: target config flags
    """
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        X, y, X_te, y_te = load_arrays(data_dir)
        X = ensure_3d_X(X)
        X_te = ensure_3d_X(X_te)
        y, out_dim, seq_to_seq = make_targets(y)
        y_te, _, _ = make_targets(y_te)

        # Fit normalizer on full training inputs (consistent with prior behavior)
        mu = X.mean(axis=(0, 1), keepdims=True)
        std = X.std(axis=(0, 1), keepdims=True) + 1e-8

        self.mu = mu
        self.std = std
        self.X = (X - mu) / std
        self.X_te = (X_te - mu) / std
        self.y = y
        self.y_te = y_te
        self.out_dim = out_dim
        self.seq_to_seq = seq_to_seq

    def tensors(self):
        X_t = torch.from_numpy(self.X.astype(np.float32))
        y_t = torch.from_numpy(self.y.astype(np.float32))
        Xte_t = torch.from_numpy(self.X_te.astype(np.float32))
        yte_t = torch.from_numpy(self.y_te.astype(np.float32))
        return X_t, y_t, Xte_t, yte_t


_DATA_CACHE = None  # module-level cache


def get_data(cfg: Config) -> SequenceData:
    global _DATA_CACHE
    if _DATA_CACHE is None or getattr(_DATA_CACHE, "data_dir", None) != cfg.data_dir:
        _DATA_CACHE = SequenceData(cfg.data_dir)
    return _DATA_CACHE


def train_model(cfg: Config) -> dict:
    set_seed(cfg.seed)
    device = select_device()

    # Use cached data (loaded once per process)
    data = get_data(cfg)
    X_t, y_t, Xte_t, yte_t = data.tensors()
    X = data.X
    y = data.y
    X_te = data.X_te
    y_te = data.y_te
    out_dim = data.out_dim
    seq_to_seq = data.seq_to_seq

    # Train/Val split
    Xtr_t, Xval_t = train_val_split_tensor(X_t, cfg.val_ratio)
    ytr_t, yval_t = train_val_split_tensor(y_t, cfg.val_ratio)

    # DataLoaders
    train_ds = torch.utils.data.TensorDataset(Xtr_t, ytr_t)
    val_ds = torch.utils.data.TensorDataset(Xval_t, yval_t)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    # Model
    input_size = X.shape[-1]
    model = LSTMRegressor(
        input_size=input_size,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        out_dim=out_dim,
        seq_to_seq=seq_to_seq,
    ).to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    train_losses = []
    val_losses = []

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

        # Val
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

        if epoch % max(1, cfg.max_epochs // 10) == 0 or epoch == 1 or epoch == cfg.max_epochs:
            print(f"Epoch {epoch:03d}/{cfg.max_epochs} | train MSE={train_loss:.6f} | val MSE={val_loss:.6f}")

    # Test evaluation
    model.eval()
    with torch.no_grad():
        yhat_te = model(Xte_t.to(device))
        test_mse = criterion(yhat_te, yte_t.to(device)).item()
        # yhat_te_np = yhat_te.detach().cpu().numpy()  # no prediction plotting needed

    # Plot only MSE curves (no scatter/prediction plot)
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(train_losses, label="train", linestyle='--')
    ax.plot(val_losses, label="val", linestyle='-')
    ax.set_title("MSE vs Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    fig.savefig(cfg.plot_path, dpi=150)
    print(f"Saved plot to {cfg.plot_path}")

    # Also show the plot interactively
    try:
        plt.show(block=False)
    except Exception as e:
        print(f"Interactive display failed: {e}")
    finally:
        plt.close(fig)

    results = {
        "device": str(device),
        "input_shape": list(X.shape),
        "target_shape": list(y.shape),
        "test_input_shape": list(X_te.shape),
        "test_target_shape": list(y_te.shape),
        "seq_to_seq": bool(seq_to_seq),
        "test_mse": float(test_mse),
        "plot_path": cfg.plot_path,
    }
    with open(os.path.join(os.path.dirname(__file__), "basic_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("Results:")
    print(json.dumps(results, indent=2))

    return results


if __name__ == '__main__':
    cfg = Config()
    _ = train_model(cfg)
