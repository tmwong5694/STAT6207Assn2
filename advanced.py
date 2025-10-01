"""
Advanced sequence regression using a GRU-based RNN in PyTorch (no LSTM, no Transformer).

What this file does
- Loads training/test arrays from the local data folder and keeps them cached on the Model instance.
- Normalizes inputs using statistics computed on the training set only.
- Trains a GRU-based regressor while comparing different hidden sizes (hidden_size_grid),
  with a fixed number of epochs per run (max_epochs, default 15).
- Plots MSE vs Epoch for each run (train dashed, val solid) using fixed colors (red/green/blue),
  saves to advanced_results.png, and also shows the figure interactively.
- Generates predictions for X_test2.npy and writes them to a2_test.json with string keys '0', '1', ...

Input/Output shapes (defensive handling)
- X: (N, T) or (N, T, F) -> coerced to (N, T, F)
- y: (N,), (N, 1) for sequence-level targets or (N, T)[,1] for per-timestep targets
  * seq_to_one: model outputs (N, 1)
  * seq_to_seq: model outputs (N, T, 1)

Notes
- Device selection prefers Apple Silicon MPS, then CUDA, else CPU.
- The same train/val split is shared across all hidden-size runs for a fair comparison.
"""
# Advanced RNN (GRU) training script for sequence regression using PyTorch
# - Trains a GRU-based regressor (no LSTM, no Transformer) on X.npy, y.npy
# - Compares different values of one hyperparameter (now: number of epochs)
# - Evaluates on X_test.npy, y_test.npy and saves a plot (PNG) + displays it
# - Runs inference on X_test2.npy and saves predictions to a2_test.json
#
# Shapes:
#   * X may be (N, T) or (N, T, F) -> we ensure (N, T, F)
#   * y may be (N,), (N,1) per-sequence or (N, T)[,1] per-timestep
#     - seq_to_one: output (N, 1)
#     - seq_to_seq: output (N, T, 1)

import os
import json
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class Model:
    """End-to-end trainer/evaluator for a GRU regressor.

    Keeps data arrays, normalization stats, and the best trained model in memory so
    subsequent calls do not reload from disk. Use:

        model = Model()
        model.load_data('./data')
        model.train()            # runs multiple hidden sizes in hidden_size_grid
        test_mse, yhat, y2 = model.evaluate()
        model.plot_and_save(test_mse, yhat)
        model.save_results(test_mse, y2)
    """
    def __init__(self,
                 lr: float = 1e-3,
                 weight_decay: float = 1e-5,
                 batch_size: int = 128,
                 max_epochs: int = 15,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 val_ratio: float = 0.1,
                 seed: int = 42,
                 # Hyperparameter comparison now focuses on hidden size
                 hidden_size_grid: Tuple[int, ...] = (16, 32, 64),
                 # Base/initial hidden size (not swept if grid provided)
                 base_hidden_size: int = 64,
                 a2_json_path: str = None,
                 plot_path: str = None):
        # Hyperparameters
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.num_layers = num_layers
        self.dropout = dropout
        self.val_ratio = val_ratio
        self.seed = seed
        self.hidden_size_grid = hidden_size_grid
        self.base_hidden_size = base_hidden_size

        # Output paths (default to repo root)
        base_dir = os.path.dirname(__file__)
        self.a2_json_path = a2_json_path or os.path.join(base_dir, "a2_test.json")
        self.plot_path = plot_path or os.path.join(base_dir, "advanced_results.png")

        # Runtime state
        self.device = self._select_device()
        self._set_seed(self.seed)

        # Data placeholders (set by load_data)
        self.X = self.y = self.X_test = self.y_test = self.X_test2 = None
        self.mu = self.std = None
        self.seq_to_seq = False
        self.out_dim = 1

        # Trained model / best config
        self.best_model: nn.Module | None = None
        self.best_hidden_size: int | None = None
        self.best_max_epochs: int | None = None
        self.runs: List[Dict] = []

    # ------------------
    # Utilities
    # ------------------
    def _set_seed(self, seed: int):
        """Seed python, numpy, and torch RNGs for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _select_device(self):
        """Choose device in order: MPS (Apple Silicon) > CUDA > CPU."""
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    # ------------------
    # Data
    # ------------------
    def load_data(self, folder: str):
        """Load arrays from folder and normalize inputs using train-set stats.

        Caches arrays as attributes so repeated training doesn't reload from disk.
        Returns the raw (unnormalized) arrays for convenience, mirroring test.py style.
        """
        X = np.load(os.path.join(folder, 'X.npy'))
        y = np.load(os.path.join(folder, 'y.npy'))
        X_test = np.load(os.path.join(folder, 'X_test.npy'))
        y_test = np.load(os.path.join(folder, 'y_test.npy'))
        X_test2 = np.load(os.path.join(folder, 'X_test2.npy'))

        X = self._ensure_3d_X(X)
        X_test = self._ensure_3d_X(X_test)
        X_test2 = self._ensure_3d_X(X_test2)

        y, out_dim, seq_to_seq = self._make_targets(y)
        y_test, _, _ = self._make_targets(y_test)

        # Normalize by train statistics
        mu = X.mean(axis=(0, 1), keepdims=True)
        std = X.std(axis=(0, 1), keepdims=True) + 1e-8

        self.X = (X - mu) / std
        self.y = y
        self.X_test = (X_test - mu) / std
        self.y_test = y_test
        self.X_test2 = (X_test2 - mu) / std
        self.mu, self.std = mu, std
        self.out_dim = out_dim
        self.seq_to_seq = seq_to_seq

        return X, y, X_test, y_test, X_test2

    @staticmethod
    def _ensure_3d_X(X: np.ndarray) -> np.ndarray:
        """Ensure input X has shape (N, T, F); append a unit feature dim if needed."""
        if X.ndim == 3:
            return X
        if X.ndim == 2:
            return X[..., None]
        raise ValueError(f"X must be 2D or 3D, got shape {X.shape}")

    @staticmethod
    def _make_targets(y: np.ndarray) -> Tuple[np.ndarray, int, bool]:
        """Coerce y into supported targets and return (y_processed, out_dim, seq_to_seq)."""
        if y.ndim == 1:
            y = y[:, None]
        elif y.ndim == 3 and y.shape[-1] == 1:
            y = y.squeeze(-1)  # (N, T)
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

    # ------------------
    # Model (RNN/GRU)
    # ------------------
    class RNNRegressor(nn.Module):
        """GRU backbone for sequence regression.

        - If seq_to_seq=True, predicts a value at each time step (B, T, 1).
        - Otherwise, predicts a single value per sequence from the final hidden state (B, 1).
        """
        def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float, out_dim: int, seq_to_seq: bool):
            super().__init__()
            self.seq_to_seq = seq_to_seq
            self.hidden_size = hidden_size
            # Project features to hidden size if needed
            self.input_proj = nn.Linear(input_size, hidden_size) if input_size != hidden_size else nn.Identity()
            # Use GRU (no LSTM)
            self.rnn = nn.GRU(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True,
            )
            if seq_to_seq:
                self.head_seq = nn.Linear(hidden_size, out_dim)
                self.head_final = None
            else:
                self.head_seq = None
                self.head_final = nn.Linear(hidden_size, out_dim)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass for the GRU regressor.

            Args
            - x: (B, T, F) float32 tensor
            Returns
            - (B, T, out_dim) if seq_to_seq else (B, out_dim)
            """
            # x: (B, T, F)
            x = self.input_proj(x)  # (B, T, H)
            out, h_n = self.rnn(x)  # out: (B, T, H), h_n: (L, B, H)
            if self.seq_to_seq:
                y = self.head_seq(self.dropout(out))  # (B, T, out_dim)
                return y
            else:
                last = h_n[-1]  # (B, H) last layer's hidden state
                last = self.dropout(last)
                return self.head_final(last)  # (B, out_dim)

    def _build_model(self, hidden_size: int) -> nn.Module:
        """Construct a GRU regressor with the configured depth/dropout and given hidden size."""
        input_size = self.X.shape[-1]
        return Model.RNNRegressor(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            out_dim=self.out_dim,
            seq_to_seq=self.seq_to_seq,
        ).to(self.device)

    # ------------------
    # Training / Eval
    # ------------------
    @staticmethod
    def _make_split_indices(n: int, val_ratio: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create random train/val index tensors with at least one validation sample."""
        n_val = max(1, int(n * val_ratio)) if n > 10 else max(1, int(n * 0.2))
        idx = torch.randperm(n)
        val_idx = idx[:n_val]
        train_idx = idx[n_val:]
        return train_idx, val_idx

    def _train_one(self, hidden_size: int, num_epochs: int, Xtr_t, ytr_t, Xval_t, yval_t) -> Dict:
        """Train one run for a fixed number of epochs and return learning curves and the model.

        Uses Adam + MSELoss and keeps the best validation checkpoint.
        """
        model = self._build_model(hidden_size)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        train_ds = torch.utils.data.TensorDataset(Xtr_t, ytr_t)
        val_ds = torch.utils.data.TensorDataset(Xval_t, yval_t)
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, num_workers=0)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=self.batch_size, shuffle=False, num_workers=0)

        train_losses: List[float] = []
        val_losses: List[float] = []
        best_val = float('inf')
        best_state = None

        for epoch in range(1, num_epochs + 1):
            model.train()
            running = 0.0
            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                # optional: gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                running += loss.item() * xb.size(0)
            train_loss = running / len(train_ds)
            train_losses.append(train_loss)

            model.eval()
            with torch.no_grad():
                running = 0.0
                for xb, yb in val_loader:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    pred = model(xb)
                    loss = criterion(pred, yb)
                    running += loss.item() * xb.size(0)
                val_loss = running / max(1, len(val_ds))
            val_losses.append(val_loss)

            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            if epoch % max(1, num_epochs // 5) == 0 or epoch == 1 or epoch == num_epochs:
                print(f"[epochs={num_epochs}, hidden_size={hidden_size}] Epoch {epoch:03d}/{num_epochs} | train MSE={train_loss:.6f} | val MSE={val_loss:.6f}")

        if best_state is not None:
            model.load_state_dict(best_state)

        return {
            "hidden_size": hidden_size,
            "max_epochs": int(num_epochs),
            "model": model,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_val_mse": float(best_val),
        }

    def train(self):
        """Run multiple training runs over hidden_size_grid and keep the best model (by val MSE)."""
        assert self.X is not None, "Call load_data(folder) first."
        # Prepare tensors
        X_t = torch.from_numpy(self.X.astype(np.float32))
        y_t = torch.from_numpy(self.y.astype(np.float32))
        # Shared split indices to keep X and y aligned
        n = X_t.shape[0]
        train_idx, val_idx = self._make_split_indices(n, self.val_ratio)
        Xtr_t, Xval_t = X_t[train_idx], X_t[val_idx]
        ytr_t, yval_t = y_t[train_idx], y_t[val_idx]

        self.runs = []
        for h in self.hidden_size_grid:
            res = self._train_one(int(h), int(self.max_epochs), Xtr_t, ytr_t, Xval_t, yval_t)
            self.runs.append(res)

        best = min(self.runs, key=lambda r: r["best_val_mse"])
        self.best_model = best["model"]
        self.best_hidden_size = best["hidden_size"]
        self.best_max_epochs = int(self.max_epochs)
        return best

    def evaluate(self) -> Tuple[float, np.ndarray, np.ndarray]:
        """Evaluate the best model on the held-out test set and prepare test2 predictions."""
        assert self.best_model is not None, "Train the model first."
        Xte_t = torch.from_numpy(self.X_test.astype(np.float32)).to(self.device)
        yte_t = torch.from_numpy(self.y_test.astype(np.float32)).to(self.device)

        self.best_model.eval()
        criterion = nn.MSELoss()
        with torch.no_grad():
            yhat_te = self.best_model(Xte_t)
            test_mse = criterion(yhat_te, yte_t).item()
            yhat_te_np = yhat_te.detach().cpu().numpy()

            Xte2_t = torch.from_numpy(self.X_test2.astype(np.float32)).to(self.device)
            yhat_te2 = self.best_model(Xte2_t).detach().cpu().numpy()

        return float(test_mse), yhat_te_np, yhat_te2

    # ------------------
    # Reporting
    # ------------------
    def plot_and_save(self, test_mse: float, yhat_te_np: np.ndarray):
        """Plot MSE vs epoch for each hidden size, save PNG, and show interactively.

        Styling:
        - Colors: red, green, blue mapped to the unique hidden sizes.
        - Train curves: dashed; Validation curves: solid.
        """
        # Single-plot figure: MSE vs epoch for each hidden size
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))

        # Prepare colors for hidden sizes
        sorted_runs = sorted(self.runs, key=lambda x: x["hidden_size"]) if self.runs else []
        unique_H = sorted({r["hidden_size"] for r in sorted_runs})
        base_colors = ['red', 'green', 'blue']
        color_map = {H: base_colors[i % len(base_colors)] for i, H in enumerate(unique_H)}

        for r in sorted_runs:
            H = r["hidden_size"]
            color = color_map[H]
            # Validation: solid line; Training: dashed line
            ax.plot(r["val_losses"], color=color, linestyle='-', label=f"val (H={H})")
            ax.plot(r["train_losses"], color=color, linestyle='--', alpha=0.9, label=f"train (H={H})")

        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE")
        ax.set_title(f"MSE vs Epoch (hidden sizes {unique_H}, epochs={self.max_epochs})")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save PNG and also display
        fig.savefig(self.plot_path, dpi=150)
        print(f"Saved plot to {self.plot_path}")
        try:
            plt.show(block=False)
        except Exception as e:
            print(f"Interactive display failed: {e}")
        finally:
            plt.close(fig)

    def save_results(self, test_mse: float, yhat_te2: np.ndarray):
        """Save only a2_test.json with predictions for X_test2.

        Keys are strings '0', '1', ... so the file matches the expected submission format.
        For seq-to-seq targets, each value is a list of per-step predictions.
        """
        # Only save a2_test.json with predictions for X_test2
        submit: Dict[str, object] = {}
        if not self.seq_to_seq:
            vals = yhat_te2.squeeze(-1).tolist()
            for i, v in enumerate(vals):
                submit[str(i)] = float(v)
        else:
            vals = yhat_te2.squeeze(-1).tolist()
            for i, arr in enumerate(vals):
                submit[str(i)] = arr
        with open(self.a2_json_path, "w") as f:
            json.dump(submit, f, indent=2)
        print(f"Saved predictions to {self.a2_json_path}")


if __name__ == '__main__':
    # Follow the simple format pattern from test.py
    data_folder = os.path.join(os.path.dirname(__file__), 'data')
    model = Model()
    model.load_data(data_folder)
    model.train()
    test_mse, yhat_te_np, yhat_te2 = model.evaluate()
    model.plot_and_save(test_mse, yhat_te_np)
    model.save_results(test_mse, yhat_te2)

    pass