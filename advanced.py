# Advanced RNN (GRU) training script for sequence regression using PyTorch
# - Trains a GRU-based regressor (no LSTM, no Transformer) on X.npy, y.npy
# - Compares different values of one hyperparameter (now: number of epochs)
# - Evaluates on X_test.npy, y_test.npy and saves plots to advanced_results.pdf
# - Runs inference on X_test2.npy and saves predictions to a2_test.json
# - Saves a summary to advanced_results.json
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
    def __init__(self,
                 lr: float = 1e-3,
                 weight_decay: float = 1e-5,
                 batch_size: int = 128,
                 max_epochs: int = 15,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 val_ratio: float = 0.1,
                 seed: int = 42,
                 # Hyperparameter comparison now focuses on training epochs
                 epochs_grid: Tuple[int, ...] = (5, 10, 15),
                 # Fixed RNN hidden size
                 base_hidden_size: int = 64,
                 plot_pdf_path: str = None,
                 results_json_path: str = None,
                 a2_json_path: str = None):
        # Hyperparameters
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs  # default cap; per-run uses epochs_grid value
        self.num_layers = num_layers
        self.dropout = dropout
        self.val_ratio = val_ratio
        self.seed = seed
        self.epochs_grid = epochs_grid
        self.base_hidden_size = base_hidden_size

        # Output paths (default to repo root)
        base_dir = os.path.dirname(__file__)
        self.plot_pdf_path = plot_pdf_path or os.path.join(base_dir, "advanced_results.pdf")
        self.results_json_path = results_json_path or os.path.join(base_dir, "advanced_results.json")
        self.a2_json_path = a2_json_path or os.path.join(base_dir, "a2_test.json")

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
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _select_device(self):
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    # ------------------
    # Data
    # ------------------
    def load_data(self, folder: str):
        """Load arrays and keep them on the instance to avoid reloading repeatedly."""
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

        return X, y, X_test, y_test

    @staticmethod
    def _ensure_3d_X(X: np.ndarray) -> np.ndarray:
        if X.ndim == 3:
            return X
        if X.ndim == 2:
            return X[..., None]
        raise ValueError(f"X must be 2D or 3D, got shape {X.shape}")

    @staticmethod
    def _make_targets(y: np.ndarray) -> Tuple[np.ndarray, int, bool]:
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
        n_val = max(1, int(n * val_ratio)) if n > 10 else max(1, int(n * 0.2))
        idx = torch.randperm(n)
        val_idx = idx[:n_val]
        train_idx = idx[n_val:]
        return train_idx, val_idx

    def _train_one(self, hidden_size: int, num_epochs: int, Xtr_t, ytr_t, Xval_t, yval_t) -> Dict:
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
        for num_epochs in self.epochs_grid:
            res = self._train_one(self.base_hidden_size, int(num_epochs), Xtr_t, ytr_t, Xval_t, yval_t)
            self.runs.append(res)

        best = min(self.runs, key=lambda r: r["best_val_mse"])
        self.best_model = best["model"]
        self.best_hidden_size = best["hidden_size"]
        self.best_max_epochs = best["max_epochs"]
        return best

    def evaluate(self) -> Tuple[float, np.ndarray, np.ndarray]:
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
        # Single-plot figure: MSE vs epoch for each epochs setting
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))

        # Prepare colors: same color for the same hyperparameter (epochs)
        sorted_runs = sorted(self.runs, key=lambda x: x["max_epochs"]) if self.runs else []
        unique_E = sorted({r["max_epochs"] for r in sorted_runs})
        cmap = plt.cm.get_cmap('tab10', max(1, len(unique_E)))
        color_map = {E: cmap(i % cmap.N) for i, E in enumerate(unique_E)}

        for r in sorted_runs:
            E = r["max_epochs"]
            color = color_map[E]
            # Validation: solid line; Training: dashed line
            ax.plot(r["val_losses"], color=color, linestyle='-', label=f"val (E={E})")
            ax.plot(r["train_losses"], color=color, linestyle='--', alpha=0.9, label=f"train (E={E})")

        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE")
        ax.set_title(f"MSE vs Epoch (RNN hidden_size={self.best_hidden_size})")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(self.plot_pdf_path, dpi=150)
        print(f"Saved PDF plot to {self.plot_pdf_path}")

        try:
            plt.show(block=False)
        except Exception as e:
            print(f"Interactive display failed: {e}")
        finally:
            plt.close(fig)

    def save_results(self, test_mse: float, yhat_te2: np.ndarray):
        # advanced_results.json
        results = {
            "device": str(self.device),
            "input_shape": list(self.X.shape),
            "target_shape": list(self.y.shape),
            "test_input_shape": list(self.X_test.shape),
            "test_target_shape": list(self.y_test.shape),
            "seq_to_seq": bool(self.seq_to_seq),
            "base_hidden_size": int(self.base_hidden_size),
            "epochs_grid": list(self.epochs_grid),
            "val_mse_by_max_epochs": {str(r["max_epochs"]): float(r["best_val_mse"]) for r in self.runs},
            "best_hidden_size": int(self.best_hidden_size),
            "best_max_epochs": int(self.best_max_epochs),
            "test_mse": float(test_mse),
            "plot_pdf_path": self.plot_pdf_path,
        }
        with open(self.results_json_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved results to {self.results_json_path}")

        # a2_test.json
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