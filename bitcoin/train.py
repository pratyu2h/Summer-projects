"""
Training for the full RAF pipeline: base GRU/LSTM forecaster -> FAISS
retrieval index -> blend head. Three stages, run in order:

  1. train_base_model   -- proper mini-batch training with DataLoader
                            (the original notebook built a DataLoader and
                            never used it, training full-batch instead)
  2. build_retriever     -- encode the (non-shuffled) training set, index it
  3. train_blend_head    -- small MLP learns how much to trust retrieval
                             vs. the raw forecast, using retrieval stats
                             computed for every train/val window

evaluate() at the end reports RMSE for raw-only vs. blended predictions
on the untouched test split, so you can see whether retrieval actually
helps before writing it up as a resume bullet.
"""
import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import BTCWindowDataset
from model import BTCForecaster, RAFBlendHead
from retrieval import MarketRetriever, build_retriever


def train_base_model(model, train_loader, val_loader, device, epochs=30, lr=1e-3,
                      patience=5, checkpoint_path="base_model.pth"):
    model = model.to(device)
    loss_fn = nn.HuberLoss()  # less sensitive to outlier moves than MSE, per README roadmap
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_val_loss, best_state, epochs_no_improve = float("inf"), None, 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # LSTMs/GRUs can explode without this
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                val_losses.append(loss_fn(pred, y).item())

        train_loss, val_loss = float(np.mean(train_losses)), float(np.mean(val_losses))
        scheduler.step(val_loss)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"epoch {epoch}/{epochs}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss, best_state, epochs_no_improve = val_loss, copy.deepcopy(model.state_dict()), 0
            torch.save(best_state, checkpoint_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  early stopping at epoch {epoch} (no val improvement for {patience} epochs)")
                break

    model.load_state_dict(best_state)
    return model, history


def compute_retrieval_features(model, loader, retriever: MarketRetriever, device, k=10):
    """For every window in `loader`, get the base model's raw prediction plus
    retrieval stats (weighted mean / std of the k nearest training-set neighbors)."""
    model.eval()
    raw_preds, retrieval_means, retrieval_stds, targets = [], [], [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            emb = model.encode(X).cpu().numpy()
            raw = model.head(model.encode(X)).squeeze(-1).cpu().numpy()

            w_mean, std = retriever.query_batch_stats(emb, k=k)

            raw_preds.append(raw)
            retrieval_means.append(w_mean)
            retrieval_stds.append(std)
            targets.append(y.numpy())

    return (np.concatenate(raw_preds), np.concatenate(retrieval_means),
            np.concatenate(retrieval_stds), np.concatenate(targets))


def train_blend_head(raw_pred, retrieval_mean, retrieval_std, y, val_raw_pred, val_retrieval_mean,
                      val_retrieval_std, val_y, device, epochs=100, lr=1e-2):
    blend = RAFBlendHead().to(device)
    optimizer = torch.optim.Adam(blend.parameters(), lr=lr)
    loss_fn = nn.HuberLoss()

    to_t = lambda a: torch.from_numpy(a).float().to(device)
    raw_pred, retrieval_mean, retrieval_std, y = map(to_t, (raw_pred, retrieval_mean, retrieval_std, y))
    val_raw_pred, val_retrieval_mean, val_retrieval_std, val_y = map(
        to_t, (val_raw_pred, val_retrieval_mean, val_retrieval_std, val_y))

    best_val_loss, best_state = float("inf"), None
    for epoch in range(1, epochs + 1):
        blend.train()
        pred = blend(raw_pred, retrieval_mean, retrieval_std)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        blend.eval()
        with torch.no_grad():
            val_pred = blend(val_raw_pred, val_retrieval_mean, val_retrieval_std)
            val_loss = loss_fn(val_pred, val_y).item()
        if val_loss < best_val_loss:
            best_val_loss, best_state = val_loss, copy.deepcopy(blend.state_dict())

    blend.load_state_dict(best_state)
    if epochs > 0:
        print(f"blend head trained, best val_loss={best_val_loss:.6f}")
    return blend


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - target) ** 2)))


if __name__ == "__main__":
    # end-to-end smoke test on synthetic data -- proves the full RAF
    # pipeline (base model -> retriever -> blend head -> eval) runs
    import pandas as pd

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    device = torch.device("cpu")

    N, T, F = 300, 20, 20
    X = rng.normal(size=(N, T, F)).astype(np.float32)
    y = rng.normal(scale=0.01, size=N).astype(np.float32)
    ts = pd.Series(pd.date_range("2024-01-01", periods=N, freq="min"))

    n_train, n_val = 200, 50
    train_ds = BTCWindowDataset(X[:n_train], y[:n_train])
    val_ds = BTCWindowDataset(X[n_train:n_train + n_val], y[n_train:n_train + n_val])
    test_ds = BTCWindowDataset(X[n_train + n_val:], y[n_train + n_val:])

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    train_loader_noshuffle = DataLoader(train_ds, batch_size=16, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    model = BTCForecaster(num_features=F, hidden_size=16, num_layers=1)
    model, history = train_base_model(model, train_loader, val_loader, device, epochs=3, checkpoint_path="/tmp/smoke_base.pth")
    assert len(history) > 0

    retriever = build_retriever(model, train_loader_noshuffle, ts[:n_train], device)

    tr_raw, tr_rmean, tr_rstd, tr_y = compute_retrieval_features(model, train_loader_noshuffle, retriever, device)
    va_raw, va_rmean, va_rstd, va_y = compute_retrieval_features(model, val_loader, retriever, device)
    te_raw, te_rmean, te_rstd, te_y = compute_retrieval_features(model, test_loader, retriever, device)

    blend = train_blend_head(tr_raw, tr_rmean, tr_rstd, tr_y, va_raw, va_rmean, va_rstd, va_y, device, epochs=20)

    with torch.no_grad():
        blended_test_pred = blend(
            torch.from_numpy(te_raw).float(), torch.from_numpy(te_rmean).float(), torch.from_numpy(te_rstd).float()
        ).numpy()

    raw_rmse = rmse(te_raw, te_y)
    blended_rmse = rmse(blended_test_pred, te_y)
    print(f"[ok] test RMSE raw={raw_rmse:.6f}  blended={blended_rmse:.6f}")
    assert np.isfinite(raw_rmse) and np.isfinite(blended_rmse)

    print("\nALL SMOKE TESTS PASSED")
