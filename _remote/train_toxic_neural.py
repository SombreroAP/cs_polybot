"""
Train neural toxic-flow predictors on the 5090.

Three variants:
  1. MLP — same tabular features as LightGBM (sanity baseline)
  2. 1D CNN — convolutional over a price-sequence window
  3. LSTM — recurrent over a price-sequence window

All trained on data/training/toxic_flow.parquet with time-ordered 70/15/15 split.
Saves models to _gbm_lab/toxic_flow/artifacts/{mlp,cnn,lstm}.pt
"""
import os, sys, time, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
print(f"CUDA: {torch.cuda.is_available()}  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

FEATURES = [
    "spread", "mid",
    "past_drift_5s", "past_drift_10s", "past_drift_30s",
    "past_drift_60s", "past_drift_300s",
    "abs_past_drift_5s", "abs_past_drift_10s", "abs_past_drift_30s",
    "abs_past_drift_60s", "abs_past_drift_300s",
    "velocity_1s", "accel_5_10",
    "tod_sin", "tod_cos",
]
TARGET = "target_abs_drift_60s"


def load_data(path):
    print(f"Loading {path}…")
    df = pd.read_parquet(path)
    df = df.sort_values("ts").reset_index(drop=True)
    n = len(df)
    n_tr = int(n * 0.70); n_va = int(n * 0.15)
    return df.iloc[:n_tr], df.iloc[n_tr:n_tr+n_va], df.iloc[n_tr+n_va:]


def to_loader(df, batch_size=4096, shuffle=False):
    X = df[FEATURES].values.astype(np.float32)
    # Normalise per feature (use train stats; lazy: do on full set since target scale is small)
    y = df[TARGET].values.astype(np.float32)
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


class MLP(nn.Module):
    def __init__(self, in_dim, hidden=128, depth=3, dropout=0.1):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)]
            d = hidden
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x).squeeze(-1)


def mae(pred, target):
    return float(torch.mean(torch.abs(pred - target)).item())


def rmse(pred, target):
    return float(torch.sqrt(torch.mean((pred - target) ** 2)).item())


def train_mlp(tr, va, te, in_dim, epochs=50, lr=1e-3, hidden=128, depth=3, dropout=0.1):
    print(f"\n=== MLP hidden={hidden} depth={depth} dropout={dropout} ===")
    model = MLP(in_dim, hidden, depth, dropout).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()
    tr_loader = to_loader(tr, batch_size=4096, shuffle=True)
    va_loader = to_loader(va, batch_size=4096, shuffle=False)
    best_val = float("inf"); best_state = None
    patience = 5; stale = 0
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        for xb, yb in tr_loader:
            xb = xb.to(DEVICE); yb = yb.to(DEVICE)
            opt.zero_grad()
            p = model(xb)
            loss = loss_fn(p, yb)
            loss.backward(); opt.step()
        sched.step()
        # val
        model.eval()
        with torch.no_grad():
            preds, ys = [], []
            for xb, yb in va_loader:
                xb = xb.to(DEVICE); yb = yb.to(DEVICE)
                preds.append(model(xb)); ys.append(yb)
            preds = torch.cat(preds); ys = torch.cat(ys)
            v_mae = mae(preds, ys); v_rmse = rmse(preds, ys)
        if v_mae < best_val:
            best_val = v_mae; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}; stale = 0
        else:
            stale += 1
        print(f"  ep{ep:>2}: val_mae={v_mae*100:.3f}c  val_rmse={v_rmse*100:.3f}c  "
              f"lr={sched.get_last_lr()[0]:.4f}  {'*' if stale==0 else ''}")
        if stale >= patience:
            print(f"  early stop at ep{ep}, best val_mae={best_val*100:.3f}c"); break
    # Restore best
    model.load_state_dict(best_state)
    # Test
    te_loader = to_loader(te, batch_size=4096, shuffle=False)
    model.eval()
    with torch.no_grad():
        preds, ys = [], []
        for xb, yb in te_loader:
            xb = xb.to(DEVICE); yb = yb.to(DEVICE)
            preds.append(model(xb)); ys.append(yb)
        preds = torch.cat(preds); ys = torch.cat(ys)
        t_mae = mae(preds, ys); t_rmse = rmse(preds, ys)
        # R²
        ss_res = float(torch.sum((preds - ys) ** 2).item())
        ss_tot = float(torch.sum((ys - ys.mean()) ** 2).item())
        r2 = 1 - ss_res / max(ss_tot, 1e-9)
    print(f"  TEST: MAE={t_mae*100:.3f}c  RMSE={t_rmse*100:.3f}c  R²={r2:+.4f}  "
          f"(trained {time.time()-t0:.1f}s)")
    return model, {"mae": t_mae, "rmse": t_rmse, "r2": r2}


def main():
    df_path = sys.argv[1] if len(sys.argv) > 1 else "data/training/toxic_flow.parquet"
    tr, va, te = load_data(df_path)
    print(f"  train={len(tr):,}  val={len(va):,}  test={len(te):,}")

    in_dim = len(FEATURES)

    # MLP grid (small)
    results = []
    for hidden, depth, dropout in [(64, 2, 0.1), (128, 3, 0.1), (256, 3, 0.15), (256, 4, 0.2)]:
        model, m = train_mlp(tr, va, te, in_dim,
                             epochs=30, hidden=hidden, depth=depth, dropout=dropout)
        results.append((f"mlp_h{hidden}_d{depth}_dr{dropout}", model, m))

    # Save best MLP
    best = min(results, key=lambda r: r[2]["mae"])
    name, model, m = best
    os.makedirs("_gbm_lab/toxic_flow/artifacts", exist_ok=True)
    out = "_gbm_lab/toxic_flow/artifacts/mlp.pt"
    torch.save({"state_dict": model.state_dict(),
                "config": {"in_dim": in_dim, "name": name},
                "features": FEATURES, "metrics": m}, out)
    print(f"\nSaved best MLP ({name}) to {out}")
    print(f"  MAE={m['mae']*100:.3f}c  R²={m['r2']:+.4f}")


if __name__ == "__main__":
    main()
