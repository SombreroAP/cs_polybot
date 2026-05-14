"""
1D CNN toxic-flow predictor on 5090.

Takes a short sequence of recent mid prices (last 60 samples = 60s) plus
the v2 tabular features and predicts |mid drift in next 60s|.

Architecture:
  Input: (batch, 60) raw mid sequence + (batch, 26) v2 tabular features
  CNN branch: 3-layer 1D conv + global avg pool → 64-d
  MLP branch: 26 → 64 → 64 (relu + dropout)
  Fused: concat 128 → 128 → 1

The 5090 trains this in ~minutes. 9800X3D handles the dataset prep
(building 60-step sequences from the parquet).
"""
import os, sys, time, pickle, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}  cuda={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"CPU threads available: {os.cpu_count()}")
torch.set_num_threads(os.cpu_count() or 8)


V2_TAB_FEATURES = [
    "spread",
    "past_drift_5s", "past_drift_10s", "past_drift_30s",
    "past_drift_60s", "past_drift_300s",
    "abs_past_drift_5s", "abs_past_drift_10s", "abs_past_drift_30s",
    "abs_past_drift_60s", "abs_past_drift_300s",
    "velocity_1s", "accel_5_10",
    "vol_30s", "vol_60s", "vol_300s",
    "spread_change_30s",
    "tod_sin", "tod_cos", "dow_sin", "dow_cos",
    "comp_drift_30s", "abs_comp_drift_30s", "comp_mid",
    "comp_sum", "comp_sum_dev",
]
SEQ_LEN = 60


def build_sequences(df):
    """For each row, build a 60-step mid sequence using the past_drift columns.

    We don't have raw per-second mids in the parquet, but we DO have past
    drifts at 5/10/30/60/300s and current mid. Reconstruct an
    approximate sequence by interpolating: mid_at(t-k) ≈ mid - past_drift_K
    where K is the nearest known window.

    Simpler approach: just use the multi-window past-drift values as the
    sequence (5 known points × 12 = 60 length). But that's lossy.

    For this experiment we use a SYNTHETIC 60-step sequence derived from
    the available past drifts, padded with mid for missing intermediate
    points. The CNN should still learn patterns from this lossy view.
    """
    # Build a sequence of relative mids: mid_at(t-k) - mid_now for k in 0..59
    # We know exact values at 0, 1, 5, 10, 30, 60s. Interpolate elsewhere.
    n = len(df)
    seqs = np.zeros((n, SEQ_LEN), dtype=np.float32)
    # known offsets and corresponding feature names
    known = [(0, None), (1, "velocity_1s"), (5, None), (10, None),
             (30, "past_drift_30s"), (60, "past_drift_60s")]
    # Fill exact: for each row, compute mid_at(t-k) - mid_now = -past_drift_k
    # at k=1, velocity_1s = mid - mid_at(t-1) → mid_at(t-1) - mid = -velocity_1s
    past_30 = df["past_drift_30s"].values
    past_60 = df["past_drift_60s"].values
    v1 = df["velocity_1s"].values
    # crude approximation: linearly interp from 0 to k=30s using past_30
    for i in range(n):
        # mid_at(t-k) - mid_now for k=0..59
        # known: k=0 → 0; k=1 → -v1[i]; k=30 → -past_30[i]; k=60 → -past_60[i] (but k≤59)
        for k in range(SEQ_LEN):
            if k == 0: seqs[i, k] = 0.0
            elif k == 1: seqs[i, k] = -v1[i]
            elif k <= 30: seqs[i, k] = -past_30[i] * (k / 30.0)
            else: seqs[i, k] = -past_60[i] * (k / 60.0)
    return seqs


class FusedDS(Dataset):
    def __init__(self, df):
        self.X_tab = df[V2_TAB_FEATURES].values.astype(np.float32)
        self.seq = build_sequences(df)
        self.y = df["target_abs_drift_60s"].values.astype(np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (torch.from_numpy(self.X_tab[i]),
                torch.from_numpy(self.seq[i]).unsqueeze(0),  # (1, 60) → channel dim
                torch.tensor(self.y[i], dtype=torch.float32))


class FusedCNN(nn.Module):
    def __init__(self, tab_dim, seq_len, dropout=0.15):
        super().__init__()
        # CNN branch on 1×seq_len sequence
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # → (B, 64, 1)
        )
        # MLP branch on tabular features
        self.mlp = nn.Sequential(
            nn.Linear(tab_dim, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 64), nn.ReLU(),
        )
        # Fused head
        self.head = nn.Sequential(
            nn.Linear(64 + 64, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x_tab, x_seq):
        c = self.conv(x_seq).squeeze(-1)
        m = self.mlp(x_tab)
        return self.head(torch.cat([c, m], dim=1)).squeeze(-1)


def metrics_t(pred, target):
    err = pred - target
    mae = float(torch.mean(torch.abs(err)).item())
    rmse = float(torch.sqrt(torch.mean(err ** 2)).item())
    ss_res = float(torch.sum(err ** 2).item())
    ss_tot = float(torch.sum((target - target.mean()) ** 2).item())
    return mae, rmse, 1 - ss_res / max(ss_tot, 1e-9)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "data/training/toxic_flow_v2.parquet"
    print(f"Loading {path}…")
    df = pd.read_parquet(path).sort_values("ts").reset_index(drop=True)
    n = len(df); n_tr = int(n * 0.7); n_va = int(n * 0.15)
    tr = df.iloc[:n_tr]; va = df.iloc[n_tr:n_tr+n_va]; te = df.iloc[n_tr+n_va:]
    print(f"  train={len(tr):,}  val={len(va):,}  test={len(te):,}")

    print("Building sequences (CPU-parallel via numpy)…")
    t0 = time.time()
    tr_ds = FusedDS(tr); va_ds = FusedDS(va); te_ds = FusedDS(te)
    print(f"  done in {time.time()-t0:.1f}s")

    bs = 4096
    tr_l = DataLoader(tr_ds, batch_size=bs, shuffle=True, num_workers=2)
    va_l = DataLoader(va_ds, batch_size=bs, shuffle=False, num_workers=2)
    te_l = DataLoader(te_ds, batch_size=bs, shuffle=False, num_workers=2)

    model = FusedCNN(len(V2_TAB_FEATURES), SEQ_LEN).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=40)
    loss_fn = nn.SmoothL1Loss()

    best_val = float("inf"); best_state = None; patience = 6; stale = 0
    t0 = time.time()
    for ep in range(40):
        model.train()
        for x_tab, x_seq, y in tr_l:
            x_tab = x_tab.to(DEVICE); x_seq = x_seq.to(DEVICE); y = y.to(DEVICE)
            opt.zero_grad()
            p = model(x_tab, x_seq)
            loss = loss_fn(p, y)
            loss.backward(); opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            ps, ys = [], []
            for x_tab, x_seq, y in va_l:
                x_tab = x_tab.to(DEVICE); x_seq = x_seq.to(DEVICE); y = y.to(DEVICE)
                ps.append(model(x_tab, x_seq)); ys.append(y)
            ps = torch.cat(ps); ys = torch.cat(ys)
            v_mae, v_rmse, v_r2 = metrics_t(ps, ys)

        if v_mae < best_val:
            best_val = v_mae; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}; stale = 0
        else:
            stale += 1
        flag = " *" if stale == 0 else ""
        print(f"  ep{ep:>2}: val_mae={v_mae*100:.3f}c  rmse={v_rmse*100:.3f}c  "
              f"R²={v_r2:+.4f}  lr={sched.get_last_lr()[0]:.4f}{flag}")
        if stale >= patience:
            print(f"  early stop at ep{ep}, best val_mae={best_val*100:.3f}c"); break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        ps, ys = [], []
        for x_tab, x_seq, y in te_l:
            x_tab = x_tab.to(DEVICE); x_seq = x_seq.to(DEVICE); y = y.to(DEVICE)
            ps.append(model(x_tab, x_seq)); ys.append(y)
        ps = torch.cat(ps); ys = torch.cat(ys)
        t_mae, t_rmse, t_r2 = metrics_t(ps, ys)
    print(f"\nTEST: MAE={t_mae*100:.3f}c  RMSE={t_rmse*100:.3f}c  R²={t_r2:+.4f}  "
          f"(trained {time.time()-t0:.1f}s)")

    out = "_gbm_lab/toxic_flow/artifacts/cnn_fused.pt"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "config": {"tab_dim": len(V2_TAB_FEATURES), "seq_len": SEQ_LEN},
                "tab_features": V2_TAB_FEATURES,
                "metrics": {"mae": t_mae, "rmse": t_rmse, "r2": t_r2}}, out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
