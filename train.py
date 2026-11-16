"""
Bidirectional LSTM for Ventilator Pressure Prediction (DSC 102 PA3).

Treats each breath as one sequence of 80 time steps and predicts pressure at
every step. Fixes the usual reasons an RNN "gets stuck":
  * input shaped (breaths, 80, features), not one row at a time
  * features scaled with RobustScaler
  * masked L1 loss on inspiratory rows only (u_out == 0), matching the metric
  * gradient clipping + learning rate halved when validation stops improving
  * long training (hundreds of epochs), with checkpoints so it can resume

Idea credit: architecture and features follow common public approaches to the
original Kaggle "Google Brain - Ventilator Pressure Prediction" competition
(bidirectional LSTM on whole breaths, u_in cumsum/lag features, R/C one-hot,
snapping predictions to the discrete pressure grid).

Usage (from the ventilator-pressure-prediction directory):
  python -u train.py --smoke              # 2-minute bug check on a small subset
  python -u train.py                      # full run
  python -u train.py --time-limit-hours 4 # stop early, still writes submission
Re-running the same command resumes from ckpt.pt.
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import RobustScaler

p = argparse.ArgumentParser()
p.add_argument("--data-dir", default="data")
p.add_argument("--epochs", type=int, default=250)
p.add_argument("--batch", type=int, default=512)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--hidden", type=int, default=256)
p.add_argument("--layers", type=int, default=4)
p.add_argument("--dropout", type=float, default=0.0)
p.add_argument("--val-frac", type=float, default=0.1)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--time-limit-hours", type=float, default=5.5)
p.add_argument("--out", default="submission.csv")
p.add_argument("--ckpt", default="ckpt.pt")
p.add_argument("--best", default="best.pt")
p.add_argument("--log", default="trials.csv")
p.add_argument("--smoke", action="store_true", help="tiny run to check for bugs")
args = p.parse_args()

if args.smoke:
    args.epochs = 3
    args.ckpt, args.best, args.log = "smoke_ckpt.pt", "smoke_best.pt", "smoke_trials.csv"
    args.out = "smoke_submission.csv"

torch.manual_seed(args.seed)
np.random.seed(args.seed)
dev = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"device: {dev}", flush=True)
T0 = time.time()
STEPS = 80


# ---------------------------------------------------------------- features
def add_features(df):
    df = df.sort_values(["breath_id", "time_step"]).reset_index(drop=True)
    g = df.groupby("breath_id")
    df["u_in_cumsum"] = g["u_in"].cumsum()
    df["time_diff"] = g["time_step"].diff().fillna(0)
    df["area"] = df["time_diff"] * df["u_in"]
    df["area"] = df.groupby("breath_id")["area"].cumsum()
    for k in (1, 2, 3, 4):
        df[f"u_in_lag{k}"] = g["u_in"].shift(k).fillna(0)
        df[f"u_in_lead{k}"] = g["u_in"].shift(-k).fillna(0)
        df[f"u_out_lag{k}"] = g["u_out"].shift(k).fillna(0)
    df["u_in_diff1"] = df["u_in"] - df["u_in_lag1"]
    df["u_in_diff2"] = df["u_in"] - df["u_in_lag2"]
    df["u_in_mean"] = g["u_in"].transform("mean")
    df["u_in_max"] = g["u_in"].transform("max")
    df["u_in_minus_mean"] = df["u_in"] - df["u_in_mean"]
    df["step"] = g.cumcount()
    df["R_C"] = df["R"].astype(str) + "_" + df["C"].astype(str)
    df = pd.get_dummies(df, columns=["R", "C", "R_C"], dtype=float)
    return df


print("loading data...", flush=True)
train = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
test = pd.read_csv(os.path.join(args.data_dir, "test.csv"))
if args.smoke:
    keep_tr = train["breath_id"].unique()[:2000]
    keep_te = test["breath_id"].unique()[:500]
    train = train[train["breath_id"].isin(keep_tr)]
    test = test[test["breath_id"].isin(keep_te)]

pressure_grid = np.sort(train["pressure"].unique())
train = add_features(train)
test = add_features(test)
assert len(train) % STEPS == 0 and len(test) % STEPS == 0, "breaths must have 80 rows"

feat_cols = [c for c in train.columns if c not in ("id", "breath_id", "pressure")]
test = test.reindex(columns=list(test.columns.union(feat_cols)), fill_value=0.0)
print(f"{len(feat_cols)} features: {feat_cols}", flush=True)

y_all = train["pressure"].to_numpy(np.float32).reshape(-1, STEPS)
m_all = (train["u_out"].to_numpy() == 0).astype(np.float32).reshape(-1, STEPS)
test_ids = test["id"].to_numpy()

scaler = RobustScaler()
X_all = scaler.fit_transform(train[feat_cols].to_numpy(np.float32))
X_test = scaler.transform(test[feat_cols].to_numpy(np.float32))
X_all = X_all.astype(np.float32).reshape(-1, STEPS, len(feat_cols))
X_test = X_test.astype(np.float32).reshape(-1, STEPS, len(feat_cols))
del train, test

# Hold out whole breaths (fixed seed, so the split is the same every run).
n = len(X_all)
perm = np.random.RandomState(args.seed).permutation(n)
n_val = int(n * args.val_frac)
va_idx, tr_idx = perm[:n_val], perm[n_val:]

to_t = lambda a: torch.from_numpy(a).to(dev)
Xtr, ytr, mtr = to_t(X_all[tr_idx]), to_t(y_all[tr_idx]), to_t(m_all[tr_idx])
Xva, yva, mva = to_t(X_all[va_idx]), to_t(y_all[va_idx]), to_t(m_all[va_idx])
Xte = to_t(X_test)
del X_all, X_test
print(f"train breaths {len(Xtr)}, val breaths {len(Xva)}, test breaths {len(Xte)} "
      f"| prep {time.time() - T0:.0f}s", flush=True)


# ---------------------------------------------------------------- model
class Net(nn.Module):
    def __init__(self, n_feat, h, layers, dropout):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(n_feat, h), nn.LayerNorm(h), nn.GELU())
        self.lstm = nn.LSTM(h, h, num_layers=layers, batch_first=True,
                            bidirectional=True, dropout=dropout if layers > 1 else 0)
        self.head = nn.Sequential(nn.Linear(2 * h, h), nn.GELU(), nn.Linear(h, 1))

    def forward(self, x):
        z, _ = self.lstm(self.inp(x))
        return self.head(z).squeeze(-1)


def masked_l1(pred, y, m):
    return ((pred - y).abs() * m).sum() / m.sum()


@torch.no_grad()
def predict(model, X, bs=64):
    model.eval()
    return torch.cat([model(X[i:i + bs]) for i in range(0, len(X), bs)]).cpu().numpy()


def snap(x):
    i = np.clip(np.searchsorted(pressure_grid, x), 1, len(pressure_grid) - 1)
    lo, hi = pressure_grid[i - 1], pressure_grid[i]
    return np.where(x - lo < hi - x, lo, hi)


model = Net(len(feat_cols), args.hidden, args.layers, args.dropout).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=10)
start_epoch, best_mae = 0, float("inf")

if os.path.exists(args.ckpt):
    ck = torch.load(args.ckpt, map_location=dev)
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["opt"])
    sched.load_state_dict(ck["sched"])
    start_epoch, best_mae = ck["epoch"] + 1, ck["best_mae"]
    print(f"resumed from epoch {start_epoch}, best val MAE {best_mae:.5f}", flush=True)

if not os.path.exists(args.log):
    with open(args.log, "w") as f:
        f.write("epoch,train_mae,val_mae,val_mae_snapped,lr,epoch_sec,total_min\n")


def write_submission():
    raw = predict(model, Xte).reshape(-1)
    vraw = predict(model, Xva)
    mask = mva.cpu().numpy() > 0
    yv = yva.cpu().numpy()
    use_snap = np.abs(snap(vraw) - yv)[mask].mean() < np.abs(vraw - yv)[mask].mean()
    pred = snap(raw) if use_snap else raw
    pd.DataFrame({"id": test_ids, "pressure": pred}).sort_values("id").to_csv(args.out, index=False)
    return use_snap


limit_sec = args.time_limit_hours * 3600
for epoch in range(start_epoch, args.epochs):
    t = time.time()
    model.train()
    order = torch.randperm(len(Xtr), device=dev)
    tot, cnt = 0.0, 0
    for i in range(0, len(order), args.batch):
        b = order[i:i + args.batch]
        loss = masked_l1(model(Xtr[b]), ytr[b], mtr[b])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tot += loss.item() * len(b)
        cnt += len(b)

    vp = predict(model, Xva)
    mask = mva.cpu().numpy() > 0
    yv = yva.cpu().numpy()
    val_mae = float(np.abs(vp - yv)[mask].mean())
    val_snap = float(np.abs(snap(vp) - yv)[mask].mean())
    sched.step(val_mae)
    lr = opt.param_groups[0]["lr"]

    note = ""
    if val_mae < best_mae:
        best_mae = val_mae
        torch.save(model.state_dict(), args.best)
        snapped = write_submission()  # always keep a submission from the best model
        note = f"  * new best, wrote {args.out}" + (" (snapped)" if snapped else "")

    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "sched": sched.state_dict(), "epoch": epoch, "best_mae": best_mae}, args.ckpt)
    el = time.time() - t
    total = (time.time() - T0) / 60
    with open(args.log, "a") as f:
        f.write(f"{epoch},{tot / cnt:.5f},{val_mae:.5f},{val_snap:.5f},{lr:.2e},{el:.1f},{total:.1f}\n")
    print(f"epoch {epoch:3d} train {tot / cnt:.4f} val {val_mae:.4f} snap {val_snap:.4f} "
          f"lr {lr:.1e} {el:.0f}s total {total:.0f}m{note}", flush=True)

    if time.time() - T0 > limit_sec:
        print("time limit reached, stopping", flush=True)
        break

print(f"done. best val MAE {best_mae:.5f}; submission from best epoch is in {args.out}", flush=True)
