from pathlib import Path
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

BASE_DIR = Path(__file__).resolve().parent
torch.manual_seed(42)
np.random.seed(42)
torch.set_num_threads(4)

# Reuse the feature engineering and validation setup.
# The original script ends with model training, so execute
# only the portion before the model is created.
source = (BASE_DIR / "train_lightgbm_multiscale.py").read_text()
feature_code = source.split("model = lgb.LGBMRegressor(")[0]
namespace = {"__file__": str(BASE_DIR / "train_lightgbm_multiscale.py")}
exec(compile(feature_code, "train_lightgbm_multiscale.py", "exec"), namespace)

df = namespace["df"]
features = namespace["features"]
train_df = namespace["train_df"]
val_df = namespace["val_df"]

# Each breath must contain exactly 80 consecutive rows.
assert (df.groupby("breath_id").size() == 80).all()
assert not df[features].isna().any().any()

def make_lgb():
    return lgb.LGBMRegressor(
        n_estimators=3000,
        learning_rate=0.03,
        num_leaves=255,
        random_state=42,
        n_jobs=4,
        verbosity=-1,
    )

start = time.perf_counter()

# Generate out-of-fold predictions only on training breaths.
oof = np.empty(len(train_df), dtype=np.float32)
folds = GroupKFold(n_splits=3)

for fold, (fit_idx, hold_idx) in enumerate(
    folds.split(train_df, groups=train_df["breath_id"]), 1
):
    fit_rows = train_df.iloc[fit_idx]
    hold_rows = train_df.iloc[hold_idx]
    fit_insp = fit_rows[fit_rows["u_out"] == 0]

    print(f"Training LightGBM fold {fold}/3...", flush=True)
    tree = make_lgb()
    tree.fit(fit_insp[features], fit_insp["pressure"])
    oof[hold_idx] = tree.predict(hold_rows[features]).astype(np.float32)

# Train the final baseline on the complete training split.
print("Training validation baseline...", flush=True)
baseline = make_lgb()
fit_insp = train_df[train_df["u_out"] == 0]
baseline.fit(fit_insp[features], fit_insp["pressure"])
val_pred = baseline.predict(val_df[features]).astype(np.float32)

val_mask = val_df["u_out"].to_numpy() == 0
baseline_mae = np.abs(
    val_df["pressure"].to_numpy()[val_mask] - val_pred[val_mask]
).mean()
print(f"Baseline validation MAE: {baseline_mae:.6f}", flush=True)

# Standardize the input features using training breaths only.
train_mean = train_df[features].mean().to_numpy(dtype=np.float32)
train_std = train_df[features].std().to_numpy(dtype=np.float32)
train_std = np.maximum(train_std, 1e-5)

def make_sequences(frame, tree_predictions):
    # The data are ordered in complete 80-step breaths.
    x = (
        (frame[features].to_numpy(dtype=np.float32) - train_mean)
        / train_std
    )
    x = np.clip(x, -10, 10)
    p = tree_predictions.reshape(-1, 1)
    x = np.concatenate([x, p / 30.0], axis=1)

    y = frame["pressure"].to_numpy(dtype=np.float32)
    mask = (frame["u_out"].to_numpy() == 0).astype(np.float32)

    return (
        torch.from_numpy(x.reshape(-1, 80, x.shape[1])),
        torch.from_numpy(y.reshape(-1, 80)),
        torch.from_numpy(mask.reshape(-1, 80)),
        torch.from_numpy(tree_predictions.reshape(-1, 80)),
    )

train_data = make_sequences(train_df, oof)
val_data = make_sequences(val_df, val_pred)

loader = DataLoader(
    TensorDataset(*train_data),
    batch_size=128,
    shuffle=True,
)

val_loader = DataLoader(
    TensorDataset(*val_data),
    batch_size=128,
    shuffle=False,
)

class ResidualGRU(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1,
        )
        self.head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x, baseline_pressure):
        hidden, _ = self.gru(x)
        correction = self.head(hidden).squeeze(-1)
        return baseline_pressure + correction

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"Neural network device: {device}", flush=True)

net = ResidualGRU(train_data[0].shape[-1]).to(device)
optimizer = torch.optim.AdamW(net.parameters(), lr=0.001)

best_mae = float("inf")
best_state = None

# Start with two epochs to test training speed.
for epoch in range(1, 3):
    net.train()
    epoch_start = time.perf_counter()

    for batch_idx, (x, y, mask, base) in enumerate(loader, 1):
        x = x.to(device)
        y = y.to(device)
        mask = mask.to(device)
        base = base.to(device)

        optimizer.zero_grad()
        prediction = net(x, base)

        loss = (
            ((prediction - y).abs() * mask).sum()
            / mask.sum().clamp(min=1)
        )

        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()

        if batch_idx % 50 == 0:
            print(
                f"Epoch {epoch} | Batch {batch_idx}/{len(loader)} "
                f"| Loss {loss.item():.5f}",
                flush=True,
            )

    net.eval()
    total_error = 0.0
    total_points = 0.0

    with torch.no_grad():
        for x, y, mask, base in val_loader:
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)
            base = base.to(device)

            prediction = net(x, base)
            total_error += (
                (prediction - y).abs() * mask
            ).sum().item()
            total_points += mask.sum().item()

    mae = total_error / total_points

    print(
        f"Epoch {epoch:02d}: hybrid MAE = {mae:.6f} "
        f"| Epoch time: {time.perf_counter() - epoch_start:.1f}s",
        flush=True,
    )

    if mae < best_mae:
        best_mae = mae
        best_state = {
            key: value.detach().cpu().clone()
            for key, value in net.state_dict().items()
        }

torch.save(
    {
        "model_state": best_state,
        "features": features,
        "train_mean": train_mean,
        "train_std": train_std,
        "best_validation_mae": best_mae,
    },
    BASE_DIR / "hybrid_gru_checkpoint.pt",
)

print(f"Best hybrid validation MAE: {best_mae:.6f}")
print(f"Baseline validation MAE: {baseline_mae:.6f}")
print(f"Total runtime: {time.perf_counter() - start:.1f} seconds")
print("Saved hybrid_gru_checkpoint.pt")