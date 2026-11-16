from pathlib import Path
import json
import random
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print(f"Using device: {DEVICE}", flush=True)

EPOCHS = 15
BATCH_SIZE = 256
LEARNING_RATE = 0.001
PATIENCE = 4

print("Loading training data...", flush=True)
df = pd.read_csv(DATA_DIR / "train.csv")

assert len(df) % 80 == 0
assert (df.groupby("breath_id").size() == 80).all()

df["u_in_cumsum"] = df.groupby("breath_id")["u_in"].cumsum()

breaths = df.groupby("breath_id", sort=False)

df["delta_time"] = (
    breaths["time_step"].diff().fillna(0).clip(lower=0)
)

df["u_in_diff"] = breaths["u_in"].diff().fillna(0)

df["u_in_time_cumsum"] = (
    (df["u_in"] * df["delta_time"])
    .groupby(df["breath_id"], sort=False)
    .cumsum()
)

features = [
    "R", "C", "time_step", "u_in", "u_out",
    "u_in_cumsum", "delta_time", "u_in_diff",
    "u_in_time_cumsum"
]

df["R"] = df["R"] / 50.0
df["C"] = df["C"] / 50.0
df["time_step"] = df["time_step"] / 3.0
df["u_in"] = df["u_in"] / 100.0
df["u_in_cumsum"] = df["u_in_cumsum"] / 4000.0
df["delta_time"] = df["delta_time"] / 0.1
df["u_in_diff"] = df["u_in_diff"] / 100.0
df["u_in_time_cumsum"] = df["u_in_time_cumsum"] / 100.0

X = df[features].to_numpy(dtype=np.float32).reshape(-1, 80, 9)
y = df["pressure"].to_numpy(dtype=np.float32).reshape(-1, 80)
u_out = df["u_out"].to_numpy(dtype=np.float32).reshape(-1, 80)

breath_ids = df["breath_id"].to_numpy().reshape(-1, 80)[:, 0]

train_idx, val_idx = train_test_split(
    np.arange(len(breath_ids)),
    test_size=0.20,
    random_state=SEED
)

assert set(breath_ids[train_idx]).isdisjoint(
    set(breath_ids[val_idx])
)

train_data = TensorDataset(
    torch.from_numpy(X[train_idx]),
    torch.from_numpy(y[train_idx]),
    torch.from_numpy(u_out[train_idx])
)

val_data = TensorDataset(
    torch.from_numpy(X[val_idx]),
    torch.from_numpy(y[val_idx]),
    torch.from_numpy(u_out[val_idx])
)

train_loader = DataLoader(
    train_data, batch_size=BATCH_SIZE, shuffle=True
)

val_loader = DataLoader(
    val_data, batch_size=BATCH_SIZE, shuffle=False
)

class VentilatorLSTM(nn.Module):
    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=9,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )

        self.head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x, _ = self.lstm(x)
        return self.head(x).squeeze(-1)


model = VentilatorLSTM().to(DEVICE)
optimizer = torch.optim.Adam(
    model.parameters(), lr=LEARNING_RATE
)

best_mae = float("inf")
epochs_without_improvement = 0

start = time.perf_counter()

print("Starting LSTM training...", flush=True)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for xb, yb, maskb in train_loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)
        maskb = maskb.to(DEVICE)

        optimizer.zero_grad()

        pred = model(xb)

        mask = 1.0 - maskb
        loss = (
            (pred - yb).abs() * mask
        ).sum() / mask.sum().clamp(min=1)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0
        )
        optimizer.step()

        total_loss += loss.item()

    model.eval()
    total_error = 0.0
    total_count = 0.0

    with torch.no_grad():
        for xb, yb, maskb in val_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            maskb = maskb.to(DEVICE)

            pred = model(xb)
            mask = 1.0 - maskb

            total_error += (
                (pred - yb).abs() * mask
            ).sum().item()
            total_count += mask.sum().item()

    val_mae = total_error / total_count

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train loss: {total_loss / len(train_loader):.5f} | "
        f"Validation MAE: {val_mae:.5f}",
        flush=True
    )

    if val_mae < best_mae:
        best_mae = val_mae
        epochs_without_improvement = 0

        torch.save(
            model.state_dict(),
            BASE_DIR / "best_lstm_advanced.pt"
        )
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:
        print("Early stopping triggered.")
        break

runtime = time.perf_counter() - start

results = {
    "experiment": "bidirectional_lstm",
    "best_validation_mae": best_mae,
    "runtime_seconds": runtime,
    "device": str(DEVICE),
    "epochs_completed": epoch + 1
}

with open(BASE_DIR / "lstm_advanced_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"Best validation MAE: {best_mae:.6f}")
print(f"Total runtime: {runtime:.1f} seconds")
print("Saved best_lstm_features.pt and lstm_features_results.json")