from pathlib import Path
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

BASE_DIR = Path(__file__).resolve().parent
matches = list((BASE_DIR / "data").rglob("train.csv"))

if len(matches) != 1:
    raise FileNotFoundError(
        f"Expected one train.csv under {BASE_DIR / 'data'}, "
        f"found {len(matches)}"
    )

train_path = matches[0]
print(f"Loading: {train_path}", flush=True)

df = pd.read_csv(train_path)

breaths = df.groupby("breath_id", sort=False)

df["u_in_cumsum"] = breaths["u_in"].cumsum()

df["delta_time"] = (
    breaths["time_step"].diff()
    .fillna(0)
    .clip(lower=0)
)

df["u_in_time_cumsum"] = (
    (df["u_in"] * df["delta_time"])
    .groupby(df["breath_id"], sort=False)
    .cumsum()
)

for lag in [1, 2, 3, 5]:
    df[f"u_in_lag_{lag}"] = (
        breaths["u_in"].shift(lag).fillna(0)
    )

features = [
    "delta_time",
    "u_in_time_cumsum",
    "R",
    "C",
    "time_step",
    "u_in",
    "u_out",
    "u_in_cumsum",
    "u_in_lag_1",
    "u_in_lag_2",
    "u_in_lag_3",
    "u_in_lag_5"
]

target = "pressure"

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, val_idx = next(
    splitter.split(df, groups=df["breath_id"])
)

train_df = df.iloc[train_idx]
val_df = df.iloc[val_idx]

assert set(train_df["breath_id"]).isdisjoint(
    set(val_df["breath_id"])
)

print(f"Training rows: {len(train_df):,}", flush=True)
print(f"Validation rows: {len(val_df):,}", flush=True)

model = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    random_state=42,
    n_jobs=4,
    verbosity=-1
)

start = time.perf_counter()

model.fit(
    train_df[features],
    train_df[target]
)

predictions = model.predict(val_df[features])
runtime = time.perf_counter() - start

mask = val_df["u_out"].to_numpy() == 0
actual = val_df[target].to_numpy()

mae = float(
    np.mean(np.abs(actual[mask] - predictions[mask]))
)

print(f"Validation inspiratory MAE: {mae:.6f}")
print(f"Training + prediction time: {runtime:.1f} seconds")

result = {
    "experiment": "lightgbm_time_weighted_input",
    "features": features,
    "validation_fraction": 0.20,
    "random_state": 42,
    "validation_inspiratory_mae": mae,
    "runtime_seconds": runtime
}

with open(BASE_DIR / "time_weighted_results.json", "w") as f:
    json.dump(result, f, indent=2)

print("Saved time_weighted_results.json")
