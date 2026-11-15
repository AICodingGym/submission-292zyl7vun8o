from pathlib import Path

import lightgbm as lgb
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

print("Loading training and test data...", flush=True)

train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")
sample = pd.read_csv(DATA_DIR / "sample_submission.csv")

features = [
    "R", "C", "time_step", "u_in", "u_out",
    "u_in_cumsum",
    "u_in_lag_1", "u_in_lag_2",
    "u_in_lag_3", "u_in_lag_5",
    "delta_time", "u_in_time_cumsum"
]

def add_features(df):
    df = df.copy()
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

    return df

print("Creating features...", flush=True)
train = add_features(train)
test = add_features(test)

print("Training on all available training rows...", flush=True)

model = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    random_state=42,
    n_jobs=4,
    verbosity=-1
)

model.fit(train[features], train["pressure"])

print("Predicting test pressures...", flush=True)
predictions = model.predict(test[features])

assert len(predictions) == len(sample)
assert test["id"].reset_index(drop=True).equals(
    sample["id"].reset_index(drop=True)
)

submission = sample[["id"]].copy()
submission["pressure"] = predictions

output_path = BASE_DIR / "submission.csv"
submission.to_csv(output_path, index=False)

print(f"Saved {output_path}", flush=True)
print(f"Submission rows: {len(submission):,}", flush=True)
print(submission.head())