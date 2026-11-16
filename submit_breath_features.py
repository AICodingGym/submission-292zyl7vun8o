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
    "breath_u_in_max",
    "breath_u_in_mean",
    "breath_u_in_std",
    "breath_u_in_sum",
    "u_in_R",
    "volume_C",
    "RC",
    "u_in_diff",
    "u_in_rate",
    "delta_time",
    "u_in_time_cumsum",
    "u_in_lead_1",
    "u_in_lead_2",
    "u_in_lead_3",
    "u_in_rolling_mean_3",
    "u_in_rolling_mean_5",
    "u_in_rolling_mean_10",
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

    for lead in [1, 2, 3]:
        df[f"u_in_lead_{lead}"] = (
            breaths["u_in"].shift(-lead).fillna(0)
        )

    for window in [3, 5, 10]:
        df[f"u_in_rolling_mean_{window}"] = (
            df.groupby("breath_id", sort=False)["u_in"]
            .transform(
                lambda values: values.rolling(
                    window=window, min_periods=1
                ).mean()
            )
        )

    df["u_in_R"] = df["u_in"] * df["R"]
    df["RC"] = df["R"] * df["C"]

    df["u_in_diff"] = breaths["u_in"].diff().fillna(0)

    df["u_in_rate"] = (
        df["u_in_diff"]
        / breaths["time_step"].diff().clip(lower=1e-4).fillna(1e-4)
    )

    df["volume_C"] = df["u_in_time_cumsum"] / df["C"]

    # Breath-level airflow statistics
    breath_airflow = df.groupby("breath_id")["u_in"]

    df["breath_u_in_max"] = breath_airflow.transform("max")
    df["breath_u_in_mean"] = breath_airflow.transform("mean")
    df["breath_u_in_std"] = breath_airflow.transform("std").fillna(0)
    df["breath_u_in_sum"] = breath_airflow.transform("sum")
    
    return df

print("Creating features...", flush=True)
train = add_features(train)
test = add_features(test)

print("Training on all available training rows...", flush=True)

model = lgb.LGBMRegressor(
    n_estimators=3000,
    learning_rate=0.03,
    num_leaves=255,
    max_depth=-1,
    random_state=42,
    n_jobs=4,
    verbosity=-1
)

inspiratory_train = train[train["u_out"] == 0]

model.fit(
    inspiratory_train[features],
    inspiratory_train["pressure"]
)

print("Predicting test pressures...", flush=True)
predictions = model.predict(test[features])

assert len(predictions) == len(sample)
assert test["id"].reset_index(drop=True).equals(
    sample["id"].reset_index(drop=True)
)

submission = sample[["id"]].copy()
submission["pressure"] = predictions

output_path = BASE_DIR / "submission_breath_features.csv"
submission.to_csv(output_path, index=False)

print(f"Saved {output_path}", flush=True)
print(f"Submission rows: {len(submission):,}", flush=True)
print(submission.head())