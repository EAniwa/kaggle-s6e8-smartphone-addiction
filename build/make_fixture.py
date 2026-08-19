"""Synthetic stand-in for the real competition data, matching the confirmed schema.

Used ONLY to execute the notebook end-to-end and prove every cell runs. Column
names, dtypes, ranges, categorical levels and missingness rates mirror the real
train.csv/test.csv. Row count is reduced so the test run is fast.
"""
import numpy as np, pandas as pd, pathlib, sys

N_TRAIN, N_TEST = int(sys.argv[1]) if len(sys.argv) > 1 else 12000, 5000
rng = np.random.default_rng(0)

MISS = {  # train missingness rates, from the real data
    "age": .0418, "daily_screen_time_hours": .1386, "social_media_hours": .1938,
    "gaming_hours": .1834, "work_study_hours": .0745, "sleep_hours": .0643,
    "notifications_per_day": .0978, "app_opens_per_day": .1167,
    "weekend_screen_time": .1621, "gender": .0420, "stress_level": .0798,
    "academic_work_impact": .0640,
}

def build(n, start_id):
    d = {}
    d["id"] = np.arange(start_id, start_id + n)
    d["age"] = rng.uniform(18, 35, n)
    d["daily_screen_time_hours"] = np.clip(rng.normal(7.64, 3.0, n), .5, 15)
    d["social_media_hours"] = np.clip(rng.normal(2.47, 1.6, n), 0, 8)
    d["gaming_hours"] = np.clip(rng.normal(1.46, 1.0, n), 0, 4)
    d["work_study_hours"] = np.clip(rng.normal(2.37, 1.5, n), 0, 6)
    d["sleep_hours"] = np.clip(rng.normal(6.80, 1.1, n), 4.5, 9)
    d["notifications_per_day"] = np.clip(rng.normal(145.9, 55, n), 20, 250)
    d["app_opens_per_day"] = np.clip(rng.normal(102.6, 40, n), 15, 180)
    d["weekend_screen_time"] = np.clip(rng.normal(9.48, 3.6, n), .51, 17.56)
    d["gender"] = rng.choice(["Male", "Female", "Other"], n)
    d["stress_level"] = rng.choice(["High", "Low", "Medium"], n)
    d["academic_work_impact"] = rng.choice(["Yes", "No"], n)
    df = pd.DataFrame(d)

    # Label rule mirroring the community finding on the source data, plus noise.
    hard1 = (df.daily_screen_time_hours > 8) | (df.social_media_hours > 4)
    hard0 = (df.daily_screen_time_hours <= 6) & (df.social_media_hours <= 4)
    p = np.where(hard1, .97, np.where(hard0, .06, .5))
    df["addicted_label"] = (rng.random(n) < p).astype(int)

    for col, rate in MISS.items():                    # punch holes AFTER labelling
        df.loc[rng.random(n) < rate, col] = np.nan
    return df

tr = build(N_TRAIN, 0)
te = build(N_TEST, N_TRAIN).drop(columns=["addicted_label"])

out = pathlib.Path("data"); out.mkdir(exist_ok=True)
tr.to_csv(out / "train.csv", index=False)
te.to_csv(out / "test.csv", index=False)
pd.DataFrame({"id": te.id, "addicted_label": 0.709424}).to_csv(
    out / "sample_submission.csv", index=False)
print(f"fixture: train {tr.shape}, test {te.shape}, pos rate {tr.addicted_label.mean():.4f}")
