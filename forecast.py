"""Hourly job: pull the latest KABQ observations, archive them, issue a new 24-h forecast."""
from datetime import timedelta
import os
import sys

import pandas as pd

import tamb_core as tc


def main():
    if not (tc.MODEL_DIR / "model.pt").exists():
        sys.exit("models/model.pt missing - run the 'Train model' workflow first")
    now = tc.utcnow()
    hourly = tc.fetch_hourly(now - timedelta(days=7), now + timedelta(days=1))
    hourly = hourly[hourly.time_utc <= pd.Timestamp(now)]
    tc.upsert_obs(hourly.dropna(subset=["T_amb_C"]))

    model, sc, ck = tc.load_model()
    f = tc.features(hourly)
    origin = f["T"].last_valid_index()
    if origin is None:
        sys.exit("no recent observations")
    gap = f["T"].loc[:origin].iloc[-ck["lookback"]:].isna().sum()
    if gap > 6:
        print(f"skip: {gap} missing hours in the last {ck['lookback']} h")
        return
    done = tc.read_forecasts(origin.to_pydatetime())
    if len(done) and ((done.issued_utc == origin) & (done.source == "live")).any():
        print(f"forecast for {origin} already issued")
        return
    rows = tc.forecast_rows(model, sc, ck, f, pd.DatetimeIndex([origin]), "live")
    tc.write_forecasts(rows)
    r = rows[rows.model == ck["name"]]
    print(f"issued {origin} UTC: next 24 h {r.T_pred_C.min():.1f} .. {r.T_pred_C.max():.1f} C")
    if os.environ.get("GITHUB_OUTPUT"):              # tell the workflow a new forecast exists
        with open(os.environ["GITHUB_OUTPUT"], "a") as fh:
            fh.write("new=true\n")


if __name__ == "__main__":
    main()
