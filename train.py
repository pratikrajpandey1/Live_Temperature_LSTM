"""
Train the 24-h model on KABQ history, save models/model.pt, and write an out-of-sample
hindcast for the most recent HINDCAST_DAYS (held out from training) so the website has
data immediately.

  python train.py                 # defaults
  python train.py --start 2016-01-01 --hindcast-days 14 --models convlstm lstm_multi
"""
import argparse, json, time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import tamb_core as tc


def fit(name, Xtr, ytr, Xva, yva, a):
    torch.manual_seed(0)
    m = tc.build(name)
    opt = torch.optim.Adam(m.parameters(), lr=a.lr)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    lf = nn.MSELoss()
    Xt, yt = torch.tensor(Xtr), torch.tensor(ytr)
    Xv, yv = torch.tensor(Xva), torch.tensor(yva)
    best, state, wait = np.inf, None, 0
    for ep in range(1, a.epochs + 1):
        t0 = time.time()
        m.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(perm), a.batch):
            b = perm[i:i + a.batch]
            opt.zero_grad()
            loss = lf(m(Xt[b]), yt[b])
            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
        m.eval()
        with torch.no_grad():
            vl = sum(lf(m(Xv[i:i + 4096]), yv[i:i + 4096]).item() * len(Xv[i:i + 4096])
                     for i in range(0, len(Xv), 4096)) / len(Xv)
        sch.step(vl)
        print(f"  [{name}] ep {ep:3d} val {vl:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if vl < best - 1e-5:
            best, wait, state = vl, 0, {k: v.clone() for k, v in m.state_dict().items()}
        else:
            wait += 1
            if wait >= a.patience:
                break
    m.load_state_dict(state)
    return m, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--hindcast-days", type=int, default=14)
    ap.add_argument("--val-days", type=int, default=365)
    ap.add_argument("--models", nargs="+", default=["convlstm", "lstm_multi"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    a = ap.parse_args()

    now = tc.utcnow().replace(minute=0, second=0, microsecond=0)
    start = datetime.fromisoformat(a.start).replace(tzinfo=timezone.utc)
    print("downloading history ...")
    hourly = tc.fetch_hourly(start, now + timedelta(days=1))
    tc.upsert_obs(hourly[hourly.time_utc >= now - timedelta(days=a.hindcast_days + 40)])
    f = tc.features(hourly)
    print(f"history {f.index.min()} -> {f.index.max()}  ({len(f):,} h)")

    cut = pd.Timestamp(now - timedelta(days=a.hindcast_days))
    vcut = cut - pd.Timedelta(days=a.val_days)
    X, y, t = tc.windows(f[f.index < cut], 72)   # all targets before the hindcast window
    tr, va = t < vcut, (t >= vcut) & (t < cut)
    print(f"windows train {tr.sum():,}  val {va.sum():,}")
    sc = tc.Scaler().fit(X[tr])
    Xs, ys = sc.x(X), sc.y(y)

    results = {}
    for name in a.models:
        m, vl = fit(name, Xs[tr], ys[tr], Xs[va], ys[va], a)
        p = sc.y_inv(tc.predict(m, Xs[va]))
        rmse = np.sqrt(((p - y[va]) ** 2).mean(0))
        results[name] = (m, float(np.sqrt(((p - y[va]) ** 2).mean())), rmse)
        print(f"{name}: val RMSE {results[name][1]:.3f} C")
    pers = X[va][:, -24:, 0]
    pers_rmse = np.sqrt(((pers - y[va]) ** 2).mean(0))
    best = min(results, key=lambda k: results[k][1])
    m, vr, lead_rmse = results[best]
    print(f"selected {best}")

    tc.MODEL_DIR.mkdir(exist_ok=True)
    torch.save({"state": m.state_dict(), "name": best, "mu": sc.mu, "sd": sc.sd,
                "lookback": 72, "steps": 24, "features": tc.FEATURES}, tc.MODEL_DIR / "model.pt")
    meta = {"model": best, "trained_utc": now.isoformat(), "train_start": str(f.index.min()),
            "train_end": str(vcut), "val_end": str(cut),
            "val_rmse_C": {k: round(v[1], 3) for k, v in results.items()},
            "val_rmse_by_lead_C": [round(float(x), 3) for x in lead_rmse],
            "val_persistence_rmse_by_lead_C": [round(float(x), 3) for x in pers_rmse]}
    (tc.MODEL_DIR / "meta.json").write_text(json.dumps(meta, indent=1))

    # out-of-sample hindcast over the held-out window
    model, sc2, ck = tc.load_model()
    origins = f.index[(f.index >= cut) & (f.index <= f["T"].last_valid_index()) & f["T"].notna()]
    rows = tc.forecast_rows(model, sc2, ck, f, origins, "hindcast")
    if len(rows):
        tc.write_forecasts(rows)
    print(f"hindcast: {rows.issued_utc.nunique() if len(rows) else 0} origins written")


if __name__ == "__main__":
    main()
