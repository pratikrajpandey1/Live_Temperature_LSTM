"""Shared code: IEM download, hourly cleaning, features, models, storage."""
import io, os, time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
OBS_DIR = ROOT / "data" / "obs"
FC_DIR = ROOT / "data" / "forecasts"
MODEL_DIR = ROOT / "models"
DOCS = ROOT / "docs"

STATION = os.environ.get("TAMB_STATION", "ABQ")          # IEM id (ICAO KABQ)
LOCAL_TZ = os.environ.get("TAMB_TZ", "America/Denver")
H = 24
ASOS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
UA = {"User-Agent": "Tamb_Prediction live-forecast (GitHub Actions)"}
SKY = {"CLR": 0.0, "SKC": 0.0, "NSC": 0.0, "NCD": 0.0, "FEW": 0.25, "SCT": 0.5,
       "BKN": 0.75, "OVC": 1.0, "VV": 1.0}
FEATURES = ["T", "Td", "RH", "u", "v", "p", "cloud", "hod_sin", "hod_cos", "doy_sin", "doy_cos"]
OBS_COLS = ["time_utc", "T_amb_C", "T_dew_C", "RH_pct", "wind_speed_ms", "wind_dir_deg",
            "p_alt_hPa", "sky_cover"]
torch.set_num_threads(max(1, os.cpu_count() or 1))


def utcnow():
    return datetime.now(timezone.utc)


# ----------------------------------------------------------------------------- download
def _get(params, tries=6):
    for i in range(tries):
        try:
            r = requests.get(ASOS_URL, params=params, headers=UA, timeout=300)
            if r.status_code == 200 and not r.text.startswith("#ERROR"):
                return r.text
            print(f"  IEM HTTP {r.status_code}: {r.text[:120]!r}")
        except requests.RequestException as e:
            print(f"  IEM {e}")
        time.sleep(5 * 2 ** i)
    raise RuntimeError("IEM download failed")


def fetch_metar(start, end):
    """Raw METARs (routine + specials) between start and end (UTC datetimes)."""
    off = os.environ.get("TAMB_OFFLINE_CSV")          # testing hook: hourly CSV instead of IEM
    if off:
        d = pd.read_csv(off, parse_dates=["time_utc"])
        d["time_utc"] = pd.to_datetime(d["time_utc"], utc=True)
        return d[(d.time_utc >= pd.Timestamp(start)) & (d.time_utc <= pd.Timestamp(end))]
    frames, s = [], start
    while s < end:
        e = min(end, s + timedelta(days=366))
        print(f"  IEM {s:%Y-%m-%d} -> {e:%Y-%m-%d}")
        params = [("station", STATION), ("data", "tmpf"), ("data", "dwpf"), ("data", "relh"),
                  ("data", "drct"), ("data", "sknt"), ("data", "alti"), ("data", "skyc1"),
                  ("year1", s.year), ("month1", s.month), ("day1", s.day),
                  ("year2", e.year), ("month2", e.month), ("day2", e.day),
                  ("tz", "Etc/UTC"), ("format", "onlycomma"), ("latlon", "no"), ("elev", "no"),
                  ("missing", "M"), ("trace", "T"), ("direct", "no"),
                  ("report_type", "3"), ("report_type", "4")]
        frames.append(pd.read_csv(io.StringIO(_get(params)), na_values=["M"], low_memory=False))
        s = e
        time.sleep(1)
    df = pd.concat(frames, ignore_index=True)
    df["valid"] = pd.to_datetime(df["valid"], utc=True)
    return df.drop_duplicates("valid").sort_values("valid")


def to_hourly(df):
    """Top-of-hour series from the nearest report within +/-20 min. Already hourly -> passthrough."""
    if "time_utc" in df.columns:
        return df[OBS_COLS].reset_index(drop=True)
    p = df.copy()
    for c in ["tmpf", "dwpf", "relh", "drct", "sknt", "alti"]:
        p[c] = pd.to_numeric(p[c], errors="coerce")
    p = p.dropna(subset=["tmpf"])
    grid = pd.DataFrame({"time_utc": pd.date_range(p.valid.min().ceil("h"), (p.valid.max() + pd.Timedelta("20min")).floor("h"),
                                                   freq="1h", tz="UTC")})
    h = pd.merge_asof(grid, p, left_on="time_utc", right_on="valid", direction="nearest",
                      tolerance=pd.Timedelta("20min"))
    out = pd.DataFrame({"time_utc": h.time_utc})
    out["T_amb_C"] = (h.tmpf - 32) * 5 / 9
    out["T_dew_C"] = (h.dwpf - 32) * 5 / 9
    out["RH_pct"] = h.relh
    out["wind_speed_ms"] = h.sknt * 0.514444
    out["wind_dir_deg"] = h.drct
    out["p_alt_hPa"] = h.alti * 33.8639
    out["sky_cover"] = h.skyc1
    c = ["T_amb_C", "T_dew_C", "RH_pct", "wind_speed_ms", "p_alt_hPa"]
    out[c] = out.set_index("time_utc")[c].interpolate("time", limit=3, limit_area="inside").values
    return out


def fetch_hourly(start, end):
    return to_hourly(fetch_metar(start, end))


# ----------------------------------------------------------------------------- storage
def upsert_obs(hourly):
    OBS_DIR.mkdir(parents=True, exist_ok=True)
    h = hourly.copy()
    h["m"] = h.time_utc.dt.strftime("%Y-%m")
    for m, g in h.groupby("m"):
        f = OBS_DIR / f"KABQ_{m}.csv"
        g = g.drop(columns="m")
        if f.exists():
            old = pd.read_csv(f, parse_dates=["time_utc"])
            old["time_utc"] = pd.to_datetime(old.time_utc, utc=True)
            g = pd.concat([old, g]).dropna(subset=["T_amb_C"])
        g.drop_duplicates("time_utc", keep="last").sort_values("time_utc").to_csv(f, index=False)


def read_obs(since):
    fs = sorted(OBS_DIR.glob("KABQ_*.csv"))
    if not fs:
        return pd.DataFrame(columns=OBS_COLS)
    d = pd.concat([pd.read_csv(f) for f in fs if f.stem[5:] >= since.strftime("%Y-%m")] or
                  [pd.DataFrame(columns=OBS_COLS)])
    d["time_utc"] = pd.to_datetime(d.time_utc, utc=True)
    return d[d.time_utc >= pd.Timestamp(since)].sort_values("time_utc")


def write_forecasts(rows):
    """rows: DataFrame issued_utc, valid_utc, lead_h, model, T_pred_C, source."""
    FC_DIR.mkdir(parents=True, exist_ok=True)
    rows = rows.copy()
    rows["day"] = rows.issued_utc.dt.strftime("%Y-%m-%d")
    for day, g in rows.groupby("day"):
        f = FC_DIR / f"{day}.csv"
        g = g.drop(columns="day")
        if f.exists():
            old = pd.read_csv(f)
            for c in ["issued_utc", "valid_utc"]:
                old[c] = pd.to_datetime(old[c], utc=True)
            g = pd.concat([old, g])
        g = g.drop_duplicates(["issued_utc", "lead_h", "model"], keep="last")
        g.sort_values(["issued_utc", "model", "lead_h"]).to_csv(f, index=False, float_format="%.3f")


def read_forecasts(since):
    fs = [f for f in sorted(FC_DIR.glob("*.csv")) if f.stem >= since.strftime("%Y-%m-%d")]
    if not fs:
        return pd.DataFrame(columns=["issued_utc", "valid_utc", "lead_h", "model", "T_pred_C", "source"])
    d = pd.concat([pd.read_csv(f) for f in fs])
    for c in ["issued_utc", "valid_utc"]:
        d[c] = pd.to_datetime(d[c], utc=True)
    return d


# ----------------------------------------------------------------------------- features
def features(hourly):
    d = hourly.set_index("time_utc").sort_index()
    d = d[~d.index.duplicated()].asfreq("1h")
    f = pd.DataFrame(index=d.index)
    f["T"], f["Td"], f["RH"] = d.T_amb_C, d.T_dew_C, d.RH_pct
    ws, wd = d.wind_speed_ms, np.deg2rad(d.wind_dir_deg)
    calm = ws == 0
    f["u"] = np.where(calm, 0.0, -ws * np.sin(wd))
    f["v"] = np.where(calm, 0.0, -ws * np.cos(wd))
    f["p"] = d.p_alt_hPa
    f["cloud"] = d.sky_cover.astype(str).str.strip().map(SKY)
    for c in ["u", "v", "p", "cloud", "Td", "RH"]:
        f[c] = f[c].interpolate(limit=3, limit_area="inside")
    loc = f.index.tz_convert(LOCAL_TZ)
    hod, doy = loc.hour, loc.dayofyear
    f["hod_sin"], f["hod_cos"] = np.sin(2 * np.pi * hod / 24), np.cos(2 * np.pi * hod / 24)
    f["doy_sin"], f["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    return f[FEATURES]


def windows(f, lookback, with_target=True):
    X = f.values.astype(np.float32)
    T = f["T"].values.astype(np.float32)
    n = len(f)
    last = n - H if with_target else n
    t0 = np.arange(lookback - 1, last)
    xin = np.lib.stride_tricks.sliding_window_view(X, lookback, axis=0)[: len(t0)].transpose(0, 2, 1)
    ok = np.isfinite(xin).all(axis=(1, 2))
    if with_target:
        y = np.lib.stride_tricks.sliding_window_view(T[lookback:], H)[: len(t0)]
        ok &= np.isfinite(y).all(axis=1)
        return xin[ok], y[ok], f.index[t0[ok]]
    return xin[ok], None, f.index[t0[ok]]


class Scaler:
    def __init__(self, mu=None, sd=None):
        self.mu, self.sd = mu, sd

    def fit(self, x):
        z = x.reshape(-1, x.shape[-1])
        self.mu, self.sd = z.mean(0), z.std(0) + 1e-6
        return self

    def x(self, x): return ((x - self.mu) / self.sd).astype(np.float32)
    def y(self, y): return ((y - self.mu[0]) / self.sd[0]).astype(np.float32)
    def y_inv(self, y): return y * self.sd[0] + self.mu[0]


# ----------------------------------------------------------------------------- models
class LSTMNet(nn.Module):
    def __init__(self, n_in, hidden=64, layers=2, drop=0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_in, hidden, layers, batch_first=True, dropout=drop)
        self.head = nn.Sequential(nn.Linear(hidden, 128), nn.ReLU(), nn.Dropout(drop), nn.Linear(128, H))

    def forward(self, x):
        o, _ = self.lstm(x)
        return self.head(o[:, -1])


class ConvLSTMCell(nn.Module):
    def __init__(self, c_in, c_hid, k=(1, 3)):
        super().__init__()
        self.c_hid = c_hid
        self.conv = nn.Conv2d(c_in + c_hid, 4 * c_hid, k, padding=(k[0] // 2, k[1] // 2))

    def forward(self, x, h, c):
        i, f, o, g = torch.chunk(self.conv(torch.cat([x, h], 1)), 4, 1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        return torch.sigmoid(o) * torch.tanh(c), c


class ConvLSTMNet(nn.Module):
    """Kreuzer et al. (2020)-style ConvLSTM: 72 h lookback = 3 x 24 h 'images', features as channels."""
    def __init__(self, n_in, n_seq, n_steps, hidden=32, drop=0.2):
        super().__init__()
        self.n_seq, self.n_steps = n_seq, n_steps
        self.c1, self.c2 = ConvLSTMCell(n_in, hidden), ConvLSTMCell(hidden, hidden)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(hidden * n_steps, 128), nn.ReLU(),
                                  nn.Dropout(drop), nn.Linear(128, H))

    def forward(self, x):
        B, L, F = x.shape
        x = x.reshape(B, self.n_seq, self.n_steps, F).permute(0, 1, 3, 2).unsqueeze(3)
        z = x.new_zeros(B, self.c1.c_hid, 1, self.n_steps)
        h1, c1, h2, c2 = z, z, z, z
        for s in range(self.n_seq):
            h1, c1 = self.c1(x[:, s], h1, c1)
            h2, c2 = self.c2(h1, h2, c2)
        return self.head(h2)


def build(name, lookback=72, steps=24):
    n = len(FEATURES)
    if name == "lstm_multi":
        return LSTMNet(n)
    if name == "convlstm":
        return ConvLSTMNet(n, lookback // steps, steps)
    raise ValueError(name)


def predict(model, X, bs=4096):
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(torch.tensor(X[i:i + bs])).numpy() for i in range(0, len(X), bs)])


def load_model():
    ck = torch.load(MODEL_DIR / "model.pt", weights_only=False)
    m = build(ck["name"], ck["lookback"], ck["steps"])
    m.load_state_dict(ck["state"])
    return m, Scaler(ck["mu"], ck["sd"]), ck


def forecast_rows(model, sc, ck, f, origins, source):
    """Model + persistence forecasts for each origin (timestamps in f.index)."""
    L = ck["lookback"]
    X = f.values.astype(np.float32)
    pos = f.index.get_indexer(origins)
    keep = [(o, p) for o, p in zip(origins, pos) if p >= L - 1]
    if not keep:
        return pd.DataFrame()
    xs = []
    for _, p in keep:
        w = pd.DataFrame(X[p - L + 1:p + 1]).interpolate(limit_direction="both").values
        xs.append(w)
    xs = np.stack(xs).astype(np.float32)
    ok = np.isfinite(xs).all(axis=(1, 2))
    xs, keep = xs[ok], [k for k, g in zip(keep, ok) if g]
    if not keep:
        return pd.DataFrame()
    yhat = sc.y_inv(predict(model, sc.x(xs)))
    T = f["T"].values
    out = []
    lead = np.arange(1, H + 1)
    for (o, p), yh in zip(keep, yhat):
        valid = o + pd.to_timedelta(lead, "h")
        pers = np.array([T[p + h - 24] if p + h - 24 >= 0 else np.nan for h in lead])
        for name, vals in [(ck["name"], yh), ("persistence", pers)]:
            out.append(pd.DataFrame({"issued_utc": o, "valid_utc": valid, "lead_h": lead,
                                     "model": name, "T_pred_C": vals, "source": source}))
    return pd.concat(out, ignore_index=True)
