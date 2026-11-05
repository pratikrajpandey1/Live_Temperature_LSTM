"""Build docs/index.html: live forecast vs. observed for KABQ (published with GitHub Pages)."""
import json
from datetime import timedelta

import numpy as np
import pandas as pd

import tamb_core as tc

WINDOW_DAYS = 30      # verification window
SHOW_DAYS = 7         # time-series chart span
Z80 = 1.2816          # 80 % band from per-lead RMSE (normal approx.)


def loc(ts):
    return ts.dt.tz_convert(tc.LOCAL_TZ).dt.strftime("%Y-%m-%d %H:%M").tolist()


def r(x, n=2):
    return [None if (v is None or not np.isfinite(v)) else round(float(v), n) for v in x]


def main():
    now = tc.utcnow()
    since = now - timedelta(days=WINDOW_DAYS + 1)
    obs = tc.read_obs(since)
    fc = tc.read_forecasts(since)
    meta = json.loads((tc.MODEL_DIR / "meta.json").read_text()) if (tc.MODEL_DIR / "meta.json").exists() else {}
    name = meta.get("model", "model")
    data = {"station": "KABQ - Albuquerque International Sunport, NM", "tz": tc.LOCAL_TZ,
            "updated_local": pd.Timestamp(now).tz_convert(tc.LOCAL_TZ).strftime("%Y-%m-%d %H:%M %Z"),
            "model": name, "meta": meta, "ready": bool(len(obs) and len(fc))}

    if data["ready"]:
        o = obs.set_index("time_utc")["T_amb_C"]
        fc = fc[fc.issued_utc >= pd.Timestamp(since)]
        v = fc.merge(o.rename("T_obs"), left_on="valid_utc", right_index=True, how="left")
        v["err"] = v.T_pred_C - v.T_obs
        ver = v.dropna(subset=["err"])

        g = ver.groupby(["model", "lead_h"])["err"]
        lead = np.arange(1, tc.H + 1)
        rm = lambda m: [float(np.sqrt((g.get_group((m, h)) ** 2).mean())) if (m, h) in g.groups else np.nan for h in lead]
        model_rmse, pers_rmse = rm(name), rm("persistence")

        mv = ver[ver.model == name]
        pv = ver[ver.model == "persistence"]
        day = lambda d: d.assign(day=d.valid_utc.dt.tz_convert(tc.LOCAL_TZ).dt.strftime("%Y-%m-%d")) \
                         .groupby("day")["err"].apply(lambda e: e.abs().mean())
        dm, dp = day(mv), day(pv)
        days = sorted(set(dm.index) | set(dp.index))[:-1] or sorted(set(dm.index))   # drop partial today

        show0 = pd.Timestamp(now - timedelta(days=SHOW_DAYS))
        os_ = obs[obs.time_utc >= show0]
        sel = lambda L: fc[(fc.model == name) & (fc.lead_h == L) & (fc.valid_utc >= show0)].sort_values("valid_utc")
        l24, l6 = sel(24), sel(6)

        last_issue = fc[fc.model == name].issued_utc.max()
        lat = fc[(fc.issued_utc == last_issue) & (fc.model == name)].sort_values("lead_h")
        latp = fc[(fc.issued_utc == last_issue) & (fc.model == "persistence")].sort_values("lead_h")
        band = np.array([x if np.isfinite(x) else (meta.get("val_rmse_by_lead_C") or [2.0] * 24)[i]
                         for i, x in enumerate(model_rmse)])
        live = fc[fc.source == "live"].issued_utc
        live_start = live.min() if len(live) else None

        def tile(L):
            m, p = model_rmse[L - 1], pers_rmse[L - 1]
            return {"rmse": r([m])[0], "pers": r([p])[0],
                    "skill": r([1 - m / p])[0] if np.isfinite(m) and np.isfinite(p) and p > 0 else None}

        data.update({
            "obs": {"t": loc(os_.time_utc), "T": r(os_.T_amb_C)},
            "lead24": {"t": loc(l24.valid_utc), "T": r(l24.T_pred_C)},
            "lead6": {"t": loc(l6.valid_utc), "T": r(l6.T_pred_C)},
            "latest": {"issued_local": loc(pd.Series([last_issue]))[0],
                       "t": loc(lat.valid_utc), "T": r(lat.T_pred_C),
                       "lo": r(lat.T_pred_C.values - Z80 * band[: len(lat)]),
                       "hi": r(lat.T_pred_C.values + Z80 * band[: len(lat)]),
                       "pers": r(latp.T_pred_C)},
            "lead": {"h": lead.tolist(), "model": r(model_rmse, 3), "pers": r(pers_rmse, 3)},
            "daily": {"day": days, "model": r([dm.get(d, np.nan) for d in days], 3),
                      "pers": r([dp.get(d, np.nan) for d in days], 3)},
            "tiles": {"h1": tile(1), "h6": tile(6), "h24": tile(24)},
            "n_issued": int(fc[fc.model == name].issued_utc.nunique()),
            "n_verified": int(len(mv)),
            "live_start_local": loc(pd.Series([live_start]))[0] if live_start is not None else None,
            "show_start_local": loc(pd.Series([show0]))[0],
            "window_days": WINDOW_DAYS,
        })

    tc.DOCS.mkdir(exist_ok=True)
    (tc.DOCS / "data.json").write_text(json.dumps(data))
    html = TEMPLATE.replace("/*__DATA__*/null", json.dumps(data))
    (tc.DOCS / "index.html").write_text(html, encoding="utf-8")
    (tc.DOCS / ".nojekyll").write_text("")
    print(f"site built: {tc.DOCS / 'index.html'}")


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Albuquerque 24-h temperature forecast - live</title>
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script>
<style>
:root{color-scheme:light;
 --page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;
 --grid:#e1e0d9;--axis:#c3c2b7;--ring:rgba(11,11,11,.10);
 --s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--obs:#0b0b0b;--hind:rgba(11,11,11,.045)}
@media (prefers-color-scheme:dark){:root{color-scheme:dark;
 --page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;
 --grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);
 --s1:#3987e5;--s2:#d95926;--s3:#199e70;--obs:#f0efec;--hind:rgba(255,255,255,.05)}}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);
 font:15px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;font-variant-numeric:tabular-nums}
main{max-width:1120px;margin:0 auto;padding:28px 20px 48px}
h1{font-size:24px;margin:0 0 4px;font-weight:650;letter-spacing:-.01em}
h2{font-size:16px;margin:0 0 2px;font-weight:600}
.sub{color:var(--ink2);margin:0}
.note{color:var(--muted);font-size:13px;margin:2px 0 8px}
.bar{display:flex;justify-content:space-between;align-items:flex-end;gap:16px;flex-wrap:wrap;margin-bottom:20px}
.seg{display:inline-flex;border:1px solid var(--ring);border-radius:8px;overflow:hidden}
.seg button{border:0;background:transparent;color:var(--ink2);padding:6px 14px;font:inherit;cursor:pointer}
.seg button[aria-pressed=true]{background:var(--surface);color:var(--ink);font-weight:600}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin-bottom:16px}
.tile,.card{background:var(--surface);border:1px solid var(--ring);border-radius:12px}
.tile{padding:14px 16px}
.tile .k{color:var(--ink2);font-size:13px}
.tile .v{font-size:28px;font-weight:650;letter-spacing:-.02em;margin:2px 0}
.tile .d{color:var(--muted);font-size:13px}
.card{padding:16px 16px 8px;margin-bottom:16px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media (max-width:820px){.grid2{grid-template-columns:1fr}}
.plot{width:100%;height:380px}.plot.sm{height:300px}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{text-align:right;padding:4px 8px;border-bottom:1px solid var(--grid)}
th:first-child,td:first-child{text-align:left}th{color:var(--ink2);font-weight:600}
details summary{cursor:pointer;color:var(--ink2);margin:6px 0 10px}
.wait{padding:40px;text-align:center;color:var(--ink2)}
footer{color:var(--muted);font-size:13px;margin-top:24px}
footer a{color:var(--ink2)}
</style>
</head>
<body>
<main>
 <div class="bar">
  <div>
   <h1>Albuquerque, NM - next-24-hour air temperature</h1>
   <p class="sub" id="sub"></p>
  </div>
  <div class="seg" role="group" aria-label="Units">
   <button id="uC" aria-pressed="true">°C</button><button id="uF" aria-pressed="false">°F</button>
  </div>
 </div>
 <div id="app"></div>
 <footer id="foot"></footer>
</main>
<script>
const D = /*__DATA__*/null;
let unit = "C"; try { unit = localStorage.getItem("tamb_unit") || "C"; } catch (e) {}
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const cv = (a, delta) => a == null ? a : (Array.isArray(a) ? a.map(x => cv(x, delta))
  : (x => x == null ? null : unit === "C" ? x : (delta ? x * 1.8 : x * 1.8 + 32))(a));
const fmt = (x, d = 1) => x == null ? "–" : x.toFixed(d);
const U = () => "°" + unit;

function layout(ytitle, extra = {}) {
  return Object.assign({
    paper_bgcolor: css("--surface"), plot_bgcolor: css("--surface"),
    font: {family: "system-ui,-apple-system,Segoe UI,Roboto,sans-serif", size: 12, color: css("--ink2")},
    margin: {l: 52, r: 16, t: 8, b: 40}, hovermode: "x unified",
    hoverlabel: {bgcolor: css("--surface"), bordercolor: css("--axis"), font: {color: css("--ink")}},
    legend: {orientation: "h", y: 1.12, x: 0, font: {color: css("--ink2")}},
    xaxis: {gridcolor: css("--grid"), linecolor: css("--axis"), zeroline: false, tickfont: {color: css("--muted")}},
    yaxis: {title: {text: ytitle, font: {color: css("--ink2")}}, gridcolor: css("--grid"),
            linecolor: css("--axis"), zeroline: false, tickfont: {color: css("--muted")}}
  }, extra);
}
const cfg = {responsive: true, displaylogo: false, displayModeBar: false};

function render() {
  document.getElementById("uC").setAttribute("aria-pressed", unit === "C");
  document.getElementById("uF").setAttribute("aria-pressed", unit === "F");
  const app = document.getElementById("app");
  document.getElementById("sub").textContent = D.station + " · page updated " + D.updated_local;
  document.getElementById("foot").innerHTML =
    "Observations: Iowa Environmental Mesonet ASOS archive (KABQ METARs, nearest report to each hour). " +
    "Model: " + D.model + " trained on hourly KABQ history (temperature, dew point, humidity, wind, pressure, cloud, time of day/year; " +
    "72-h look-back, 24 hourly outputs), after Kreuzer, Munz &amp; Schlüter (2020). Baseline: persistence (same hour yesterday). " +
    "Shaded band: ±1.28 × recent RMSE at each lead (≈80%). Rebuilt every hour by GitHub Actions." +
    (D.meta && D.meta.trained_utc ? " Model last trained " + D.meta.trained_utc.slice(0, 10) + "." : "");
  if (!D.ready) { app.innerHTML = '<div class="card wait">Waiting for the first forecast run…</div>'; return; }

  const T = D.tiles, Lt = D.latest;
  const tile = (k, t) => `<div class="tile"><div class="k">${k}</div>
    <div class="v">${fmt(cv(t.rmse, 1), 2)} ${U()}</div>
    <div class="d">RMSE, last ${D.window_days} d · persistence ${fmt(cv(t.pers, 1), 2)}${t.skill != null ? " · skill " + Math.round(t.skill * 100) + "%" : ""}</div></div>`;
  const hi = Math.max(...Lt.T), lo = Math.min(...Lt.T);
  app.innerHTML = `
   <div class="tiles">
    <div class="tile"><div class="k">Next 24 h (issued ${Lt.issued_local})</div>
      <div class="v">${fmt(cv(lo))} – ${fmt(cv(hi))} ${U()}</div>
      <div class="d">forecast low – high</div></div>
    ${tile("1 hour ahead", T.h1)}${tile("6 hours ahead", T.h6)}${tile("24 hours ahead", T.h24)}
   </div>
   <div class="card"><h2>Forecast vs. observed</h2>
    <p class="note">Observed: measured at KABQ. Blue / orange: what the model predicted 24 h / 6 h earlier for that hour. Green: the newest forecast for the next 24 h.${D.live_start_local ? "" : " All forecasts so far are hindcasts (held-out period)."}</p>
    <div id="ts" class="plot"></div></div>
   <div class="grid2">
    <div class="card"><h2>Error by lead time</h2><p class="note">RMSE over the last ${D.window_days} days. Lower is better.</p><div id="lead" class="plot sm"></div></div>
    <div class="card"><h2>Daily mean absolute error</h2><p class="note">All lead times, by local day.</p><div id="daily" class="plot sm"></div></div>
   </div>
   <div class="card"><details><summary>Latest forecast as a table</summary>
    <table><thead><tr><th>Local time</th><th>Forecast (${U()})</th><th>80% range</th><th>Persistence</th></tr></thead><tbody>
    ${Lt.t.map((t, i) => `<tr><td>${t}</td><td>${fmt(cv(Lt.T[i]))}</td><td>${fmt(cv(Lt.lo[i]))} – ${fmt(cv(Lt.hi[i]))}</td><td>${fmt(cv(Lt.pers[i]))}</td></tr>`).join("")}
    </tbody></table></details></div>`;

  const s1 = css("--s1"), s2 = css("--s2"), s3 = css("--s3"), ob = css("--obs");
  const hov = U() === "°C" ? "%{y:.1f} °C" : "%{y:.1f} °F";
  const shapes = [], ann = [];
  const hindEnd = D.live_start_local || Lt.issued_local;
  if (hindEnd > D.show_start_local) {
    shapes.push({type: "rect", xref: "x", yref: "paper", x0: D.show_start_local, x1: hindEnd, y0: 0, y1: 1,
                 fillcolor: css("--hind"), line: {width: 0}, layer: "below"});
    ann.push({xref: "x", yref: "paper", x: D.show_start_local, y: 1, xanchor: "left", yanchor: "top",
              text: D.live_start_local ? "hindcast" : "hindcast (held-out)", showarrow: false,
              font: {size: 11, color: css("--muted")}});
  }
  shapes.push({type: "line", xref: "x", yref: "paper", x0: Lt.issued_local, x1: Lt.issued_local, y0: 0, y1: 1,
               line: {color: css("--axis"), width: 1, dash: "dot"}});
  ann.push({xref: "x", yref: "paper", x: Lt.issued_local, y: 1, xanchor: "left", yanchor: "top",
            text: " now", showarrow: false, font: {size: 11, color: css("--muted")}});

  Plotly.react("ts", [
    {x: Lt.t, y: cv(Lt.hi), mode: "lines", line: {width: 0}, hoverinfo: "skip", showlegend: false},
    {x: Lt.t, y: cv(Lt.lo), mode: "lines", line: {width: 0}, fill: "tonexty", fillcolor: s3 + "2e",
     name: "80% range", hovertemplate: hov},
    {x: D.lead24.t, y: cv(D.lead24.T), name: "Predicted 24 h earlier", mode: "lines", line: {color: s1, width: 2}, hovertemplate: hov},
    {x: D.lead6.t, y: cv(D.lead6.T), name: "Predicted 6 h earlier", mode: "lines", line: {color: s2, width: 2}, hovertemplate: hov},
    {x: Lt.t, y: cv(Lt.T), name: "Latest forecast", mode: "lines", line: {color: s3, width: 2.5}, hovertemplate: hov},
    {x: D.obs.t, y: cv(D.obs.T), name: "Observed", mode: "lines", line: {color: ob, width: 2}, hovertemplate: hov},
  ], layout("Air temperature (" + U() + ")", {shapes, annotations: ann}), cfg);

  const lx = {title: {text: "Hours ahead", font: {color: css("--ink2")}}, dtick: 3};
  Plotly.react("lead", [
    {x: D.lead.h, y: cv(D.lead.model, 1), name: D.model, mode: "lines+markers", line: {color: s1, width: 2},
     marker: {size: 8, color: s1, line: {color: css("--surface"), width: 2}}, hovertemplate: "%{y:.2f}"},
    {x: D.lead.h, y: cv(D.lead.pers, 1), name: "Persistence", mode: "lines+markers", line: {color: s2, width: 2, dash: "dash"},
     marker: {size: 8, symbol: "square", color: s2, line: {color: css("--surface"), width: 2}}, hovertemplate: "%{y:.2f}"},
  ], (l => (Object.assign(l.xaxis, lx), l.yaxis.rangemode = "tozero", l))(layout("RMSE (" + U() + ")")), cfg);

  Plotly.react("daily", [
    {x: D.daily.day, y: cv(D.daily.model, 1), name: D.model, mode: "lines+markers", line: {color: s1, width: 2},
     marker: {size: 8, color: s1, line: {color: css("--surface"), width: 2}}, hovertemplate: "%{y:.2f}"},
    {x: D.daily.day, y: cv(D.daily.pers, 1), name: "Persistence", mode: "lines+markers", line: {color: s2, width: 2, dash: "dash"},
     marker: {size: 8, symbol: "square", color: s2, line: {color: css("--surface"), width: 2}}, hovertemplate: "%{y:.2f}"},
  ], (l => (l.yaxis.rangemode = "tozero", l))(layout("MAE (" + U() + ")")), cfg);
}
["C", "F"].forEach(u => document.getElementById("u" + u).onclick = () => {
  unit = u; try { localStorage.setItem("tamb_unit", u); } catch (e) {} render(); });
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", render);
render();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
