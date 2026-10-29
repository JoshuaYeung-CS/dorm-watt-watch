"""Render a self-contained HTML report (charts embedded as PNG data URIs, works from file://)."""

from __future__ import annotations

import base64
import html
import io
import json
import os
from datetime import date
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "dormwatt-mpl"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .disaggregate import BULB_WATTS, CATEGORIES, LABELS, LAMP_ON_SHARE, Checklist, Estimate  # noqa: E402
from .factors import GridFactors  # noqa: E402
from .recommend import Plan  # noqa: E402
from .sources import ACCESSED, SOURCES  # noqa: E402

# Okabe-Ito palette (colour-blind safe)
COLORS = {"always_on": "#0072B2", "heating": "#D55E00", "cooling": "#56B4E9",
          "lighting": "#E69F00", "electronics": "#009E73"}
INK = "#1f2933"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#9aa5b1", "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False,
                     "axes.spines.right": False, "font.family": "DejaVu Sans"})


def _png(fig, save_to: Path | None = None) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    data = buf.getvalue()
    if save_to:
        Path(save_to).write_bytes(data)
    return "data:image/png;base64," + base64.b64encode(data).decode()


def chart_breakdown(est: Estimate, save_to=None) -> str:
    f = 365.0 / est.days
    cats = [c for c in CATEGORIES if est.totals[c] > 0.5]
    vals = [est.totals[c] * f for c in cats]
    lo = [est.intervals[c][0] * f for c in cats]
    hi = [est.intervals[c][1] * f for c in cats]
    fig, ax = plt.subplots(figsize=(7.2, 0.55 * len(cats) + 0.8))
    y = np.arange(len(cats))[::-1]
    ax.barh(y, vals, color=[COLORS[c] for c in cats], height=0.6)
    has_ci = all(np.isfinite(lo)) and all(np.isfinite(hi))
    if has_ci:
        ax.errorbar(vals, y, xerr=[np.subtract(vals, lo), np.subtract(hi, vals)], fmt="none",
                    ecolor=INK, capsize=3, lw=1)
    for yi, v in zip(y, vals):
        ax.text(v, yi + 0.36, f" {v:,.0f}", va="center", ha="left", fontsize=9, color=INK)
    ax.set_yticks(y, [LABELS[c] for c in cats])
    ax.set_xlabel("kWh per year" + ("  (bars: estimate, whiskers: 90% bootstrap interval)" if has_ci else ""))
    ax.set_xlim(0, max(hi if has_ci else vals) * 1.15)
    return _png(fig, save_to)


def chart_monthly(est: Estimate, save_to=None) -> str:
    months = est.hours.astype("datetime64[M]")
    um = np.unique(months)
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    bottom = np.zeros(len(um))
    for c in CATEGORIES:
        v = np.array([est.components[c][months == m].sum() for m in um])
        if v.sum() < 0.5:
            continue
        ax.bar(np.arange(len(um)), v, bottom=bottom, color=COLORS[c], label=LABELS[c], width=0.75)
        bottom += v
    ax.set_xticks(np.arange(len(um)), [date.fromisoformat(str(m) + "-01").strftime("%b") for m in um])
    ax.set_ylabel("kWh per month")
    ax.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    return _png(fig, save_to)


def chart_day(est: Estimate, month: int | None = None, save_to=None) -> str:
    """Average day (home days only) in a chosen month: metered line over stacked estimate."""
    hrs = est.hours
    hod = (hrs - hrs.astype("datetime64[D]")).astype(int)
    mon = hrs.astype("datetime64[M]").astype(int) % 12 + 1
    day = hrs.astype("datetime64[D]")
    home = ~np.isin(day, est.away_days)
    if month is None:
        month = int(np.bincount(mon[home]).argmax())
    sel = home & (mon == month) & np.isfinite(est.y)
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    bottom = np.zeros(24)
    for c in CATEGORIES:
        v = np.array([est.components[c][sel & (hod == h)].mean() if (sel & (hod == h)).any() else 0 for h in range(24)])
        if v.sum() < 1e-3:
            continue
        ax.fill_between(np.arange(24), bottom, bottom + v, step="mid", color=COLORS[c], alpha=0.9, label=LABELS[c])
        bottom += v
    meas = [est.y[sel & (hod == h)].mean() for h in range(24)]
    ax.plot(np.arange(24), meas, color=INK, lw=1.5, label="Metered (average)")
    ax.set_xticks(range(0, 24, 3), [f"{h:02d}:00" for h in range(0, 24, 3)])
    ax.set_ylabel("kWh per hour")
    ax.set_title(f"Average day at home, {date(2000, month, 1).strftime('%B')}", fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    return _png(fig, save_to)


def chart_savings(plan: Plan, save_to=None) -> str:
    s = plan.savings
    fig, ax = plt.subplots(figsize=(7.2, 0.5 * len(s) + 0.8))
    y = np.arange(len(s))[::-1]
    top = {x.key for x in plan.top}
    ax.barh(y, [x.kwh_year for x in s], color=["#009E73" if x.key in top else "#b8c2cc" for x in s], height=0.6)
    for yi, x in zip(y, s):
        ax.text(x.kwh_year, yi, f"  {x.kwh_year:,.0f} kWh  ${x.usd_year:,.0f}  {x.kg_year:,.0f} kg", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, [x.title for x in s], fontsize=8)
    ax.set_xlabel("Saving, kWh per year (green: top three)")
    ax.set_xlim(0, max(x.kwh_year for x in s) * 1.9 if s else 1)
    return _png(fig, save_to)


def chart_validation(validation: dict, save_to=None) -> str:
    runs = validation["year_15min_runs"]
    cats = [c for c in CATEGORIES if any(r["truth"][c] > 1 for r in runs)]
    fig, axes = plt.subplots(1, len(cats), figsize=(2.3 * len(cats), 2.6))
    for ax, c in zip(np.atleast_1d(axes), cats):
        t = np.array([r["truth"][c] for r in runs])
        e = np.array([r["est"][c] for r in runs])
        m = max(t.max(), e.max()) * 1.05 + 1
        ax.plot([0, m], [0, m], color="#9aa5b1", lw=1)
        ax.scatter(t, e, s=12, color=COLORS[c])
        ax.set_xlim(0, m)
        ax.set_ylim(0, m)
        ax.set_title(LABELS[c].split(" (")[0].replace(", cooking and other plug loads", " + other"), fontsize=8)
        ax.set_xlabel("true kWh/yr", fontsize=8)
        ax.tick_params(labelsize=7)
    np.atleast_1d(axes)[0].set_ylabel("estimated kWh/yr", fontsize=8)
    fig.tight_layout()
    return _png(fig, save_to)


CSS = """
:root{--bg:#f7f9fb;--card:#ffffff;--ink:#1f2933;--muted:#52606d;--line:#d9e2ec;--accent:#007a5a;--warn:#8a4b00}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#12171c;--card:#1b2229;--ink:#e4e7eb;--muted:#a9b4be;--line:#33404d;--accent:#3ccf9c;--warn:#f0b35a}}
:root[data-theme="dark"]{--bg:#12171c;--card:#1b2229;--ink:#e4e7eb;--muted:#a9b4be;--line:#33404d;--accent:#3ccf9c;--warn:#f0b35a}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:920px;margin:0 auto;padding:24px 16px 64px}h1{font-size:1.9rem;margin:.2em 0}h2{font-size:1.3rem;margin:2em 0 .6em;border-bottom:1px solid var(--line);padding-bottom:.3em}
h3{font-size:1.05rem;margin:.2em 0 .4em}p,li{max-width:72ch}.muted{color:var(--muted)}.small{font-size:.88rem}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:16px 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.big{font-size:1.6rem;font-weight:700}.win{border-left:5px solid var(--accent)}.win .big{color:var(--accent)}
figure{margin:12px 0;background:#fff;border:1px solid var(--line);border-radius:10px;padding:8px}figure img{width:100%;height:auto;display:block}
figcaption{color:#52606d;font-size:.85rem;padding:4px 6px}
table{border-collapse:collapse;width:100%;font-size:.92rem;background:var(--card)}th,td{border:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}
th{background:color-mix(in srgb,var(--card) 85%,var(--ink) 15%)}td.n{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.tablewrap{overflow-x:auto}.warn{color:var(--warn)}code{font-size:.9em}a{color:var(--accent)}
.badge{display:inline-block;font-size:.75rem;border:1px solid var(--line);border-radius:999px;padding:1px 8px;margin-left:6px;color:var(--muted)}
"""


def _fmt(x: float, d: int = 0) -> str:
    return f"{x:,.{d}f}"


def build_html(est: Estimate, plan: Plan, cl: Checklist, g: GridFactors, title_note: str = "",
               validation: dict | None = None, chart_dir: Path | None = None) -> str:
    e = html.escape
    f = 365.0 / est.days
    total_y = est.total_kwh * f
    cd = (lambda n: Path(chart_dir) / n) if chart_dir else (lambda n: None)
    img_break = chart_breakdown(est, cd("breakdown.png"))
    img_month = chart_monthly(est, cd("monthly.png"))
    img_day = chart_day(est, 1 if (est.hours.astype("datetime64[M]").astype(int) % 12 == 0).sum() > 24 * 20 else None, cd("winter_day.png"))
    img_save = chart_savings(plan, cd("savings.png"))
    img_val = chart_validation(validation, cd("validation.png")) if validation and "year_15min_runs" in validation else None

    top = plan.top
    tk = sum(s.kwh_year for s in top)
    tu = sum(s.usd_year for s in top)
    tg = sum(s.kg_year for s in top)
    tks = sum(s.kwh_semester for s in top)
    tus = sum(s.usd_semester for s in top)
    tgs = sum(s.kg_semester for s in top)
    s0, s1 = plan.semester

    out = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
           f"<title>Dorm Watt Watch report</title><style>{CSS}</style></head><body><main>"]
    out.append("<p class='muted small'>Dorm Watt Watch</p><h1>Where your electricity goes, and the three changes that save the most</h1>")
    if title_note:
        out.append(f"<p class='warn'>{e(title_note)}</p>")
    out.append(f"<p class='muted'>{e(str(est.hours[0])[:10])} to {e(str(est.hours[-1])[:10])} &middot; "
               f"{est.days:.0f} days of data &middot; grid region {e(g.subregion)} ({e(g.subregion_name)}) &middot; "
               f"price {g.price_usd_per_kwh * 100:.2f}&cent;/kWh ({e(g.state)})</p>")
    out.append("<div class='cards'>"
               f"<div class='card'><div class='muted small'>Electricity per year</div><div class='big'>{_fmt(total_y)} kWh</div>"
               f"<div class='small'>${_fmt(total_y * g.price_usd_per_kwh)} &middot; {_fmt(total_y * g.kg_co2e_per_kwh)} kg CO2e</div></div>"
               f"<div class='card win'><div class='muted small'>Top 3 changes save, per year</div><div class='big'>{_fmt(tk)} kWh</div>"
               f"<div class='small'>${_fmt(tu)} &middot; {_fmt(tg)} kg CO2e ({tk / total_y * 100:.0f}% of your use)</div></div>"
               f"<div class='card win'><div class='muted small'>Per semester ({e(str(s0)[:10])} to {e(str(s1 - 1)[:10])})</div><div class='big'>{_fmt(tks)} kWh</div>"
               f"<div class='small'>${_fmt(tus)} &middot; {_fmt(tgs)} kg CO2e</div></div></div>")

    out.append("<h2>Your three biggest wins</h2><div class='cards'>")
    for i, s in enumerate(top, 1):
        out.append(f"<div class='card win'><h3>{i}. {e(s.title)}</h3><p class='small'>{e(s.action)}</p>"
                   f"<div class='big'>{_fmt(s.kwh_year)} kWh/yr</div><div class='small'>${_fmt(s.usd_year)}/yr &middot; {_fmt(s.kg_year)} kg CO2e/yr<br>"
                   f"this semester: {_fmt(s.kwh_semester)} kWh &middot; ${_fmt(s.usd_semester)} &middot; {_fmt(s.kg_semester)} kg</div>"
                   f"<p class='small muted'><strong>Method:</strong> {e(s.method)}"
                   + "".join(f"<br><em>Assumption:</em> {e(a)}" for a in s.assumptions) + "</p></div>")
    out.append("</div>")

    out.append("<h2>Where your electricity goes</h2>"
               f"<figure><img src='{img_break}' alt='Horizontal bar chart of estimated kWh per year by category with 90 percent intervals; values are in the table below.'>"
               "<figcaption>Estimated energy by category. Whiskers are 90% day-block bootstrap intervals (sampling noise only). Model error is larger: see the 'typical error in tests' column and Accuracy."
               + (" Lighting's range is the &plusmn;40% range of its wattage and on-share assumptions." if est.lighting_method == "bottom-up" else "")
               + "</figcaption></figure>")
    vst = (validation or {}).get("settings", {}).get("year_15min", {})
    out.append("<div class='tablewrap'><table><tr><th>Category</th><th>kWh/yr</th><th>90% sampling interval</th>"
               "<th>Typical error in tests</th><th>Share</th><th>$/yr</th><th>kg CO2e/yr</th></tr>")
    for c in CATEGORIES:
        v = est.totals[c] * f
        if v < 0.5 and not (c == "heating" and cl.space_heater):
            continue
        lo, hi = (x * f for x in est.intervals[c])
        iv = f"{_fmt(lo)} - {_fmt(hi)}" if np.isfinite(lo) else "n/a"
        te = vst.get(c, {})
        terr = f"&plusmn;{te['median_ape_pct']:.0f}% (1 in 5: &gt;{te['p80_ape_pct']:.0f}%)" if te.get("median_ape_pct") is not None else "n/a"
        out.append(f"<tr><td>{e(LABELS[c])}</td><td class='n'>{_fmt(v)}</td><td class='n'>{iv}</td><td class='n'>{terr}</td><td class='n'>{v / total_y * 100:.0f}%</td>"
                   f"<td class='n'>${_fmt(v * g.price_usd_per_kwh)}</td><td class='n'>{_fmt(v * g.kg_co2e_per_kwh)}</td></tr>")
    out.append("</table></div>")
    if cl.mini_fridge:
        out.append(f"<p class='small muted'>Inside the always-on load, the mini-fridge is taken as about {_fmt(plan.fridge_kwh_year)} kWh/yr: {e(plan.fridge_method)}</p>")
    out.append(f"<figure><img src='{img_month}' alt='Stacked monthly bar chart of estimated kWh by category.'><figcaption>Month by month. Breaks show up as months with only the always-on load.</figcaption></figure>")
    out.append(f"<figure><img src='{img_day}' alt='Average day at home: stacked estimated components by hour with the metered average as a line.'><figcaption>The black line is the meter; the colours are the estimate. Fit on home days: R&sup2; {est.r2_hourly:.2f} hourly, {est.r2_daily:.2f} daily.</figcaption></figure>")

    out.append("<h2>Every option, ranked</h2>"
               f"<figure><img src='{img_save}' alt='Bar chart of yearly kWh saved by each option; values in the table below.'></figure>"
               "<div class='tablewrap'><table><tr><th>Change</th><th>kWh/yr</th><th>$/yr</th><th>kg CO2e/yr</th><th>kWh this semester</th><th>How it is calculated</th></tr>")
    for s in plan.savings:
        src = ", ".join(SOURCES[k]["publisher"].replace("U.S. ", "") for k in s.sources) or "assumption"
        out.append(f"<tr><td>{e(s.title)}{'<span class=badge>top 3</span>' if s in top else ''}</td><td class='n'>{_fmt(s.kwh_year)}</td>"
                   f"<td class='n'>${_fmt(s.usd_year)}</td><td class='n'>{_fmt(s.kg_year)}</td><td class='n'>{_fmt(s.kwh_semester)}</td>"
                   f"<td class='small'>{e(s.method)} <span class='muted'>[{e(src)}]</span>"
                   + "".join(f"<br><em>Assumption:</em> {e(a)}" for a in s.assumptions) + "</td></tr>")
    out.append("</table></div>")
    out.append(f"<p class='small'>CO2e uses EPA eGRID's average (total output) rate for {e(g.subregion)}, "
               f"{g.co2e_lb_per_mwh:.1f} lb/MWh, grossed up for {g.grid_gross_loss * 100:.1f}% grid loss: "
               f"<strong>{g.kg_co2e_per_kwh:.3f} kg CO2e per kWh</strong> at your meter. Power saved at the margin is "
               f"usually from fossil plants; with eGRID's non-baseload rate ({g.nonbaseload_co2e_lb_per_mwh:.0f} lb/MWh, "
               f"{g.kg_co2e_per_kwh_nonbaseload:.3f} kg/kWh) the top three would avoid "
               f"<strong>{_fmt(sum(s.kg_year_marginal for s in top))} kg CO2e/yr</strong>. The average rate is the headline because it is the conservative one here.</p>")
    if g.note:
        out.append(f"<p class='small warn'>{e(g.note)}</p>")

    out.append("<h2>Method</h2><ol class='small'>"
               f"<li><strong>Read the meter file.</strong> {est.days:.0f} days of interval readings summed into clock hours.</li>"
               f"<li><strong>Always-on load.</strong> {e(est.baseload_method)}: {est.baseload_kw * 1000:.0f} W, "
               f"{len(est.away_days)} away days found (a day is 'away' when its mean and peak stay near the lowest days' minimum).</li>"
               "<li><strong>Regression on physical drivers.</strong> On home days the hourly use above the always-on load"
               + (" and the lighting estimate" if est.lighting_method == "bottom-up" else "") + " is fitted, "
               "with non-negative least squares, to columns built from outdoor temperature (degree-hours from balance points chosen by least squares"
               + (f"; heating {est.balance_heat_f:.0f} F" if est.balance_heat_f is not None else "")
               + (f"; cooling {est.balance_cool_f:.0f} F" if est.balance_cool_f is not None else "") + "), "
               "a monotone response to the day's mean temperature (non-negative ramps, so heating can only rise as it gets colder), "
               "and your own weekly activity pattern (electronics and everything else). "
               "The appliance checklist only switches parts off (no AC means no cooling term).</li>"
               + ("<li><strong>Lighting, bottom-up.</strong> A whole-home meter cannot tell evening lights from evening electronics, so lighting = "
                  f"your {cl.bulbs} {e(cl.lighting)} bulbs x a typical {e(str(int(BULB_WATTS.get(cl.lighting, 0))))} W x {LAMP_ON_SHARE:.0%} on-share, "
                  "in every hour the meter shows you up (use 50 Wh above the always-on load) while the sun is low at your location. "
                  "Wattage and on-share are assumptions; the range shown is &plusmn;40%.</li>"
                  if est.lighting_method == "bottom-up" else
                  "<li><strong>Lighting by regression</strong> on activity x darkness from the sun's position. Give your bulb count for a steadier estimate.</li>")
               + "<li><strong>Hour-by-hour shares.</strong> Each hour's metered kWh is split in proportion to the fitted parts, so the categories add up to your bill.</li>"
               "<li><strong>Uncertainty.</strong> Days are resampled with replacement and the fit repeated (day-block bootstrap) for 90% intervals.</li>"
               "<li><strong>Savings.</strong> Each option multiplies the estimated kWh it affects by a sourced reduction factor or a labelled assumption; "
               "money uses the EIA state residential price and CO2e uses EPA eGRID.</li></ol>")

    out.append("<h2>Accuracy</h2>")
    if validation:
        st = validation["settings"]["year_15min"]
        base = validation["settings"].get("checklist_only_baseline", {})
        sem = validation["settings"].get("fall_semester_15min", {})
        out.append(f"<p class='small'>Tested on {validation['n_households']} synthetic student households simulated minute by minute "
                   "(thermostat-driven heater and AC on real 2025 Ithaca weather, cycling fridge, randomly timed lights, gaming, TV, cooking, breaks). "
                   "The estimator never sees their true schedules. Median absolute error per category, over households that have it:</p>"
                   "<div class='tablewrap'><table><tr><th>Category</th><th>Median error, full year</th><th>80th pct error</th><th>Mean abs error</th><th>Median error, fall semester only</th><th>No-meter guess (median error)</th></tr>")
        for c in CATEGORIES:
            r = st[c]
            if not r["n_present"]:
                continue
            out.append(f"<tr><td>{e(LABELS[c])}</td><td class='n'>{r['median_ape_pct']:.0f}%</td><td class='n'>{r['p80_ape_pct']:.0f}%</td>"
                       f"<td class='n'>{r['mae_kwh']:.0f} kWh</td><td class='n'>{(sem.get(c) or {}).get('median_ape_pct') or float('nan'):.0f}%</td>"
                       f"<td class='n'>{(base.get(c) or {}).get('median_ape_pct') or float('nan'):.0f}%</td></tr>")
        out.append("</table></div>")
        t3 = st.get("_top3_savings")
        if t3:
            out.append(f"<p class='small'><strong>The savings themselves:</strong> feeding each test household's <em>true</em> category energy into the same "
                       f"savings rules, the estimated top-three saving was off by a median {t3['median_ape_pct']:.0f}% (1 in 5 households: over {t3['p80_ape_pct']:.0f}%), "
                       f"and the same three changes came out on top in {t3['same_top3_share'] * 100:.0f}% of households. "
                       "(The computer-sleep saving comes mostly from the checklist, which helps this agreement.)</p>")
        if img_val:
            out.append(f"<figure><img src='{img_val}' alt='Scatter plots of estimated versus true kWh per year for each category across synthetic households.'><figcaption>Each dot is one synthetic household; the grey line is a perfect estimate.</figcaption></figure>")
    if vst:
        def m(c):
            return vst[c]["median_ape_pct"]
        out.append("<p class='small'><strong>Read the numbers this way:</strong> in these tests the median error was "
                   f"{m('heating'):.0f}% for heating, {m('always_on'):.0f}% for the always-on load (it reads low), "
                   f"{m('electronics'):.0f}% for electronics and other plug loads and {m('lighting'):.0f}% for lighting "
                   "(the lighting test is optimistic: the simulator and the estimator both assume lights are on when you are up after dark). "
                   "The 90% sampling intervals cover only day-to-day noise and are too narrow as error bars. Savings inherit these errors.</p>")
    out.append("<h2>Limits</h2><ul class='small'>"
               "<li>A whole-home meter cannot see individual devices. Categories are estimated from when and how use changes with weather, daylight and your routine; devices that share a pattern (TV vs. PC, cooking vs. chargers) cannot be separated and are grouped.</li>"
               "<li>A heater that runs whatever the weather looks like any other plug load; heating is identified only through its link to outdoor temperature.</li>"
               "<li>Lighting depends on the bulb count and type you enter.</li>"
               "<li>The fridge/standby split inside the always-on load is a prior from ENERGY STAR data, not a measurement.</li>"
               "<li>The always-on load is measured on away days; in winter breaks a cold, empty room makes a fridge work less, so it reads about 10% low in tests.</li>"
               "<li>Without a temperature file, heating and cooling cannot be separated from other loads; a file with only cold months cannot separate a heater's steady part from plug loads.</li>"
               "<li>Accuracy was measured on simulated households, not on labelled real homes.</li>"
               "<li>CO2 uses annual-average eGRID2023 rates; hour-by-hour grid intensity is not modelled. Prices are the state average, not your tariff.</li>"
               "<li>Per-year figures scale the data period to 365 days" + (" (this file is shorter than a year, so seasonal loads are extrapolated)." if est.days < 330 else ".") + "</li></ul>")
    if est.warnings:
        out.append("<p class='small warn'>" + "<br>".join(e(w) for w in est.warnings) + "</p>")

    out.append("<h2>Sources</h2><ul class='small'>")
    for k, s in SOURCES.items():
        out.append(f"<li><strong>{e(s['publisher'])}</strong>: {e(s['title'])}. {e(s['vintage'])}. "
                   f"<a href='{e(s['url'])}'>{e(s['url'])}</a> (accessed {ACCESSED})</li>")
    out.append("</ul><p class='small muted'>Wattages of bulb types, the 3 lit hours/day lighting prior, PC idle watts when not given, "
               "the 25% standby cut and the meal-time shape are assumptions and are labelled wherever they are used.</p>")
    out.append("</main></body></html>")
    return "".join(out)


def to_json(est: Estimate, plan: Plan, g: GridFactors) -> dict:
    f = 365.0 / est.days
    return {
        "period": [str(est.hours[0]), str(est.hours[-1])], "days": est.days,
        "grid": {"subregion": g.subregion, "kg_co2e_per_kwh": g.kg_co2e_per_kwh,
                 "kg_co2e_per_kwh_nonbaseload": g.kg_co2e_per_kwh_nonbaseload,
                 "state": g.state, "usd_per_kwh": g.price_usd_per_kwh},
        "kwh_per_year": {c: est.totals[c] * f for c in CATEGORIES},
        "interval_90_kwh_per_year": {c: [x * f for x in est.intervals[c]] for c in CATEGORIES},
        "baseload_w": est.baseload_kw * 1000, "away_days": len(est.away_days),
        "r2_hourly": est.r2_hourly, "r2_daily": est.r2_daily,
        "semester": [str(plan.semester[0]), str(plan.semester[1])],
        "savings": [{k: getattr(s, k) for k in ("key", "title", "kwh_year", "kwh_semester", "usd_year", "usd_semester",
                                                 "kg_year", "kg_semester", "kg_year_marginal", "method", "sources", "assumptions")}
                    for s in plan.savings],
        "top3": [s.key for s in plan.top],
        "warnings": est.warnings,
    }


def load_validation(path: Path) -> dict | None:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
