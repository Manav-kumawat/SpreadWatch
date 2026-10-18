"""
L5 Output - generated annexure.

    python report.py --mode sample      (or --mode real)

Writes outputs/<mode>/SpreadWatch_Annexure.html (open in a browser, print to PDF
for the annexure) and outputs/<mode>/figures/*.png (numbered figures for the
report and the PowerPoint deck).
"""
import argparse
import base64
import html
import json
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from commentary import DISCLAIMER, generate  # noqa: E402
from config import (BASELINE_WINDOW, LGD, OUTPUT_DIR, PRE_WINDOW, REAL_DIR, SAMPLE_DIR, SOURCES_CSV,  # noqa: E402
                    YIELD_TOLERANCE_BPS, Z_THRESHOLD)
from engine.ratings import classify_action  # noqa: E402
from engine.transition import CATEGORIES  # noqa: E402

NAVY, RED, GREY, GOLD = "#1f3864", "#c0392b", "#8c8c8c", "#c9a227"
plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return base64.b64encode(path.read_bytes()).decode()


def fig_event_path(od, figdir, tag):
    pe = pd.read_csv(od / "path_events.csv")
    pp = pd.read_csv(od / "path_placebo.csv")
    fig, ax = plt.subplots(figsize=(8, 4))
    for df, name, col in [(pe, "Downgrade episodes", RED), (pp, "Placebo (random) dates", GREY)]:
        if df.empty:
            continue
        ax.fill_between(df.rel_week, df.ci_lo, df.ci_hi, color=col, alpha=0.15)
        ax.plot(df.rel_week, df.mean_bps, color=col, lw=2, label=name)
    ax.axvline(0, color="k", ls="--", lw=1)
    ax.axvspan(PRE_WINDOW[0] / 7, 0, color=GOLD, alpha=0.12, label="pre-event window")
    ax.set_xlabel("Weeks relative to rating action")
    ax.set_ylabel("Abnormal spread vs baseline (bps)")
    ax.set_title(f"Figure 1: Average abnormal spread around downgrades{tag}")
    ax.legend(frameon=False)
    return _save(fig, figdir / "fig1_event_path.png")


def fig_case(od, ddir, figdir, issuer, tag):
    ab = pd.read_csv(od / "abnormal_spreads.csv", parse_dates=["date"])
    s = ab[ab.issuer == issuer].sort_values("date")
    acts = pd.read_csv(ddir / "rating_actions.csv")
    acts = acts[(acts.issuer == issuer) & acts.rating_from.notna()]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(s.date, s.spread_bps, color=NAVY, lw=1.2, marker=".", ms=3, label=f"{issuer} spread")
    ax.plot(s.date, s.benchmark_bps, color=GREY, ls=":", label="benchmark")
    for _, a in acts.iterrows():
        k = classify_action(a.rating_from, a.rating_to)
        x = pd.Timestamp(a.action_date)
        if k in ("downgrade", "default", "upgrade"):
            col = RED if k != "upgrade" else "green"
            ax.axvline(x, color=col, ls="--", lw=1.2)
            short = lambda r: str(r).replace("CRISIL ", "").replace("CARE ", "").replace("[ICRA]", "")  # noqa: E731
            ax.text(x, ax.get_ylim()[1], f" {short(a.rating_from)}→{short(a.rating_to)}", color=col,
                    fontsize=8, va="top", rotation=90)
        else:                                   # reaffirmations / outlook changes: faint, unlabelled
            ax.axvline(x, color=GREY, ls=":", lw=0.6, alpha=0.5)
    ax.set_ylabel("Spread over matched-tenor G-sec (bps)")
    ax.set_title(f"Figure 2: {issuer} - spread and rating actions{tag}")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    return _save(fig, figdir / "fig2_case_issuer.png")


def fig_matrix(od, figdir, tag):
    p = pd.read_csv(od / "transition_probs.csv", index_col=0)
    c = pd.read_csv(od / "transition_counts.csv", index_col=0)
    rows = [r for r in CATEGORIES if r != "D"]
    z = p.reindex(index=rows, columns=CATEGORIES)
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.imshow(np.nan_to_num(z.values) * 100, cmap="Blues", vmin=0, vmax=100)
    for i, r in enumerate(rows):
        for j, col in enumerate(CATEGORIES):
            v = z.loc[r, col]
            if pd.notna(v) and c.loc[r].sum() > 0:
                ax.text(j, i, f"{100 * v:.1f}", ha="center", va="center", fontsize=8,
                        color="white" if v > 0.6 else "black")
    ax.set_xticks(range(len(CATEGORIES)), CATEGORIES)
    ax.set_yticks(range(len(rows)), [f"{r} (n={int(c.loc[r].sum())})" for r in rows])
    ax.set_xlabel("Rating one year later")
    ax.set_ylabel("Rating at cohort start")
    ax.set_title(f"Figure 3: One-year transition matrix, %{tag}")
    return _save(fig, figdir / "fig3_transition_matrix.png")


def fig_validation(od, figdir, tag):
    sp = pd.read_csv(od / "trade_spreads.csv")
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(sp.yield_diff_bps.dropna().clip(-10, 10), bins=60, color=NAVY)
    ax.axvline(-YIELD_TOLERANCE_BPS, color=RED, ls="--")
    ax.axvline(YIELD_TOLERANCE_BPS, color=RED, ls="--", label=f"±{YIELD_TOLERANCE_BPS:.0f} bps tolerance")
    ax.set_xlabel("Recomputed minus printed yield (bps, clipped ±10)")
    ax.set_ylabel("Trades")
    ax.set_title(f"Figure 4: Validation - recomputed vs exchange-printed yield{tag}")
    ax.legend(frameon=False)
    return _save(fig, figdir / "fig4_validation.png")


def fig_pd(od, figdir, tag):
    imp = pd.read_csv(od / "implied_vs_historical_pd.csv", index_col=0)
    imp = imp.reindex([c for c in CATEGORIES if c in imp.index])
    x = np.arange(len(imp))
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(x - 0.2, imp.implied_pd_1y * 100, 0.4, color=NAVY, label=f"Market-implied (spread / {LGD:.0%} LGD)")
    ax.bar(x + 0.2, imp.historical_pd_1y.fillna(0) * 100, 0.4, color=GOLD, label="Historical (transition matrix)")
    ax.set_xticks(x, imp.index)
    ax.set_ylabel("One-year PD, %")
    ax.set_title(f"Figure 5: Implied vs historical PD by grade{tag}")
    ax.legend(frameon=False)
    return _save(fig, figdir / "fig5_implied_vs_historical_pd.png")


def table_html(df, floatfmt="{:,.2f}", index=True):
    return df.to_html(index=index, float_format=lambda v: floatfmt.format(v), border=0, classes="t", na_rep="-")


def build(mode: str):
    od = OUTPUT_DIR / mode
    ddir = SAMPLE_DIR if mode == "sample" else REAL_DIR
    figdir = od / "figures"
    figdir.mkdir(exist_ok=True)
    tag = " (SYNTHETIC)" if mode == "sample" else ""
    S = json.load(open(od / "summary.json"))
    T, V = S["event_study"], S["validation"]
    comm = generate(mode, use_llm=False)

    ev = pd.read_csv(od / "events.csv")
    flagged = ev[(ev.usable == True) & (ev.flagged == True)]  # noqa: E712
    case_issuer = flagged.sort_values("z_pre", ascending=False).issuer.iloc[0] if len(flagged) else ev.issuer.iloc[0]
    figs = [fig_event_path(od, figdir, tag), fig_case(od, ddir, figdir, case_issuer, tag),
            fig_matrix(od, figdir, tag), fig_validation(od, figdir, tag), fig_pd(od, figdir, tag)]

    tests = pd.DataFrame([(k, v) for k, v in T.items()], columns=["statistic", "value"])
    events_tab = ev[ev.usable == True][["issuer", "agency", "action_date", "rating_from", "rating_to",  # noqa: E712
                                       "dS_pre_bps", "z_pre", "flagged", "lead_days", "dS_post_bps"]]
    pd_tab = pd.read_csv(od / "pd_by_grade.csv", index_col=0) * 100
    tsum = pd.read_csv(od / "transition_summary.csv", index_col=0)
    src = pd.read_csv(SOURCES_CSV).fillna("") if SOURCES_CSV.exists() else pd.DataFrame()

    banner = ('<div class="warn"><b>SYNTHETIC SAMPLE DATA.</b> This annexure demonstrates the method. '
              'None of its numbers is a finding about Indian corporate bonds.</div>' if mode == "sample" else "")
    img = lambda b64: f'<img src="data:image/png;base64,{b64}"/>'  # noqa: E731
    comm_html = "".join(f"<p>{html.escape(p)}</p>" for p in comm["text"].split("\n\n"))

    doc = f"""<!doctype html><html><head><meta charset="utf-8"><title>SpreadWatch Annexure</title>
<style>
body{{font-family:'Times New Roman',serif;font-size:12pt;line-height:1.5;max-width:900px;margin:40px auto;color:#111;padding:0 16px}}
h1,h2{{color:{NAVY}}} h1{{font-size:18pt}} h2{{font-size:14pt;border-bottom:1px solid #ccc;margin-top:28px}}
img{{max-width:100%;margin:8px 0}} .t{{border-collapse:collapse;font-size:9.5pt;margin:8px 0}}
.t td,.t th{{border-bottom:1px solid #ddd;padding:3px 7px;text-align:right}} .t th{{background:#eef1f7}}
.warn{{background:#fff3cd;border-left:4px solid {GOLD};padding:10px;margin:12px 0}}
.small{{font-size:9.5pt;color:#555}}
</style></head><body>
<h1>SpreadWatch - Credit Spread Analyser: Generated Annexure</h1>
<p class="small">Generated {datetime.now():%d %B %Y %H:%M} · dataset: {mode} · period {S['period'][0]} to {S['period'][1]}</p>
{banner}
<h2>A. Research question and answer</h2>
<p><b>Do credit spreads on Indian corporate bonds move before the rating agencies act?</b></p>
<p>Verdict: <b>{T.get('verdict')}</b> - {html.escape(str(T.get('verdict_reason', '')))}</p>
<p class="small">Method: abnormal spread = issuer spread minus the weekly median spread of issuers with no downgrade within
180 days. Baseline days {BASELINE_WINDOW[0]} to {BASELINE_WINDOW[1]}; pre-event window days {PRE_WINDOW[0]} to {PRE_WINDOW[1]};
early-warning breach at {Z_THRESHOLD:.0f} standard deviations above the baseline mean.</p>
{img(figs[0])}
<h3>Table A1: Event-study test statistics</h3>{table_html(tests.astype(str), index=False)}
<h3>Table A2: Sensitivity of the answer to window and threshold choices</h3>{table_html(pd.read_csv(od / "sensitivity.csv").round(4), "{:,.4f}", index=False)}
<h3>Table A3: Measured downgrade episodes</h3>{table_html(events_tab.round(2), index=False)}
<h2>B. Case study</h2>{img(figs[1])}
<h2>C. Rating transition matrix and PD</h2>{img(figs[2])}
<h3>Table C1: Stability, downgrade, upgrade, default and withdrawal rates</h3>{table_html(tsum, "{:,.3f}")}
<h3>Table C2: Probability of default by horizon, %</h3>{table_html(pd_tab)}
{img(figs[4])}
<h2>D. Validation</h2>
<p>{V.get('n_checked', 0):,} trades checked; {100 * float(V.get('pass_rate', 0) or 0):.1f}% within ±{YIELD_TOLERANCE_BPS:.0f} bps of the printed
yield; median absolute difference {float(V.get('median_abs_diff_bps', 0) or 0):.3f} bps; maximum {float(V.get('max_abs_diff_bps', 0) or 0):.3f} bps.
{S['n_trades_rejected']:,} rows were rejected by input validation.</p>
{img(figs[3])}
<h2>E. Generated commentary (verified, every sentence cited)</h2>{comm_html}
<p class="small">Verifier kept {int(comm['verification']['kept'].sum())} of {len(comm['verification'])} sentences.</p>
<h2>F. Data provenance (sources.csv)</h2>{table_html(src, index=False)}
<p class="small"><i>{html.escape(DISCLAIMER)}</i></p>
</body></html>"""
    out = od / "SpreadWatch_Annexure.html"
    out.write_text(doc)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sample", "real"], default="sample")
    print("Written:", build(ap.parse_args().mode))
