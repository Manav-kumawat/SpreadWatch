"""
L5 Output - SpreadWatch application.

    streamlit run app.py

Runs fully offline once outputs/<mode>/ exists (python run_pipeline.py --mode ...).
"""
import json
from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import OUTPUT_DIR, SAMPLE_DIR, REAL_DIR, SOURCES_CSV, CURVE_FILE, YIELD_TOLERANCE_BPS
from commentary import DISCLAIMER, generate
from engine.bond_math import Bond, accrued_interest, macaulay_duration, residual_tenor_years, yield_from_clean_price
from engine.curve import CurveError, GsecCurve, to_annual_effective
from engine.spreads import TradeValidationError, compute_trade_spreads
from engine.transition import CATEGORIES

st.set_page_config(page_title="SpreadWatch", page_icon="📉", layout="wide")

NAVY, RED, GREY, GOLD = "#1f3864", "#c0392b", "#8c8c8c", "#c9a227"


# ----------------------------------------------------------------------------
@st.cache_data
def load(mode: str, stamp: float):
    od = OUTPUT_DIR / mode
    r = lambda n, **k: pd.read_csv(od / n, **k) if (od / n).exists() else pd.DataFrame()  # noqa: E731
    return {
        "summary": json.load(open(od / "summary.json")),
        "spreads": r("trade_spreads.csv", parse_dates=["trade_date", "maturity_date"]),
        "rejected": r("rejected_trades.csv"),
        "abnormal": r("abnormal_spreads.csv", parse_dates=["date"]),
        "events": r("events.csv"),
        "placebos": r("placebos.csv"),
        "path_events": r("path_events.csv"),
        "path_placebo": r("path_placebo.csv"),
        "counts": r("transition_counts.csv", index_col=0),
        "probs": r("transition_probs.csv", index_col=0),
        "pd": r("pd_by_grade.csv", index_col=0),
        "tsum": r("transition_summary.csv", index_col=0),
        "implied": r("implied_vs_historical_pd.csv", index_col=0),
    }


def data_dir(mode):
    return SAMPLE_DIR if mode == "sample" else REAL_DIR


@st.cache_resource
def curve_for(mode: str):
    return GsecCurve(pd.read_csv(data_dir(mode) / CURVE_FILE))


def bad_demo(mode: str) -> pd.DataFrame:
    """Six trades from the current dataset; rows 1-5 are broken on purpose, row 6 is left intact."""
    if mode == "sample":
        return pd.read_csv(SAMPLE_DIR / "bad_trades_demo.csv")
    t = pd.read_csv(data_dir(mode) / "trades.csv")
    ok = pd.read_csv(OUTPUT_DIR / mode / "trade_spreads.csv")
    ok = ok[ok["yield_check"] == "PASS"]
    passed = set(zip(ok["isin"], pd.to_datetime(ok["trade_date"]).dt.date.astype(str)))
    t = t[[(i, str(d)) in passed for i, d in zip(t["isin"], pd.to_datetime(t["trade_date"]).dt.date)]]
    t = t.sample(6, random_state=1).reset_index(drop=True)     # all six passed validation before breaking
    t.loc[0, "maturity_date"] = "2010-01-01"                                  # matures before it trades
    t.loc[1, "price"] = float(t.loc[1, "price"]) * 100                        # price per 10,000 instead of per 100
    t.loc[2, "printed_yield_pct"] = float(t.loc[2, "printed_yield_pct"]) + 1.5  # printed yield 150 bps off
    t.loc[3, "coupon_pct"] = -1                                               # impossible coupon
    t.loc[4, "trade_date"] = "2031-02-30"                                     # not a date
    return t


def fmt(x, nd=1, suffix=""):
    try:
        return f"{float(x):,.{nd}f}{suffix}"
    except (TypeError, ValueError):
        return "n/a"


# ----------------------------------------------------------------------------
modes = [m for m in ("real", "sample") if (OUTPUT_DIR / m / "summary.json").exists()]
if not modes:
    st.error("No engine output found. Run:  python run_pipeline.py --mode sample")
    st.stop()

with st.sidebar:
    st.title("SpreadWatch")
    st.caption("Credit Spread & Rating Migration Analyser")
    mode = st.radio("Dataset", modes, format_func=lambda m: "Real data" if m == "real" else "SYNTHETIC sample")
    if st.button("Re-run engine on this dataset"):
        from run_pipeline import run
        with st.spinner("Running L3 engine..."):
            run(mode)
        st.cache_data.clear()
        st.cache_resource.clear()
    st.divider()
    st.caption(DISCLAIMER)

D = load(mode, (OUTPUT_DIR / mode / "summary.json").stat().st_mtime)
S, T = D["summary"], D["summary"]["event_study"]

if mode == "sample":
    st.warning("**SYNTHETIC SAMPLE DATA.** These numbers demonstrate that the method works. "
               "They are not findings about Indian bonds and must not appear as results in the report.")

st.markdown("### Do credit spreads on Indian corporate bonds move before the rating agencies act?")

tabs = st.tabs(["1 · Answer", "2 · Issuer spreads", "3 · Transition matrix", "4 · Validation",
                "5 · Bond calculator", "6 · Commentary", "7 · Provenance"])

# ---------------------------------------------------------------- 1. Answer
with tabs[0]:
    verdict = T.get("verdict", "n/a")
    colour = {"YES": "green", "NO": "red", "PARTIAL": "orange"}.get(verdict, "gray")
    st.markdown(f"## Verdict: :{colour}[{verdict}]")
    st.caption(T.get("verdict_reason", ""))
    c = st.columns(5)
    c[0].metric("Usable downgrade episodes", f"{T.get('n_events_usable', 0)} / {T.get('n_events_total', 0)}")
    c[1].metric("Mean pre-event widening", fmt(T.get("mean_dS_pre_bps"), 1, " bps"),
                help="Mean abnormal spread in days -30..-1 minus the baseline (days -180..-91)")
    c[2].metric("t-test p-value (one-sided)", fmt(T.get("t_pvalue"), 4))
    c[3].metric("Hit rate vs false alarms", f"{fmt(100 * float(T.get('hit_rate', 0) or 0), 0)}% vs "
                f"{fmt(100 * float(T.get('false_alarm_rate', 0) or 0), 0)}%",
                help="Share of downgrades where the 2-sd rule fired, vs share of random placebo dates")
    c[4].metric("Median lead time", fmt(T.get("median_lead_days"), 0, " days"))

    pe, pp = D["path_events"], D["path_placebo"]
    if not pe.empty:
        fig = go.Figure()
        for df, name, col in [(pe, "Downgrade episodes", RED), (pp, "Placebo dates", GREY)]:
            if df.empty:
                continue
            fig.add_trace(go.Scatter(x=list(df.rel_week) + list(df.rel_week[::-1]),
                                     y=list(df.ci_hi) + list(df.ci_lo[::-1]), fill="toself",
                                     line=dict(width=0), fillcolor=col, opacity=0.15, showlegend=False,
                                     hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=df.rel_week, y=df.mean_bps, name=name, line=dict(color=col, width=3)))
        fig.add_vline(x=0, line_dash="dash", annotation_text="rating action")
        fig.add_vrect(x0=-30 / 7, x1=0, fillcolor=GOLD, opacity=0.08, line_width=0,
                      annotation_text="pre-event window", annotation_position="top left")
        fig.update_layout(title="Average abnormal spread vs its own baseline, in weeks around the event (95% CI)",
                          xaxis_title="weeks relative to the rating action", yaxis_title="bps above baseline",
                          height=430, legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, width="stretch")
    sens_path = OUTPUT_DIR / mode / "sensitivity.csv"
    if sens_path.exists():
        st.markdown("#### Does the answer survive other reasonable settings?")
        st.dataframe(pd.read_csv(sens_path).round(4), width="stretch", hide_index=True)
    with st.expander("All test statistics"):
        st.json(T)
    ev = D["events"]
    if not ev.empty:
        st.markdown("#### Downgrade episodes")
        show = ev[["issuer", "agency", "action_date", "rating_from", "rating_to", "usable", "dS_pre_bps",
                   "z_pre", "flagged", "lead_days", "dS_post_bps", "pre_share"]]
        st.dataframe(show.round(2), width="stretch", hide_index=True)

# ---------------------------------------------------------------- 2. Issuer
with tabs[1]:
    ab, ev = D["abnormal"], D["events"]
    issuers = sorted(ab["issuer"].unique())
    flagged = ev[(ev.get("flagged") == True)] if not ev.empty else ev  # noqa: E712
    default_issuer = flagged.sort_values("z_pre", ascending=False)["issuer"].iloc[0] if not flagged.empty else issuers[0]
    who = st.selectbox("Issuer", issuers, index=issuers.index(default_issuer))
    s = ab[ab["issuer"] == who].sort_values("date")
    acts = pd.read_csv(data_dir(mode) / "rating_actions.csv")
    acts = acts[acts["issuer"] == who]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s["date"], y=s["spread_bps"], mode="lines+markers", marker=dict(size=4),
                             name=f"{who} spread", line=dict(color=NAVY)))
    fig.add_trace(go.Scatter(x=s["date"], y=s["benchmark_bps"], name="benchmark (quiet issuers, weekly median)",
                             line=dict(color=GREY, dash="dot")))
    for _, a in acts.iterrows():
        if pd.isna(a["rating_from"]) or not str(a["rating_from"]).strip():
            continue
        from engine.ratings import classify_action
        kind = classify_action(a["rating_from"], a["rating_to"])
        if kind not in ("downgrade", "default", "upgrade"):     # reaffirmations: faint, no label
            fig.add_vline(x=pd.Timestamp(a["action_date"]), line_color=GREY, line_dash="dot", line_width=1, opacity=0.4)
            continue
        col = RED if kind != "upgrade" else "green"
        fig.add_vline(x=pd.Timestamp(a["action_date"]), line_color=col, line_dash="dash")
        fig.add_annotation(x=pd.Timestamp(a["action_date"]), y=1.02, yref="paper", showarrow=False,
                           text=f"{a['rating_from']}→{a['rating_to']}", font=dict(color=col, size=11))
    fig.update_layout(title=f"Spread over matched-tenor G-sec - {who}", yaxis_title="bps", height=460)
    st.plotly_chart(fig, width="stretch")
    st.caption("Red dashed lines = downgrades, green = upgrades, faint dotted = reaffirmations. Each point is the value-weighted spread of that day's trades.")
    st.markdown("**Rating actions**")
    st.dataframe(acts, width="stretch", hide_index=True)
    st.markdown("**Measured episodes for this issuer**")
    st.dataframe(ev[ev["issuer"] == who].round(2) if not ev.empty else ev, width="stretch", hide_index=True)

# ---------------------------------------------------------------- 3. Transition matrix
with tabs[2]:
    probs, counts = D["probs"], D["counts"]
    st.markdown(f"One-year cohort transition matrix ({S.get('cohorts', 0)} annual cohorts formed on 1 April, "
                "withdrawal-adjusted)")
    z = probs.reindex(index=[c for c in CATEGORIES if c != "D"], columns=CATEGORIES)
    txt = [[("" if pd.isna(v) else f"{100 * v:.1f}%<br>n={int(counts.loc[r, c])}") for c, v in row.items()]
           for r, row in z.iterrows()]
    fig = go.Figure(go.Heatmap(z=z.values * 100, x=z.columns, y=z.index, text=txt, texttemplate="%{text}",
                               colorscale="Blues", zmin=0, zmax=100, colorbar=dict(title="%")))
    fig.update_layout(xaxis_title="rating one year later", yaxis_title="rating at cohort start",
                      yaxis_autorange="reversed", height=480)
    st.plotly_chart(fig, width="stretch")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Stability, downgrade, default and withdrawal rates**")
        st.dataframe((D["tsum"] * 1).round(3), width="stretch")
    with c2:
        st.markdown("**Probability of default by horizon (matrix powers)**")
        st.dataframe((D["pd"] * 100).round(2).astype(str) + "%", width="stretch")
    imp = D["implied"]
    if not imp.empty:
        imp = imp.reindex([c for c in CATEGORIES if c in imp.index])
        fig = go.Figure([go.Bar(x=imp.index, y=imp["implied_pd_1y"] * 100, name="Market-implied (spread / LGD)",
                                marker_color=NAVY),
                         go.Bar(x=imp.index, y=imp["historical_pd_1y"] * 100, name="Historical (transition matrix)",
                                marker_color=GOLD)])
        fig.update_layout(barmode="group", title="One-year PD: what spreads imply vs what actually happened",
                          yaxis_title="%", height=400)
        st.plotly_chart(fig, width="stretch")
        st.caption("The gap between the bars is the part of the spread that compensates for things other than "
                   "expected default loss - liquidity, risk aversion, uncertainty about the rating itself.")

# ---------------------------------------------------------------- 4. Validation
with tabs[3]:
    v = S["validation"]
    st.markdown("**Validation test:** every trade's yield is recomputed from its traded price and compared with the "
                "yield printed in the exchange's debt segment data.")
    c = st.columns(4)
    c[0].metric("Trades checked", f"{v.get('n_checked', 0):,}")
    c[1].metric(f"Within ±{YIELD_TOLERANCE_BPS:.0f} bps", fmt(100 * float(v.get("pass_rate", 0) or 0), 1, "%"))
    c[2].metric("Median |difference|", fmt(v.get("median_abs_diff_bps"), 3, " bps"))
    c[3].metric("Max |difference|", fmt(v.get("max_abs_diff_bps"), 3, " bps"))
    sp = D["spreads"]
    if not sp.empty and sp["yield_diff_bps"].notna().any():
        fig = go.Figure(go.Histogram(x=sp["yield_diff_bps"].clip(-10, 10), nbinsx=80, marker_color=NAVY))
        fig.update_layout(title="Recomputed minus printed yield (bps, clipped at ±10)", height=320)
        st.plotly_chart(fig, width="stretch")
    st.markdown(f"**Rejected rows in this dataset:** {len(D['rejected'])}")
    if len(D["rejected"]):
        st.dataframe(D["rejected"], width="stretch", hide_index=True)

    st.divider()
    st.markdown("#### Check a trade file  (live failure demonstration)")
    up = st.file_uploader("Upload a trade CSV in the canonical format", type="csv")
    use_bad = st.button("…or load the deliberately broken demo file")
    df_in = pd.read_csv(up) if up is not None else (bad_demo(mode) if use_bad else None)
    if df_in is not None:
        st.dataframe(df_in, width="stretch", hide_index=True)
        try:
            ok, rej = compute_trade_spreads(df_in, curve_for(mode))
            if len(rej):
                st.error(f"{len(rej)} row(s) REFUSED")
                st.dataframe(rej[["trade_date", "isin", "price", "coupon_pct", "maturity_date", "reject_reason"]],
                             width="stretch", hide_index=True)
            fails = ok[ok["yield_check"] == "FAIL"] if not ok.empty else ok
            if len(fails):
                st.error(f"{len(fails)} row(s) FAILED the printed-yield check - the printed yield is not used")
                st.dataframe(fails[["trade_date", "isin", "price", "printed_yield_pct", "computed_yield_pct",
                                    "yield_diff_bps"]].round(4), width="stretch", hide_index=True)
            good = ok[ok["yield_check"] == "PASS"] if not ok.empty else ok
            if len(good):
                st.success(f"{len(good)} row(s) accepted")
                st.dataframe(good[["trade_date", "isin", "computed_yield_pct", "printed_yield_pct",
                                   "gsec_yield_used_pct", "spread_bps"]].round(4), width="stretch", hide_index=True)
        except TradeValidationError as e:
            st.error(f"File refused: {e}")

# ---------------------------------------------------------------- 5. Calculator
with tabs[4]:
    st.markdown("Price → yield → spread for one bond, using the same engine functions as the batch run.")
    c = st.columns(5)
    cpn = c[0].number_input("Coupon %", value=8.50, step=0.05)
    mat = c[1].date_input("Maturity", value=date(2029, 3, 31), min_value=date(2000, 1, 1), max_value=date(2070, 1, 1))
    curve = curve_for(mode)
    last_curve = pd.Timestamp(curve.dates[-1]).date()
    td = c[2].date_input("Trade date", value=last_curve, min_value=date(2000, 1, 1), max_value=date(2070, 1, 1))
    px = c[3].number_input("Clean price (per 100)", value=99.50, step=0.05)
    freq = c[4].selectbox("Coupons / year", [1, 2, 4, 12])
    try:
        b = Bond(cpn, mat, freq)
        y = yield_from_clean_price(b, td, px)
        ten = residual_tenor_years(mat, td)
        g, ex = curve.matched_yield(td, ten)
        g_eff = to_annual_effective(g)
        r = st.columns(5)
        r[0].metric("Yield (XIRR, annual)", f"{y:.4f}%")
        r[1].metric("Accrued interest", f"{accrued_interest(b, td):.4f}")
        r[2].metric("Residual tenor", f"{ten:.2f} y")
        r[3].metric("Matched G-sec (annual eff.)", f"{g_eff:.4f}%", help=f"quoted semi-annual {g:.4f}%" +
                    (" - EXTRAPOLATED beyond the curve" if ex else ""))
        r[4].metric("Spread", f"{(y - g_eff) * 100:.1f} bps")
        st.caption(f"Macaulay duration {macaulay_duration(b, td, y):.2f} years. Curve used: latest on or before {td}.")
    except (ValueError, CurveError) as e:
        st.error(f"Input refused: {e}")

# ---------------------------------------------------------------- 6. Commentary
with tabs[5]:
    use_llm = st.toggle("Let Claude write the text (needs ANTHROPIC_API_KEY; falls back to template offline)", value=False)
    if st.button("Generate commentary"):
        with st.spinner("Writing and verifying..."):
            st.session_state["comm"] = generate(mode, use_llm)
    comm = st.session_state.get("comm")
    if comm and comm["mode"] == mode:
        if comm["note"]:
            st.info(comm["note"])
        st.caption(f"Writer: {comm['writer']}. Every sentence carries a citation: [E:...] engine fact, "
                   "[S:...] sources.csv row, [R:...] rationale paragraph.")
        st.markdown(comm["text"])
        st.caption(comm["disclaimer"])
        log = comm["verification"]
        st.markdown(f"**Verifier:** kept {int(log['kept'].sum())} of {len(log)} sentences")
        if (~log["kept"]).any():
            st.dataframe(log[~log["kept"]], width="stretch", hide_index=True)
        with st.expander("Facts the writer was allowed to use"):
            st.json(comm["facts"])
        with st.expander("Retrieved rationale paragraphs (L2)"):
            for h in comm["rag_hits"]:
                st.markdown(f"**{h['chunk_id']}** (score {h['score']:.2f})  \n{h['text']}")

# ---------------------------------------------------------------- 7. Provenance
with tabs[6]:
    st.markdown("**sources.csv** - every input, with publisher, URL and date pulled")
    st.dataframe(pd.read_csv(SOURCES_CSV) if SOURCES_CSV.exists() else pd.DataFrame(), width="stretch",
                 hide_index=True)
    lg = OUTPUT_DIR / "ingest_log.csv"
    if lg.exists() and mode == "real":
        st.markdown("**Ingest log** - file hashes and row counts")
        st.dataframe(pd.read_csv(lg), width="stretch", hide_index=True)
    st.markdown("**This run**")
    st.json({k: v for k, v in S.items() if k not in ("event_study", "validation")})
