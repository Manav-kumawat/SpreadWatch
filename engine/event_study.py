"""
Event study: do spreads move BEFORE the rating agency acts?

Event   = a downgrade (or default) by any agency, for an issuer we have spread
          data on. Only the FIRST downgrade of an episode is used: a downgrade
          preceded by another downgrade of the same issuer within
          EVENT_CLUSTER_DAYS is dropped, because its "before" window is
          already contaminated by the earlier action.

Abnormal spread  AS(t) = issuer spread(t) - benchmark(t)
          benchmark(t) = median spread, that week, of issuers with no
          downgrade within +/- EVENT_CLUSTER_DAYS (the "quiet" universe).
          This strips out market-wide moves (e.g. a liquidity squeeze that
          widens every spread) so that what remains is issuer-specific.

Per event (all windows in calendar days relative to the action date, day 0):
  baseline mean / sd   AS over BASELINE_WINDOW  (-180..-91)
  dS_pre               mean AS over PRE_WINDOW (-30..-1)  - baseline mean
  z_pre                dS_pre / baseline sd
  flagged              z_pre >= Z_THRESHOLD   (early-warning signal fired)
  lead_days            first day in DETECTION_WINDOW (-90..-1) where AS breaches
                       baseline mean + Z x sd on two consecutive observations
  dS_post              mean AS over POST_WINDOW (0..30) - mean AS over PRE_WINDOW
  pre_share            dS_pre / (dS_pre + dS_post), capped at 1: share of the total
                       widening that happened before the agency acted

Placebo: the same calculation on random dates for issuers that were never
downgraded. If spreads "widen before downgrades" only as often as they widen
on random dates, the signal is noise.

Hypothesis tests (one-sided, H1: spreads widen before the action):
  * t-test and Wilcoxon signed-rank on dS_pre across events (H0: mean <= 0)
  * Mann-Whitney U: event dS_pre vs placebo dS_pre
  * Fisher exact: event hit rate vs placebo false-alarm rate
"""
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats

from config import (BASELINE_WINDOW, DETECTION_WINDOW, EVENT_CLUSTER_DAYS, MIN_BASELINE_OBS,
                    MIN_PRE_OBS, N_PLACEBO_PER_ISSUER, POST_WINDOW, PRE_WINDOW, RANDOM_SEED,
                    SIGNIFICANCE, Z_THRESHOLD)
from engine.ratings import NOTCH, classify_action, normalise


# ----------------------------------------------------------------------------
# 1. Events
# ----------------------------------------------------------------------------
def select_events(actions: pd.DataFrame, issuers_with_data) -> pd.DataFrame:
    df = actions.copy()
    df["action_date"] = pd.to_datetime(df["action_date"])
    df["action"] = [classify_action(a, b) for a, b in zip(df["rating_from"], df["rating_to"])]
    neg = df[df["action"].isin(["downgrade", "default"])].sort_values("action_date")
    neg = neg[neg["issuer"].isin(set(issuers_with_data))]

    keep = []
    for issuer, g in neg.groupby("issuer"):
        last = None
        for _, r in g.iterrows():
            if last is None or (r["action_date"] - last).days > EVENT_CLUSTER_DAYS:
                keep.append(r)
            last = r["action_date"]          # any downgrade restarts the clock
    ev = pd.DataFrame(keep).reset_index(drop=True)
    if ev.empty:
        return ev
    ev["notches"] = [
        (_n(b) - _n(a)) if _n(a) and _n(b) else np.nan
        for a, b in zip(ev["rating_from"], ev["rating_to"])
    ]
    return ev


def _n(r):
    s = normalise(r)
    return NOTCH.get(s)


def negative_action_dates(actions: pd.DataFrame) -> Dict[str, pd.DatetimeIndex]:
    df = actions.copy()
    df["action_date"] = pd.to_datetime(df["action_date"])
    df["action"] = [classify_action(a, b) for a, b in zip(df["rating_from"], df["rating_to"])]
    neg = df[df["action"].isin(["downgrade", "default"])]
    return {i: pd.DatetimeIndex(g["action_date"]) for i, g in neg.groupby("issuer")}


# ----------------------------------------------------------------------------
# 2. Benchmark and abnormal spreads
# ----------------------------------------------------------------------------
def abnormal_spreads(daily: pd.DataFrame, actions: pd.DataFrame,
                     use_benchmark: bool = True) -> pd.DataFrame:
    df = daily.copy()
    df["date"] = pd.to_datetime(df["date"])
    neg = negative_action_dates(actions)

    def near_action(row_issuer, dates):
        acts = neg.get(row_issuer)
        if acts is None or len(acts) == 0:
            return np.zeros(len(dates), dtype=bool)
        diff = np.abs((dates.values[:, None] - acts.values[None, :]).astype("timedelta64[D]").astype(int))
        return (diff <= EVENT_CLUSTER_DAYS).any(axis=1)

    df["quiet"] = True
    for issuer, idx in df.groupby("issuer").groups.items():
        df.loc[idx, "quiet"] = ~near_action(issuer, df.loc[idx, "date"])

    if use_benchmark:
        q = df[df["quiet"]].copy()
        q["week"] = q["date"].dt.to_period("W-FRI")
        bench = q.groupby("week")["spread_bps"].median()
        df["week"] = df["date"].dt.to_period("W-FRI")
        df["benchmark_bps"] = df["week"].map(bench)
        # carry the last known benchmark forward over thin weeks
        full = bench.reindex(pd.period_range(bench.index.min(), bench.index.max(), freq="W-FRI")).ffill()
        df["benchmark_bps"] = df["benchmark_bps"].fillna(df["week"].map(full))
        df = df.drop(columns="week")
    else:
        df["benchmark_bps"] = 0.0
    df["abnormal_bps"] = df["spread_bps"] - df["benchmark_bps"]
    return df.dropna(subset=["abnormal_bps"])


# ----------------------------------------------------------------------------
# 3. Per-event measurement
# ----------------------------------------------------------------------------
def _window(series: pd.DataFrame, lo: int, hi: int) -> pd.DataFrame:
    return series[(series["rel_day"] >= lo) & (series["rel_day"] <= hi)]


def measure_window(issuer_series: pd.DataFrame, event_date: pd.Timestamp) -> dict:
    s = issuer_series.copy()
    s["rel_day"] = (s["date"] - event_date).dt.days
    base = _window(s, *BASELINE_WINDOW)["abnormal_bps"]
    pre = _window(s, *PRE_WINDOW)["abnormal_bps"]
    post = _window(s, *POST_WINDOW)["abnormal_bps"]
    det = _window(s, *DETECTION_WINDOW).sort_values("rel_day")

    out = {"n_baseline": len(base), "n_pre": len(pre), "n_post": len(post)}
    if len(base) < MIN_BASELINE_OBS or len(pre) < MIN_PRE_OBS:
        out["usable"] = False
        return out

    mu, sd = base.mean(), base.std(ddof=1)
    sd = sd if sd and sd > 1e-9 else np.nan
    d_pre = pre.mean() - mu
    z = d_pre / sd if pd.notna(sd) else np.nan
    threshold = mu + Z_THRESHOLD * sd if pd.notna(sd) else np.nan

    lead = np.nan
    if pd.notna(threshold) and len(det) >= 2:
        above = (det["abnormal_bps"] > threshold).to_numpy()
        for i in range(len(above) - 1):
            if above[i] and above[i + 1]:
                lead = -int(det["rel_day"].iloc[i])
                break

    d_post = post.mean() - pre.mean() if len(post) else np.nan
    total = d_pre + d_post if pd.notna(d_post) else np.nan
    # Capped at 1: above 1 means the spread partly retraced after the action.
    share = min(1.0, d_pre / total) if pd.notna(total) and total > 0 and d_pre > 0 else np.nan

    out.update({
        "usable": True, "baseline_mean_bps": mu, "baseline_sd_bps": sd,
        "pre_mean_bps": pre.mean(), "dS_pre_bps": d_pre, "z_pre": z,
        "flagged": bool(pd.notna(z) and z >= Z_THRESHOLD),
        "lead_days": lead, "dS_post_bps": d_post, "pre_share": share,
    })
    return out


def run_events(abn: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, e in events.iterrows():
        s = abn[abn["issuer"] == e["issuer"]]
        m = measure_window(s, pd.Timestamp(e["action_date"]))
        rows.append({"issuer": e["issuer"], "agency": e.get("agency", ""),
                     "action_date": pd.Timestamp(e["action_date"]).date(),
                     "rating_from": e["rating_from"], "rating_to": e["rating_to"],
                     "notches": e.get("notches", np.nan), "source_id": e.get("source_id", ""), **m})
    return pd.DataFrame(rows)


def run_placebos(abn: pd.DataFrame, actions: pd.DataFrame, n_per_issuer: int = N_PLACEBO_PER_ISSUER,
                 seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    neg = negative_action_dates(actions)
    controls = [i for i in abn["issuer"].unique() if i not in neg]
    rows = []
    for issuer in sorted(controls):
        s = abn[abn["issuer"] == issuer]
        lo = s["date"].min() + pd.Timedelta(days=-BASELINE_WINDOW[0])
        hi = s["date"].max() - pd.Timedelta(days=POST_WINDOW[1])
        if hi <= lo:
            continue
        span = (hi - lo).days
        for k in range(n_per_issuer):
            d = lo + pd.Timedelta(days=int(rng.integers(0, span + 1)))
            m = measure_window(s, d)
            rows.append({"issuer": issuer, "placebo_date": d.date(), **m})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 4. Average path in event time (for the chart)
# ----------------------------------------------------------------------------
def event_time_path(abn: pd.DataFrame, dates: pd.DataFrame, date_col: str,
                    weeks=(-26, 4)) -> pd.DataFrame:
    paths = []
    for _, e in dates.iterrows():
        d0 = pd.Timestamp(e[date_col])
        s = abn[abn["issuer"] == e["issuer"]].copy()
        s["rel_day"] = (s["date"] - d0).dt.days
        base = _window(s, *BASELINE_WINDOW)["abnormal_bps"]
        if len(base) < MIN_BASELINE_OBS:
            continue
        s["rel_week"] = np.floor(s["rel_day"] / 7).astype(int)
        s = s[(s["rel_week"] >= weeks[0]) & (s["rel_week"] <= weeks[1])]
        w = (s.groupby("rel_week")["abnormal_bps"].mean() - base.mean()).rename(len(paths))
        paths.append(w)
    if not paths:
        return pd.DataFrame(columns=["rel_week", "mean_bps", "ci_lo", "ci_hi", "n"])
    mat = pd.concat(paths, axis=1).sort_index()      # union of weeks, in time order
    mean = mat.mean(axis=1)
    n = mat.count(axis=1)
    se = mat.std(axis=1, ddof=1) / np.sqrt(n)
    return pd.DataFrame({"rel_week": mat.index, "mean_bps": mean.values,
                         "ci_lo": (mean - 1.96 * se).values, "ci_hi": (mean + 1.96 * se).values,
                         "n": n.values})


# ----------------------------------------------------------------------------
# 5. Tests and verdict
# ----------------------------------------------------------------------------
def hypothesis_tests(ev: pd.DataFrame, pl: pd.DataFrame) -> dict:
    e = ev[ev.get("usable", False) == True].copy()  # noqa: E712
    p = pl[pl.get("usable", False) == True].copy() if not pl.empty else pl  # noqa: E712
    e["flagged"] = e["flagged"].astype(bool)
    if not p.empty:
        p["flagged"] = p["flagged"].astype(bool)
    res = {"n_events_total": int(len(ev)), "n_events_usable": int(len(e)),
           "n_placebo_usable": int(len(p))}
    if len(e) < 3:
        res["verdict"] = "INSUFFICIENT DATA"
        res["verdict_reason"] = f"only {len(e)} usable events (need at least 3)"
        return res

    x = e["dS_pre_bps"].astype(float)
    res["mean_dS_pre_bps"] = float(x.mean())
    res["median_dS_pre_bps"] = float(x.median())
    res["share_events_widened"] = float((x > 0).mean())
    t = stats.ttest_1samp(x, 0.0, alternative="greater")
    res["t_stat"], res["t_pvalue"] = float(t.statistic), float(t.pvalue)
    try:
        w = stats.wilcoxon(x, alternative="greater")
        res["wilcoxon_pvalue"] = float(w.pvalue)
    except ValueError:
        res["wilcoxon_pvalue"] = np.nan

    res["hit_rate"] = float(e["flagged"].mean())
    lead = e.loc[e["flagged"] & e["lead_days"].notna(), "lead_days"]
    res["median_lead_days"] = float(lead.median()) if len(lead) else np.nan
    res["median_pre_share"] = float(e["pre_share"].dropna().median()) if e["pre_share"].notna().any() else np.nan

    if len(p) >= 3:
        y = p["dS_pre_bps"].astype(float)
        res["placebo_mean_dS_pre_bps"] = float(y.mean())
        res["false_alarm_rate"] = float(p["flagged"].mean())
        mw = stats.mannwhitneyu(x, y, alternative="greater")
        res["mannwhitney_pvalue"] = float(mw.pvalue)
        table = [[int(e["flagged"].sum()), int((~e["flagged"]).sum())],
                 [int(p["flagged"].sum()), int((~p["flagged"]).sum())]]
        res["fisher_pvalue"] = float(stats.fisher_exact(table, alternative="greater")[1])

    sig_level = res["t_pvalue"] < SIGNIFICANCE
    sig_vs_placebo = res.get("fisher_pvalue", 1.0) < SIGNIFICANCE and res.get("mannwhitney_pvalue", 1.0) < SIGNIFICANCE
    if sig_level and sig_vs_placebo:
        res["verdict"] = "YES"
        res["verdict_reason"] = ("spreads widened significantly in the 30 days before downgrades, "
                                 "and more often than on random dates")
    elif sig_level or sig_vs_placebo:
        res["verdict"] = "PARTIAL"
        res["verdict_reason"] = "one test is significant but not both - the evidence is mixed"
    else:
        res["verdict"] = "NO"
        res["verdict_reason"] = "pre-downgrade widening is not distinguishable from zero or from random dates"
    return res


def run_event_study(daily: pd.DataFrame, actions: pd.DataFrame, use_benchmark: bool = True) -> dict:
    abn = abnormal_spreads(daily, actions, use_benchmark=use_benchmark)
    events = select_events(actions, abn["issuer"].unique())
    ev = run_events(abn, events) if not events.empty else pd.DataFrame()
    pl = run_placebos(abn, actions)
    tests = hypothesis_tests(ev, pl) if not ev.empty else {"verdict": "INSUFFICIENT DATA",
                                                         "verdict_reason": "no downgrade events with spread data"}
    usable_ev = ev[ev["usable"] == True] if not ev.empty else ev  # noqa: E712
    usable_pl = pl[pl["usable"] == True] if not pl.empty else pl  # noqa: E712
    return {
        "abnormal": abn, "events": ev, "placebos": pl, "tests": tests,
        "path_events": event_time_path(abn, usable_ev.rename(columns={"action_date": "d"}), "d")
        if not usable_ev.empty else pd.DataFrame(),
        "path_placebo": event_time_path(abn, usable_pl.rename(columns={"placebo_date": "d"}), "d")
        if not usable_pl.empty else pd.DataFrame(),
    }


# ----------------------------------------------------------------------------
# 6. Sensitivity of the answer to the window / threshold choices
# ----------------------------------------------------------------------------
SENSITIVITY_VARIANTS = [
    ("base case", {}),
    ("pre-window 60 days", {"PRE_WINDOW": (-60, -1)}),
    ("pre-window 14 days", {"PRE_WINDOW": (-14, -1)}),
    ("threshold 1.5 sd", {"Z_THRESHOLD": 1.5}),
    ("threshold 2.5 sd", {"Z_THRESHOLD": 2.5}),
    ("no benchmark (raw spread)", {"_use_benchmark": False}),
]


def sensitivity(daily: pd.DataFrame, actions: pd.DataFrame, variants=SENSITIVITY_VARIANTS) -> pd.DataFrame:
    """Re-run the whole study with one setting changed at a time; settings are restored afterwards."""
    g = globals()
    rows = []
    for name, changes in variants:
        saved = {k: g[k] for k in changes if not k.startswith("_")}
        try:
            g.update({k: v for k, v in changes.items() if not k.startswith("_")})
            t = run_event_study(daily, actions, use_benchmark=changes.get("_use_benchmark", True))["tests"]
        finally:
            g.update(saved)
        rows.append({"variant": name, "usable_events": t.get("n_events_usable"),
                     "mean_dS_pre_bps": t.get("mean_dS_pre_bps"), "t_pvalue": t.get("t_pvalue"),
                     "hit_rate": t.get("hit_rate"), "false_alarm_rate": t.get("false_alarm_rate"),
                     "verdict": t.get("verdict")})
    return pd.DataFrame(rows)
