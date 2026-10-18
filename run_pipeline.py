"""
Run the full L3 engine on either the SYNTHETIC sample or your real data and
write every output table to outputs/<mode>/.

    python run_pipeline.py --mode sample
    python run_pipeline.py --mode real
"""
import argparse
import json
from datetime import datetime

import numpy as np
import pandas as pd

from config import (CURVE_FILE, OUTPUT_DIR, RATINGS_FILE, REAL_DIR, SAMPLE_DIR, TRADES_FILE,
                    YIELD_TOLERANCE_BPS)
from engine.curve import GsecCurve
from engine.event_study import run_event_study, sensitivity
from engine.ratings import category
from engine.spreads import compute_trade_spreads, daily_issuer_spreads, validation_summary
from engine.transition import (cohort_counts, implied_pd_from_spread, multi_year_pd,
                               probability_matrix, summary_stats)


def data_dir(mode: str):
    return SAMPLE_DIR if mode == "sample" else REAL_DIR


def out_dir(mode: str):
    d = OUTPUT_DIR / mode
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_inputs(mode: str):
    d = data_dir(mode)
    missing = [f for f in (TRADES_FILE, CURVE_FILE, RATINGS_FILE) if not (d / f).exists()]
    if missing:
        raise FileNotFoundError(f"{missing} not found in {d}. "
                                f"{'Run python ingest.py --mode real first.' if mode == 'real' else ''}")
    trades = pd.read_csv(d / TRADES_FILE)
    curve = pd.read_csv(d / CURVE_FILE)
    actions = pd.read_csv(d / RATINGS_FILE, dtype={"rating_from": str, "rating_to": str}).fillna({"rating_from": ""})
    return trades, curve, actions


def rating_on_dates(daily: pd.DataFrame, actions: pd.DataFrame) -> pd.Series:
    """Rating category in force for each issuer-date (worst across agencies)."""
    from engine.ratings import NOTCH, normalise
    a = actions.copy()
    a["action_date"] = pd.to_datetime(a["action_date"])
    a["notch"] = a["rating_to"].map(lambda r: NOTCH.get(normalise(r)))
    out = pd.Series(index=daily.index, dtype=object)
    for issuer, idx in daily.groupby("issuer").groups.items():
        acts = a[a["issuer"] == issuer].dropna(subset=["notch"])
        if acts.empty:
            continue
        dates = pd.to_datetime(daily.loc[idx, "date"])
        cats = []
        for d in dates:
            cur = acts[acts["action_date"] <= d]
            if cur.empty:
                cats.append(None)
                continue
            worst = cur.sort_values("action_date").groupby("agency").tail(1)["rating_to"]
            cats.append(category(max(worst, key=lambda r: NOTCH.get(normalise(r), 0))))
        out.loc[idx] = cats
    return out


def run(mode: str) -> dict:
    trades, curve_df, actions = load_inputs(mode)
    od = out_dir(mode)
    curve = GsecCurve(curve_df)

    # 1. Spreads + validation test
    spreads, rejected = compute_trade_spreads(trades, curve)
    val = validation_summary(spreads)
    daily = daily_issuer_spreads(spreads)
    spreads.to_csv(od / "trade_spreads.csv", index=False)
    rejected.to_csv(od / "rejected_trades.csv", index=False)
    daily.to_csv(od / "daily_spreads.csv", index=False)

    # 2. Event study
    es = run_event_study(daily, actions)
    for k in ("events", "placebos", "path_events", "path_placebo"):
        es[k].to_csv(od / f"{k}.csv", index=False)
    es["abnormal"].to_csv(od / "abnormal_spreads.csv", index=False)
    sens = sensitivity(daily, actions)
    sens.to_csv(od / "sensitivity.csv", index=False)

    # 3. Transition matrix and PDs
    counts = cohort_counts(actions)
    probs = probability_matrix(counts)
    pd_tab = multi_year_pd(probs)
    stats_tab = summary_stats(counts)
    counts.to_csv(od / "transition_counts.csv")
    probs.to_csv(od / "transition_probs.csv")
    pd_tab.to_csv(od / "pd_by_grade.csv")
    stats_tab.to_csv(od / "transition_summary.csv")
    counts.attrs["log"].to_csv(od / "cohort_log.csv", index=False)

    # 4. Market-implied vs historical PD by grade
    daily["category"] = rating_on_dates(daily, actions)
    by_grade = daily.dropna(subset=["category"]).groupby("category")["spread_bps"].agg(["mean", "median", "count"])
    by_grade["implied_pd_1y"] = by_grade["median"].map(implied_pd_from_spread)
    by_grade["historical_pd_1y"] = pd_tab["PD_1y"].reindex(by_grade.index)
    by_grade["implied_to_historical"] = by_grade["implied_pd_1y"] / by_grade["historical_pd_1y"].replace(0, np.nan)
    by_grade.to_csv(od / "implied_vs_historical_pd.csv")

    meta = {
        "mode": mode, "synthetic": mode == "sample",
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "n_trades_input": int(len(trades)), "n_trades_used": int(len(spreads)),
        "n_trades_rejected": int(len(rejected)), "n_issuers": int(daily["issuer"].nunique()),
        "n_rating_actions": int(len(actions)),
        "period": [str(pd.to_datetime(trades["trade_date"]).min().date()),
                   str(pd.to_datetime(trades["trade_date"]).max().date())],
        "cohorts": int(counts.attrs["log"]["cohort"].nunique()) if not counts.attrs["log"].empty else 0,
        "validation": val, "tolerance_bps": YIELD_TOLERANCE_BPS,
        "event_study": es["tests"],
        "sensitivity_verdicts": dict(zip(sens["variant"], sens["verdict"])),
    }
    with open(od / "summary.json", "w") as f:
        json.dump(meta, f, indent=2, default=lambda x: None if pd.isna(x) else str(x))
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["real", "sample"], default="sample")
    args = ap.parse_args()
    m = run(args.mode)
    print(json.dumps(m, indent=2, default=str))
