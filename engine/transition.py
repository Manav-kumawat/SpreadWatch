"""
Rating transition matrix - cohort method.

Method (the same approach Indian CRAs use in their SEBI-mandated
transition/default studies):

  1. Each (issuer, agency) pair is one rated unit. Its rating on any date is
     the rating set by its most recent action on or before that date.
  2. A cohort is formed on 1 April of each year: every unit holding a live
     rating (not D, not withdrawn) on that date.
  3. One year later we record where each unit ended up. If it hit D at any
     time during the year it counts as D, even if it was later upgraded.
  4. Count matrix N[i, j] = number of units starting in category i and ending
     in category j. Pool all cohorts.
  5. Probability P[i, j] = N[i, j] / (row total excluding withdrawals)
     ("withdrawal-adjusted"); withdrawals are reported separately.

From P we also get:
  * 1-year PD by grade  = P[i, D]
  * n-year PD by grade  = (P^n)[i, D]   (assumes a time-homogeneous Markov chain)
  * market-implied PD from a spread s via the credit triangle:
        hazard = s / LGD ;  PD(1y) = 1 - exp(-hazard)
"""
from datetime import date
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config import COHORT_MONTH_DAY, HORIZON_YEARS, LGD
from engine.ratings import CATEGORIES, WITHDRAWN, category

LIVE = [c for c in CATEGORIES if c != "D"]           # rows: AAA..C
COLS = CATEGORIES + [WITHDRAWN]                       # columns: AAA..C, D, WR


def build_histories(actions: pd.DataFrame) -> Dict[tuple, pd.DataFrame]:
    df = actions.copy()
    df["action_date"] = pd.to_datetime(df["action_date"])
    df["cat_to"] = df["rating_to"].map(category)
    df = df.dropna(subset=["cat_to"]).sort_values("action_date")
    return {k: g[["action_date", "cat_to"]].reset_index(drop=True)
            for k, g in df.groupby(["issuer", "agency"])}


def _rating_on(hist: pd.DataFrame, when: pd.Timestamp) -> Optional[str]:
    h = hist[hist["action_date"] <= when]
    return None if h.empty else h["cat_to"].iloc[-1]


def cohort_counts(actions: pd.DataFrame, start_year: Optional[int] = None,
                  end_date=None) -> pd.DataFrame:
    """Pooled count matrix (rows LIVE categories, columns COLS) plus a per-cohort log."""
    hists = build_histories(actions)
    all_dates = pd.to_datetime(actions["action_date"])
    end_date = pd.Timestamp(end_date) if end_date is not None else all_dates.max()
    first_year = start_year or all_dates.min().year
    m, d = COHORT_MONTH_DAY

    counts = pd.DataFrame(0, index=LIVE, columns=COLS, dtype=int)
    log: List[dict] = []
    year = first_year
    while True:
        c_start = pd.Timestamp(date(year, m, d))
        c_end = pd.Timestamp(date(year + HORIZON_YEARS, m, d))
        if c_end > end_date:
            break
        for unit, h in hists.items():
            start_cat = _rating_on(h, c_start)
            if start_cat is None or start_cat in ("D", WITHDRAWN):
                continue
            during = h[(h["action_date"] > c_start) & (h["action_date"] <= c_end)]
            if (during["cat_to"] == "D").any():
                end_cat = "D"
            else:
                end_cat = _rating_on(h, c_end)
            counts.loc[start_cat, end_cat] += 1
            log.append({"cohort": c_start.date(), "issuer": unit[0], "agency": unit[1],
                        "from": start_cat, "to": end_cat})
        year += 1
    counts.attrs["log"] = pd.DataFrame(log)
    return counts


def probability_matrix(counts: pd.DataFrame) -> pd.DataFrame:
    """Withdrawal-adjusted row probabilities over CATEGORIES (AAA..C, D). Empty rows -> NaN."""
    core = counts[CATEGORIES].astype(float)
    totals = core.sum(axis=1)
    return core.div(totals.replace(0, np.nan), axis=0)


def square_matrix(prob: pd.DataFrame) -> pd.DataFrame:
    """Add the absorbing D row (D stays D) and fill empty rows with 'stay put' so powers exist."""
    sq = prob.reindex(index=CATEGORIES, columns=CATEGORIES).copy()
    sq.loc["D"] = 0.0
    sq.loc["D", "D"] = 1.0
    for r in CATEGORIES:
        if sq.loc[r].isna().all():
            sq.loc[r] = 0.0
            sq.loc[r, r] = 1.0
    return sq.fillna(0.0)


def multi_year_pd(prob: pd.DataFrame, years=(1, 2, 3, 5)) -> pd.DataFrame:
    sq = square_matrix(prob).to_numpy()
    out = {}
    pn = np.eye(len(sq))
    # P^n by repeated multiplication. errstate silences a spurious numpy-2/macOS
    # Accelerate warning; the values are finite and are checked in the tests.
    with np.errstate(all="ignore"):
        for n in range(1, max(years) + 1):
            pn = pn @ sq
            if n in years:
                out[f"PD_{n}y"] = pn[:, CATEGORIES.index("D")].copy()
    res = pd.DataFrame(out, index=CATEGORIES).loc[LIVE]
    observed = prob.notna().any(axis=1).reindex(LIVE)
    res.loc[~observed] = np.nan
    return res


def summary_stats(counts: pd.DataFrame) -> pd.DataFrame:
    """Per-grade stability, downgrade, upgrade, default and withdrawal rates."""
    rows = []
    for i, r in enumerate(LIVE):
        n_total = counts.loc[r].sum()
        n_adj = counts.loc[r, CATEGORIES].sum()
        if n_total == 0:
            rows.append({"grade": r, "units": 0})
            continue
        down = counts.loc[r, CATEGORIES[i + 1:]].sum()
        up = counts.loc[r, CATEGORIES[:i]].sum()
        rows.append({
            "grade": r, "units": int(n_total),
            "stable": counts.loc[r, r] / n_adj if n_adj else np.nan,
            "downgrade": down / n_adj if n_adj else np.nan,
            "upgrade": up / n_adj if n_adj else np.nan,
            "default": counts.loc[r, "D"] / n_adj if n_adj else np.nan,
            "withdrawn_share": counts.loc[r, WITHDRAWN] / n_total,
        })
    return pd.DataFrame(rows).set_index("grade")


def implied_pd_from_spread(spread_bps: float, lgd: float = LGD, years: float = 1.0) -> float:
    """Credit triangle: spread = hazard x LGD  ->  PD(t) = 1 - exp(-hazard x t)."""
    hazard = max(spread_bps, 0.0) / 10000.0 / lgd
    return 1.0 - float(np.exp(-hazard * years))
