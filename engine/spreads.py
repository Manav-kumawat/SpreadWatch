"""
Credit spread engine.

For every reported corporate bond trade:

  1. Validate the row (dates, price range, coupon, maturity after trade date).
  2. Recompute the yield from the traded clean price          (bond_math)
  3. Compare it with the yield printed by the exchange        (VALIDATION TEST)
  4. Read the matched-tenor G-sec yield off that day's curve  (curve)
  5. Spread (bps) = corporate yield - matched G-sec yield, both on an annual
     effective basis, x 100.

Then collapse trades into one spread per issuer per day (value-weighted).
"""
from typing import Tuple

import numpy as np
import pandas as pd

from config import DEFAULT_COUPON_FREQ, SPREAD_BASIS, YIELD_TOLERANCE_BPS
from engine.bond_math import Bond, residual_tenor_years, yield_from_clean_price
from engine.curve import CurveError, GsecCurve, to_annual_effective

REQUIRED_TRADE_COLUMNS = ["trade_date", "isin", "issuer", "coupon_pct", "maturity_date", "price"]


class TradeValidationError(ValueError):
    pass


def validate_trades(trades: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split trades into (accepted, rejected). Every rejected row carries a reason,
    so a bad input is refused loudly rather than silently producing a number.
    """
    missing = [c for c in REQUIRED_TRADE_COLUMNS if c not in trades.columns]
    if missing:
        raise TradeValidationError(f"trade file is missing required columns: {missing}")

    df = trades.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    df["maturity_date"] = pd.to_datetime(df["maturity_date"], errors="coerce")
    for c in ["coupon_pct", "price", "printed_yield_pct", "value_cr", "coupon_freq"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "coupon_freq" not in df.columns:
        df["coupon_freq"] = DEFAULT_COUPON_FREQ
    df["coupon_freq"] = df["coupon_freq"].fillna(DEFAULT_COUPON_FREQ).astype(int)

    reasons = pd.Series("", index=df.index)

    def flag(mask, why):
        reasons[mask & (reasons == "")] = why

    flag(df["trade_date"].isna(), "unreadable trade_date")
    flag(df["maturity_date"].isna(), "unreadable maturity_date")
    flag(df["maturity_date"] <= df["trade_date"], "maturity on/before trade date")
    flag(df["price"].isna() | (df["price"] <= 0), "missing or non-positive price")
    flag((df["price"] < 20) | (df["price"] > 200), "price outside 20-200 per 100 face (likely unit error)")
    flag(df["coupon_pct"].isna() | (df["coupon_pct"] < 0) | (df["coupon_pct"] > 30), "coupon missing or outside 0-30%")
    flag(~df["coupon_freq"].isin([1, 2, 4, 12]), "unsupported coupon frequency")
    if "printed_yield_pct" in df.columns:
        flag((df["printed_yield_pct"] < -5) | (df["printed_yield_pct"] > 60), "printed yield outside -5%..60%")

    rejected = df[reasons != ""].assign(reject_reason=reasons[reasons != ""])
    accepted = df[reasons == ""].copy()
    return accepted, rejected


def compute_trade_spreads(trades: pd.DataFrame, curve: GsecCurve,
                          tolerance_bps: float = YIELD_TOLERANCE_BPS) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (spreads, rejected). `spreads` has one row per accepted trade with
    the recomputed yield, the validation result and the spread in bps.
    """
    accepted, rejected = validate_trades(trades)
    rows = []
    extra_rejects = []
    for idx, r in accepted.iterrows():
        bond = Bond(coupon_pct=float(r["coupon_pct"]), maturity=r["maturity_date"].date(),
                    freq=int(r["coupon_freq"]))
        settle = r["trade_date"]
        try:
            computed = yield_from_clean_price(bond, settle, float(r["price"]))
        except ValueError as e:
            extra_rejects.append({**r.to_dict(), "reject_reason": f"yield solve failed: {e}"})
            continue
        tenor = residual_tenor_years(bond.maturity, settle)
        try:
            g_quoted, extrap = curve.matched_yield(settle, tenor)
        except CurveError as e:
            extra_rejects.append({**r.to_dict(), "reject_reason": str(e)})
            continue

        printed = r.get("printed_yield_pct", np.nan)
        if pd.notna(printed):
            diff_bps = (computed - printed) * 100.0
            check = "PASS" if abs(diff_bps) <= tolerance_bps else "FAIL"
        else:
            diff_bps, check = np.nan, "NO_PRINTED_YIELD"

        # Market yield used for the spread: the exchange-printed yield when it
        # passed validation, otherwise our recomputed yield (never a failed print).
        corp_yield = printed if check == "PASS" else computed
        g_for_spread = to_annual_effective(g_quoted) if SPREAD_BASIS == "annual_effective" else g_quoted

        rows.append({
            "trade_date": settle, "isin": r["isin"], "issuer": r["issuer"],
            "coupon_pct": r["coupon_pct"], "maturity_date": r["maturity_date"],
            "price": r["price"], "value_cr": r.get("value_cr", np.nan),
            "printed_yield_pct": printed, "computed_yield_pct": computed,
            "yield_diff_bps": diff_bps, "yield_check": check,
            "tenor_years": tenor, "gsec_yield_quoted_pct": g_quoted,
            "gsec_yield_used_pct": g_for_spread, "gsec_extrapolated": extrap,
            "corp_yield_used_pct": corp_yield,
            "spread_bps": (corp_yield - g_for_spread) * 100.0,
            "source_id": r.get("source_id", ""),
        })
    spreads = pd.DataFrame(rows)
    if extra_rejects:
        rejected = pd.concat([rejected, pd.DataFrame(extra_rejects)], ignore_index=True)
    return spreads, rejected


def validation_summary(spreads: pd.DataFrame, tolerance_bps: float = YIELD_TOLERANCE_BPS) -> dict:
    """Headline numbers for the validation test (Chapter 11)."""
    checked = spreads[spreads["yield_check"].isin(["PASS", "FAIL"])]
    if checked.empty:
        return {"n_checked": 0}
    d = checked["yield_diff_bps"].abs()
    return {
        "n_checked": int(len(checked)),
        "n_pass": int((checked["yield_check"] == "PASS").sum()),
        "pass_rate": float((checked["yield_check"] == "PASS").mean()),
        "tolerance_bps": tolerance_bps,
        "mean_abs_diff_bps": float(d.mean()),
        "median_abs_diff_bps": float(d.median()),
        "p95_abs_diff_bps": float(d.quantile(0.95)),
        "max_abs_diff_bps": float(d.max()),
    }


def daily_issuer_spreads(spreads: pd.DataFrame) -> pd.DataFrame:
    """
    One spread per issuer per day: traded-value-weighted average of that day's
    trades (equal weights where the traded value is missing). FAILed prints are excluded.
    """
    ok = spreads[spreads["yield_check"] != "FAIL"].copy()
    ok["w"] = ok["value_cr"].fillna(1.0).clip(lower=1e-9) if "value_cr" in ok else 1.0
    ok["ws"] = ok["w"] * ok["spread_bps"]
    g = ok.groupby(["issuer", "trade_date"]).agg(ws=("ws", "sum"), w=("w", "sum"),
                                                  n_trades=("spread_bps", "size"))
    g["spread_bps"] = g["ws"] / g["w"]
    return g.reset_index().rename(columns={"trade_date": "date"})[["issuer", "date", "spread_bps", "n_trades"]]
