"""
Bond price <-> yield mathematics for Indian corporate bonds.

Convention (see assumptions.md A1-A3)
-------------------------------------
* Coupons are paid `freq` times a year; the schedule is built backwards
  from the maturity date.
* Time to each cash flow is measured in years as (cash-flow date - settlement
  date) / 365  (Actual/365).
* The yield is an annualised, annually-compounded rate (the "XIRR" convention
  used by Indian exchanges and FIMMDA for corporate bond yields):

        Dirty price = sum_i  CF_i / (1 + y) ** t_i

* Clean price = Dirty price - Accrued interest
  Accrued interest = annual coupon x (days since last coupon) / 365

The yield is found by bisection: we keep halving an interval [lo, hi] that is
known to contain the answer. It is slower than Newton's method but it always
converges and is easy to explain line by line.
"""
from dataclasses import dataclass
from datetime import date
from typing import List, Tuple

import pandas as pd

from config import DAY_COUNT_BASIS, FACE_VALUE


@dataclass
class Bond:
    coupon_pct: float          # annual coupon, % of face (e.g. 8.5)
    maturity: date
    freq: int = 1              # coupons per year
    face: float = FACE_VALUE

    def validate(self) -> None:
        if self.coupon_pct < 0 or self.coupon_pct > 30:
            raise ValueError(f"coupon {self.coupon_pct}% outside 0-30%")
        if self.freq not in (1, 2, 4, 12):
            raise ValueError(f"coupon frequency {self.freq} not supported (1, 2, 4, 12)")
        if self.face <= 0:
            raise ValueError("face value must be positive")


def _to_date(d) -> date:
    return pd.Timestamp(d).date()


def coupon_schedule(bond: Bond, settle) -> Tuple[date, List[date]]:
    """
    Return (previous coupon date, list of future coupon dates incl. maturity).

    Built backwards from maturity in steps of 12/freq months.
    """
    settle = _to_date(settle)
    maturity = _to_date(bond.maturity)
    if settle >= maturity:
        raise ValueError(f"settlement {settle} is on/after maturity {maturity}")
    step = 12 // bond.freq
    future: List[date] = []
    k = 0
    while True:
        d = (pd.Timestamp(maturity) - pd.DateOffset(months=step * k)).date()
        if d <= settle:
            prev = d
            break
        future.append(d)
        k += 1
    future.reverse()
    return prev, future


def accrued_interest(bond: Bond, settle) -> float:
    prev, _ = coupon_schedule(bond, settle)
    days = (_to_date(settle) - prev).days
    return bond.face * bond.coupon_pct / 100.0 * days / DAY_COUNT_BASIS


def cash_flows(bond: Bond, settle) -> List[Tuple[float, float]]:
    """List of (time in years, cash amount) for every future payment."""
    settle = _to_date(settle)
    _, dates = coupon_schedule(bond, settle)
    cpn = bond.face * bond.coupon_pct / 100.0 / bond.freq
    flows = []
    for i, d in enumerate(dates):
        amount = cpn + (bond.face if i == len(dates) - 1 else 0.0)
        flows.append(((d - settle).days / DAY_COUNT_BASIS, amount))
    return flows


def dirty_price_from_yield(bond: Bond, settle, ytm_pct: float) -> float:
    y = ytm_pct / 100.0
    return sum(cf / (1.0 + y) ** t for t, cf in cash_flows(bond, settle))


def clean_price_from_yield(bond: Bond, settle, ytm_pct: float) -> float:
    return dirty_price_from_yield(bond, settle, ytm_pct) - accrued_interest(bond, settle)


def yield_from_clean_price(bond: Bond, settle, clean_price: float,
                           lo: float = -5.0, hi: float = 60.0, tol: float = 1e-9) -> float:
    """
    Solve for the yield (in %) that reprices the bond to `clean_price`.

    Price falls as yield rises, so if price(mid) is too high the yield must be
    higher than mid and we move `lo` up; otherwise we move `hi` down.
    """
    bond.validate()
    if clean_price <= 0:
        raise ValueError(f"price must be positive, got {clean_price}")
    target = clean_price + accrued_interest(bond, settle)
    flows = cash_flows(bond, settle)          # built once, re-priced many times

    def pv(ytm_pct: float) -> float:
        y = ytm_pct / 100.0
        return sum(cf / (1.0 + y) ** t for t, cf in flows)

    f_lo = pv(lo) - target
    f_hi = pv(hi) - target
    if f_lo * f_hi > 0:
        raise ValueError(
            f"price {clean_price} implies a yield outside {lo}%-{hi}% - check the input"
        )
    for _ in range(200):
        mid = (lo + hi) / 2.0
        f_mid = pv(mid) - target
        if abs(f_mid) < tol or (hi - lo) / 2.0 < tol:
            return mid
        if f_mid > 0:      # price too high -> yield must be higher
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def residual_tenor_years(maturity, settle) -> float:
    return (_to_date(maturity) - _to_date(settle)).days / DAY_COUNT_BASIS


def macaulay_duration(bond: Bond, settle, ytm_pct: float) -> float:
    y = ytm_pct / 100.0
    flows = cash_flows(bond, settle)
    pv = [cf / (1 + y) ** t for t, cf in flows]
    return sum(t * p for (t, _), p in zip(flows, pv)) / sum(pv)
