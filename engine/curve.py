"""
Government securities (G-sec) yield curve: look-up and matched-tenor interpolation.

Input: a long table  date | tenor_years | yield_pct  (e.g. the FBIL par yield
curve published every business day).

The "matched-tenor" G-sec yield for a corporate bond with residual maturity T
is read off that day's curve by straight-line interpolation between the two
curve points that bracket T:

    y(T) = y1 + (y2 - y1) * (T - T1) / (T2 - T1)

Beyond the shortest/longest point we hold the end value flat and flag it.
"""
from typing import Tuple

import numpy as np
import pandas as pd

from config import GSEC_COMPOUNDING, MAX_CURVE_STALENESS_DAYS


class CurveError(ValueError):
    pass


class GsecCurve:
    def __init__(self, curve_df: pd.DataFrame):
        df = curve_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.dropna(subset=["tenor_years", "yield_pct"])
        if (df["tenor_years"] <= 0).any():
            raise CurveError("curve contains non-positive tenors")
        self._by_date = {
            d: g.sort_values("tenor_years")[["tenor_years", "yield_pct"]].to_numpy()
            for d, g in df.groupby("date")
        }
        self.dates = np.array(sorted(self._by_date), dtype="datetime64[ns]")

    def curve_on(self, when) -> Tuple[pd.Timestamp, np.ndarray]:
        """Latest curve on or before `when`, refusing anything older than the staleness limit."""
        when = pd.Timestamp(when)
        idx = np.searchsorted(self.dates, np.datetime64(when), side="right") - 1
        if idx < 0:
            raise CurveError(f"no G-sec curve on or before {when.date()}")
        d = pd.Timestamp(self.dates[idx])
        if (when - d).days > MAX_CURVE_STALENESS_DAYS:
            raise CurveError(
                f"latest G-sec curve before {when.date()} is {d.date()} "
                f"(> {MAX_CURVE_STALENESS_DAYS} days stale)"
            )
        return d, self._by_date[d]

    def matched_yield(self, when, tenor_years: float) -> Tuple[float, bool]:
        """Return (interpolated G-sec yield in %, extrapolated flag)."""
        _, pts = self.curve_on(when)
        tenors, ylds = pts[:, 0], pts[:, 1]
        extrapolated = bool(tenor_years < tenors[0] or tenor_years > tenors[-1])
        return float(np.interp(tenor_years, tenors, ylds)), extrapolated


def to_annual_effective(yield_pct: float, compounding: int = GSEC_COMPOUNDING) -> float:
    """
    Convert a yield compounded `compounding` times a year to annual effective.
    Example: 7.00% semi-annual -> (1 + 0.07/2)^2 - 1 = 7.1225%
    """
    return ((1 + yield_pct / 100.0 / compounding) ** compounding - 1) * 100.0
