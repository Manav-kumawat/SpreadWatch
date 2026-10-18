"""
Hand-worked example 2 - matched-tenor G-sec yield and the spread.

Curve on 30-Jun-2024: 1y = 6.50%, 5y = 7.00% (semi-annual par yields).
A bond with 3 years left reads   6.50 + (7.00 - 6.50) x (3 - 1)/(5 - 1) = 6.75%.

Spread for a corporate bond yielding 9.00% against a flat 7.00% curve:
    7.00% semi-annual -> annual effective (1 + 0.035)^2 - 1 = 7.1225%
    spread = (9.00 - 7.1225) x 100 = 187.75 bps
"""
from datetime import date

import pandas as pd
import pytest

from engine.bond_math import Bond, clean_price_from_yield
from engine.curve import CurveError, GsecCurve, to_annual_effective
from engine.spreads import compute_trade_spreads, validate_trades, validation_summary

CURVE = GsecCurve(pd.DataFrame({
    "date": ["2024-06-28"] * 2,
    "tenor_years": [1.0, 5.0],
    "yield_pct": [6.50, 7.00],
}))
FLAT = GsecCurve(pd.DataFrame({"date": ["2024-06-28"] * 2, "tenor_years": [0.5, 30.0],
                               "yield_pct": [7.0, 7.0]}))


def test_linear_interpolation():
    y, extrap = CURVE.matched_yield("2024-06-30", 3.0)
    assert y == pytest.approx(6.75)
    assert extrap is False


def test_flat_extrapolation_is_flagged():
    y, extrap = CURVE.matched_yield("2024-06-30", 10.0)
    assert y == pytest.approx(7.00)
    assert extrap is True


def test_stale_curve_is_refused():
    with pytest.raises(CurveError):
        CURVE.matched_yield("2024-08-30", 3.0)


def test_semi_annual_to_annual_effective():
    assert to_annual_effective(7.0) == pytest.approx(7.1225, abs=1e-6)


def _trade(printed_yield, true_yield=9.0):
    bond = Bond(8.0, date(2027, 6, 30))
    price = round(clean_price_from_yield(bond, date(2024, 6, 30), true_yield), 4)
    return pd.DataFrame([{
        "trade_date": "2024-06-30", "isin": "TEST01", "issuer": "Test Ltd", "coupon_pct": 8.0,
        "maturity_date": "2027-06-30", "price": price, "printed_yield_pct": printed_yield,
        "value_cr": 10.0,
    }])


def test_spread_matches_hand_calculation():
    out, rej = compute_trade_spreads(_trade(9.0), FLAT)
    assert rej.empty
    row = out.iloc[0]
    assert row["yield_check"] == "PASS"
    assert row["spread_bps"] == pytest.approx(187.75, abs=0.01)


def test_validation_fails_when_printed_yield_disagrees():
    out, _ = compute_trade_spreads(_trade(10.0), FLAT)        # printed 100 bps too high
    row = out.iloc[0]
    assert row["yield_check"] == "FAIL"
    assert row["yield_diff_bps"] == pytest.approx(-100.0, abs=0.1)
    # the failed print is NOT used for the spread - the recomputed 9% is
    assert row["spread_bps"] == pytest.approx(187.75, abs=0.05)
    assert validation_summary(out)["pass_rate"] == 0.0


def test_bad_rows_are_rejected_with_reasons():
    bad = pd.concat([_trade(9.0)] * 4, ignore_index=True)
    bad.loc[0, "maturity_date"] = "2010-01-01"
    bad.loc[1, "price"] = 9850.0
    bad.loc[2, "coupon_pct"] = -1
    bad.loc[3, "trade_date"] = "2031-02-30"
    ok, rejected = validate_trades(bad)
    assert ok.empty
    assert set(rejected["reject_reason"]) == {
        "maturity on/before trade date",
        "price outside 20-200 per 100 face (likely unit error)",
        "coupon missing or outside 0-30%",
        "unreadable trade_date",
    }


def test_missing_column_raises():
    from engine.spreads import TradeValidationError
    with pytest.raises(TradeValidationError):
        validate_trades(pd.DataFrame({"isin": ["X"]}))
