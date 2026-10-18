"""
Hand-worked example 1 - a 3-year 8% annual-pay bond priced at a 9% yield.

Settlement 30-Jun-2024, coupons 30-Jun-2025/26/27. No 29-Feb falls inside
any of those years, so the times are exactly 1, 2 and 3 years.

    Price = 8/1.09 + 8/1.09^2 + 108/1.09^3
          = 7.33945 + 6.73344 + 83.39582
          = 97.46871

Work it in Excel:  =8/1.09 + 8/1.09^2 + 108/1.09^3
"""
from datetime import date

import pytest

from engine.bond_math import (Bond, accrued_interest, clean_price_from_yield, coupon_schedule,
                              yield_from_clean_price)

BOND = Bond(coupon_pct=8.0, maturity=date(2027, 6, 30), freq=1)
SETTLE = date(2024, 6, 30)


def test_schedule_built_back_from_maturity():
    prev, future = coupon_schedule(BOND, SETTLE)
    assert prev == date(2024, 6, 30)
    assert future == [date(2025, 6, 30), date(2026, 6, 30), date(2027, 6, 30)]


def test_price_from_yield_matches_hand_calculation():
    assert clean_price_from_yield(BOND, SETTLE, 9.0) == pytest.approx(97.46871, abs=1e-4)


def test_yield_from_price_inverts_the_price():
    assert yield_from_clean_price(BOND, SETTLE, 97.46871) == pytest.approx(9.0, abs=1e-4)


def test_par_bond_yields_its_coupon():
    assert yield_from_clean_price(BOND, SETTLE, 100.0) == pytest.approx(8.0, abs=1e-6)


def test_accrued_interest_actual_365():
    # 9% coupon, 90 days since last coupon: 100 x 9% x 90/365 = 2.219178
    b = Bond(coupon_pct=9.0, maturity=date(2027, 6, 30))
    assert accrued_interest(b, date(2024, 9, 28)) == pytest.approx(2.219178, abs=1e-6)


def test_mid_period_round_trip():
    settle = date(2025, 1, 15)
    p = clean_price_from_yield(BOND, settle, 8.75)
    assert yield_from_clean_price(BOND, settle, p) == pytest.approx(8.75, abs=1e-6)


def test_rejects_settlement_after_maturity():
    with pytest.raises(ValueError):
        yield_from_clean_price(BOND, date(2028, 1, 1), 100.0)


def test_rejects_absurd_price():
    with pytest.raises(ValueError):
        yield_from_clean_price(BOND, SETTLE, 9850.0)   # quoted per 10,000 by mistake
