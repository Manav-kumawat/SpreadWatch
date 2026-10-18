"""
Hand-worked example 3 - a four-issuer transition matrix.

Cohort 1-Apr-2020, observed 1-Apr-2021:
  U1  AA  -> stays AA
  U2  AA  -> A   (downgraded Oct-2020)
  U3  A   -> D   (defaulted Jan-2021; its Mar-2021 upgrade to C does not undo the default)
  U4  A   -> withdrawn (Sep-2020)

Counts:            AA   A   D   WR
          AA        1   1   0   0
          A         0   0   1   1
Withdrawal-adjusted probabilities:
          AA row: AA 0.5, A 0.5        A row: D 1.0   (WR left out of the denominator)
Two-year PD for AA = P(AA->A) x P(A->D) = 0.5 x 1.0 = 0.5
"""
import pandas as pd
import pytest

from engine import ratings
from engine.transition import (cohort_counts, implied_pd_from_spread, multi_year_pd,
                               probability_matrix, summary_stats)

ACTIONS = pd.DataFrame([
    ("U1", "X", "2019-05-01", "", "CRISIL AA/Stable"),
    ("U2", "X", "2019-05-01", "", "CRISIL AA"),
    ("U2", "X", "2020-10-01", "CRISIL AA", "CRISIL A"),
    ("U3", "X", "2019-05-01", "", "[ICRA]A (Stable)"),
    ("U3", "X", "2021-01-10", "[ICRA]A", "[ICRA]D"),
    ("U3", "X", "2021-03-10", "[ICRA]D", "[ICRA]C"),
    ("U4", "X", "2019-05-01", "", "IND A-"),
    ("U4", "X", "2020-09-01", "IND A-", "Withdrawn"),
], columns=["issuer", "agency", "action_date", "rating_from", "rating_to"])


def counts():
    return cohort_counts(ACTIONS, start_year=2020, end_date="2021-04-01")


def test_counts():
    c = counts()
    assert c.loc["AA", "AA"] == 1 and c.loc["AA", "A"] == 1
    assert c.loc["A", "D"] == 1 and c.loc["A", "WR"] == 1
    assert c.to_numpy().sum() == 4


def test_probabilities_rows_sum_to_one():
    p = probability_matrix(counts())
    assert p.loc["AA", "AA"] == pytest.approx(0.5)
    assert p.loc["A", "D"] == pytest.approx(1.0)
    observed = p.dropna(how="all")
    assert observed.sum(axis=1).tolist() == pytest.approx([1.0] * len(observed))


def test_multi_year_pd():
    pd_tab = multi_year_pd(probability_matrix(counts()), years=(1, 2))
    assert pd_tab.loc["AA", "PD_1y"] == pytest.approx(0.0)
    assert pd_tab.loc["AA", "PD_2y"] == pytest.approx(0.5)
    assert pd.isna(pd_tab.loc["BBB", "PD_1y"])          # no observations -> no PD claimed


def test_summary_stats():
    s = summary_stats(counts())
    assert s.loc["AA", "downgrade"] == pytest.approx(0.5)
    assert s.loc["A", "default"] == pytest.approx(1.0)
    assert s.loc["A", "withdrawn_share"] == pytest.approx(0.5)


def test_implied_pd_credit_triangle():
    # 180 bps / 45% LGD = 4% hazard -> 1 - e^-0.04 = 3.9211%
    assert implied_pd_from_spread(180) == pytest.approx(0.039211, abs=1e-6)


@pytest.mark.parametrize("raw,sym", [
    ("CRISIL AA+/Stable", "AA+"), ("[ICRA]A- (Negative)", "A-"), ("IND BBB+ (CE)", "BBB+"),
    ("CARE D", "D"), ("Provisional CARE AA", "AA"), ("Withdrawn", "WR"),
    ("CRISIL A1+", None), ("ACUITE BB-", "BB-"), ("AAA", "AAA"),
])
def test_rating_normalisation(raw, sym):
    assert ratings.normalise(raw) == sym


def test_action_classified_by_notches_not_words():
    assert ratings.classify_action("CRISIL AAA", "CRISIL AA+") == "downgrade"
    assert ratings.classify_action("[ICRA]BBB", "[ICRA]D") == "default"
    assert ratings.classify_action("CARE A", "CARE A+") == "upgrade"
    assert ratings.classify_action("CARE A", "CARE A") == "reaffirm"
