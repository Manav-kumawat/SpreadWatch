"""
Hand-worked example 4 - the early-warning measurement on a made-up issuer.

Issuer "W" (downgraded on day 0):
  days -180..-31 : abnormal spread alternates 98 / 102 bps  (mean 100, sd ~2)
  days  -30..-1  : 150 bps
  days    0..30  : 170 bps
  => dS_pre  = 150 - 100 = 50 bps, z = 50 / 2.0 ~ 25 -> flagged
     first breach of 100 + 2 x 2 = 104 is at day -30 -> lead time 30 days
     dS_post = 170 - 150 = 20 bps; share of widening before the action = 50 / 70 = 0.714

Issuer "Q" (downgraded on day 0) never moves before the action -> not flagged.
"""
import numpy as np
import pandas as pd
import pytest

from engine.event_study import abnormal_spreads, measure_window, run_event_study, select_events

D0 = pd.Timestamp("2022-06-01")


def series(issuer, pre_level, post_level):
    rows = []
    for rel in range(-180, 31, 3):
        if rel < -30:
            v = 98.0 if (rel // 3) % 2 == 0 else 102.0
        elif rel < 0:
            v = pre_level
        else:
            v = post_level
        rows.append({"issuer": issuer, "date": D0 + pd.Timedelta(days=rel), "spread_bps": v,
                     "abnormal_bps": v})
    return pd.DataFrame(rows)


def test_widening_issuer_is_flagged_with_lead_time():
    m = measure_window(series("W", 150.0, 170.0), D0)
    assert m["usable"]
    assert m["baseline_mean_bps"] == pytest.approx(100.0, abs=0.2)
    assert m["dS_pre_bps"] == pytest.approx(50.0, abs=0.2)
    assert m["flagged"]
    assert 28 <= m["lead_days"] <= 30
    assert m["dS_post_bps"] == pytest.approx(20.0)
    assert m["pre_share"] == pytest.approx(50 / 70, abs=0.01)


def test_quiet_issuer_is_not_flagged():
    s = series("Q", 100.0, 160.0)
    s.loc[s["date"] < D0, "abnormal_bps"] = np.where(
        (np.arange((s["date"] < D0).sum()) % 2) == 0, 98.0, 102.0)
    m = measure_window(s, D0)
    assert not m["flagged"]
    assert np.isnan(m["lead_days"])


def test_clustered_downgrades_keep_only_the_first():
    actions = pd.DataFrame([
        ("W", "A", "2022-06-01", "AA", "A"),
        ("W", "A", "2022-08-01", "A", "BBB"),      # 61 days later -> same episode, dropped
        ("W", "A", "2023-09-01", "BBB", "BB"),     # new episode
        ("W", "A", "2023-12-01", "BB", "BB+"),     # upgrade, never an event
    ], columns=["issuer", "agency", "action_date", "rating_from", "rating_to"])
    ev = select_events(actions, ["W"])
    assert list(pd.to_datetime(ev["action_date"]).dt.date.astype(str)) == ["2022-06-01", "2023-09-01"]


def test_too_few_observations_is_unusable():
    s = series("W", 150.0, 170.0)
    m = measure_window(s[s["date"] > D0 - pd.Timedelta(days=60)], D0)
    assert m["usable"] is False


def test_benchmark_removes_market_move():
    # Two issuers move up 40 bps together; one is "quiet" so the move is market-wide.
    dates = pd.date_range("2022-01-03", periods=10, freq="W-MON")
    daily = pd.DataFrame({
        "issuer": ["A"] * 10 + ["B"] * 10,
        "date": list(dates) * 2,
        "spread_bps": [100] * 5 + [140] * 5 + [200] * 5 + [240] * 5,
    })
    actions = pd.DataFrame(columns=["issuer", "agency", "action_date", "rating_from", "rating_to"])
    abn = abnormal_spreads(daily, actions)
    a = abn[abn["issuer"] == "A"]["abnormal_bps"]
    assert a.iloc[:5].mean() == pytest.approx(a.iloc[5:].mean())   # the common 40 bps is gone


def test_end_to_end_runs_and_gives_a_verdict():
    daily = pd.concat([series(f"E{i}", 150.0 + i, 170.0 + i) for i in range(5)])
    actions = pd.DataFrame([(f"E{i}", "A", D0.date().isoformat(), "AA", "A") for i in range(5)],
                           columns=["issuer", "agency", "action_date", "rating_from", "rating_to"])
    res = run_event_study(daily[["issuer", "date", "spread_bps"]], actions, use_benchmark=False)
    assert res["tests"]["n_events_usable"] == 5
    assert res["tests"]["mean_dS_pre_bps"] > 40
    assert res["tests"]["t_pvalue"] < 0.01
