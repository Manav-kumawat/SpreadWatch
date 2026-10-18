"""
Generate a SYNTHETIC sample dataset so the whole system can run end to end
before the real data has been collected.

!!! NOTHING PRODUCED BY THIS SCRIPT MAY APPEAR AS A FINDING IN THE REPORT !!!

Issuers are called "SYN Issuer NN Ltd", agencies "SYN-AGENCY-A/B", ISINs start
with "SYN". The generator deliberately mixes downgrades that are anticipated by
the market (spread drifts wider in the 30-90 days before) with downgrades that
are not, so the event study has something non-trivial to detect. The share of
anticipated downgrades (ANTICIPATED_SHARE) is a choice made here - which is
exactly why synthetic results prove the code works, not that the hypothesis holds.

Usage:  python make_sample_data.py
"""
import numpy as np
import pandas as pd

from config import SAMPLE_DIR
from engine.bond_math import Bond, clean_price_from_yield
from engine.curve import to_annual_effective
from engine.ratings import SCALE

SEED = 7
START, END = "2016-01-01", "2024-12-31"
N_ISSUERS = 50
TRADE_PROB = 0.10                 # chance a given bond trades on a given business day
ANTICIPATED_SHARE = 0.65          # share of downgrades the market "sees coming"
TENORS = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 14, 20, 30]

# Spread level (bps over G-sec) by notch - a stylised credit curve
BASE_SPREAD = dict(zip(SCALE, [55, 85, 110, 140, 185, 230, 280, 360, 430, 510,
                               700, 800, 900, 1050, 1200, 1350, 1600, 1800, 2000, 3500]))


def gsec_curve(rng, days):
    """Nelson-Siegel-shaped par curve whose level and slope follow slow random walks."""
    n = len(days)
    level = 7.0 + np.cumsum(rng.normal(0, 0.012, n))
    level = np.clip(level, 5.5, 8.5)
    slope = -1.2 + np.cumsum(rng.normal(0, 0.008, n))
    slope = np.clip(slope, -2.5, 0.2)
    rows = []
    for d, b0, b1 in zip(days, level, slope):
        for t in TENORS:
            lam = 2.0
            f = (1 - np.exp(-t / lam)) / (t / lam)
            rows.append((d.date(), t, round(b0 + b1 * f, 4)))
    return pd.DataFrame(rows, columns=["date", "tenor_years", "yield_pct"])


def rating_path(rng, months, start_notch):
    """Monthly rating path; downgrade risk rises as the rating falls."""
    path, notch = [], start_notch
    for _ in months:
        if notch < 20:
            p_down = 0.002 + 0.0018 * notch
            p_up = 0.003 if notch > 1 else 0.0
            u = rng.random()
            if u < p_down:
                step = rng.choice([1, 1, 1, 2, 2, 3])
                if notch >= 11 and rng.random() < 0.25:
                    step = 20 - notch            # sudden default from sub-investment grade
                notch = min(20, notch + step)
            elif u < p_down + p_up:
                notch -= 1
        path.append(notch)
    return path


REASONS = [
    "The downgrade reflects a sharp weakening in the liquidity profile of the company, with large "
    "debt repayments falling due over the next two quarters against limited unencumbered cash.",
    "The rating action follows a significant increase in gross non-performing assets in the wholesale "
    "lending book and a slowdown in collections over the last three quarters.",
    "Refinancing risk has risen materially as access to commercial paper and bank lines has tightened, "
    "and the company has relied on asset sales to meet maturities.",
    "Operating margins have contracted because of higher input costs, pushing net leverage above the "
    "level expected for the earlier rating.",
    "Delays in receiving payments from state distribution utilities have stretched the working capital "
    "cycle and weakened debt service coverage.",
]


def write_rationales(rng, actions: pd.DataFrame):
    """SYNTHETIC rating rationales in two common layouts, for the L2 extraction and retrieval demo."""
    out = SAMPLE_DIR / "rationales"
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*.txt"):
        f.unlink()
    moves = actions[actions["rating_from"] != ""]
    for i, r in enumerate(moves.itertuples()):
        d = pd.Timestamp(r.action_date)
        verb = "downgraded" if SCALE.index(r.rating_to) > SCALE.index(r.rating_from) else "upgraded"
        reason = REASONS[int(rng.integers(len(REASONS)))] if verb == "downgraded" else \
            "The upgrade reflects steady improvement in operating cash flows and lower leverage."
        if i % 2 == 0:   # layout 1: "Rating downgraded to X from Y"
            body = (f"SYNTHETIC DOCUMENT - NOT A REAL RATING RATIONALE\n\nRating Rationale\n"
                    f"{d.strftime('%B %d, %Y')} | Mumbai\n{r.issuer}\n\n"
                    f"Rating {verb} to 'SYN {r.rating_to}/{r.outlook}' from 'SYN {r.rating_from}/Stable' "
                    f"on the Non-Convertible Debentures of {r.issuer}.\n\n"
                    f"Rationale\n\n{reason} SYN-AGENCY-A will continue to monitor the refinancing "
                    f"plan and the liquidity position closely.\n\n"
                    f"Key rating drivers\n\nLiquidity: the company had cash and equivalents that cover "
                    f"only part of its scheduled repayments over the next twelve months.\n")
        else:            # layout 2: "[X] (Outlook); downgraded from [Y]"
            body = (f"SYNTHETIC DOCUMENT - NOT A REAL RATING RATIONALE\n\n{r.issuer}\n"
                    f"{d.day:02d} {d.strftime('%B %Y')}\n\nSummary of rating action\n\n"
                    f"Non-convertible debenture programme: [SYN]{r.rating_to} ({r.outlook}); "
                    f"{verb} from [SYN]{r.rating_from} (Stable).\n\n"
                    f"Rationale\n\n{reason} The agency has also taken note of the company's plan to "
                    f"raise equity, the timing of which remains uncertain.\n")
        fname = f"SYN_{r.issuer.replace(' ', '_')}_{d.date().isoformat()}.txt"
        (out / fname).write_text(body)


def main():
    rng = np.random.default_rng(SEED)
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    days = pd.bdate_range(START, END)
    curve = gsec_curve(rng, days)
    curve10 = curve[curve["tenor_years"] == 10].set_index("date")["yield_pct"]
    curve_by_date = {d: g.sort_values("tenor_years") for d, g in curve.groupby("date")}

    months = pd.date_range(START, END, freq="MS")
    market = np.cumsum(rng.normal(0, 3, len(days)))          # market-wide spread factor (bps)
    market = market - pd.Series(market).rolling(250, min_periods=1).mean().to_numpy()

    trades, actions = [], []
    start_dist = [1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6, 7, 8, 9, 10]
    for k in range(1, N_ISSUERS + 1):
        issuer = f"SYN Issuer {k:02d} Ltd"
        agency = "SYN-AGENCY-A" if k % 2 else "SYN-AGENCY-B"
        n0 = int(rng.choice(start_dist))
        path = rating_path(rng, months, n0)
        actions.append(dict(issuer=issuer, agency=agency, action_date="2015-11-15",
                            instrument="NCD", rating_from="", rating_to=SCALE[n0 - 1],
                            outlook="Stable", source_id="SYN01"))

        # Daily "fair" spread: base level of the current rating, plus pre-event drift
        month_idx = np.searchsorted(months.values, days.values, side="right") - 1
        notch_daily = np.array(path)[month_idx]
        target = np.array([BASE_SPREAD[SCALE[n - 1]] for n in notch_daily], dtype=float)
        prev = n0
        for m, n in zip(months, path):
            if n == prev:
                continue
            # the action date is a random day inside the month
            act_day = m + pd.Timedelta(days=int(rng.integers(0, 27)))
            actions.append(dict(issuer=issuer, agency=agency, action_date=act_day.date().isoformat(),
                                instrument="NCD", rating_from=SCALE[prev - 1], rating_to=SCALE[n - 1],
                                outlook="Negative" if n > prev else "Stable", source_id="SYN01"))
            if n > prev and rng.random() < ANTICIPATED_SHARE:
                lead = int(rng.integers(45, 100))
                jump = BASE_SPREAD[SCALE[n - 1]] - BASE_SPREAD[SCALE[prev - 1]]
                ramp_start = act_day - pd.Timedelta(days=lead)
                mask = (days >= ramp_start) & (days < m)
                frac = ((days[mask] - ramp_start).days / lead).to_numpy()
                target[mask] += 0.7 * jump * frac
            prev = n

        idio = np.zeros(len(days))
        for i in range(1, len(days)):
            idio[i] = 0.97 * idio[i - 1] + rng.normal(0, 4)
        spread = target + 0.6 * market + idio

        # Bonds: a 7-10y bond issued at the start, replaced when it gets short
        bonds, issue = [], pd.Timestamp(START) - pd.Timedelta(days=int(rng.integers(30, 700)))
        while issue < pd.Timestamp(END):
            mat = issue + pd.DateOffset(years=int(rng.integers(7, 11)))
            cpn = round(float(curve10.iloc[0]) + BASE_SPREAD[SCALE[n0 - 1]] / 100 + rng.normal(0, 0.3), 2)
            bonds.append((f"SYN{k:02d}{len(bonds):02d}XX0{len(bonds)}", cpn, mat.date()))
            issue = mat - pd.DateOffset(years=2)

        for i, d in enumerate(days):
            if notch_daily[i] == 20:          # defaulted paper: stop simulating trades
                continue
            for isin, cpn, mat in bonds:
                if not (pd.Timestamp(mat) - d).days > 180 or rng.random() > TRADE_PROB:
                    continue
                if d < pd.Timestamp(mat) - pd.DateOffset(years=11):
                    continue
                tenor = (pd.Timestamp(mat) - d).days / 365
                c = curve_by_date[d.date()]
                g = float(np.interp(tenor, c["tenor_years"], c["yield_pct"]))
                y = round(to_annual_effective(g) + spread[i] / 100 + rng.normal(0, 0.02), 4)
                price = round(clean_price_from_yield(Bond(cpn, mat), d, y), 4)
                trades.append(dict(trade_date=d.date().isoformat(), isin=isin, issuer=issuer,
                                   coupon_pct=cpn, coupon_freq=1, maturity_date=mat.isoformat(),
                                   price=price, printed_yield_pct=y,
                                   value_cr=round(float(rng.lognormal(2.5, 0.8)), 2), source_id="SYN01"))

    pd.DataFrame(trades).to_csv(SAMPLE_DIR / "trades.csv", index=False)
    curve.assign(source_id="SYN01").to_csv(SAMPLE_DIR / "gsec_curve.csv", index=False)
    pd.DataFrame(actions).to_csv(SAMPLE_DIR / "rating_actions.csv", index=False)

    # A deliberately broken file for the live "failure" demonstration
    bad = pd.DataFrame(trades[:6]).copy()
    bad.loc[0, "maturity_date"] = "2010-01-01"           # matures before it trades
    bad.loc[1, "price"] = 9850.0                          # price quoted per 10,000 not per 100
    bad.loc[2, "printed_yield_pct"] = bad.loc[2, "printed_yield_pct"] + 1.5   # printed yield 150 bps off
    bad.loc[3, "coupon_pct"] = -1                         # impossible coupon
    bad.loc[4, "trade_date"] = "2031-02-30"               # not a date
    bad.to_csv(SAMPLE_DIR / "bad_trades_demo.csv", index=False)

    write_rationales(rng, pd.DataFrame(actions))

    print(f"SYNTHETIC data written to {SAMPLE_DIR}: {len(trades)} trades, "
          f"{len(actions)} rating actions, {curve['date'].nunique()} curve dates")


if __name__ == "__main__":
    main()
