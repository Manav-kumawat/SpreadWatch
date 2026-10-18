# Assumptions

Every assumption below is a line in `config.py` or a stated rule in `engine/`. For each one: what we assumed, why, and what would change if it were wrong. Both group members must be able to defend every row.

## Bond mathematics

| # | Assumption | Why | If wrong |
|---|---|---|---|
| A1 | Corporate bond yields are **annualised, annually compounded, Actual/365** (the "XIRR" convention): `Dirty price = Σ CF / (1+y)^(days/365)` | This is the market convention for reported Indian corporate bond yields (look up and cite SEBI's circular on the day count convention for debt securities). We do not rely on the citation alone: The validation test checks it directly: if our convention differed from the exchange's, recomputed yields would not match printed yields. | The validation pass rate would drop, and we would see it. |
| A2 | Accrued interest = annual coupon × days since last coupon / 365 | Consistent with A1. A leap-year period strictly uses 366 under Actual/Actual. | Error of at most 1/365 of a day's coupon, roughly 0.002 per 100 face. Far below the 5 bp tolerance. |
| A3 | Coupon dates roll back from maturity in steps of 12/frequency months; default frequency is annual | Most Indian corporate NCDs pay annually; the frequency is read from the data where available. | A semi-annual bond treated as annual would fail the yield check, so it would be caught. |
| A4 | Validation tolerance = **5 bps** between recomputed and printed yield | Prices are reported to 4 decimals. One paisa of price on a 5-year bond is roughly 0.25 bp, so 5 bps allows for rounding and small settlement-date conventions without hiding real errors. | Tighter tolerance means more FAILs from rounding. Looser tolerance risks accepting misprints. The full distribution of differences is reported, not only the pass rate. |

## Spread

| # | Assumption | Why | If wrong |
|---|---|---|---|
| A5 | The risk-free benchmark is the **FBIL G-sec par yield curve**, linearly interpolated at the bond's residual maturity, flat beyond the ends (flagged) | FBIL is the RBI-recognised benchmark administrator. Linear interpolation is transparent and easy to reproduce by hand. (State the tenor points of the FBIL file you actually use in Chapter 10.) | Spline interpolation changes spreads by a few bps between nodes. Spread *changes* around an event are almost unaffected because the same method is used on both sides. |
| A5b | G-sec par yields (semi-annual) are converted to **annual effective** before subtracting: (1 + y/2)² − 1 | Otherwise the spread mixes two compounding bases. At a 7% G-sec yield the mismatch is about **12 bps** (7.1225% vs 7.00%). | Using quoted yields (`SPREAD_BASIS = "quoted"`) shifts every spread by roughly 10–15 bps but leaves the event-study changes nearly identical. |
| A5c | Use the latest curve on or before the trade date, never older than 7 calendar days | Covers weekends and holidays without using stale data. | Trades with no recent curve are rejected and listed, not silently priced. |
| A5d | An issuer's daily spread is the **traded-value-weighted** average of its trades that day | Large trades carry more price information than small, possibly retail, trades. | Equal weighting gives more weight to odd-lot prints. Easy to test by switching. |

## Event study

| # | Assumption | Why | If wrong |
|---|---|---|---|
| A6 | An event is a **downgrade or default by any agency**. Only the **first** downgrade of an episode is used; downgrades within 180 days of an earlier one are dropped. | A second downgrade's "before" window already contains the first downgrade's reaction, so it would overstate early movement. | Including them inflates the count and the apparent lead. |
| A7 | **Abnormal spread** = issuer spread − weekly median spread of issuers with no downgrade within ±180 days | Removes market-wide moves (rate cycles, liquidity squeezes such as the 2018 NBFC episode or March 2020) so the test isolates issuer-specific information. | Without it, a market-wide widening would look like "anticipation". You can switch it off (`use_benchmark=False`) and compare. |
| A8 | Windows: baseline days −180 to −91, pre-event −30 to −1, detection −90 to −1, post 0 to +30 (calendar days) | The baseline ends three months before the event so it is not contaminated by early leakage; 30 days is "just before the agency acts". | The report should show a sensitivity check with a 60-day pre-window. |
| A9 | Early-warning breach = abnormal spread above baseline mean + **2 standard deviations** on **two consecutive** observations | 2 sd is a conventional threshold. Requiring two observations stops a single bad print triggering it. | A lower threshold gives a higher hit rate *and* a higher false-alarm rate. The placebo test shows the trade-off. |
| A10 | An event needs at least 4 baseline and 2 pre-event observations | Corporate bonds trade thinly; with fewer points the standard deviation is meaningless. | Events that fail are reported as unusable, not dropped silently. How many fail is itself a finding about market liquidity. |
| A10b | Placebo dates are drawn at random (seed 42) for issuers never downgraded in the sample | Gives the false-alarm rate of the same rule on dates when nothing happened. | A different seed changes placebo numbers slightly; the seed is fixed so results reproduce exactly. |
| A10c | Verdict = YES only if the one-sided t-test is significant at 5% **and** both the Fisher (hit vs false alarm) and Mann-Whitney (events vs placebo) tests are significant | The question must be answerable either way. Requiring both the level test and the comparison with random dates guards against a widening that happens all the time anyway. | Weaker rules would say YES more often. |

## Transition matrix and PD

| # | Assumption | Why | If wrong |
|---|---|---|---|
| A11 | **Cohort method**, annual cohorts formed on **1 April**, one-year horizon, each (issuer, agency) pair is a unit, ratings grouped into AAA/AA/A/BBB/BB/B/C/D | The standard method in the agencies' own published transition studies; 1 April matches the Indian financial year. | Monthly cohorts or the duration (continuous-time) method give smoother estimates; the cohort method is the simplest to verify by hand. |
| A11b | A unit that hits D at any time within the year counts as D, even if later upgraded | Default is the event that matters for PD. | Counting only the year-end rating would understate default rates. |
| A11c | **Withdrawn** ratings are excluded from the denominator ("withdrawal-adjusted"); the withdrawal share is reported | A withdrawn rating is not a credit outcome. | If withdrawals hide defaults, PDs are understated. This is a known limitation and is disclosed. |
| A12 | Multi-year PD = (P^n)[grade, D]: a **time-homogeneous Markov chain** | Standard simplification. | Real ratings show momentum (a downgraded issuer is more likely to be downgraded again), so multi-year PD for recently downgraded issuers is understated. |
| A13 | Market-implied PD uses the **credit triangle** spread = hazard × LGD with **LGD = 45%** | 45% is the Basel II foundation-IRB supervisory LGD for senior unsecured claims, which RBI also uses. | The whole spread is treated as default compensation, so implied PD is an *upper* bound. The gap to historical PD measures the liquidity and risk premium. |

## Data

| # | Assumption | Why | If wrong |
|---|---|---|---|
| A14 | Reported OTC trades on NSE/BSE are representative prices | They are the only public transaction prices for Indian corporate bonds. | Many bonds trade rarely; stale prices bias spreads toward no change. This biases the test *against* finding early movement, which makes a YES conservative. |
| A15 | Rating action dates are the press-release dates on the agency websites | That is when the information became public. | If an agency informed the issuer earlier, "early" market moves could be leakage rather than independent analysis. Discuss this in Chapter 12. |
