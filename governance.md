# Governance note (Module 7 artifact)

## 1. Purpose and intended use
SpreadWatch is an academic tool. It measures whether corporate bond spreads move before credit rating actions, and it compares agency rating migration with market-implied default risk. It is **not** a trading signal, rating, or investment recommendation.

> **Disclaimer (shown in the app, annexure and commentary):** Educational use only. Nothing here is investment advice or a recommendation to buy, sell or hold any security (SEBI (Research Analysts) Regulations, 2014).

## 2. Model inventory and risk tier

| Component | Type | Risk | Control |
|---|---|---|---|
| `engine/bond_math.py` | Deterministic formula | Low | Hand-worked tests; recomputed yield must match the exchange-printed yield within 5 bps on every trade |
| `engine/curve.py` | Interpolation | Low | Tests; stale curves (> 7 days) refused; extrapolated tenors flagged |
| `engine/event_study.py` | Statistical inference | **Medium**: choice of windows and thresholds drives the answer | Windows fixed *before* looking at real results (`config.py`); placebo control; two-test verdict rule; sensitivity analysis reported |
| `engine/transition.py` | Statistical estimate | Medium: small samples per grade | Unit counts printed in every row; PD not reported for grades with no observations |
| `extract.py` | Rule-based NLP | Medium: a wrong rating changes the event list | Evidence sentence stored for every row; `verified = N` until a person checks it; accuracy measured against a hand-labelled sample |
| `commentary.py` (LLM mode) | Generative AI | **High** if uncontrolled | See section 4 |

## 3. Data ethics and provenance
* Secondary, published data only. No personal data is processed.
* Every raw file must have a row in `sources.csv` (publisher, URL, date pulled); `ingest.py` refuses unregistered files and records a SHA-256 hash of each file it reads.
* Exchange and agency websites have terms of use. Data is downloaded manually for academic, non-commercial analysis. If a site's terms forbid redistribution, keep the raw files out of the public repository and publish only `sources.csv` with the URLs and hashes.
* Synthetic data is labelled SYNTHETIC in file names, issuer names, every chart title and an app banner, and is never used for findings.

## 4. Controls on the language model (L4)
1. **No arithmetic in the model.** Every number comes from the L3 engine, passed to the writer as a named fact.
2. **Mandatory citation.** Every sentence must end with `[E:fact]`, `[S:source]` or `[R:paragraph]` tags that exist.
3. **Number verification.** Every number in a sentence must appear in the facts that sentence cites (numbers inside a quoted rationale excerpt must come from that excerpt).
4. **Fail closed.** Sentences that fail are dropped and logged in `commentary_verification.csv`. If the model is unavailable or declines, the deterministic template is used and the app says so.
5. **No advice.** The prompt forbids buy/sell/hold language; the disclaimer is appended to every output.
6. **Reproducibility.** Template mode is deterministic. LLM mode records the writer used. The model name is configurable (`SPREADWATCH_MODEL`).

## 5. Known limitations (carry into Chapter 9)
* Thin secondary trading: many downgraded bonds do not trade often enough to measure (reported as "unusable" events).
* Reported OTC trades may include non-arm's-length or odd-lot prints; value weighting reduces but does not remove this.
* Rating actions are public on the release date; we cannot observe earlier private communication with the issuer.
* The cohort transition matrix is estimated from a small sample of issuers; confidence intervals are wide.
* The credit-triangle PD treats the whole spread as default compensation (an upper bound).

## 6. Regulatory context (verify each reference and cite it in the bibliography before use)
* SEBI (Credit Rating Agencies) Regulations, 1999, and SEBI's master circular for CRAs: rating actions must be published as press releases, and CRAs must disclose rating transition and default statistics. This is why our transition matrix can be compared with the agencies' own published studies.
* SEBI (Research Analysts) Regulations, 2014: the reason for the disclaimer.
* Basel II foundation-IRB / RBI capital framework: the 45% supervisory LGD for senior unsecured exposures (assumption A13).
* FBIL as the administrator of the benchmark G-sec curve used for the spread.

## 7. Change log
Record any change to `config.py` after real results were first seen, with the reason. Changing a window or threshold after seeing the answer without disclosing it is data dredging.

### Change log entries (real-data build, 18 October 2026)
| Date | What changed | Why | Effect |
|---|---|---|---|
| 18-Oct-2026 | First real run: bonds per issuer chosen by total trading days 2018-2022 (3 per issuer) | Initial rule | 2 of 5 downgrade episodes measurable -> verdict INSUFFICIENT DATA |
| 18-Oct-2026 | **Bond selection changed** after the first run: for downgraded issuers, rank bonds on trading days in [first downgrade - 400 days, first downgrade + 60 days], up to 5 bonds | The first rule picked bonds that mostly traded *after* the downgrades, so the pre-event window had no data. The new rule uses rating dates only, not spread results | DHFL and Tata Motors episodes measurable; others still too thin |
| 18-Oct-2026 | **Three issuers added** (Tata Motors Finance, PNB Housing Finance, ECL Finance) after the first run | Raise the number of measurable episodes; chosen because they were downgraded in 2019-20 AND are among the more traded issuers on NSE | 3 of 8 episodes measurable -> verdict PARTIAL |
| 18-Oct-2026 | Test windows, z threshold, minimum observations, benchmark | **Not changed** at any point | - |

Both changes affect *which data enter the study*, not *how the test is scored*. They are disclosed here and in Chapter 10.
