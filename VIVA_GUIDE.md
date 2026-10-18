# Viva guide

Twenty of the fifty marks test whether you can explain your own calculation. Work through this file together until each of you can redo every formula on paper.

---

## 1. Formula sheet (each one is a test in `tests/`)

**Price from yield** (`engine/bond_math.py`, test_bond_math.py)
`Dirty price = Σ CFᵢ / (1 + y)^(tᵢ)`, with tᵢ = days to cash flow / 365
`Clean price = Dirty price − Accrued interest`
`Accrued = Face × Coupon × days since last coupon / 365`

*Worked:* 3-year 8% annual bond at a 9% yield, on a coupon date
= 8/1.09 + 8/1.09² + 108/1.09³ = 7.33945 + 6.73344 + 83.39582 = **97.46871**

**Yield from price:** bisection. Start with [−5%, 60%], price the bond at the midpoint, and keep the half that still contains the answer. Price falls as yield rises: if the price at the midpoint is too high, the yield must be higher. Stop when the interval is narrower than 1e-9.
*Why not Newton's method?* Bisection always converges and is easy to explain. Speed is irrelevant at our data size.

**Matched-tenor G-sec yield** (`engine/curve.py`)
`y(T) = y₁ + (y₂ − y₁)(T − T₁)/(T₂ − T₁)`
*Worked:* 1y = 6.50%, 5y = 7.00%, bond has 3 years left → 6.50 + 0.50 × 2/4 = **6.75%**

**Compounding conversion:** FBIL par yields are semi-annual. Annual effective = (1 + y/2)² − 1
*Worked:* 7.00% → **7.1225%**. Ignoring this would overstate every spread by about **12 bps**.

**Spread** (`engine/spreads.py`): `Spread (bps) = (corporate yield − G-sec yield, annual effective) × 100`
*Worked:* 9.00% − 7.1225% = **187.75 bps**

**Abnormal spread** (`engine/event_study.py`): `AS = issuer spread − weekly median spread of quiet issuers`
**Pre-event widening:** `ΔS_pre = mean AS(days −30..−1) − mean AS(days −180..−91)`
**z-score:** `z = ΔS_pre / sd(AS in baseline)`; flagged if z ≥ 2
**Lead time:** first day in −90..−1 when AS > baseline mean + 2 sd on two consecutive observations
**Share before the action:** `ΔS_pre / (ΔS_pre + ΔS_post)`, capped at 1

*Worked (test_event_study.py):* baseline 98/102 alternating → mean 100, sd ≈ 2; pre-window 150; post 170
→ ΔS_pre = 50 bps, z ≈ 25, flagged, lead 30 days, share = 50/70 = **71%**

**Transition matrix** (`engine/transition.py`): `P[i,j] = N[i,j] / Σ_j N[i,j]` (withdrawals excluded)
*Worked:* 2 AA units, one stays and one falls to A → P(AA→AA) = 0.5, P(AA→A) = 0.5.
**n-year PD** = (Pⁿ)[i, D]. *Worked:* P(AA→A) = 0.5, P(A→D) = 1 → 2-year PD of AA = 0.5 × 1 = **0.5**

**Spread-implied PD (credit triangle):** hazard λ = s / LGD; PD(1y) = 1 − e^(−λ)
*Worked:* 180 bps, LGD 45% → λ = 0.018/0.45 = 4% → PD = 1 − e^(−0.04) = **3.92%**

**Tests:** one-sided t-test on ΔS_pre (H₀: mean ≤ 0); Wilcoxon signed-rank (the same without assuming normality); Mann-Whitney (events vs placebo); Fisher exact (hit rate vs false-alarm rate).

---

## 2. The three-minute demonstration

Rehearse it with a stopwatch. Open `streamlit run app.py` before you enter the room.

| Time | What you do | What you say |
|---|---|---|
| 0:00 | Tab **1 · Answer** | "Our question is whether spreads move before the agencies act. The verdict is ___ . Mean abnormal widening in the 30 days before a downgrade was ___ bps, p = ___. The red line is downgrades, the grey line is random dates." |
| 0:45 | Tab **3 · Transition matrix** | "This is the one-year matrix from ___ cohorts. ___% of A-rated issuers were downgraded within a year. Bottom chart: spreads imply far higher default probabilities than actually happened. That gap is the liquidity and risk premium." |
| 1:30 | Tab **2 · Issuer spreads**, the pre-selected issuer | "Here is one issuer. The spread left its benchmark ___ days before the downgrade, marked by the red line." |
| 2:15 | Tab **4 · Validation** → *load the deliberately broken demo file* | "The system refuses bad input. This row matures before it trades, this price is quoted per 10,000, this printed yield is 150 bps away from what the price implies. Each one is rejected with a reason and never reaches the spread." |
| 2:50 | Stop | "Every number traces to `sources.csv` and a test." |

Backup: if the laptop fails, `outputs/real/SpreadWatch_Annexure.html` has every figure offline.

---

## 3. Questions to expect, with answers

**"Where does this number come from?"** Open `outputs/real/summary.json` or the facts panel in the Commentary tab. Every fact has an id, the engine file that computed it, and the input file it came from (`sources.csv` and `ingest_log.csv` with the file hash).

**"Why spread over G-sec and not over the swap (OIS) curve?"** G-secs are the risk-free benchmark in India, FBIL publishes an official curve, and the brief specifies it. The OIS market is liquid only at short tenors.

**"Isn't part of the spread liquidity, not credit?"** Yes. That's why we use *changes* in an *abnormal* spread around events: liquidity premia that are stable over time cancel out. Figure 5 shows the size of the non-default part.

**"Why 30 days? Why 2 standard deviations?"** Both were fixed in `config.py` before the real results were seen (`governance.md` §7). Chapter 11 shows a sensitivity check with a 60-day window and a 1.5-sd threshold.

**"Your verdict could just be market-wide stress, e.g. IL&FS in September 2018."** The benchmark removes moves shared by quiet issuers, and the placebo test measures how often the same rule fires when nothing happens.

**"What if the bond didn't trade before the downgrade?"** The event is marked unusable (A10) and counted. The share of unusable events is a finding: a thin secondary market limits price discovery.

**"Why does a YES not mean agencies are useless?"** The market may be reacting to the same public information the agency later formalises, or to the agency's outlook/watch changes. Ratings still drive regulatory and investment-mandate constraints.

**"Why is the transition matrix different from CRISIL's published one?"** Our sample is small and deliberately over-weights downgraded issuers (selection on the outcome). The matrix illustrates the method and matches the published one on ___ (compare in Chapter 11).

**"Where is the AI?"** L2 extracts rating actions from unstructured rationale documents and retrieves the paragraph that explains each action. L4 writes the commentary, optionally with Claude, under a verifier that blocks any uncited sentence or any number that is not in the engine output. The mathematics is deliberately kept out of the model (Rule 2).

**"Show us a failure."** Validation tab → broken demo file; or Commentary tab → the verification log shows sentences the verifier dropped.

---

## 4. Macro and regulatory talking points (verify each and cite it before using)

* **The 2018 IL&FS default** and the NBFC/HFC funding stress that followed: the obvious test case for "did the market move before the agencies?" Agencies were criticised and SEBI later tightened disclosure rules for CRAs. Find the specific SEBI orders and circulars and cite them.
* **SEBI CRA regulations:** press releases for every rating action; disclosure of default and transition statistics, which is why our matrix can be benchmarked.
* **Rate cycle:** RBI cuts in 2019–20 and hikes from May 2022 moved the whole G-sec curve. Measuring spreads (not yields) and using a benchmark removes this.
* **Market structure:** India's corporate bond market is dominated by AAA/AA PSU and financial issuers, with thin secondary trading. Mention SEBI's RFQ platform and reporting requirements.
* **Basel / RBI:** the 45% LGD; also, bank capital charges depend on external ratings, which is why rating timeliness matters for banks.

---

## 5. Who presents what

Both of you should be able to answer anything, but split the lead:
* **Member 1:** data (L1, L2), validation test, transition matrix.
* **Member 2:** event study (L3), commentary governance (L4), app and findings (L5).
