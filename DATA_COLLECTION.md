# Collecting the real data

> The full step-by-step version, with the 24 named companies, source URLs and a two-day plan, is **docs/SpreadWatch_Guidebook.docx**. Templates are in `templates/`. Bond terms missing from trade files go in `data/raw/master/bond_master.csv`; per-maturity G-sec files go in `data/raw/gsec_tenors/` and are combined with `python prepare_curve.py`.

The system already runs end to end on synthetic data. To produce **findings**, you need three real inputs. Exchange and agency websites block automated downloads, so collect them by hand. That's also your evidence of data provenance in the viva.

**Golden rule:** every time you save a file, add one row to `sources.csv` straight away:

```
source_id,file,publisher,url,date_pulled,description,synthetic
NSE01,data/raw/trades/nse_cbm_2019-01.csv,National Stock Exchange of India,<exact URL you downloaded from>,2026-10-20,"Reported corporate bond trades, Jan 2019",N
```

`ingest.py` refuses any file in `data/raw/` that has no row here.

---

## Step 1: Choose the sample (do this first, ~1 day)

You need two groups of issuers over one window (suggested **April 2017 to March 2025**, which covers the 2018–20 NBFC/HFC stress, COVID-19 and the 2022–23 rate hikes):

* **Event issuers (15–25):** companies with listed NCDs or bonds that were **downgraded** in the window. Good places to look are the agencies' rating-action lists and their annual default studies. Housing finance companies, NBFCs, infrastructure, power and telecom issuers were heavily downgraded in this period. Candidates worth checking (verify each one yourself): IL&FS group entities, Dewan Housing Finance, Reliance Capital / Reliance Home Finance, Altico Capital, Srei Infrastructure Finance, Yes Bank (Tier II / AT1 bonds).
* **Control issuers (15–25):** issuers with **stable** ratings and frequently traded bonds, e.g. AAA/AA public-sector financiers and large private issuers. They feed the benchmark and the placebo test.

For every issuer, note 1–3 ISINs of bonds that traded regularly. Bonds that never trade cannot be measured; record how many you dropped. That's a finding about market liquidity.

## Step 2: Corporate bond trades → `data/raw/trades/`

**Publisher:** NSE and/or BSE (debt segment, reported OTC corporate bond trades). On the exchange website, go to *Market Data → Debt / Corporate Bonds → reported trades* (the archive of daily or monthly reports). Download the files covering your window.

Columns `ingest.py` recognises (it maps common header names automatically):

| needed | typical header in the exchange file |
|---|---|
| trade_date | Trade Date / Date |
| isin | ISIN |
| issuer | Issuer Name / Description |
| coupon_pct | Coupon / Coupon Rate (%) |
| maturity_date | Maturity Date |
| price | Price / Weighted Average Price |
| printed_yield_pct | Yield / Weighted Average Yield |
| value_cr | Trade Value (Rs. Cr) |
| coupon_freq *(optional)* | Frequency |

If the trade file lacks coupon, maturity or frequency, take them from the **NSDL security master** (look up the ISIN) and add them as columns. Log NSDL as a separate source.

Keep only your sample ISINs (filter in Excel or pandas) so the files stay small.

> **The validation test lives here.** The exchange prints both price and yield. SpreadWatch recomputes the yield from the price; agreement within 5 bps proves the bond maths and conventions are right.

## Step 3: G-sec curve → `data/raw/gsec/`

**Publisher:** FBIL (Financial Benchmarks India Pvt Ltd), *G-Sec par yield curve* (daily). Download the daily files for your window. Either layout works:

* long: `date, tenor, yield`
* wide: `date, 0.25, 0.5, 1, 2, ... 30` (or `3M, 6M, 1Y, ...`)

Fallback: RBI DBIE government security yields. FBIL is the benchmark administrator, so prefer it.

## Step 4: Rating actions → `data/raw/ratings/`

**Publishers:** CRISIL Ratings, ICRA, CARE Ratings, India Ratings (each agency's *rating rationale / press release* pages, searchable by company).

For each issuer, open every rationale in your window and record one row per action:

```
issuer,agency,action_date,instrument,rating_from,rating_to,outlook,url
Example Housing Finance Ltd,CRISIL,2019-02-03,NCD,CRISIL AA+/Stable,CRISIL AA/Negative,Negative,https://...
```

* Include the **first rating you can find before the window starts** (with `rating_from` blank), so the transition matrix knows where each issuer began.
* Include reaffirmations: they show the agency looked and did not act.
* Use the agency's own symbols ("CRISIL AA+", "[ICRA]A-", "IND BBB"); the parser normalises them.
* Keep the issuer name **exactly** the same as in the trade files.

**Faster route with L2:** save the rationale PDFs in `data/raw/rationales/` and run `python extract.py`. It drafts `extracted_rating_actions.csv`, with the evidence sentence for each row. Check every row against the PDF, fix mistakes, set `verified` to `Y`, then save the checked file in `data/raw/ratings/` and register it in `sources.csv`. Hand-label 15–20 rationales and run `python extract.py --evaluate your_labels.csv` to report extraction accuracy in Chapter 10.

## Step 5: Cross-check the transition matrix (recommended)

SEBI requires agencies to publish transition and default statistics. Download one agency's published one-year transition matrix for a similar period. Compare its diagonal (stability) and default column with yours in Chapter 11, and explain the differences: your sample is small and deliberately includes many downgraded issuers.

## Step 6: Run

```bash
python ingest.py --mode real
python run_pipeline.py --mode real
python report.py --mode real
streamlit run app.py         # choose "Real data" in the sidebar
```

Check `outputs/real/rejected_trades.csv` and the validation tab. Explain every rejection category in Chapter 11.
