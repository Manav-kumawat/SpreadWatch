"""
Build the model's real-data inputs from the downloaded public files.

Run after tools/fetch_nse_cbm.py, tools/fetch_fbil_gsec.py, tools/crisil_history.py
and tools/care_history.py:

    python tools/build_real_dataset.py

1. Reads every NSE CBM security master in data/raw/original/nse_cbm/master/ and
   keeps PLAIN bonds only: fixed coupon > 0, coupon paid yearly / half-yearly /
   quarterly / monthly, Actual/365, no step-up, no floating benchmark, and not
   tax-free (type PF or 'TF'/'TAX FREE' in the description), perpetual (DP),
   zero-coupon (DC) or market-linked.
2. Maps NSE issuer names to the sample issuers (ISSUERS below).
3. Reads every daily bhavcopy, keeps those ISINs, and for each issuer picks the
   MAX_ISINS bonds with the most trading days (the most liquid ones).
4. Writes
     data/raw/master/bond_master.csv          isin, issuer, coupon_pct, maturity_date, coupon_freq
     data/raw/trades/nse_cbm_trades.csv       trade_date, isin, price, printed_yield_pct, value_cr
5. Registers these files, the FBIL curve and the rating files in sources.csv.

Selection rules are written here so the sample can be defended in the viva:
nothing is chosen by hand after looking at results.
"""
import csv
import re
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
NSE = RAW / "original" / "nse_cbm"
MAX_ISINS = 3            # stable issuers
MAX_ISINS_EVENT = 5      # downgraded issuers (bonds trade thinly before a downgrade)
MIN_TRADE_DAYS = 15
WINDOW = ("2018-01-01", "2022-06-30")

# NSE issuer name (regex, case-insensitive) -> name used everywhere in the project
ISSUERS = {
    r"^dewan housing finance": "Dewan Housing Finance Corporation Ltd",
    r"^reliance home finance": "Reliance Home Finance Ltd",
    r"^indiabulls housing finance": "Indiabulls Housing Finance Ltd",
    r"^vodafone idea": "Vodafone Idea Ltd",
    r"^tata motors limited": "Tata Motors Ltd",
    r"^piramal capital & (?:hsg|housing)": "Piramal Capital & Housing Finance Ltd",
    r"^housing development finance": "Housing Development Finance Corporation Ltd",
    r"^lic housing finance": "LIC Housing Finance Ltd",
    r"^reliance industries": "Reliance Industries Ltd",
    r"^power finance corporation limited": "Power Finance Corporation Ltd",
    r"^(?:rec ltd|rec limited|rural electrification)": "REC Ltd",
    r"^indian railway finance": "Indian Railway Finance Corporation Ltd",
    r"^tata motors finance limited": "Tata Motors Finance Ltd",
    r"^pnb housing finance": "PNB Housing Finance Ltd",
    r"^ecl finance": "ECL Finance Ltd",
}
FREQ = {"yearly": 1, "half yearly": 2, "quarterly": 4, "monthly": 12}
PLAIN_TYPES = {"DB", "BB", "PT", "ID", "BP"}
EXCLUDE_DESC = re.compile(r"(?i)tax\s*free|\bTF\b|perp|step|mld|market link|zero|\b0%")


def load_masters() -> pd.DataFrame:
    """
    All security-master snapshots, oldest first. Each ISIN keeps the record from
    the EARLIEST snapshot it appears in: after the 2021 DHFL-Piramal merger, old
    DHFL bonds are listed under the Piramal name in later snapshots, and we want
    the issuer as it was when the bond was issued.
    """
    files = sorted((NSE / "master").glob("Corporate_bond_report_*.csv"),
                   key=lambda f: pd.to_datetime(f.stem.replace("Corporate_bond_report_", ""), format="%d-%b-%Y"))
    frames = []
    for f in files:
        m = pd.read_csv(f, skiprows=5, dtype=str)
        m.columns = [c.strip() for c in m.columns]
        m["master_file"] = f.name
        frames.append(m)
    m = pd.concat(frames, ignore_index=True)
    m["ISIN"] = m["ISIN"].str.strip()
    return m.drop_duplicates("ISIN", keep="first")


def plain_bonds(m: pd.DataFrame) -> pd.DataFrame:
    m = m.copy()
    m["issuer_std"] = None
    for pat, name in ISSUERS.items():
        m.loc[m["Issuer"].str.lower().str.strip().str.contains(pat, regex=True, na=False), "issuer_std"] = name
    m = m[m["issuer_std"].notna()]
    m["coupon_pct"] = pd.to_numeric(m["Issue Name"].str.extract(r"^\s*(\d+(?:\.\d+)?)\s*%\s*$")[0], errors="coerce")
    m["coupon_freq"] = m["Coupon Frequency"].str.strip().str.lower().map(FREQ)
    keep = (m["Sectype"].isin(PLAIN_TYPES) & (m["coupon_pct"] > 0) & m["coupon_freq"].notna()
            & (m["Day Count Convention"].str.strip() == "ACTUALby365")
            & m["Step up Coupons"].isna() & m["Floating Benchmark"].isna()
            & ~m["Issue Desc"].str.contains(EXCLUDE_DESC, na=False))
    out = m[keep].copy()
    out["maturity_date"] = pd.to_datetime(out["Maturity Date"], format="%d-%b-%Y", errors="coerce").dt.date
    return out[["ISIN", "issuer_std", "coupon_pct", "maturity_date", "coupon_freq", "Issue Desc", "master_file"]]


def load_bhavcopies(isins) -> pd.DataFrame:
    frames = []
    for f in sorted((NSE / "bhavcopy").glob("cbm_trd*.csv")):
        b = pd.read_csv(f, dtype=str)
        b.columns = [" ".join(c.split()) for c in b.columns]
        b["ISIN"] = b["ISIN"].str.strip()
        b = b[b["ISIN"].isin(isins)]
        if len(b):
            frames.append(b)
    b = pd.concat(frames, ignore_index=True)
    out = pd.DataFrame({
        "trade_date": pd.to_datetime(b["Trade Date"], format="%d-%b-%Y").dt.date,
        "isin": b["ISIN"],
        "price": pd.to_numeric(b["Weighted Average Price (Rs.)"], errors="coerce"),
        "printed_yield_pct": pd.to_numeric(b["Weighted Average Yield (YTM) (%)"], errors="coerce"),
        "value_cr": pd.to_numeric(b["Total Trade Value (Rs. in lacs)"], errors="coerce") / 100.0,
    })
    out.loc[out["printed_yield_pct"] <= 0, "printed_yield_pct"] = None    # 0.0000 = yield not computed by NSE
    return out[(out["price"] > 0)]


def first_downgrades() -> dict:
    """Date of each issuer's first downgrade or default in the rating files (any agency)."""
    import sys
    sys.path.insert(0, str(ROOT))
    from engine.ratings import classify_action
    frames = [pd.read_csv(f, dtype=str).fillna("") for f in (RAW / "ratings").glob("*.csv")]
    if not frames:
        return {}
    r = pd.concat(frames)
    r["action"] = [classify_action(a, b) for a, b in zip(r["rating_from"], r["rating_to"])]
    r = r[r["action"].isin(["downgrade", "default"])]
    r = r[(pd.to_datetime(r["action_date"]) >= WINDOW[0]) & (pd.to_datetime(r["action_date"]) <= WINDOW[1])]
    return pd.to_datetime(r.groupby("issuer")["action_date"].min()).to_dict()


def register(rows):
    src = ROOT / "sources.csv"
    old = pd.read_csv(src, dtype=str).fillna("")
    new = pd.DataFrame(rows)
    out = pd.concat([old[~old["source_id"].isin(new["source_id"])], new], ignore_index=True)
    out.to_csv(src, index=False, quoting=csv.QUOTE_MINIMAL)


def first_download_date(log: Path) -> str:
    d = pd.read_csv(log)
    return str(pd.to_datetime(d["downloaded_at"]).min().date()) if len(d) else str(date.today())


def main():
    master = plain_bonds(load_masters())
    print(f"{len(master)} plain bonds of sample issuers in the NSE security masters")
    trades = load_bhavcopies(set(master["ISIN"]))
    trades = trades[(pd.to_datetime(trades["trade_date"]) >= WINDOW[0]) & (pd.to_datetime(trades["trade_date"]) <= WINDOW[1])]
    days = trades.groupby("isin")["trade_date"].nunique().rename("trade_days")
    master = master.merge(days, left_on="ISIN", right_index=True, how="inner")
    master = master[master["trade_days"] >= MIN_TRADE_DAYS]

    # Liquidity score. For an issuer that was downgraded, what matters is trading in the
    # year before (and just after) its FIRST downgrade, so bonds are ranked on trading
    # days in [first downgrade - 400 days, first downgrade + 60 days]. Other issuers are
    # ranked on trading days over the whole window. The rule uses rating dates only,
    # never spread results.
    first_dg = first_downgrades()
    t_dates = pd.to_datetime(trades["trade_date"])
    score = {}
    for isin, issuer in zip(master["ISIN"], master["issuer_std"]):
        mask = trades["isin"] == isin
        if issuer in first_dg:
            d0 = first_dg[issuer]
            mask &= (t_dates >= d0 - pd.Timedelta(days=400)) & (t_dates <= d0 + pd.Timedelta(days=60))
        score[isin] = trades.loc[mask, "trade_date"].nunique()
    master["liquidity_score"] = master["ISIN"].map(score)
    master["n_max"] = master["issuer_std"].map(lambda i: MAX_ISINS_EVENT if i in first_dg else MAX_ISINS)
    master = master[master["liquidity_score"] > 0]
    chosen = (master.sort_values(["issuer_std", "liquidity_score"], ascending=[True, False])
              .groupby("issuer_std", group_keys=False).apply(lambda g: g.head(int(g["n_max"].iloc[0]))))

    bm = chosen.rename(columns={"ISIN": "isin", "issuer_std": "issuer"})[
        ["isin", "issuer", "coupon_pct", "maturity_date", "coupon_freq", "Issue Desc", "trade_days", "liquidity_score",
         "master_file"]]
    bm = bm.rename(columns={"Issue Desc": "nse_description"})
    (RAW / "master").mkdir(parents=True, exist_ok=True)
    bm.to_csv(RAW / "master" / "bond_master.csv", index=False)
    t = trades[trades["isin"].isin(bm["isin"])].sort_values(["isin", "trade_date"])
    (RAW / "trades").mkdir(parents=True, exist_ok=True)
    t.to_csv(RAW / "trades" / "nse_cbm_trades.csv", index=False)
    print(bm.groupby("issuer").agg(bonds=("isin", "size"), trade_days=("trade_days", "sum")).to_string())
    print(f"{len(t)} bond-day trade records -> data/raw/trades/nse_cbm_trades.csv")

    nse_date = first_download_date(NSE / "download_log.csv")
    fbil_log = RAW / "original" / "fbil" / "download_log.csv"
    rows = [
        dict(source_id="NSE01", file="data/raw/trades/nse_cbm_trades.csv", publisher="National Stock Exchange of India",
             url="https://nsearchives.nseindia.com/archives/debt/cbm/cbm_trd<YYYYMMDD>.csv", date_pulled=nse_date,
             description=f"NSE Corporate Bond Market daily bhavcopies {WINDOW[0]} to {WINDOW[1]} (originals and download log in "
                         f"data/raw/original/nse_cbm/), filtered by tools/build_real_dataset.py to {len(bm)} sample ISINs; "
                         "weighted average price and yield, total traded value", synthetic="N"),
        dict(source_id="NSE02", file="data/raw/master/bond_master.csv", publisher="National Stock Exchange of India",
             url="https://nsearchives.nseindia.com/content/debt/Corporate_bond_report_<DD-Mon-YYYY>.csv", date_pulled=nse_date,
             description="NSE CBM security master (issuer, coupon, maturity, frequency, day count); plain bonds of sample "
                         "issuers, most-traded bonds selected by tools/build_real_dataset.py", synthetic="N"),
        dict(source_id="FBIL01", file="data/raw/gsec/fbil_par_yield.csv", publisher="Financial Benchmarks India Pvt Ltd",
             url="https://www.fbil.org.in/wasdm/gsec/downloadPublished?date=<YYYY-MM-DD>",
             date_pulled=first_download_date(fbil_log) if fbil_log.exists() else nse_date,
             description="FBIL G-Sec par yield curve (semi-annual YTM, tenors 0.25-40y), last published day of each week, "
                         "extracted from the daily 'FBIL GOI Prices including Par Yield' files by tools/fetch_fbil_gsec.py",
             synthetic="N"),
    ]
    for fname, sid, pub in [("crisil_rating_actions.csv", "CRISIL01", "CRISIL Ratings Ltd"),
                            ("care_rating_actions.csv", "CARE01", "CARE Ratings Ltd")]:
        if (RAW / "ratings" / fname).exists():
            rows.append(dict(source_id=sid, file=f"data/raw/ratings/{fname}", publisher=pub,
                             url="Rationale URL on every row; copies in data/raw/rationales/", date_pulled=nse_date,
                             description=f"Rating actions read from {pub} rationale 'rating history' tables by "
                                         f"tools/{'crisil' if sid.startswith('CRISIL') else 'care'}_history.py", synthetic="N"))
    register(rows)
    print("sources.csv updated:", [r["source_id"] for r in rows])


if __name__ == "__main__":
    main()
