"""
Combine one-file-per-tenor G-sec yield histories into a single curve file.

Use this when your G-sec source gives one download per maturity (for example a
"India 5-Year Bond Yield Historical Data.csv" file for each of 1Y, 2Y, 3Y, 5Y,
7Y, 10Y, 15Y, 30Y) instead of one daily curve file.

    1. Put the per-tenor files in data/raw/gsec_tenors/ (one file per maturity;
       the maturity must appear in the file name: "1-Year", "5-Year", "3-Month" ...)
    2. python prepare_curve.py
    3. Register BOTH the per-tenor files and the combined file
       (data/raw/gsec/gsec_curve_combined.csv) in sources.csv - the combined
       file's description should say it is derived from the per-tenor files.

Expected columns in each file: a date column ("Date") and a yield column
("Price", "Close", "Yield" or "Yield (%)"). Month-first dates (03/29/2022)
are detected automatically.
"""
import re
import sys
from pathlib import Path

import pandas as pd

from config import RAW_DIR

SRC = RAW_DIR / "gsec_tenors"
OUT = RAW_DIR / "gsec" / "gsec_curve_combined.csv"
TENOR_IN_NAME = re.compile(r"(\d+(?:\.\d+)?)\s*[- _]?\s*(year|yr|y|month|mon|m)\b", re.I)
YIELD_COLS = ["price", "close", "yield", "yield (%)", "ytm", "last"]


def tenor_from_name(name: str) -> float:
    m = TENOR_IN_NAME.search(name.replace("_", " "))
    if not m:
        raise ValueError(f"cannot find a maturity such as '5-Year' or '3-Month' in the file name: {name}")
    v = float(m.group(1))
    return v / 12.0 if m.group(2).lower().startswith("m") else v


def parse_dates(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    for fmt in ("%m/%d/%Y", "%d-%m-%Y", "%Y-%m-%d", "%d/%m/%Y", "%b %d, %Y", "%d-%b-%Y", "%d %b %Y"):
        d = pd.to_datetime(s, format=fmt, errors="coerce")
        if d.notna().mean() > 0.95:
            return d
    return pd.to_datetime(s, errors="coerce")


def load_one(path: Path) -> pd.Series:
    df = pd.read_excel(path) if path.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(path)
    cols = {c.strip().lower(): c for c in df.columns}
    date_col = cols.get("date")
    ycol = next((cols[c] for c in YIELD_COLS if c in cols), None)
    if date_col is None or ycol is None:
        raise ValueError(f"{path.name}: need a 'Date' column and one of {YIELD_COLS}; found {list(df.columns)}")
    d = parse_dates(df[date_col])
    y = pd.to_numeric(df[ycol].astype(str).str.replace("%", "").str.replace(",", ""), errors="coerce")
    out = pd.Series(y.values, index=d.values).dropna()
    out = out[~out.index.isna()]
    return out[~out.index.duplicated()].sort_index()


def combine(src: Path = SRC) -> pd.DataFrame:
    files = sorted(p for p in src.glob("*") if p.suffix.lower() in (".csv", ".xlsx", ".xls"))
    if not files:
        raise FileNotFoundError(f"no files in {src}")
    series = {}
    for p in files:
        t = tenor_from_name(p.name)
        if t in series:
            raise ValueError(f"two files for the same maturity {t}y: check {p.name}")
        series[t] = load_one(p)
        print(f"  {p.name}: {t:g} years, {len(series[t])} days, {series[t].index.min().date()} to {series[t].index.max().date()}")
    wide = pd.DataFrame(series).sort_index(axis=1)
    wide = wide.dropna(thresh=max(2, int(0.6 * wide.shape[1])))   # keep days where most tenors are present
    wide.index.name = "date"
    wide.columns = [f"{c:g}" for c in wide.columns]
    return wide.reset_index().assign(date=lambda d: d["date"].dt.date)


if __name__ == "__main__":
    try:
        w = combine()
    except (FileNotFoundError, ValueError) as e:
        print("ERROR:", e)
        sys.exit(1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    w.to_csv(OUT, index=False)
    print(f"Combined curve: {len(w)} days x {w.shape[1] - 1} tenors -> {OUT}")
    print("Now register this file AND the per-tenor files in sources.csv.")
