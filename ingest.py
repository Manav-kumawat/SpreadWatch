"""
L1 Ingest - load every input from a named public source and log where it came from.

Real mode
---------
Put the files you downloaded into:
    data/raw/trades/    NSE / BSE corporate bond reported-trade files (.csv / .xlsx)
    data/raw/gsec/      FBIL G-sec par yield curve files (.csv / .xlsx)
    data/raw/ratings/   rating actions you compiled from CRISIL / ICRA / CARE rationales (.csv)
    data/raw/master/    (optional) bond master: isin, issuer, coupon_pct, maturity_date, coupon_freq -
                        used to fill bond terms missing from the trade files, matched by ISIN

and add ONE row per file to sources.csv (source_id, file, publisher, url,
date_pulled, description, synthetic). Ingest REFUSES any raw file that is not
in sources.csv - an input with no provenance never reaches the engine.

Output: canonical CSVs in data/real/ plus outputs/ingest_log.csv (file hash,
rows read, rows kept) so every number can be traced back to a file and a URL.

Usage:
    python ingest.py --mode real
    python ingest.py --mode sample     (checks / regenerates the SYNTHETIC sample)
"""
import argparse
import hashlib
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from config import (CURVE_FILE, OUTPUT_DIR, RATINGS_FILE, RAW_DIR, REAL_DIR, ROOT, SAMPLE_DIR,
                    SOURCES_CSV, TRADES_FILE)

SOURCE_COLUMNS = ["source_id", "file", "publisher", "url", "date_pulled", "description", "synthetic"]

# Header aliases seen in exchange / FBIL downloads -> canonical column name.
TRADE_ALIASES = {
    "trade_date": ["trade_date", "trade date", "date", "deal date", "trade dt"],
    "isin": ["isin", "isin no", "isin code", "isin number"],
    "issuer": ["issuer", "issuer name", "name of issuer", "description", "security description",
               "security name", "symbol"],
    "coupon_pct": ["coupon_pct", "coupon", "coupon rate", "coupon (%)", "coupon rate (%)", "coupon(%)"],
    "coupon_freq": ["coupon_freq", "coupon frequency", "frequency", "interest frequency"],
    "maturity_date": ["maturity_date", "maturity date", "maturity", "redemption date", "maturity dt"],
    "price": ["price", "clean price", "trade price", "weighted average price", "wap", "wtd avg price",
              "last traded price", "ltp"],
    "printed_yield_pct": ["printed_yield_pct", "yield", "yield (%)", "traded yield", "weighted average yield",
                          "wa yield", "wtd avg yield", "ytm", "yield(%)"],
    "value_cr": ["value_cr", "trade value", "trade value (rs. cr)", "value (rs. cr)", "traded value (rs. cr)",
                 "amount (rs. cr)", "value (rs cr)", "value", "turnover (rs. cr)"],
}
RATING_ALIASES = {
    "issuer": ["issuer", "company", "company name", "issuer name"],
    "agency": ["agency", "cra", "rating agency"],
    "action_date": ["action_date", "date", "rating date", "press release date"],
    "instrument": ["instrument", "instrument type", "facility"],
    "rating_from": ["rating_from", "previous rating", "old rating", "from"],
    "rating_to": ["rating_to", "rating", "current rating", "new rating", "to"],
    "outlook": ["outlook"],
    "url": ["url", "rationale url", "link"],
}
FREQ_WORDS = {"annual": 1, "yearly": 1, "semi": 2, "half": 2, "quarter": 4, "month": 12}



def parse_dates(s: pd.Series) -> pd.Series:
    """
    ISO dates (2019-02-03) are read as year-month-day; everything else
    (03-02-2019, 03/02/2019, 03-Feb-2019) as day-first, the Indian convention.
    Reading ISO dates day-first would silently swap day and month.
    """
    txt = s.astype(str).str.strip()
    iso = pd.to_datetime(txt, format="%Y-%m-%d", errors="coerce")
    rest = pd.to_datetime(txt.where(iso.isna()), dayfirst=True, errors="coerce", format="mixed")
    return iso.fillna(rest).dt.date


class ProvenanceError(RuntimeError):
    pass


# ----------------------------------------------------------------------------
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_sources() -> pd.DataFrame:
    if not SOURCES_CSV.exists():
        pd.DataFrame(columns=SOURCE_COLUMNS).to_csv(SOURCES_CSV, index=False)
    src = pd.read_csv(SOURCES_CSV, dtype=str).fillna("")
    missing = [c for c in SOURCE_COLUMNS if c not in src.columns]
    if missing:
        raise ProvenanceError(f"sources.csv is missing columns {missing}")
    return src


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path)
    return pd.read_csv(path)


def rename_by_alias(df: pd.DataFrame, aliases: dict) -> pd.DataFrame:
    lookup = {re.sub(r"\s+", " ", str(c)).strip().lower(): c for c in df.columns}
    mapping = {}
    for canon, names in aliases.items():
        for n in names:
            if n in lookup and lookup[n] not in mapping:
                mapping[lookup[n]] = canon
                break
    return df.rename(columns=mapping)


def parse_tenor(label) -> float:
    """'3M' -> 0.25, '1Y' -> 1.0, '0.5' -> 0.5, '10 Yr' -> 10.0. Raises ValueError if not a tenor."""
    s = str(label).strip().upper().replace("YEARS", "Y").replace("YEAR", "Y").replace("YR", "Y")
    s = s.replace(" ", "")
    m = re.fullmatch(r"(\d+(?:\.\d+)?)([MY]?)", s)
    if not m:
        raise ValueError(label)
    v = float(m.group(1))
    return v / 12.0 if m.group(2) == "M" else v


def canonical_curve(df: pd.DataFrame) -> pd.DataFrame:
    """Accept long (date, tenor, yield) or wide (date + one column per tenor) curve files."""
    df = rename_by_alias(df, {"date": ["date", "as on date", "curve date", "trade date"],
                              "tenor_years": ["tenor_years", "tenor", "maturity (years)", "ttm"],
                              "yield_pct": ["yield_pct", "yield", "par yield", "yield (%)"]})
    if {"date", "tenor_years", "yield_pct"}.issubset(df.columns):
        out = df[["date", "tenor_years", "yield_pct"]].copy()
        out["tenor_years"] = out["tenor_years"].map(parse_tenor)
    else:
        tenor_cols = {}
        for c in df.columns:
            try:
                tenor_cols[c] = parse_tenor(c)
            except ValueError:
                continue
        if "date" not in df.columns or not tenor_cols:
            raise ValueError("curve file has neither long (date, tenor, yield) nor wide (date + tenor columns) layout")
        out = df.melt(id_vars="date", value_vars=list(tenor_cols), var_name="tenor", value_name="yield_pct")
        out["tenor_years"] = out["tenor"].map(tenor_cols)
        out = out.drop(columns="tenor")
    out["date"] = parse_dates(out["date"])
    out["yield_pct"] = pd.to_numeric(out["yield_pct"], errors="coerce")
    return out.dropna()


def canonical_trades(df: pd.DataFrame) -> pd.DataFrame:
    df = rename_by_alias(df, TRADE_ALIASES)
    for c in ["trade_date", "maturity_date"]:
        if c in df.columns:
            df[c] = parse_dates(df[c])
    if "coupon_freq" in df.columns and df["coupon_freq"].dtype == object:
        def f(v):
            s = str(v).lower()
            for k, n in FREQ_WORDS.items():
                if k in s:
                    return n
            try:
                return int(float(s))
            except ValueError:
                return 1
        df["coupon_freq"] = df["coupon_freq"].map(f)
    keep = [c for c in TRADE_ALIASES if c in df.columns]
    return df[keep]


def canonical_ratings(df: pd.DataFrame) -> pd.DataFrame:
    df = rename_by_alias(df, RATING_ALIASES)
    need = ["issuer", "agency", "action_date", "rating_to"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"rating file missing columns {miss}")
    df["action_date"] = parse_dates(df["action_date"])
    for c in ["rating_from", "instrument", "outlook", "url"]:
        if c not in df.columns:
            df[c] = ""
    return df[["issuer", "agency", "action_date", "instrument", "rating_from", "rating_to", "outlook", "url"]]


def canonical_master(df: pd.DataFrame) -> pd.DataFrame:
    """Bond master (one row per ISIN): issuer name, coupon, maturity, frequency - from NSDL / the term sheet."""
    df = rename_by_alias(df, {k: v for k, v in TRADE_ALIASES.items()
                              if k in ("isin", "issuer", "coupon_pct", "coupon_freq", "maturity_date")})
    if "isin" not in df.columns:
        raise ValueError("bond master needs an 'isin' column")
    df = canonical_trades(df)            # same header mapping and date parsing as trade files
    return df.drop_duplicates(subset="isin", keep="last")


def apply_master(trades: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    """
    Fill / override bond terms in the trade file from the bond master, by ISIN.
    The master's issuer name wins so that it matches the rating-actions file exactly.
    """
    t = trades.copy()
    m = master.set_index("isin")
    for col in ("issuer", "coupon_pct", "coupon_freq", "maturity_date"):
        if col not in m.columns:
            continue
        mapped = t["isin"].map(m[col])
        if col == "issuer" or col not in t.columns:
            t[col] = mapped.where(mapped.notna(), t[col] if col in t.columns else None)
        else:
            t[col] = t[col].where(t[col].notna(), mapped)
    return t


# ----------------------------------------------------------------------------
def ingest_real() -> pd.DataFrame:
    src = load_sources()
    registered = {Path(f).as_posix(): sid for f, sid in zip(src["file"], src["source_id"]) if f}
    groups = {"master": (canonical_master, None),
              "trades": (canonical_trades, TRADES_FILE),
              "gsec": (canonical_curve, CURVE_FILE),
              "ratings": (canonical_ratings, RATINGS_FILE)}
    master = None
    log, unregistered = [], []
    REAL_DIR.mkdir(parents=True, exist_ok=True)

    for sub, (fn, out_name) in groups.items():
        folder = RAW_DIR / sub
        files = sorted(p for p in folder.glob("*") if p.suffix.lower() in (".csv", ".xlsx", ".xls")) \
            if folder.exists() else []
        frames = []
        for p in files:
            rel = p.relative_to(ROOT).as_posix()
            sid = registered.get(rel)
            if sid is None:
                unregistered.append(rel)
                continue
            raw = read_table(p)
            canon = fn(raw)
            canon["source_id"] = sid
            frames.append(canon)
            log.append({"source_id": sid, "file": rel, "sha256": sha256(p), "rows_read": len(raw),
                        "rows_kept": len(canon), "ingested_at": datetime.now().isoformat(timespec="seconds")})
        if not frames:
            continue
        combined = pd.concat(frames, ignore_index=True).drop_duplicates()
        if sub == "master":
            master = combined.drop_duplicates(subset="isin", keep="last")
            continue
        if sub == "trades" and master is not None:
            combined = apply_master(combined, master)
        combined.to_csv(REAL_DIR / out_name, index=False)

    if unregistered:
        raise ProvenanceError(
            "These raw files are not registered in sources.csv, so they were refused:\n  "
            + "\n  ".join(unregistered)
            + "\nAdd a row (source_id, file, publisher, url, date_pulled, description, synthetic=N) for each."
        )
    missing_registered = [f for f in registered if f.startswith("data/raw/") and not (ROOT / f).exists()]
    if missing_registered:
        print("WARNING: sources.csv lists files that are not on disk:", missing_registered)

    OUTPUT_DIR.mkdir(exist_ok=True)
    log_df = pd.DataFrame(log)
    log_df.to_csv(OUTPUT_DIR / "ingest_log.csv", index=False)
    return log_df


def ingest_sample() -> pd.DataFrame:
    if not (SAMPLE_DIR / TRADES_FILE).exists():
        import make_sample_data
        make_sample_data.main()
    src = load_sources()
    if "SYN01" not in set(src["source_id"]):
        src = pd.concat([src, pd.DataFrame([{
            "source_id": "SYN01", "file": "make_sample_data.py", "publisher": "SYNTHETIC - generated by this project",
            "url": "n/a", "date_pulled": datetime.now().date().isoformat(),
            "description": "Synthetic trades, G-sec curve and rating actions for development and the demo only. "
                           "Not used for any finding.", "synthetic": "Y"}])], ignore_index=True)
        src.to_csv(SOURCES_CSV, index=False)
    rows = []
    for name in (TRADES_FILE, CURVE_FILE, RATINGS_FILE):
        p = SAMPLE_DIR / name
        rows.append({"source_id": "SYN01", "file": p.relative_to(ROOT).as_posix(), "sha256": sha256(p),
                     "rows_read": sum(1 for _ in open(p)) - 1, "rows_kept": None,
                     "ingested_at": datetime.now().isoformat(timespec="seconds")})
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["real", "sample"], default="real")
    args = ap.parse_args(argv)
    try:
        log = ingest_real() if args.mode == "real" else ingest_sample()
    except ProvenanceError as e:
        print(f"INGEST REFUSED: {e}")
        sys.exit(2)
    print(log.to_string(index=False) if not log.empty else "No raw files found - see DATA_COLLECTION.md")


if __name__ == "__main__":
    main()
