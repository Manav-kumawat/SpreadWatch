"""
Download FBIL's published G-sec valuation files and extract the par yield curve.

    python tools/fetch_fbil_gsec.py --start 2018-03-31 --end 2022-06-30 --every week

FBIL (Financial Benchmarks India Pvt Ltd) publishes, every Mumbai business day,
"FBIL GOI Prices including Par Yield". The public site lists the published
days at
    https://www.fbil.org.in/wasdm/gsec/fetchfiltered?fromDate=YYYY-MM-DD&toDate=YYYY-MM-DD&authenticated=false
and serves each day's Excel file at
    https://www.fbil.org.in/wasdm/gsec/downloadPublished?date=YYYY-MM-DD

The 'Par-Yield' sheet holds tenors 0.25 to 40 years with the YTM on a
semi-annual basis and on an annualised basis.

Outputs
  data/raw/original/fbil/FBIL_GSec_<YYYY-MM-DD>.xls     untouched downloads
  data/raw/original/fbil/download_log.csv              provenance log
  data/raw/gsec/fbil_par_yield.csv                     date, tenor_years, yield_pct
                                                       (semi-annual basis -> GSEC_COMPOUNDING = 2)
"""
import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
ORIG = ROOT / "data" / "raw" / "original" / "fbil"
OUT = ROOT / "data" / "raw" / "gsec" / "fbil_par_yield.csv"
BASE = "https://www.fbil.org.in/wasdm/gsec"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36")


def published_dates(s, start, end):
    """All dates FBIL published a G-sec file for, queried a month at a time."""
    out = set()
    for m0 in pd.date_range(pd.Timestamp(start).replace(day=1), end, freq="MS"):
        m1 = min(m0 + pd.offsets.MonthEnd(0), pd.Timestamp(end))
        r = s.get(f"{BASE}/fetchfiltered", params={"fromDate": f"{max(m0, pd.Timestamp(start)):%Y-%m-%d}",
                                                   "toDate": f"{m1:%Y-%m-%d}", "authenticated": "false"},
                  timeout=60)
        r.raise_for_status()
        out |= {pd.Timestamp(x["processRunDate"]) for x in r.json()}
        time.sleep(0.3)
    return sorted(out)


def par_curve(xls_path: Path) -> pd.DataFrame:
    raw = pd.read_excel(xls_path, sheet_name="Par-Yield", header=None)
    # the header cell reads "Tenor (Year)" from Aug 2018 and "Maturity" before that
    hdr = raw.index[raw[0].astype(str).str.contains("Tenor|Maturity", case=False, na=False)][0]
    body = raw.iloc[hdr + 1:, :3].copy()
    body.columns = ["tenor_years", "ytm_semi_annual", "ytm_annualised"]
    body = body.apply(pd.to_numeric, errors="coerce").dropna()
    return body


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2018-03-31")
    ap.add_argument("--end", default="2022-06-30")
    ap.add_argument("--every", choices=["day", "week"], default="week",
                    help="'week' keeps the last published day of each week")
    a = ap.parse_args(argv)

    s = requests.Session()
    s.headers.update({"User-Agent": UA, "Referer": "https://www.fbil.org.in/"})
    days = published_dates(s, a.start, a.end)
    if a.every == "week":
        days = list(pd.Series(days).groupby(pd.Series(days).dt.to_period("W-FRI")).max())
    print(f"{len(days)} files to fetch")

    ORIG.mkdir(parents=True, exist_ok=True)
    new_log = not (ORIG / "download_log.csv").exists()
    frames = []
    with open(ORIG / "download_log.csv", "a", newline="") as fh:
        log = csv.writer(fh)
        if new_log:
            log.writerow(["file", "url", "http_status", "bytes", "downloaded_at"])
        for i, d in enumerate(days, 1):
            path = ORIG / f"FBIL_GSec_{d:%Y-%m-%d}.xls"
            url = f"{BASE}/downloadPublished?date={d:%Y-%m-%d}"
            if not path.exists():
                r = s.get(url, timeout=90)
                log.writerow([path.name, url, r.status_code, len(r.content), datetime.now().isoformat(timespec="seconds")])
                if r.status_code != 200 or len(r.content) < 1000:
                    continue
                path.write_bytes(r.content)
                time.sleep(0.4)
            try:
                c = par_curve(path)
            except Exception as e:  # unreadable file: logged, skipped
                print(f"  {path.name}: could not read Par-Yield sheet ({e})")
                continue
            frames.append(pd.DataFrame({"date": d.date(), "tenor_years": c["tenor_years"],
                                        "yield_pct": c["ytm_semi_annual"]}))
            if i % 25 == 0:
                print(f"  {i}/{len(days)}", flush=True)
    curve = pd.concat(frames, ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    curve.to_csv(OUT, index=False)
    print(f"{curve['date'].nunique()} curve dates, {len(curve)} rows -> {OUT}")


if __name__ == "__main__":
    main()
