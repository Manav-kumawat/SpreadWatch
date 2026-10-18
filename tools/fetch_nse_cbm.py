"""
Download NSE's daily corporate bond files (public archives) for a date range.

    python tools/fetch_nse_cbm.py --start 2018-01-01 --end 2022-06-30

Two kinds of file are saved, untouched, under data/raw/original/nse_cbm/:

  bhavcopy/cbm_trd<YYYYMMDD>.csv
      NSE Corporate Bond Market daily bhavcopy: one row per ISIN traded that
      day - last trade price/yield, weighted average price, weighted average
      yield (YTM) and traded value.
      https://nsearchives.nseindia.com/archives/debt/cbm/cbm_trd<YYYYMMDD>.csv

  master/Corporate_bond_report_<DD-Mon-YYYY>.csv     (with --master-dates)
      NSE CBM security master: issuer, coupon, issue and maturity dates,
      coupon frequency and day-count convention for every listed bond.
      https://nsearchives.nseindia.com/content/debt/Corporate_bond_report_<DD-Mon-YYYY>.csv

Holidays return 404 and are skipped. A download log (file, URL, HTTP status,
bytes, time) is written next to the files - it is the provenance evidence for
sources.csv. Requests are spaced out to be polite to the exchange's servers.
"""
import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "original" / "nse_cbm"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36")
BHAV = "https://nsearchives.nseindia.com/archives/debt/cbm/cbm_trd{d:%Y%m%d}.csv"
MASTER = "https://nsearchives.nseindia.com/content/debt/Corporate_bond_report_{d:%d-%b-%Y}.csv"


def session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": UA, "Accept": "*/*", "Accept-Language": "en-US,en;q=0.9",
                      "Referer": "https://www.nseindia.com/"})
    try:
        s.get("https://www.nseindia.com/api/marketStatus", timeout=20)   # sets the site cookies
    except requests.RequestException:
        pass
    return s


def fetch(s, url, path, log, pause):
    if path.exists() and path.stat().st_size > 0:
        return "exists"
    for attempt in range(3):
        try:
            r = s.get(url, timeout=30)
            break
        except requests.RequestException:
            time.sleep(2 + attempt * 3)
    else:
        log.writerow([path.name, url, "ERROR", 0, datetime.now().isoformat(timespec="seconds")])
        return "error"
    ok = r.status_code == 200 and not r.text.lstrip().lower().startswith("<!doctype html")
    if ok:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(r.content)
    log.writerow([path.name, url, r.status_code if ok else f"{r.status_code} (no file)", len(r.content) if ok else 0,
                  datetime.now().isoformat(timespec="seconds")])
    time.sleep(pause)
    return "ok" if ok else "missing"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--master-dates", nargs="*", default=[], help="dates for security-master snapshots, YYYY-MM-DD")
    ap.add_argument("--pause", type=float, default=0.4, help="seconds between requests")
    a = ap.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    new_log = not (OUT / "download_log.csv").exists()
    with open(OUT / "download_log.csv", "a", newline="") as fh:
        log = csv.writer(fh)
        if new_log:
            log.writerow(["file", "url", "http_status", "bytes", "downloaded_at"])
        s = session()
        counts = {}
        days = pd.bdate_range(a.start, a.end)
        for i, d in enumerate(days, 1):
            res = fetch(s, BHAV.format(d=d), OUT / "bhavcopy" / f"cbm_trd{d:%Y%m%d}.csv", log, a.pause)
            counts[res] = counts.get(res, 0) + 1
            if i % 50 == 0:
                print(f"  {i}/{len(days)} days {counts}", flush=True)
                fh.flush()
        for ds in a.master_dates:
            d = pd.Timestamp(ds)
            res = fetch(s, MASTER.format(d=d), OUT / "master" / f"Corporate_bond_report_{d:%d-%b-%Y}.csv", log, a.pause)
            print(f"  master {d.date()}: {res}")
    print("done:", counts)


if __name__ == "__main__":
    sys.exit(main())
