"""
Turn CARE Ratings press releases (PDF) into rating-action rows.

CARE press releases carry "Annexure-2: Rating History of last three years":
for each instrument, every rating with its date in brackets, e.g.
    "3)CARE AA+ (Under Credit watch with Developing Implications) (03-Feb-19)".
This script downloads each PDF, keeps an untouched copy, takes the long-term
debenture (or long-term loan) row of that annexure, and writes one row per
dated action, plus the rating given in the press release itself on its date.

    python tools/care_history.py --list tools/care_docs.csv     # columns: issuer,url

Output: data/raw/ratings/care_rating_actions.csv (same columns as crisil_history.py)
"""
import argparse
import io
import re
from pathlib import Path

import pandas as pd
import requests
from pypdf import PdfReader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.ratings import NOTCH, normalise  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SAVE = ROOT / "data" / "raw" / "rationales"
OUT = ROOT / "data" / "raw" / "ratings" / "care_rating_actions.csv"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36")
SYM = r"(AAA|AA|A|BBB|BB|B|C|D)\s?([+-])?(?![A-Za-z0-9])"
DATED = re.compile(r"CARE\s+" + SYM + r"((?:(?!CARE\s).){0,160}?)\((\d{1,2}-[A-Za-z]{3}-\d{2,4})\)")
WITHDRAWN = re.compile(r"Withdrawn\s*\((\d{1,2}-[A-Za-z]{3}-\d{2,4})\)", re.I)
MONTH_DATE = re.compile(r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}")


def outlook_of(s):
    s = " ".join(s.split())
    if re.search(r"(?i)negative implications", s):
        return "Watch Negative"
    if re.search(r"(?i)developing implications", s):
        return "Watch Developing"
    m = re.search(r"(?i)\b(Stable|Negative|Positive)\b", s)
    return m.group(1).title() if m else ""


def parse(text):
    flat = " ".join(text.split())
    md = MONTH_DATE.search(flat)
    doc_date = pd.to_datetime(md.group(0)).date() if md else None
    a = flat.find("Annexure-2")
    a = flat.find("Rating History", a if a >= 0 else 0)
    if a < 0:
        raise ValueError("no rating history annexure")
    block = flat[a:]
    # split into numbered instrument rows: "1. Name ... LT", "2. ..."
    parts = re.split(r"\s(\d{1,2})\.\s(?=[A-Z])", block)
    rows = []
    for k in range(1, len(parts) - 1, 2):
        body = parts[k + 1]
        name = body[:80]
        if re.search(r"\bLT\b", body[:160]) and not re.search(r"(?i)commercial paper|short term", name):
            rows.append(body)
    if not rows:
        raise ValueError("no long-term row in the rating history")
    pref = [r for r in rows if re.search(r"(?i)debentures|bonds", r[:80])] or rows

    def events(r):
        ev = {}
        for m in DATED.finditer(r):
            d = pd.to_datetime(m.group(4), format="%d-%b-%y" if len(m.group(4).split("-")[-1]) == 2 else "%d-%b-%Y").date()
            ev.setdefault(d, ("CARE " + m.group(1) + (m.group(2) or ""), outlook_of(m.group(3))))
        for m in WITHDRAWN.finditer(r):
            d = pd.to_datetime(m.group(1), format="%d-%b-%y").date()
            ev.setdefault(d, ("Withdrawn", ""))
        return ev

    best = max(pref, key=lambda r: len(events(r)))
    ev = events(best)
    # The action of this release itself: the WORST current long-term rating across all
    # rows (an issuer with any long-term debt rated D is in default).
    current = []
    for r in rows:
        cur = re.search(r"\bLT\s+[\d,.]+\s+CARE\s+" + SYM, r)
        if cur:
            current.append("CARE " + cur.group(1) + (cur.group(2) or ""))
    if current and doc_date:
        worst = max(current, key=lambda x: NOTCH.get(normalise(x), 0))
        ev[doc_date] = (worst, "")
    name = " ".join(best[:70].split(" LT ")[0].split())
    return name, doc_date, dict(sorted(ev.items()))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", required=True)
    a = ap.parse_args(argv)
    s = requests.Session()
    s.headers["User-Agent"] = UA
    rows = []
    for job in pd.read_csv(a.list).itertuples():
        try:
            r = s.get(job.url, timeout=90)
            r.raise_for_status()
            text = "\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(r.content)).pages)
            name, ddate, ev = parse(text)
        except Exception as e:
            print(f"  FAILED {job.issuer}: {e}")
            continue
        SAVE.mkdir(parents=True, exist_ok=True)
        (SAVE / f"CARE_{re.sub(r'[^A-Za-z0-9]+', '', job.issuer)}_{ddate}.pdf").write_bytes(r.content)
        for d, (sym, ol) in ev.items():
            rows.append(dict(issuer=job.issuer, agency="CARE", action_date=d.isoformat(), instrument=name,
                             rating_from="", rating_to=sym, outlook=ol, url=job.url,
                             source_type="agency_rationale" if d == ddate else "rating_history_table",
                             notes=f"from press release dated {ddate}"))
        print(f"  {job.issuer}: {len(ev)} rows ({name})")
    old = pd.read_csv(OUT, dtype=str).fillna("") if OUT.exists() else pd.DataFrame()
    df = pd.concat([old, pd.DataFrame(rows)], ignore_index=True).astype(str)
    df = df.sort_values(["issuer", "action_date", "source_type"]).drop_duplicates(["issuer", "agency", "action_date"])
    df["rating_from"] = df.groupby(["issuer", "agency"])["rating_to"].shift(1).fillna("")
    df.to_csv(OUT, index=False)
    print(f"{len(df)} rows -> {OUT}")


if __name__ == "__main__":
    main()
