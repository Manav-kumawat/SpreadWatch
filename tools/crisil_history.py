"""
Turn CRISIL rating rationales into rating-action rows.

Every CRISIL rationale ends with a table "Rating History for last 3 Years"
listing, for each instrument, every rating action (date + rating) in the
current and previous three calendar years, plus the rating at the start of
the oldest year. This script downloads each rationale given on the command
line (or listed in a text file), saves an untouched copy, reads the long-term
debt row of that table and writes one row per dated action.

    python tools/crisil_history.py --issuer "Indiabulls Housing Finance Ltd" URL [URL ...]
    python tools/crisil_history.py --list tools/crisil_docs.csv     # columns: issuer,url

Output: data/raw/ratings/crisil_rating_actions.csv (appended / de-duplicated)
        columns: issuer, agency, action_date, instrument, rating_from, rating_to,
                 outlook, url, source_type, notes
The rows are a machine-read draft: check a sample against the documents.
"""
import argparse
import html
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
SAVE = ROOT / "data" / "raw" / "rationales"
OUT = ROOT / "data" / "raw" / "ratings" / "crisil_rating_actions.csv"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36")
LT_PREFERENCE = ["non convertible debentures", "bonds", "long term borrowing", "subordinated debt",
                 "retail bond", "debentures", "long term", "fund based", "term loan"]
DATE = re.compile(r"^\d{2}-\d{2}-\d{2,4}$")
RATING = re.compile(r"CRISIL\s+(AAA|AA|A|BBB|BB|B|C|D)\s?[+-]?(\s?\((?:CE|SO)\))?(/[A-Za-z ]+)?|Withdrawn|Suspended", re.I)


def cell_text(s):
    s = re.sub(r"<[^>]+>", " ", s)
    return " ".join(html.unescape(s).split())


def lt_part(rating):
    """'CRISIL A1+ / CRISIL AAA/Stable' -> 'CRISIL AAA/Stable' (keep the long-term part)."""
    if not rating or rating in ("--", "-"):
        return None
    parts = [p.strip() for p in re.split(r"\s/\s", rating)]
    for p in parts:
        m = RATING.search(p)
        if m and not re.search(r"A[1-4]\+?(?!\w)", m.group(0).replace("CRISIL", "")):
            return m.group(0).strip()
    m = RATING.search(rating)
    return m.group(0).strip() if m else None


def split_outlook(r):
    if r is None:
        return None, ""
    if "/" in r:
        sym, outlook = r.split("/", 1)
        return sym.strip(), outlook.strip()
    return r.strip(), ""


def doc_date(txt):
    m = re.search(r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}", txt)
    return pd.to_datetime(m.group(0)).date() if m else None


def parse(raw_html: str):
    txt = cell_text(raw_html[:20000])
    ddate = doc_date(txt)
    i = raw_html.lower().find("rating history")
    if i < 0:
        raise ValueError("no 'Rating History' section")
    tbl = raw_html[i:]
    j = tbl.lower().find("</table>")
    tbl = tbl[:j] if j > 0 else tbl
    rows = []
    for tr in re.findall(r"(?is)<tr[^>]*>(.*?)</tr>", tbl):
        cells = [cell_text(c) for c in re.findall(r"(?is)<t[dh][^>]*>(.*?)</t[dh]>", tr)]
        if cells:
            rows.append(cells)
    # header row with years, e.g. ['', 'Current', '2021 (History)', '2020', '2019', '2018', 'Start of 2018']
    start_label = next((c for r in rows for c in r if c.lower().startswith("start of")), "")
    start_year = int(re.search(r"\d{4}", start_label).group(0)) if start_label else None

    # group rows by instrument (continuation rows have an empty first cell)
    groups, cur = [], None
    for r in rows:
        if len(r) >= 3 and r[0] and r[1] in ("LT", "ST", "LT/ST"):
            cur = {"name": r[0], "type": r[1], "rows": [r]}
            groups.append(cur)
        elif cur is not None and len(r) >= 3 and not r[0]:
            cur["rows"].append(r)
    lt = [g for g in groups if g["type"].startswith("LT")]
    if not lt:
        raise ValueError("no long-term instrument in the history table")
    def history(g):
        events = {}
        first = g["rows"][0]
        current = lt_part(first[3]) if len(first) > 3 else None
        if current and ddate:
            events[ddate] = current
        for r in g["rows"]:
            for k in range(4, len(r) - 1):
                if DATE.match(r[k]):
                    rt = lt_part(r[k + 1])
                    if rt:
                        d = datetime.strptime(r[k], "%d-%m-%y" if len(r[k]) == 8 else "%d-%m-%Y").date()
                        events.setdefault(d, rt)
        start = lt_part(first[-1]) if start_year else None
        return dict(sorted(events.items())), start

    # Use the long-term instrument with the longest rating history (an instrument
    # first rated recently would miss earlier actions); ties go to bonds/NCDs.
    def rank(g):
        ev, start = history(g)
        pref = next((k for k, p in enumerate(LT_PREFERENCE) if p in g["name"].lower()), 99)
        return (-(len(ev) + (1 if start else 0)), min(ev) if ev else datetime.max.date(), pref)
    # Perpetual / subordinated / Tier I-II / hybrid instruments are rated below senior
    # debt; mixing them with senior ratings would create false rating moves.
    senior = [g for g in lt if not re.search(r"(?i)perpetual|subordinat|tier|hybrid|infrastructure bond", g["name"])]
    g = sorted(senior or lt, key=rank)[0]
    events, start = history(g)
    return g["name"], ddate, start_year, start, events


def rows_for(issuer, url, s):
    r = s.get(url, timeout=60)
    r.raise_for_status()
    raw = r.content.decode("utf-8", "ignore")
    instrument, ddate, start_year, start, events = parse(raw)
    SAVE.mkdir(parents=True, exist_ok=True)
    fname = f"CRISIL_{re.sub(r'[^A-Za-z0-9]+', '', issuer)}_{ddate}.html"
    (SAVE / fname).write_bytes(r.content)
    out = []
    if start:
        sym, ol = split_outlook(start)
        out.append(dict(issuer=issuer, agency="CRISIL", action_date=f"{start_year - 1}-12-31", instrument=instrument,
                        rating_from="", rating_to=sym, outlook=ol, url=url, source_type="rating_history_table",
                        notes=f"rating in force at start of {start_year} (history table of rationale dated {ddate})"))
    for d, rt in events.items():
        sym, ol = split_outlook(rt)
        out.append(dict(issuer=issuer, agency="CRISIL", action_date=d.isoformat(), instrument=instrument,
                        rating_from="", rating_to=sym, outlook=ol, url=url,
                        source_type="agency_rationale" if d == ddate else "rating_history_table",
                        notes=f"from rationale dated {ddate}"))
    return out


def finalise(df):
    """De-duplicate across documents and fill rating_from from the previous action of the same issuer."""
    df = df.sort_values(["issuer", "action_date", "source_type"]).drop_duplicates(["issuer", "agency", "action_date"], keep="first")
    df["rating_from"] = df.groupby(["issuer", "agency"])["rating_to"].shift(1).fillna("")
    return df


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--issuer")
    ap.add_argument("--list")
    ap.add_argument("urls", nargs="*")
    a = ap.parse_args(argv)
    jobs = [(a.issuer, u) for u in a.urls]
    if a.list:
        jobs += [(r.issuer, r.url) for r in pd.read_csv(a.list).itertuples()]
    s = requests.Session()
    s.headers["User-Agent"] = UA
    new = []
    for issuer, url in jobs:
        try:
            got = rows_for(issuer, url, s)
            print(f"  {issuer}: {len(got)} rows from {url[-60:]}")
            new += got
        except Exception as e:
            print(f"  FAILED {issuer}: {e} ({url})")
    old = pd.read_csv(OUT, dtype=str).fillna("") if OUT.exists() else pd.DataFrame()
    df = finalise(pd.concat([old, pd.DataFrame(new)], ignore_index=True).astype(str))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"{len(df)} rows -> {OUT}")


if __name__ == "__main__":
    main()
