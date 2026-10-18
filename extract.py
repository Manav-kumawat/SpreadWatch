"""
L2 Extract - pull rating actions out of rating rationale documents.

Input : rating rationale press releases saved from the agency websites
        (CRISIL, ICRA, CARE, India Ratings ...) as .pdf / .txt / .html in
        data/raw/rationales/
Output: data/raw/ratings/extracted_rating_actions.csv - one row per rating change
        found, with the sentence it came from ("evidence") and verified = N.

The extractor is RULE-BASED (regular expressions), not a language model, so
every extracted row can be traced to an exact sentence and the rules can be
explained. You must review each row, correct it if needed and set verified = Y
before using the file - and register the file in sources.csv.

How the rules work
------------------
1. Read the text; find the press-release date (first date in the document)
   and the issuer (first line ending in Limited / Ltd that is not the agency).
2. Split into sentences; keep sentences that contain an action word
   (downgraded, upgraded, revised, reaffirmed, withdrawn, assigned).
3. Find every long-term rating symbol in the sentence that carries an agency
   prefix ("CRISIL AA+", "[ICRA]A-", "IND BBB", "CARE D" ...). Short-term
   symbols such as A1+ are ignored.
4. If the sentence contains "from": the first symbol after "from" is the OLD
   rating; the NEW rating is the symbol after "to", or else the last symbol
   before "from".  Otherwise a single symbol is the current rating.
5. The action type is decided by comparing notches (engine.ratings), not by
   the words used.

Usage:
    python extract.py                     # data/raw/rationales -> extracted_rating_actions.csv
    python extract.py --src data/sample/rationales --out outputs/sample/extracted_rating_actions.csv
    python extract.py --evaluate labels.csv   # accuracy against your hand-labelled file
"""
import argparse
import re
from pathlib import Path
from typing import List, Optional

import pandas as pd

from config import OUTPUT_DIR, RAW_DIR
from engine.ratings import classify_action, normalise

AGENCY_PREFIX = r"(?:\[\s*(?:ICRA|SYN)\s*\]\s*|CRISIL\s+|ICRA\s+|IND\s+|CARE\s+|ACUITE\s+|Acuite\s+|BWR\s+|SYN\s+)"
RATING_TOKEN = re.compile(AGENCY_PREFIX + r"(?:AAA|AA|A|BBB|BB|B|C|D)\s?[+-]?(?![0-9A-Za-z+])(?:\s*\((?:CE|SO)\))?")
OUTLOOK = re.compile(r"(Watch\s+(?:with\s+)?Negative|Watch\s+Developing|Negative|Stable|Positive|Developing)", re.I)
ACTION_WORDS = re.compile(r"downgrad|upgrad|revis|reaffirm|withdraw|assign|migrat|placed", re.I)
MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December|" \
         "Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
DATE_PATTERNS = [
    re.compile(rf"\b({MONTHS})\.?\s+(\d{{1,2}}),?\s+(\d{{4}})\b"),        # December 03, 2018
    re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({MONTHS})\.?,?\s+(\d{{4}})\b"),  # 03 December 2018
    re.compile(r"\b(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})\b"),              # 03-12-2018 (day first)
]
AGENCY_NAMES = {"CRISIL": "CRISIL", "ICRA": "ICRA", "IND": "India Ratings", "CARE": "CARE",
                "ACUITE": "Acuite", "BWR": "Brickwork", "SYN": "SYN-AGENCY"}
AGENCY_SELF = re.compile(r"(CRISIL|ICRA|CARE|India Ratings|Acuite|Brickwork)\s+(Ratings\s+)?(Limited|Ltd)", re.I)


def read_document(path: Path) -> str:
    suf = path.suffix.lower()
    if suf == ".pdf":
        from pypdf import PdfReader
        return "\n".join((pg.extract_text() or "") for pg in PdfReader(str(path)).pages)
    text = path.read_text(encoding="utf-8", errors="ignore")
    if suf in (".html", ".htm"):
        text = re.sub(r"<(script|style).*?</\1>", " ", text, flags=re.S | re.I)
        text = re.sub(r"<br\s*/?>|</p>|</tr>|</div>", "\n", text, flags=re.I)
        text = re.sub(r"<[^>]+>", " ", text)
        import html as _html
        text = _html.unescape(text)
    return text


def find_date(text: str) -> Optional[str]:
    head = text[:3000]
    best = None
    for pat in DATE_PATTERNS:
        m = pat.search(head)
        if m and (best is None or m.start() < best[0]):
            best = (m.start(), m.group(0))
    if not best:
        return None
    d = pd.to_datetime(re.sub(r"\bSept\b", "Sep", best[1]), dayfirst=True, errors="coerce")
    return None if pd.isna(d) else d.date().isoformat()


def find_issuer(text: str) -> Optional[str]:
    for line in [l.strip() for l in text.splitlines() if l.strip()][:20]:
        if re.search(r"\b(Limited|Ltd\.?|LLP)\s*$", line) and not AGENCY_SELF.search(line) and len(line) < 120:
            return line
    return None


def agency_of(token: str) -> str:
    t = token.strip().lstrip("[").upper()
    for k, v in AGENCY_NAMES.items():
        if t.startswith(k):
            return v
    return ""


def sentences(text: str) -> List[str]:
    text = re.sub(r"[ \t]+", " ", text)
    parts = re.split(r"(?<=[.;])\s+(?=[A-Z(\[])|\n{1,}", text)
    return [p.strip() for p in parts if p.strip()]


def extract_actions_from_sentence(s: str) -> List[dict]:
    toks = [(m.start(), m.end(), m.group(0)) for m in RATING_TOKEN.finditer(s)]
    if not toks or not ACTION_WORDS.search(s):
        return []
    out = []
    m_from = re.search(r"\bfrom\b", s, re.I)
    if m_from and len(toks) >= 2:
        after = [t for t in toks if t[0] > m_from.end()]
        before = [t for t in toks if t[1] <= m_from.start()]
        if not after:
            return []
        old = after[0]
        m_to = re.search(r"\bto\b", s[old[1]:], re.I)
        new = None
        if m_to:
            nxt = [t for t in after[1:] if t[0] > old[1] + m_to.start()]
            new = nxt[0] if nxt else None
        if new is None and before:
            new = before[-1]
        if new is None:
            return []
        tail = s[new[1]:new[1] + 40]
        out.append({"rating_from": old[2], "rating_to": new[2], "agency": agency_of(new[2]),
                    "outlook": (OUTLOOK.search(tail).group(1) if OUTLOOK.search(tail) else "")})
    elif re.search(r"withdraw", s, re.I):
        out.append({"rating_from": toks[0][2], "rating_to": "Withdrawn", "agency": agency_of(toks[0][2]),
                    "outlook": ""})
    elif len(toks) == 1:
        tail = s[toks[0][1]:toks[0][1] + 40]
        out.append({"rating_from": "", "rating_to": toks[0][2], "agency": agency_of(toks[0][2]),
                    "outlook": (OUTLOOK.search(tail).group(1) if OUTLOOK.search(tail) else "")})
    return out


def extract_document(path: Path) -> List[dict]:
    text = read_document(path)
    date, issuer = find_date(text), find_issuer(text)
    rows, seen = [], set()
    for s in sentences(text):
        for a in extract_actions_from_sentence(s):
            key = (normalise(a["rating_from"]), normalise(a["rating_to"]))
            if key in seen or key[1] is None:
                continue
            seen.add(key)
            action = classify_action(a["rating_from"], a["rating_to"])
            rows.append({
                "issuer": issuer, "agency": a["agency"], "action_date": date, "instrument": "",
                "rating_from": a["rating_from"], "rating_to": a["rating_to"], "outlook": a["outlook"],
                "action": action, "url": path.name, "evidence": s[:300],
                "extraction_method": "rules", "verified": "N",
            })
    # A rationale usually reports one change for many instruments; keep the
    # changes first and drop plain reaffirmations if a change was found.
    if any(r["action"] in ("downgrade", "upgrade", "default", "withdrawn") for r in rows):
        rows = [r for r in rows if r["action"] not in ("reaffirm", "assigned")]
    if not rows:
        rows.append({"issuer": issuer, "agency": "", "action_date": date, "instrument": "", "rating_from": "",
                     "rating_to": "", "outlook": "", "action": "NOT_FOUND", "url": path.name,
                     "evidence": "", "extraction_method": "rules", "verified": "N"})
    return rows


def extract_folder(src: Path) -> pd.DataFrame:
    files = sorted(p for p in src.glob("*") if p.suffix.lower() in (".pdf", ".txt", ".html", ".htm"))
    rows = [r for p in files for r in extract_document(p)]
    return pd.DataFrame(rows)


def evaluate(extracted: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Field-level accuracy against a hand-labelled file with columns url, issuer, action_date, rating_from, rating_to."""
    ex = extracted.copy()
    lab = labels.copy()
    for df in (ex, lab):
        df["from_n"] = df["rating_from"].map(normalise)
        df["to_n"] = df["rating_to"].map(normalise)
    m = lab.merge(ex, on="url", how="left", suffixes=("_true", "_pred"))
    res = {
        "documents": lab["url"].nunique(),
        "date_correct": (m["action_date_true"].astype(str) == m["action_date_pred"].astype(str)).mean(),
        "issuer_correct": (m["issuer_true"].str.lower().str.strip() == m["issuer_pred"].str.lower().str.strip()).mean(),
        "rating_from_correct": (m["from_n_true"].fillna("") == m["from_n_pred"].fillna("")).mean(),
        "rating_to_correct": (m["to_n_true"] == m["to_n_pred"]).mean(),
    }
    return pd.DataFrame([res])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=str(RAW_DIR / "rationales"))
    ap.add_argument("--out", default=str(RAW_DIR / "ratings" / "extracted_rating_actions.csv"))
    ap.add_argument("--evaluate", help="hand-labelled CSV to score the extraction against")
    a = ap.parse_args(argv)
    df = extract_folder(Path(a.src))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"{len(df)} rows from {df['url'].nunique() if len(df) else 0} documents -> {a.out}")
    if a.evaluate:
        acc = evaluate(df, pd.read_csv(a.evaluate))
        acc.to_csv(OUTPUT_DIR / "extraction_accuracy.csv", index=False)
        print(acc.to_string(index=False))


if __name__ == "__main__":
    main()
