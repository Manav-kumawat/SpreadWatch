"""
L2 extraction rules tested on sentences written in the layouts the agencies use.
These are illustrative format examples written for the test - not quotations
from real rating rationales.
"""
from pathlib import Path

import pandas as pd

from engine.ratings import normalise
from extract import extract_actions_from_sentence, extract_document, find_date, find_issuer


def one(s):
    out = extract_actions_from_sentence(s)
    assert len(out) == 1, out
    return normalise(out[0]["rating_from"]), normalise(out[0]["rating_to"]), out[0]["agency"]


def test_crisil_layout_downgraded_to_from():
    assert one("Rating downgraded to 'CRISIL A+/Watch Negative' from 'CRISIL AA/Stable'") == ("AA", "A+", "CRISIL")


def test_icra_layout_new_first_then_from():
    assert one("NCD programme: [ICRA]BBB (Negative); downgraded from [ICRA]A (Stable)") == ("A", "BBB", "ICRA")


def test_revised_from_to():
    assert one("The long-term rating has been revised from CARE AA- to CARE BB+.") == ("AA-", "BB+", "CARE")


def test_short_term_symbols_ignored():
    assert extract_actions_from_sentence("Commercial paper reaffirmed at CRISIL A1+") == []


def test_withdrawal():
    out = extract_actions_from_sentence("The rating of IND AA- has been withdrawn at the request of the company.")
    assert normalise(out[0]["rating_to"]) == "WR"


def test_sentence_without_action_word_is_skipped():
    assert extract_actions_from_sentence("CRISIL AA bonds are widely held.") == []


def test_date_formats():
    assert find_date("Rating Rationale\nDecember 03, 2018 | Mumbai") == "2018-12-03"
    assert find_date("XYZ Limited\n05 June 2019\n") == "2019-06-05"
    assert find_date("Date: 14-08-2020") == "2020-08-14"


def test_issuer_skips_agency_name():
    text = "CRISIL Ratings Limited\nRating Rationale\nAbc Housing Finance Limited\n"
    assert find_issuer(text) == "Abc Housing Finance Limited"


def test_whole_document(tmp_path: Path):
    doc = tmp_path / "abc.txt"
    doc.write_text("Rating Rationale\nMarch 15, 2019 | Mumbai\nAbc Infra Finance Limited\n\n"
                   "Rating downgraded to 'CRISIL BBB/Negative' from 'CRISIL A/Stable'; CP reaffirmed at CRISIL A1.\n")
    rows = extract_document(doc)
    assert len(rows) == 1
    r = rows[0]
    assert (r["issuer"], r["action_date"], r["action"]) == ("Abc Infra Finance Limited", "2019-03-15", "downgrade")
    assert r["outlook"] == "Negative" and r["verified"] == "N"


def test_synthetic_rationales_extract_correctly():
    """Accuracy on the synthetic sample (both layouts) must be 100% - it is a regression check."""
    from config import SAMPLE_DIR
    from extract import extract_folder
    src = SAMPLE_DIR / "rationales"
    if not src.exists():
        return
    ex = extract_folder(src)
    truth = pd.read_csv(SAMPLE_DIR / "rating_actions.csv").dropna(subset=["rating_from"])
    truth["key"] = truth["issuer"] + "|" + truth["action_date"]
    ex["key"] = ex["issuer"] + "|" + ex["action_date"]
    m = truth.merge(ex, on="key", suffixes=("_t", "_e"))
    assert len(m) == len(truth)
    assert (m["rating_to_t"].map(normalise) == m["rating_to_e"].map(normalise)).all()
    assert (m["rating_from_t"].map(normalise) == m["rating_from_e"].map(normalise)).all()
