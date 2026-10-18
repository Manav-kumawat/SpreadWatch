"""The L4 verifier must block uncited sentences, unknown citations and invented numbers."""
from commentary import verify

FACTS = {"es.mean_pre": "the mean abnormal widening was 121.5 bps",
         "es.lead": "the median lead time was 28 days"}
SRC = {"S01"}
HITS = [{"chunk_id": "doc.pdf#p2", "text": "Liquidity weakened as repayments of Rs 500 crore fell due."}]


def run(text):
    kept, log = verify(text, FACTS, SRC, HITS)
    return kept, log


def test_cited_sentence_with_matching_number_is_kept():
    kept, log = run("Spreads widened by 121.5 bps on average [E:es.mean_pre].")
    assert log["kept"].all()


def test_invented_number_is_dropped():
    kept, log = run("Spreads widened by 150 bps on average [E:es.mean_pre].")
    assert not log["kept"].any()
    assert "150" in log["reason"].iloc[0]


def test_number_from_another_fact_is_not_enough():
    # 28 exists in es.lead, but this sentence only cites es.mean_pre
    kept, log = run("The rule fired 28 days early [E:es.mean_pre].")
    assert not log["kept"].any()


def test_uncited_and_unknown_citations_are_dropped():
    kept, log = run("Spreads always lead ratings. Ratings lag [E:does.not.exist]. Source checked [S:S99].")
    assert not log["kept"].any()
    assert list(log["reason"].str[:7]) == ["no cita", "unknown", "unknown"]


def test_rationale_citation_allows_its_numbers():
    kept, log = run("Repayments of Rs 500 crore fell due [R:doc.pdf#p2].")
    assert log["kept"].all()


def test_mixed_paragraph_keeps_only_good_sentences():
    kept, log = run("Widening was 121.5 bps [E:es.mean_pre]. It was 99 bps for banks [E:es.mean_pre].")
    assert kept == "Widening was 121.5 bps [E:es.mean_pre]."
