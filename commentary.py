"""
L4 Commentary - writes over the engine's output only, and cites every claim.

Citation tags
  [E:<fact_id>]   a number computed by the L3 engine (outputs/<mode>/...)
  [S:<source_id>] a row of sources.csv (L1 provenance)
  [R:<chunk_id>]  a paragraph of a rating rationale retrieved by rag/ (L2)

Two writers:
  * template (default) - deterministic sentences filled from the engine facts.
    Works offline, gives the same text every run.
  * llm (optional)     - Claude rewrites the commentary from the same facts.
    It is told to use only the supplied facts and to tag every sentence.

Both go through the same VERIFIER before anything is shown:
  1. every sentence must carry at least one tag, and every tag must exist;
  2. every number in a sentence must appear in the cited facts.
Sentences that fail are dropped and logged - the language model can never
introduce a number of its own. See governance.md.
"""
import json
import os
import re
from typing import Dict, List, Tuple

import pandas as pd

from config import BASELINE_WINDOW, LGD, OUTPUT_DIR, PRE_WINDOW, SAMPLE_DIR, SOURCES_CSV, Z_THRESHOLD, RAW_DIR
from rag import RationaleIndex

DISCLAIMER = ("Educational use only. SpreadWatch is an academic project; nothing here is investment advice "
              "or a recommendation to buy, sell or hold any security (SEBI (Research Analysts) "
              "Regulations, 2014).")
TAG = re.compile(r"\[(E|S|R):([^\]]+)\]")
NUM = re.compile(r"(?<![\w.])-?\d+(?:,\d{3})*(?:\.\d+)?")


def _fmt(x, nd=1):
    return f"{x:,.{nd}f}"


def _pct(x, nd=1):
    return f"{100 * x:.{nd}f}%"


# ----------------------------------------------------------------------------
# Facts
# ----------------------------------------------------------------------------
def build_facts(mode: str) -> Tuple[Dict[str, str], dict]:
    """fact_id -> human-readable statement containing the number, as the engine produced it."""
    od = OUTPUT_DIR / mode
    s = json.load(open(od / "summary.json"))
    v, t = s["validation"], s["event_study"]
    f: Dict[str, str] = {}

    f["data.period"] = f"trade data run from {s['period'][0]} to {s['period'][1]}"
    f["data.trades"] = f"{s['n_trades_used']:,} trades for {s['n_issuers']} issuers were priced"
    f["data.rejected"] = f"{s['n_trades_rejected']:,} trade rows were rejected by validation"
    if v.get("n_checked"):
        f["val.pass_rate"] = (f"{_pct(float(v['pass_rate']))} of {int(v['n_checked']):,} recomputed yields were within "
                              f"{_fmt(float(v['tolerance_bps']), 0)} bps of the exchange-printed yield")
        f["val.median_diff"] = f"the median absolute yield difference was {_fmt(float(v['median_abs_diff_bps']), 2)} bps"
        f["val.p95_diff"] = f"the 95th percentile absolute difference was {_fmt(float(v['p95_abs_diff_bps']), 2)} bps"

    f["es.windows"] = (f"the baseline window is days {BASELINE_WINDOW[0]} to {BASELINE_WINDOW[1]} and the pre-event "
                       f"window is days {PRE_WINDOW[0]} to {PRE_WINDOW[1]}, with a breach at {Z_THRESHOLD:.0f} standard deviations")
    f["es.verdict"] = f"verdict: {t.get('verdict')} - {t.get('verdict_reason')}"
    if "mean_dS_pre_bps" in t:
        f["es.events"] = f"{t['n_events_usable']} of {t['n_events_total']} downgrade episodes had enough trades to measure"
        f["es.mean_pre"] = f"the mean abnormal widening in the pre-event window was {_fmt(float(t['mean_dS_pre_bps']))} bps"
        f["es.median_pre"] = f"the median abnormal widening was {_fmt(float(t['median_dS_pre_bps']))} bps"
        f["es.share_widened"] = f"{_pct(float(t['share_events_widened']))} of episodes widened before the action"
        f["es.ttest"] = f"one-sided t-test t = {float(t['t_stat']):.2f}, p = {float(t['t_pvalue']):.4f}"
        f["es.hit_rate"] = f"the early-warning rule fired before {_pct(float(t['hit_rate']))} of downgrades"
        if t.get("median_lead_days") not in (None, "None"):
            f["es.lead"] = f"the median lead time when it fired was {_fmt(float(t['median_lead_days']), 0)} days"
        if t.get("median_pre_share") not in (None, "None"):
            f["es.pre_share"] = (f"the median share of total widening that happened before the action was "
                                 f"{_pct(float(t['median_pre_share']), 0)}")
        if "false_alarm_rate" in t:
            f["es.false_alarm"] = (f"the same rule fired on {_pct(float(t['false_alarm_rate']))} of "
                                   f"{t['n_placebo_usable']} random placebo dates")
            f["es.fisher"] = f"Fisher exact test hit rate vs false-alarm rate p = {float(t['fisher_pvalue']):.4f}"
            f["es.placebo_mean"] = f"placebo mean change was {_fmt(float(t['placebo_mean_dS_pre_bps']))} bps"

    probs = pd.read_csv(od / "transition_probs.csv", index_col=0)
    pdt = pd.read_csv(od / "pd_by_grade.csv", index_col=0)
    for g in ["AAA", "AA", "A", "BBB"]:
        if g in probs.index and probs.loc[g].notna().any():
            f[f"tm.stay.{g}"] = f"{_pct(float(probs.loc[g, g]))} of {g}-rated units kept their category over one year"
            down = probs.loc[g, probs.columns[list(probs.columns).index(g) + 1:]].sum()
            f[f"tm.down.{g}"] = f"{_pct(float(down))} of {g}-rated units were downgraded within a year"
        if g in pdt.index and pd.notna(pdt.loc[g, "PD_1y"]):
            f[f"pd.{g}"] = (f"historical one-year PD for {g} is {_pct(float(pdt.loc[g, 'PD_1y']), 2)} and "
                            f"five-year PD is {_pct(float(pdt.loc[g, 'PD_5y']), 2)}")
    imp = pd.read_csv(od / "implied_vs_historical_pd.csv", index_col=0)
    for g in ["AA", "A", "BBB"]:
        if g in imp.index:
            f[f"imp.{g}"] = (f"the median {g} spread of {_fmt(float(imp.loc[g, 'median']), 0)} bps implies a one-year PD of "
                             f"{_pct(float(imp.loc[g, 'implied_pd_1y']), 2)} at {_pct(LGD, 0)} LGD")

    # The case study: the flagged event with the largest z-score
    case = {}
    ev = pd.read_csv(od / "events.csv")
    flagged = ev[(ev["usable"] == True) & (ev["flagged"] == True)]  # noqa: E712
    if not flagged.empty:
        c = flagged.sort_values("z_pre", ascending=False).iloc[0]
        case = c.to_dict()
        f["case.action"] = (f"{c['issuer']} was downgraded from {c['rating_from']} to {c['rating_to']} "
                            f"on {c['action_date']}")
        f["case.widening"] = (f"its abnormal spread was {_fmt(c['dS_pre_bps'])} bps above its baseline in the 30 days "
                              f"before the action (z = {c['z_pre']:.1f})")
        if pd.notna(c["lead_days"]):
            f["case.lead"] = f"the early-warning rule first fired {int(c['lead_days'])} days before the action"
    return f, case


def sources_table() -> pd.DataFrame:
    return pd.read_csv(SOURCES_CSV, dtype=str).fillna("") if SOURCES_CSV.exists() else pd.DataFrame()


def retrieve_case_context(mode: str, case: dict, k: int = 2) -> List[dict]:
    folder = SAMPLE_DIR / "rationales" if mode == "sample" else RAW_DIR / "rationales"
    idx = RationaleIndex(folder)
    if not case or not len(idx):
        return []
    q = ("rating action reflects weakening liquidity refinancing risk repayments leverage "
         "non-performing collections margins working capital")
    key = str(case["issuer"]).replace(" ", "_")
    hits = idx.search(q, k=k, file_contains=key) or idx.search(q, k=k)
    date = str(case.get("action_date", ""))
    hits.sort(key=lambda h: date not in h["file"])           # prefer the rationale of that action
    return hits[:k]


# ----------------------------------------------------------------------------
# Writers
# ----------------------------------------------------------------------------
def template_commentary(facts: Dict[str, str], src: pd.DataFrame, rag_hits: List[dict], mode: str) -> str:
    sid = src["source_id"].iloc[0] if not src.empty else "SYN01"
    if mode == "sample":
        sid = "SYN01"
    P = []

    def has(*keys):
        return all(k in facts for k in keys)

    p1 = []
    if has("es.verdict"):
        p1.append(f"On the research question - whether credit spreads on corporate bonds move before the rating "
                  f"agencies act - the engine's answer on this dataset is "
                  f"{facts['es.verdict'].split(':')[1].split(' - ')[0].strip()} [E:es.verdict].")
    if has("es.events", "es.mean_pre"):
        p1.append(f"Across the sample, {facts['es.events']} [E:es.events], and {facts['es.mean_pre']} [E:es.mean_pre].")
    if has("es.ttest"):
        p1.append(f"The widening is tested against zero with a {facts['es.ttest']} [E:es.ttest].")
    if has("es.hit_rate", "es.false_alarm"):
        p1.append(f"As a signal, {facts['es.hit_rate']}, while {facts['es.false_alarm']} "
                  f"[E:es.hit_rate][E:es.false_alarm].")
    if has("es.lead"):
        p1.append(f"In those episodes {facts['es.lead']} [E:es.lead].")
    if has("es.pre_share"):
        p1.append(f"Measured against the full move, {facts['es.pre_share']} [E:es.pre_share].")
    P.append(" ".join(p1))

    p2 = []
    if has("case.action", "case.widening"):
        p2.append(f"Case study: {facts['case.action']} [E:case.action][S:{sid}].")
        p2.append(f"Before that, {facts['case.widening']} [E:case.widening].")
        if has("case.lead"):
            p2.append(f"Put differently, {facts['case.lead']} [E:case.lead].")
        for h in rag_hits[:1]:
            excerpt = " ".join(h["text"].split()[:40])
            p2.append(f"The agency's rationale explains the action as follows: \"{excerpt}...\" [R:{h['chunk_id']}].")
    if p2:
        P.append(" ".join(p2))

    p3 = []
    downs = [g for g in ["AA", "A"] if has(f"tm.down.{g}")]
    if downs:
        p3.append("On the transition matrix, " + ", and ".join(facts[f"tm.down.{g}"] for g in downs) + " "
                  + "".join(f"[E:tm.down.{g}]" for g in downs) + ".")
    for g in ["A", "BBB"]:
        if has(f"pd.{g}"):
            p3.append(f"The {facts[f'pd.{g}']} [E:pd.{g}].")
        if has(f"imp.{g}"):
            p3.append(f"By contrast, {facts[f'imp.{g}']} [E:imp.{g}].")
    if p3:
        P.append(" ".join(p3))

    p4 = []
    if has("val.pass_rate"):
        p4.append(f"Validation: {facts['val.pass_rate']} [E:val.pass_rate], and {facts['val.median_diff']} "
                  f"[E:val.median_diff].")
    if mode == "sample":
        p4.append(f"All figures above come from the SYNTHETIC sample and demonstrate the method only [S:{sid}].")
    P.append(" ".join(p4))
    return "\n\n".join(p for p in P if p)


LLM_SYSTEM = """You write the commentary section of an academic credit-spread study.
Rules you must follow exactly:
- Use ONLY the facts and rationale excerpts provided. Do not add any number, date, name or claim that is not in them.
- Every sentence must end with one or more citation tags copied exactly from the provided ids, e.g. [E:es.mean_pre] or [R:file.txt#p2] or [S:SYN01].
- Copy numbers exactly as written in the facts. Do not round, convert or compute new numbers.
- Never recommend buying, selling or holding any security.
- Write 3 to 4 short paragraphs: the answer to the research question, one issuer case study, the transition matrix and PD comparison, and the validation result.
Return only the commentary text."""


def llm_commentary(facts: Dict[str, str], rag_hits: List[dict], sid: str) -> str:
    import anthropic
    client = anthropic.Anthropic()
    payload = {"facts": facts, "rationale_excerpts": {h["chunk_id"]: h["text"] for h in rag_hits},
               "source_ids": [sid]}
    resp = client.beta.messages.create(
        model=os.environ.get("SPREADWATCH_MODEL", "claude-opus-5-5"),
        max_tokens=4000,
        system=LLM_SYSTEM,
        output_config={"effort": "medium"},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        messages=[{"role": "user", "content": json.dumps(payload, indent=1)}],
    )
    if resp.stop_reason == "refusal":
        raise RuntimeError("model declined the request")
    return "".join(b.text for b in resp.content if b.type == "text").strip()


# ----------------------------------------------------------------------------
# Verifier
# ----------------------------------------------------------------------------
def _numbers(text: str) -> List[float]:
    return [float(n.replace(",", "")) for n in NUM.findall(text)]


def verify(text: str, facts: Dict[str, str], src_ids, rag_hits: List[dict]) -> Tuple[str, pd.DataFrame]:
    rag_ids = {h["chunk_id"]: h["text"] for h in rag_hits}
    kept_pars, log = [], []
    for par in text.split("\n\n"):
        kept = []
        # protect quoted excerpts so the splitter does not cut inside them
        quotes = re.findall(r"\"[^\"]*\"", par)
        masked = par.strip()
        for i, q in enumerate(quotes):
            masked = masked.replace(q, f"\x00{i}\x00", 1)
        for sent in re.split(r"(?<=[.!?\]])\s+(?=[A-Z\"(\x00])", masked):
            for i, q in enumerate(quotes):
                sent = sent.replace(f"\x00{i}\x00", q)
            if not sent.strip():
                continue
            tags = TAG.findall(sent)
            reason = ""
            if not tags:
                reason = "no citation"
            else:
                bad = [f"{k}:{v}" for k, v in tags if (k == "E" and v not in facts) or
                       (k == "S" and v not in src_ids) or (k == "R" and v not in rag_ids)]
                if bad:
                    reason = f"unknown citation {bad}"
            if not reason:
                body = TAG.sub("", sent)
                quoted = " ".join(re.findall(r"\"[^\"]*\"", body))
                body_nums = _numbers(re.sub(r"\"[^\"]*\"", "", body))
                cited_text = " ".join(facts.get(v, "") if k == "E" else rag_ids.get(v, "") if k == "R" else ""
                                      for k, v in tags)
                allowed = set(_numbers(cited_text)) | set(_numbers(quoted))
                stray = [n for n in body_nums if n not in allowed]
                if stray:
                    reason = f"number(s) not in cited facts: {stray}"
            log.append({"sentence": sent, "kept": not reason, "reason": reason})
            if not reason:
                kept.append(sent)
        if kept:
            kept_pars.append(" ".join(kept))
    return "\n\n".join(kept_pars), pd.DataFrame(log)


def generate(mode: str = "sample", use_llm: bool = False) -> dict:
    facts, case = build_facts(mode)
    src = sources_table()
    src_ids = set(src["source_id"]) if not src.empty else {"SYN01"}
    hits = retrieve_case_context(mode, case)
    sid = "SYN01" if mode == "sample" else (sorted(src_ids - {"SYN01"}) or ["SYN01"])[0]
    writer, note = "template", ""
    raw = template_commentary(facts, src, hits, mode)
    if use_llm:
        try:
            raw = llm_commentary(facts, hits, sid)
            writer = "llm"
        except Exception as e:  # no key, no network, refusal -> fall back, and say so
            note = f"LLM unavailable ({type(e).__name__}: {e}); template commentary used instead."
    text, log = verify(raw, facts, src_ids, hits)
    out = {"mode": mode, "writer": writer, "note": note, "text": text, "raw": raw,
           "facts": facts, "rag_hits": hits, "verification": log, "disclaimer": DISCLAIMER}
    od = OUTPUT_DIR / mode
    (od / "commentary.md").write_text(f"{text}\n\n_{DISCLAIMER}_\n")
    log.to_csv(od / "commentary_verification.csv", index=False)
    json.dump(facts, open(od / "facts.json", "w"), indent=1)
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sample", "real"], default="sample")
    ap.add_argument("--llm", action="store_true", help="let Claude write the text (needs ANTHROPIC_API_KEY)")
    a = ap.parse_args()
    r = generate(a.mode, a.llm)
    print(r["note"] or f"writer: {r['writer']}")
    print(r["text"])
    v = r["verification"]
    print(f"\nVerifier: kept {int(v['kept'].sum())} of {len(v)} sentences")
    for _, row in v[~v["kept"]].iterrows():
        print("  DROPPED:", row["reason"], "|", row["sentence"][:100])
