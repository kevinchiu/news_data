"""Trace a factual statement back to its earliest report in the corpus.

  python -I scripts/trace_claim.py --sid 12043:2                  # a sentence from the corpus
  python -I scripts/trace_claim.py "Statement text" [--headline "..."] [--as-of 1987-04-01]

The statement is first split into atomic, self-contained facts (Haiku 5.5, effort low); each
"new" fact is traced separately (--all-facts also traces background facts, --no-split traces
the sentence whole). Retrieval: BM25 over sentences from strictly earlier stories. Verification: Claude Haiku 5.5
(effort low, with headlines) decides which candidates report the same fact. Each hop re-searches
before the earliest match found so far, using both the claim's and that match's wording, until
no earlier match turns up.
"""
import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from retrieval import SentenceIndex  # noqa: E402
from split_claims import split  # noqa: E402
from verify import LLMCache  # noqa: E402

MODEL, EFFORT, BATCH = "claude-haiku-5-5", "low", 15  # batch size matches the bench

ATTRIBUTION = [
    re.compile(r"according to ([^,.;\"]{3,60})"),
    re.compile(r"(?:,|\")\s*((?:the |a |an )?[A-Za-z][\w&.'/ -]{2,60}?)\s+(?:said|told \w+|reported|announced|estimated)\b"),
    re.compile(r"^((?:the |a |an )?[A-Z][\w&.'/ -]{2,60}?)\s+(?:said|told \w+|reported|announced|estimated)\b"),
    re.compile(r"\b(told Reuters|told reporters)\b"),
]


def attribution(text):
    for rx in ATTRIBUTION:
        m = rx.search(text)
        if m:
            return re.sub(r"^(?:but|and|while|as)\s+", "", m.group(1).strip())
    return None


def verify(cache, claim, headline, date, cands):
    """Verify candidates in bench-sized batches; identical texts are checked once."""
    groups = {}
    for c in cands:
        groups.setdefault(" ".join(c["text"].lower().split()), []).append(c)
    reps = [g[0] for g in groups.values()]
    batches = [reps[i:i + BATCH] for i in range(0, len(reps), BATCH)]
    with ThreadPoolExecutor(4) as ex:
        results = list(ex.map(lambda b: cache.verify(claim, b, headline, MODEL, EFFORT, date), batches))
    matched, calls = [], {"made": 0, "cached": 0}
    for b, r in zip(batches, results):
        calls["cached" if r["cached"] else "made"] += 1
        for i in r["matches"]:
            matched.extend(groups[" ".join(b[i]["text"].lower().split())])
    return matched, calls


def trace(idx, cache, claim, headline=None, as_of="9999", exclude_docs=(), k=30, max_hops=4):
    found, checked = {}, set()
    calls = {"made": 0, "cached": 0}
    queries, cutoff = [claim], as_of
    for hop in range(max_hops):
        cands = {}
        for q in queries:
            for c in idx.search(q, before=cutoff, k=k, exclude_docs=exclude_docs):
                if c["sid"] not in checked:
                    cands.setdefault(c["sid"], c)
        checked |= set(cands)
        if not cands:
            break
        matched, n = verify(cache, claim, headline, None if as_of == "9999" else as_of, list(cands.values()))
        calls = {key: calls[key] + n[key] for key in calls}
        for m in matched:
            found[m["sid"]] = dict(m, hop=hop)
        if not matched:
            break
        earliest = min(found.values(), key=lambda r: (r["date"], r["sid"]))
        if earliest["date"] >= cutoff:
            break
        cutoff, queries = earliest["date"], [claim, earliest["text"]]
    timeline = sorted(found.values(), key=lambda r: (r["date"], r["sid"]))
    for r in timeline:
        r["attribution"] = attribution(r["text"])
    return {"claim": claim, "headline": headline, "as_of": as_of, "origin": timeline[0] if timeline else None,
            "timeline": timeline, "llm_calls": calls}


def show(res, indent=""):
    p = lambda *a: print(indent + " ".join(str(x) for x in a))  # noqa: E731
    if not res["timeline"]:
        p("No earlier report found in the corpus.")
        return
    o = res["origin"]
    p(f"EARLIEST REPORT: {o['date'][:16]}  doc {o['doc_id']}  {o['headline']}")
    p(f"  {o['text']}")
    if o["attribution"]:
        p(f"  attributed to: {o['attribution']}")
    p(f"TIMELINE ({len(res['timeline'])} matching sentences):")
    for r in res["timeline"]:
        src = f"  [{r['attribution']}]" if r["attribution"] else ""
        p(f"  {r['date'][:16]}  doc {r['doc_id']:<6} {r['headline'][:60]}{src}")
        p(f"      {r['text'][:160]}")


def show_all(out):
    print(f"CLAIM: {out['claim']}")
    if out["headline"]:
        print(f"  from: {out['headline']}  ({out['as_of'][:16]})")
    for i, f in enumerate(out["facts"], 1):
        print(f"\nFACT {i} ({f['kind']}): {f['text']}")
        if f.get("trace"):
            show(f["trace"], "   ")
        else:
            print("   (background, not traced; use --all-facts)")
    c = out["llm_calls"]
    print(f"\n({c['made']} Haiku calls, {c['cached']} from cache)")


def trace_split(idx, cache, claim, headline, as_of, exclude_docs, k, all_facts=False, do_split=True):
    calls = {"made": 0, "cached": 0}
    if do_split:
        sp = split(cache, claim, headline, None if as_of == "9999" else as_of)
        calls["cached" if sp["cached"] else "made"] += 1
        facts = [dict(f) for f in sp["facts"]] or [{"text": claim, "kind": "new"}]
    else:
        facts = [{"text": claim, "kind": "new"}]
    for f in facts:
        if f["kind"] == "new" or all_facts:
            f["trace"] = trace(idx, cache, f["text"], headline, as_of, exclude_docs, k)
            calls = {key: calls[key] + f["trace"]["llm_calls"][key] for key in calls}
    return {"claim": claim, "headline": headline, "as_of": as_of, "facts": facts, "llm_calls": calls}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("statement", nargs="?")
    ap.add_argument("--sid", help="trace a corpus sentence by id (doc_id:index)")
    ap.add_argument("--headline")
    ap.add_argument("--as-of", default="9999", help="only consider reports before this ISO date")
    ap.add_argument("--k", type=int, default=30, help="BM25 candidates per query per hop")
    ap.add_argument("--all-facts", action="store_true", help="also trace background facts")
    ap.add_argument("--no-split", action="store_true", help="trace the statement whole")
    ap.add_argument("--json", help="also write the full result to this path")
    a = ap.parse_args()
    if not (a.statement or a.sid):
        ap.error("give a statement or --sid")

    idx = SentenceIndex("data/processed/sentences.jsonl", "data/processed/reuters21578.jsonl")
    cache = LLMCache()
    if a.sid:
        row = idx.rows[idx.by_sid[a.sid]]
        args = (row["text"], idx.headline(row["doc_id"]), row["date"], (row["doc_id"],))
    else:
        args = (a.statement, a.headline, a.as_of, ())
    res = trace_split(idx, cache, *args, a.k, a.all_facts, not a.no_split)
    show_all(res)
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
