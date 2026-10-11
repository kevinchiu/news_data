"""Score a tracer config on a split of the Quotebank benchmark.

Usage: python -I scripts/evaluate.py <config.json> <train|val|test> [--mode paraphrase|verbatim]
           [--limit N] [--retrieval-only] [--nonce STR] [--workers 4]

Metrics (all fractions of items unless noted):
  recall_at_k       ground-truth quote (or a direct variant) among the retrieved candidates  [no LLM]
  verifier_recall   ground-truth quote matched, given it was retrieved
  fp_earlier        a wrongly matched quote pulled the first appearance earlier than the truth
  first_any         first appearance correct: same URL as ground truth, or a different article with
                    the identical timestamp (syndicated copies often publish the same second; both
                    sides rank only mentions with a real time of day, so date-only ties can't count)
  first_any_tie     the subset of first_any that matched on timestamp, not URL
  first_original    first original-outlet appearance correct (same URL)
  calls_per_claim   Haiku calls per item (cached calls included)
"""
import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qb_config  # noqa: E402
from trace_quote import candidates, trace  # noqa: E402
from verify import LLMCache  # noqa: E402

BENCH = "bench/qb_bench.jsonl"


def load_items(split, limit=None):
    items = [json.loads(l) for l in open(BENCH)]
    items = [it for it in items if it["split"] == split]
    return items[:limit] if limit else items


def evaluate(cfg, items, mode="paraphrase", retrieval_only=False, nonce="", workers=4, con=None):
    con = con or duckdb.connect("data/news.duckdb", read_only=True)
    con.execute("LOAD fts")
    cache, local = LLMCache(), threading.local()

    def cursor():
        if not hasattr(local, "con"):
            local.con = con.cursor()
            local.con.execute("LOAD fts")
        return local.con

    def one(it):
        ids, gt = set(it["variant_ids"]), it["gt_first_any"]
        if retrieval_only:
            cands = candidates(cursor(), it[mode], it["as_of"], cfg)
            return {"quote_id": it["quote_id"], "retrieved": any(c["quote_id"] in ids for c in cands)}
        res = trace(cursor(), cache, it[mode], it["as_of"], cfg, nonce=nonce)
        pred, matched = res["first_any"], [m["quote_id"] for m in res["matched_quotes"]]
        orig, gto = res["first_original"], it["gt_first_original"]
        return {
            "quote_id": it["quote_id"],
            "retrieved": any(c["quote_id"] in ids for c in res["candidates"]),
            "gt_matched": any(q in ids for q in matched),
            "fp_earlier": bool(pred and pred["date"][:19] < gt["date"][:19]),
            "first_any": bool(pred and (pred["url"] == gt["url"] or pred["date"][:19] == gt["date"][:19])),
            "first_any_tie": bool(pred and pred["url"] != gt["url"] and pred["date"][:19] == gt["date"][:19]),
            "first_original": bool(orig and gto and orig["url"] == gto["url"]),
            "calls": res["llm_calls"]["made"] + res["llm_calls"]["cached"],
            "made": res["llm_calls"]["made"],
            "matched": matched,
            "pred_first_any": pred and {k: pred[k] for k in ("date", "domain", "url")},
        }

    with ThreadPoolExecutor(workers) as ex:
        rows = list(ex.map(one, items))
    n = len(rows)
    m = {"n": n, "recall_at_k": sum(r["retrieved"] for r in rows) / n}
    if not retrieval_only:
        retrieved = [r for r in rows if r["retrieved"]]
        m |= {
            "verifier_recall": sum(r["gt_matched"] for r in retrieved) / max(1, len(retrieved)),
            "fp_earlier": sum(r["fp_earlier"] for r in rows) / n,
            "first_any": sum(r["first_any"] for r in rows) / n,
            "first_any_tie": sum(r["first_any_tie"] for r in rows) / n,
            "first_original": sum(r["first_original"] for r in rows) / n,
            "calls_per_claim": sum(r["calls"] for r in rows) / n,
            "new_calls": sum(r["made"] for r in rows),
        }
    return m, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("split")
    ap.add_argument("--mode", default="paraphrase")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--retrieval-only", action="store_true")
    ap.add_argument("--nonce", default="")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", help="write per-item rows (JSONL)")
    a = ap.parse_args()
    m, rows = evaluate(qb_config.load(a.config), load_items(a.split, a.limit), a.mode, a.retrieval_only,
                       a.nonce, a.workers)
    print(json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in m.items()}))
    if a.out:
        Path(a.out).write_text("".join(json.dumps(r) + "\n" for r in rows))


if __name__ == "__main__":
    main()
