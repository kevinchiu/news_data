"""Score verifier runs against a gold run.

Usage: python -I scripts/score_bench.py bench/queries.jsonl <gold_label> <label> [<label> ...]
"""
import json
import sys
from statistics import mean, median


def load(label):
    return {r["qid"]: r for r in map(json.loads, open(f"bench/runs/{label}.jsonl"))}


def earliest(q, matches):
    if not matches:
        return None
    return min(matches, key=lambda i: (q["candidates"][i]["date"], q["candidates"][i]["sid"]))


def main(qpath, gold_label, *labels):
    qs = {q["qid"]: q for q in map(json.loads, open(qpath))}
    gold = load(gold_label)
    n_pos = sum(len(g["matches"]) for g in gold.values())
    n_q_pos = sum(bool(g["matches"]) for g in gold.values())
    print(f"gold={gold_label}: {len(gold)} queries, {n_pos} matching pairs, {n_q_pos} queries with >=1 match\n")
    hdr = f"{'run':<22}{'n':>4}{'prec':>7}{'rec':>7}{'F1':>7}{'origin':>8}{'any':>7}{'think':>7}{'p50 s':>7}{'$/q':>9}"
    print(hdr)
    for label in labels:
        run = load(label)
        common = [qid for qid in run if qid in gold]
        tp = fp = fn = origin_ok = any_ok = 0
        for qid in common:
            g, r = set(gold[qid]["matches"]), set(run[qid]["matches"])
            tp += len(g & r); fp += len(r - g); fn += len(g - r)
            origin_ok += earliest(qs[qid], g) == earliest(qs[qid], r)
            any_ok += bool(g) == bool(r)
        p = tp / max(1, tp + fp); rc = tp / max(1, tp + fn); f1 = 2 * p * rc / max(1e-9, p + rc)
        rs = [run[q] for q in common]
        print(f"{label:<22}{len(common):>4}{p:>7.2f}{rc:>7.2f}{f1:>7.2f}"
              f"{origin_ok / len(common):>8.2f}{any_ok / len(common):>7.2f}"
              f"{mean(r['thinking_tokens'] for r in rs):>7.0f}{median(r['latency_s'] for r in rs):>7.1f}"
              f"{mean(r['cost_usd'] for r in rs):>9.5f}")
    print("\norigin = earliest match agrees with gold (incl. both 'none'); any = agrees on whether any match exists;"
          "\nthink = mean thinking tokens; $/q = notional list-price cost per query (subscription-billed here)")


if __name__ == "__main__":
    main(*sys.argv[1:])
