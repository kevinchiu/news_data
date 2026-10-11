"""Build the effort-benchmark set: query sentences + BM25 candidates from strictly earlier stories.

Usage: python -I scripts/make_bench.py data/processed/sentences.jsonl bench/queries.jsonl [n_queries] [k]
"""
import json
import random
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from retrieval import BM25, toks  # noqa: E402


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / max(1, len(a | b))


def main(src, dst, n=80, k=15):
    rs = [json.loads(l) for l in open(src)]
    corpus = [toks(r["text"]) for r in rs]
    bm25 = BM25(corpus)
    dates = np.array([r["date"] for r in rs])
    docs = np.array([r["doc_id"] for r in rs])

    rng = random.Random(0)
    eligible = [i for i, r in enumerate(rs)
                if r["kind"] == "sentence" and r["date"] >= "1987-03-02"
                and 10 <= len(r["text"].split()) <= 45
                and (re.search(r"\d", r["text"]) or " said" in r["text"])]
    pool = rng.sample(eligible, 1500)

    scored = []
    for qi in pool:
        q = rs[qi]
        s = bm25.get_scores(corpus[qi])
        s[(dates >= q["date"]) | (docs == q["doc_id"])] = -1
        top = np.argsort(-s)[:k]
        top = [int(j) for j in top if s[j] > 0]
        if len(top) < k:
            continue
        scored.append((qi, top, jaccard(corpus[qi], corpus[top[0]])))

    # stratify so the set has plenty of likely positives, not just novel facts
    hi = [x for x in scored if x[2] >= 0.35]
    mid = [x for x in scored if 0.15 <= x[2] < 0.35]
    lo = [x for x in scored if x[2] < 0.15]
    picks = rng.sample(hi, min(len(hi), n * 4 // 10))
    picks += rng.sample(mid, min(len(mid), n * 3 // 10))
    picks += rng.sample(lo, n - len(picks))
    print(f"pool strata hi={len(hi)} mid={len(mid)} lo={len(lo)}; picked {len(picks)}")

    with open(dst, "w") as fh:
        for qi, top, jac in picks:
            q = rs[qi]
            fh.write(json.dumps({
                "qid": q["sid"], "date": q["date"], "text": q["text"], "top_jaccard": round(jac, 3),
                "candidates": [{"sid": rs[j]["sid"], "date": rs[j]["date"], "kind": rs[j]["kind"],
                                "text": rs[j]["text"]} for j in top],
            }) + "\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], *(int(a) for a in sys.argv[3:]))
