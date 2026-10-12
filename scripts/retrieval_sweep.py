"""Exhaustive sweep of the retrieval subspace: recall@k on the 145 train items, no LLM calls.

  .venv/bin/python -I scripts/retrieval_sweep.py [--workers 8] [--check]

Grid: k in {5,10,15,20,30,45,60} x retrieval.index x retrieval.query_template x
retrieval.speaker_filter (84 configs), for both query forms (paraphrase, paraphrase_hard).
The verify knobs are held at config/base.json (they do not affect retrieval).

BM25 retrieval is a deterministic ranking cut at k (ORDER BY rounded score, quote_id LIMIT k),
so the top-k list for any k is a prefix of the top-60 list. Each (index, template, filter)
combination is therefore retrieved once at k=60, and the rank of the first ground-truth variant
is stored per item; recall@k for every k in 1..60 follows exactly. `--check` re-scores a few grid
cells with evaluate.evaluate(retrieval_only=True) to confirm this equivalence.

Results are cached in experiments/retrieval_sweep.json (resumable: finished combinations are
skipped). DuckDB is opened read-only with one thread per query; `--workers` queries run at once.
"""
import argparse
import itertools
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import qb_config  # noqa: E402
from trace_quote import candidates  # noqa: E402

OUT = ROOT / "experiments" / "retrieval_sweep.json"
BENCH = ROOT / "bench" / "qb_bench.jsonl"
DB = ROOT / "data" / "news.duckdb"
K_GRID = [5, 10, 15, 20, 30, 45, 60]
K_MAX = 60
MODES = ["paraphrase", "paraphrase_hard"]
INDEXES = ["quotation", "quotation_speaker"]
TEMPLATES = ["raw", "content", "content_plus_speaker"]
FILTERS = ["off", "surname"]


def combos():
    return list(itertools.product(INDEXES, TEMPLATES, FILTERS))


def key(mode, index, template, filt):
    return f"{mode}|{index}|{template}|{filt}"


def load_items():
    items = [json.loads(l) for l in open(BENCH)]
    return [it for it in items if it["split"] == "train"]


def load_cache():
    try:
        return json.loads(OUT.read_text())
    except (OSError, ValueError):
        return {}


def save_cache(data):
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data))
    tmp.replace(OUT)


def make_cfg(base, index, template, filt, k=K_MAX):
    return qb_config.apply(base, [{"path": "retrieval.k", "value": k}, {"path": "retrieval.index", "value": index},
                                  {"path": "retrieval.query_template", "value": template},
                                  {"path": "retrieval.speaker_filter", "value": filt}])


def first_hit_ranks(con, items, mode, cfg, workers):
    """0-based rank of the first ground-truth variant in the top-K_MAX list, or None."""
    local = threading.local()

    def cursor():
        if not hasattr(local, "con"):
            local.con = con.cursor()
            local.con.execute("LOAD fts")
        return local.con

    def one(it):
        ids = set(it["variant_ids"])
        cands = candidates(cursor(), it[mode], it["as_of"], cfg)
        return next((i for i, c in enumerate(cands) if c["quote_id"] in ids), None)

    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(one, items))


def recall(ranks, k):
    return sum(1 for r in ranks if r is not None and r < k) / len(ranks)


def grid_rows(data):
    rows = []
    for mode in MODES:
        for index, template, filt in combos():
            cell = data.get("cells", {}).get(key(mode, index, template, filt))
            if not cell:
                continue
            for k in K_GRID:
                rows.append({"mode": mode, "index": index, "query_template": template, "speaker_filter": filt,
                             "k": k, "recall_at_k": recall(cell["ranks"], k)})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=8, help="concurrent BM25 queries (one DuckDB thread each)")
    ap.add_argument("--check", action="store_true", help="re-score a few cells with evaluate() and compare")
    a = ap.parse_args()

    con = duckdb.connect(str(DB), read_only=True, config={"threads": 1})
    con.execute("LOAD fts")
    items = load_items()
    base = qb_config.load()
    data = load_cache()
    data.setdefault("cells", {})
    data["meta"] = {"split": "train", "n_items": len(items), "k_grid": K_GRID, "k_max": K_MAX, "modes": MODES,
                    "indexes": INDEXES, "templates": TEMPLATES, "filters": FILTERS,
                    "fixed": {"min_urls": base["retrieval"].get("min_urls"),
                              "window_days": base["retrieval"].get("window_days")},
                    "quote_ids": [it["quote_id"] for it in items]}
    t_all = time.time()
    todo = [(m, *c) for m in MODES for c in combos() if key(m, *c) not in data["cells"]]
    print(f"{len(todo)} of {len(MODES) * len(combos())} combinations to run "
          f"({len(todo) * len(K_GRID)} grid cells), {len(items)} items, workers={a.workers}", flush=True)
    for mode, index, template, filt in todo:
        t0 = time.time()
        ranks = first_hit_ranks(con, items, mode, make_cfg(base, index, template, filt), a.workers)
        dt = time.time() - t0
        data["cells"][key(mode, index, template, filt)] = {
            "mode": mode, "index": index, "query_template": template, "speaker_filter": filt,
            "ranks": ranks, "elapsed_s": round(dt, 1), "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
        data["grid"] = grid_rows(data)
        data["meta"]["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        save_cache(data)
        print(f"{mode:<16} {index:<18} {template:<21} {filt:<8} "
              + " ".join(f"@{k}={recall(ranks, k):.3f}" for k in K_GRID) + f"  ({dt:.0f}s)", flush=True)
    data["grid"] = grid_rows(data)
    data["meta"]["total_elapsed_s"] = round(sum(c["elapsed_s"] for c in data["cells"].values()), 1)
    save_cache(data)
    print(f"done in {time.time() - t_all:.0f}s this run; "
          f"{data['meta']['total_elapsed_s']:.0f}s of query time across all runs", flush=True)

    for mode in MODES:
        best = sorted((r for r in data["grid"] if r["mode"] == mode), key=lambda r: (-r["recall_at_k"], r["k"]))[:5]
        print(f"best {mode}: " + "; ".join(f"{r['index']}/{r['query_template']}/{r['speaker_filter']} k={r['k']} "
                                           f"{r['recall_at_k']:.3f}" for r in best))

    if a.check:
        from evaluate import evaluate
        cells = [("paraphrase", "quotation_speaker", "content_plus_speaker", "off", 30),
                 ("paraphrase", "quotation", "raw", "off", 30),
                 ("paraphrase_hard", "quotation", "content", "surname", 15),
                 ("paraphrase_hard", "quotation_speaker", "raw", "off", 60)]
        for mode, index, template, filt, k in cells:
            m, _ = evaluate(make_cfg(base, index, template, filt, k), items, mode=mode, retrieval_only=True,
                            workers=a.workers, con=con)
            mine = recall(data["cells"][key(mode, index, template, filt)]["ranks"], k)
            print(f"check {mode} {index}/{template}/{filt} k={k}: evaluate {m['recall_at_k']:.4f} "
                  f"vs sweep {mine:.4f} {'OK' if abs(m['recall_at_k'] - mine) < 1e-9 else 'MISMATCH'}", flush=True)


if __name__ == "__main__":
    main()
