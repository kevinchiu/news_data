"""Score an event-tracer config on a split of the event benchmark (bench/ev_bench.jsonl, GT v3).

Usage: python -I scripts/evaluate_event.py <config.json> <train|val|test|watch> [--mode query|query_title]
           [--limit N] [--retrieval-only] [--nonce STR] [--workers 4] [--out rows.jsonl]

The tracer's first report is correct when it is one of the ground truth's first-wave articles
(the labelled first report and same-event articles up to 30 minutes after it), a same-event
article from that half hour, or a syndicated copy of a first-wave headline from that half hour.

Metrics (fractions of items unless noted):
  recall_at_k       a first-wave headline is among the retrieved candidates  [no LLM]
  verifier_recall   a first-wave headline is matched, given it was retrieved
  first_report      first report correct (headline metric)
  early             predicted first report is earlier than the ground truth's: a wrong match, or a
                    report the label missed (these are logged for audit, not silently trusted)
  late              predicted first report comes after the first wave
  none              nothing matched
  calls_per_claim   Haiku calls per item (cached calls included)
"""
import argparse
import datetime as dt
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ev_config  # noqa: E402
from trace_event import candidates, title_key, trace  # noqa: E402
from verify import LLMCache  # noqa: E402

BENCH, WATCH = "bench/ev_bench.jsonl", "bench/ev_watch.jsonl"
WAVE = dt.timedelta(minutes=30)


def load_items(split, limit=None):
    items = [json.loads(l) for l in open(WATCH if split == "watch" else BENCH)]
    items = [it for it in items if it["split"] == split]
    return items[:limit] if limit else items


def item_id(it):
    return str(it["event_id"])


def gt_sets(it):
    t0 = dt.datetime.fromisoformat(it["gt_t0"])
    wave = {a["url"] for a in it["gt_first_wave"]}
    same = {a["url"]: dt.datetime.fromisoformat(a["date"]) for a in it["gt_same_event"]}
    keys = {title_key(a["title"]) for a in it["gt_first_wave"]}
    return t0, wave, same, keys


def in_wave(a, t0, wave, same, keys):
    d = dt.datetime.fromisoformat(a["date"])
    return a["url"] in wave or (d <= t0 + WAVE and (a["url"] in same or title_key(a["title"]) in keys))


def evaluate(cfg, items, mode="query", retrieval_only=False, nonce="", workers=4, con=None):
    con = con or duckdb.connect("data/news.duckdb", read_only=True)
    con.execute("LOAD fts")
    cache, local = LLMCache(), threading.local()

    def cursor():
        if not hasattr(local, "con"):
            local.con = con.cursor()
            local.con.execute("LOAD fts")
            local.con.execute("SET threads = 3")
        return local.con

    def one(it):
        t0, wave, same, keys = gt_sets(it)
        hit = lambda g: any(in_wave(a, t0, wave, same, keys) for a in g["articles"])  # noqa: E731
        if retrieval_only:
            cands = candidates(cursor(), it[mode], it["as_of"], cfg, cache)
            return {"item_id": item_id(it), "retrieved": any(map(hit, cands))}
        res = trace(cursor(), cache, it[mode], it["as_of"], cfg, nonce=nonce)
        pred = res["first"]
        d = dt.datetime.fromisoformat(pred["date"]) if pred else None
        correct = bool(pred and in_wave(pred, t0, wave, same, keys))
        return {
            "item_id": item_id(it),
            "retrieved": any(map(hit, res["candidates"])),
            "gt_matched": any(map(hit, res["matched"])),
            "first_report": correct,
            "early": bool(pred and not correct and d < t0),
            "late": bool(pred and not correct and d >= t0),
            "none": pred is None,
            "calls": res["llm_calls"]["made"] + res["llm_calls"]["cached"],
            "made": res["llm_calls"]["made"],
            "matched": [g["title"] for g in res["matched"]],
            "pred": pred and {k: pred[k] for k in ("date", "source", "url", "title")},
        }

    with ThreadPoolExecutor(workers) as ex:
        rows = list(ex.map(one, items))
    n = len(rows)
    m = {"n": n, "recall_at_k": sum(r["retrieved"] for r in rows) / n}
    if not retrieval_only:
        retrieved = [r for r in rows if r["retrieved"]]
        m |= {
            "verifier_recall": sum(r["gt_matched"] for r in retrieved) / max(1, len(retrieved)),
            "first_report": sum(r["first_report"] for r in rows) / n,
            "early": sum(r["early"] for r in rows) / n,
            "late": sum(r["late"] for r in rows) / n,
            "none": sum(r["none"] for r in rows) / n,
            "calls_per_claim": sum(r["calls"] for r in rows) / n,
            "new_calls": sum(r["made"] for r in rows),
        }
    return m, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("split")
    ap.add_argument("--mode", default="query")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--retrieval-only", action="store_true")
    ap.add_argument("--nonce", default="")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", help="write per-item rows (JSONL)")
    a = ap.parse_args()
    m, rows = evaluate(ev_config.load(a.config), load_items(a.split, a.limit), a.mode, a.retrieval_only,
                       a.nonce, a.workers)
    print(json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in m.items()}))
    if a.out:
        Path(a.out).write_text("".join(json.dumps(r) + "\n" for r in rows))


if __name__ == "__main__":
    main()
