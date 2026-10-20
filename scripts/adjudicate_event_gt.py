"""GT v3.2: let the labeller adjudicate earlier reports that a tracer run found.

A label can only be as early as its candidate pool, so a tracer that finds a real earlier report
is otherwise scored wrong ("early"); likewise a same-event report inside the first 30 minutes that
the labeller never saw is scored "late". This pass takes those predictions in evaluate_event.py
row files, asks the GT labeller (label_event_gt.MODEL / EFFORT, not the tracer's verifier) whether
each reports the same event as the query, and adds the accepted ones to the ground truth
(re-deriving GT_t0 and the first wave). v3.2 = the baseline's predictions on every split, before
the loop; v3.3 = the predictions of every config the loop scored in full (train/val) and of the
baseline and the final candidates (all splits), before the test score; v3.4 adds "late" answers
inside the first wave (and the early answers of the tracer fixes). Predictions already
adjudicated are skipped; the previous version's files are kept as *.jsonl.v<old version>.

Usage: python -I scripts/adjudicate_event_gt.py VERSION ROWS.jsonl [ROWS.jsonl ...]
"""
import datetime as dt
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from label_event_gt import EFFORT, MODEL  # noqa: E402
from verify import LLMCache, claude_json  # noqa: E402

PATHS = ("bench/ev_bench.jsonl", "bench/ev_watch.jsonl")
BATCH = 10
SYSTEM = """You check ground-truth labels for a news-provenance benchmark. Each item gives a \
later article's headline (the QUERY), the event it reports, the earliest report of that event \
found so far, and one CANDIDATE article published before it or within 30 minutes after it (times in UTC). Say whether the \
candidate reports the same event: a wire copy or independent report of it counts, even if worded \
differently, naming people the query only describes (or the reverse), or lacking a detail the \
query adds; reports of the same incident while it is unfolding count. Plans or previews ("to \
arrive", "set to", "heads to"), separate earlier developments, similar events elsewhere and \
re-posts of an older story about an earlier occurrence do not. Judge from headlines and times; \
when unsure, answer false."""
SCHEMA = json.dumps({"type": "object", "properties": {"items": {"type": "array", "items": {
    "type": "object", "properties": {"item": {"type": "integer"}, "same_event": {"type": "boolean"}},
    "required": ["item", "same_event"], "additionalProperties": False}}},
    "required": ["items"], "additionalProperties": False})


def main(version, *row_files):
    items = {str(it["event_id"]): (p, it) for p in PATHS for it in map(json.loads, open(p))}
    old = {p: str(max(json.loads(l)["gt_version"] for l in open(p))).replace(".", "_") for p in PATHS}
    early = {iid: {a["url"]: None for a in it.get("gt_adjudicated", [])} for iid, (p, it) in items.items()}
    for f in row_files:
        for r in map(json.loads, open(f)):
            if r["item_id"] not in items or not r.get("pred") or r["pred"]["url"] in early.get(r["item_id"], {}):
                continue
            t0 = dt.datetime.fromisoformat(items[r["item_id"]][1]["gt_t0"])
            in_wave_late = r.get("late") and dt.datetime.fromisoformat(r["pred"]["date"]) <= t0 + dt.timedelta(minutes=30)
            if r.get("early") or in_wave_late:
                early.setdefault(r["item_id"], {})[r["pred"]["url"]] = r["pred"]
    todo = [(iid, c) for iid, cs in early.items() for c in cs.values() if c]
    print(f"{len(todo)} earlier predictions to adjudicate", flush=True)
    cache = LLMCache()

    def judge(batch):
        lines = []
        for i, (iid, c) in enumerate(batch, 1):
            it = items[iid][1]
            first = it["gt_first_wave"][0]
            lines += [f"ITEM {i}", f"QUERY ({it['as_of']}): {it['query_title']}",
                      f"EVENT: {it['gt_label']['query_event']}",
                      f"EARLIEST SO FAR: {first['date'][:16]} {first['source']}: {first['title']}",
                      f"CANDIDATE: {c['date'][:16]} {c['source']}: {c['title']}", ""]
        p = "\n".join(lines)

        def compute():
            out, stats = claude_json(p, SYSTEM, SCHEMA, MODEL, EFFORT)
            return {**out, **stats}
        r = cache.cached([MODEL, EFFORT, SYSTEM, p], compute)
        return {x["item"]: x["same_event"] for x in r["items"]}

    batches = [todo[i:i + BATCH] for i in range(0, len(todo), BATCH)]
    with ThreadPoolExecutor(4) as ex:
        verdicts = list(ex.map(judge, batches))
    accepted = 0
    for batch, v in zip(batches, verdicts):
        for i, (iid, c) in enumerate(batch, 1):
            it = items[iid][1]
            it.setdefault("gt_adjudicated", []).append({**c, "same_event": bool(v.get(i))})
            if v.get(i):
                accepted += 1
                it["gt_same_event"].append({k: c[k] for k in ("url", "source", "date", "title")})
    for p, it in items.values():
        it["gt_version"] = float(version)
        if not any(a["same_event"] for a in it.get("gt_adjudicated", [])):
            continue  # (an item accepted in an earlier pass is re-derived identically)
        same = sorted({(a["url"], a["source"], dt.datetime.fromisoformat(a["date"]), a["title"]) for a in it["gt_same_event"]},
                      key=lambda a: (a[2], a[0]))
        t0 = same[0][2]
        it.update({
            "gt_t0": t0.isoformat(),
            "gt_first_wave": [{"url": u, "source": s, "date": d.isoformat(), "title": t}
                              for u, s, d, t in same if (d - t0).total_seconds() <= 1800],
            "gt_first_slot": [u for u, s, d, t in same if d == t0],
            "gt_same_event": [{"url": u, "source": s, "date": d.isoformat(), "title": t} for u, s, d, t in same],
        })
    for p in PATHS:
        backup = Path(p).with_suffix(f".jsonl.v{old[p]}")
        if not backup.exists():
            backup.write_text(Path(p).read_text())
        rows = [it for q, it in items.values() if q == p]
        Path(p).write_text("".join(json.dumps(it) + "\n" for it in rows))
    print(f"accepted {accepted} of {len(todo)}; previous files kept as *.jsonl.v{old[PATHS[0]]}")


if __name__ == "__main__":
    main(*sys.argv[1:])
