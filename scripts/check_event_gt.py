"""Opus spot-check of the event benchmark's ground truth (bench/ev_bench.jsonl) before any loop run.

For a sample of train/val items (test stays unseen), Claude Opus 5.5 sees the query (headline
and rewritten sentence), the ground truth's earliest first-wave articles, and up to 10 earlier
GKG articles (72 h before GT_t0) whose titles share the most content words with the query or
the first wave. It judges whether the ground truth reports the same event as the query, and
whether any earlier article does (a GT miss). Results are cached, so re-runs are free.

Usage: python -I scripts/check_event_gt.py [bench/ev_bench.jsonl] [n=50] [out/event_gt_check.jsonl]
"""
import datetime as dt
import hashlib
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_event_bench import content_grams, grams, STOP  # noqa: E402
from verify import LLMCache, claude_json  # noqa: E402

MODEL, EFFORT = "claude-opus-5-5", "medium"
SYSTEM = """You audit a news-provenance benchmark. An item names a news event through a later \
article's headline, and claims that a set of earlier articles were the first to report that same \
event. Judge from headlines, outlets and times only. "Same event" means the same specific \
happening (the same announcement, incident, ruling, death, release...), not merely the same topic \
or an earlier stage of a developing story. Be strict and concrete."""
SCHEMA = json.dumps({"type": "object", "properties": {
    "gt_same_event": {"type": "string", "enum": ["yes", "no", "unclear"]},
    "earlier_same_event": {"type": "array", "items": {"type": "integer"},
                           "description": "numbers of EARLIER candidates that report the same event"},
    "note": {"type": "string"}},
    "required": ["gt_same_event", "earlier_same_event", "note"], "additionalProperties": False})


def ensure_title_df(path="data/news.duckdb"):
    """Document frequency of content words over all GKG titles (table gkg_title_df), built once."""
    with duckdb.connect(path, read_only=True) as ro:
        if ro.execute("SELECT count(*) FROM information_schema.tables WHERE table_name = 'gkg_title_df'").fetchone()[0]:
            return
    con = duckdb.connect(path)
    con.execute("SET threads = 3")
    stop = "[" + ", ".join(f"'{w}'" for w in STOP) + "]"
    con.execute(f"""
    CREATE TABLE gkg_title_df AS
      SELECT w, count(*) AS n FROM (
        SELECT DISTINCT url, w FROM gdelt_gkg, unnest(string_split(regexp_replace(lower(title), '[^a-z0-9 ]+', ' ', 'g'), ' ')) AS t(w)
        WHERE title IS NOT NULL AND length(w) > 2 AND NOT list_contains({stop}, w))
      GROUP BY 1""")
    con.close()


def earlier_candidates(con, it, lo=None, hi=None, limit=10, min_share=0.3):
    """GKG articles in [lo, hi) whose titles match the query's (and first wave's) content words,
    scored by summed IDF; at least min_share of the query title's IDF mass must match."""
    words = set()
    for t in [it["query_title"]] + [a["title"] for a in it["gt_first_wave"][:5]]:
        words |= {w for w in grams(t, 1)[0] if w not in STOP and len(w) > 2 and w.isalnum()}
    if not words:
        return []
    n_docs = con.execute("SELECT count(*) FROM gdelt_gkg").fetchone()[0]
    df = dict(con.execute("SELECT w, n FROM gkg_title_df WHERE w IN (SELECT unnest(?))", [sorted(words)]).fetchall())
    idf = {w: math.log(n_docs / df.get(w, 1)) for w in words}
    q_mass = sum(idf[w] for w in grams(it["query_title"], 1)[0] if w in idf)
    t0 = dt.datetime.fromisoformat(it["gt_t0"])
    score = " + ".join(f"regexp_matches(lower(title), '\\b{w}\\b')::INT * {idf[w]:.3f}" for w in sorted(words))
    return con.execute(f"""
      SELECT url, source, date, title, s FROM (
        SELECT url, source, date, title, {score} AS s FROM gdelt_gkg
        WHERE date >= ? AND date < ? AND title IS NOT NULL)
      WHERE s >= ? ORDER BY s DESC, date, url LIMIT {limit}""",
                       [lo or t0 - dt.timedelta(hours=72), hi or t0, min_share * q_mass]).fetchall()


def prompt(it, earlier):
    wave = it["gt_first_wave"][:5]
    lines = [f"QUERY (seen {it['as_of']}): headline: {it['query_title']}", f"  rewritten: {it['query']}", "",
             f"GROUND TRUTH, first reports (earliest {it['gt_t0']}):"]
    lines += [f"  - {a['date']} {a['source']}: {a['title']}" for a in wave]
    lines += ["", "EARLIER candidates (before the ground truth):"]
    lines += [f"  {i}. {d} {src}: {title}" for i, (url, src, d, title, s) in enumerate(earlier, 1)] or ["  (none)"]
    lines += ["", "Does the ground truth report the same event as the query? Which earlier candidates, if any, "
              "report that same event?"]
    return "\n".join(lines)


def main(path="bench/ev_bench.jsonl", n=50, dst="out/event_gt_check.jsonl"):
    items = [json.loads(l) for l in open(path)]
    items = sorted((it for it in items if it["split"] in ("train", "val")),
                   key=lambda it: hashlib.md5(str(it["event_id"]).encode()).hexdigest())[:n]
    con = duckdb.connect("data/news.duckdb", read_only=True)
    con.execute("SET threads = 3")
    cache = LLMCache()

    def check(it):
        earlier = earlier_candidates(con.cursor(), it)
        p = prompt(it, earlier)

        def compute():
            out, stats = claude_json(p, SYSTEM, SCHEMA, MODEL, EFFORT)
            return {**out, **stats}
        r = cache.cached([MODEL, EFFORT, SYSTEM, p], compute)
        return {"event_id": it["event_id"], "split": it["split"], "query_title": it["query_title"],
                "gt_title": it["gt_first_wave"][0]["title"], "gt_same_event": r["gt_same_event"],
                "earlier": [{"date": str(earlier[i - 1][2]), "source": earlier[i - 1][1], "title": earlier[i - 1][3],
                             "url": earlier[i - 1][0]} for i in r["earlier_same_event"] if 1 <= i <= len(earlier)],
                "note": r["note"]}

    with ThreadPoolExecutor(4) as ex:
        rows = list(ex.map(check, items))
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in rows)
    k = {v: sum(r["gt_same_event"] == v for r in rows) for v in ("yes", "no", "unclear")}
    miss = sum(bool(r["earlier"]) for r in rows if r["gt_same_event"] == "yes")
    print(f"{len(rows)} items: GT same event {k} -> precision {k['yes'] / len(rows):.2f}; "
          f"GT correct but an earlier report exists: {miss} -> {dst}")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(*[x for x in a if not x.isdigit()][:1], *(int(x) for x in a if x.isdigit()),
         *[x for x in a if not x.isdigit()][1:])
