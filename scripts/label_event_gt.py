"""Model-labelled ground truth (GT v3.1) for the event benchmark.

Titles alone cannot decide "same event" (the audits of structural GT v1/v2 found 18-24% wrong
events and missed earlier reports in about half of the rest), so the judgement is made once, by
Claude Sonnet 5.5 (low effort; a stronger model than the tracer's Haiku verifier), and frozen; the
loop cannot change it.

Candidates are found without a model, from three searches over GKG articles seen from 72 h before
the structural GT_t0 (bench/ev_bench.jsonl.v2) up to the query time: content-word IDF overlap with
the query and first wave, BM25 over headlines (gkg_titles), and shared GKG person/organisation
names with the query article and first wave (finds first-hand outlets that name people the
query's headline only describes). Each time window gets N_CAND / 2 headlines, filled round-robin
from the three searches, plus the structural first wave. Syndicated copies are one candidate.
The model sees BATCH items per call (headlines, outlets and times only), says whether the query
reports a discrete event, and lists the candidates that report that same event.

GT v3.1 changes, after two Fable reviews of 20 v3 items (13 right, 4 missed earlier reports, 2
wrong event, 1 unsure; 4 items were not discrete events): the name and BM25 searches, a rubric
for unfolding incidents, previews and stale re-posts, and dropping non-events. (A rule dropping a
lone report seen hours before any other outlet's was tried and removed: adjudication showed it
mostly discarded genuine first reports whose followers the candidate pool had missed.)
GT_t0 = the earliest remaining same-event article; the first wave = those up to GT_t0 + 30 min.

Usage: python -I scripts/label_event_gt.py [bench/ev_bench.jsonl bench/ev_watch.jsonl]
"""
import datetime as dt
import html
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_event_gt import earlier_candidates, ensure_title_df  # noqa: E402
from make_event_bench import grams  # noqa: E402
from verify import LLMCache, claude_json  # noqa: E402

MODEL, EFFORT, BATCH, N_CAND, WORKERS = "claude-sonnet-5-5", "low", 8, 30, 4
SYSTEM = """You label a news-provenance benchmark. Each item names a news event through a later \
article's headline (the QUERY) and lists numbered CANDIDATE articles published before it, with \
time (UTC), outlet and headline; syndicated copies are grouped as one candidate.

First decide whether the query reports a discrete event: a specific happening such as an \
announcement, incident, ruling, death, release, decision, filing or arrival. Analyses, \
explainers, reviews, columns, previews of scheduled events and descriptions of an ongoing \
situation are not discrete events.

Then list every candidate that reports that same event, including wire copies and independent \
reports, even if worded differently, naming people the query only describes (or the reverse), or \
lacking a detail the query adds (a later death toll, the victims' nationality). Reports of the \
same incident while it is unfolding (a shooting under way, a gunfight that later ends) count. Do \
not list articles that only share the topic, announce plans or previews ("to arrive", "set to", \
"heads to"), report an earlier or later separate development (an earlier case count, a different \
city's measure), report a similar event elsewhere, or re-post an older story about an earlier \
occurrence (use the times). Judge from headlines; leave out headlines too vague to tell. Also \
state the query event in one short sentence."""
SCHEMA = json.dumps({"type": "object", "properties": {"items": {"type": "array", "items": {
    "type": "object", "properties": {
        "item": {"type": "integer"},
        "discrete_event": {"type": "boolean"},
        "query_event": {"type": "string"},
        "same_event": {"type": "array", "items": {"type": "integer"}}},
    "required": ["item", "discrete_event", "query_event", "same_event"], "additionalProperties": False}}},
    "required": ["items"], "additionalProperties": False})


def names(con, urls, limit=4):
    """The most frequent GKG person/organisation names of these articles (most specific first)."""
    counts = {}
    for persons, orgs in con.execute("SELECT persons, organizations FROM gdelt_gkg WHERE url IN (SELECT unnest(?))",
                                     [list(urls)]).fetchall():
        for field in (persons, orgs):
            for n in {x.rsplit(",", 1)[0].strip().lower() for x in (field or "").split(";") if x}:
                if len(n) >= 5 and "'" not in n:
                    counts[n] = counts.get(n, 0) + 1
    return [n for n, _ in sorted(counts.items(), key=lambda kv: (-kv[1], -len(kv[0]), kv[0]))[:limit]]


def name_candidates(con, it, lo, hi, limit):
    ns = names(con, [it["query_url"]] + [a["url"] for a in it["gt_first_wave"][:5]])
    if not ns:
        return []
    need = min(2, len(ns))
    score = " + ".join(f"(contains(lower(coalesce(persons, '')), '{n}') OR contains(lower(coalesce(organizations, '')), '{n}'))::INT"
                       for n in ns)
    return con.execute(f"""
      SELECT url, source, date, title FROM (
        SELECT url, source, date, title, {score} AS s FROM gdelt_gkg
        WHERE date >= ? AND date < ? AND title IS NOT NULL)
      WHERE s >= {need} ORDER BY s DESC, date, url LIMIT {limit}""", [lo, hi]).fetchall()


def bm25_candidates(con, it, lo, hi, limit):
    return con.execute("""
      SELECT url, source, date, title FROM (
        SELECT *, fts_main_gkg_titles.match_bm25(url, ?) AS score FROM gkg_titles WHERE date >= ? AND date < ?)
      WHERE score IS NOT NULL ORDER BY round(score, 6) DESC, url LIMIT ?""",
                       [re.sub(r"[^\w\s]", " ", it["query_title"]), lo, hi, limit]).fetchall()


def candidates(con, it):
    """Structural first wave + three searches per time window (before GT_t0, then GT_t0 .. query
    time), round-robin, grouped by normalised title, sorted by the earliest copy's time."""
    t0, as_of = dt.datetime.fromisoformat(it["gt_t0"]), dt.datetime.fromisoformat(it["as_of"])
    wave = [(a["url"], a["source"], dt.datetime.fromisoformat(a["date"]), a["title"]) for a in it["gt_first_wave"]]
    windows = []
    for lo, hi in ((t0 - dt.timedelta(hours=72), t0), (t0, as_of)):
        lists = [[r[:4] for r in earlier_candidates(con, it, lo=lo, hi=hi, limit=2 * N_CAND, min_share=0.4)],
                 bm25_candidates(con, it, lo, hi, 2 * N_CAND), name_candidates(con, it, lo, hi, 2 * N_CAND)]
        windows.append([r for k in range(2 * N_CAND) for lst in lists if k < len(lst) for r in [lst[k]]])
    groups, per_source = {}, {}
    for part, rows in enumerate([wave] + windows):
        added = 0
        for url, src, d, title in rows:
            if url == it["query_url"] or not title:
                continue
            title = html.unescape(title)
            # first 8 words without numbers: photo-gallery pages and minor title edits group together
            key = " ".join([w for w in grams(title, 1)[0] if not w.isdigit()][:8])
            if key not in groups:
                # neighbours: N_CAND / 2 headlines per window, at most 2 per outlet
                if part and (added >= N_CAND // 2 or per_source.get(src, 0) >= 2):
                    continue
                per_source[src] = per_source.get(src, 0) + 1
                added += 1
            groups.setdefault(key, {}).setdefault(url, (url, src, d, title))
    return sorted((sorted(g.values(), key=lambda c: (c[2], c[0])) for g in groups.values()), key=lambda g: (g[0][2], g[0][0]))


def prompt(batch):
    lines = []
    for i, (it, cands) in enumerate(batch, 1):
        lines += [f"ITEM {i}", f"QUERY ({it['as_of']}): {it['query_title']}", "CANDIDATES:"]
        for j, g in enumerate(cands, 1):
            url, src, d, title = g[0]
            more = f" (+{len(g) - 1} copies)" if len(g) > 1 else ""
            lines.append(f"  {j}. {d:%Y-%m-%d %H:%M} {src}: {title}{more}")
        lines.append("")
    return "\n".join(lines)


def label_batches(items, cands, model, effort):
    """Label items BATCH per call; returns ([{item no.: label}, ...], batches)."""
    cache = LLMCache()
    batches = [list(zip(items, cands))[i:i + BATCH] for i in range(0, len(items), BATCH)]

    def label(batch):
        p = prompt(batch)

        def compute():
            out, stats = claude_json(p, SYSTEM, SCHEMA, model, effort)
            return {**out, **stats}
        r = cache.cached([model, effort, SYSTEM, p], compute)
        return {x["item"]: x for x in r["items"]}, r.get("output_tokens", 0), r.get("cached")

    with ThreadPoolExecutor(WORKERS) as ex:
        results = list(ex.map(label, batches))
    print(f"{len(batches)} {model} calls ({sum(not c for _, _, c in results)} new), "
          f"{sum(t for _, t, _ in results):,} output tokens", flush=True)
    return [r[0] for r in results], batches


def main(*paths):
    paths = paths or ("bench/ev_bench.jsonl", "bench/ev_watch.jsonl")
    # the structural GT v2 files are the input; they are kept as *.jsonl.v2 on the first run
    src = [Path(p).with_suffix(".jsonl.v2") if Path(p).with_suffix(".jsonl.v2").exists() else Path(p) for p in paths]
    items = [json.loads(l) for p in src for l in open(p)]
    ensure_title_df()
    con = duckdb.connect("data/news.duckdb", read_only=True)
    con.execute("SET threads = 3")
    con.execute("LOAD fts")
    with ThreadPoolExecutor(WORKERS) as ex:
        cands = list(ex.map(lambda it: candidates(con.cursor(), it), items))
    print(f"{len(items)} items, {sum(map(len, cands)) / len(items):.1f} candidate headlines each", flush=True)
    results, batches = label_batches(items, cands, MODEL, EFFORT)
    kept = {p: [] for p in paths}
    dropped = {"not a discrete event": 0, "no same-event candidate": 0}
    for labels, batch in zip(results, batches):
        for i, (it, cs) in enumerate(batch, 1):
            lab = labels.get(i, {"same_event": [], "query_event": "", "discrete_event": False})
            same = sorted({c for j in lab["same_event"] if 1 <= j <= len(cs) for c in cs[j - 1]}, key=lambda c: (c[2], c[0]))
            if not lab["discrete_event"]:
                dropped["not a discrete event"] += 1
                continue
            if not same:
                dropped["no same-event candidate"] += 1
                continue
            t0 = same[0][2]
            wave = [c for c in same if (c[2] - t0).total_seconds() <= 1800]
            it.update({
                "gt_t0": t0.isoformat(),
                "gt_first_wave": [{"url": u, "source": s, "date": d.isoformat(), "title": t} for u, s, d, t in wave],
                "gt_first_slot": [u for u, s, d, t in same if d == t0],
                "gt_same_event": [{"url": u, "source": s, "date": d.isoformat(), "title": t} for u, s, d, t in same],
                "gt_label": {"model": MODEL, "effort": EFFORT, "query_event": lab["query_event"],
                             "n_candidates": len(cs), "n_candidate_articles": sum(map(len, cs))},
                "gt_version": 3.1,
            })
            kept[paths[0] if it["split"] != "watch" else paths[-1]].append(it)
    for p, rows in kept.items():
        if not Path(p).with_suffix(".jsonl.v2").exists():
            Path(p).with_suffix(".jsonl.v2").write_text(Path(p).read_text())
        with open(p, "w") as fh:
            fh.writelines(json.dumps(it) + "\n" for it in rows)
    split = {s: sum(it["split"] == s for rows in kept.values() for it in rows) for s in ("train", "val", "test", "watch")}
    print(f"kept {split}; dropped {dropped}; v2 files saved as *.jsonl.v2")


if __name__ == "__main__":
    main(*sys.argv[1:])
