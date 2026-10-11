"""Build a held-out "watch set" for the Quotebank provenance benchmark.

Items are selected and labelled exactly like bench/qb_bench.jsonl (see make_qb_bench.py: same
quote filters, same ground truth, same cached Haiku paraphrase), but:
  - any quote whose id or any of whose linked variant ids appears in the bench (quote_id or
    variant_ids) is excluded, and so is any quote sharing ids with an earlier watch item;
  - candidates are taken in a different deterministic order, hash(quote_id || 'watch');
  - every item has "split": "watch".

The improvement loop never reads this file; scripts/watch_scorer.py scores it for display only.

Usage: python -I scripts/make_watch_set.py [bench/qb_watch.jsonl] [n=100] [min_domains=5]
"""
import datetime as dt
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_qb_bench import paraphrase  # noqa: E402
from verify import LLMCache  # noqa: E402

BENCH = "bench/qb_bench.jsonl"


def bench_ids(path=BENCH):
    ids = set()
    for line in open(path):
        it = json.loads(line)
        ids.add(it["quote_id"])
        ids.update(it["variant_ids"])
    return ids


def main(dst="bench/qb_watch.jsonl", n=100, min_domains=5):
    excluded = bench_ids()
    con = duckdb.connect("data/news.duckdb", read_only=True)
    con.execute("CREATE TEMP TABLE bench_ids(id VARCHAR)")
    con.executemany("INSERT INTO bench_ids VALUES (?)", [[i] for i in excluded])
    # same filters as make_qb_bench.py, plus the bench exclusion and a different order; take
    # a surplus because a few candidates fail the timed-mention check or overlap each other
    rows = con.execute("""
      WITH m AS (
        SELECT m.*, coalesce(d.kind, 'outlet') AS kind FROM quote_mentions m LEFT JOIN domains d USING (domain)),
      per_quote AS (
        SELECT quote_id, count(DISTINCT domain) AS n_domains, min(date) AS first_date FROM m GROUP BY 1)
      SELECT q.quote_id, q.quotation, q.speaker, p.n_domains
      FROM per_quote p JOIN quotes q USING (quote_id)
      WHERE p.n_domains >= ? AND q.speaker IS NOT NULL AND q.speaker_proba >= 0.7
        AND len(string_split(q.quotation, ' ')) BETWEEN 12 AND 60
        AND EXISTS (SELECT 1 FROM m WHERE m.quote_id = p.quote_id AND m.date >= p.first_date + INTERVAL 6 HOUR
                    AND m.has_time)
        AND q.quote_id NOT IN (SELECT id FROM bench_ids)
        AND NOT EXISTS (SELECT 1 FROM quote_links l WHERE l.a = q.quote_id AND l.b IN (SELECT id FROM bench_ids))
        AND NOT EXISTS (SELECT 1 FROM quote_links l WHERE l.b = q.quote_id AND l.a IN (SELECT id FROM bench_ids))
      ORDER BY hash(q.quote_id || 'watch') LIMIT ?""", [min_domains, 3 * n]).fetchall()
    print(f"{len(rows)} candidates")

    # ground truth exactly as make_qb_bench.build (the paraphrase is made later, for kept items only)
    def label(row):
        qid, quote, speaker, n_domains = row
        cur = con.cursor()
        ids = [qid] + [r[0] for r in cur.execute(
            "SELECT b FROM quote_links WHERE a = ? UNION SELECT a FROM quote_links WHERE b = ?", [qid, qid]).fetchall()]
        ms = cur.execute("""
          SELECT m.date, m.domain, m.url, coalesce(d.kind, 'outlet'), m.has_time FROM quote_mentions m
          LEFT JOIN domains d USING (domain) WHERE m.quote_id IN (SELECT unnest(?)) ORDER BY m.date, m.url""",
                         [ids]).fetchall()
        ms = [r for r in ms if r[4]]
        later = next((r for r in ms if r[0] >= ms[0][0] + dt.timedelta(hours=6)), None)
        if later is None:
            return None
        before = [r for r in ms if r[0] < later[0]]
        orig = [r for r in before if r[3] == "outlet"]
        return {
            "quote_id": qid, "split": "watch", "variant_ids": ids, "speaker": speaker, "quotation": quote,
            "n_domains": n_domains, "as_of": later[0].isoformat(), "query_url": later[2],
            "verbatim": f'{speaker} said: "{quote}"',
            "gt_first_any": {"date": before[0][0].isoformat(), "domain": before[0][1], "url": before[0][2],
                             "has_time": before[0][4]},
            "gt_first_original": ({"date": orig[0][0].isoformat(), "domain": orig[0][1], "url": orig[0][2]}
                                  if orig else None),
        }

    with ThreadPoolExecutor(6) as ex:
        labelled = list(ex.map(label, rows))
    kept, seen = [], set(excluded)
    for it in labelled:
        if it is None or seen & set(it["variant_ids"]):
            continue
        seen.update(it["variant_ids"])
        kept.append(it)
        if len(kept) == n:
            break
    print(f"{len(kept)} items kept (need {n})")

    cache = LLMCache()
    with ThreadPoolExecutor(6) as ex:
        paras = list(ex.map(lambda it: paraphrase(cache, it["speaker"], it["quotation"]), kept))
    out = []
    for it, p in zip(kept, paras):
        # same key order as the bench items
        out.append({k: it[k] for k in ("quote_id", "split", "variant_ids", "speaker", "quotation", "n_domains",
                                       "as_of", "query_url", "verbatim")}
                   | {"paraphrase": p, "gt_first_any": it["gt_first_any"],
                      "gt_first_original": it["gt_first_original"]})

    overlap = [it["quote_id"] for it in out if excluded & set(it["variant_ids"])]
    assert not overlap, f"overlap with bench: {overlap}"
    with open(dst, "w") as fh:
        fh.writelines(json.dumps(it) + "\n" for it in out)
    print(f"{len(out)} watch items -> {dst} (0 ids shared with {BENCH}'s {len(excluded)} ids)")


if __name__ == "__main__":
    main(*(sys.argv[1:2] or ["bench/qb_watch.jsonl"]), *(int(a) for a in sys.argv[2:]))
