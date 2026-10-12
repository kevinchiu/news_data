"""Ground truth v2: widen each benchmark item's variant set to near-identical quotes the 8-gram
links missed, then recompute the earliest mention.

v1 used only direct quote_links, so an earlier, slightly different version of the same quote
(e.g. a fuller or trimmed rendering from the same speaker) counted as a *wrong* earlier match.
v2 also accepts a quote as a variant when, relative to the item's quote:
  - its speaker matches (surname) or is unattributed,
  - the shorter of the two quotes has >= 6 words and >= 70% of its words occur in the longer one,
  - they share at least one 5-word run (rules out bag-of-words coincidences),
  - it was first seen within 14 days of the item's quote.
Candidates come from full-text search on the item's quotation. as_of and the query are unchanged.

Usage: python -I scripts/fix_ground_truth.py bench/qb_bench.jsonl [bench/qb_watch.jsonl ...]
Writes each file in place, keeping the previous version as <file>.v1.
"""
import datetime as dt
import json
import re
import shutil
import sys

import duckdb

MIN_WORDS, MIN_CONTAIN, RUN, WINDOW_DAYS, SEARCH_K = 6, 0.7, 5, 14, 50


def words(s):
    return re.findall(r"[a-z0-9']+", s.lower())


def runs(ws, n=RUN):
    return {" ".join(ws[i:i + n]) for i in range(len(ws) - n + 1)}


def same_speaker(item_speaker, speaker):
    if speaker is None:
        return True
    last = item_speaker.split()[-1].lower()
    return last in speaker.lower()


def fuzzy_variants(con, it):
    w0 = words(it["quotation"])
    first0 = con.execute("SELECT first_date FROM quotes WHERE quote_id = ?", [it["quote_id"]]).fetchone()[0]
    rows = con.execute(f"""
      SELECT q.quote_id, q.quotation, q.speaker, q.first_date FROM (
        SELECT quote_id, fts_main_quotes.match_bm25(quote_id, ?) AS s FROM quotes) x JOIN quotes q USING (quote_id)
      WHERE x.s IS NOT NULL ORDER BY round(x.s, 6) DESC, q.quote_id LIMIT {SEARCH_K}""", [it["quotation"]]).fetchall()
    out = []
    for qid, text, speaker, first in rows:
        if qid in it["variant_ids"] or not same_speaker(it["speaker"], speaker):
            continue
        if abs((first - first0).total_seconds()) > WINDOW_DAYS * 86400:
            continue
        w1 = words(text)
        short, long_ = (w0, w1) if len(w0) <= len(w1) else (w1, w0)
        if len(short) < MIN_WORDS:
            continue
        contain = sum(1 for w in set(short) if w in set(long_)) / len(set(short))
        if contain >= MIN_CONTAIN and runs(w0) & runs(w1):
            out.append(qid)
    return out


def recompute(con, it, ids):
    ms = con.execute("""
      SELECT m.date, m.domain, m.url, coalesce(d.kind, 'outlet') FROM quote_mentions m
      LEFT JOIN domains d USING (domain)
      WHERE m.quote_id IN (SELECT unnest(?)) AND m.has_time AND m.date < ? ORDER BY m.date, m.url""",
                     [ids, dt.datetime.fromisoformat(it["as_of"])]).fetchall()
    orig = [r for r in ms if r[3] == "outlet"]
    return ({"date": ms[0][0].isoformat(), "domain": ms[0][1], "url": ms[0][2], "has_time": True},
            {"date": orig[0][0].isoformat(), "domain": orig[0][1], "url": orig[0][2]} if orig else None)


def main(paths):
    con = duckdb.connect("data/news.duckdb", read_only=True)
    con.execute("LOAD fts")
    for path in paths:
        items = [json.loads(l) for l in open(path)]
        if items and items[0].get("gt_version") == 2:
            print(f"{path}: already v2, skipped")
            continue
        shutil.copy(path, path + ".v1")
        widened = moved = 0
        for it in items:
            extra = fuzzy_variants(con, it)
            it["direct_variant_ids"] = list(it["variant_ids"])
            it["fuzzy_variant_ids"] = extra
            it["variant_ids"] = it["direct_variant_ids"] + extra
            old = it["gt_first_any"]["url"]
            it["gt_first_any"], it["gt_first_original"] = recompute(con, it, it["variant_ids"])
            it["gt_version"] = 2
            widened += bool(extra)
            moved += it["gt_first_any"]["url"] != old
        with open(path, "w") as fh:
            fh.writelines(json.dumps(it) + "\n" for it in items)
        print(f"{path}: {len(items)} items, {widened} gained fuzzy variants, first appearance moved for {moved}")


if __name__ == "__main__":
    main(sys.argv[1:])
