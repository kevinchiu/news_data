"""Trace a statement to the first outlet that carried the quotation it reports (Quotebank).

  python -I scripts/trace_quote.py "Statement, e.g. a paraphrase of what someone said" \
      [--as-of 2020-03-15T12:00] [--config config/base.json]

All tunable behaviour comes from the config (scripts/qb_config.py), which the improvement loop edits.
1. Full-text search (BM25) over unique quotations first seen before --as-of.
2. Claude Haiku 5.5 picks the quotations that state the claim.
3. Matched quotations are expanded to their variants (none / direct links / whole cluster).
4. Every mention of those quotations before --as-of gives the timeline: first appearance
   anywhere, first original outlet (not wire/press release/aggregator/feed), and spread by domain.
   With --with-wire, wire/press-release articles found via GDELT (wire_quote_matches; partial,
   paused) are merged in; their time is when GDELT saw them, an upper bound on publication.
"""
import argparse
import json
import sys
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qb_config  # noqa: E402
from verify import LLMCache  # noqa: E402

FTS = {"quotation": ("quotes", "fts_main_quotes"), "quotation_speaker": ("quotes_qs", "fts_main_quotes_qs")}


def query_text(statement, cfg):
    speaker, content = qb_config.split_statement(statement)
    mode = cfg["retrieval"]["query_template"]
    if mode == "content":
        return content
    if mode == "content_plus_speaker":
        return f"{content} {speaker or ''}".strip()
    return statement


def candidates(con, statement, as_of, cfg):
    r = cfg["retrieval"]
    table, fts = FTS[r["index"]]
    window = f"AND q.first_date >= TIMESTAMP '{as_of[:19].replace('T', ' ')}' - INTERVAL {int(r['window_days'])} DAY" \
        if r.get("window_days") and not as_of.startswith("9999") else ""
    name = qb_config.surname(qb_config.split_statement(statement)[0]) if r["speaker_filter"] == "surname" else None
    # unknown speakers stay in: Quotebank leaves many quotes unattributed
    speaker_sql = "AND (q.speaker IS NULL OR lower(q.speaker) LIKE '%' || lower(?) || '%')" if name else ""
    params = [query_text(statement, cfg), as_of, r.get("min_urls", 1)] + ([name] if name else []) + [r["k"]]
    # BM25 scores wobble in the last bits between runs (parallel summation), so they are rounded and
    # ties (e.g. identical quote variants) broken by quote_id: the verifier prompt, and its
    # cache key, are the same on every run
    return [dict(zip(("quote_id", "text", "speaker", "date", "num_urls"), row)) for row in con.execute(f"""
        SELECT q.quote_id, q.quotation, coalesce(q.speaker, 'None'), strftime(q.first_date, '%Y-%m-%dT%H:%M'), q.num_urls
        FROM (SELECT quote_id, {fts}.match_bm25(quote_id, ?) AS score FROM {table}) s JOIN quotes q USING (quote_id)
        WHERE s.score IS NOT NULL AND q.first_date < ? AND q.num_urls >= ? {window} {speaker_sql}
        ORDER BY round(s.score, 6) DESC, q.quote_id LIMIT ?""", params).fetchall()]


def verify(cache, statement, cands, cfg, nonce=""):
    v = cfg["verify"]
    name = qb_config.surname(qb_config.split_statement(statement)[0]) if v.get("speaker_hint") else None

    def hint(c):
        if not v.get("speaker_hint"):
            return ""
        if c["speaker"] == "None":
            return " (speaker unknown)"
        if name:
            return " (speaker matches claim)" if name.lower() in c["speaker"].lower() else " (different speaker?)"
        return ""
    shown = [{"text": f'{c["speaker"]}: "{c["text"]}"' + hint(c)
                      + (f' [carried by {c["num_urls"]} articles]' if v.get("show_num_urls") else ""),
              "date": c["date"] if v["show_date"] else None} for c in cands]
    matched, calls = [], {"made": 0, "cached": 0}
    for i in range(0, len(shown), v["batch"]):
        r = cache.verify(statement, shown[i:i + v["batch"]], None, v["model"], v["effort"], None, v["prompt"], nonce)
        calls["cached" if r["cached"] else "made"] += 1
        matched += [cands[i + j] for j in r["matches"]]
    return matched, calls


def timeline(con, quote_ids, as_of, cfg, include_wire=False):
    expand = {
        "none": "SELECT unnest(?) AS quote_id WHERE ? IS NOT NULL AND ? IS NOT NULL",
        "direct": """SELECT unnest(?) AS quote_id
          UNION SELECT b FROM quote_links WHERE a IN (SELECT unnest(?))
          UNION SELECT a FROM quote_links WHERE b IN (SELECT unnest(?))""",
        "cluster": """SELECT quote_id FROM quote_clusters WHERE cluster_id IN (
          SELECT cluster_id FROM quote_clusters WHERE quote_id IN (SELECT unnest(?)))
          AND ? IS NOT NULL AND ? IS NOT NULL""",
    }[cfg["timeline"]["variants"]]
    rows = con.execute(f"""
        WITH ids AS ({expand})
        SELECT * FROM (
          SELECT m.date, m.has_time, m.domain, d.kind, d.grp, m.url, m.title, m.quote_id, m.speaker,
                 'quotebank' AS via
          FROM quote_mentions m JOIN ids USING (quote_id) LEFT JOIN domains d USING (domain)
          UNION ALL
          SELECT DISTINCT a.date, true, a.source,
                 CASE WHEN a.source IN ('prnewswire.com', 'businesswire.com', 'globenewswire.com')
                      THEN 'press_release' ELSE 'wire' END, a.source, a.url, a.title, w.quote_id, NULL, 'gdelt'
          FROM wire_quote_matches w JOIN ids USING (quote_id) JOIN gdelt_wire_articles a USING (gkg_id))
        WHERE date < ? AND (? OR via = 'quotebank') ORDER BY date, domain""",
        [quote_ids] * 3 + [as_of, include_wire]).fetchall()
    keys = ("date", "has_time", "domain", "kind", "grp", "url", "title", "quote_id", "speaker", "via")
    out = []
    for r in rows:
        t = dict(zip(keys, r), date=r[0].isoformat())
        t["kind"], t["grp"] = qb_config.domain_kind(cfg, t["domain"], t["kind"], t["grp"])
        out.append(t)
    return out


def trace(con, cache, statement, as_of="9999-12-31", cfg=None, include_wire=False, nonce=""):
    cfg = cfg or qb_config.load()
    cands = candidates(con, statement, as_of, cfg)
    matched, calls = verify(cache, statement, cands, cfg, nonce) if cands else ([], {"made": 0, "cached": 0})
    tl = timeline(con, [m["quote_id"] for m in matched], as_of, cfg, include_wire) if matched else []
    # rank "first" among timed mentions (date-only stamps sort artificially early); keep all in spread
    timed = [t for t in tl if t["has_time"]] or tl
    original = [t for t in timed if t["kind"] not in cfg["timeline"]["original_excludes"]]
    first_by_domain = {}
    for t in tl:
        first_by_domain.setdefault(t["domain"], t)
    return {"statement": statement, "as_of": as_of, "candidates": cands, "matched_quotes": matched,
            "first_any": timed[0] if timed else None, "first_original": original[0] if original else None,
            "spread": list(first_by_domain.values()), "n_mentions": len(tl), "llm_calls": calls}


def show(res):
    print(f"STATEMENT: {res['statement']}")
    if not res["matched_quotes"]:
        print("\nNo quotation in the corpus states this.")
    for q in res["matched_quotes"]:
        print(f"\nMATCHED QUOTE ({q['speaker']}, first seen {q['date']}): \"{q['text'][:200]}\"")
    for label in ("first_any", "first_original"):
        t = res[label]
        if t:
            hour = "" if t["has_time"] else "  (date only)"
            print(f"\n{label.upper().replace('_', ' ')}: {t['date'][:16]}{hour}  {t['domain']} [{t['kind']}]")
            print(f"  {t['title'][:120] if t['title'] else ''}\n  {t['url']}")
    if res["spread"]:
        print(f"\nSPREAD ({res['n_mentions']} mentions, {len(res['spread'])} domains); first 12:")
        for t in res["spread"][:12]:
            via = "  (seen by GDELT; publish time <= this)" if t["via"] == "gdelt" else ""
            print(f"  {t['date'][:16]}  {t['domain']:<32} {t['kind']}{via}")
    c = res["llm_calls"]
    print(f"\n({c['made']} Haiku calls, {c['cached']} from cache)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("statement")
    ap.add_argument("--as-of", default="9999-12-31")
    ap.add_argument("--config", default=str(qb_config.CONFIG_DIR / "base.json"))
    ap.add_argument("--with-wire", action="store_true", help="merge GDELT wire rows (partial data)")
    ap.add_argument("--json")
    a = ap.parse_args()
    con = duckdb.connect("data/news.duckdb", read_only=True)
    con.execute("LOAD fts")
    res = trace(con, LLMCache(), a.statement, a.as_of, qb_config.load(a.config), include_wire=a.with_wire)
    show(res)
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
