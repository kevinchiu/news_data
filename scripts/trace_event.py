"""Trace a news event to the first report of it (GDELT GKG headlines, Jan-Apr 2020).

  python -I scripts/trace_event.py "One sentence reporting the event" --as-of 2020-03-15T12:00 \
      [--config config/event_base.json]

All tunable behaviour comes from the config (scripts/ev_config.py), which the improvement loop edits.
1. BM25 over article headlines seen in the WINDOW_DAYS before --as-of (table gkg_titles, built by
   --build-index); syndicated copies (same normalised headline) count as one candidate. With
   query_template "headline" the statement is first rewritten as a news headline by Haiku (reworded
   statements drift from headline vocabulary); "raw+headline" interleaves both searches' results.
2. Claude Haiku 5.5 picks the headlines that report the same event as the statement.
3. The matched headlines' articles give the timeline: the first report is the earliest of them.
   Times are when GDELT saw each article (15-minute steps), an upper bound on publication.
"""
import argparse
import json
import re
import sys
from itertools import chain, zip_longest
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ev_config  # noqa: E402
from verify import LLMCache, claude_json  # noqa: E402

# fixed, not a loop knob: the benchmark's first reports lie within 72 h of the query's story
WINDOW_DAYS = 7
OUTLET_SUFFIX = re.compile(r"\s+[|–—-]\s+[^|–—-]{1,40}$")
HEADLINE_MODEL, HEADLINE_EFFORT = "claude-haiku-5-5", "low"
HEADLINE_SYSTEM = """Rewrite a sentence from a news story as the headline a news site would most likely \
have given the first report of it. Use the plain words headlines use (e.g. "new cases" not "fresh \
infections"), keep every name, place and number, drop dates and reporting frames ("officials said"). \
One headline, at most 15 words."""
HEADLINE_SCHEMA = json.dumps({"type": "object", "properties": {"headline": {"type": "string"}},
                              "required": ["headline"], "additionalProperties": False})


def title_key(title):
    """Normalised headline: syndicated copies and photo-gallery variants share it."""
    t = OUTLET_SUFFIX.sub("", title or "").lower()
    return " ".join([w for w in re.sub(r"[^a-z0-9 ]+", " ", t).split() if not w.isdigit()][:8])


def build_index(path="data/news.duckdb"):
    con = duckdb.connect(path)
    con.execute("SET threads = 3")
    con.execute("INSTALL fts; LOAD fts")
    con.execute("""
      CREATE OR REPLACE TABLE gkg_titles AS
        SELECT url, any_value(source) AS source, min(date) AS date, any_value(title) AS title
        FROM gdelt_gkg WHERE title IS NOT NULL AND length(title) >= 15 GROUP BY url""")
    con.execute("PRAGMA create_fts_index('gkg_titles', 'url', 'title', stemmer = 'porter', overwrite = 1)")
    print("gkg_titles", con.execute("SELECT count(*) FROM gkg_titles").fetchone()[0])


def headline_query(cache, statement):
    """Haiku's headline-style rewrite of the statement (cached)."""
    def compute():
        out, stats = claude_json(statement, HEADLINE_SYSTEM, HEADLINE_SCHEMA, HEADLINE_MODEL, HEADLINE_EFFORT)
        return {**out, **stats}
    return cache.cached([HEADLINE_MODEL, HEADLINE_EFFORT, HEADLINE_SYSTEM, statement], compute)


def query_text(statement, cfg):
    if cfg["retrieval"]["query_template"] == "content":
        # drop reporting frames ("officials said on Tuesday that ..."), keep names and facts
        return re.sub(r"\b(said|says|reported|according to|on (monday|tuesday|wednesday|thursday|friday|saturday|sunday))\b",
                      " ", statement, flags=re.I)
    return statement


def candidates(con, statement, as_of, cfg, cache=None):
    """Top-k headline groups by BM25, each with its articles (earliest first)."""
    t = cfg["retrieval"]["query_template"]
    if t in ("headline", "raw+headline"):
        h = headline_query(cache or LLMCache(), statement)["headline"]
        if t == "headline":
            return search(con, h, as_of, cfg)
        # interleave the two rankings (raw first), one group per normalised headline, k in all
        out, seen = [], set()
        for g in chain.from_iterable(zip_longest(search(con, statement, as_of, cfg), search(con, h, as_of, cfg))):
            if g and title_key(g["title"]) not in seen and len(out) < cfg["retrieval"]["k"]:
                seen.add(title_key(g["title"]))
                out.append(g)
        return out
    return search(con, query_text(statement, cfg), as_of, cfg)


def search(con, query, as_of, cfg):
    r = cfg["retrieval"]
    as_of = as_of[:19].replace("T", " ")
    rows = con.execute(f"""
      SELECT url, source, strftime(date, '%Y-%m-%dT%H:%M'), title FROM (
        SELECT *, fts_main_gkg_titles.match_bm25(url, ?) AS score FROM gkg_titles
        WHERE date < TIMESTAMP '{as_of}' AND date >= TIMESTAMP '{as_of}' - INTERVAL {WINDOW_DAYS} DAY)
      WHERE score IS NOT NULL ORDER BY round(score, 6) DESC, url LIMIT ?""",
                       [query, r["k"] * r["pool"]]).fetchall()
    groups = {}
    for url, src, d, title in rows:
        key = title_key(title)
        if key not in groups and len(groups) >= r["k"]:
            continue  # keep collecting copies of the k groups already in
        groups.setdefault(key, {"title": title, "articles": []})["articles"].append(
            {"url": url, "source": src, "date": d, "title": title})
    out = []
    for g in groups.values():
        g["articles"].sort(key=lambda a: (a["date"], a["url"]))
        out.append({"title": g["title"], "date": g["articles"][0]["date"], "source": g["articles"][0]["source"],
                    "n_copies": len(g["articles"]) - 1, "articles": g["articles"]})
    return out


def verify(cache, statement, cands, cfg, nonce="", as_of=None):
    v = cfg["verify"]
    claim_date = as_of if v.get("show_claim_date") else None
    shown = [{"text": (f"{c['source']}: " if v["show_source"] else "") + c["title"]
                      + (f" (+{c['n_copies']} copies)" if v["show_copies"] and c["n_copies"] else ""),
              "date": c["date"] if v["show_date"] else None} for c in cands]
    if v["order"] == "time":
        order = sorted(range(len(cands)), key=lambda i: (cands[i]["date"], i))
        cands, shown = [cands[i] for i in order], [shown[i] for i in order]
    matched, calls = [], {"made": 0, "cached": 0}
    for i in range(0, len(shown), v["batch"]):
        r = cache.verify(statement, shown[i:i + v["batch"]], None, v["model"], v["effort"], claim_date, v["prompt"], nonce)
        calls["cached" if r["cached"] else "made"] += 1
        matched += [cands[i + j] for j in r["matches"] if 0 <= j < len(shown[i:i + v["batch"]])]
    return matched, calls


def recheck(cache, statement, cands, matched, cfg, nonce=""):
    """Second pass anchored on the matched headlines: the verifier's errors are mostly earlier
    occurrences of a similar event accepted, or the true first report rejected. Take reference
    headlines from the matches (most copies, or most recent), then re-ask, headline against those
    headlines, about every candidate seen before the earliest reference."""
    rc, v = cfg["recheck"], cfg["verify"]
    if rc["ref_by"] == "copies":
        ref = sorted(matched, key=lambda g: (-len(g["articles"]), g["date"]))[:rc["ref"]]
    else:  # "recent": the matches nearest the statement's own story
        ref = sorted(matched, key=lambda g: g["date"], reverse=True)[:rc["ref"]]
    anchor = min(g["date"] for g in ref)
    ids, was = {id(g) for g in ref}, {id(g) for g in matched}
    todo = [c for c in cands if c["date"] < anchor and id(c) not in ids]
    keep = [g for g in matched if g["date"] >= anchor]
    claim = statement + "\n\nREFERENCE HEADLINES (these report the claim's event):\n" + "\n".join(
        f"- {g['title']} ({g['date']})" for g in sorted(ref, key=lambda g: g["date"]))
    calls = {"made": 0, "cached": 0}
    for i in range(0, len(todo), v["batch"]):
        b = todo[i:i + v["batch"]]
        r = cache.verify(claim, [{"text": c["title"], "date": c["date"]} for c in b], None,
                         rc["model"], rc["effort"], None, rc["prompt"], nonce)
        calls["cached" if r["cached"] else "made"] += 1
        # "union": the second pass can also accept candidates the first rejected; "intersect": it only drops
        keep += [b[j] for j in r["matches"] if 0 <= j < len(b) and (rc["mode"] == "union" or id(b[j]) in was)]
    return keep, calls


def trace(con, cache, statement, as_of="2020-05-01", cfg=None, nonce=""):
    cfg = cfg or ev_config.load()
    cands = candidates(con, statement, as_of, cfg, cache)
    matched, calls = verify(cache, statement, cands, cfg, nonce, as_of) if cands else ([], {"made": 0, "cached": 0})
    if matched and cfg.get("recheck", {}).get("enabled"):
        matched, c2 = recheck(cache, statement, cands, matched, cfg, nonce)
        calls = {k: calls[k] + c2[k] for k in calls}
    timeline = sorted((a for g in matched for a in g["articles"]), key=lambda a: (a["date"], a["url"]))
    first_by_source = {}
    for a in timeline:
        first_by_source.setdefault(a["source"], a)
    return {"statement": statement, "as_of": as_of, "candidates": cands, "matched": matched,
            "first": timeline[0] if timeline else None, "spread": list(first_by_source.values()),
            "n_articles": len(timeline), "llm_calls": calls}


def show(res):
    print(f"STATEMENT: {res['statement']}  (as of {res['as_of']})")
    if not res["matched"]:
        print("\nNo headline before then reports this event.")
    f = res["first"]
    if f:
        print(f"\nFIRST REPORT: {f['date']}  {f['source']}\n  {f['title']}\n  {f['url']}")
        print(f"\nSPREAD ({res['n_articles']} articles, {len(res['spread'])} outlets); first 12:")
        for a in res["spread"][:12]:
            print(f"  {a['date']}  {a['source']:<30} {a['title'][:70]}")
    c = res["llm_calls"]
    print(f"\n({c['made']} Haiku calls, {c['cached']} from cache)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("statement", nargs="?")
    ap.add_argument("--as-of", default="2020-05-01")
    ap.add_argument("--config", default=str(ev_config.CONFIG_DIR / "event_base.json"))
    ap.add_argument("--build-index", action="store_true", help="(re)build gkg_titles and its BM25 index")
    ap.add_argument("--json")
    a = ap.parse_args()
    if a.build_index:
        build_index()
        return
    con = duckdb.connect("data/news.duckdb", read_only=True)
    con.execute("LOAD fts")
    res = trace(con, LLMCache(), a.statement, a.as_of, ev_config.load(a.config))
    show(res)
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
