"""Build the event-provenance benchmark ("who reported this event first?") from GDELT 2.0.

Ground truth is structural (GDELT mention timing + title/URL near-duplicates), not model labels.
Design revised after a Fable review of the first draft (2026-10-18):
  - a GDELT event ID's own earliest mention is just the article GDELT coded it from, and is
    usually a passing mention, so the answer is built from *anchor* mentions (lead sentences:
    SentenceID <= 2, Confidence >= 50) and expanded to syndicated copies by title/URL slug;
  - multi-source events are mostly one wire story on member sites, so the whole first wave
    (every story article up to GT_t0 + 30 min) counts as correct.

Steps:
  1. Candidate events: >= 5 mention sources, mentions spanning >= 6 h, an actor with a country
     or known-group code, EventRootCode not 01/02 (statements/appeals), >= 3 anchor mentions
     from >= 3 sources.
  2. Seeds in hash order; each seed is star-merged with up to 9 other event IDs whose anchor
     URLs overlap it (|A∩B| / min(|A|,|B|) >= 0.5), same ActionGeo country, Day ± 1.
  3. Story = the cluster's anchor articles + GKG articles seen from 72 h before the earliest
     anchor to 30 min after it whose normalised title shares 4-gram Jaccard >= 0.5 with an anchor
     title, or whose URL slug (first 5 alphabetic tokens) equals an anchor's.
  4. GT_t0 = earliest story article; first wave = story articles up to GT_t0 + 30 min; also the
     earliest 15-minute slot (secondary, exact).
  5. Coherence: GDELT event IDs bundle unrelated articles, so anchors are linked when their titles
     share a content-word 3-gram, two 2-grams or three content words, and only the largest linked group (>= 3 sources)
     and copies of it are kept; title matches in step 3 need titles of >= 6 words. Titles/slugs a source reuses on >= 3
     days (section pages) are ignored.
  6. Query = a coherent anchor article >= 6 h after GT_t0 that is not a copy of a first-wave article
     (title 4-gram Jaccard < 0.3 with every first-wave title); Haiku rewrites its title as one reported sentence;
     as_of = when GDELT saw it.
  6b. Linked reports (GT v2): the Opus audit of v1 found an earlier report of the same event
     outside the GT in 23 of 41 correct items, mostly outlets GDELT coded elsewhere. So GKG
     articles from 24 h before GT_t0 whose titles share a content 3-gram or two 2-grams with
     core anchors from >= 2 sources join the story, and GT_t0 / first wave / query are
     re-selected. (Three shared words, or a 72 h window, drifted to earlier stages of the story.)
  7. Seeds whose story shares an article or event ID with an accepted item are skipped, so
     splits don't share stories.

All thresholds above are frozen: the improvement loop must not change them.

Usage: python -I scripts/make_event_bench.py [bench/ev_bench.jsonl bench/ev_watch.jsonl] [n_train n_val n_test n_watch]
"""
import datetime as dt
import hashlib
import html
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify import LLMCache, claude_json  # noqa: E402

N_SEEDS = 8000
STOP = ("a an the of to in on at for and or but is are was were be been as by with from that this it its "
        "after over into about says said say amid new his her their he she they we you not no will has have "
        "had than up out more who what why how when vs").split()

REWRITE_SYSTEM = """Rewrite a news headline as one sentence that a journalist would write to \
report the event. Keep the names, places and figures. Do not copy more than three consecutive \
words from the headline and do not use quotation marks."""
REWRITE_SCHEMA = json.dumps({"type": "object", "properties": {"sentence": {"type": "string"}},
                             "required": ["sentence"], "additionalProperties": False})

# lowercased title without a trailing " - Outlet" / " | Outlet" segment, as a word list
NORM = r"""list_filter(string_split(regexp_replace(lower(regexp_replace({t}, '\s+[|–—-]\s+[^|–—-]{{1,40}}$', '')),
                                                  '[^a-z0-9 ]+', ' ', 'g'), ' '), x -> x <> '')"""
# first 5 alphabetic tokens of the URL's longest path segment (NULL if fewer)
SLUG = r"""(SELECT CASE WHEN len(s) >= 5 THEN array_to_string(s[1:5], ' ') END FROM (SELECT list_filter(
             string_split(regexp_replace(lower(list_sort(list_transform(
               string_split(regexp_replace(regexp_replace({u}, '[?#].*$', ''), '^[a-z]+://[^/]+', ''), '/'),
               x -> lpad(length(x)::VARCHAR, 5, '0') || x), 'DESC')[1][6:]),
             '[^a-z]+', ' ', 'g'), ' '), x -> length(x) > 1) AS s))"""


def rewrite(cache, title):
    def compute():
        out, stats = claude_json(f"HEADLINE: {title}", REWRITE_SYSTEM, REWRITE_SCHEMA, "claude-haiku-5-5", "low")
        return {"sentence": out["sentence"], **stats}
    return cache.cached(["claude-haiku-5-5", "low", REWRITE_SYSTEM, title], compute)["sentence"]


def build_clusters(con):
    con.execute(f"""
    CREATE TEMP TABLE anchors AS
      SELECT GlobalEventID AS eid, MentionIdentifier AS url, MentionSourceName AS src, min(MentionTimeDate) AS t
      FROM gdelt_mentions WHERE SentenceID <= 2 AND Confidence >= 50 GROUP BY ALL;
    CREATE TEMP TABLE ast AS SELECT eid, count(*) AS na, count(DISTINCT src) AS nas FROM anchors GROUP BY 1;
    CREATE TEMP TABLE est AS
      SELECT GlobalEventID AS eid, count(DISTINCT MentionSourceName) AS ns,
             max(MentionTimeDate) - min(MentionTimeDate) AS span
      FROM gdelt_mentions WHERE GlobalEventID IN (SELECT eid FROM ast WHERE nas >= 3) GROUP BY 1;
    CREATE TEMP TABLE ev AS
      SELECT DISTINCT ON (GlobalEventID) GlobalEventID AS eid, Day, ActionGeo_CountryCode AS cc,
             Actor1Name, EventCode, Actor2Name, ActionGeo_FullName
      FROM gdelt_events WHERE GlobalEventID IN (SELECT eid FROM ast);
    CREATE TEMP TABLE seeds AS
      SELECT ev.eid, ast.na FROM ev JOIN ast USING (eid) JOIN est USING (eid) JOIN gdelt_events g ON g.GlobalEventID = ev.eid
      WHERE est.ns >= 5 AND est.span >= INTERVAL 6 HOUR AND ast.nas >= 3
        AND g.EventRootCode NOT IN ('01', '02') AND g.Day BETWEEN '20200101' AND '20200430'
        AND (coalesce(g.Actor1CountryCode, '') <> '' OR coalesce(g.Actor1KnownGroupCode, '') <> ''
             OR coalesce(g.Actor2CountryCode, '') <> '' OR coalesce(g.Actor2KnownGroupCode, '') <> '')
      QUALIFY row_number() OVER (PARTITION BY ev.eid) = 1
      ORDER BY hash(ev.eid) LIMIT {N_SEEDS};
    CREATE TEMP TABLE members AS
      WITH pairs AS (
        SELECT s.eid AS seed, a.eid AS other, count(DISTINCT a.url) AS shared
        FROM anchors sa JOIN seeds s ON s.eid = sa.eid JOIN anchors a ON a.url = sa.url AND a.eid <> sa.eid GROUP BY 1, 2)
      SELECT seed, other AS eid FROM (
        SELECT p.*, row_number() OVER (PARTITION BY seed ORDER BY shared DESC, other) AS r
        FROM pairs p JOIN ast sa ON sa.eid = p.seed JOIN ast oa ON oa.eid = p.other
        JOIN ev se ON se.eid = p.seed JOIN ev oe ON oe.eid = p.other
        WHERE p.shared >= 0.5 * least(sa.na, oa.na) AND coalesce(se.cc, '') = coalesce(oe.cc, '')
          AND abs(datediff('day', strptime(se.Day, '%Y%m%d'), strptime(oe.Day, '%Y%m%d'))) <= 1)
      WHERE r <= 9
      UNION ALL SELECT eid, eid FROM seeds;
    -- anchor articles per cluster, with GKG titles
    CREATE TEMP TABLE cl_anchor AS
      SELECT m.seed, a.url, any_value(a.src) AS src, min(a.t) AS t, any_value(g.title) AS title
      FROM members m JOIN anchors a USING (eid) LEFT JOIN gdelt_gkg g ON g.url = a.url GROUP BY 1, 2;
    """)
    print("seeds", con.execute("SELECT count(*) FROM seeds").fetchone()[0],
          "| cluster anchors", con.execute("SELECT count(*) FROM cl_anchor").fetchone()[0], flush=True)

    # story expansion: title 4-gram Jaccard >= 0.5 or equal URL slug, within the window
    con.execute(f"""
    CREATE TEMP TABLE win AS SELECT seed, min(t) - INTERVAL 72 HOUR AS lo, min(t) + INTERVAL 30 MINUTE AS hi
      FROM cl_anchor GROUP BY 1;
    CREATE TEMP TABLE a_sh AS
      SELECT DISTINCT seed, url, hash(array_to_string(w[i:i + 3], ' ')) AS h, len(w) - 3 AS n
      FROM (SELECT seed, url, {NORM.format(t='title')} AS w FROM cl_anchor WHERE title IS NOT NULL),
           unnest(generate_series(1, greatest(len(w) - 3, 0))) AS t(i);
    CREATE TEMP TABLE g_win AS
      SELECT g.url, g.source, g.date, g.title, {SLUG.format(u='g.url')} AS slug,
             array_to_string({NORM.format(t='g.title')}, ' ') AS ntitle FROM gdelt_gkg g
      WHERE g.date BETWEEN (SELECT min(lo) FROM win) AND (SELECT max(hi) FROM win);
    CREATE TEMP TABLE g_sh AS
      SELECT url, h, n FROM (
        SELECT DISTINCT url, hash(array_to_string(w[i:i + 3], ' ')) AS h, len(w) - 3 AS n
        FROM (SELECT url, {NORM.format(t='title')} AS w FROM g_win WHERE title IS NOT NULL),
             unnest(generate_series(1, greatest(len(w) - 3, 0))) AS t(i))
      WHERE h IN (SELECT h FROM a_sh);
    CREATE TEMP TABLE story AS
      WITH title_hits AS (
        SELECT a.seed, a.url AS via, g.url FROM a_sh a JOIN g_sh g USING (h)
        WHERE a.n >= 3 AND g.n >= 3
        GROUP BY a.seed, a.url, g.url, a.n, g.n HAVING count(*) >= 0.5 * (a.n + g.n - count(*))),
      a_slug AS (SELECT DISTINCT seed, url, {SLUG.format(u='url')} AS slug FROM cl_anchor),
      slug_hits AS (
        SELECT s.seed, s.url AS via, g.url FROM a_slug s JOIN g_win g ON g.slug = s.slug WHERE s.slug IS NOT NULL),
      hits AS (
        SELECT seed, url, list(DISTINCT via) AS vias FROM (SELECT * FROM title_hits UNION SELECT * FROM slug_hits) GROUP BY 1, 2)
      SELECT h.seed, h.url, g.source AS src, g.date AS t, g.title, false AS anchor, h.vias, g.slug, g.ntitle
      FROM hits h JOIN g_win g USING (url) JOIN win w USING (seed)
      WHERE g.date BETWEEN w.lo AND w.hi AND h.url NOT IN (SELECT url FROM cl_anchor c WHERE c.seed = h.seed)
      UNION ALL
      SELECT c.seed, c.url, c.src, c.t, c.title, true, []::VARCHAR[], g.slug, g.ntitle
      FROM cl_anchor c LEFT JOIN g_win g USING (url);
    -- a title or slug a source reuses on >= 3 different days is a section/landing page, not a story
    CREATE TEMP TABLE generic AS
      SELECT 'title' AS kind, source, ntitle AS key FROM g_win
      WHERE (source, ntitle) IN (SELECT src, ntitle FROM story) GROUP BY ALL HAVING count(DISTINCT date::DATE) >= 3
      UNION ALL
      SELECT 'slug', source, slug FROM g_win
      WHERE (source, slug) IN (SELECT src, slug FROM story) GROUP BY ALL HAVING count(DISTINCT date::DATE) >= 3;
    """)
    print("story articles", con.execute("SELECT count(*), count(*) FILTER (WHERE NOT anchor) FROM story").fetchone(),
          flush=True)


def grams(title, n):
    t = re.sub(r"\s+[|–—-]\s+[^|–—-]{1,40}$", "", title or "")
    w = [x for x in re.sub(r"[^a-z0-9 ]+", " ", t.lower()).split() if x]
    return w, {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def jaccard(a, b):
    return len(a & b) / len(a | b) if a and b else 0.0


def content_words(title):
    return [x for x in grams(title, 1)[0] if x not in STOP and len(x) > 2]


def content_grams(title):
    w = content_words(title)
    return (set(w), {" ".join(w[i:i + 2]) for i in range(len(w) - 1)}, {" ".join(w[i:i + 3]) for i in range(len(w) - 2)})


def linked(g, h, words=True):
    """Two titles are about the same story if they share a content-word 3-gram or two 2-grams, or
    (words=True, used only within one GDELT event's anchors) three content words."""
    return bool(g[2] & h[2]) or len(g[1] & h[1]) >= 2 or (words and len(g[0] & h[0]) >= 3)


def core(anchors):
    """The largest group of anchors linked by shared title content: GDELT event IDs bundle
    unrelated articles, and this keeps one story."""
    g = [content_grams(a["title"]) for a in anchors]
    parent = list(range(len(anchors)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(len(anchors)):
        for j in range(i + 1, len(anchors)):
            if linked(g[i], g[j]):
                parent[find(i)] = find(j)
    groups = {}
    for i, a in enumerate(anchors):
        groups.setdefault(find(i), []).append(a)
    return max(groups.values(), key=lambda grp: (len({a["source"] for a in grp}), len(grp)), default=[])


def linked_articles(con, cands):
    """Earlier articles of the same story that GDELT did not code under these event IDs (often
    independent outlets with different headlines): GKG articles from 24 h before GT_t0 to 30 min
    after whose titles share a content 3-gram or two 2-grams with core anchors from >= 2 sources.
    Returns {seed: [article, ...]}."""
    con.execute("CREATE OR REPLACE TEMP TABLE p1_core (seed BIGINT, url VARCHAR, src VARCHAR, title VARCHAR)")
    con.executemany("INSERT INTO p1_core VALUES (?, ?, ?, ?)",
                    [(seed, a["url"], a["source"], a["title"]) for seed, c in cands.items() for a in c["core"]])
    con.execute("CREATE OR REPLACE TEMP TABLE p1_win (seed BIGINT, lo TIMESTAMP, hi TIMESTAMP)")
    con.executemany("INSERT INTO p1_win VALUES (?, ?, ?)",
                    [(seed, c["t0"] - dt.timedelta(hours=24), c["t0"] + dt.timedelta(minutes=30)) for seed, c in cands.items()])
    stop = "[" + ", ".join(f"'{w}'" for w in STOP) + "]"

    def cw(t):
        return f"list_distinct(list_filter({NORM.format(t=t)}, x -> length(x) > 2 AND NOT list_contains({stop}, x)))"
    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE aw AS SELECT DISTINCT seed, url AS via, w FROM p1_core, unnest({cw('title')}) AS t(w);
    CREATE OR REPLACE TEMP TABLE gw AS
      SELECT g.url, g.date, w FROM gdelt_gkg g, unnest({cw('g.title')}) AS t(w)
      WHERE g.title IS NOT NULL AND g.date BETWEEN (SELECT min(lo) FROM p1_win) AND (SELECT max(hi) FROM p1_win)
        AND w IN (SELECT w FROM aw);
    CREATE OR REPLACE TEMP TABLE df AS SELECT w, count(*) AS n FROM gw GROUP BY 1;
    CREATE OR REPLACE TEMP TABLE lk AS
      SELECT a.seed, gw.url FROM aw a JOIN df USING (w) JOIN gw USING (w) JOIN p1_win p ON p.seed = a.seed
      WHERE df.n <= 5000 AND gw.date BETWEEN p.lo AND p.hi
      GROUP BY a.seed, a.via, gw.url HAVING count(*) >= 2;
    """)
    out = {}
    for seed, url, src, d, title in con.execute("""
        SELECT DISTINCT l.seed, g.url, g.source, g.date, g.title FROM (SELECT DISTINCT seed, url FROM lk) l
        JOIN gdelt_gkg g ON g.url = l.url ORDER BY 1, 4""").fetchall():
        c = cands[seed]
        g = content_grams(title)
        srcs = {a["source"] for a, ag in zip(c["core"], c["core_grams"]) if linked(g, ag, words=False)}
        if len(srcs - {src}) >= 2:
            out.setdefault(seed, []).append({"url": url, "source": src, "date": d, "title": html.unescape(title or ""),
                                             "anchor": False, "vias": [], "linked": True})
    return out


def main(dst="bench/ev_bench.jsonl", watch_dst="bench/ev_watch.jsonl", n_train=150, n_val=75, n_test=150,
         n_watch=100):
    con = duckdb.connect("data/news.duckdb", read_only=True)
    con.execute("SET threads = 3")
    build_clusters(con)
    order = [r[0] for r in con.execute("SELECT eid FROM seeds ORDER BY hash(eid)").fetchall()]
    generic = {(k, src, key) for k, src, key in con.execute("SELECT * FROM generic").fetchall()}
    story = {}
    for seed, url, src, t, title, anchor, vias, slug, ntitle in con.execute(
            "SELECT * FROM story ORDER BY seed, t, url").fetchall():
        if ("title", src, ntitle) in generic or (not anchor and ("slug", src, slug) in generic):
            continue
        story.setdefault(seed, []).append({"url": url, "source": src, "date": t, "anchor": anchor, "vias": vias,
                                           "title": html.unescape(title) if title else title})
    eids = {}
    for seed, eid in con.execute("SELECT seed, eid FROM members").fetchall():
        eids.setdefault(seed, []).append(eid)
    meta = {r[0]: r[1:] for r in con.execute(
        "SELECT eid, Actor1Name, EventCode, Actor2Name, ActionGeo_FullName FROM ev WHERE eid IN (SELECT eid FROM seeds)").fetchall()}

    n_total = n_train + n_val + n_test + n_watch
    reasons = {}

    def drop(why):
        reasons[why] = reasons.get(why, 0) + 1

    def select(arts, anchors):
        """GT_t0, first wave and the query for a story, or None if there is no usable query."""
        arts = sorted(arts, key=lambda a: (a["date"], a["url"]))
        t0 = arts[0]["date"]
        wave = [a for a in arts if (a["date"] - t0).total_seconds() <= 1800]
        wave_grams = [grams(a["title"], 4)[1] for a in wave if a["title"]]
        queries = [a for a in anchors if (a["date"] - t0).total_seconds() >= 6 * 3600 and len(grams(a["title"], 1)[0]) >= 6
                   and all(jaccard(grams(a["title"], 4)[1], g) < 0.3 for g in wave_grams)]
        if not queries:
            return None
        return arts, t0, wave, min(queries, key=lambda a: hashlib.md5(a["url"].encode()).hexdigest())

    # pass 1: coherent stories from GDELT anchors and their syndicated copies
    cands = {}
    for seed in order:
        if len(cands) >= 3 * n_total:
            break
        arts = story.get(seed, [])
        anchors = core([a for a in arts if a["anchor"] and a["title"]])
        if len({a["source"] for a in anchors}) < 3:
            drop("< 3 sources with coherent anchor titles"); continue
        core_urls = {a["url"] for a in anchors}
        arts = [a for a in arts if a["url"] in core_urls or set(a["vias"]) & core_urls]
        sel = select(arts, anchors)
        if sel is None:
            drop("no non-copy lead-sentence query >= 6 h later"); continue
        cands[seed] = {"arts": arts, "core": anchors, "core_grams": [content_grams(a["title"]) for a in anchors],
                       "t0": sel[1]}
    print(f"pass 1: {len(cands)} candidate stories", flush=True)

    # pass 2: add earlier same-story articles GDELT coded elsewhere, then re-select
    extra = linked_articles(con, cands)
    used_urls, used_eids, picked = set(), set(), []
    for seed, c in cands.items():
        if len(picked) >= n_total:
            break
        have = {a["url"] for a in c["arts"]}
        arts = c["arts"] + [a for a in extra.get(seed, []) if a["url"] not in have]
        if used_eids & set(eids[seed]) or used_urls & {a["url"] for a in arts}:
            drop("overlaps accepted item"); continue
        sel = select(arts, c["core"])
        if sel is None:
            drop("no query >= 6 h after the linked first report"); continue
        arts, t0, wave, q = sel
        used_urls |= {a["url"] for a in arts}
        used_eids |= set(eids[seed])
        a1, code, a2, geo = meta[seed]
        picked.append({
            "event_id": seed, "event_ids": sorted(eids[seed]),
            "gdelt_coding": {"actor1": a1, "event_code": code, "actor2": a2, "geo": geo},
            "n_story_sources": len({a["source"] for a in arts}), "n_anchor_sources": len({a["source"] for a in c["core"]}),
            "as_of": q["date"].isoformat(), "query_url": q["url"], "query_source": q["source"], "query_title": q["title"],
            "gt_t0": t0.isoformat(),
            "gt_first_wave": [{"url": a["url"], "source": a["source"], "date": a["date"].isoformat(), "title": a["title"],
                               "anchor": a["anchor"], "linked": a.get("linked", False)} for a in wave],
            "gt_first_slot": [a["url"] for a in arts if a["date"] == t0],
            "gt_version": 2,
        })
    print(f"{len(picked)} items; dropped: {reasons}", flush=True)

    cache = LLMCache()
    with ThreadPoolExecutor(6) as ex:
        sentences = list(ex.map(lambda it: rewrite(cache, it["query_title"]), picked))
    for i, (it, s) in enumerate(zip(picked, sentences)):
        it["query"] = s
        it["split"] = ("train" if i < n_train else "val" if i < n_train + n_val
                       else "test" if i < n_train + n_val + n_test else "watch")
    for path, rows in ((dst, [it for it in picked if it["split"] != "watch"]),
                       (watch_dst, [it for it in picked if it["split"] == "watch"])):
        with open(path, "w") as fh:
            fh.writelines(json.dumps(it) + "\n" for it in rows)
    print({s: sum(it["split"] == s for it in picked) for s in ("train", "val", "test", "watch")}, f"-> {dst}, {watch_dst}")


if __name__ == "__main__":
    a = sys.argv[1:]
    paths = [x for x in a if not x.isdigit()]
    main(*paths, *(int(x) for x in a if x.isdigit()))
