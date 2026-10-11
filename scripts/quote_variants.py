"""Group Quotebank quote variants (trimmed / lightly edited versions of one quote) into clusters.

A light stand-in for MemeTracker's phrase graph: two quotes are linked when they share word
8-grams covering at least half of the shorter quote; 8-grams shared by more than 50 quotes
(stock phrases) are ignored. Connected components become clusters, identified by their
earliest quote. Writes tables quote_clusters(quote_id, cluster_id) and quote_links(a, b) into
data/news.duckdb.

Usage: python -I scripts/quote_variants.py
"""
import time

import duckdb
import numpy as np

N, MAX_POSTING, MIN_SHARE = 8, 50, 0.5

con = duckdb.connect("data/news.duckdb")
t0 = time.time()
con.execute(f"""
CREATE OR REPLACE TEMP TABLE toks AS
  SELECT quote_id, first_date,
         string_split(trim(regexp_replace(lower(quotation), '[^a-z0-9 ]+', ' ', 'g')), ' ') AS w
  FROM quotes;
CREATE OR REPLACE TEMP TABLE shingles AS
  SELECT quote_id, hash(array_to_string(list_slice(list_filter(w, x -> x <> ''), i, i + {N - 1}), ' ')) AS h
  FROM (SELECT quote_id, w, generate_series(1, greatest(len(list_filter(w, x -> x <> '')) - {N - 1}, 0)) AS i_list
        FROM toks), unnest(i_list) AS t(i);
""")
print("shingles", con.execute("SELECT count(*) FROM shingles").fetchone()[0], round(time.time() - t0), "s")
con.execute(f"""
CREATE OR REPLACE TEMP TABLE nsh AS SELECT quote_id, count(DISTINCT h) AS n FROM shingles GROUP BY 1;
CREATE OR REPLACE TEMP TABLE posting AS
  SELECT h, list(DISTINCT quote_id) AS ids FROM shingles GROUP BY h
  HAVING count(DISTINCT quote_id) BETWEEN 2 AND {MAX_POSTING};
CREATE OR REPLACE TEMP TABLE pairs AS
  SELECT a, b, count(*) AS shared FROM (
    SELECT x.a, y.b FROM (SELECT h, unnest(ids) AS a FROM posting) x
    JOIN (SELECT h, unnest(ids) AS b FROM posting) y ON x.h = y.h AND x.a < y.b) GROUP BY a, b;
CREATE OR REPLACE TEMP TABLE links AS
  SELECT p.a, p.b FROM pairs p JOIN nsh na ON na.quote_id = p.a JOIN nsh nb ON nb.quote_id = p.b
  WHERE p.shared >= {MIN_SHARE} * least(na.n, nb.n);
""")
# keep direct variant links too: long speech excerpts chain many distinct sentences into one
# cluster, so tracing expands a quote only to its direct variants, not the whole cluster
con.execute("CREATE OR REPLACE TABLE quote_links AS SELECT a, b FROM links")
links = con.execute("SELECT a, b FROM links").fetchall()
print("links", len(links), round(time.time() - t0), "s")

# union-find over linked quotes
ids = {q: i for i, q in enumerate({x for ab in links for x in ab})}
parent = np.arange(len(ids))


def find(i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


for a, b in links:
    ra, rb = find(ids[a]), find(ids[b])
    if ra != rb:
        parent[max(ra, rb)] = min(ra, rb)
inv = list(ids)
rows = [(q, inv[find(i)]) for q, i in ids.items()]
con.execute("CREATE OR REPLACE TEMP TABLE uf (quote_id VARCHAR, root VARCHAR)")
con.executemany("INSERT INTO uf VALUES (?, ?)", rows) if len(rows) < 50_000 else \
    con.execute("INSERT INTO uf SELECT * FROM (SELECT unnest(?) AS quote_id, unnest(?) AS root)",
                [[r[0] for r in rows], [r[1] for r in rows]])
# cluster id = earliest quote (quote ids encode first-seen date); singletons are their own cluster
con.execute("""
CREATE OR REPLACE TABLE quote_clusters AS
  WITH c AS (SELECT u.quote_id, min(u.quote_id) OVER (PARTITION BY u.root) AS cluster_id FROM uf u)
  SELECT q.quote_id, coalesce(c.cluster_id, q.quote_id) AS cluster_id
  FROM quotes q LEFT JOIN c USING (quote_id);
""")
print(con.execute("""SELECT count(*) quotes, count(DISTINCT cluster_id) clusters,
  sum(CASE WHEN quote_id <> cluster_id THEN 1 ELSE 0 END) merged FROM quote_clusters""").fetchone(),
      round(time.time() - t0), "s")
