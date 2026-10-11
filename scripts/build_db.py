"""Build (or refresh) data/news.duckdb from the processed files.

Tables: reuters_articles, reuters_sentences, quotes (+ full-text index), quote_urls,
        domain_stats, domain_labels.
        quote_mentions (from the article-centric Parquet, once it exists; sorted by quote_id).
Views:  domains (stats + labels).
Run scripts/quote_variants.py afterwards for quote_clusters / quote_links.

Usage: python -I scripts/build_db.py [--mentions-only]
"""
import sys
from pathlib import Path

import duckdb

DB = "data/news.duckdb"
P = "data/processed"

con = duckdb.connect(DB)
MENTIONS_ONLY = "--mentions-only" in sys.argv
if not MENTIONS_ONLY:
    con.execute(f"""
CREATE OR REPLACE TABLE reuters_articles AS
  SELECT * REPLACE (CAST(date AS TIMESTAMP) AS date) FROM read_json_auto('{P}/reuters21578.jsonl');
CREATE OR REPLACE TABLE reuters_sentences AS
  SELECT * REPLACE (CAST(date AS TIMESTAMP) AS date) FROM read_json_auto('{P}/sentences.jsonl');

CREATE OR REPLACE TABLE quotes AS SELECT * FROM '{P}/quotes-2020-quotes.parquet';
CREATE OR REPLACE TABLE quote_urls AS SELECT * FROM '{P}/quotes-2020-urls.parquet';

-- volume per domain, the basis for labelling wires/aggregators
CREATE OR REPLACE TABLE domain_stats AS
  SELECT domain, count(*) AS n_mentions, count(DISTINCT quote_id) AS n_quotes,
         count(DISTINCT url) AS n_urls
  FROM quote_urls GROUP BY domain;
""")

    con.execute("INSTALL fts; LOAD fts;")
    con.execute("PRAGMA create_fts_index('quotes', 'quote_id', 'quotation', stemmer='english', "
                "stopwords='english', overwrite=1)")
    # second index that also matches the speaker's name (a retrieval option for the tracer)
    con.execute("CREATE OR REPLACE TABLE quotes_qs AS "
                "SELECT quote_id, quotation || ' ' || coalesce(speaker, '') AS text FROM quotes")
    con.execute("PRAGMA create_fts_index('quotes_qs', 'quote_id', 'text', stemmer='english', "
                "stopwords='english', overwrite=1)")

    # hand-curated labels (wire / press_release / aggregator / feed, plus ownership group) by LIKE pattern
    con.execute("CREATE OR REPLACE TABLE domain_labels AS SELECT * FROM read_csv('data/domain_labels.csv', header=true)")
    con.execute("""
CREATE OR REPLACE VIEW domains AS
  SELECT s.*, coalesce(l.kind, 'outlet') AS kind,
         -- an RSS/feed mirror belongs to its parent site
         coalesce(l.grp, CASE WHEN l.kind = 'feed' THEN regexp_replace(s.domain, '^(feeds?|rss)\\.', '') END) AS grp
  FROM domain_stats s LEFT JOIN LATERAL (
    SELECT kind, grp FROM domain_labels WHERE s.domain LIKE pattern ORDER BY length(pattern) DESC LIMIT 1) l ON true""")

mentions = Path(f"{P}/quotebank-2020-mentions")  # directory of Parquet parts
if mentions.exists():
    # midnight stamps are date-only in the source: treat their hour as unknown.
    # sorted by quote_id so per-quote timeline lookups skip most row groups
    con.execute("DROP VIEW IF EXISTS quote_mentions")
    con.execute(f"""
    CREATE OR REPLACE TABLE quote_mentions AS
      SELECT *, strftime(date, '%H:%M:%S') <> '00:00:00' AS has_time
      FROM '{mentions}/*.parquet' ORDER BY quote_id, date""")

for (t,) in con.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'main' ORDER BY 1").fetchall():
    print(f"{t:<20} {con.execute(f'SELECT count(*) FROM {t}').fetchone()[0]:>12,}")
