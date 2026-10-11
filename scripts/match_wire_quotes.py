"""Load GDELT wire articles into DuckDB and link their quotations to Quotebank quotes.

Tables written to data/news.duckdb:
  gdelt_wire_articles(gkg_id, date, source, url, title, persons, organizations)
  wire_quote_matches(gkg_id, quote_id, shared, gdelt_quote)
A wire quote is linked to a Quotebank quote when they share word 6-grams covering at least half
of the shorter one (GDELT quotes are often fragments of the full quotation), the wire quote has
at least 8 words (shorter fragments are stock phrases), and the quote was first seen in
Quotebank within a week of the wire article.

Usage: python -I scripts/match_wire_quotes.py data/processed/gdelt_gkg_wire
"""
import sys
import time

import duckdb

N, MIN_SHARE, MAX_POSTING = 6, 0.5, 200


def main(parts_dir):
    con = duckdb.connect("data/news.duckdb")
    t0 = time.time()
    con.execute(f"""
    CREATE OR REPLACE TABLE gdelt_wire_articles AS
      SELECT gkg_id, date, source, url, title, persons, organizations FROM '{parts_dir}/*.parquet';
    CREATE OR REPLACE TEMP TABLE wq AS
      SELECT gkg_id, row_number() OVER () AS wid, q.quote AS gdelt_quote
      FROM '{parts_dir}/*.parquet', unnest(quotations) AS t(q);
    """)
    print("wire articles", con.execute("SELECT count(*) FROM gdelt_wire_articles").fetchone()[0],
          "| wire quotes", con.execute("SELECT count(*) FROM wq").fetchone()[0])

    def shingle_sql(table, key, text):
        return f"""
          SELECT {key} AS k, hash(array_to_string(list_slice(w, i, i + {N - 1}), ' ')) AS h
          FROM (SELECT {key}, list_filter(string_split(regexp_replace(lower({text}), '[^a-z0-9 ]+', ' ', 'g'), ' '),
                                          x -> x <> '') AS w FROM {table}),
               unnest(generate_series(1, greatest(len(w) - {N - 1}, 0))) AS t(i)"""
    con.execute(f"CREATE OR REPLACE TEMP TABLE wsh AS {shingle_sql('wq', 'wid', 'gdelt_quote')}")
    # only Quotebank shingles that also occur in wire quotes are needed
    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE qsh AS
      SELECT * FROM ({shingle_sql('quotes', 'quote_id', 'quotation')}) WHERE h IN (SELECT h FROM wsh)""")
    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE common AS
      SELECT h FROM qsh GROUP BY h HAVING count(DISTINCT k) <= {MAX_POSTING};
    CREATE OR REPLACE TABLE wire_quote_matches AS
      WITH pairs AS (
        SELECT w.k AS wid, q.k AS quote_id, count(DISTINCT w.h) AS shared
        FROM wsh w JOIN qsh q USING (h) WHERE h IN (SELECT h FROM common) GROUP BY 1, 2),
      nw AS (SELECT k, count(DISTINCT h) n FROM wsh GROUP BY 1)
      SELECT wq.gkg_id, p.quote_id, p.shared, wq.gdelt_quote
      FROM pairs p JOIN wq USING (wid) JOIN nw ON nw.k = p.wid
      JOIN gdelt_wire_articles a ON a.gkg_id = wq.gkg_id JOIN quotes qq ON qq.quote_id = p.quote_id
      JOIN (SELECT quote_id, len(list_filter(string_split(regexp_replace(lower(quotation), '[^a-z0-9 ]+', ' ', 'g'), ' '),
                                            x -> x <> '')) - {N - 1} AS n FROM quotes) nq2 ON nq2.quote_id = p.quote_id
      WHERE nw.n >= 3 AND p.shared >= greatest(3, {MIN_SHARE} * least(nw.n, nq2.n))
        AND abs(epoch(qq.first_date) - epoch(a.date)) <= 7 * 86400;
    """)
    print("matches", con.execute("SELECT count(*), count(DISTINCT quote_id), count(DISTINCT gkg_id) FROM wire_quote_matches").fetchone(),
          round(time.time() - t0), "s")


if __name__ == "__main__":
    main(sys.argv[1])
