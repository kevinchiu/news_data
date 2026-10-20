"""Load the raw GDELT 2.0 Events (export) and Mentions files into DuckDB.

Tables written to data/news.duckdb:
  gdelt_events(GlobalEventID, ..., DATEADDED, SOURCEURL)   one row per coded event (61 columns)
  gdelt_mentions(GlobalEventID, EventTimeDate, MentionTimeDate, ..., MentionIdentifier, ...)
    one row per article mentioning an event; MentionTimeDate is when GDELT saw the article
    (15-minute resolution), an upper bound on publication.
Column names follow the GDELT 2.0 codebooks. The zips are unpacked into one temporary
tab-separated file per table, which is deleted after loading.

Usage: python -I scripts/load_gdelt_events.py data/raw/gdelt
"""
import subprocess
import sys
import time
from pathlib import Path

import duckdb

ACTOR = ["Code", "Name", "CountryCode", "KnownGroupCode", "EthnicCode", "Religion1Code", "Religion2Code",
         "Type1Code", "Type2Code", "Type3Code"]
GEO = ["Type", "FullName", "CountryCode", "ADM1Code", "ADM2Code", "Lat", "Long", "FeatureID"]
EVENT_COLS = (["GlobalEventID", "Day", "MonthYear", "Year", "FractionDate"]
              + [f"Actor1{c}" for c in ACTOR] + [f"Actor2{c}" for c in ACTOR]
              + ["IsRootEvent", "EventCode", "EventBaseCode", "EventRootCode", "QuadClass", "GoldsteinScale",
                 "NumMentions", "NumSources", "NumArticles", "AvgTone"]
              + [f"{a}Geo_{c}" for a in ("Actor1", "Actor2", "Action") for c in GEO]
              + ["DATEADDED", "SOURCEURL"])
MENTION_COLS = ["GlobalEventID", "EventTimeDate", "MentionTimeDate", "MentionType", "MentionSourceName",
                "MentionIdentifier", "SentenceID", "Actor1CharOffset", "Actor2CharOffset", "ActionCharOffset",
                "InRawText", "Confidence", "MentionDocLen", "MentionDocTone", "MentionDocTranslationInfo", "Extras"]
# codes keep leading zeros (EventCode "010"), so only these columns are converted
TYPED = {
    "gdelt_events": {"GlobalEventID": "BIGINT", "NumMentions": "INTEGER", "NumSources": "INTEGER",
                     "NumArticles": "INTEGER", "GoldsteinScale": "DOUBLE", "AvgTone": "DOUBLE",
                     "QuadClass": "INTEGER", "IsRootEvent": "INTEGER", "ActionGeo_Lat": "DOUBLE",
                     "ActionGeo_Long": "DOUBLE"},
    "gdelt_mentions": {"GlobalEventID": "BIGINT", "MentionType": "INTEGER", "SentenceID": "INTEGER",
                       "Actor1CharOffset": "INTEGER", "Actor2CharOffset": "INTEGER",
                       "ActionCharOffset": "INTEGER", "InRawText": "INTEGER", "Confidence": "INTEGER",
                       "MentionDocLen": "INTEGER", "MentionDocTone": "DOUBLE"},
}
TIMES = {"gdelt_events": ["DATEADDED"], "gdelt_mentions": ["EventTimeDate", "MentionTimeDate"]}


def load(con, table, cols, zips, tmp):
    with open(tmp, "wb") as out:
        for z in zips:
            subprocess.run(["unzip", "-p", str(z)], stdout=out, check=True)
    spec = "{" + ", ".join(f"'{c}': 'VARCHAR'" for c in cols) + "}"
    sel = ", ".join(
        f"strptime({c}, '%Y%m%d%H%M%S') AS {c}" if c in TIMES[table]
        else f"TRY_CAST({c} AS {TYPED[table][c]}) AS {c}" if c in TYPED[table] else c
        for c in cols)
    con.execute(f"""CREATE OR REPLACE TABLE {table} AS SELECT {sel}
                    FROM read_csv('{tmp}', delim='\t', header=false, quote='', escape='', columns={spec},
                                  ignore_errors=true, store_rejects=true)""")
    rejects = con.execute("SELECT count(*) FROM reject_errors").fetchone()[0]
    tmp.unlink()
    n, lo, hi = con.execute(f"SELECT count(*), min({TIMES[table][-1]}), max({TIMES[table][-1]}) FROM {table}").fetchone()
    print(f"{table}: {n:,} rows ({lo} .. {hi}), {rejects} rejected lines", flush=True)


def main(raw_dir):
    raw = Path(raw_dir)
    con = duckdb.connect("data/news.duckdb")
    con.execute("SET threads = 3")
    t0 = time.time()
    load(con, "gdelt_events", EVENT_COLS, sorted((raw / "export").glob("*.zip")), raw / "export.tsv")
    load(con, "gdelt_mentions", MENTION_COLS, sorted((raw / "mentions").glob("*.zip")), raw / "mentions.tsv")
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main(sys.argv[1])
