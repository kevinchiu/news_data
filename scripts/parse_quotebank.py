"""Flatten a Quotebank article-centric file into a Parquet table of quote mentions.

One row per (article, quotation): when and where (domain) a quote appeared, who said it, and
the text around it — the raw material for tracing which outlet carried a quote first.

Usage: python -I scripts/parse_quotebank.py data/raw/quotebank/quotebank-2020.json.bz2 data/processed/quotebank-2020-mentions.parquet
"""
import bz2
import json
import sys
from urllib.parse import urlparse

import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA = pa.schema([
    ("quote_id", pa.string()), ("article_id", pa.string()), ("date", pa.timestamp("s")),
    ("domain", pa.string()), ("url", pa.string()), ("title", pa.string()),
    ("speaker", pa.string()), ("speaker_proba", pa.float32()), ("quotation", pa.string()),
    ("left_context", pa.string()), ("right_context", pa.string()), ("quote_occurrences", pa.int32()),
    ("article_length", pa.int32()), ("phase", pa.string()),
])
BATCH = 250_000


def domain(url):
    host = urlparse(url).netloc.lower().split(":")[0]
    return host[4:] if host.startswith("www.") else host


def rows(path):
    with bz2.open(path, "rt") as f:
        for line in f:
            a = json.loads(line)
            dom = domain(a["url"])
            for q in a.get("quotations") or []:
                probas = q.get("localProbas") or []
                top = probas[0] if probas else ["None", "0"]
                yield {
                    "quote_id": q["quoteID"], "article_id": a["articleID"], "date": a["date"],
                    "domain": dom, "url": a["url"], "title": a.get("title"),
                    "speaker": None if q["localTopSpeaker"] == "None" else q["localTopSpeaker"],
                    "speaker_proba": float(top[1]), "quotation": q["quotation"],
                    "left_context": q.get("leftContext"), "right_context": q.get("rightContext"),
                    "quote_occurrences": q.get("numOccurrences"), "article_length": a.get("articleLength"),
                    "phase": a.get("phase"),
                }


def main(src, dst):
    import datetime as dt
    writer, buf, n = pq.ParquetWriter(dst, SCHEMA, compression="zstd"), [], 0
    for r in rows(src):
        r["date"] = dt.datetime.strptime(r["date"], "%Y-%m-%d %H:%M:%S")
        buf.append(r)
        if len(buf) >= BATCH:
            writer.write_table(pa.Table.from_pylist(buf, SCHEMA)); n += len(buf); buf = []
            print(f"{n:,} mentions", flush=True)
    if buf:
        writer.write_table(pa.Table.from_pylist(buf, SCHEMA)); n += len(buf)
    writer.close()
    print(f"done: {n:,} mentions -> {dst}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
